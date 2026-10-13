"""ConvLines-BB (bb/DESIGN.md section 4): arch.net's ConvLines carried to the bivariate bicycle codes on the l x m torus.

Input: the X-memory detectors on the (T, l, m) grid of bb.code.Layout, two channels per site (the site's X- and Z-check
detector; no Z detector exists at t = 0 and t = T - 1, which the per-layer bias tells the network).
  stem      one convolution, circular in (a, b) and zero-padded in t, whose taps are exactly the DEM's co-firing offsets
            (dt, da, db) (bb.design: every pair of detectors inside one fault), 2 -> C channels, plus a learned bias per
            time layer: every site sees every elementary-error relation in one step, the codes' non-local monomials
            (x^3, y^3) included
  blocks    L pre-norm bottleneck residual blocks as arch.net: LayerNorm -> C -> Cb -> SiLU -> circular tap convolution
            Cb -> Cb (taps: the Tanner offsets x dt in {-1, 0, 1} by default, taps_from_design) -> SiLU -> Cb -> C
            (zero-initialised)
  read-out  one query per translate (k, g) of each logical k (k G of them, G = l m): a content gate shared by all
            queries (sigmoid(q0 . key)) times a per-logical position gate sigmoid(b[k, h, t, site - g]) indexed by the
            site's offset from the translate (translation-equivariant; initialised to prefer the X checks adjacent to
            the logical's support), summed over all sites / N0 (sums keep counts, as arch.net); out-projection,
            residual with the logical's query, LayerNorm + MLP -> one logit z[k, g] per translate
  decision  the joint vote over the 2^k logical classes y: score(y) = sum_{k,g} (-1)^{phi_kg(y)} s_kg + sum_j (y_j - 1/2)
            beta_j with s_kg = -(-1)^{g_kg} z_kg w_k / 2, phi_kg(y) = (y Mcol)_kg and g_kg = (det Gdet)_kg the exact
            detector parity (bb.code.translates). With all translates equivalent this is arch.net's vote
            c = sum_j w_j (-1)^g_j z_j + b0 per logical; in general it is the log-likelihood of y under a model that
            treats the translate logits as independent, which they are not (translates of one syndrome are
            correlated): score is a learned ranking of the classes, not a posterior. Decision = argmax_y score; the
            margin (top-1 minus top-2 score) is a ranking statistic for routing, uncalibrated (bb/DESIGN.md 4.4).
Training loss: CE(score, y) + mean over translates of BCE(z_kg, y_kg), y_kg the exact translate flips."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N0 = 64.0


class TapConv(nn.Module):
    """Convolution over (T, l, m), circular in (l, m), zero-padded in T, with weights only on the taps (dt, da, db):
    out(P) = sum_taps W_tap x(P + tap) (a dense conv3d on the circularly padded grid with a fixed 0/1 kernel mask)."""

    def __init__(self, cin, cout, taps, l, m, bias=True):
        super().__init__()
        taps = sorted({(int(t[0]), int(t[1]), int(t[2])) for t in taps})
        self.taps = taps
        self.l, self.m = l, m
        self.dt = max(abs(t[0]) for t in taps)
        self.alo, self.ahi = min(t[1] for t in taps), max(t[1] for t in taps)
        self.blo, self.bhi = min(t[2] for t in taps), max(t[2] for t in taps)
        assert self.ahi - self.alo < l and self.bhi - self.blo < m, "taps alias on the torus"
        k = (2 * self.dt + 1, self.ahi - self.alo + 1, self.bhi - self.blo + 1)
        mask = torch.zeros(k)
        for dt, da, db in taps:
            mask[dt + self.dt, da - self.alo, db - self.blo] = 1
        self.register_buffer("mask", mask, persistent=False)
        bound = 1.0 / math.sqrt(cin * len(taps))
        self.weight = nn.Parameter(torch.empty(cout, cin, *k).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(cout)) if bias else None

    def forward(self, x):                                  # (B, C, T, l, m)
        x = F.pad(x, (-self.blo, self.bhi, -self.alo, self.ahi, 0, 0), mode="circular")
        return F.conv3d(x, self.weight * self.mask, self.bias, padding=(self.dt, 0, 0))


class Block(nn.Module):
    """dil > 0: a time-only block (gross2 --tblocks): the conv is kernel (3, 1, 1) with dilation (dil, 1, 1),
    zero-padded in t, so a site meets its own layers t - dil, t, t + dil."""

    def __init__(self, C, Cb, taps, l, m, dil=0):
        super().__init__()
        self.ln = nn.LayerNorm(C)
        self.down = nn.Linear(C, Cb)
        self.conv = nn.Conv3d(Cb, Cb, (3, 1, 1), dilation=(dil, 1, 1), padding=(dil, 0, 0)) if dil else \
            TapConv(Cb, Cb, taps, l, m)
        self.up = nn.Linear(Cb, C)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):                                  # (B, T, l, m, C)
        u = F.silu(self.down(self.ln(x)))
        v = F.silu(self.conv(u.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1))
        return x + self.up(v)


class TranslateQueries(nn.Module):
    """k G translate queries over the site features (B, S = T n2, C) -> logits (B, k, G). bins (time-bin edges, gross2
    --rbins): the read-out branch R adds to each logit an MLP of the query's pooled summaries kept apart per time bin
    (the same gate and position weights, the sum taken per bin), its final linear zero-initialised."""

    def __init__(self, C, heads, K, T, l, m, near, mlp=2, bins=()):
        super().__init__()
        n2 = l * m
        self.K, self.G, self.T, self.n2, self.h, self.dh = K, n2, T, n2, heads, C // heads
        self.ln = nn.LayerNorm(C)
        self.key = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.e = nn.Parameter(torch.zeros(K, C))
        self.bc = nn.Parameter(torch.zeros(heads))
        b0 = torch.where(torch.as_tensor(near, dtype=torch.bool), -1.0, -3.0)          # (K, n2)
        self.b = nn.Parameter(b0[:, None, None, :].repeat(1, heads, T, 1).contiguous())  # (K, H, T, n2)
        a, bb = np.divmod(np.arange(n2), m)
        rel = ((a[None, :] - a[:, None]) % l) * m + (bb[None, :] - bb[:, None]) % m       # rel[g, s] = s - g
        self.register_buffer("rel", torch.as_tensor(rel, dtype=torch.long), persistent=False)
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))
        self.bins, self.rb = list(zip(bins[:-1], bins[1:])), None
        if bins:
            assert bins[0] == 0 and bins[-1] == T and all(x < y for x, y in self.bins), bins
            nb = len(self.bins)
            self.rb = nn.Sequential(nn.LayerNorm(nb * C), nn.Linear(nb * C, C), nn.SiLU(), nn.Linear(C, 1))
            nn.init.zeros_(self.rb[-1].weight)
            nn.init.zeros_(self.rb[-1].bias)

    def position(self):                                    # (H, K G, S)
        P = torch.sigmoid(self.b)[:, :, :, self.rel]       # (K, H, T, G, n2)
        return P.permute(1, 0, 3, 2, 4).reshape(self.h, self.K * self.G, self.T * self.n2) / N0

    def forward(self, f):
        B, S, C = f.shape
        H, dh = self.h, self.dh
        y = self.ln(f)
        gate = torch.sigmoid(torch.einsum("bshe,he->bsh", self.key(y).view(B, S, H, dh), self.q0.view(H, dh))
                             * dh ** -0.5 + self.bc)
        U = gate.unsqueeze(-1) * self.v(y).view(B, S, H, dh)
        P = self.position().to(U.dtype)
        a = torch.einsum("hqs,bshe->bqhe", P, U).reshape(B, self.K * self.G, C)
        u = (self.q0 + self.e).repeat_interleave(self.G, 0) + self.o(a)
        z = self.mlp(u).squeeze(-1)
        if self.rb is not None:                            # S is t-major: bin [t0, t1) = sites t0 n2 .. t1 n2 - 1
            n = self.n2
            ab = [torch.einsum("hqs,bshe->bqhe", P[:, :, t0 * n:t1 * n], U[:, t0 * n:t1 * n]).reshape(B, -1, C)
                  for t0, t1 in self.bins]
            z = z + self.rb(torch.cat(ab, -1)).squeeze(-1)
        return z.view(B, self.K, self.G)


def mean_gate0(near):
    """TranslateQueries' mean position gate at initialisation over logicals, sites and translates: sigmoid(-1) on the X
    checks of the logical's support, sigmoid(-3) elsewhere (translation averages over the offset s - g)."""
    return float(np.mean(1.0 / (1.0 + np.exp(-np.where(np.asarray(near, bool), -1.0, -3.0)))))


class PooledQueriesBB(nn.Module):
    """A1's pooled head on BB (B-P, B-PA; revision/REVISION_PLAN.md 2.1): Q = K G learned queries without translate
    geometry over the site features plus a learned position embedding pos (S, C), with TranslateQueries' gated sum: one
    content gate per site and head shared by all queries (sigmoid(q0 . key / sqrt(dh) + bc)) times a per-query position
    gate, here a free learned table sigmoid(b[q, h, s]) (no tie to the offset s - g, no logical's support), started at
    the logit of the translate read-out's mean initial gate (mean_gate0); summed over the sites / N0, out-projection,
    residual with the query's own vector q0 + e_q, LayerNorm + MLP -> one output u_q per query (B, Q). The network maps
    them to the K logical logits with a learned linear map. Keeping the shared content gate keeps the multiply-adds of
    the deployed read-out (a content score per query would add Q S C, about 30 %). Construction draws ln, key, v, o, q0
    and the MLP in TranslateQueries' order (from the same head seed they start equal to its); e and pos (N(0, 0.02^2))
    are drawn after them."""

    def __init__(self, C, heads, Q, S, mlp=2, gate0=0.5):
        super().__init__()
        self.Q, self.S, self.h, self.dh = Q, S, heads, C // heads
        self.ln = nn.LayerNorm(C)
        self.key = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.bc = nn.Parameter(torch.zeros(heads))
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))
        self.e = nn.Parameter(torch.randn(Q, C) * 0.02)
        self.pos = nn.Parameter(torch.randn(S, C) * 0.02)
        self.b = nn.Parameter(torch.full((Q, heads, S), float(np.log(gate0 / (1.0 - gate0)))))

    def position(self):                                    # (H, Q, S)
        return (torch.sigmoid(self.b) / N0).permute(1, 0, 2)

    def forward(self, f):
        B, S, C = f.shape
        H, dh = self.h, self.dh
        y = self.ln(f + self.pos)
        gate = torch.sigmoid(torch.einsum("bshe,he->bsh", self.key(y).view(B, S, H, dh), self.q0.view(H, dh))
                             * dh ** -0.5 + self.bc)
        U = gate.unsqueeze(-1) * self.v(y).view(B, S, H, dh)
        a = torch.einsum("hqs,bshe->bqhe", self.position().to(U.dtype), U).reshape(B, self.Q, C)
        u = (self.q0 + self.e) + self.o(a)
        return self.mlp(u).squeeze(-1)


HEADS = ("joint", "pooled")


class ConvLinesBB(nn.Module):
    """tblocks (dilations) and rbins (time-bin edges) are gross2's additions (empty by default: the network, its cfg and
    its initialisation are then exactly the stage-2 ones); both start at zero (zero-initialised outputs).
    head (A1, revision/REVISION_PLAN.md 2.1): "joint" the deployed translate read-out and joint vote; "pooled"
    PooledQueriesBB and a learned linear map lmap (Q -> K) to the K logical logits, decision bits = logit > 0,
    confidence min_k |logit_k|. aux_translates (pooled only) adds a training-only translate read-out `aux` (a
    TranslateQueries over the trunk features) whose K G logits carry the translate loss; it is not on the decision path
    and export_decision leaves it out. head_seed: the torch seed drawn before each head module (decision head, then
    aux), so arms that share a head module start it identically; None keeps the stage-2 order. The configuration
    records head and aux_translates only for a head other than the deployed one."""

    def __init__(self, code, p, stem_taps, blk_taps, width=128, bott=32, blocks=3, heads=4, mlp=2,
                 tblocks=(), rbins=(), head="joint", aux_translates=False, head_seed=None):
        super().__init__()
        assert head in HEADS and (not aux_translates or head == "pooled"), (head, aux_translates)
        assert head == "joint" or not (tblocks or rbins), "the A1 heads use the stage-2 trunk and read-out"
        from bb import code as bcode
        tr = bcode.translates(code, p)
        lay = tr["layout"]
        self.cfg = dict(arch="convlines_bb", code=code, p=p, width=width, bott=bott, blocks=blocks, heads=heads, mlp=mlp,
                        stem_taps=[list(map(int, t[:3])) for t in stem_taps],
                        blk_taps=[list(map(int, t[:3])) for t in blk_taps])
        if tblocks:
            self.cfg["tblocks"] = [int(d) for d in tblocks]
        if rbins:
            self.cfg["rbins"] = [int(t) for t in rbins]
        if head != "joint" or aux_translates:
            self.cfg.update(head=head, aux_translates=bool(aux_translates))
        self.kind = head
        self.l, self.m, self.T, self.n2, self.D, self.K = lay.l, lay.m, lay.T, lay.n2, lay.D, lay.k
        self.G = lay.n2
        T, l, m, C = self.T, self.l, self.m, width
        gidx = np.full(T * 2 * self.n2, self.D, np.int64)  # grid (t, type, site) -> detector (D: none)
        gidx[lay.flat] = np.arange(self.D)
        self.register_buffer("gidx", torch.as_tensor(gidx), persistent=False)
        self.stem = TapConv(2, C, stem_taps, l, m)
        self.tbias = nn.Parameter(torch.zeros(T, C))
        self.blocks = nn.ModuleList(Block(C, bott, blk_taps, l, m) for _ in range(blocks))
        hx = lay.hx.astype(np.int64)
        near = (hx @ lay.logicals.T.astype(np.int64) > 0).T                                 # (K, n2) X checks on the support
        K, G = self.K, self.G
        if head_seed is not None:
            torch.manual_seed(head_seed)
        if head == "pooled":
            self.head = PooledQueriesBB(C, heads, K * G, T * self.n2, mlp, gate0=mean_gate0(near))
            self.lmap = nn.Linear(K * G, K)
        else:
            self.head = TranslateQueries(C, heads, self.K, T, l, m, near, mlp, tuple(int(t) for t in rbins))
        self.aux = None
        if aux_translates:
            if head_seed is not None:
                torch.manual_seed(head_seed)
            self.aux = TranslateQueries(C, heads, self.K, T, l, m, near, mlp)
        Y = (np.arange(2 ** K)[:, None] >> np.arange(K)[None, :]) & 1                       # (2^K, K) class bits
        phi = (Y @ tr["Mcol"].astype(np.int64)) % 2                                         # (2^K, K G)
        self.register_buffer("Ybits", torch.as_tensor(Y, dtype=torch.uint8), persistent=False)
        self.register_buffer("Psi", torch.as_tensor(1 - 2 * phi, dtype=torch.float32), persistent=False)
        self.register_buffer("Mcol", torch.as_tensor(tr["Mcol"], dtype=torch.float32), persistent=False)
        self.register_buffer("Gdet", torch.as_tensor(tr["Gdet"], dtype=torch.float32), persistent=False)
        self.register_buffer("pow2", torch.as_tensor(2 ** np.arange(K), dtype=torch.long), persistent=False)
        stab = [[g for g in range(G) if (tr["M"][g, kk] == np.eye(K, dtype=np.uint8)[kk]).all()] for kk in range(K)]
        sm = np.zeros((K, G), np.float32)
        for kk, s in enumerate(stab):
            sm[kk, s] = 1
        self.register_buffer("stabmask", torch.as_tensor(sm), persistent=False)         # translates equivalent to ell_k
        if head == "joint":                                # the joint vote's weights (the pooled head maps with lmap)
            self.w = nn.Parameter(torch.full((K,), 1.0 / G))
            self.beta = nn.Parameter(torch.zeros(K))
        self.tblocks = nn.ModuleList(Block(C, bott, None, l, m, int(d)) for d in tblocks)  # last: keeps the init above

    def detparity(self, det):                              # (B, D) -> (B, K G) float 0/1: g_kg
        with torch.autocast(det.device.type, enabled=False):
            return torch.remainder(det.to(self.Gdet.dtype) @ self.Gdet, 2.0)

    def labels(self, det, obs, g=None):                    # exact translate flips (B, K G) and class index (B,)
        with torch.autocast(det.device.type, enabled=False):   # g: forward's detparity(det), the same parity
            par = det.to(self.Gdet.dtype) @ self.Gdet if g is None else g.to(self.Mcol.dtype)
            y = torch.remainder(obs.to(self.Mcol.dtype) @ self.Mcol + par, 2.0)
        return y, (obs.long() * self.pow2).sum(1)

    def features(self, det):                               # (B, D) -> (B, S, C)
        B = det.shape[0]
        T, l, m = self.T, self.l, self.m
        pad = torch.cat([det, det.new_zeros(B, 1)], 1)
        x = pad[:, self.gidx].view(B, T, 2, l, m).transpose(1, 2).to(self.stem.weight.dtype)
        x = self.stem(x).permute(0, 2, 3, 4, 1) + self.tbias[None, :, None, None, :]
        for blk in (*self.blocks, *self.tblocks):
            x = blk(x)
        return x.reshape(B, T * self.n2, -1)

    def vote(self, z, g):                                  # z (B, K, G), g (B, K G) -> scores (B, 2^K)
        with torch.autocast(z.device.type, enabled=False):
            s = -(1.0 - 2.0 * g) * z.to(self.Psi.dtype).reshape(z.shape[0], -1) * self.w.repeat_interleave(self.G) / 2
            return s @ self.Psi.t() + (self.Ybits.to(self.beta.dtype) - 0.5) @ self.beta

    def logits(self, u):                                   # pooled: query outputs (B, Q) -> logical logits (B, K), fp32
        with torch.autocast(u.device.type, enabled=False):
            return F.linear(u.float(), self.lmap.weight, self.lmap.bias)

    def forward(self, det):
        """(B, D) uint8/float -> the decision path: joint (scores (B, 2^K), z (B, K, G), g (B, K G)); pooled
        (logical logits (B, K), query outputs u (B, Q), None)."""
        if self.kind == "pooled":
            u = self.head(self.features(det))
            return self.logits(u), u, None
        z = self.head(self.features(det))
        g = self.detparity(det)
        return self.vote(z, g), z, g

    def forward_train(self, det):
        """-> (decision output (scores or logical logits), the translate logits (B, K, G) of the translate loss or None,
        g (B, K G) or None)."""
        if self.kind == "joint":
            return self.forward(det)
        f = self.features(det)
        out = self.logits(self.head(f))
        if self.aux is None:
            return out, None, None
        return out, self.aux(f), self.detparity(det)

    @torch.no_grad()
    def decide(self, scores):
        """(class (B,), bits (B, K) uint8, margin (B,)): joint, the argmax class and top-1 minus top-2 score; pooled
        (scores = the logical logits), bits = logit > 0, class their index, margin min_k |logit_k|."""
        if self.kind == "pooled":
            bits = (scores > 0).to(torch.uint8)
            return (bits.long() * self.pow2).sum(1), bits, scores.abs().min(1).values
        top = torch.topk(scores, 2, dim=1)
        cls = top.indices[:, 0]
        return cls, self.Ybits[cls], top.values[:, 0] - top.values[:, 1]

    @torch.no_grad()
    def strict(self, z, g):
        """Diagnostic: per logical, the vote over its own equivalent translates only -> bits (B, K)."""
        c = ((1.0 - 2.0 * g) * z.float().reshape(z.shape[0], -1)).view(-1, self.K, self.G)
        return ((c * self.stabmask).sum(-1) > 0).to(torch.uint8)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def macs(model):
    """Multiply-adds per shot of the inference form (only the taps carrying weights), by part; the decision path alone
    (a training-only read-out excluded). The pooled head: the same read-out rows, its linear map (Q K) for the vote, and
    its position embedding as additions (no multiply)."""
    c = model.cfg
    C, Cb, L, H = c["width"], c["bott"], c["blocks"], c["heads"]
    S = model.T * model.n2
    Q = model.K * model.G
    out = {"stem": S * len(c["stem_taps"]) * 2 * C,
           "blocks (down + taps + up)": L * S * (C * Cb + len(c["blk_taps"]) * Cb * Cb + Cb * C),
           "read-out key, value, gate": S * (2 * C * C + C),
           "read-out pooling": Q * S * C,
           "read-out out-proj + MLP": Q * (C * C + C * c["mlp"] * C + c["mlp"] * C)}
    if c.get("head", "joint") == "pooled":
        out["linear map (Q K)"] = Q * model.K
    else:
        out["vote"] = 2 ** model.K * Q + model.D * Q
    if c.get("tblocks"):
        out["time blocks (down + 3 taps + up)"] = len(c["tblocks"]) * S * (C * Cb + 3 * Cb * Cb + Cb * C)
    if c.get("rbins"):
        out["read-out branch R (bin pooling + MLP)"] = Q * S * C + Q * ((len(c["rbins"]) - 1) * C * C + C)
    out["total"] = sum(out.values())
    return out


def build(cfg):
    return ConvLinesBB(cfg["code"], cfg["p"], cfg["stem_taps"], cfg["blk_taps"], cfg["width"], cfg["bott"],
                       cfg["blocks"], cfg["heads"], cfg["mlp"], cfg.get("tblocks", ()), cfg.get("rbins", ()),
                       head=cfg.get("head", "joint"), aux_translates=cfg.get("aux_translates", False))


def trunk_keys(model):
    """State-dict keys of the trunk (stem, per-layer bias, blocks): what a trunk-only --init carries."""
    return [k for k in model.state_dict() if k.startswith(("stem.", "tbias", "blocks."))]


def readout_macs(model):
    """The read-out's share of macs(model): every row but the stem and the blocks."""
    return sum(v for k, v in macs(model).items() if k not in ("stem", "blocks (down + taps + up)", "total"))


def export_decision(ck):
    """A checkpoint dict {'state', 'cfg', ...} -> the same without the training-only read-out (aux.*): the inference
    graph of every arm is build(cfg) with aux_translates False, loaded strictly from this state."""
    cfg = dict(ck["cfg"])
    drop = [k for k in ck["state"] if k.startswith("aux.")]
    assert bool(drop) == bool(cfg.get("aux_translates", False)), "aux weights and aux_translates disagree"
    if "aux_translates" in cfg:
        cfg["aux_translates"] = False
    return {**ck, "state": {k: v for k, v in ck["state"].items() if not k.startswith("aux.")}, "cfg": cfg,
            "dropped": drop}


def taps_from_design(path, blk="tanner", dt_blk=1):
    """(stem taps, block taps) from bb.design's JSON. The stem: every co-firing offset of the DEM (every elementary
    error in one step). The blocks, x dt in [-dt_blk, dt_blk]: 'tanner' = the offsets between checks that share one
    data qubit (X-X, Z-Z, X -> Z, Z -> X; the code's own neighbourhood, the analogue of ConvLines' 3 x 3 x 3), or
    'dense' = every spatial offset of the stem (on BB72 the whole torus, on gross 65 of 72)."""
    import json
    d = json.load(open(path))
    stem = [t[:3] for t in d["taps"]["stem"]]
    if blk == "tanner":
        tn = d["taps"]["tanner"]
        sp = sorted({tuple(x) for x in tn["xx"] + tn["xz"] + tn["zx"]})
    else:
        sp = [tuple(x) for x in d["taps"]["block_spatial"]]
    return stem, [[dt, a, b] for a, b in sp for dt in range(-dt_blk, dt_blk + 1)]
