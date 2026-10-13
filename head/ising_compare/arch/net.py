"""Stage G1 (arch/README.md): the ConvLines network (CL) at distance d.

Input: the dense detector grid. Every detector sits at (t, i, j) on a T x H x W grid (arch.geom.layout; d = 9:
10 x 10 x 10, 720 detectors; X-type sites at (i + j) even, Z-type at (i + j) odd; grid positions without a detector are
zero padding). Constant channels per site: X type, Z type, first layer, last layer (the data-measurement layer).

  stem      one type-specific 3D convolution (kernel 3 x 5 x 5 in (t, i, j): every single-mechanism relation of the DEM,
            matching edges and X-Z pairs, lies within (+-1, +-2, +-2)), detector bit + constant channels -> C; X- and
            Z-centred sites have their own kernels ("direction- and type-aware": one weight matrix per tap and type)
  blocks    L pre-norm bottleneck residual blocks: LayerNorm -> 1x1x1 C -> Cb -> SiLU -> type-specific 3 x 3 x 3
            convolution Cb -> Cb -> SiLU -> 1x1x1 Cb -> C (zero-initialised) added to the stream; off-site positions
            stay 0, so they act as zero padding
  read-out  d logical-line queries (line j = data-qubit row j, between grid rows j and j + 1; a shared query plus a
            per-line offset) cross-attend over the 720 site features (H heads; score bias by the site's signed distance
            to the line (half-line units, icmp.lines.line_offsets), its time layer and its type; initialised to prefer
            sites near the line) with sigmoid gates and a sum, a = sum_s sigmoid(score) v_s / N0 (N0 = 64): a line's
            flip is the parity of the chains crossing it, so the pooled value keeps counts (a softmax average would
            divide them out; Cascade pools by an average over a fixed support, i.e. a sum); then out-projection,
            residual, LayerNorm + MLP -> one logit z_j per line
  decision  the standard observable's logit c = sum_j w_j (-1)^g_j z_j + b0, g_j = y_j XOR y_0 an exact GF(2)
            function of the detectors (icmp.lines.gmap); w_j initialised to 1 / d. Decision = c > 0.

Training loss: mean_j BCE(z_j, y_j) + BCE(c, y_0), y_j the sampled line flips (icmp.lines.line_circuit).
`deploy_tables` folds the trained network into the inference form (per-site tables, per-type tap weights, the
read-out's query-key product and bias table); `DeployRef` runs that form with plain torch ops (the reference of the
engines in arch.engine and the source of the ONNX export)."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from icmp import lines

N0 = 64.0                                                # the read-out's constant sum normalisation


class Layout:
    """Geometry of distance d (numpy)."""

    def __init__(self, d):
        from arch.geom import layout
        mc, g, idx = layout(d)
        self.d, self.m = d, len(g)
        self.T, self.H, self.W = (int(idx[:, k].max()) + 1 for k in range(3))
        self.n = self.T * self.H * self.W
        self.idx = idx
        self.is_x = g[:, 3].astype(bool)
        self.flat = (idx[:, 0] * self.H + idx[:, 1]) * self.W + idx[:, 2]
        self.first = idx[:, 0] == 0
        self.last = idx[:, 0] == self.T - 1
        self.off = lines.line_offsets(d)                 # (m, d) signed distance to line j, half-line units
        self.A = lines.gmap(d)[0]                        # (m, d) uint8: g = bits A mod 2 (column 0 zero)
        pos = {tuple(int(v) for v in r): k for k, r in enumerate(idx)}
        self.pos = pos

    def site_order(self):
        """Engine order of the detectors: X sites, then Z sites, each in (t, i, j) order. (perm (m,), number of X)."""
        key = np.lexsort((self.flat, ~self.is_x))
        return key, int(self.is_x.sum())

    def neighbours(self, order, k):
        """(m, K) for a kernel k = (kt, ki, kj) in conv3d tap order: entry = engine index of the detector at
        (t, i, j) + (a - kt // 2, b - ki // 2, c - kj // 2), or -1."""
        inv = np.empty(self.m, np.int64)
        inv[order] = np.arange(self.m)
        taps = [(a - k[0] // 2, b - k[1] // 2, c - k[2] // 2) for a in range(k[0]) for b in range(k[1]) for c in range(k[2])]
        out = np.full((self.m, len(taps)), -1, np.int64)
        for s, det in enumerate(order):
            t, i, j = (int(v) for v in self.idx[det])
            for o, (dt, di, dj) in enumerate(taps):
                q = self.pos.get((t + dt, i + di, j + dj))
                if q is not None:
                    out[s, o] = inv[q]
        return out


class Block(nn.Module):
    def __init__(self, C, Cb, k):
        super().__init__()
        self.Cb = Cb
        self.ln = nn.LayerNorm(C)
        self.down = nn.Linear(C, Cb)
        self.conv = nn.Conv3d(Cb, 2 * Cb, k, padding=tuple(v // 2 for v in k))     # out: X-site weights, Z-site weights
        self.up = nn.Linear(Cb, C)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, mx, mz):                       # x (B, T, H, W, C) channel-last; mx / mz (1, T, H, W, 1)
        on = mx + mz
        u = F.silu(self.down(self.ln(x))) * on
        v = self.conv(u.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        v = F.silu(v[..., :self.Cb] * mx + v[..., self.Cb:] * mz)
        return x + self.up(v) * on


class LineQueries(nn.Module):
    """Line queries over the site features (B, m, C) -> line logits (B, n): all d lines (the deployed read-out), or the
    lines `lines` alone (A1's single-line head S-L: one line, n = 1). self.d is the number of queries n."""

    def __init__(self, C, heads, lay, mlp=2, lines=None):
        super().__init__()
        self.lines = list(range(lay.d)) if lines is None else [int(j) for j in lines]
        off = lay.off[:, self.lines]                       # (m, n) signed distance to each read line
        d = len(self.lines)
        self.d, self.h, self.dh = d, heads, C // heads
        self.ln = nn.LayerNorm(C)
        self.k = nn.Linear(C, C, bias=False)
        self.wq = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.e = nn.Parameter(torch.zeros(d, C))
        R = int(np.abs(off).max())
        bins = torch.arange(-R, R + 1, dtype=torch.float32)
        self.b_line = nn.Parameter((-1.0 - 0.5 * bins.abs()).repeat(heads, 2, 1))  # (H, type, 2R + 1): near the line
        self.n0 = N0
        self.b_t = nn.Parameter(torch.zeros(heads, 2, lay.T))                     # (H, type, T)
        self.register_buffer("off", torch.as_tensor(off + R, dtype=torch.long), persistent=False)           # (m, n)
        self.register_buffer("tt", torch.as_tensor(lay.idx[:, 0], dtype=torch.long), persistent=False)
        self.register_buffer("ty", torch.as_tensor((~lay.is_x).astype(np.int64)), persistent=False)        # 0 X, 1 Z
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))

    def bias(self):                                       # (m, d, H)
        bl = self.b_line[:, self.ty[:, None], self.off]   # (H, m, d)
        bt = self.b_t[:, self.ty, self.tt]                # (H, m)
        return (bl + bt[:, :, None]).permute(1, 2, 0)

    def queries(self):                                    # (d, C)
        return self.q0 + self.e

    def forward(self, f):
        B, m, C = f.shape
        d, H, dh = self.d, self.h, self.dh
        y = self.ln(f)
        q = self.wq(self.queries()).view(d, H, dh)
        k = self.k(y).view(B, m, H, dh)
        v = self.v(y).view(B, m, H, dh)
        s = torch.einsum("bmhe,jhe->bjhm", k, q) * dh ** -0.5 + self.bias().permute(1, 2, 0)[None].to(k.dtype)
        p = (torch.sigmoid(s.float()) / self.n0).to(v.dtype)
        a = torch.einsum("bjhm,bmhe->bjhe", p, v).reshape(B, d, C)
        u = self.queries() + self.o(a)
        return self.mlp(u).squeeze(-1)

    @torch.no_grad()
    def gates(self, f):                                   # fp32 site features -> gates (B, n, H, m), pooled (B, n, H, dh)
        B, m, C = f.shape
        d, H, dh = self.d, self.h, self.dh
        y = self.ln(f)
        q = self.wq(self.queries()).view(d, H, dh)
        k = self.k(y).view(B, m, H, dh)
        v = self.v(y).view(B, m, H, dh)
        g = torch.sigmoid(torch.einsum("bmhe,jhe->bjhm", k, q) * dh ** -0.5 + self.bias().permute(1, 2, 0)[None])
        return g, torch.einsum("bjhm,bmhe->bjhe", g, v) / self.n0


def mean_gate0(lay):
    """The line read-out's mean gate at initialisation over sites, lines, heads and types: sigmoid of the initial score
    bias -1 - |offset| / 2 (b_t = 0; the query-key term starts near 0). The pooled head's score bias starts at its logit,
    so both heads start with the same gate mass per query."""
    return float(np.mean(1.0 / (1.0 + np.exp(1.0 + 0.5 * np.abs(lay.off)))))


class PooledQueries(nn.Module):
    """A1's pooled head (S-P, S-PA; revision/REVISION_PLAN.md 2.1): n learned queries without line geometry cross-attend
    over the site features plus a learned position embedding pos (m, C), with the line read-out's gated sum: score =
    query . key / sqrt(dh) + a score bias, sigmoid gate, sum over the sites / N0; then out-projection, residual,
    LayerNorm and MLP -> one output u_j per query (B, n); the network maps them to c with a learned linear map
    (ConvLines). The score bias is a free learned table b[j, h, s] (the line read-out's table is built from each site's
    distance to the line and its layer; this one has no geometry), started at the logit of gate0 everywhere; the query
    offsets e and pos start random (N(0, 0.02^2)), since no geometry tells the queries or the sites apart. The same rule
    as bb.net.PooledQueriesBB: the deployed read-out's computation, its geometric table replaced by a free one.
    Construction draws ln, k, wq, v, o, q0 and the MLP in LineQueries' order, so from the same head seed they start
    equal to the line read-out's; e and pos are drawn after them."""

    def __init__(self, C, heads, m, n, mlp=2, gate0=0.5):
        super().__init__()
        self.d, self.h, self.dh = n, heads, C // heads
        self.ln = nn.LayerNorm(C)
        self.k = nn.Linear(C, C, bias=False)
        self.wq = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.n0 = N0
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))
        self.e = nn.Parameter(torch.randn(n, C) * 0.02)
        self.pos = nn.Parameter(torch.randn(m, C) * 0.02)
        self.b = nn.Parameter(torch.full((n, heads, m), float(np.log(gate0 / (1.0 - gate0)))))

    def queries(self):                                    # (n, C)
        return self.q0 + self.e

    def forward(self, f):
        B, m, C = f.shape
        n, H, dh = self.d, self.h, self.dh
        y = self.ln(f + self.pos)
        q = self.wq(self.queries()).view(n, H, dh)
        k = self.k(y).view(B, m, H, dh)
        v = self.v(y).view(B, m, H, dh)
        s = torch.einsum("bmhe,jhe->bjhm", k, q) * dh ** -0.5 + self.b[None].to(k.dtype)
        p = (torch.sigmoid(s.float()) / self.n0).to(v.dtype)
        a = torch.einsum("bjhm,bmhe->bjhe", p, v).reshape(B, n, C)
        u = self.queries() + self.o(a)
        return self.mlp(u).squeeze(-1)

    @torch.no_grad()
    def gates(self, f):
        B, m, C = f.shape
        n, H, dh = self.d, self.h, self.dh
        y = self.ln(f + self.pos)
        q = self.wq(self.queries()).view(n, H, dh)
        k = self.k(y).view(B, m, H, dh)
        v = self.v(y).view(B, m, H, dh)
        g = torch.sigmoid(torch.einsum("bmhe,jhe->bjhm", k, q) * dh ** -0.5 + self.b[None])
        return g, torch.einsum("bjhm,bmhe->bjhe", g, v) / self.n0


HEADS = ("vote", "line", "pooled")


class ConvLines(nn.Module):
    """head (A1, revision/REVISION_PLAN.md 2.1): "vote" the deployed read-out (every line, c = sum_j w_j (-1)^g_j z_j
    + b0); "line" the line line_index alone (default the centre (d - 1) / 2), parity-corrected like a vote term,
    c = w (-1)^g_j z_j + b0 with w started at 1; "pooled" PooledQueries with d queries and a learned linear map
    c = sum_j w_j u_j + b0 (no parity), w started uniform in +-1 / sqrt(d) (drawn after the head) and b0 at 0. aux_lines (pooled only) adds a training-only line read-out `aux` (a LineQueries
    over the trunk features) whose line logits carry the line loss; it is not on the decision path, and export_decision
    leaves it out. head_seed: the torch seed drawn before each head module (decision head, then aux), so arms that share
    a head module start it identically whatever the trunk; None keeps the old order (heads drawn after the trunk). The
    configuration records head, line_index and aux_lines only for a head other than the deployed one."""

    def __init__(self, d, width=128, bott=32, blocks=2, heads=4, stem_k=(3, 5, 5), block_k=(3, 3, 3), mlp=2,
                 head="vote", line_index=None, aux_lines=False, head_seed=None):
        super().__init__()
        assert head in HEADS and (not aux_lines or head == "pooled"), (head, aux_lines)
        lay = Layout(d)
        self.lay = lay
        self.cfg = dict(arch="convlines", d=d, width=width, bott=bott, blocks=blocks, heads=heads, stem_k=list(stem_k),
                        block_k=list(block_k), mlp=mlp)
        if head == "line":
            line_index = (d - 1) // 2 if line_index is None else int(line_index)
            assert 0 <= line_index < d, line_index
        else:
            line_index = None
        if head != "vote" or aux_lines:
            self.cfg.update(head=head, line_index=line_index, aux_lines=bool(aux_lines))
        self.kind = head
        self.d, self.m, self.C = d, lay.m, width
        T, H, W, n = lay.T, lay.H, lay.W, lay.n
        self.shape = (T, H, W)
        gidx = np.full(n, lay.m, np.int64)
        gidx[lay.flat] = np.arange(lay.m)
        self.register_buffer("gidx", torch.as_tensor(gidx), persistent=False)                   # grid -> detector (m: none)
        self.register_buffer("flat", torch.as_tensor(lay.flat), persistent=False)              # detector -> grid
        mx = np.zeros(n, np.float32)
        mz = np.zeros(n, np.float32)
        mx[lay.flat[lay.is_x]] = 1
        mz[lay.flat[~lay.is_x]] = 1
        fl = np.zeros(n, np.float32)
        ll = np.zeros(n, np.float32)
        fl[lay.flat[lay.first]] = 1
        ll[lay.flat[lay.last]] = 1
        self.register_buffer("mx", torch.as_tensor(mx).view(1, T, H, W, 1), persistent=False)
        self.register_buffer("mz", torch.as_tensor(mz).view(1, T, H, W, 1), persistent=False)
        self.register_buffer("const", torch.as_tensor(np.stack([mx, mz, fl, ll])).view(1, 4, T, H, W), persistent=False)
        self.register_buffer("A", torch.as_tensor(lay.A, dtype=torch.float32))                  # stored with the weights
        self.stem = nn.Conv3d(5, 2 * width, stem_k, padding=tuple(v // 2 for v in stem_k))
        self.blocks = nn.ModuleList(Block(width, bott, tuple(block_k)) for _ in range(blocks))
        if head_seed is not None:
            torch.manual_seed(head_seed)
        if head == "pooled":
            self.head = PooledQueries(width, heads, lay.m, d, mlp, gate0=mean_gate0(lay))
        else:
            self.head = LineQueries(width, heads, lay, mlp, lines=None if head == "vote" else [line_index])
        n = self.head.d
        self.w = nn.Parameter(torch.empty(n).uniform_(-n ** -0.5, n ** -0.5) if head == "pooled" else
                              torch.full((n,), 1.0 / n))
        self.b0 = nn.Parameter(torch.zeros(()))
        self.aux = None
        if aux_lines:
            if head_seed is not None:
                torch.manual_seed(head_seed)
            self.aux = LineQueries(width, heads, lay, mlp)
        if head != "pooled":                               # the read lines' columns of A (the vote: all of them)
            self.register_buffer("Ah", self.A[:, self.head.lines].clone(), persistent=False)

    def gsign(self, bits):                                 # (B, m) -> (B, d) float, (-1)^g_j
        with torch.autocast("cuda", enabled=False):
            g = torch.remainder(bits.float() @ self.A, 2.0)
        return 1.0 - 2.0 * g

    def head_sign(self, bits):                             # (B, m) -> (B, n) float, (-1)^g_j of the head's read lines
        with torch.autocast("cuda", enabled=False):
            g = torch.remainder(bits.float() @ self.Ah, 2.0)
        return 1.0 - 2.0 * g

    def features(self, bits):                              # (B, m) -> (B, m, C)
        B = bits.shape[0]
        T, H, W = self.shape
        C = self.C
        pad = torch.cat([bits.to(self.const.dtype), bits.new_zeros(B, 1, dtype=self.const.dtype)], 1)
        grid = pad[:, self.gidx].view(B, 1, T, H, W)
        st = self.stem
        y = F.conv3d(grid.to(st.weight.dtype), st.weight[:, :1], None, padding=st.padding) + \
            F.conv3d(self.const, st.weight[:, 1:], st.bias, padding=st.padding)
        y = y.permute(0, 2, 3, 4, 1)
        x = y[..., :C] * self.mx + y[..., C:] * self.mz
        for blk in self.blocks:
            x = blk(x, self.mx, self.mz)
        return x.reshape(B, T * H * W, C)[:, self.flat]

    def decide(self, bits, z):                             # head outputs (B, n) -> c (B,) fp32
        if self.kind == "pooled":
            return (z.float() * self.w).sum(1) + self.b0
        return (z.float() * self.head_sign(bits) * self.w).sum(1) + self.b0

    def forward(self, bits):                               # (B, m) -> (c (B,), z (B, n)): the decision path alone
        z = self.head(self.features(bits))
        return self.decide(bits, z), z

    def forward_train(self, bits):                         # (B, m) -> (c, line logits (B, d) of the line loss or None)
        f = self.features(bits)
        z = self.head(f)
        c = self.decide(bits, z)
        if self.kind == "vote":
            return c, z
        return c, (self.aux(f) if self.aux is not None else None)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def trunk_keys(model):
    """State-dict keys of the trunk (stem and blocks): what a trunk-only --init carries."""
    return [k for k in model.state_dict() if k.startswith(("stem.", "blocks."))]


def export_decision(ck):
    """A checkpoint dict {'state', 'cfg', ...} -> the same without the training-only read-out (aux.*): the inference
    graph of every arm is ConvLines(**cfg) with aux_lines False, loaded strictly from this state."""
    cfg = dict(ck["cfg"])
    drop = [k for k in ck["state"] if k.startswith("aux.")]
    assert bool(drop) == bool(cfg.get("aux_lines", False)), "aux weights and aux_lines disagree"
    if "aux_lines" in cfg:
        cfg["aux_lines"] = False
    return {**ck, "state": {k: v for k, v in ck["state"].items() if not k.startswith("aux.")}, "cfg": cfg,
            "dropped": drop}


def net_from_cfg(cfg):
    """ConvLines from a saved configuration (old configurations: the vote head)."""
    return ConvLines(cfg["d"], **{k: cfg[k] for k in ("width", "bott", "blocks", "heads", "stem_k", "block_k", "mlp")},
                     head=cfg.get("head", "vote"), line_index=cfg.get("line_index"), aux_lines=cfg.get("aux_lines", False))


def head_macs(cfg, m):
    """Multiply-adds per shot of the read-out on the decision path (a training-only read-out excluded), n queries:
    inference form (net.flops' read-out rows: the query-key product folded into one (C, n H) matrix, as deploy_tables
    does; it folds the pooled head's constant queries alike) and training form (keys m C C, then scores m n C), plus the
    parity products of the read lines (bits (m) x A (m, n), counted m n; none for the pooled head) and the n-term vote or
    linear map. The pooled head's position embedding is an addition (m C adds, no multiply)."""
    C, d, H, mlp = cfg["width"], cfg["d"], cfg["heads"], cfg["mlp"]
    head = cfg.get("head", "vote")
    n = {"vote": d, "line": 1, "pooled": d}[head]
    common = {"values (m C C)": m * C * C, "gated sum (m n C)": m * n * C,
              "out-proj + MLP": n * (C * C + C * mlp * C + mlp * C), "parity (m n)": 0 if head == "pooled" else m * n,
              "vote / linear map (n)": n}
    inf = {"scores (m C n H, folded)": m * C * n * H, **common}
    inf["total"] = sum(inf.values())
    return {"n_queries": n, "inference": inf, "training_form_total": inf["total"] - m * C * n * H + m * C * C + m * n * C}


def flops(cfg, m):
    """Multiply-adds per shot of the inference form (arch.engine), by part."""
    C, Cb, L, d, H = cfg["width"], cfg["bott"], cfg["blocks"], cfg["d"], cfg["heads"]
    ks, kb = int(np.prod(cfg["stem_k"])), int(np.prod(cfg["block_k"]))
    mlp = cfg["mlp"]
    out = {"stem (dense tap matmul)": m * ks * C, "down-projections": L * m * C * Cb, "block convolutions": L * m * kb * Cb * Cb,
           "up-projections": L * m * Cb * C, "read-out scores + values": m * C * (d * H + C),
           "read-out pooling": m * d * C, "read-out out-proj + MLP": d * (C * C + C * mlp * C + mlp * C)}
    out["total"] = sum(out.values())
    return out


@torch.no_grad()
def deploy_tables(model):
    """The inference form (fp32 tensors on the model's device) of a ConvLines model, engine site order."""
    assert model.kind == "vote", "the engines' inference form exists for the vote head alone"
    lay = model.lay
    dev = model.A.device
    order, nx = lay.site_order()
    C, d = model.C, model.d
    T, H, W = model.shape
    st = model.stem
    cfg = model.cfg
    ks, kb = tuple(cfg["stem_k"]), tuple(cfg["block_k"])
    oidx = torch.as_tensor(order, device=dev)
    typ = torch.as_tensor((~lay.is_x[order]).astype(np.int64), device=dev)                        # 0 X, 1 Z
    yc = F.conv3d(model.const, st.weight[:, 1:], st.bias, padding=st.padding)[0].permute(1, 2, 3, 0).reshape(-1, 2 * C)
    yc = yc[model.flat[oidx]]                                                                     # (m, 2C)
    P = torch.where(typ[:, None] == 0, yc[:, :C], yc[:, C:])
    wb = st.weight[:, 0].reshape(2 * C, -1).t()                                                  # (taps, 2C)
    t = {"order": oidx, "nx": nx, "det": oidx, "P": P.contiguous(),
         "Wb": torch.stack([wb[:, :C], wb[:, C:]]).contiguous(),                                 # (2, taps, C)
         "nb_stem": torch.as_tensor(lay.neighbours(order, ks), device=dev),
         "nb_blk": torch.as_tensor(lay.neighbours(order, kb), device=dev), "blocks": []}
    for blk in model.blocks:
        Cb = blk.Cb
        wc = blk.conv.weight.reshape(2 * Cb, Cb, -1).permute(2, 1, 0)                            # (taps, Cb_in, 2Cb)
        t["blocks"].append({"ln_w": blk.ln.weight.clone(), "ln_b": blk.ln.bias.clone(), "eps": blk.ln.eps,
                            "Wd": blk.down.weight.t().contiguous(), "bd": blk.down.bias.clone(),
                            "Wc": torch.stack([wc[..., :Cb].reshape(-1, Cb), wc[..., Cb:].reshape(-1, Cb)]).contiguous(),
                            "bc": torch.stack([blk.conv.bias[:Cb], blk.conv.bias[Cb:]]).contiguous(),
                            "Wu": blk.up.weight.t().contiguous(), "bu": blk.up.bias.clone()})
    hd = model.head
    Hh, dh = hd.h, hd.dh
    q = hd.wq(hd.queries()).view(d, Hh, dh) * dh ** -0.5                                            # (d, H, dh)
    Qm = torch.zeros(C, d * Hh, device=dev)
    for j in range(d):
        for h in range(Hh):
            Qm[h * dh:(h + 1) * dh, j * Hh + h] = q[j, h]
    t["head"] = {"ln_w": hd.ln.weight.clone(), "ln_b": hd.ln.bias.clone(), "eps": hd.ln.eps,
                 "Wsc": (hd.k.weight.t() @ Qm).contiguous(),                                           # (C, d H)
                 "BIAS": hd.bias()[oidx].reshape(lay.m, d * Hh).contiguous(),                          # (m, d H)
                 "Wv": hd.v.weight.t().contiguous(), "bv": hd.v.bias.clone(),
                 "Wo": hd.o.weight.t().contiguous(), "bo": hd.o.bias.clone(), "U0": hd.queries().clone(),
                 "m_ln_w": hd.mlp[0].weight.clone(), "m_ln_b": hd.mlp[0].bias.clone(), "m_eps": hd.mlp[0].eps,
                 "W1": hd.mlp[1].weight.t().contiguous(), "b1": hd.mlp[1].bias.clone(),
                 "W2": hd.mlp[3].weight.t().contiguous(), "b2": hd.mlp[3].bias.clone(), "H": Hh, "dh": dh,
                 "n0": hd.n0}
    t["A"] = model.A[oidx].contiguous()                                                           # (m, d), engine order
    t["shape"] = (T, H, W)
    t["tij"] = torch.as_tensor(lay.idx[order], device=dev)                                       # (m, 3) grid position
    t["w"], t["b0"] = model.w.clone(), model.b0.clone()
    t["cfg"] = dict(cfg, m=lay.m)
    return _detached(t)


def _detached(t):
    """Every tensor of the tables detached from autograd and contiguous (views of parameters would otherwise keep
    requires_grad, which the ONNX export refuses as a constant and torch.save keeps)."""
    if torch.is_tensor(t):
        return t.detach().clone().contiguous()
    if isinstance(t, dict):
        return {k: _detached(v) for k, v in t.items()}
    if isinstance(t, list):
        return [_detached(v) for v in t]
    return t


class DeployRef(nn.Module):
    """The inference form with plain torch ops: detector bits (B, m) uint8 -> (c (B,), z (B, d)). dtype: the matmul
    input type (torch.float32 for the reference; torch.float16 with fp32 stream otherwise). Missing neighbours index the
    zero row m (no negative gather indices, for the ONNX / TensorRT export)."""

    def __init__(self, t, dtype=torch.float32):
        super().__init__()
        self.t, self.dt = t, dtype
        self.nx = t["nx"]
        m = t["cfg"]["m"]
        self.nbs = torch.where(t["nb_stem"] >= 0, t["nb_stem"], m)
        self.nbb = torch.where(t["nb_blk"] >= 0, t["nb_blk"], m)

    def mm(self, a, w):
        return (a.to(self.dt) @ w.to(self.dt)).float()

    def typed(self, a, W, b=None):                         # rows [0, nx) X weights, [nx, m) Z weights
        nx = self.nx
        y = torch.cat([self.mm(a[:, :nx], W[0]), self.mm(a[:, nx:], W[1])], 1)
        if b is not None:
            y = y + torch.cat([b[0].expand(nx, -1), b[1].expand(a.shape[1] - nx, -1)], 0)
        return y

    def conv(self, b):                                     # (B, m) engine-order bits -> site features (B, m, C)
        t = self.t
        B = b.shape[0]
        bp = torch.cat([b, b.new_zeros(B, 1)], 1)
        x = t["P"] + self.typed(bp[:, self.nbs], t["Wb"])
        for p in t["blocks"]:
            u = F.silu(self.mm(F.layer_norm(x, x.shape[-1:], p["ln_w"], p["ln_b"], p["eps"]), p["Wd"]) + p["bd"])
            up = torch.cat([u, u.new_zeros(B, 1, u.shape[-1])], 1)
            g = up[:, self.nbb].reshape(B, u.shape[1], -1)                                       # (B, m, taps Cb)
            v = F.silu(self.typed(g, p["Wc"], p["bc"]))
            x = x + self.mm(v, p["Wu"]) + p["bu"]
        return x

    def readout(self, x, b):                               # site features (B, m, C), bits (B, m), engine order
        t, h = self.t, self.t["head"]
        B, m = b.shape
        y = F.layer_norm(x, x.shape[-1:], h["ln_w"], h["ln_b"], h["eps"])
        s = self.mm(y, h["Wsc"]) + h["BIAS"]                                                      # (B, m, d H)
        pr = torch.sigmoid(s) / h["n0"]
        v = self.mm(y, h["Wv"]) + h["bv"]                                                         # (B, m, C)
        d, H, dh = t["cfg"]["d"], h["H"], h["dh"]
        a = self.mm(pr.view(B, m, d, H).permute(0, 3, 2, 1), v.view(B, m, H, dh).permute(0, 2, 1, 3))   # (B, H, d, dh)
        a = a.permute(0, 2, 1, 3).reshape(B, d, H * dh)
        u = h["U0"] + self.mm(a, h["Wo"]) + h["bo"]
        z = (self.mm(F.silu(self.mm(F.layer_norm(u, u.shape[-1:], h["m_ln_w"], h["m_ln_b"], h["m_eps"]), h["W1"]) +
                            h["b1"]), h["W2"]) + h["b2"]).squeeze(-1)                             # (B, d)
        gs = b @ t["A"]
        g = gs - 2.0 * torch.floor(gs * 0.5)
        c = (z * (1.0 - 2.0 * g) * t["w"]).sum(1) + t["b0"]
        return c, z

    def forward(self, bits):
        b = bits[:, self.t["det"]].float()
        return self.readout(self.conv(b), b)


class DenseRef(nn.Module):
    """The training form's dense 3D convolutions (cuDNN) followed by DeployRef's read-out: bits (B, m) -> (c, z)."""

    def __init__(self, model, t, dtype=torch.float32):
        super().__init__()
        self.model, self.ref = model, DeployRef(t, dtype)

    def forward(self, bits):
        b = bits[:, self.ref.t["det"]].float()
        x = self.model.features(bits)[:, self.ref.t["det"]].float()
        return self.ref.readout(x, b)
