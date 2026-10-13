"""Cascade (Gu, Bonilla Ataides, Lukin & Yelin, arXiv:2604.08358), re-implemented for BB codes on the torus: the network.
Every choice the paper leaves open was listed in the re-implementation's notes, which are not part of this release;
the short form is here.

Nodes. Check nodes are the detectors' places (bb.code.Layout): index (c T + t) n2 + s with c = 0 X / 1 Z, t = 0 .. T-1
(T = R + 1: R rounds and the final data-measurement layer), s = a m + b the site of x^a y^b. In an X memory the Z checks
have no detector at t = 0 and t = R; those places hold the token 'absent'. Data nodes have the same index with
d = 0 L / 1 R in place of c, so data and check nodes share the time axis.

Convolution (Methods, "Geometric structure and convolution"): a check -> check convolution on the torus factored into
two bipartite steps, check -> data -> check, each over the 6 Tanner neighbours of the target at 2 time offsets (12
relations): data (d, t, s) <- checks at t and t + 1; check (c, t, s) <- data at t - 1 and t (so the composite reaches
checks at t - 1, t, t + 1: the paper's "3 temporal offsets"). Sources outside 0 .. T-1 are zero (zero padding in time);
space is periodic. Each target type (L, R, X, Z) has its own 12 relation matrices; a relation is (neighbour, dt), and
the neighbour is a fixed offset on the torus, so the convolution is translation equivariant. The two steps are linear
(a factorisation of the check -> check convolution: no norm or activation between them) and bias-free.

Block (Extended Data Fig. 1): BN -> SiLU -> project H -> H/4 -> BN -> SiLU -> convolution (at H/4) -> BN -> SiLU ->
project H/4 -> H, and h + f(h) / sqrt(2 L) ('branch'; 'skip' = the figure's literal h / sqrt(2 L) + f(h)).
Read-out (Methods, "Architecture"): BN -> SiLU, a check -> data scatter convolution at width H (12 relations per data
type, as above), the average over every data node (all T layers) in each observable's support, then a two-layer MLP
(H -> 2 H -> 1, SiLU) shared by the observables: one logit per observable. Scatter and pooling are both linear, so they
are computed as pooling first (the pooled sources of every relation) and one matrix product (exact; test_cascade)."""
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from qnd import bb as qbb

try:                                                    # the Triton kernels (GPU hosts); the PyTorch path otherwise
    from bb.cascade_triton import GatherMMTriton
except ImportError:
    GatherMMTriton = None

PARAMS = qbb.PARAMS                                    # code -> (l, m, a, b)
ROUNDS = {"bb72": 6, "bb144": 12}
DT_C2D = (0, 1)                                         # data (t) <- checks (t + dt)
DT_D2C = (-1, 0)                                        # check (t) <- data (t + dt)
ABSENT = 4                                              # tokens: 2 c + bit for a detector, 4 for a place without one


def neighbour_offsets(code):
    """The Tanner graph as offsets on the torus. -> {('c2d', d): [(c, (da, db)) x 6], ('d2c', c): [(d, (da, db)) x 6]}:
    the target of type d (data L/R) or c (check X/Z) at site s is adjacent to the source of the listed type at site
    s + (da, db), for every s (checked here). Sources are listed X then Z (or L then R), each sorted by offset."""
    l, m = PARAMS[code][:2]
    n2 = l * m
    hx, hz, _, _ = qbb.matrices(code)
    H = (hx, hz)
    a, b = np.divmod(np.arange(n2), m)

    def offs(nbrs, s):
        return sorted(((int(a[u] - a[s]) % l, int(b[u] - b[s]) % m) for u in nbrs))

    out = {}
    for d in range(2):
        nb = []
        for c in range(2):
            blk = H[c][:, d * n2:(d + 1) * n2]          # (check sites, data sites of type d)
            o0 = offs(np.flatnonzero(blk[:, 0]), 0)
            for s in range(n2):
                if offs(np.flatnonzero(blk[:, s]), s) != o0:
                    raise ValueError(f"{code}: the Tanner graph is not translation invariant at data ({d}, {s})")
            nb += [(c, o) for o in o0]
        out["c2d", d] = nb
    for c in range(2):
        nb = []
        for d in range(2):
            blk = H[c][:, d * n2:(d + 1) * n2]
            o0 = offs(np.flatnonzero(blk[0]), 0)
            for s in range(n2):
                if offs(np.flatnonzero(blk[s]), s) != o0:
                    raise ValueError(f"{code}: the Tanner graph is not translation invariant at check ({c}, {s})")
            nb += [(d, o) for o in o0]
        out["d2c", c] = nb
    return out


class Geometry:
    """Node layout and gather tables of a BB code with R rounds (see the module docstring).
    c2d[d]: (T n2, 12) source check index of target data (d, t, s) for relation r = 2 j + k (neighbour j of
            nb[('c2d', d)], dt = DT_C2D[k]); N = the zero pad row.  d2c[c]: (T n2, 12) source data index, likewise.
    The inverse tables (inv_*) give, for every source node u and relation r, the flat position n 12 + r of the target
    n that reads u through r (N_t 12 = none); every relation reads each source at most once."""

    def __init__(self, code, rounds=None):
        self.code = code
        self.l, self.m = l, m = PARAMS[code][:2]
        self.n2 = n2 = l * m
        self.R = ROUNDS[code] if rounds is None else rounds
        self.T = T = self.R + 1
        self.N = N = 2 * T * n2                         # check nodes = data nodes
        self.nb = nb = neighbour_offsets(code)
        if any(len(v) != 6 for v in nb.values()):
            raise ValueError(f"{code}: expected 6 Tanner neighbours per node")
        a, b = np.divmod(np.arange(n2), m)
        t = np.arange(T)

        def table(src_nb, dts):
            idx = np.full((T, n2, 12), N, np.int64)
            for j, (ty, (da, db)) in enumerate(src_nb):
                site = ((a + da) % l) * m + (b + db) % m
                for k, dt in enumerate(dts):
                    ts = t + dt
                    ok = (ts >= 0) & (ts < T)
                    idx[ok, :, 2 * j + k] = (ty * T + ts[ok])[:, None] * n2 + site[None, :]
            return idx.reshape(T * n2, 12)

        self.c2d = [table(nb["c2d", d], DT_C2D) for d in range(2)]
        self.d2c = [table(nb["d2c", c], DT_D2C) for c in range(2)]
        self.inv_c2d = [self.inverse(x) for x in self.c2d]
        self.inv_d2c = [self.inverse(x) for x in self.d2c]

    def inverse(self, idx):
        nt, K = idx.shape
        inv = np.full((self.N, K), nt * K, np.int64)
        n, r = np.nonzero(idx < self.N)
        u = idx[n, r]
        if len(np.unique(u * K + r)) != len(u):
            raise ValueError("a relation reads one source twice")
        inv[u, r] = n * K + r
        return inv

    def site_perm(self, g):
        """Site permutation of the translation by g: s = (a, b) -> (a + g0, b + g1)."""
        a, b = np.divmod(np.arange(self.n2), self.m)
        return ((a + g[0]) % self.l) * self.m + (b + g[1]) % self.m

    def node_perm(self, g):
        """Node permutation (checks or data alike): node (ty, t, s) -> (ty, t, s + g)."""
        ty_t, s = np.divmod(np.arange(self.N), self.n2)
        return ty_t * self.n2 + self.site_perm(g)[s]


class _GatherMM(torch.autograd.Function):
    """y[b, n] = W cat_r x[b, idx[n, r]] (x padded with a zero row at index N): one bipartite step as an implicit GEMM.
    Saves x, not the 12-fold gathered tensor, and gathers again in the backward; the input gradient is a gather through
    the inverse table (deterministic, no atomics)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x, w, idx, inv):
        B, N, C = x.shape
        nt, K = idx.shape
        xp = torch.cat([x, x.new_zeros(B, 1, C)], 1)
        g = xp.index_select(1, idx.reshape(-1)).view(B, nt, K * C)
        ctx.save_for_backward(x, w, idx, inv)
        return g @ w.t()

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gy):
        x, w, idx, inv = ctx.saved_tensors
        B, N, C = x.shape
        nt, K = idx.shape
        gy = gy.contiguous()
        gx = gw = None
        if ctx.needs_input_grad[1]:
            xp = torch.cat([x, x.new_zeros(B, 1, C)], 1)
            g = xp.index_select(1, idx.reshape(-1)).view(B * nt, K * C)
            gw = gy.view(B * nt, -1).t() @ g
        if ctx.needs_input_grad[0]:
            gg = (gy @ w).view(B, nt * K, C)
            gg = torch.cat([gg, gg.new_zeros(B, 1, C)], 1)
            gx = gg.index_select(1, inv.reshape(-1)).view(B, N, K, C).sum(2)
        return gx, gw, None, None


def gather_mm(x, w, idx, inv):
    return _GatherMM.apply(x, w, idx, inv)


def gather_mm_ref(x, w, idx):
    """Plain autograd reference of gather_mm (tests)."""
    B, N, C = x.shape
    xp = torch.cat([x, x.new_zeros(B, 1, C)], 1)
    return xp[:, idx.reshape(-1)].reshape(B, idx.shape[0], -1) @ w.t()


_RECOMPUTE = [False]                                    # True while activation checkpointing recomputes a block


@contextmanager
def _recomputing():
    _RECOMPUTE[0] = True
    try:
        yield
    finally:
        _RECOMPUTE[0] = False


def bn(mod, x):
    """BatchNorm over every node of every shot, per channel ((B, N, C) -> the same). Running statistics are not
    updated while a checkpointed block is recomputed (they were updated in its first forward)."""
    C = x.shape[-1]
    x2 = x.reshape(-1, C)
    if not mod.training:
        return F.batch_norm(x2, mod.running_mean, mod.running_var, mod.weight, mod.bias, False, 0.0, mod.eps).view(x.shape)
    rm, rv = mod.running_mean, mod.running_var
    if _RECOMPUTE[0]:
        rm, rv = rm.clone(), rv.clone()                 # same graph as the first forward, statistics left alone
    else:
        mod.num_batches_tracked.add_(1)
    mom = mod.momentum if mod.momentum is not None else 1.0 / float(mod.num_batches_tracked)
    return F.batch_norm(x2, rm, rv, mod.weight, mod.bias, True, mom, mod.eps).view(x.shape)


class BipartiteConv(nn.Module):
    """check -> data -> check at width C (12 relations per target type and step; linear, no bias). impl 'torch':
    _GatherMM (gather + GEMM); 'triton': cascade_triton.GatherMMTriton (implicit GEMM, the same arithmetic)."""

    def __init__(self, C, geo, impl="torch"):
        super().__init__()
        self.C, self.impl = C, impl
        std = 1 / math.sqrt(12 * C)
        self.w_c2d = nn.ParameterList([nn.Parameter(torch.randn(C, 12 * C) * std) for _ in range(2)])
        self.w_d2c = nn.ParameterList([nn.Parameter(torch.randn(C, 12 * C) * std) for _ in range(2)])
        for name in ("c2d", "d2c"):
            for k in range(2):
                self.register_buffer(f"{name}{k}", torch.as_tensor(getattr(geo, name)[k]), persistent=False)
                self.register_buffer(f"inv_{name}{k}", torch.as_tensor(getattr(geo, "inv_" + name)[k]), persistent=False)

    def forward(self, x):                               # (B, N checks, C) -> (B, N checks, C)
        op = GatherMMTriton.apply if self.impl == "triton" and x.is_cuda and GatherMMTriton is not None else gather_mm
        data = torch.cat([op(x, self.w_c2d[d], getattr(self, f"c2d{d}"), getattr(self, f"inv_c2d{d}"))
                          for d in range(2)], 1)
        return torch.cat([op(data, self.w_d2c[c], getattr(self, f"d2c{c}"), getattr(self, f"inv_d2c{c}"))
                          for c in range(2)], 1)


class Block(nn.Module):
    """Bottleneck residual block (Extended Data Fig. 1)."""

    def __init__(self, H, geo, L, res="branch", impl="torch"):
        super().__init__()
        Hb = H // 4
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(H), nn.BatchNorm1d(Hb), nn.BatchNorm1d(Hb)
        self.down = nn.Linear(H, Hb, bias=False)
        self.conv = BipartiteConv(Hb, geo, impl)
        self.up = nn.Linear(Hb, H, bias=False)
        for lin in (self.down, self.up):
            nn.init.normal_(lin.weight, std=1 / math.sqrt(lin.in_features))
        self.res, self.scale = res, 1 / math.sqrt(2 * L)

    def branch(self, h):
        u = self.down(F.silu(bn(self.bn1, h)))
        u = self.conv(F.silu(bn(self.bn2, u)))
        return self.up(F.silu(bn(self.bn3, u)))

    def forward(self, h):
        u = self.branch(h)
        if self.res == "branch":
            return h + self.scale * u
        return self.scale * h + u


class CascadeBB(nn.Module):
    """Cascade for a BB code memory: detector bits (B, D) -> one logit per observable (B, k).
    det_t, det_typ, det_site: each detector's place (bb.code.Layout); logicals (k, 2 n2): the observables' supports on
    the data (L sites then R sites)."""

    def __init__(self, code, H, L, det_t, det_typ, det_site, logicals, res="branch", act_ckpt=False, rounds=None,
                 resid_bf16=False, conv_impl="torch"):
        super().__init__()
        if H % 4:
            raise ValueError("H must be a multiple of 4")
        geo = self.geo = Geometry(code, rounds)
        self.H, self.L, self.act_ckpt, self.resid_bf16 = H, L, act_ckpt, resid_bf16
        self.D = D = len(det_t)
        T, n2, N = geo.T, geo.n2, geo.N
        node = (np.asarray(det_typ) * T + np.asarray(det_t)) * n2 + np.asarray(det_site)
        if len(np.unique(node)) != D or node.max() >= N:
            raise ValueError("detector places collide or fall outside the grid")
        gidx = np.full(N, D, np.int64)
        gidx[node] = np.arange(D)
        tbase = np.full(N, ABSENT, np.int64)
        tbase[node] = 2 * np.asarray(det_typ)
        self.register_buffer("gidx", torch.as_tensor(gidx), persistent=False)
        self.register_buffer("tbase", torch.as_tensor(tbase), persistent=False)
        self.embed = nn.Embedding(5, H)
        self.blocks = nn.ModuleList([Block(H, geo, L, res, conv_impl) for _ in range(L)])
        self.bn_out = nn.BatchNorm1d(H)
        # read-out: scatter (12 relations per data type, H -> H) + average over each support's data nodes
        logicals = np.asarray(logicals, np.int64)
        self.k = k = len(logicals)
        self.support = [np.flatnonzero(r) for r in logicals]   # indices d n2 + s
        pool = np.zeros((k, 2, 12, N + 1))
        for o in range(k):
            sup = np.flatnonzero(logicals[o])
            w = 1.0 / (T * len(sup))
            for q in sup:
                d, s = divmod(int(q), n2)
                for t in range(T):
                    np.add.at(pool[o, d], (np.arange(12), geo.c2d[d][t * n2 + s]), w)
        self.register_buffer("pool", torch.as_tensor(pool[..., :N].reshape(k * 24, N), dtype=torch.float32),
                             persistent=False)
        self.scatter = nn.Parameter(torch.randn(H, 24 * H) / math.sqrt(12 * H))   # (out, (d, r, in))
        self.mlp1 = nn.Linear(H, 2 * H)
        self.mlp2 = nn.Linear(2 * H, 1)
        nn.init.normal_(self.mlp1.weight, std=1 / math.sqrt(H))
        nn.init.zeros_(self.mlp1.bias)
        nn.init.zeros_(self.mlp2.weight)
        nn.init.zeros_(self.mlp2.bias)

    def tokens(self, det):
        det = F.pad(det.to(torch.uint8), (0, 1))
        return self.tbase + det[:, self.gidx].long()

    def backbone(self, tok):
        h = self.embed(tok)
        if self.resid_bf16 and torch.is_autocast_enabled():
            h = h.to(torch.bfloat16)                    # the residual stream in bf16 (an option; default fp32)
        for blk in self.blocks:
            if self.act_ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False,
                                                      context_fn=lambda: (_null(), _recomputing()))
            else:
                h = blk(h)
        return F.silu(bn(self.bn_out, h))

    def readout(self, h):
        B = h.shape[0]
        pooled = torch.matmul(self.pool.to(h.dtype), h)            # (B, 24 k, H): pooled sources of every relation
        z = pooled.view(B, self.k, 24 * self.H) @ self.scatter.t().to(pooled.dtype)
        return self.mlp2(F.silu(self.mlp1(z))).squeeze(-1)

    def readout_ref(self, h):
        """The read-out as written in the paper (scatter to data nodes, then the average over each support): tests."""
        geo, B, H = self.geo, h.shape[0], self.H
        W = self.scatter.view(H, 2, 12 * H)
        data = torch.stack([gather_mm_ref(h, W[:, d], torch.as_tensor(geo.c2d[d], device=h.device))
                            for d in range(2)], 1)                  # (B, 2, T n2, H)
        data = data.view(B, 2, geo.T, geo.n2, H).transpose(2, 3).reshape(B, 2 * geo.n2, geo.T, H)
        z = torch.stack([data[:, torch.as_tensor(sup, device=h.device)].mean((1, 2)) for sup in self.support], 1)
        return self.mlp2(F.silu(self.mlp1(z))).squeeze(-1)

    def forward(self, det):
        return self.readout(self.backbone(self.tokens(det)))


@contextmanager
def _null():
    yield


def muon_params(model):
    """The matrices Muon trains (Methods: "matrix-valued parameters (convolution weights)"): the backbone's projections
    and relation matrices. Everything else (embedding, BN, read-out scatter and MLP) goes to Lion."""
    return [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]


def build(cfg, layout=None):
    """cfg: {code, H, L, res, act_ckpt} -> CascadeBB on the code's Relay-BP circuit layout (bb.code.Layout)."""
    if layout is None:
        from bb import code as bbcode
        layout = bbcode.Layout(cfg["code"])
    m = CascadeBB(cfg["code"], cfg["H"], cfg["L"], layout.t, layout.typ, layout.site, layout.logicals,
                  res=cfg.get("res", "branch"), act_ckpt=cfg.get("act_ckpt", False),
                  resid_bf16=cfg.get("resid_bf16", False), conv_impl=cfg.get("conv_impl", "torch"))
    m.cfg = dict(cfg)
    return m


def macs_per_shot(H, L, N=504, k=12):
    """Multiply-accumulates per shot: per block 2 projections (N H H/4 each) and 2 bipartite steps (N 12 (H/4)^2
    each); the read-out's pooled scatter and MLP."""
    Hb = H // 4
    block = 2 * N * H * Hb + 2 * N * 12 * Hb * Hb
    return L * block + 24 * k * N * H + k * 24 * H * H + k * (2 * H * H + 2 * H)
