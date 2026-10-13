"""Cascade (Gu, Bonilla Ataides, Lukin & Yelin, arXiv:2604.08358) for a rotated surface-code memory: the network.
The BB re-implementation (ising_compare/bb/cascade_net.py on ising-cascade-net, 64b2e25d) with the code convolution
replaced by the paper's surface-code one; every choice the paper leaves open is listed in cascade_compare/SURFACE_PLAN.md
(section 3), the short form is here.

Grid (Methods, "Geometric structure and convolution": "For surface codes, the syndrome is a 3D tensor of shape
(R, d+1, d+1)"). Stabilizer places of the rotated code on the (d+1) x (d+1) plaquette grid, measure qubit (2j, 2i) of
Stim's coordinates at row i, column j, so X and Z alternate on a checkerboard. Time t = 0 .. T-1 with T = R + 1: the R
rounds and the layer of detectors built from the final data measurement, so every detector has its own place (the
paper's R slices would merge the last two; its d = 19 count, 6,840 = 19 x 360 positions, equals the detector count of
this layout). Node n = (t G + i) G + j, G = d + 1. A place without a detector (a grid point without a stabilizer, the
Z checks at t = 0 and t = R of an X memory) holds the token 'absent'; a detector holds 2 c + bit (c = 0 X, 1 Z).

Convolution: "a standard 3D convolution with zero padding at boundaries" (3 x 3 x 3, ED Fig. 4: K = 27), stride 1,
bias-free, at width H/4 inside the bottleneck block. Block (Extended Data Fig. 1): BN -> SiLU -> project H -> H/4 ->
BN -> SiLU -> 3x3x3 convolution -> BN -> SiLU -> project H/4 -> H; residual 'skip' = h / sqrt(2 L) + f(h) (the figure's
literal form, the recipe the BB pilot chose: ising_compare/bb/cascade/stage2/PILOT.md) or 'branch' = h + f(h) / sqrt(2L).
Tensors are (B, C, T, G, G) in channels_last_3d memory, so the projections are 1x1x1 convolutions (GEMMs over C).

Read-out (Methods, "Architecture"): BN -> SiLU, "a convolution scatters the check-node representations to data
qubits", the average over the data qubits of the observable's support (all T layers), a two-layer MLP (H -> 2H -> 1,
SiLU): one logit. The scatter is the surface-code analogue of the BB one: data qubit (r, c) at Stim's (2c + 1, 2r + 1)
reads the four plaquette places around it, (i, j) = (r + di, c + dj) with di, dj in {0, 1}, at times t and t + 1
(zero past T - 1): a 2 x 2 x 2 kernel, width H -> H, data nodes (T, d, d). Scatter and average are linear, so they are
computed as the average of each kernel tap's sources followed by one matrix product (readout_ref is the literal form;
test_surface checks that they agree)."""
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

ABSENT = 4                                              # tokens: 2 c + bit for a detector, 4 for a place without one
TAPS = [(dt, di, dj) for dt in (0, 1) for di in (0, 1) for dj in (0, 1)]   # the read-out scatter's kernel taps

_RECOMPUTE = [False]                                    # True while activation checkpointing recomputes a block


@contextmanager
def _recomputing():
    _RECOMPUTE[0] = True
    try:
        yield
    finally:
        _RECOMPUTE[0] = False


@contextmanager
def _null():
    yield


def bn(mod, x):
    """BatchNorm over every node of every shot, per channel ((B, C, T, G, G)). Running statistics are not updated while
    a checkpointed block is recomputed (they were updated in its first forward)."""
    if not mod.training:
        return F.batch_norm(x, mod.running_mean, mod.running_var, mod.weight, mod.bias, False, 0.0, mod.eps)
    rm, rv = mod.running_mean, mod.running_var
    if _RECOMPUTE[0]:
        rm, rv = rm.clone(), rv.clone()                 # same graph as the first forward, statistics left alone
    else:
        mod.num_batches_tracked.add_(1)
    mom = mod.momentum if mod.momentum is not None else 1.0 / float(mod.num_batches_tracked)
    return F.batch_norm(x, rm, rv, mod.weight, mod.bias, True, mom, mod.eps)


def _conv(cin, cout, k):
    c = nn.Conv3d(cin, cout, k, padding=k // 2, bias=False)
    nn.init.normal_(c.weight, std=1 / math.sqrt(cin * k ** 3))   # fan-in, as the BB projections and relations
    return c


class Block(nn.Module):
    """Bottleneck residual block (Extended Data Fig. 1) with the 3x3x3 space-time convolution."""

    def __init__(self, H, L, res="skip"):
        super().__init__()
        Hb = H // 4
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm3d(H), nn.BatchNorm3d(Hb), nn.BatchNorm3d(Hb)
        self.down = _conv(H, Hb, 1)
        self.conv = _conv(Hb, Hb, 3)
        self.up = _conv(Hb, H, 1)
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


class CascadeSurface(nn.Module):
    """Cascade for a rotated surface-code memory of distance d over R rounds: detector bits (B, D) -> one logit (B, 1).
    det_t, det_typ, det_i, det_j: each detector's place (surface_layout.Layout); support: the observable's data qubits
    as (r, c) pairs."""

    def __init__(self, d, R, H, L, det_t, det_typ, det_i, det_j, support, res="skip", act_ckpt=False, resid_bf16=False):
        super().__init__()
        if H % 4:
            raise ValueError("H must be a multiple of 4")
        self.d, self.R, self.H, self.L, self.act_ckpt, self.resid_bf16 = d, R, H, L, act_ckpt, resid_bf16
        self.T, self.G = T, G = R + 1, d + 1
        self.N = N = T * G * G
        self.D = D = len(det_t)
        self.k = 1
        node = (np.asarray(det_t) * G + np.asarray(det_i)) * G + np.asarray(det_j)
        if len(np.unique(node)) != D or node.min() < 0 or node.max() >= N:
            raise ValueError("detector places collide or fall outside the grid")
        gidx = np.full(N, D, np.int64)
        gidx[node] = np.arange(D)
        tbase = np.full(N, ABSENT, np.int64)
        tbase[node] = 2 * np.asarray(det_typ)
        self.register_buffer("gidx", torch.as_tensor(gidx), persistent=False)
        self.register_buffer("tbase", torch.as_tensor(tbase), persistent=False)
        self.embed = nn.Embedding(5, H)
        self.blocks = nn.ModuleList([Block(H, L, res) for _ in range(L)])
        self.bn_out = nn.BatchNorm3d(H)
        # read-out: scatter (2 x 2 x 2 taps, H -> H) + average over the support's data nodes, as pooling then a product
        self.support = np.asarray(support, np.int64).reshape(-1, 2)
        pool = np.zeros((len(TAPS), N))
        w = 1.0 / (T * len(self.support))
        for k, (dt, di, dj) in enumerate(TAPS):
            for r, c in self.support:
                for t in range(T - dt):                 # the source at t + dt = T is the zero pad
                    pool[k, ((t + dt) * G + r + di) * G + c + dj] += w
        self.register_buffer("pool", torch.as_tensor(pool, dtype=torch.float32), persistent=False)
        self.scatter = nn.Parameter(torch.randn(H, len(TAPS) * H) / math.sqrt(len(TAPS) * H))   # (out, (tap, in))
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
        B, T, G = tok.shape[0], self.T, self.G
        h = self.embed(tok).view(B, T, G, G, self.H).permute(0, 4, 1, 2, 3)   # (B, H, T, G, G), channels last
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
        hn = h.permute(0, 2, 3, 4, 1).reshape(B, self.N, self.H)        # a view for channels_last_3d
        pooled = torch.matmul(self.pool.to(hn.dtype), hn)                # (B, 8, H): pooled sources of every tap
        z = pooled.reshape(B, len(TAPS) * self.H) @ self.scatter.t().to(pooled.dtype)
        return self.mlp2(F.silu(self.mlp1(z)))

    def readout_ref(self, h):
        """The read-out as written in the paper (scatter to data nodes, then the average over the support): tests."""
        B, H, T, G = h.shape[0], self.H, self.T, self.G
        W = self.scatter.view(H, 2, 2, 2, H).permute(0, 4, 1, 2, 3)     # Conv3d weight (out, in, dt, di, dj)
        data = F.conv3d(F.pad(h, (0, 0, 0, 0, 0, 1)), W.to(h.dtype))      # (B, H, T, d, d): sources t, t + 1
        r, c = (torch.as_tensor(x, device=h.device) for x in self.support.T)
        z = data[:, :, :, r, c].mean((2, 3))
        return self.mlp2(F.silu(self.mlp1(z)))

    def forward(self, det):
        return self.readout(self.backbone(self.tokens(det)))


def muon_params(model):
    """The matrices Muon trains (Methods: "matrix-valued parameters (convolution weights)"): the blocks' projections and
    3x3x3 convolutions. Everything else (embedding, BN, read-out scatter and MLP) goes to Lion."""
    return [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim >= 2]


def build(cfg, layout):
    """cfg: {d, R, H, L, res, act_ckpt, resid_bf16} -> CascadeSurface on the layout's detector places, channels last."""
    m = CascadeSurface(cfg["d"], cfg["R"], cfg["H"], cfg["L"], layout.t, layout.typ, layout.i, layout.j,
                       layout.support, res=cfg.get("res", "skip"), act_ckpt=cfg.get("act_ckpt", False),
                       resid_bf16=cfg.get("resid_bf16", False))
    m.cfg = dict(cfg)
    return m.to(memory_format=torch.channels_last_3d)


def macs_per_shot(H, L, d, R=None):
    """Multiply-accumulates per shot: per block and node two projections (H H/4 each) and the 3x3x3 convolution
    (27 (H/4)^2); the read-out's scatter (8 H H on the pooled taps) and MLP."""
    R = d if R is None else R
    N = (R + 1) * (d + 1) ** 2
    Hb = H // 4
    return L * N * (2 * H * Hb + 27 * Hb * Hb) + 8 * H * H + 2 * H * H + 2 * H
