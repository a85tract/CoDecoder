"""The ConvLines network at distance d (the code that trained and deployed the five released networks; only the source
of the detector geometry differs: here it is read from weights/surface/grid_d{d}.npz instead of being rebuilt from NVIDIA's
MemoryCircuit, see scripts/make_circuits.py).

Input: the dense detector grid. Every detector sits at (t, i, j) on a T x H x W grid (d = 9: 10 x 10 x 10, 720
detectors; X-type sites at (i + j) even, Z-type at (i + j) odd; grid positions without a detector are zero padding).
Constant channels per site: X type, Z type, first layer, last layer (the data-measurement layer).

  stem      one type-specific 3D convolution (kernel 3 x 5 x 5 in (t, i, j): every single-mechanism relation of the DEM,
            matching edges and X-Z pairs, lies within (+-1, +-2, +-2)), detector bit + constant channels -> C; X- and
            Z-centred sites have their own kernels (one weight matrix per tap and type)
  blocks    L pre-norm bottleneck residual blocks: LayerNorm -> 1x1x1 C -> Cb -> SiLU -> type-specific 3 x 3 x 3
            convolution Cb -> Cb -> SiLU -> 1x1x1 Cb -> C (zero-initialised) added to the stream; off-site positions
            stay 0, so they act as zero padding
  read-out  d logical-line queries (line j = data-qubit row j; a shared query plus a per-line offset) cross-attend over
            the site features (H heads; score bias by the site's signed distance to the line in half-line units, its
            time layer and its type) with sigmoid gates and a sum, a = sum_s sigmoid(score) v_s / N0 (N0 = 64): a
            line's flip is the parity of the chains crossing it, so the pooled value keeps counts; then out-projection,
            residual, LayerNorm + MLP -> one logit z_j per line
  decision  the standard observable's logit c = sum_j w_j (-1)^g_j z_j + b0, g_j = y_j XOR y_0 an exact GF(2) function
            of the detectors (the buffer A, stored with the weights); decision = c > 0.

Training loss: mean_j BCE(z_j, y_j) + BCE(c, y_0), y_j the sampled line flips. `deploy_tables` folds a trained network
into the inference form that timing/engine.py (the deployed Triton engine) runs."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N0 = 64.0                                                # the read-out's constant sum normalisation


class Layout:
    """Geometry of distance d (numpy), from weights/surface/grid_d{d}.npz: idx (m, 3) = (t, i, j) grid position of every
    detector in the circuit's detector order, is_x (m,) = X-type check, line_off (m, d) = signed distance of the
    detector to logical line j in half-line units."""

    def __init__(self, d, geometry):
        g = np.load(geometry)
        idx = g["idx"].astype(np.int64)
        assert idx.shape[1] == 3 and g["line_off"].shape == (len(idx), d), f"{geometry}: not a d = {d} geometry"
        self.d, self.m = d, len(idx)
        self.T, self.H, self.W = (int(idx[:, k].max()) + 1 for k in range(3))
        self.n = self.T * self.H * self.W
        self.idx = idx
        self.is_x = g["is_x"].astype(bool)
        self.flat = (idx[:, 0] * self.H + idx[:, 1]) * self.W + idx[:, 2]
        self.first = idx[:, 0] == 0
        self.last = idx[:, 0] == self.T - 1
        self.off = g["line_off"].astype(np.int64)        # (m, d) signed distance to line j, half-line units
        self.A = np.zeros((self.m, d), np.uint8)         # g = bits A mod 2; the trained value comes with the weights
        pos = {tuple(int(v) for v in r): k for k, r in enumerate(idx)}
        assert len(pos) == self.m, "two detectors share a grid site"
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
    """d line queries over the site features (B, m, C) -> line logits (B, d)."""

    def __init__(self, C, heads, lay, mlp=2):
        super().__init__()
        d = lay.d
        self.d, self.h, self.dh = d, heads, C // heads
        self.ln = nn.LayerNorm(C)
        self.k = nn.Linear(C, C, bias=False)
        self.wq = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.e = nn.Parameter(torch.zeros(d, C))
        R = int(np.abs(lay.off).max())
        bins = torch.arange(-R, R + 1, dtype=torch.float32)
        self.b_line = nn.Parameter((-1.0 - 0.5 * bins.abs()).repeat(heads, 2, 1))  # (H, type, 2R + 1): near the line
        self.n0 = N0
        self.b_t = nn.Parameter(torch.zeros(heads, 2, lay.T))                     # (H, type, T)
        self.register_buffer("off", torch.as_tensor(lay.off + R, dtype=torch.long), persistent=False)       # (m, d)
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


class ConvLines(nn.Module):
    def __init__(self, d, geometry, width=128, bott=32, blocks=2, heads=4, stem_k=(3, 5, 5), block_k=(3, 3, 3), mlp=2):
        super().__init__()
        lay = Layout(d, geometry)
        self.lay = lay
        self.cfg = dict(arch="convlines", d=d, width=width, bott=bott, blocks=blocks, heads=heads, stem_k=list(stem_k),
                        block_k=list(block_k), mlp=mlp)
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
        self.head = LineQueries(width, heads, lay, mlp)
        self.w = nn.Parameter(torch.full((d,), 1.0 / d))
        self.b0 = nn.Parameter(torch.zeros(()))

    def gsign(self, bits):                                 # (B, m) -> (B, d) float, (-1)^g_j
        with torch.autocast("cuda", enabled=False):
            g = torch.remainder(bits.float() @ self.A, 2.0)
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

    def forward(self, bits):                               # (B, m) -> (c (B,), z (B, d))
        z = self.head(self.features(bits))
        c = (z.float() * self.gsign(bits) * self.w).sum(1) + self.b0
        return c, z


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def deploy_tables(model):
    """The inference form (fp32 tensors on the model's device) of a ConvLines model, engine site order."""
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
    """Every tensor of the tables detached from autograd and contiguous."""
    if torch.is_tensor(t):
        return t.detach().clone().contiguous()
    if isinstance(t, dict):
        return {k: _detached(v) for k, v in t.items()}
    if isinstance(t, list):
        return [_detached(v) for v in t]
    return t
