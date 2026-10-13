"""The deployed Triton engine of a ConvLines network (convlines.net), detector bits (B, m) uint8 on the GPU
-> standard-observable logit c (B,), line logits z (B, d) and decision (B,) uint8 on the GPU, in L + 2 kernels:

  stem          per site tile: the detector bits at the stem's taps (a (BM, taps) 0/1 tile gathered in the program from
                the shot's bits, read once) times the tile type's tap weights + the folded per-site table P -> residual
                x; epilogue: block 1's
                LayerNorm, down-projection and SiLU -> u
  block l < L   per site tile: gather u at the 27 taps of each site, one (BM, Cb) x (Cb, Cb) product per tap with the
                tile type's weights, SiLU, up-projection, residual add; epilogue: block l + 1's LayerNorm,
                down-projection, SiLU -> u'. u lives on a zero-padded dense grid, so every tap is a constant offset
                from the site's grid position and the 27 gathers are independent loads
  block L       as above, epilogue: the read-out's LayerNorm, its d x H attention scores (query-key product folded into
                one matrix, plus the constant bias table) and values, and the tile's partial pooled sums
                sum_s sigmoid(score) v_s per (line, head)
  read-out      one program per (shot, line): line j's partial sums added over the tiles (/ N0), out-projection,
                residual with the query, LayerNorm, MLP -> z_j; the shot's last program (atomic counter) forms
                c = sum_j w_j (-1)^g_j z_j + b0 in line order, g_j from the stem's per-tile partial sums

Tiles hold sites of one type (engine order: X sites, then Z sites; convlines.net.Layout.site_order), so a tile uses one
weight set. Arithmetic: fp16 matmul inputs, fp32 accumulation, fp32 residual stream, LayerNorm, gates and read-out
state. With pdl=True every kernel after the first is launched with programmatic dependent launch (Triton launch_pdl;
it loads constants, then waits (griddepcontrol.wait) for its predecessor's outputs).

This is the production file, unchanged apart from comments and docstrings. The studies ran it with FROZEN below at
d = 9-15 (three pipeline stages at every d) as frozen(deploy_tables(model)), the tables built on the CPU in fp32."""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


def _p2(n):
    return 1 << (int(n) - 1).bit_length()


@triton.jit
def _ln(x, w_ptr, b_ptr, cols, C: tl.constexpr, eps):
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / C + eps)
    return xc * rstd[:, None] * tl.load(w_ptr + cols)[None, :] + tl.load(b_ptr + cols)[None, :]


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _tile(tile, NTX, nx, m, BM: tl.constexpr):
    is_z = tile >= NTX
    r0 = tl.where(is_z, nx + (tile - NTX) * BM, tile * BM)
    rend = tl.where(is_z, m, nx)
    rows = r0 + tl.arange(0, BM)
    return rows, rows < rend, is_z.to(tl.int32)


@triton.jit
def _down(x, gp, rm, b, ln_w, ln_b, eps, wd_ptr, bd_ptr, u_ptr, NG: tl.constexpr, C: tl.constexpr, CB: tl.constexpr):
    """u = SiLU(LayerNorm(x) Wd + bd) for the tile rows -> u_ptr (B, NG, CB) fp16 at their padded-grid positions gp."""
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    y = _ln(x, ln_w, ln_b, cols, C, eps)
    wd = tl.load(wd_ptr + cols[:, None] * CB + cb[None, :])
    u = _silu(tl.dot(y.to(tl.float16), wd) + tl.load(bd_ptr + cb)[None, :])
    tl.store(u_ptr + (b * NG + gp)[:, None] * CB + cb[None, :], u.to(tl.float16), mask=rm[:, None])


@triton.jit
def _k_stem(bits_ptr, nbd_ptr, gpos_ptr, det_ptr, A_ptr, gp_ptr, P_ptr, Wb_ptr, x_ptr, u_ptr, ln_w, ln_b, eps, wd_ptr,
            bd_ptr, NT, NTX, nx, m, NG: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, KSP: tl.constexpr,
            DP: tl.constexpr, MP: tl.constexpr, BM: tl.constexpr, PDL: tl.constexpr):
    """The shot's bits are read once, contiguously (MP >= m bytes, zero beyond m); the (BM, taps) bit tile and the
    sites' own bits are gathered from them in the program (tl.gather), so the stem waits on one level of loads."""
    if PDL:
        gdc_launch_dependents()
    b = tl.program_id(0)
    tile = tl.program_id(1)
    rows, rm, typ = _tile(tile, NTX, nx, m, BM)
    cols = tl.arange(0, C)
    taps = tl.arange(0, KSP)
    mp = tl.arange(0, MP)
    allb = tl.load(bits_ptr + b * m + mp, mask=mp < m, other=0).to(tl.float16)                     # (MP,)
    nd = tl.load(nbd_ptr + rows[:, None] * KSP + taps[None, :], mask=rm[:, None], other=MP - 1)    # MP - 1: a zero bit
    bit = tl.reshape(tl.gather(allb, tl.reshape(nd, (BM * KSP,)), 0), (BM, KSP))                   # (BM, KSP)
    jj = tl.arange(0, DP)                                                    # this tile's share of g_j = bits . A_j
    own = tl.gather(allb, tl.load(det_ptr + rows, mask=rm, other=MP - 1), 0).to(tl.float32)
    At = tl.load(A_ptr + rows[:, None] * DP + jj[None, :], mask=rm[:, None], other=0.0)
    tl.store(gp_ptr + (b * NT + tile) * DP + jj, tl.sum(own[:, None] * At, 0))
    wb = tl.load(Wb_ptr + typ * KSP * C + taps[:, None] * C + cols[None, :])                    # (KSP, C)
    x = tl.load(P_ptr + rows[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0) + tl.dot(bit, wb)
    tl.store(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
    gp = tl.load(gpos_ptr + rows, mask=rm, other=0)
    _down(x, gp, rm, b, ln_w, ln_b, eps, wd_ptr, bd_ptr, u_ptr, NG, C, CB)


@triton.jit
def _conv_up(x_ptr, u_ptr, gp, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, m, NG: tl.constexpr,
             GH: tl.constexpr, GW: tl.constexpr, KT: tl.constexpr, KI: tl.constexpr, KJ: tl.constexpr,
             C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, STAGES: tl.constexpr, TPI: tl.constexpr):
    """x[rows] + Up(SiLU(sum over taps of u[neighbour] Wc[typ, tap] + bc[typ])) (BM, C) fp32. u lives on the padded
    dense grid (zero off the sites), so a tap's neighbour is at the constant offset (dt GH + di) GW + dj from gp: the
    taps' gathers do not depend on each other. TPI taps per iteration: one (BM, TPI CB) x (TPI CB, CB) product (the
    weights are padded to a whole number of iterations with zero taps)."""
    NTAP: tl.constexpr = KT * KI * KJ
    NIT: tl.constexpr = (NTAP + TPI - 1) // TPI
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    qq = tl.arange(0, TPI * CB)
    x = tl.load(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0)
    acc = tl.zeros((BM, CB), dtype=tl.float32)
    base = (b * NG + gp)[:, None] * CB
    for it in tl.range(0, NIT, num_stages=STAGES):
        o = it * TPI + qq // CB
        delta = ((o // (KI * KJ) - KT // 2) * GH + (o // KJ % KI - KI // 2)) * GW + (o % KJ - KJ // 2)
        a = tl.load(u_ptr + base + (delta * CB + qq % CB)[None, :], mask=rm[:, None] & (o < NTAP)[None, :], other=0.0)
        w = tl.load(Wc_ptr + (typ * NIT * TPI * CB + it * TPI * CB + qq[:, None]) * CB + cb[None, :])
        acc += tl.dot(a, w)
    v = _silu(acc + tl.load(bc_ptr + typ * CB + cb)[None, :])
    wu = tl.load(Wu_ptr + cb[:, None] * C + cols[None, :])
    up = tl.dot(v.to(tl.float16), wu) + tl.load(bu_ptr + cols)[None, :]
    return x + up


@triton.jit
def _k_block(x_ptr, u_ptr, u2_ptr, gpos_ptr, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, ln_w, ln_b, eps, wd_ptr, bd_ptr, NTX, nx,
             m, NG: tl.constexpr, GH: tl.constexpr, GW: tl.constexpr, KT: tl.constexpr, KI: tl.constexpr,
             KJ: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, STAGES: tl.constexpr,
             TPI: tl.constexpr, PDL: tl.constexpr):
    b = tl.program_id(0)
    rows, rm, typ = _tile(tl.program_id(1), NTX, nx, m, BM)
    gp = tl.load(gpos_ptr + rows, mask=rm, other=0)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    x = _conv_up(x_ptr, u_ptr, gp, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, m, NG, GH, GW, KT, KI, KJ, C, CB,
                 BM, STAGES, TPI)
    cols = tl.arange(0, C)
    tl.store(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
    _down(x, gp, rm, b, ln_w, ln_b, eps, wd_ptr, bd_ptr, u2_ptr, NG, C, CB)


@triton.jit
def _k_block_last(x_ptr, u_ptr, gpos_ptr, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, ln_w, ln_b, eps, wsc_ptr, bias_ptr, wv_ptr,
                  bv_ptr, pa_ptr, NT, NTX, nx, m, NG: tl.constexpr, GH: tl.constexpr, GW: tl.constexpr,
                  KT: tl.constexpr, KI: tl.constexpr, KJ: tl.constexpr, C: tl.constexpr, CB: tl.constexpr,
                  BM: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, DP: tl.constexpr, STAGES: tl.constexpr,
                  TPI: tl.constexpr, PDL: tl.constexpr):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    rows, rm, typ = _tile(tile, NTX, nx, m, BM)
    gp = tl.load(gpos_ptr + rows, mask=rm, other=0)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    x = _conv_up(x_ptr, u_ptr, gp, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, m, NG, GH, GW, KT, KI, KJ, C, CB,
                 BM, STAGES, TPI)
    cols = tl.arange(0, C)
    y = _ln(x, ln_w, ln_b, cols, C, eps).to(tl.float16)
    hj = tl.arange(0, H * DP)                                                # score column = h DP + j
    s = tl.dot(y, tl.load(wsc_ptr + cols[:, None] * (H * DP) + hj[None, :])) + \
        tl.load(bias_ptr + rows[:, None] * (H * DP) + hj[None, :], mask=rm[:, None], other=0.0)
    g = tl.where(rm[:, None], 1.0 / (1.0 + tl.exp(-s)), 0.0)                                      # sigmoid gates
    v = tl.dot(y, tl.load(wv_ptr + cols[:, None] * C + cols[None, :])) + tl.load(bv_ptr + cols)[None, :]
    pr = tl.dot(tl.trans(g.to(tl.float16)), v.to(tl.float16))               # (H DP, C); keep head-matched blocks
    keep = (hj // DP)[:, None] == (cols // DH)[None, :]
    tl.store(pa_ptr + ((b * NT + tile) * DP + (hj % DP)[:, None]) * C + cols[None, :], pr, mask=keep)


@triton.jit
def _k_readout(gp_ptr, pa_ptr, U0_ptr, Wo_ptr, bo_ptr, mw_ptr, mb_ptr, meps, W1_ptr, b1_ptr, W2_ptr, b2, w_ptr, b0, inv_n0,
               z_ptr, ctr_ptr, c_ptr, dec_ptr, NT, D, NTP: tl.constexpr, C: tl.constexpr, DP: tl.constexpr,
               FF: tl.constexpr, FC: tl.constexpr, PDL: tl.constexpr):
    """Program (b, j): line j's pooled value = the tiles' partial sums added / N0 (one load), out-projection, residual,
    LayerNorm, MLP -> z_j; the program that finishes last for shot b (atomic counter) forms
    c = sum_j w_j (-1)^g_j z_j + b0 in line order (deterministic) and resets the counter."""
    b = tl.program_id(0)
    j = tl.program_id(1)
    cols = tl.arange(0, C)
    tt = tl.arange(0, NTP)
    wo = tl.load(Wo_ptr + cols[:, None] * C + cols[None, :])                                         # (C, C), constant
    if PDL:
        gdc_wait()
    tm = (tt < NT)[:, None]
    a = tl.sum(tl.load(pa_ptr + ((b * NT + tt[:, None]) * DP + j) * C + cols[None, :], mask=tm, other=0.0), 0) * inv_n0
    u = tl.load(U0_ptr + j * C + cols) + tl.sum(a[:, None] * wo.to(tl.float32), 0) + tl.load(bo_ptr + cols)
    mean = tl.sum(u, 0) / C
    uc = u - mean
    y = uc * tl.rsqrt(tl.sum(uc * uc, 0) / C + meps) * tl.load(mw_ptr + cols) + tl.load(mb_ptr + cols)
    z = b2
    for f0 in tl.static_range(0, FF, FC):
        ff = f0 + tl.arange(0, FC)
        w1 = tl.load(W1_ptr + cols[:, None] * FF + ff[None, :]).to(tl.float32)
        hd = tl.sum(y[:, None] * w1, 0) + tl.load(b1_ptr + ff)
        z += tl.sum(hd / (1.0 + tl.exp(-hd)) * tl.load(W2_ptr + ff), 0)
    tl.store(z_ptr + b * D + j, z)
    tl.debug_barrier()                                         # the store precedes the release below in every thread
    done = tl.atomic_add(ctr_ptr + b, 1, sem="acq_rel", scope="gpu")
    if done == D - 1:
        tl.debug_barrier()
        jj = tl.arange(0, DP)
        gs = tl.sum(tl.load(gp_ptr + (b * NT + tt[:, None]) * DP + jj[None, :], mask=tm, other=0.0), 0)
        sgn = 1.0 - 2.0 * (gs - 2.0 * tl.floor(gs * 0.5))                                           # (-1)^g_j
        zz = tl.load(z_ptr + b * D + jj, mask=jj < D, other=0.0)
        c = tl.sum(tl.where(jj < D, tl.load(w_ptr + jj) * sgn * zz, 0.0), 0) + b0
        tl.store(c_ptr + b, c)
        tl.store(dec_ptr + b, (c > 0).to(tl.uint8))
        tl.atomic_xchg(ctr_ptr + b, 0)


@triton.jit
def _k_nop(ptr, PDL: tl.constexpr):
    """Calibration: one store, nothing else (the launch floor of a kernel boundary in a graph)."""
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    tl.store(ptr + tl.program_id(0), 1.0)


def nop_chain(n, grid=23, pdl=False):
    """A call of n dependent no-op kernels (the first launched normally), for the launch-floor calibration."""
    buf = torch.zeros(grid, device="cuda")

    def fn(x):
        for i in range(n):
            _k_nop[(grid,)](buf, PDL=pdl and i > 0, **({"launch_pdl": True} if pdl and i > 0 else {}))
        return buf
    return fn


FROZEN = dict(BM=16, warps=4, stages=3, tpi=4, warps_ro=4)
"""The one engine configuration (the fastest validated launch configuration of a development sweep), used for every
read-out, calibration and timing of the released networks; PDL is on where the GPU supports it (sm_90+), which changes no
arithmetic (the logits were checked bit for bit with PDL on and off)."""


def frozen(t, pdl=None):
    """The deployed engine of deploy tables t: FROZEN, PDL on sm_90+ unless pdl is given."""
    if pdl is None:
        pdl = torch.cuda.get_device_capability() >= (9, 0)
    return Engine(t, pdl=pdl, **FROZEN)


class Engine:
    """Engine of deploy tables t (convlines.net.deploy_tables, fp32). __call__(bits (B, m) uint8) -> (c, z, decision)."""

    def __init__(self, t, BM=32, warps=4, warps_ro=4, stages=3, tpi=1, pdl=False):
        dev = t["P"].device
        cfg = t["cfg"]
        self.cfg = dict(BM=BM, warps=warps, warps_ro=warps_ro, stages=stages, tpi=tpi, pdl=pdl)
        self.m, self.nx, self.d = cfg["m"], t["nx"], cfg["d"]
        self.C, self.CB, self.L = cfg["width"], cfg["bott"], cfg["blocks"]
        h = t["head"]
        self.H, self.DH = h["H"], h["dh"]
        self.DP = max(16, _p2(self.d))
        self.FF = h["W1"].shape[1]
        m, nx = self.m, self.nx
        self.NTX = math.ceil(nx / BM)
        self.NT = self.NTX + math.ceil((m - nx) / BM)
        f16, f32 = torch.float16, torch.float32
        ks = t["Wb"].shape[1]
        self.KSP = max(16, _p2(ks))
        self.KT, self.KI, self.KJ = cfg["block_k"]
        T, Hg, Wg = t["shape"]
        self.GH, self.GW = Hg + 2 * (self.KI // 2), Wg + 2 * (self.KJ // 2)            # padded dense grid of u
        self.NG = (T + 2 * (self.KT // 2)) * self.GH * self.GW
        tij = t["tij"]                                                                    # (m, 3) engine order
        self.gpos = (((tij[:, 0] + self.KT // 2) * self.GH + tij[:, 1] + self.KI // 2) * self.GW + tij[:, 2] +
                     self.KJ // 2).to(torch.int32).contiguous()
        self.MP = _p2(m)
        assert self.MP > m, "a zero bit at MP - 1 is needed"
        nbd = torch.full((m, self.KSP), self.MP - 1, dtype=torch.int32, device=dev)
        nbs = t["nb_stem"]
        nbd[:, :ks] = torch.where(nbs >= 0, t["det"][nbs.clamp(min=0)], self.MP - 1).to(torch.int32)   # tap detector
        self.nbd = nbd.contiguous()
        self.P = t["P"].to(f32).contiguous()
        Wb = torch.zeros(2, self.KSP, self.C, dtype=f16, device=dev)
        Wb[:, :ks] = t["Wb"].to(f16)
        self.Wb = Wb.contiguous()
        self.blk = [{k: (v.to(f16).contiguous() if k in ("Wd", "Wc", "Wu") else
                         v.to(f32).contiguous() if torch.is_tensor(v) else v) for k, v in p.items()} for p in t["blocks"]]
        ntap = self.KT * self.KI * self.KJ
        pad = (-ntap) % tpi                                   # zero taps up to a whole number of TPI-tap iterations
        if pad:
            for p in self.blk:
                Wc = p["Wc"].view(2, ntap, self.CB, self.CB)
                p["Wc"] = torch.cat([Wc, Wc.new_zeros(2, pad, self.CB, self.CB)], 1).reshape(2, -1, self.CB).contiguous()
        H, DP, C, d = self.H, self.DP, self.C, self.d
        Wsc = torch.zeros(C, H, DP, device=dev)
        Wsc[:, :, :d] = h["Wsc"].view(C, d, H).permute(0, 2, 1)                                   # column (h, j)
        bias = torch.zeros(m, H, DP, device=dev)
        bias[:, :, :d] = h["BIAS"].view(m, d, H).permute(0, 2, 1)
        U0 = torch.zeros(DP, C, device=dev)
        U0[:d] = h["U0"]
        A = torch.zeros(m, DP, device=dev)
        A[:, :d] = t["A"]                                                                           # engine order
        w = torch.zeros(DP, device=dev)
        w[:d] = t["w"]
        self.ro = {"ln_w": h["ln_w"].float().contiguous(), "ln_b": h["ln_b"].float().contiguous(), "eps": h["eps"],
                   "Wsc": Wsc.reshape(C, H * DP).to(f16).contiguous(), "bias": bias.reshape(m, H * DP).contiguous(),
                   "Wv": h["Wv"].to(f16).contiguous(), "bv": h["bv"].float().contiguous(), "U0": U0.contiguous(),
                   "Wo": h["Wo"].to(f16).contiguous(), "bo": h["bo"].float().contiguous(),
                   "mw": h["m_ln_w"].float().contiguous(), "mb": h["m_ln_b"].float().contiguous(), "meps": h["m_eps"],
                   "W1": h["W1"].to(f16).contiguous(), "b1": h["b1"].float().contiguous(),
                   "W2": h["W2"].reshape(-1).float().contiguous(), "b2": float(h["b2"]), "w": w.contiguous(),
                   "b0": float(t["b0"]), "A": A.contiguous(), "inv_n0": 1.0 / float(h["n0"])}
        self.det = t["det"].to(torch.int32).contiguous()
        self.NTP = _p2(self.NT)
        self.dev = dev
        self._bufs = {}

    def buffers(self, B):
        if B not in self._bufs:
            dev, m, f32 = self.dev, self.m, torch.float32
            self._bufs[B] = {"x": torch.empty(B, m, self.C, device=dev, dtype=f32),
                             "u": [torch.zeros(B, self.NG, self.CB, device=dev, dtype=torch.float16) for _ in range(2)],
                             "pa": torch.empty(B, self.NT, self.DP, self.C, device=dev, dtype=f32),
                             "gp": torch.empty(B, self.NT, self.DP, device=dev, dtype=f32),
                             "ctr": torch.zeros(B, device=dev, dtype=torch.int32),
                             "c": torch.empty(B, device=dev, dtype=f32), "z": torch.empty(B, self.d, device=dev, dtype=f32),
                             "dec": torch.empty(B, device=dev, dtype=torch.uint8)}
        return self._bufs[B]

    def launches(self, bits):
        """[(name, launch)] of the call, in order; __call__ runs them."""
        B = bits.shape[0]
        bf = self.buffers(B)
        c, ro = self.cfg, self.ro
        pdl, BM, nw = c["pdl"], c["BM"], c["warps"]
        geo = dict(NG=self.NG, GH=self.GH, GW=self.GW, KT=self.KT, KI=self.KI, KJ=self.KJ, C=self.C, CB=self.CB, BM=BM,
                   STAGES=c["stages"], TPI=c["tpi"], PDL=pdl, num_warps=nw, **({"launch_pdl": True} if pdl else {}))
        grid = (B, self.NT)
        p0 = self.blk[0]
        steps = [("stem", lambda: _k_stem[grid](
            bits, self.nbd, self.gpos, self.det, ro["A"], bf["gp"], self.P, self.Wb, bf["x"], bf["u"][0], p0["ln_w"],
            p0["ln_b"], p0["eps"], p0["Wd"], p0["bd"], self.NT, self.NTX, self.nx, self.m, NG=self.NG, C=self.C,
            CB=self.CB, KSP=self.KSP, DP=self.DP, MP=self.MP, BM=BM, PDL=pdl, num_warps=nw))]
        for li, p in enumerate(self.blk):
            ui, uo = bf["u"][li % 2], bf["u"][(li + 1) % 2]
            if li < self.L - 1:
                q = self.blk[li + 1]
                steps.append((f"block{li}", lambda p=p, q=q, ui=ui, uo=uo: _k_block[grid](
                    bf["x"], ui, uo, self.gpos, p["Wc"], p["bc"], p["Wu"], p["bu"], q["ln_w"], q["ln_b"], q["eps"],
                    q["Wd"], q["bd"], self.NTX, self.nx, self.m, **geo)))
            else:
                steps.append((f"block{li}_ro", lambda p=p, ui=ui: _k_block_last[grid](
                    bf["x"], ui, self.gpos, p["Wc"], p["bc"], p["Wu"], p["bu"], ro["ln_w"], ro["ln_b"], ro["eps"],
                    ro["Wsc"], ro["bias"], ro["Wv"], ro["bv"], bf["pa"], self.NT, self.NTX, self.nx,
                    self.m, H=self.H, DH=self.DH, DP=self.DP, **geo)))
        steps.append(("readout", lambda: _k_readout[(B, self.d)](
            bf["gp"], bf["pa"], ro["U0"], ro["Wo"], ro["bo"], ro["mw"], ro["mb"], ro["meps"], ro["W1"], ro["b1"],
            ro["W2"], ro["b2"], ro["w"], ro["b0"], ro["inv_n0"], bf["z"], bf["ctr"], bf["c"], bf["dec"], self.NT,
            self.d, NTP=self.NTP, C=self.C, DP=self.DP, FF=self.FF, FC=min(self.FF, 128), PDL=pdl,
            num_warps=c["warps_ro"], **({"launch_pdl": True} if pdl else {}))))
        return steps, (bf["c"], bf["z"], bf["dec"])

    def __call__(self, bits):
        steps, out = self.launches(bits)
        for _, fn in steps:
            fn()
        return out
