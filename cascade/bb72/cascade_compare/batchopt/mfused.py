"""Milestone 2 (batchopt/M2.md): the batched fused engine of the ConvLines networks with several shots per program in every
kernel, each shot's arithmetic kept as pc_fused's (bb.fused's) kernels do it, so that every decision and margin stays
bitwise the registered batch-1 engine's at any batch size.

Engine(model, nmax, cfg=None) is pc_fused.Batched (bb.fused's tables, PDL, its buffers at nmax rounded up to whole
groups) with all seven launches replaced. Program (i, j) of a kernel runs tile i of the G shots j G .. j G + G - 1:

  stem     _k_stem_g: the bias and time-bias tile, the detector-index tile and the tap-weight tile are loaded once for
           the group; each shot gathers its own detector bits and gets its own (BM, KCH) x (KCH, C) product on the
           shared tile, then bb.fused's epilogue (LayerNorm, down-projection, SiLU) with its weights loaded once
  block    _k_block_g, every block but the last (milestone 1, the safe form): one (BM, TPI CB) x (TPI CB, CB)
           product per shot on each iteration's shared tap-weight tile; the gather's offsets carry their contiguity
           (runs of CB channels at multiples of CB), so that it loads 16 bytes per thread where today's loads 2; one
           pipeline stage; the epilogue (conv bias, SiLU, up-projection with the residual and its bias as the
           accumulator, the next LayerNorm, down-projection, SiLU) per shot with its weights loaded once
  last     _k_block_last_g, the last block: the same loop, then per shot the residual, the read-out's LayerNorm and its
           gated output in 64 columns at a time, each 64-column slice of the gate and value weights loaded once for the
           group and applied to each shot's LayerNorm output in turn
  pool     _k_pool_g: the position-gate tiles, the out-projection, the LayerNorm, the MLP's weights, the parity table
           and the query weights loaded once for the group, each shot's sums at today's shapes and in today's order; the
           grid ends at the last tile with a real query (27 of 32 for BB72; today the 5 tiles past it store nothing)
  vote     _k_vote_g (M1): each psi tile loaded once for the group, each shot's sums in today's layout; 8 warps; psi8 = 1
           stores the table (entries -1, 0, +1, checked exactly) as int8, loaded 8 entries per thread as the fp16 table
           is (the layout pin); psi8 = 2 drops the pin, a deliberate negative control that changes the margins' bits
  top2     pc_fused._k_top2, unchanged

Nothing moves the arithmetic of a shot: every product keeps today's shape (K per tl.dot, the accumulator carried into
the next product), every reduction today's tile and layout, every nonlinear function bb.fused's code. A shot of a
partial last group (n not a multiple of G) runs on padded buffer rows that no real shot reads and that are never
returned; it reads the detector bits of the last real shot (the input has n rows).

The launch settings (G, stages, the pool's query tiles per program) depend on the batch n (TIERS); cfg's
explicit values override them."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

import pc_fused
from bb.fused import BM, BQV, BS, BY, KCH, _p2, _silu

KERNELS = ("stem", "block", "last", "pool", "vote")
# The launch settings by batch size, set from the development sweep (M2.md section 1): (n_min, tier), the last tier
# with n_min <= n applies. Every setting is bitwise, so the table decides speed only. G: shots per program; stages:
# the software-pipeline stages (Triton's num_stages); qg: the pool's query tiles per program; seq: the last block's
# epilogue one shot after the other (else every shot's LayerNorm output live and each weight slice loaded once).
TIERS = ((1, {"G": {"stem": 1, "block": 1, "last": 1, "pool": 1, "vote": 1},
              "stages": {"stem": 2, "block": 2, "last": 2, "pool": 2, "vote": 2}, "qg": 1, "seq": True}),
         (16, {"G": {"stem": 1, "block": 1, "last": 1, "pool": 1, "vote": 2},
               "stages": {"stem": 1, "block": 1, "last": 1, "pool": 1, "vote": 2}, "qg": 1, "seq": True}),
         (64, {"G": {"stem": 4, "block": 2, "last": 2, "pool": 1, "vote": 2},
               "stages": {"stem": 1, "block": 1, "last": 1, "pool": 2, "vote": 2}, "qg": 2, "seq": False}),
         (256, {"G": {"stem": 4, "block": 8, "last": 4, "pool": 2, "vote": 2},
                "stages": {"stem": 1, "block": 1, "last": 1, "pool": 2, "vote": 2}, "qg": 2, "seq": False}),
         (1024, {"G": {"stem": 4, "block": 4, "last": 4, "pool": 2, "vote": 2},
                 "stages": {"stem": 1, "block": 1, "last": 1, "pool": 2, "vote": 2}, "qg": 2, "seq": False}),
         (4096, {"G": {"stem": 4, "block": 8, "last": 4, "pool": 2, "vote": 2},
                 "stages": {"stem": 1, "block": 1, "last": 1, "pool": 2, "vote": 2}, "qg": 2, "seq": False}))
DEFAULT = {"G": None, "stages": None, "qg": None, "seq": None, "vec": True, "warps": {"vote": 8}, "psi8": 1,
           "pskip": True}


@triton.jit
def _ln_t(x, w, b, C: tl.constexpr, eps):
    """bb.fused._ln with its weight and bias already loaded."""
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / C + eps)
    return xc * rstd[:, None] * w[None, :] + b[None, :]


@triton.jit
def _k_stem_g(det_ptr, nbs_ptr, wst_ptr, p_ptr, x_ptr, ln_w, ln_b, eps, wd_ptr, bd_ptr, u_ptr, D, S, KS, NB,
              KCH: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, PDL: tl.constexpr,
              SX: tl.constexpr, SU: tl.constexpr, G: tl.constexpr):
    b0 = tl.program_id(1).to(tl.int64) * G                                     # the group's first shot
    x_ptr += b0 * SX
    u_ptr += b0 * SU
    if PDL:
        gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    kk = tl.arange(0, KCH)
    p0 = tl.load(p_ptr + rows[:, None] * C + cols[None, :])                   # bias + time bias; 0 on padded rows
    lw = tl.load(ln_w + cols)
    lb = tl.load(ln_b + cols)
    wd = tl.load(wd_ptr + cols[:, None] * CB + cb[None, :])
    bd = tl.load(bd_ptr + cb)
    dps = ()
    accs = ()
    for g in tl.static_range(G):
        dps = dps + (det_ptr + tl.minimum(b0 + g, NB - 1) * D,)               # a padded shot reads the last real one
        accs = accs + (p0,)
    for k0 in tl.range(0, KS, KCH):
        idx = tl.load(nbs_ptr + rows[:, None] * KS + (k0 + kk)[None, :])      # detector index, D = none
        w = tl.load(wst_ptr + (k0 + kk)[:, None] * C + cols[None, :])         # once for the group
        nxt = ()
        for g in tl.static_range(G):
            bit = tl.load(dps[g] + idx, mask=idx < D, other=0).to(tl.float16)
            nxt = nxt + (accs[g] + tl.dot(bit, w),)
        accs = nxt
    for g in tl.static_range(G):                                                # bb.fused's epilogue (_down), per shot
        tl.store(x_ptr + g * SX + rows[:, None] * C + cols[None, :], accs[g])
        y = _ln_t(accs[g], lw, lb, C, eps).to(tl.float16)
        u = _silu(tl.dot(y, wd) + bd[None, :])
        tl.store(u_ptr + g * SU + rows[:, None] * CB + cb[None, :], u.to(tl.float16), mask=(rows < S)[:, None])


@triton.jit
def _conv_g(u_ptr, nb_ptr, wc_ptr, rows, NIT, SU: tl.constexpr, TPI: tl.constexpr, CB: tl.constexpr,
            BM: tl.constexpr, G: tl.constexpr, VEC: tl.constexpr):
    """A block's gathered convolution for G shots: per iteration the tap-weight tile once, one product per shot."""
    cb = tl.arange(0, CB)
    qq = tl.arange(0, TPI * CB)
    tt = tl.arange(0, TPI)
    accs = ()
    for g in tl.static_range(G):
        accs = accs + (tl.zeros((BM, CB), dtype=tl.float32),)
    for it in tl.range(0, NIT):
        w = tl.load(wc_ptr + (it * (TPI * CB) + qq)[:, None] * CB + cb[None, :])    # once for the group
        nbv = tl.load(nb_ptr + rows[:, None] * (NIT * TPI) + (it * TPI + tt)[None, :])   # neighbour site, S = zero row
        off = tl.reshape(nbv[:, :, None] * CB + cb[None, None, :], (BM, TPI * CB))
        if VEC:
            off = tl.max_contiguous(tl.multiple_of(off, [1, CB]), [1, CB])
        nxt = ()
        for g in tl.static_range(G):
            nxt = nxt + (accs[g] + tl.dot(tl.load(u_ptr + g * SU + off), w),)
        accs = nxt
    return accs


@triton.jit
def _k_block_g(x_ptr, u_ptr, nb_ptr, wc_ptr, bc_ptr, wu_ptr, bu_ptr, ln_w, ln_b, eps, wd_ptr, bd_ptr, uo_ptr, S, NIT,
               TPI: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, PDL: tl.constexpr,
               SX: tl.constexpr, SU: tl.constexpr, G: tl.constexpr, VEC: tl.constexpr):
    b0 = tl.program_id(1).to(tl.int64) * G                                     # the group's first shot
    x_ptr += b0 * SX
    u_ptr += b0 * SU
    uo_ptr += b0 * SU
    if PDL:
        gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    if PDL:
        gdc_wait()                                                              # u and x of the kernel before
    accs = _conv_g(u_ptr, nb_ptr, wc_ptr, rows, NIT, SU, TPI, CB, BM, G, VEC)
    bc = tl.load(bc_ptr + cb)
    vs = ()
    for g in tl.static_range(G):
        vs = vs + (_silu(accs[g] + bc[None, :]).to(tl.float16),)
    wu = tl.load(wu_ptr + cb[:, None] * C + cols[None, :])
    bu = tl.load(bu_ptr + cols)
    lw = tl.load(ln_w + cols)
    lb = tl.load(ln_b + cols)
    wd = tl.load(wd_ptr + cols[:, None] * CB + cb[None, :])
    bd = tl.load(bd_ptr + cb)
    for g in tl.static_range(G):                                                # bb.fused's epilogue, per shot
        x = tl.load(x_ptr + g * SX + rows[:, None] * C + cols[None, :]) + bu[None, :] + tl.dot(vs[g], wu)
        tl.store(x_ptr + g * SX + rows[:, None] * C + cols[None, :], x)
        y = _ln_t(x, lw, lb, C, eps).to(tl.float16)
        u = _silu(tl.dot(y, wd) + bd[None, :])
        tl.store(uo_ptr + g * SU + rows[:, None] * CB + cb[None, :], u.to(tl.float16), mask=(rows < S)[:, None])


@triton.jit
def _k_block_last_g(x_ptr, u_ptr, nb_ptr, wc_ptr, bc_ptr, wu_ptr, bu_ptr, ln_w, ln_b, eps, wg_ptr, bg_ptr, wv_ptr,
                    bv_ptr, uo_ptr, S, NIT, TPI: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr,
                    PDL: tl.constexpr, SX: tl.constexpr, SU: tl.constexpr, SUO: tl.constexpr, G: tl.constexpr,
                    VEC: tl.constexpr, SEQ: tl.constexpr):
    b0 = tl.program_id(1).to(tl.int64) * G                                     # the group's first shot
    x_ptr += b0 * SX
    u_ptr += b0 * SU
    uo_ptr += b0 * SUO
    if PDL:
        gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    if PDL:
        gdc_wait()                                                              # u and x of the kernel before
    accs = _conv_g(u_ptr, nb_ptr, wc_ptr, rows, NIT, SU, TPI, CB, BM, G, VEC)
    bc = tl.load(bc_ptr + cb)
    wu = tl.load(wu_ptr + cb[:, None] * C + cols[None, :])
    bu = tl.load(bu_ptr + cols)
    lw = tl.load(ln_w + cols)
    lb = tl.load(ln_b + cols)
    if SEQ:                                     # one shot after the other, the gate and value slices reread (L1)
        for g in tl.static_range(G):
            v = _silu(accs[g] + bc[None, :]).to(tl.float16)
            x = tl.load(x_ptr + g * SX + rows[:, None] * C + cols[None, :]) + bu[None, :] + tl.dot(v, wu)
            y = _ln_t(x, lw, lb, C, eps).to(tl.float16)
            for c0 in tl.static_range(0, C, 64):                               # 64 columns at a time (shared memory)
                cc = c0 + tl.arange(0, 64)
                gt = tl.sigmoid(tl.dot(y, tl.load(wg_ptr + cols[:, None] * C + cc[None, :])) + tl.load(bg_ptr + cc)[None, :])
                vv = tl.dot(y, tl.load(wv_ptr + cols[:, None] * C + cc[None, :])) + tl.load(bv_ptr + cc)[None, :]
                tl.store(uo_ptr + g * SUO + rows[:, None] * C + cc[None, :], (gt * vv).to(tl.float16),
                         mask=(rows < S)[:, None])
    else:                                       # every shot's LayerNorm output live, each slice loaded once
        ys = ()
        for g in tl.static_range(G):                                            # the residual and the read-out's LN
            v = _silu(accs[g] + bc[None, :]).to(tl.float16)
            x = tl.load(x_ptr + g * SX + rows[:, None] * C + cols[None, :]) + bu[None, :] + tl.dot(v, wu)
            ys = ys + (_ln_t(x, lw, lb, C, eps).to(tl.float16),)
        for c0 in tl.static_range(0, C, 64):                                   # 64 columns at a time (shared memory)
            cc = c0 + tl.arange(0, 64)
            wg = tl.load(wg_ptr + cols[:, None] * C + cc[None, :])            # once for the group
            bg = tl.load(bg_ptr + cc)
            wv = tl.load(wv_ptr + cols[:, None] * C + cc[None, :])
            bv = tl.load(bv_ptr + cc)
            for g in tl.static_range(G):
                gt = tl.sigmoid(tl.dot(ys[g], wg) + bg[None, :])
                vv = tl.dot(ys[g], wv) + bv[None, :]
                tl.store(uo_ptr + g * SUO + rows[:, None] * C + cc[None, :], (gt * vv).to(tl.float16),
                         mask=(rows < S)[:, None])


@triton.jit
def _k_pool_g(U_ptr, p_ptr, wo_ptr, u0_ptr, mw_ptr, mb_ptr, meps, w1_ptr, b1_ptr, w2_ptr, b2, gt_ptr, det_ptr, D, wq_ptr,
              s_ptr, z_ptr, Q, SP, NB, QP: tl.constexpr, BS: tl.constexpr, BQ: tl.constexpr, C: tl.constexpr,
              H: tl.constexpr, DH: tl.constexpr, FF: tl.constexpr, FC: tl.constexpr, NNZ: tl.constexpr,
              PDL: tl.constexpr, SUU: tl.constexpr, G: tl.constexpr, QG: tl.constexpr):
    """Program (i, j): query tiles i QG .. i QG + QG - 1 of shots j G .. j G + G - 1. Each position-gate tile is loaded
    once for the G shots, each tile of U once for the QG query tiles, the out-projection and the MLP's weights once
    for all; every (shot, query tile) keeps bb.fused's sums at today's shapes and in today's order."""
    b0 = tl.program_id(1).to(tl.int64) * G                                     # the group's first shot
    U_ptr += b0 * SUU
    s_ptr += b0 * QP
    z_ptr += b0 * QP
    if PDL:
        gdc_launch_dependents()
    cols = tl.arange(0, C)
    dc = tl.arange(0, DH)
    sb = tl.arange(0, BS)
    qrs = ()
    us = ()
    for q in tl.static_range(QG):
        qrs = qrs + ((tl.program_id(0) * QG + q) * BQ + tl.arange(0, BQ),)
    for g in tl.static_range(G):
        for q in tl.static_range(QG):
            us = us + (tl.load(u0_ptr + qrs[q][:, None] * C + cols[None, :]),)      # q0 + e_k + bo
    if PDL:
        gdc_wait()                                                              # U of the last block
    for h in tl.static_range(H):
        accs = ()
        for i in tl.static_range(G * QG):
            accs = accs + (tl.zeros((BQ, DH), dtype=tl.float32),)
        for s0 in tl.range(0, SP, BS):
            pts = ()
            for q in tl.static_range(QG):                                       # once for the G shots
                pts = pts + (tl.load(p_ptr + (h * QP + qrs[q])[:, None] * SP + (s0 + sb)[None, :]),)
            na = ()                                     # (names unique per loop: Triton carries reused ones)
            for g in tl.static_range(G):
                ut = tl.load(U_ptr + g * SUU + (s0 + sb)[:, None] * C + (h * DH + dc)[None, :])   # once for QG tiles
                for q in tl.static_range(QG):
                    na = na + (accs[g * QG + q] + tl.dot(pts[q], ut),)
            accs = na
        wo = tl.load(wo_ptr + (h * DH + dc)[:, None] * C + cols[None, :])
        nu = ()
        for i in tl.static_range(G * QG):
            nu = nu + (us[i] + tl.dot(accs[i].to(tl.float16), wo),)
        us = nu
    mw = tl.load(mw_ptr + cols)
    mb = tl.load(mb_ptr + cols)
    ys = ()
    zs = ()
    for i in tl.static_range(G * QG):
        ys = ys + (_ln_t(us[i], mw, mb, C, meps).to(tl.float16),)
        zs = zs + (tl.zeros((BQ,), dtype=tl.float32) + b2,)
    for f0 in tl.static_range(0, FF, FC):
        ff = f0 + tl.arange(0, FC)
        w1 = tl.load(w1_ptr + cols[:, None] * FF + ff[None, :])               # once for all
        b1 = tl.load(b1_ptr + ff)
        w2 = tl.load(w2_ptr + ff)
        nz_ = ()
        for i in tl.static_range(G * QG):
            hd = tl.dot(ys[i], w1) + b1[None, :]
            nz_ = nz_ + (zs[i] + tl.sum(_silu(hd) * w2[None, :], 1),)
        zs = nz_
    nz = tl.arange(0, NNZ)
    for q in tl.static_range(QG):
        di = tl.load(gt_ptr + qrs[q][:, None] * NNZ + nz[None, :])            # the column's detectors, D = none
        wq = tl.load(wq_ptr + qrs[q])
        for g in tl.static_range(G):
            dp = det_ptr + tl.minimum(b0 + g, NB - 1) * D                      # a padded shot reads the last real one
            par = tl.sum(tl.load(dp + di, mask=di < D, other=0).to(tl.int32), 1) % 2
            s = (2.0 * par.to(tl.float32) - 1.0) * zs[g * QG + q] * wq * 0.5
            tl.store(s_ptr + g * QP + qrs[q], s, mask=qrs[q] < Q)
            tl.store(z_ptr + g * QP + qrs[q], zs[g * QG + q], mask=qrs[q] < Q)


@triton.jit
def _k_vote_g(s_ptr, psi_ptr, beta_ptr, m1_ptr, i1_ptr, m2_ptr, QP, K: tl.constexpr, BY: tl.constexpr,
              BQV: tl.constexpr, PDL: tl.constexpr, NP: tl.constexpr, G: tl.constexpr, PIN: tl.constexpr):
    b0 = tl.program_id(1).to(tl.int64) * G                                     # the group's first shot
    s_ptr += b0 * QP
    m1_ptr += b0 * NP
    i1_ptr += b0 * NP
    m2_ptr += b0 * NP
    if PDL:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    yl = tl.arange(0, BY)
    ys = pid * BY + yl
    qv = tl.arange(0, BQV)
    accs = ()
    for g in tl.static_range(G):
        accs = accs + (tl.zeros((BY,), dtype=tl.float32),)
    if PDL:
        gdc_wait()                                                              # s of the pool
    for q0 in tl.range(0, QP, BQV):
        off = ys[:, None] * QP + (q0 + qv)[None, :]
        if PIN:
            off = tl.max_contiguous(off, [1, 8])                                # 8 entries per thread, as in fp16
        psi = tl.load(psi_ptr + off).to(tl.float32)                             # once for the group
        nxt = ()
        for g in tl.static_range(G):
            nxt = nxt + (accs[g] + tl.sum(psi * tl.load(s_ptr + g * QP + q0 + qv)[None, :], 1),)
        accs = nxt
    for g in tl.static_range(G):
        acc = accs[g]
        for j in tl.static_range(K):
            acc += (((ys >> j) & 1).to(tl.float32) - 0.5) * tl.load(beta_ptr + j)
        m1 = tl.max(acc, 0)
        i1 = tl.argmax(acc, 0)
        m2 = tl.max(tl.where(yl == i1, float("-inf"), acc), 0)
        tl.store(m1_ptr + g * NP + pid, m1)
        tl.store(i1_ptr + g * NP + pid, pid * BY + i1)
        tl.store(m2_ptr + g * NP + pid, m2)


def tier_for(n, cfg=None):
    """The launch settings at batch n: the TIERS entry, with cfg's explicit G, stages, qg and seq over it."""
    t = TIERS[0][1]
    for n_min, x in TIERS:
        if n >= n_min:
            t = x
    t = {"G": dict(t["G"]), "stages": dict(t["stages"]), "qg": t["qg"], "seq": t["seq"]}
    for k in ("G", "stages"):
        t[k].update((cfg or {}).get(k) or {})
    for k in ("qg", "seq"):
        if (cfg or {}).get(k) is not None:
            t[k] = cfg[k]
    return t


class Engine(pc_fused.Batched):
    """pc_fused.Batched with every launch replaced by its several-shot kernel (module docstring); called and run as
    Batched: Engine(model, nmax)(det (n, D) uint8, n <= nmax) -> (bits (n, K) uint8, margin (n,) fp32)."""

    def __init__(self, model, nmax, cfg=None):
        c = {**DEFAULT, **(cfg or {})}
        c["warps"] = {**DEFAULT["warps"], **c["warps"]}
        if set(c) != set(DEFAULT) or c["psi8"] not in (0, 1, 2) or not set(c["G"] or {}) <= set(KERNELS) \
                or not set(c["stages"] or {}) <= set(KERNELS) or c["qg"] not in (None, 1, 2, 4):
            raise ValueError(f"cfg {cfg}")
        self.cfg = c
        gmax = max(max(tier_for(n_min, c)["G"].values()) for n_min, _ in TIERS)
        super().__init__(model, -(-nmax // gmax) * gmax)                        # whole groups
        self.nmax, self.gmax = nmax, gmax
        self.NQR = -(-self.Q // self.bq) if c["pskip"] else self.NQT            # query tiles with a real query
        if c["psi8"]:
            p8 = self.psi.to(torch.int8)
            if not torch.equal(p8.to(self.psi.dtype), self.psi):
                raise ValueError("psi has entries other than -1, 0 and +1")
            self.psi8 = p8.contiguous()

    def tier(self, n):
        t = tier_for(n, self.cfg)
        if max(t["G"].values()) > self.gmax:
            raise ValueError(f"G {t['G']} at n = {n}: buffers hold groups of {self.gmax}")
        return t

    def groups(self, n):
        return self.tier(n)["G"]

    def launches(self, x, n):
        """[(name, launch)] of one call on n shots, in pc_fused.Batched's order and with its names."""
        c, C, CB, S, D, ns, pdl = self.cfg, self.C, self.CB, self.S, self.D, self.stages, self.pdl
        lp = {"launch_pdl": True} if pdl else {}           # every kernel after the first
        SX, SU, SUU = self.x[0].numel(), self.u[0][0].numel(), self.U[0].numel()
        t = self.tier(n)
        G, st, qg, vec = t["G"], t["stages"], t["qg"], bool(c["vec"])

        def ng(k):
            return -(-n // G[k])
        b0 = self.blk[0]
        out = [("stem", lambda: _k_stem_g[(self.NT, ng("stem"))](
            x, self.nbs, self.wst, self.P0, self.x, b0["ln_w"], b0["ln_b"], b0["eps"], b0["wd"], b0["bd"], self.u[0],
            D, S, self.KS, n, KCH=KCH, C=C, CB=CB, BM=BM, PDL=pdl, SX=SX, SU=SU, G=G["stem"], num_warps=4,
            num_stages=st["stem"]))]
        L = len(self.blk)
        for i, p in enumerate(self.blk):
            if i < L - 1:
                q = self.blk[i + 1]
                fn = (lambda p=p, q=q, i=i: _k_block_g[(self.NT, ng("block"))](
                    self.x, self.u[i % 2], self.nb, p["wc"], p["bc"], p["wu"], p["bu"], q["ln_w"], q["ln_b"], q["eps"],
                    q["wd"], q["bd"], self.u[(i + 1) % 2], S, self.NIT, TPI=self.tpi, C=C, CB=CB, BM=BM, PDL=pdl,
                    SX=SX, SU=SU, G=G["block"], VEC=vec, num_warps=4, num_stages=st["block"], **lp))
            else:
                q = self.ro
                fn = (lambda p=p, q=q, i=i: _k_block_last_g[(self.NT, ng("last"))](
                    self.x, self.u[i % 2], self.nb, p["wc"], p["bc"], p["wu"], p["bu"], q["ln_w"], q["ln_b"], q["eps"],
                    q["wg"], q["bg"], q["wv"], q["bv"], self.U, S, self.NIT, TPI=self.tpi, C=C, CB=CB, BM=BM, PDL=pdl,
                    SX=SX, SU=SU, SUO=SUU, G=G["last"], VEC=vec, SEQ=bool(t["seq"]), num_warps=4,
                    num_stages=st["last"], **lp))
            out.append((f"block{i}", fn))
        o = self.pool
        out.append(("pool", lambda: _k_pool_g[(-(-self.NQR // qg), ng("pool"))](
            self.U, self.P, o["wo"], o["u0"], o["mw"], o["mb"], o["meps"], o["w1"], o["b1"], o["w2"], o["b2"], self.gt, x,
            D, self.wq, self.s, self.z, self.Q, self.SP, n, QP=self.QP, BS=BS, BQ=self.bq, C=C, H=self.H, DH=self.DH,
            FF=o["FF"], FC=min(o["FF"], 64), NNZ=self.NNZ, PDL=pdl, SUU=SUU, G=G["pool"], QG=qg, num_warps=4,
            num_stages=st["pool"], **lp)))
        out.append(("vote", lambda: _k_vote_g[(self.NP, ng("vote"))](
            self.s, self.psi8 if c["psi8"] else self.psi, self.beta, self.m1, self.i1, self.m2, self.QP, K=self.K,
            BY=BY, BQV=BQV, PDL=pdl, NP=self.NP, G=G["vote"], PIN=c["psi8"] == 1, num_warps=c["warps"]["vote"],
            num_stages=st["vote"], **lp)))
        out.append(("top2", lambda: pc_fused._k_top2[(1, n)](
            self.m1, self.i1, self.m2, self.bits, self.margin, NP=self.NP, K=self.K, KP=_p2(self.K), PDL=pdl,
            num_warps=1, **lp)))
        return out


def engine(model, nmax, cfg=None, base=False):
    """Today's engine (pc_fused.Batched) if base, else Engine."""
    return pc_fused.Batched(model, nmax) if base else Engine(model, nmax, cfg)
