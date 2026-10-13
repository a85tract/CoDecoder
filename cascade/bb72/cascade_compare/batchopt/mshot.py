"""Milestone 1: several shots per program in S55's fused engine, on a residual block's kernel and on the
vote, each shot's arithmetic kept as pc_fused's (bb.fused's) kernels do it.

MultiShot(model, nmax, cfg) is pc_fused.Batched with some launches replaced (cfg keys, defaults in DEFAULT):

  blocks, bg, form, vec, bs   _k_block_g on the residual blocks listed in blocks (not the last one): program (i, j) runs
           tile i of the bg shots j * bg .. j * bg + bg - 1. Each iteration's tap-weight tile (TPI CB x CB) is loaded
           once for the group. form "safe": one (BM, TPI CB) x (TPI CB, CB) product per shot, today's shape; "stack":
           one (bg BM, TPI CB) product over the group's stacked rows. Either way an output element is the same
           mma.sync m16n8k16 chain over k in the same order (K = TPI CB per product, as today). The epilogue (conv bias,
           SiLU, up-projection with the residual and its bias as the accumulator, the next LayerNorm, down-projection,
           SiLU) runs per shot at today's shapes, its weights loaded once for the group. vec: the gather's offsets carry
           their contiguity (runs of CB channels at multiples of CB), so that it loads 16 bytes per thread where today's
           loads 2; the values reaching the MMA are the same. bs: the block's pipeline stages (0: bb.fused's 2); stages
           move data only (async copies into shared memory), never the arithmetic.
  vg, psi8, vw   _k_vote_g: program (i, j) scores classes i BY .. i BY + BY - 1 for the vg shots of group j; each psi
           tile is loaded once for the group and each shot's sums keep today's layout (8 columns per thread, 16 lanes
           across, so the same reduction tree). psi8 = 1: the psi table (entries -1, 0, +1) stored as int8 and loaded 8
           entries per thread, as the fp16 table is; psi8 = 2: int8 without that pin, a negative control (the load then
           takes 16 entries per thread and the reduction tree changes). vw: the vote's warps.
  pool     "skip": the pool's grid ends at the last tile with a real query (27 of 32 for BB72); today the 5 tiles past
           it store nothing.

Buffers hold whole groups (Batched's buffers at nmax rounded up to the largest G); the shots of a partial last group
past n run on those padded rows, which no real shot reads, and are never returned. Launch options (warps, stages, PDL)
are bb.fused's unless cfg says otherwise."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

import pc_fused
from bb.fused import BM, BQV, BS, BY, _silu

DEFAULT = {"blocks": (), "bg": 1, "form": "safe", "vec": False, "bs": 0, "vg": 0, "psi8": 0, "vw": 4, "pool": ""}


@triton.jit
def _halves(x, G: tl.constexpr, BM: tl.constexpr, N: tl.constexpr):
    """Rows (G BM, N) of G shots (shot g in rows g BM ..) -> the rows of the first and of the last G / 2 shots."""
    return tl.split(tl.permute(tl.reshape(x, (2, (G // 2) * BM, N)), (1, 2, 0)))


@triton.jit
def _unstack(x, G: tl.constexpr, BM: tl.constexpr, N: tl.constexpr):
    """Rows (G BM, N) of G shots -> a tuple of G tiles (BM, N), shot order kept; G is 1, 2, 4 or 8."""
    if G == 1:
        out = (x,)
    elif G == 2:
        out = _halves(x, 2, BM, N)
    elif G == 4:
        lo, hi = _halves(x, 4, BM, N)
        out = _halves(lo, 2, BM, N) + _halves(hi, 2, BM, N)
    else:
        lo, hi = _halves(x, 8, BM, N)
        a, b = _halves(lo, 4, BM, N)
        c, d = _halves(hi, 4, BM, N)
        out = _halves(a, 2, BM, N) + _halves(b, 2, BM, N) + _halves(c, 2, BM, N) + _halves(d, 2, BM, N)
    return out


@triton.jit
def _ln_t(x, w, b, C: tl.constexpr, eps):
    """bb.fused._ln with its weight and bias already loaded."""
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / C + eps)
    return xc * rstd[:, None] * w[None, :] + b[None, :]


@triton.jit
def _k_block_g(x_ptr, u_ptr, nb_ptr, wc_ptr, bc_ptr, wu_ptr, bu_ptr, ln_w, ln_b, eps, wd_ptr, bd_ptr, uo_ptr, S, NIT,
               TPI: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, PDL: tl.constexpr,
               SX: tl.constexpr, SU: tl.constexpr, G: tl.constexpr, STACK: tl.constexpr, VEC: tl.constexpr):
    b0 = tl.program_id(1).to(tl.int64) * G                                     # the group's first shot
    x_ptr += b0 * SX
    u_ptr += b0 * SU
    uo_ptr += b0 * SU
    if PDL:
        gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    qq = tl.arange(0, TPI * CB)
    tt = tl.arange(0, TPI)
    if STACK:
        gr = tl.arange(0, G * BM)                                               # stacked row: shot gr // BM
        srow = tl.program_id(0) * BM + gr % BM
        sh = (gr // BM) * SU
        acc = tl.zeros((G * BM, CB), dtype=tl.float32)
    else:
        accs = ()
        for g in tl.static_range(G):
            accs = accs + (tl.zeros((BM, CB), dtype=tl.float32),)
    if PDL:
        gdc_wait()                                                              # u and x of the kernel before
    for it in tl.range(0, NIT):
        w = tl.load(wc_ptr + (it * (TPI * CB) + qq)[:, None] * CB + cb[None, :])    # once for the group
        if STACK:
            nbv = tl.load(nb_ptr + srow[:, None] * (NIT * TPI) + (it * TPI + tt)[None, :])
            off = sh[:, None] + tl.reshape(nbv[:, :, None] * CB + cb[None, None, :], (G * BM, TPI * CB))
            if VEC:
                off = tl.max_contiguous(tl.multiple_of(off, [1, CB]), [1, CB])
            acc += tl.dot(tl.load(u_ptr + off), w)
        else:
            nbv = tl.load(nb_ptr + rows[:, None] * (NIT * TPI) + (it * TPI + tt)[None, :])   # neighbour site
            off = tl.reshape(nbv[:, :, None] * CB + cb[None, None, :], (BM, TPI * CB))
            if VEC:
                off = tl.max_contiguous(tl.multiple_of(off, [1, CB]), [1, CB])
            nxt = ()
            for g in tl.static_range(G):
                nxt = nxt + (accs[g] + tl.dot(tl.load(u_ptr + g * SU + off), w),)
            accs = nxt
    bc = tl.load(bc_ptr + cb)
    if STACK:
        vs = _unstack(_silu(acc + bc[None, :]).to(tl.float16), G, BM, CB)
    else:
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


class MultiShot(pc_fused.Batched):
    """pc_fused.Batched with cfg's launches replaced (module docstring); called and run as Batched."""

    def __init__(self, model, nmax, cfg):
        self.cfg = c = {**DEFAULT, **cfg}
        if set(c) != set(DEFAULT) or c["form"] not in ("safe", "stack") or c["pool"] not in ("", "skip") \
                or c["bg"] not in (1, 2, 4, 8) or c["vg"] not in (0, 1, 2, 4, 8, 16) or c["psi8"] not in (0, 1, 2) \
                or c["bs"] not in (0, 1, 2, 3) or c["vw"] not in (1, 2, 4, 8, 16):
            raise ValueError(f"cfg {cfg}")
        g = max(c["bg"], c["vg"], 1)
        super().__init__(model, -(-nmax // g) * g)                              # whole groups
        self.nmax = nmax
        if any(not 0 <= i < len(self.blk) - 1 for i in c["blocks"]):
            raise ValueError(f"blocks {c['blocks']}: the last block ({len(self.blk) - 1}) keeps its kernel")
        if c["psi8"]:
            p8 = self.psi.to(torch.int8)
            if not torch.equal(p8.to(self.psi.dtype), self.psi):
                raise ValueError("psi has entries other than -1, 0 and +1")
            self.psi8 = p8.contiguous()

    def launches(self, x, n):
        """Batched.launches with cfg's launches replaced, in the same order."""
        c, C, CB, S, ns, pdl = self.cfg, self.C, self.CB, self.S, self.stages, self.pdl
        lp = {"launch_pdl": True} if pdl else {}
        SX, SU, SUU = self.x[0].numel(), self.u[0][0].numel(), self.U[0].numel()
        out = []
        for name, fn in super().launches(x, n):
            if name.startswith("block") and int(name[5:]) in c["blocks"]:
                i, G = int(name[5:]), c["bg"]
                p, q = self.blk[i], self.blk[i + 1]
                fn = (lambda p=p, q=q, i=i, G=G: _k_block_g[(self.NT, -(-n // G))](
                    self.x, self.u[i % 2], self.nb, p["wc"], p["bc"], p["wu"], p["bu"], q["ln_w"], q["ln_b"], q["eps"],
                    q["wd"], q["bd"], self.u[(i + 1) % 2], S, self.NIT, TPI=self.tpi, C=C, CB=CB, BM=BM, PDL=pdl,
                    SX=SX, SU=SU, G=G, STACK=c["form"] == "stack", VEC=bool(c["vec"]), num_warps=4,
                    num_stages=c["bs"] or ns, **lp))
            elif name == "pool" and c["pool"] == "skip":
                o = self.pool
                fn = (lambda o=o: pc_fused._k_pool[(-(-self.Q // self.bq), n)](
                    self.U, self.P, o["wo"], o["u0"], o["mw"], o["mb"], o["meps"], o["w1"], o["b1"], o["w2"], o["b2"],
                    self.gt, x, self.D, self.wq, self.s, self.z, self.Q, self.SP, QP=self.QP, BS=BS, BQ=self.bq, C=C,
                    H=self.H, DH=self.DH, FF=o["FF"], FC=min(o["FF"], 64), NNZ=self.NNZ, PDL=pdl, SUU=SUU,
                    num_warps=4, num_stages=ns, **lp))
            elif name == "vote" and (c["vg"] or c["psi8"]):
                G = c["vg"] or 1
                fn = (lambda G=G: _k_vote_g[(self.NP, -(-n // G))](
                    self.s, self.psi8 if c["psi8"] else self.psi, self.beta, self.m1, self.i1, self.m2, self.QP,
                    K=self.K, BY=BY, BQV=BQV, PDL=pdl, NP=self.NP, G=G, PIN=c["psi8"] == 1, num_warps=c["vw"],
                    num_stages=ns, **lp))
            out.append((name, fn))
        return out


def engine(model, nmax, cfg):
    """Today's engine (pc_fused.Batched) for an empty cfg, else MultiShot."""
    return MultiShot(model, nmax, cfg) if cfg else pc_fused.Batched(model, nmax)
