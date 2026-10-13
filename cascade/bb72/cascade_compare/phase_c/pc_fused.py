"""The ConvLines networks' fused engine over a batch of shots (PHASE_C_PLAN.md 2.4): the reads of phase C and the timed
paths at batch sizes above 1.

bb.engine.Deployed(model, "fused") is registered at batch 1: one shot through bb.fused's five Triton kernels. Batched
runs the same kernels over n shots per call. They are bb.fused's kernels copied with one more grid axis, the shot:
program (i, b) runs tile i of shot b with the instructions of the batch-1 program i, on shot b's buffers. Only the base
addresses move (the first lines of each kernel); the tiles, loops, arithmetic and launch options are bb.fused's. Each
shot's decision and margin are therefore bitwise those of the batch-1 engine at any batch size. pc_speed.py shows it on
the screen's shots (C3) and requires it in the claim-point check (C4b). bb.fused's tables and constants are reused
(Engine.__init__ at the pinned commit), so its precision is fused.PRECISION."""
import hashlib
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from bb import fused
from bb.fused import BM, BQV, BS, BY, KCH, _down, _ln, _p2, _silu


@triton.jit
def _k_stem(det_ptr, nbs_ptr, wst_ptr, p_ptr, x_ptr, ln_w, ln_b, eps, wd_ptr, bd_ptr, u_ptr, D, S, KS,
            KCH: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, PDL: tl.constexpr,
            SX: tl.constexpr, SU: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)                                           # the shot
    det_ptr += b * D
    x_ptr += b * SX
    u_ptr += b * SU
    if PDL:
        gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, C)
    kk = tl.arange(0, KCH)
    acc = tl.load(p_ptr + rows[:, None] * C + cols[None, :])                  # bias + time bias; 0 on padded rows
    for k0 in tl.range(0, KS, KCH):
        idx = tl.load(nbs_ptr + rows[:, None] * KS + (k0 + kk)[None, :])      # detector index, D = none
        bit = tl.load(det_ptr + idx, mask=idx < D, other=0).to(tl.float16)
        acc += tl.dot(bit, tl.load(wst_ptr + (k0 + kk)[:, None] * C + cols[None, :]))
    tl.store(x_ptr + rows[:, None] * C + cols[None, :], acc)
    _down(acc, rows, S, ln_w, ln_b, eps, wd_ptr, bd_ptr, u_ptr, C, CB)


@triton.jit
def _k_block(x_ptr, u_ptr, nb_ptr, wc_ptr, bc_ptr, wu_ptr, bu_ptr, ln_w, ln_b, eps, wd_ptr, bd_ptr, uo_ptr, wg_ptr,
             bg_ptr, wv_ptr, bv_ptr, S, NIT, TPI: tl.constexpr, C: tl.constexpr, CB: tl.constexpr,
             BM: tl.constexpr, LAST: tl.constexpr, PDL: tl.constexpr, SX: tl.constexpr, SU: tl.constexpr,
             SUO: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)                                           # the shot
    x_ptr += b * SX
    u_ptr += b * SU
    uo_ptr += b * SUO
    if PDL:
        gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    qq = tl.arange(0, TPI * CB)
    tt = tl.arange(0, TPI)
    acc = tl.zeros((BM, CB), dtype=tl.float32)
    if PDL:
        gdc_wait()                                                              # u and x of the kernel before
    for it in tl.range(0, NIT):
        nbv = tl.load(nb_ptr + rows[:, None] * (NIT * TPI) + (it * TPI + tt)[None, :])   # neighbour site, S = zero row
        a = tl.load(u_ptr + tl.reshape(nbv[:, :, None] * CB + cb[None, None, :], (BM, TPI * CB)))
        acc += tl.dot(a, tl.load(wc_ptr + (it * (TPI * CB) + qq)[:, None] * CB + cb[None, :]))
    v = _silu(acc + tl.load(bc_ptr + cb)[None, :]).to(tl.float16)
    x = (tl.load(x_ptr + rows[:, None] * C + cols[None, :]) + tl.load(bu_ptr + cols)[None, :]
         + tl.dot(v, tl.load(wu_ptr + cb[:, None] * C + cols[None, :])))
    if LAST:
        y = _ln(x, ln_w, ln_b, cols, C, eps).to(tl.float16)
        for c0 in tl.static_range(0, C, 64):                                   # 64 columns at a time (shared memory)
            cc = c0 + tl.arange(0, 64)
            g = tl.sigmoid(tl.dot(y, tl.load(wg_ptr + cols[:, None] * C + cc[None, :])) + tl.load(bg_ptr + cc)[None, :])
            vv = tl.dot(y, tl.load(wv_ptr + cols[:, None] * C + cc[None, :])) + tl.load(bv_ptr + cc)[None, :]
            tl.store(uo_ptr + rows[:, None] * C + cc[None, :], (g * vv).to(tl.float16), mask=(rows < S)[:, None])
    else:
        tl.store(x_ptr + rows[:, None] * C + cols[None, :], x)
        _down(x, rows, S, ln_w, ln_b, eps, wd_ptr, bd_ptr, uo_ptr, C, CB)


@triton.jit
def _k_pool(U_ptr, p_ptr, wo_ptr, u0_ptr, mw_ptr, mb_ptr, meps, w1_ptr, b1_ptr, w2_ptr, b2, gt_ptr, det_ptr, D, wq_ptr,
            s_ptr, z_ptr, Q, SP, QP: tl.constexpr, BS: tl.constexpr, BQ: tl.constexpr, C: tl.constexpr,
            H: tl.constexpr, DH: tl.constexpr, FF: tl.constexpr, FC: tl.constexpr, NNZ: tl.constexpr, PDL: tl.constexpr,
            SUU: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)                                           # the shot
    U_ptr += b * SUU
    det_ptr += b * D
    s_ptr += b * QP
    z_ptr += b * QP
    if PDL:
        gdc_launch_dependents()
    qr = tl.program_id(0) * BQ + tl.arange(0, BQ)
    cols = tl.arange(0, C)
    dc = tl.arange(0, DH)
    sb = tl.arange(0, BS)
    u = tl.load(u0_ptr + qr[:, None] * C + cols[None, :])                     # q0 + e_k + bo
    if PDL:
        gdc_wait()                                                              # U of the last block
    for h in tl.static_range(H):
        acc = tl.zeros((BQ, DH), dtype=tl.float32)
        for s0 in tl.range(0, SP, BS):
            pt = tl.load(p_ptr + (h * QP + qr)[:, None] * SP + (s0 + sb)[None, :])
            acc += tl.dot(pt, tl.load(U_ptr + (s0 + sb)[:, None] * C + (h * DH + dc)[None, :]))
        u += tl.dot(acc.to(tl.float16), tl.load(wo_ptr + (h * DH + dc)[:, None] * C + cols[None, :]))
    y = _ln(u, mw_ptr, mb_ptr, cols, C, meps).to(tl.float16)
    z = tl.zeros((BQ,), dtype=tl.float32) + b2
    for f0 in tl.static_range(0, FF, FC):
        ff = f0 + tl.arange(0, FC)
        hd = tl.dot(y, tl.load(w1_ptr + cols[:, None] * FF + ff[None, :])) + tl.load(b1_ptr + ff)[None, :]
        z += tl.sum(_silu(hd) * tl.load(w2_ptr + ff)[None, :], 1)
    nz = tl.arange(0, NNZ)
    di = tl.load(gt_ptr + qr[:, None] * NNZ + nz[None, :])                    # the column's detectors, D = none
    par = tl.sum(tl.load(det_ptr + di, mask=di < D, other=0).to(tl.int32), 1) % 2
    s = (2.0 * par.to(tl.float32) - 1.0) * z * tl.load(wq_ptr + qr) * 0.5
    tl.store(s_ptr + qr, s, mask=qr < Q)
    tl.store(z_ptr + qr, z, mask=qr < Q)


@triton.jit
def _k_vote(s_ptr, psi_ptr, beta_ptr, m1_ptr, i1_ptr, m2_ptr, QP, K: tl.constexpr, BY: tl.constexpr,
            BQV: tl.constexpr, PDL: tl.constexpr, NP: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)                                           # the shot
    s_ptr += b * QP
    m1_ptr += b * NP
    i1_ptr += b * NP
    m2_ptr += b * NP
    if PDL:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    yl = tl.arange(0, BY)
    ys = pid * BY + yl
    qv = tl.arange(0, BQV)
    acc = tl.zeros((BY,), dtype=tl.float32)
    if PDL:
        gdc_wait()                                                              # s of the pool
    for q0 in tl.range(0, QP, BQV):
        psi = tl.load(psi_ptr + ys[:, None] * QP + (q0 + qv)[None, :]).to(tl.float32)
        acc += tl.sum(psi * tl.load(s_ptr + q0 + qv)[None, :], 1)
    for j in tl.static_range(K):
        acc += (((ys >> j) & 1).to(tl.float32) - 0.5) * tl.load(beta_ptr + j)
    m1 = tl.max(acc, 0)
    i1 = tl.argmax(acc, 0)
    m2 = tl.max(tl.where(yl == i1, float("-inf"), acc), 0)
    tl.store(m1_ptr + pid, m1)
    tl.store(i1_ptr + pid, pid * BY + i1)
    tl.store(m2_ptr + pid, m2)


@triton.jit
def _k_top2(m1_ptr, i1_ptr, m2_ptr, bits_ptr, margin_ptr, NP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
            PDL: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)                                           # the shot
    m1_ptr += b * NP
    i1_ptr += b * NP
    m2_ptr += b * NP
    bits_ptr += b * K
    margin_ptr += b
    pp = tl.arange(0, NP)
    if PDL:
        gdc_wait()                                                              # the vote's partial top 2
    m1 = tl.load(m1_ptr + pp)
    i1 = tl.load(i1_ptr + pp)
    m2 = tl.load(m2_ptr + pp)
    top = tl.max(m1, 0)
    ps = tl.argmax(m1, 0)
    sec = tl.maximum(tl.max(tl.where(pp == ps, float("-inf"), m1), 0), tl.sum(tl.where(pp == ps, m2, 0.0), 0))
    cls = tl.sum(tl.where(pp == ps, i1, 0), 0)
    kk = tl.arange(0, KP)
    tl.store(bits_ptr + kk, ((cls >> kk) & 1).to(tl.uint8), mask=kk < K)
    tl.store(margin_ptr, top - sec)


class Batched(fused.Engine):
    """Batched(model, nmax)(det (n, D) uint8 on the GPU, n <= nmax) -> (bits (n, K) uint8, margin (n,) fp32), views of
    static output buffers; run(det) takes any number of rows, nmax at a time."""

    def __init__(self, model, nmax):
        super().__init__(model)                            # bb.fused's tables and launch options
        dev, f16, f32 = self.psi.device, torch.float16, torch.float32
        self.nmax = nmax
        self.x = torch.empty(nmax, self.NT * BM, self.C, device=dev, dtype=f32)
        self.u = [torch.zeros(nmax, self.S + 1, self.CB, device=dev, dtype=f16) for _ in range(2)]
        self.U = torch.zeros(nmax, self.SP, self.C, device=dev, dtype=f16)
        self.s = torch.zeros(nmax, self.QP, device=dev, dtype=f32)
        self.z = torch.zeros(nmax, self.QP, device=dev, dtype=f32)
        self.m1 = torch.empty(nmax, self.NP, device=dev, dtype=f32)
        self.i1 = torch.empty(nmax, self.NP, device=dev, dtype=torch.int32)
        self.m2 = torch.empty(nmax, self.NP, device=dev, dtype=f32)
        self.bits = torch.zeros(nmax, self.K, device=dev, dtype=torch.uint8)
        self.margin = torch.zeros(nmax, device=dev, dtype=f32)

    def launches(self, x, n):
        """bb.fused.Engine.launches on n shots: the same launches with the grid (tiles, n)."""
        C, CB, S, D, ns, pdl = self.C, self.CB, self.S, self.D, self.stages, self.pdl
        lp = {"launch_pdl": True} if pdl else {}           # every kernel after the first
        SX, SU, SUU = self.x[0].numel(), self.u[0][0].numel(), self.U[0].numel()
        b0 = self.blk[0]
        out = [("stem", lambda: _k_stem[(self.NT, n)](
            x, self.nbs, self.wst, self.P0, self.x, b0["ln_w"], b0["ln_b"], b0["eps"], b0["wd"], b0["bd"], self.u[0], D, S,
            self.KS, KCH=KCH, C=C, CB=CB, BM=BM, PDL=pdl, SX=SX, SU=SU, num_warps=4, num_stages=ns))]
        L = len(self.blk)
        for i, p in enumerate(self.blk):
            last = i == L - 1
            q = self.ro if last else self.blk[i + 1]
            out.append((f"block{i}", lambda p=p, q=q, i=i, last=last: _k_block[(self.NT, n)](
                self.x, self.u[i % 2], self.nb, p["wc"], p["bc"], p["wu"], p["bu"], q["ln_w"], q["ln_b"], q["eps"],
                p["wd"] if last else q["wd"], p["bd"] if last else q["bd"], self.U if last else self.u[(i + 1) % 2],
                self.ro["wg"], self.ro["bg"], self.ro["wv"], self.ro["bv"], S, self.NIT, TPI=self.tpi, C=C, CB=CB,
                BM=BM, LAST=last, PDL=pdl, SX=SX, SU=SU, SUO=SUU if last else SU, num_warps=4, num_stages=ns, **lp)))
        o = self.pool
        out.append(("pool", lambda: _k_pool[(self.NQT, n)](
            self.U, self.P, o["wo"], o["u0"], o["mw"], o["mb"], o["meps"], o["w1"], o["b1"], o["w2"], o["b2"], self.gt, x,
            D, self.wq, self.s, self.z, self.Q, self.SP, QP=self.QP, BS=BS, BQ=self.bq, C=C, H=self.H, DH=self.DH,
            FF=o["FF"], FC=min(o["FF"], 64), NNZ=self.NNZ, PDL=pdl, SUU=SUU, num_warps=4, num_stages=ns, **lp)))
        out.append(("vote", lambda: _k_vote[(self.NP, n)](
            self.s, self.psi, self.beta, self.m1, self.i1, self.m2, self.QP, K=self.K, BY=BY, BQV=BQV, PDL=pdl,
            NP=self.NP, num_warps=4, num_stages=ns, **lp)))
        out.append(("top2", lambda: _k_top2[(1, n)](
            self.m1, self.i1, self.m2, self.bits, self.margin, NP=self.NP, K=self.K, KP=_p2(self.K), PDL=pdl,
            num_warps=1, **lp)))
        return out

    def __call__(self, x, n=None):
        n = len(x) if n is None else n
        if not 0 < n <= self.nmax or x.dtype != torch.uint8 or not x.is_contiguous() or x.shape[1] != self.D:
            raise ValueError(f"det {tuple(x.shape)} {x.dtype}: needs 1 to {self.nmax} contiguous uint8 rows of {self.D}")
        for _, fn in self.launches(x, n):
            fn()
        return self.bits[:n], self.margin[:n]

    @torch.no_grad()
    def run(self, det):
        """det (N, D) uint8 on the GPU -> (bits (N, K) uint8, margin (N,) fp32) on the GPU, nmax shots per call."""
        bits = torch.empty(len(det), self.K, dtype=torch.uint8, device=det.device)
        margin = torch.empty(len(det), dtype=torch.float32, device=det.device)
        for i in range(0, len(det), self.nmax):
            d = det[i:i + self.nmax].contiguous()
            b, m = self(d)
            bits[i:i + len(d)] = b
            margin[i:i + len(d)] = m
        return bits, margin


def signature(nmax):
    """What a fused read depends on beyond the network: the kernels' sources, the batch, the precision, Triton."""
    def sha(f):
        h = hashlib.sha256()
        h.update(open(f, "rb").read())
        return h.hexdigest()[:16]
    return {"engine": "fused (bb.fused's kernels batched by pc_fused.Batched)", "batch": nmax,
            "precision": fused.PRECISION, "triton": triton.__version__,
            "src": {"bb/fused.py": sha(fused.__file__), "pc_fused.py": sha(os.path.abspath(__file__))}}
