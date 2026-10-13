"""The engine port (bb/PLAN.md 2.4, rung 2): one shot of ConvLines-BB (bb.net) in L + 4 Triton kernels, arch.engine's
scheme carried to the torus. Detector bits (1, D) uint8 on the GPU -> decision bits (1, K) uint8 and margin (1,) fp32:

  stem       per tile of BM sites (t, a, b): the detector bits at the stem's taps x channels (one gather through a fixed
             table, circular in (a, b), zero outside 0 <= t < T) times the tap weights + the bias and time bias -> the
             residual x; epilogue: block 1's LayerNorm, down-projection and SiLU -> u
  block i    per tile: u gathered at the block's taps (fixed table; a zero row stands for t outside the grid), TPI taps
             per (BM, TPI Cb) x (TPI Cb, Cb) product, SiLU, up-projection, residual add; epilogue: block i + 1's
             LayerNorm, down-projection, SiLU -> u' (the last block: the read-out's LayerNorm, its content gate (the
             query-key product folded into one matrix per head) times the values -> U)
  pool       per tile of BQ translate queries: the pooled values (the fixed position gates P / N0, an input-independent
             table, times U, per head), out-projection, the query's residual, LayerNorm, MLP -> z; the translates'
             detector parities g (a fixed table of each column's detectors) -> s = -(-1)^g z w / 2
  vote       per tile of BY classes y: score(y) = sum_kg Psi[y, kg] s_kg + sum_j (y_j - 1/2) beta_j, the tile's top 2
  top2       one program: the global top 2 -> the decision's bits and the margin (top-1 minus top-2 score)

Arithmetic: PRECISION. No atomics and fixed launch shapes, so a call is deterministic."""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

PRECISION = "fp16 matmul inputs with fp32 accumulation; fp32 residual stream, LayerNorm, gates, vote and margin"
BM, BQ, BS, BY, BQV, TPI, KCH, STAGES = 16, 16, 64, 128, 128, 8, 64, 2


def _p2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


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
def _down(x, rows, S, ln_w, ln_b, eps, wd_ptr, bd_ptr, u_ptr, C: tl.constexpr, CB: tl.constexpr):
    """u = SiLU(LayerNorm(x) Wd + bd) -> u_ptr (S + 1, CB) fp16 for the tile's rows below S (row S stays zero)."""
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    y = _ln(x, ln_w, ln_b, cols, C, eps).to(tl.float16)
    u = _silu(tl.dot(y, tl.load(wd_ptr + cols[:, None] * CB + cb[None, :])) + tl.load(bd_ptr + cb)[None, :])
    tl.store(u_ptr + rows[:, None] * CB + cb[None, :], u.to(tl.float16), mask=(rows < S)[:, None])


@triton.jit
def _k_stem(det_ptr, nbs_ptr, wst_ptr, p_ptr, x_ptr, ln_w, ln_b, eps, wd_ptr, bd_ptr, u_ptr, D, S, KS,
            KCH: tl.constexpr, C: tl.constexpr, CB: tl.constexpr, BM: tl.constexpr, PDL: tl.constexpr):
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
             BM: tl.constexpr, LAST: tl.constexpr, PDL: tl.constexpr):
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
            H: tl.constexpr, DH: tl.constexpr, FF: tl.constexpr, FC: tl.constexpr, NNZ: tl.constexpr, PDL: tl.constexpr):
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
            BQV: tl.constexpr, PDL: tl.constexpr):
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


class Engine:
    """Engine(model)(x (1, D) uint8 on the GPU) -> (bits (1, K) uint8, margin (1,) fp32), static output buffers."""

    def __init__(self, model, tpi=TPI, stages=STAGES, bq=BQ, pdl=None):
        model = model.eval()
        if pdl is None:                                    # programmatic dependent launch where the GPU has it
            pdl = torch.cuda.get_device_capability() >= (9, 0)
        self.tpi, self.stages, self.bq, self.pdl = tpi, stages, bq, pdl
        dev = next(model.parameters()).device
        f16, f32, i32 = torch.float16, torch.float32, torch.int32
        T, l, m, n2, D, K, G = model.T, model.l, model.m, model.n2, model.D, model.K, model.G
        c = model.cfg
        C, CB, H = c["width"], c["bott"], c["heads"]
        S, Q = T * n2, K * G
        self.D, self.K, self.S, self.Q, self.C, self.CB, self.H, self.DH = D, K, S, Q, C, CB, H, C // H
        self.NT = -(-S // BM)
        SP1 = self.NT * BM                                     # x rows (tiles of BM)
        self.SP = -(-SP1 // BS) * BS                           # U rows and P columns (chunks of BS)
        self.QP = -(-Q // BQV) * BQV
        self.NQT = self.QP // bq
        self.NP = 2 ** K // BY
        t, a, b = np.unravel_index(np.arange(S), (T, l, m))

        def nbr(tap):                                          # neighbour site of every site, -1 outside 0 <= t < T
            dt, da, db = tap
            tt = t + dt
            return np.where((tt >= 0) & (tt < T), tt * n2 + ((a + da) % l) * m + (b + db) % m, -1)

        def tap_w(conv, tap):                                  # (cout, cin) weight of one tap
            dt, da, db = tap
            return (conv.weight * conv.mask)[:, :, dt + conv.dt, da - conv.alo, db - conv.blo].float()

        with torch.no_grad():
            st = model.stem
            gidx = model.gidx.cpu().numpy()
            self.KS = max(KCH, _p2(2 * len(st.taps)))
            nbs = np.full((SP1, self.KS), D, np.int64)
            wst = torch.zeros(self.KS, C, device=dev)
            for j, tap in enumerate(st.taps):
                ns = nbr(tap)
                for ch in range(2):
                    nbs[:S, 2 * j + ch] = np.where(ns >= 0, gidx[((ns // n2) * 2 + ch) * n2 + ns % n2], D)
                    wst[2 * j + ch] = tap_w(st, tap)[:, ch]
            self.nbs = torch.as_tensor(nbs, dtype=i32, device=dev).contiguous()
            self.wst = wst.to(f16).contiguous()
            P0 = torch.zeros(SP1, C, device=dev)
            P0[:S] = (st.bias[None, :] + model.tbias[torch.as_tensor(t, device=dev)]).float()
            self.P0 = P0.contiguous()
            blk_taps = model.blocks[0].conv.taps
            self.NIT = -(-len(blk_taps) // tpi)
            nb = np.full((SP1, self.NIT * tpi), S, np.int64)
            for j, tap in enumerate(blk_taps):
                ns = nbr(tap)
                nb[:S, j] = np.where(ns >= 0, ns, S)
            self.nb = torch.as_tensor(nb, dtype=i32, device=dev).contiguous()
            self.blk = []
            for blk in model.blocks:
                wc = torch.zeros(self.NIT * tpi, CB, CB, device=dev)
                for j, tap in enumerate(blk.conv.taps):
                    wc[j] = tap_w(blk.conv, tap).t()                                   # (cin, cout)
                self.blk.append({"ln_w": blk.ln.weight.float().contiguous(), "ln_b": blk.ln.bias.float().contiguous(),
                                 "eps": float(blk.ln.eps), "wd": blk.down.weight.t().to(f16).contiguous(),
                                 "bd": blk.down.bias.float().contiguous(), "wc": wc.reshape(-1, CB).to(f16).contiguous(),
                                 "bc": blk.conv.bias.float().contiguous(), "wu": blk.up.weight.t().to(f16).contiguous(),
                                 "bu": blk.up.bias.float().contiguous()})
            hd = model.head
            DH = self.DH
            q0 = hd.q0.float()
            wg = (hd.key.weight.float().view(H, DH, C) * q0.view(H, DH, 1)).sum(1) * DH ** -0.5     # (H, C)
            self.ro = {"ln_w": hd.ln.weight.float().contiguous(), "ln_b": hd.ln.bias.float().contiguous(),
                       "eps": float(hd.ln.eps), "wg": wg.t().repeat_interleave(DH, 1).to(f16).contiguous(),    # (C, C)
                       "bg": hd.bc.float().repeat_interleave(DH).contiguous(),
                       "wv": hd.v.weight.t().to(f16).contiguous(), "bv": hd.v.bias.float().contiguous()}
            P = torch.zeros(H, self.QP, self.SP, device=dev)
            P[:, :Q, :S] = hd.position().float()
            self.P = P.to(f16).contiguous()
            u0 = torch.zeros(self.QP, C, device=dev)
            u0[:Q] = (q0 + hd.e.float()).repeat_interleave(G, 0) + hd.o.bias.float()
            ml = hd.mlp
            self.pool = {"wo": hd.o.weight.t().to(f16).contiguous(), "u0": u0.contiguous(),
                         "mw": ml[0].weight.float().contiguous(), "mb": ml[0].bias.float().contiguous(),
                         "meps": float(ml[0].eps), "w1": ml[1].weight.t().to(f16).contiguous(),
                         "b1": ml[1].bias.float().contiguous(), "w2": ml[3].weight.reshape(-1).float().contiguous(),
                         "b2": float(ml[3].bias.reshape(-1)[0]), "FF": ml[1].weight.shape[0]}
            gd = model.Gdet.cpu().numpy() > 0.5                                         # (D, Q)
            self.NNZ = max(2, _p2(gd.sum(0).max()))
            gt = np.full((self.QP, self.NNZ), D, np.int64)
            for q in range(Q):
                r = np.flatnonzero(gd[:, q])
                gt[q, :len(r)] = r
            self.gt = torch.as_tensor(gt, dtype=i32, device=dev).contiguous()
            wq = torch.zeros(self.QP, device=dev)
            wq[:Q] = model.w.float().repeat_interleave(G)
            self.wq = wq.contiguous()
            psi = torch.zeros(2 ** K, self.QP, device=dev)
            psi[:, :Q] = model.Psi.float()
            self.psi = psi.to(f16).contiguous()
            self.beta = model.beta.float().contiguous()
        self.x = torch.empty(SP1, C, device=dev, dtype=f32)
        self.u = [torch.zeros(S + 1, CB, device=dev, dtype=f16) for _ in range(2)]
        self.U = torch.zeros(self.SP, C, device=dev, dtype=f16)
        self.s = torch.zeros(self.QP, device=dev, dtype=f32)
        self.z = torch.zeros(self.QP, device=dev, dtype=f32)
        self.m1 = torch.empty(self.NP, device=dev, dtype=f32)
        self.i1 = torch.empty(self.NP, device=dev, dtype=i32)
        self.m2 = torch.empty(self.NP, device=dev, dtype=f32)
        self.bits = torch.zeros(1, K, device=dev, dtype=torch.uint8)
        self.margin = torch.zeros(1, device=dev, dtype=f32)

    def launches(self, x):
        """[(name, launch)] of one call on the detector bits x, in order; __call__ runs them."""
        C, CB, S, D, ns, pdl = self.C, self.CB, self.S, self.D, self.stages, self.pdl
        lp = {"launch_pdl": True} if pdl else {}           # every kernel after the first
        b0 = self.blk[0]
        out = [("stem", lambda: _k_stem[(self.NT,)](
            x, self.nbs, self.wst, self.P0, self.x, b0["ln_w"], b0["ln_b"], b0["eps"], b0["wd"], b0["bd"], self.u[0], D, S,
            self.KS, KCH=KCH, C=C, CB=CB, BM=BM, PDL=pdl, num_warps=4, num_stages=ns))]
        L = len(self.blk)
        for i, p in enumerate(self.blk):
            last = i == L - 1
            q = self.ro if last else self.blk[i + 1]
            out.append((f"block{i}", lambda p=p, q=q, i=i, last=last: _k_block[(self.NT,)](
                self.x, self.u[i % 2], self.nb, p["wc"], p["bc"], p["wu"], p["bu"], q["ln_w"], q["ln_b"], q["eps"],
                p["wd"] if last else q["wd"], p["bd"] if last else q["bd"], self.U if last else self.u[(i + 1) % 2],
                self.ro["wg"], self.ro["bg"], self.ro["wv"], self.ro["bv"], S, self.NIT, TPI=self.tpi, C=C, CB=CB,
                BM=BM, LAST=last, PDL=pdl, num_warps=4, num_stages=ns, **lp)))
        o = self.pool
        out.append(("pool", lambda: _k_pool[(self.NQT,)](
            self.U, self.P, o["wo"], o["u0"], o["mw"], o["mb"], o["meps"], o["w1"], o["b1"], o["w2"], o["b2"], self.gt, x,
            D, self.wq, self.s, self.z, self.Q, self.SP, QP=self.QP, BS=BS, BQ=self.bq, C=C, H=self.H, DH=self.DH,
            FF=o["FF"], FC=min(o["FF"], 64), NNZ=self.NNZ, PDL=pdl, num_warps=4, num_stages=ns, **lp)))
        out.append(("vote", lambda: _k_vote[(self.NP,)](
            self.s, self.psi, self.beta, self.m1, self.i1, self.m2, self.QP, K=self.K, BY=BY, BQV=BQV, PDL=pdl,
            num_warps=4, num_stages=ns, **lp)))
        out.append(("top2", lambda: _k_top2[(1,)](
            self.m1, self.i1, self.m2, self.bits, self.margin, NP=self.NP, K=self.K, KP=_p2(self.K), PDL=pdl,
            num_warps=1, **lp)))
        return out

    def __call__(self, x):
        for _, fn in self.launches(x):
            fn()
        return self.bits, self.margin
