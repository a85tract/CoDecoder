"""Triton kernels for Cascade's bipartite convolution (bb.cascade_net, conv_impl 'triton'): the same arithmetic as
cascade_net._GatherMM without the 12-fold gathered tensor in memory (the paper: "We implement these bipartite
convolutions using custom Triton kernels that exploit the regular structure for efficient memory access").

  y[b, n] = sum_r x[b, idx[n, r]] @ W_r^T            W (C_out, 12 C_in), W_r = W[:, r C_in:(r+1) C_in]
  forward and input gradient: one implicit-GEMM kernel (_gmm), each program a BLOCK_M x BLOCK_N tile of the output,
  the K loop over the 12 relations reading the source rows by index (pad index -> zeros);
  weight gradient: _gmm_wgrad, each program one relation and one tile of W, a slice of the rows, partial sums in fp32
  reduced by torch (deterministic). bf16 inputs, fp32 accumulation, bf16 outputs (fp32 weight gradient).
  _gmm is autotuned over BLOCK_M, pipeline stages and warps per (CA, CY, K) on first use, among the configs that fit
  the card's shared memory; the forward reads a transposed copy of W so that every weight tile is loaded
  row-contiguously. The choice is cached on disk (Triton's cache_results, in TRITON_CACHE_DIR), so every later process
  on the same GPU architecture (a resumed segment) runs the same configs. Both launches are torch.library custom ops,
  opaque to torch.compile."""
import inspect

import torch
import triton
import triton.language as tl

_AUTOTUNE_CACHE = {"cache_results": True} if "cache_results" in inspect.signature(triton.autotune).parameters else {}


def _gmm_configs():
    """BLOCK_M, pipeline stages and warps; on a GPU, only the configs whose pipelined A and B tiles (BLOCK_N <= 128,
    BLOCK_K = 64, bf16) fit the card's shared memory with 8 KB to spare (an L40 or RTX PRO 6000 has about 100 KB per
    block, an H200 228 KB)."""
    cs = [(bm, st, w) for bm in (64, 128) for st in (2, 3, 4) for w in (4, 8)]
    if torch.cuda.is_available():
        smem = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        cs = [c for c in cs if c[1] * (c[0] * 64 + 64 * 128) * 2 + 8192 <= smem] or [(64, 2, 4)]
    return [triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=st) for bm, st, w in cs]


@triton.autotune(configs=_gmm_configs(), key=["CA", "CY", "K"], **_AUTOTUNE_CACHE)
@triton.jit
def _gmm(A, IDX, Bw, Y, M, NT, NS, s_r, s_k, s_j,
         CA: tl.constexpr, CY: tl.constexpr, K: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Y[m, j] = sum_r sum_k A[shot(m) NS + IDX[node(m), r], k] Bw[r s_r + k s_k + j s_j] (IDX = NS: zero row)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = rm < M
    mn = rn < CY
    shot = rm // NT
    node = rm - shot * NT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for r in tl.static_range(K):
        src = tl.load(IDX + node * K + r, mask=mm, other=NS)
        ok = mm & (src < NS)
        row = (shot * NS + src).to(tl.int64) * CA
        for k0 in tl.static_range(0, CA, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(A + row[:, None] + rk[None, :], mask=ok[:, None] & (rk[None, :] < CA), other=0.0)
            w = tl.load(Bw + r * s_r + rk[:, None] * s_k + rn[None, :] * s_j,
                        mask=(rk[:, None] < CA) & mn[None, :], other=0.0)
            acc += tl.dot(a, w)
    out = rm.to(tl.int64)[:, None] * CY + rn[None, :]
    tl.store(Y + out, acc.to(Y.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def _gmm_wgrad(X, GY, IDX, P, M, NT, NS, ROWS_PER,
               CI: tl.constexpr, CO: tl.constexpr, K: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_I: tl.constexpr):
    """P[split, o, r CI + i] = sum over this split's rows m of GY[m, o] X[shot(m) NS + IDX[node(m), r], i]."""
    pid = tl.program_id(0)                               # relation r and the (o, i) tile
    split = tl.program_id(1)
    n_i = tl.cdiv(CI, BLOCK_I)
    n_o = tl.cdiv(CO, BLOCK_O)
    r = pid // (n_o * n_i)
    rem = pid - r * (n_o * n_i)
    po = rem // n_i
    pi = rem - po * n_i
    ro = po * BLOCK_O + tl.arange(0, BLOCK_O)
    ri = pi * BLOCK_I + tl.arange(0, BLOCK_I)
    acc = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
    m0 = split * ROWS_PER
    for mb in range(m0, tl.minimum(m0 + ROWS_PER, M), BLOCK_M):
        rm = mb + tl.arange(0, BLOCK_M)
        mm = rm < tl.minimum(m0 + ROWS_PER, M)
        shot = rm // NT
        node = rm - shot * NT
        src = tl.load(IDX + node * K + r, mask=mm, other=NS)
        ok = mm & (src < NS)
        row = (shot * NS + src).to(tl.int64) * CI
        x = tl.load(X + row[:, None] + ri[None, :], mask=ok[:, None] & (ri[None, :] < CI), other=0.0)
        g = tl.load(GY + rm.to(tl.int64)[:, None] * CO + ro[None, :], mask=mm[:, None] & (ro[None, :] < CO), other=0.0)
        acc += tl.dot(tl.trans(g), x)
    out = P + split.to(tl.int64) * CO * K * CI + ro[:, None] * (K * CI) + r * CI + ri[None, :]
    tl.store(out, acc, mask=(ro[:, None] < CO) & (ri[None, :] < CI))


def _blk(c):
    return 128 if c >= 128 else (64 if c >= 64 else max(16, triton.next_power_of_2(c)))


def gmm(a, idx, w, transposed=False):
    """a (B, NS, CA) bf16 contiguous, idx (NT, K) int32, w (C_out, K C_in) bf16 -> (B, NT, CY) bf16.
    transposed=False: CA = C_in, CY = C_out (the forward); True: CA = C_out, CY = C_in (the input gradient through
    the inverse table)."""
    B, NS, CA = a.shape
    NT, K = idx.shape
    C_out, KC = w.shape
    C_in = KC // K
    if transposed:                                       # B_r[o, i] = w[o, r C_in + i]: rows contiguous in i
        bw, CY, s_r, s_k, s_j = w, C_in, C_in, K * C_in, 1
    else:                                                # B_r[k, j] = w[j, r C_in + k] = wt[r C_in + k, j]
        bw, CY, s_r, s_k, s_j = w.t().contiguous(), C_out, C_in * C_out, C_out, 1
    y = torch.empty(B, NT, CY, device=a.device, dtype=a.dtype)
    M = B * NT
    BN, BK = _blk(CY), min(64, _blk(CA))
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(CY, BN))
    _gmm[grid](a, idx, bw, y, M, NT, NS, s_r, s_k, s_j, CA=CA, CY=CY, K=K, BLOCK_N=BN, BLOCK_K=BK)
    return y


def gmm_wgrad(x, gy, idx, splits=None):
    """x (B, NS, C_in), gy (B, NT, C_out), idx (NT, K) -> dW (C_out, K C_in) fp32."""
    B, NS, CI = x.shape
    _, NT, CO = gy.shape
    K = idx.shape[1]
    M = B * NT
    BM, BO, BI = 64, _blk(CO), _blk(CI)
    tiles = K * triton.cdiv(CO, BO) * triton.cdiv(CI, BI)
    if splits is None:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        splits = max(1, min(triton.cdiv(M, 4 * BM), (4 * sms + tiles - 1) // tiles))
    rows = triton.cdiv(triton.cdiv(M, splits), BM) * BM
    splits = triton.cdiv(M, rows)
    P = torch.empty(splits, CO, K * CI, device=x.device, dtype=torch.float32)
    _gmm_wgrad[(tiles, splits)](x, gy, idx, P, M, NT, NS, rows, CI=CI, CO=CO, K=K, BLOCK_M=BM, BLOCK_O=BO,
                                BLOCK_I=BI, num_warps=8 if BO * BI >= 128 * 128 else 4, num_stages=2)
    return P.sum(0)


@torch.library.custom_op("cascade::gmm", mutates_args=())
def gmm_op(a: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, transposed: bool) -> torch.Tensor:
    """gmm as an opaque operator, so torch.compile does not trace Triton's launcher and autotuner."""
    return gmm(a, idx, w, transposed)


@gmm_op.register_fake
def _(a, idx, w, transposed):
    K = idx.shape[1]
    return a.new_empty(a.shape[0], idx.shape[0], w.shape[1] // K if transposed else w.shape[0])


@torch.library.custom_op("cascade::gmm_wgrad", mutates_args=())
def gmm_wgrad_op(x: torch.Tensor, gy: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return gmm_wgrad(x, gy, idx)


@gmm_wgrad_op.register_fake
def _(x, gy, idx):
    return x.new_empty(gy.shape[2], idx.shape[1] * x.shape[2], dtype=torch.float32)


class GatherMMTriton(torch.autograd.Function):
    """Drop-in for cascade_net._GatherMM (the same signature; idx and the inverse table as int64 on the device)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x, w, idx, inv):
        x = x.contiguous()
        w = w.contiguous()
        idx32 = idx.to(torch.int32)
        ctx.save_for_backward(x, w, idx32, inv)
        return gmm_op(x, idx32, w, False)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gy):
        x, w, idx32, inv = ctx.saved_tensors
        gy = gy.contiguous().to(x.dtype)
        K = idx32.shape[1]
        NT = idx32.shape[0]
        gx = gw = None
        if ctx.needs_input_grad[0]:
            tgt = torch.where(inv < NT * K, inv // K, torch.full_like(inv, NT)).to(torch.int32)
            gx = gmm_op(gy, tgt, w, True)
        if ctx.needs_input_grad[1]:
            gw = gmm_wgrad_op(x, gy, idx32).to(w.dtype)
        return gx, gw, None, None
