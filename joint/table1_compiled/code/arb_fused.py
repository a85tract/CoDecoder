"""S55'-deep's 4,096 class scores from the fused engine itself (PLAN.md step 2).

pc_fused.Batched computes every class score inside its vote kernel (_k_vote: acc, BY = 128 classes per program, 32
programs per shot) and keeps only each program's top two; the scores never reach memory, and the buffers are reused per
sub-batch. Scored is pc_fused.Batched with one change: its vote launch runs _k_vote_x, which is _k_vote copied line for
line with one more output, tl.store(sc_ptr + ys, acc), the final scores of the program's classes (after the beta
terms, the values whose top two the engine keeps). Every other kernel, table, tile, launch option and the batch are
pc_fused's (and so bb.fused's), so the decisions and margins are those of the read, which arb_scores.py checks bitwise
on every exported shot, together with each program's top two against the exported scores."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

import pc_fused
from bb.fused import BQV, BY


@triton.jit
def _k_vote_x(s_ptr, psi_ptr, beta_ptr, m1_ptr, i1_ptr, m2_ptr, sc_ptr, QP, K: tl.constexpr, BY: tl.constexpr,
              BQV: tl.constexpr, PDL: tl.constexpr, NP: tl.constexpr, NY: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)                                           # the shot
    s_ptr += b * QP
    m1_ptr += b * NP
    i1_ptr += b * NP
    m2_ptr += b * NP
    sc_ptr += b * NY                                                            # added: the shot's 2^K scores
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
    tl.store(sc_ptr + ys, acc)                                                  # added: the class scores


class Scored(pc_fused.Batched):
    """pc_fused.Batched whose vote kernel also writes the 2^K class scores of every shot to self.sc (n, 2^K) fp32."""

    def __init__(self, model, nmax):
        super().__init__(model, nmax)
        self.NY = self.NP * BY
        if self.NY != 2 ** self.K:
            raise ValueError(f"{self.NP} programs of {BY} classes are not the 2^{self.K} classes")
        self.sc = torch.zeros(nmax, self.NY, device=self.psi.device, dtype=torch.float32)

    def launches(self, x, n):
        out = super().launches(x, n)
        k = [i for i, (name, _) in enumerate(out) if name == "vote"]
        if len(k) != 1:
            raise RuntimeError("pc_fused.Batched.launches has no single vote launch")
        pdl, ns = self.pdl, self.stages
        lp = {"launch_pdl": True} if pdl else {}                               # as in pc_fused (not the first kernel)
        out[k[0]] = ("vote", lambda: _k_vote_x[(self.NP, n)](
            self.s, self.psi, self.beta, self.m1, self.i1, self.m2, self.sc, self.QP, K=self.K, BY=BY, BQV=BQV,
            PDL=pdl, NP=self.NP, NY=self.NY, num_warps=4, num_stages=ns, **lp))
        return out


@torch.no_grad()
def run_with_programs(eng, det, scores=False):
    """det (N, D) uint8 on the GPU -> CPU tensors: bits (N, K), margin (N,), the vote's per-program m1, i1, m2 (N, NP)
    and, for a Scored engine with scores=True, the class scores (N, 2^K); eng.nmax shots per call, as Batched.run."""
    n = len(det)
    out = {"bits": torch.empty(n, eng.K, dtype=torch.uint8), "margin": torch.empty(n, dtype=torch.float32),
           "m1": torch.empty(n, eng.NP, dtype=torch.float32), "i1": torch.empty(n, eng.NP, dtype=torch.int32),
           "m2": torch.empty(n, eng.NP, dtype=torch.float32)}
    if scores:
        out["sc"] = torch.empty(n, eng.NY, dtype=torch.float32)
    for i in range(0, n, eng.nmax):
        d = det[i:i + eng.nmax].contiguous()
        k = len(d)
        b, m = eng(d)
        out["bits"][i:i + k] = b.cpu()
        out["margin"][i:i + k] = m.cpu()
        out["m1"][i:i + k] = eng.m1[:k].cpu()
        out["i1"][i:i + k] = eng.i1[:k].cpu()
        out["m2"][i:i + k] = eng.m2[:k].cpu()
        if scores:
            out["sc"][i:i + k] = eng.sc[:k].cpu()
    return out


def signature(nmax):
    sig = pc_fused.signature(nmax)
    import hashlib
    import os
    sig["src"]["arb_fused.py"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    sig["triton"] = triton.__version__
    return sig
