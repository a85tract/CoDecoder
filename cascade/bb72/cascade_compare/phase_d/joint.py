"""The joint decision of CoDecoder (S55'-deep) and Cascade (C14') on BB72, in the timed pipelines and the reads (ARB's
rule A3 with its arithmetic fixed).

Rule. On a shot that S routes to Cascade (S's margin below the lock's tau1) and on which the two decisions differ, the
answer is
    argmax over the 4,096 classes y of  t(y) = s_N(y) + w (y . z)
with s_N the network's class scores (the values of bb.fused's vote, whose top two give S's decision and margin), z
Cascade's 12 logits on the path in use (its bf16 output in fp32) and bit j of the class index = observable j. Every other
shot is answered as in the composition (S's decision, or Cascade's on a routed shot where the two agree). w is a
parameter of the process (ARB's fit: 10^0.8 = 6.31 at p = 0.2 %, 10^0.575 = 3.76 at p = 0.3 %).

Arithmetic, fixed: fp32 on the GPU.
  s_N(y)  as the read's vote computes it: _k_jvote is pc_fused._k_vote (the read's and the batched paths' vote) line for
          line, the same tiles (BY = 128 classes per program, BQV = 128), loads, loop, sums and beta terms, launched with
          the engine's num_warps 4 and num_stages; it runs only on rows whose decisions differ and stores the scores.
  e(y)    = the sum of z_j over the set bits j of y, in increasing j, from +0.0, each step one fp32 add.
  m(y)    = w32 e(y), one IEEE fp32 multiply rounded to nearest (PTX mul.rn.f32 in inline assembly, so that no compiler
          fuses it with the next add into an FMA); w32 = float32(w), rounded to nearest.
  t(y)    = s_N(y) + m(y), one fp32 add.
  answer  the lowest class index among the maxima of t (ties go to the lowest index).
joint_np computes the same answer in numpy, bit for bit (numpy's float32 elementwise operations are IEEE operations
rounded to nearest, never fused); a3_f64 is ARB's float64 rule (arb_analysis.a3_answers) for comparison.

Where it runs. JointB is Cascade's computation (make_fwd_z: pc_speed.make_fwd's, with the logits as a third output)
followed by the joint step, captured in one CUDA graph behind pc_speed.Graph's interface: its `bits` are the
composition's decisions with the joint step, `bb` Cascade's own decisions, `conf` Cascade's confidence, `z` its logits.
Single mode: one shot; the joint step reads S's own static buffers (its pool output s and its decision bits) in place,
so a routed shot costs one replay, as before. Batched mode: the rows S routes queue on the GPU with their s rows and
S's decision bits (gathered from S's engine right after S's replay, before its buffers are reused), and each replay
of Cascade's batch of 4,096 queued rows also decides the joint step of those rows. On a row whose decisions agree both
kernels only compare the two decisions and stop."""
import hashlib
import os

import numpy as np
import torch
import triton
import triton.language as tl

from bb.fused import BQV, BY
from joint_ref import K, NY, YB, a3_f64, bits_of_cls, cls_of, joint_np, w32_of  # noqa: F401  (the CPU references)

D = 432


@triton.jit
def _mul_rn(a, b):
    """a * b in fp32: one IEEE multiply rounded to nearest (inline PTX, never contracted into an FMA)."""
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _k_jvote(s_ptr, psi_ptr, beta_ptr, sb_ptr, bb_ptr, sc_ptr, QP, K: tl.constexpr, KP: tl.constexpr,
             BY: tl.constexpr, BQV: tl.constexpr, NY: tl.constexpr):
    """Program (i, b): pc_fused._k_vote's program i on row b, only if row b's two decisions differ; stores the scores
    of its BY classes into sc (the vote's lines below are pc_fused's, the top-2 replaced by the store)."""
    b = tl.program_id(1).to(tl.int64)                                           # the row
    kk = tl.arange(0, KP)
    km = kk < K
    nd = tl.sum((tl.load(sb_ptr + b * K + kk, mask=km, other=0) !=
                 tl.load(bb_ptr + b * K + kk, mask=km, other=0)).to(tl.int32), 0)
    if nd > 0:
        s_ptr += b * QP
        sc_ptr += b * NY
        pid = tl.program_id(0)
        yl = tl.arange(0, BY)
        ys = pid * BY + yl
        qv = tl.arange(0, BQV)
        acc = tl.zeros((BY,), dtype=tl.float32)
        for q0 in tl.range(0, QP, BQV):
            psi = tl.load(psi_ptr + ys[:, None] * QP + (q0 + qv)[None, :]).to(tl.float32)
            acc += tl.sum(psi * tl.load(s_ptr + q0 + qv)[None, :], 1)
        for j in tl.static_range(K):
            acc += (((ys >> j) & 1).to(tl.float32) - 0.5) * tl.load(beta_ptr + j)
        tl.store(sc_ptr + ys, acc)


@triton.jit
def _k_jtop(sc_ptr, z_ptr, sb_ptr, bb_ptr, out_ptr, fl_ptr, w, K: tl.constexpr, KP: tl.constexpr, NY: tl.constexpr):
    """Program b: row b's answer. Decisions equal: Cascade's bits, flag 0. Different: the joint answer (the module's
    arithmetic), flag 1."""
    b = tl.program_id(0).to(tl.int64)
    kk = tl.arange(0, KP)
    km = kk < K
    sbit = tl.load(sb_ptr + b * K + kk, mask=km, other=0)
    bbit = tl.load(bb_ptr + b * K + kk, mask=km, other=0)
    nd = tl.sum((sbit != bbit).to(tl.int32), 0)
    if nd > 0:
        ys = tl.arange(0, NY)
        e = tl.zeros((NY,), dtype=tl.float32)
        for j in tl.static_range(K):
            e = e + tl.where(((ys >> j) & 1) == 1, tl.load(z_ptr + b * K + j), 0.0)
        t = tl.load(sc_ptr + b * NY + ys) + _mul_rn(e, tl.zeros((NY,), dtype=tl.float32) + w)
        cls = tl.min(tl.where(t == tl.max(t, 0), ys, NY), 0)
        tl.store(out_ptr + b * K + kk, ((cls >> kk) & 1).to(tl.uint8), mask=km)
        tl.store(fl_ptr + b, 1)
    else:
        tl.store(out_ptr + b * K + kk, bbit, mask=km)
        tl.store(fl_ptr + b, 0)


class Step:
    """The joint step on n rows (n fixed: a CUDA graph's batch) over buffers that stay in place:
    s (n, QP) fp32 S's pool output, sb (n, K) uint8 S's decisions, bb (n, K) uint8 Cascade's decisions, z (n, K) fp32
    Cascade's logits -> out (n, K) uint8 the answers, flag (n,) int8 (1 where the decisions differ), sc (n, 4,096) the
    class scores of those rows. eng is S's fused engine (bb.fused.Engine or pc_fused.Batched / mfused.Engine): its psi,
    beta, QP and num_stages."""

    def __init__(self, eng, n, w, s, sb, bb, z):
        dev = eng.psi.device
        if eng.K != K or eng.NP * BY != NY or eng.psi.dtype != torch.float16:
            raise ValueError("the joint step needs bb.fused's vote: K = 12, 4,096 classes, an fp16 psi table")
        for name, t, shape, dt in (("s", s, (n, eng.QP), torch.float32), ("sb", sb, (n, K), torch.uint8),
                                   ("bb", bb, (n, K), torch.uint8), ("z", z, (n, K), torch.float32)):
            if tuple(t.shape) != shape or t.dtype != dt or not t.is_contiguous():
                raise ValueError(f"joint step: {name} is {tuple(t.shape)} {t.dtype}, needs contiguous {shape} {dt}")
        self.eng, self.n, self.s, self.sb, self.bb, self.z = eng, n, s, sb, bb, z
        self.w, self.w32_hex = w32_of(w)
        self.sc = torch.zeros(n, NY, device=dev, dtype=torch.float32)
        self.out = torch.zeros(n, K, device=dev, dtype=torch.uint8)
        self.flag = torch.zeros(n, device=dev, dtype=torch.int8)

    def __call__(self):
        e = self.eng
        _k_jvote[(e.NP, self.n)](self.s, e.psi, e.beta, self.sb, self.bb, self.sc, e.QP, K=K, KP=16, BY=BY, BQV=BQV,
                                 NY=NY, num_warps=4, num_stages=e.stages)
        _k_jtop[(self.n,)](self.sc, self.z, self.sb, self.bb, self.out, self.flag, self.w, K=K, KP=16, NY=NY,
                           num_warps=8)
        return self.out


def make_fwd_z(pr):
    """pc_speed.make_fwd(pr, "minabs") for a Cascade network with the logits kept: det (B, 432) uint8 ->
    (bits (B, 12) uint8, conf (B,) fp32, z (B, 12) fp32)."""
    if pr.kind != "cascade":
        raise ValueError("make_fwd_z: a Cascade network")

    def fwd(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = pr.model(x).float()
            return (z > 0).to(torch.uint8), z.abs().min(1).values, z
    return fwd


def fused_engine(g):
    """S's fused engine behind a ConvLines path object (pc_speed.FusedGraph: bb.engine.Deployed's bb.fused.Engine;
    pc_speed.Graph of an engine: its .engine)."""
    if hasattr(g, "e") and hasattr(g.e, "fused"):
        return g.e.fused
    if getattr(g, "engine", None) is not None:
        return g.engine
    raise ValueError(f"{type(g).__name__}: no fused engine behind this path")


class JointB:
    """Cascade's computation fn (make_fwd_z, compiled or not) at batch B followed by the joint step, one CUDA graph,
    behind pc_speed.Graph's interface (x, bits, conf, B, replay, run). Single mode (B = 1): s_src and sb_src are S's own
    buffers (its engine's s row and decision bits), read in place. Batched mode: s_src = sb_src = None, and the caller
    fills self.s and self.sb (the queued rows' s and S's decisions) before each replay. Outputs: bits (the
    composition's decisions with the joint step), bb (Cascade's decisions), conf (Cascade's confidence, min |z|), z,
    flag, sc."""

    def __init__(self, fn, B, eng, w, s_src=None, sb_src=None, dev=None):
        dev = dev or torch.device("cuda")
        self.B, self.fn = B, fn
        self.x = torch.zeros(B, D, dtype=torch.uint8, device=dev)
        self.s = s_src if s_src is not None else torch.zeros(B, eng.QP, dtype=torch.float32, device=dev)
        self.sb = sb_src if sb_src is not None else torch.zeros(B, K, dtype=torch.uint8, device=dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(st):
            for _ in range(3):
                bb, conf, z = fn(self.x)
                Step(eng, B, w, self.s, self.sb, bb, z)()
        torch.cuda.current_stream().wait_stream(st)
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g):
            self.bb, self.conf, self.z = fn(self.x)
            self.step = Step(eng, B, w, self.s, self.sb, self.bb, self.z)
            self.bits = self.step()
        self.flag, self.sc, self.w, self.w32_hex = self.step.flag, self.step.sc, self.step.w, self.step.w32_hex

    def replay(self):
        self.g.replay()


class GraphZ:
    """pc_speed.Graph for a computation with the logits kept (make_fwd_z): fn captured in a CUDA graph at batch B on a
    static input (three warm-up calls on a side stream first); run(det) -> (bits, conf, z) on the device, B rows per
    replay, the last batch zero-padded, as pc_speed.Graph.run. The reads' compiled@4096 with the logits."""

    def __init__(self, fn, B, dev=None):
        dev = dev or torch.device("cuda")
        self.B = B
        self.x = torch.zeros(B, D, dtype=torch.uint8, device=dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(st):
            for _ in range(3):
                fn(self.x)
        torch.cuda.current_stream().wait_stream(st)
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g):
            self.bits, self.conf, self.z = fn(self.x)

    def replay(self):
        self.g.replay()

    def run(self, det):
        n, ob, oc, oz = len(det), [], [], []
        for i in range(0, n, self.B):
            d = det[i:i + self.B]
            self.x.zero_()
            self.x[:len(d)].copy_(d)
            self.g.replay()
            ob.append(self.bits[:len(d)].clone())
            oc.append(self.conf[:len(d)].clone())
            oz.append(self.z[:len(d)].clone())
        return torch.cat(ob), torch.cat(oc), torch.cat(oz)


def compiled_z(pr):
    """Cascade's compiled computation with the logits kept (torch.compile of make_fwd_z, default mode, dynamic=False):
    the computation of JointB on the compiled path, before its graph capture."""
    return torch.compile(make_fwd_z(pr), dynamic=False)


def build_b(pr, path, B, eng, w, s_src=None, sb_src=None):
    """Cascade on its path ("compiled": torch.compile, default mode, dynamic=False, as pc_speed.build; "graph": the
    module) at batch B with the joint step -> JointB."""
    fn = make_fwd_z(pr)
    if path == "compiled":
        fn = torch.compile(fn, dynamic=False)
    elif path != "graph":
        raise ValueError(f"the joint step runs behind Cascade's graph or compiled path, not {path!r}")
    return JointB(fn, B, eng, w, s_src, sb_src)


def joint_gpu(eng, s, sb, bb, z, w, chunk=4096):
    """The joint step on rows given as tensors on the GPU (eager launches, chunk rows at a time) -> answers (n, K) uint8,
    flag (n,), scores (n, 4,096) on the CPU, the scores only on rows that differ (zero elsewhere)."""
    n = len(s)
    out = torch.empty(n, K, dtype=torch.uint8)
    flag = torch.empty(n, dtype=torch.int8)
    sc = torch.zeros(n, NY, dtype=torch.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        st = Step(eng, hi - lo, w, s[lo:hi].contiguous(), sb[lo:hi].contiguous(), bb[lo:hi].contiguous(),
                  z[lo:hi].contiguous())
        st()
        out[lo:hi], flag[lo:hi], sc[lo:hi] = st.out.cpu(), st.flag.cpu(), st.sc.cpu()
    return out, flag, sc


def _dev_f32(x, dev):
    """numpy array or tensor -> contiguous fp32 tensor on dev."""
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=torch.float32).contiguous()
    return torch.as_tensor(np.ascontiguousarray(x, np.float32), device=dev)


def joint_scores_gpu(sc, z, w, chunk=16384):
    """_k_jtop on given scores (n, 4,096) and logits (n, K) (numpy or tensors), every row taken as differing -> classes
    (n,) int64 (numpy). The GPU's arithmetic on rows whose scores are known (the check on ARB's rows)."""
    dev = torch.device("cuda")
    n = len(z)
    out = np.empty(n, np.int64)
    pw = torch.as_tensor(1 << np.arange(K), device=dev)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        m = hi - lo
        s_ = _dev_f32(sc[lo:hi], dev)
        z_ = _dev_f32(z[lo:hi], dev)
        sb = torch.zeros(m, K, dtype=torch.uint8, device=dev)
        bb = torch.ones(m, K, dtype=torch.uint8, device=dev)         # every row "differs"
        o = torch.empty(m, K, dtype=torch.uint8, device=dev)
        fl = torch.empty(m, dtype=torch.int8, device=dev)
        w32, _ = w32_of(w)
        _k_jtop[(m,)](s_, z_, sb, bb, o, fl, w32, K=K, KP=16, NY=NY, num_warps=8)
        out[lo:hi] = (o.long() * pw).sum(1).cpu().numpy()
    return out


def scores_gpu(eng, s, chunk=4096):
    """_k_jvote's class scores of every row of s (n, QP) (numpy or tensor), every row taken as differing -> (n, 4,096)
    float32 numpy. For the check of the vote's scores against the read's (Scored's)."""
    dev = eng.psi.device
    n = len(s)
    out = np.empty((n, NY), np.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        m = hi - lo
        s_ = _dev_f32(s[lo:hi], dev)
        sb = torch.zeros(m, K, dtype=torch.uint8, device=dev)
        bb = torch.ones(m, K, dtype=torch.uint8, device=dev)
        sc = torch.zeros(m, NY, dtype=torch.float32, device=dev)
        _k_jvote[(eng.NP, m)](s_, eng.psi, eng.beta, sb, bb, sc, eng.QP, K=K, KP=16, BY=BY, BQV=BQV, NY=NY,
                              num_warps=4, num_stages=eng.stages)
        out[lo:hi] = sc.cpu().numpy()
    return out


def source():
    """sha256 (16 hex digits) of this file and joint_ref.py: the kernels, the arithmetic and its reference."""
    h = hashlib.sha256(open(os.path.abspath(__file__), "rb").read())
    h.update(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_ref.py"), "rb").read())
    return h.hexdigest()[:16]


def record(w):
    w32, hx = w32_of(w)
    return {"w": float(w), "w32": w32, "w32_hex": hx, "joint_py": source(), "triton": triton.__version__,
            "rule": "argmax_y s_N(y) + w (y . z) on routed shots whose decisions differ; fp32, ties to the lowest class",
            "impl": "JointB: Cascade's computation + _k_jvote + _k_jtop in one CUDA graph"}
