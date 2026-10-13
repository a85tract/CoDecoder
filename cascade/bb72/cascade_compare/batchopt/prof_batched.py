"""The profile: where the batched decoding of the BB72 comparison spends its time, and what bounds it (batchopt/PROFILE.md).

The first step of the optimization stage; phase C's formal run is not touched. The networks are loaded with
bb.cascade_eval.Predictor and built on their frozen batched paths exactly as pc_speed.py builds them (S55:
pc_fused.Batched in a CUDA graph, fused@4096; C14: torch.compile, then a CUDA graph, compiled@4096), on whatever card
runs the process: no card is refused, and every output records the card, its driver, torch, Triton and the SM clock
samples. A profile, not a phase-C measurement: one process per step, no repetitions, no screen.

  shots     the inputs: kind-12 shots of the uniform setting at --p (0.3 %), chunks 0-9 (the screen's) -> shots.npz
  env       the card and its measured ceilings: GEMM per precision (cuBLAS, 8192^3), DRAM and L2-resident copies,
            pinned PCIe copies, the replay of a one-kernel CUDA graph -> env.json
  work      the work per shot from the model definitions: S55's fused engine per kernel (tensor-core and other FLOPs,
            the bytes its programs request, the compulsory activation bytes), its useful MACs (bb.net.macs) and the
            PyTorch module's FLOPs (torch.utils.flop_counter); C14's FLOPs (flop_counter on the gather-matmul path, the
            products the Triton kernel computes), the dtypes of its matrix products and a bytes model -> work.json
  s55       S55 alone on its fused path at each --batches: per shot (input resident and host to host, as C3), per replay
            (CUDA events), per kernel (torch.profiler: graph replays with PDL as frozen, eager launches without PDL),
            bitwise against the reads' engine, the kernels' tensor-core instructions (PTX) -> s55.json
  c14       C14 alone on its compiled path at each --batches: the same by kernel and category, the reference outputs of
            the frozen path, the Triton convolution's autotuned config and PTX -> c14.json
  pipe      S55 -> C14 batched as pc_speed.pipeline runs it (Relay-BP aside) at each --q1 on --n shots (the unique shots
            cycled): S, copies, routing, queue, B and the GPU's idle gaps from CUDA events and host clocks; one
            torch.profiler trace; then S and B replayed on two streams (overlap) -> pipe.json
  ncuwork   one NVTX range "prof" around one call of S55 or C14 at --batch: the workload Nsight Compute profiles
  ncuparse  the key metrics of an ncu raw CSV export -> JSON
  report    the roofline numbers and the breakdown -> summary.json, summary.md
  selftest  CPU checks of the parsing and summary code on synthetic inputs"""
import argparse
import csv
import gzip
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase_c"))
import pc_common as pc  # noqa: E402

SMI = ("name", "driver_version", "mig.mode.current", "memory.total", "clocks.max.sm", "clocks.max.mem",
       "clocks.applications.graphics", "power.limit", "power.max_limit", "pcie.link.gen.max", "pcie.link.width.max",
       "pcie.link.gen.current", "persistence_mode", "compute_mode", "ecc.mode.current")
# Vendor figures: dense tensor-core TFLOP/s with FP32 accumulation, FP32 (non-tensor) TFLOP/s, DRAM GB/s. The RTX PRO
# 6000's tensor figures are its datasheet's 4 PFLOPS (FP4, sparse) divided down (FP8 = FP4 / 2, BF16 = FP8 / 2, dense =
# sparse / 2); 188 SMs x 2.617 GHz x 1,024 FP16 FLOP per SM and clock gives the same 504.
NOMINAL = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": {"fp16": 504, "bf16": 504, "fp8": 1008, "tf32": 252,
                                                            "fp32": 126, "dram_gbs": 1792},
           "NVIDIA H100 80GB HBM3": {"fp16": 989, "bf16": 989, "fp8": 1979, "tf32": 495, "fp32": 67, "dram_gbs": 3350}}
MMA = re.compile(r"\b(mma\.sync\.aligned\.m\d+n\d+k\d+[\w.]*|wgmma\.mma_async[\w.]*|tcgen05\.mma[\w.]*)")
Q1S = (0.0, 0.01, 0.02, 0.05, 0.10, 0.40)
RATIOS = ((100, 1), (20, 1), (10, 1), (5, 2))     # S batches per B batch at q1 = 1, 5, 10 and 40 %


# ---------------------------------------------------------------------------------------------------------- the card
def smi_index():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def smi(fields):
    out = {}
    for f in fields:                                    # one query per field: an unsupported field fails alone
        r = subprocess.run(["nvidia-smi", "-i", smi_index(), f"--query-gpu={f}", "--format=csv,noheader"],
                           capture_output=True, text=True)
        out[f] = r.stdout.strip() if r.returncode == 0 else None
    return out


def card(dev):
    """The card (recorded, never refused): its driver, clocks and limits (nvidia-smi), torch, CUDA, Triton, the host."""
    import torch
    import triton
    p = torch.cuda.get_device_properties(dev)
    s = smi(SMI)
    cpu = subprocess.run("lscpu | sed -n 's/^Model name: *//p'", shell=True, capture_output=True, text=True).stdout
    return {"gpu": torch.cuda.get_device_name(dev), "capability": list(torch.cuda.get_device_capability(dev)),
            "sms": p.multi_processor_count, "mem_gib": round(p.total_memory / 2 ** 30, 1),
            "l2_mib": round(getattr(p, "L2_cache_size", 0) / 2 ** 20, 1),
            "smem_per_block_optin_kib": getattr(p, "shared_memory_per_block_optin", 0) // 1024,
            "driver": s.get("driver_version"), "smi": s, "torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": triton.__version__, "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
            "host": platform.node(), "cpu": cpu.strip(), "cpus": len(os.sched_getaffinity(0)),
            "job": os.environ.get("SLURM_JOB_ID", ""), "tree": os.environ.get("PC_TREE", "")}


def setup():
    """pc_speed.setup's backend settings (TF32 off, cuDNN deterministic, no benchmarking), any card admitted and
    recorded."""
    import pc_speed
    pc.gpu_env = card
    return pc_speed.setup()


class Watch:
    """nvidia-smi samples of this GPU every 100 ms while a measurement runs (bb.timing.gpu_watch's sampling, more
    fields); stop() -> their summary."""
    Q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks.max.sm"

    def __init__(self):
        self.p = subprocess.Popen(["nvidia-smi", "-i", smi_index(), f"--query-gpu={self.Q}",
                                   "--format=csv,noheader,nounits", "-lms", "100"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def stop(self):
        self.p.terminate()
        rows = [[float(x) if re.fullmatch(r"[0-9.]+", x.strip()) else np.nan for x in ln.split(",")]
                for ln in self.p.communicate()[0].splitlines() if ln.count(",") == 5]
        if not rows:
            return {"samples": 0}
        a = np.array(rows)

        def q(c):
            v = a[:, c][~np.isnan(a[:, c])]
            return [float(v.min()), float(np.median(v)), float(v.max())] if len(v) else None
        mx = float(np.nanmax(a[:, 5])) if not np.isnan(a[:, 5]).all() else None
        return {"samples": len(a), "sm_mhz": q(0), "mem_mhz": q(1), "power_w": q(2), "temp_c": q(3), "util": q(4),
                "sm_max_mhz": mx, "sm_min_ratio": float(np.nanmin(a[:, 0]) / mx) if mx else None}


# ------------------------------------------------------------------------------------------------------------ helpers
def dump(out, name, obj):
    f = os.path.join(out, name)
    with open(f + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(f + ".tmp", f)


def load_shots(out):
    f = os.path.join(out, "shots.npz")
    if not os.path.exists(f):
        raise SystemExit(f"{f}: run the shots step first")
    z = np.load(f)
    return z["det"], z["obs"]


def load(tag, dev):
    from bb.cascade_eval import Predictor
    return Predictor(pc.NETS[tag][0], dev, "ema", "bf16")          # as pc_speed.nets and pc_speed.pipeline load it


def ev_s(fn, reps, warm=3):
    """Seconds per call of fn, from CUDA events around reps back-to-back calls."""
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / 1e3 / reps


def replay_us(gr, target_s=0.5, reps=3):
    """GPU time of one replay (us): reps x calls back-to-back replays timed with CUDA events, calls ~ target_s."""
    import torch
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    gr.replay()
    torch.cuda.synchronize()
    calls = max(3, int(target_s / max(time.perf_counter() - t, 1e-6)))
    v = [1e6 * ev_s(gr.replay, calls, warm=0) for _ in range(reps)]
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "calls": calls, "reps": reps}


def iters_for(us):
    return max(2, min(50, int(2e5 / max(us, 1.0))))


def union_us(iv):
    tot, s0, e0 = 0.0, None, None
    for s, e in sorted(iv):
        if e0 is None or s > e0:
            tot += 0.0 if e0 is None else e0 - s0
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    return tot + (0.0 if e0 is None else e0 - s0)


# ------------------------------------------------------------------------------------------------ kernels and traces
def kprof(fn, iters, path):
    """torch.profiler (CPU and CUDA activity) over iters calls of fn; the Chrome trace goes to path.json.gz -> its GPU
    events (kernels, copies, memsets)."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(path + ".json")
    with open(path + ".json", "rb") as fi:
        raw = fi.read()
    with gzip.open(path + ".json.gz", "wb") as fo:
        fo.write(raw)
    os.remove(path + ".json")
    tr = json.loads(raw)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    return [x for x in evs if x.get("ph") == "X" and x.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]


def kcat(n, cat=None):
    s = n.lower()
    if cat in ("gpu_memcpy", "gpu_memset") or s.startswith(("memcpy", "memset")):
        return "copy/memset"
    if s.startswith("_k_"):
        return "fused engine"
    if "_gmm" in s:
        return "bipartite conv (Triton _gmm)"
    if s.startswith("triton_"):
        return {"poi": "inductor pointwise", "red": "inductor reduction", "per": "inductor persistent reduction",
                "tem": "inductor template"}.get(s.split("_")[1], "inductor other")
    if re.search(r"gemm|gemv|cutlass|xmma|nvjet|cublas|splitk|s16816|tensorop", s):
        return "cuBLAS GEMM"
    if re.search(r"index|gather|scatter|embedding", s):
        return "gather/index (aten)"
    if re.search(r"cat|copy|fill|elementwise|vectorized|reduce|norm", s):
        return "aten elementwise/reduce"
    return "other"


def tc_name(n):
    """What a kernel's name says about tensor cores and precision (cuBLAS and CUTLASS names carry both; Triton's and
    Inductor's do not: their PTX does, see ptx_summary)."""
    s = n.lower()
    tc = bool(re.search(r"tensorop|xmma|hmma|gmma|s16816|s1688|16x8x16|nvjet|tensor\d+x\d+x\d+|tcgen05|umma", s))
    prec = None
    for p_, pat in (("fp8", r"e4m3|e5m2"), ("bf16", r"bf16"), ("tf32", r"tf32"), ("fp16", r"f16|h16816|h1688|fp16"),
                    ("fp32", r"sgemm|f32f32_f32")):
        if re.search(pat, s):
            prec = p_
            break
    return {"tensor_cores": tc, "precision": prec}


def ksum(evs, iters, shots):
    """GPU events of a trace -> (one row per name, by time: calls and us per iteration, us per shot, share, category,
    the name's tensor-core reading, Kineto's launch fields; totals: the sum of durations, the busy time (their union)
    and the span, per iteration)."""
    if not evs:
        return [], {}
    by = {}
    for x in evs:
        d = by.setdefault(x["name"], {"n": 0, "us": 0.0, "args": x.get("args", {}), "cat": x.get("cat")})
        d["n"] += 1
        d["us"] += float(x.get("dur", 0.0))
    tot = sum(d["us"] for d in by.values())
    rows = []
    for n, d in sorted(by.items(), key=lambda kv: -kv[1]["us"]):
        g = d["args"]
        rows.append({"name": n[:200], "cat": kcat(n, d["cat"]), "tc": tc_name(n), "calls": d["n"] / iters,
                     "us_iter": d["us"] / iters, "us_shot": d["us"] / iters / shots,
                     "share": d["us"] / tot if tot else 0.0,
                     "grid": g.get("grid"), "block": g.get("block"), "regs": g.get("registers per thread"),
                     "smem": g.get("shared memory"), "occ_est": g.get("est. achieved occupancy %"),
                     "blocks_per_sm": g.get("blocks per SM"), "warps_per_sm": g.get("warps per SM")})
    iv = [(float(x["ts"]), float(x["ts"]) + float(x.get("dur", 0.0))) for x in evs]
    return rows, {"sum_us_iter": tot / iters, "busy_us_iter": union_us(iv) / iters,
                  "span_us_iter": (max(e for _, e in iv) - min(s for s, _ in iv)) / iters}


def categories(rows):
    out = {}
    for r in rows:
        c = out.setdefault(r["cat"], {"us_iter": 0.0, "us_shot": 0.0, "share": 0.0, "kernels": 0})
        c["us_iter"] += r["us_iter"]
        c["us_shot"] += r["us_shot"]
        c["share"] += r["share"]
        c["kernels"] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["us_iter"]))


def ksum_fused(evs, iters, shots, names):
    """S55's kernels labelled by their place in pc_fused.Batched.launches (stem, block0 .., pool, vote, top2): in start
    order the trace's kernels repeat that sequence once per call (PDL lets a kernel start before the last one ends,
    never before it has started)."""
    ks = sorted((x for x in evs if x.get("cat") == "kernel"), key=lambda x: float(x["ts"]))
    want = {"stem": "_k_stem", "pool": "_k_pool", "vote": "_k_vote", "top2": "_k_top2"}
    ok = len(ks) == iters * len(names) and all(
        x["name"].startswith(want.get(names[i % len(names)], "_k_block")) for i, x in enumerate(ks))
    if not ok:
        rows, tot = ksum(evs, iters, shots)
        return {"labelled": False, "kernel_events": len(ks), "rows": rows, **tot}
    lab = [{**x, "name": names[i % len(names)]} for i, x in enumerate(ks)]
    rows, tot = ksum(lab + [x for x in evs if x.get("cat") != "kernel"], iters, shots)
    order = {n: i for i, n in enumerate(names)}
    rows.sort(key=lambda r: order.get(r["name"], len(order)))
    return {"labelled": True, "kernel_events": len(ks), "rows": rows, **tot}


def ptx_summary(name, k):
    """A compiled Triton kernel: launch options, registers, spills and the tensor-core instructions of its PTX."""
    asm = getattr(k, "asm", None) or {}
    ptx = asm.get("ptx", "") if hasattr(asm, "get") else ""
    ins = {}
    for x in MMA.findall(ptx if isinstance(ptx, str) else ""):
        ins[x] = ins.get(x, 0) + 1
    md = getattr(k, "metadata", None)

    def m(f):
        return md.get(f) if isinstance(md, dict) else getattr(md, f, None)
    return {"name": name, "ptx": bool(ptx), "mma": ins, "tensor_cores": bool(ins), "num_warps": m("num_warps"),
            "num_stages": m("num_stages"), "shared_bytes": m("shared"), "regs": getattr(k, "n_regs", None),
            "spills": getattr(k, "n_spills", None)}


def compiled_kernels(jit):
    """The kernels Triton compiled for a JITFunction (its device_caches' kernel dicts; the layout varies by version)."""
    ks = []
    for v in getattr(jit, "device_caches", {}).values():
        for x in (v if isinstance(v, tuple) else (v,)):
            if isinstance(x, dict):
                ks += [k for k in x.values() if hasattr(k, "asm")]
    return ks


# ------------------------------------------------------------------------------------------------------------- work
def fused_work(e):
    """Per shot and kernel of pc_fused.Batched (bb.fused's launch geometry; padding included, as executed): programs,
    tensor-core FLOPs (every tl.dot, 2 per MAC), other FLOPs (the vote, the MLP's last layer; LayerNorm, SiLU and gates
    approximately), the bytes its programs request (weights and tables re-read by every program, gathers, activations:
    L2 traffic at least) and the compulsory activation bytes (each per-shot buffer written once and read once: what DRAM
    sees when the batch's buffers exceed L2)."""
    from bb.fused import BM, BY
    C, CB, S, H, K, D, FF = e.C, e.CB, e.S, e.H, e.K, e.D, e.pool["FF"]
    NT, SP, QP, NQT, NP, KS, NNZ, BQ = e.NT, e.SP, e.QP, e.NQT, e.NP, e.KS, e.NNZ, e.bq
    SP1, TAPS = NT * BM, e.NIT * e.tpi
    f32, f16, i32, ln = 4, 2, 4, 2 * C * 4
    out = {}
    prog = (BM * C * f32 + BM * KS * (i32 + 1) + KS * C * f16 + ln + C * CB * f16 + CB * f32 + BM * C * f32
            + BM * CB * f16)
    out["stem"] = {"programs": NT, "mma": 2 * SP1 * KS * C + 2 * SP1 * C * CB, "simt": SP1 * (10 * C + 6 * CB),
                   "l2": NT * prog, "dram": D + SP1 * C * f32 + S * CB * f16}
    L = len(e.blk)
    for i in range(L):
        prog = BM * TAPS * (i32 + CB * f16) + TAPS * CB * CB * f16 + CB * f32 + BM * C * f32 + C * f32 + CB * C * f16
        mma, simt = 2 * SP1 * TAPS * CB * CB + 2 * SP1 * CB * C, SP1 * (6 * CB + C)
        if i == L - 1:                                  # the read-out's LayerNorm, gate and values -> U
            prog += ln + 2 * C * C * f16 + 2 * C * f32 + BM * C * f16
            mma, simt = mma + 4 * SP1 * C * C, simt + SP1 * 16 * C
            dram = S * CB * f16 + SP1 * C * f32 + S * C * f16
        else:                                           # the next block's LayerNorm, down-projection, SiLU -> u'
            prog += BM * C * f32 + ln + C * CB * f16 + CB * f32 + BM * CB * f16
            mma, simt = mma + 2 * SP1 * C * CB, simt + SP1 * (10 * C + 6 * CB)
            dram = S * CB * f16 + 2 * SP1 * C * f32 + S * CB * f16
        out[f"block{i}"] = {"programs": NT, "mma": mma, "simt": simt, "l2": NT * prog, "dram": dram}
    prog = (BQ * C * f32 + H * BQ * SP * f16 + SP * C * f16 + C * C * f16 + ln + C * FF * f16 + 2 * FF * f32
            + BQ * NNZ * (i32 + 1) + 3 * BQ * f32)
    out["pool"] = {"programs": NQT, "mma": 2 * QP * SP * C + 2 * QP * C * C + 2 * QP * C * FF,
                   "simt": QP * (10 * C + 7 * FF + 2 * NNZ + 4), "l2": NQT * prog,
                   "dram": SP * C * f16 + D + 2 * QP * f32}
    out["vote"] = {"programs": NP, "mma": 0, "simt": 2 * 2 ** K * QP + 2 * 2 ** K * K,
                   "l2": NP * (BY * QP * f16 + QP * f32 + K * f32 + 12), "dram": QP * f32 + NP * 12}
    out["top2"] = {"programs": 1, "mma": 0, "simt": 4 * NP, "l2": NP * 12 + K + 4, "dram": NP * 12 + K + 4}
    return out


def fused_geometry(e):
    from bb.fused import BM, BS, BY, BQV, KCH
    return {k: int(v) for k, v in dict(C=e.C, CB=e.CB, S=e.S, Q=e.Q, H=e.H, DH=e.DH, K=e.K, D=e.D, NT=e.NT, BM=BM,
                                         SP=e.SP, QP=e.QP, NQT=e.NQT, BQ=e.bq, NP=e.NP, BY=BY, BQV=BQV, BS=BS, KS=e.KS,
                                         KCH=KCH, NIT=e.NIT, TPI=e.tpi, NNZ=e.NNZ, FF=e.pool["FF"],
                                         blocks=len(e.blk)).items()}


def cascade_bytes(N, H, L, k, D):
    """A bytes model of Cascade's compiled forward per shot (bf16 autocast, fp32 residual stream), drawn from the fusion
    barriers torch.compile cannot cross: the cuBLAS projections, the opaque Triton convolution and the concatenation of
    its outputs. Between them Inductor fuses the pointwise work into one kernel each (the residual update with the next
    block's BatchNorm, SiLU and bf16 cast; BatchNorm and SiLU after each projection and after the convolution). Each
    tensor crosses DRAM once per kernel that writes or reads it (at batch 4,096 every activation exceeds L2); the
    convolution's 12-fold gather is served from L2 after one read; weights are left out (amortized over the batch)."""
    Hb, f32, bf = H // 4, 4, 2
    h, a, s = N * H * f32, N * H * bf, N * Hb * bf
    blk = {"residual update + BN1 + SiLU + cast": 2 * h + 2 * a, "down projection (cuBLAS)": a + s, "BN2 + SiLU": 2 * s,
           "conv check -> data (2 Triton calls)": 3 * s, "cat (data)": 2 * s,
           "conv data -> check (2 Triton calls)": 3 * s,
           "cat (checks)": 2 * s, "BN3 + SiLU": 2 * s, "up projection (cuBLAS)": s + a}
    ends = {"tokens + embedding": D + N * 8 + h, "BN + SiLU + read-out": h + 2 * a + k * 24 * H * bf}
    per = sum(blk.values())
    return {"per_block": blk, "per_block_total": per, "blocks_total": L * per, "ends": ends,
            "total": L * per + sum(ends.values())}


def mm_logger():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class MMLog(TorchDispatchMode):
        """The matrix products a forward calls: op, input shapes and dtypes, counted per distinct call."""
        OPS = ("mm", "addmm", "bmm", "baddbmm", "convolution", "_convolution", "_scaled_mm")

        def __init__(self):
            super().__init__()
            self.seen = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.overloadpacket.__name__ in self.OPS:
                ts = [x for x in args if isinstance(x, torch.Tensor)]
                key = (func.overloadpacket.__name__, tuple(tuple(t.shape) for t in ts),
                       tuple(str(t.dtype).replace("torch.", "") for t in ts))
                self.seen[key] = self.seen.get(key, 0) + 1
            return func(*args, **(kwargs or {}))

        def summary(self):
            return [{"op": k[0], "shapes": k[1], "dtypes": k[2], "calls": v} for k, v in self.seen.items()]
    return MMLog()


def count(m, x, cuda):
    """FLOPs per shot of the module on rows x (torch.utils.flop_counter; bf16 autocast on a GPU, as the timed paths run;
    fp32 on a CPU: the same count) and the matrix products it calls."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    log = mm_logger()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        with FlopCounterMode(display=False) as fc, log:
            m(x)
    n = len(x)
    return ({"total": fc.get_total_flops() / n,
             "by_op": {str(k): v / n for k, v in fc.get_flop_counts().get("Global", {}).items()}}, log.summary())


def work(a):
    import torch
    from bb import cascade_net as cnet
    from bb import fused
    from bb import net as bnet
    cuda = torch.cuda.is_available()
    if not cuda:                                        # bb.fused builds its tables where the model is; only its
        torch.cuda.get_device_capability = lambda *x: (0, 0)    # PDL choice asks the GPU
    import pc_fused
    dev = torch.device("cuda" if cuda else "cpu")
    x = torch.as_tensor(load_shots(a.out)[0][:8], device=dev)
    out = {"step": "work", "device": str(dev), "nets": {}}
    S = load(a.s, dev)
    e = pc_fused.Batched(S.model, 1)
    fw = fused_work(e)
    fl, mm = count(S.model, x, cuda)
    out["nets"][a.s] = {"sha256": S.sig["sha256"], "precision": fused.PRECISION, "geometry": fused_geometry(e),
                        "stem_taps": len(S.model.cfg["stem_taps"]), "blk_taps": len(S.model.cfg["blk_taps"]),
                        "engine": fw, "engine_total": {k: sum(v[k] for v in fw.values()) for k in ("mma", "simt", "l2",
                                                                                                    "dram")},
                        "useful_macs": bnet.macs(S.model), "module_flops": fl, "module_matmuls": mm}
    del S, e
    B = load(a.b, dev)
    m = B.model
    impl = [blk.conv.impl for blk in m.blocks]
    for blk in m.blocks:                                # the gather-matmul path: the Triton kernel's products
        blk.conv.impl = "torch"
    fl, mm = count(m, x, cuda)
    for blk, i in zip(m.blocks, impl):
        blk.conv.impl = i
    out["nets"][a.b] = {"sha256": B.sig["sha256"], "H": m.H, "L": m.L, "N": m.geo.N, "conv_impl": impl[0],
                        "resid_bf16": m.resid_bf16, "flops": fl, "matmuls": mm,
                        "macs_per_shot": cnet.macs_per_shot(m.H, m.L, N=m.geo.N, k=m.k),
                        "bytes_model": cascade_bytes(m.geo.N, m.H, m.L, m.k, m.D)}
    dump(a.out, "work.json", out)
    print(json.dumps({"work": {t: v.get("engine_total", v.get("flops", {}).get("total"))
                               for t, v in out["nets"].items()}}))


# ------------------------------------------------------------------------------------------------------- the steps
def shots(a):
    from bb import cascade_data as cdata
    det, obs = cdata.eval_shots(pc.CODE, a.p, range(a.chunks), kind=a.kind, procs=a.procs, family="uniform")
    np.savez(os.path.join(a.out, "shots.npz"), det=det, obs=obs, p=a.p, kind=a.kind, chunks=a.chunks)
    print(json.dumps({"shots": det.shape, "p": a.p, "kind": a.kind, "chunks": a.chunks}))


def env(a):
    import torch
    dev, cd = setup()
    out = {"step": "env", "env": cd, "nominal": NOMINAL.get(cd["gpu"]), "ceilings": {}}
    c, n, w = out["ceilings"], 8192, Watch()
    for name, dt, tf32 in (("bf16", torch.bfloat16, False), ("fp16", torch.float16, False),
                           ("tf32", torch.float32, True), ("fp32", torch.float32, False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        x, y = torch.randn(n, n, device=dev, dtype=dt), torch.randn(n, n, device=dev, dtype=dt)
        c[f"gemm_{name}_tflops"] = 2 * n ** 3 / ev_s(lambda: x @ y, 10 if name != "fp32" else 4) / 1e12
        del x, y
    torch.backends.cuda.matmul.allow_tf32 = False
    try:                                                # FP8 e4m3 x e4m3 -> bf16, unit scales (cuBLASLt)
        x = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        y = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        one = torch.ones((), device=dev)
        for fast in (False, True):
            c[f"gemm_fp8{'_fastacc' if fast else ''}_tflops"] = 2 * n ** 3 / ev_s(
                lambda: torch._scaled_mm(x, y.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16,
                                         use_fast_accum=fast), 10) / 1e12
        del x, y
    except Exception as ex:
        c["gemm_fp8_error"] = repr(ex)[:300]
    x = torch.empty(2 ** 28, device=dev)                # 1 GiB
    y = torch.empty_like(x)
    c["dram_copy_gbs"] = 2 * x.numel() * 4 / ev_s(lambda: y.copy_(x), 10) / 1e9
    c["dram_read_gbs"] = x.numel() * 4 / ev_s(lambda: x.sum(), 10) / 1e9
    del x, y
    m = max(2 ** 18, int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 2 ** 24) // 16))
    x = torch.empty(m, device=dev)                      # source and destination: half of L2 together
    y = torch.empty_like(x)
    c["l2_copy_gbs"] = 2 * m * 4 / ev_s(lambda: y.copy_(x), 400, warm=20) / 1e9
    h = torch.empty(2 ** 26).pin_memory()
    d = torch.empty(2 ** 26, device=dev)
    c["h2d_gbs"] = h.numel() * 4 / ev_s(lambda: d.copy_(h, non_blocking=True), 5) / 1e9
    c["d2h_gbs"] = h.numel() * 4 / ev_s(lambda: h.copy_(d, non_blocking=True), 5) / 1e9
    z = torch.zeros(1, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        z.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        z.add_(1)
    c["graph_replay_us"] = 1e6 * ev_s(g.replay, 2000, warm=50)
    out["clock"] = w.stop()
    dump(a.out, "env.json", out)
    print(json.dumps({"gpu": cd["gpu"], **{k: round(v, 1) for k, v in c.items() if isinstance(v, float)}}))


def s55(a):
    import torch
    import pc_fused
    import pc_speed
    from bb import fused
    dev, cd = setup()
    pr = load(a.net, dev)
    m = pr.model
    det = load_shots(a.out)[0]
    dg = torch.as_tensor(det, device=dev)
    ref_b, ref_m = pc_fused.Batched(m, pc.FUSED_READ_B).run(dg)          # the reads' engine: the bitwise reference
    np.savez(os.path.join(a.out, f"ref_{a.net}.npz"), bits=ref_b.cpu().numpy(), margin=ref_m.cpu().numpy())
    out = {"step": "s55", "net": a.net, "sha256": pr.sig["sha256"], "env": cd, "precision": fused.PRECISION,
           "rows": len(det), "batches": {}}
    for B in [int(b) for b in a.batches.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        gr = pc_speed.fused_batch(m, B)                                    # the fused path at B, as pc_speed builds it
        e = gr.engine
        row = {"build_s": round(time.time() - t0, 1), "pdl": bool(e.pdl)}
        n = min(len(dg), max(2 * B, 32768))
        bits, mg = gr.run(dg[:n])
        row.update(bitwise=bool(torch.equal(bits, ref_b[:n]) and torch.equal(mg, ref_m[:n])), bitwise_rows=n)
        w = Watch()
        row["amortized_us"] = pc_speed.amortized(gr, dg, secs=a.secs)    # C3's metric: input resident / host to host
        row["replay_us"] = replay_us(gr)
        row["clock"] = w.stop()
        row["gpu_us_shot"] = row["replay_us"]["mean"] / B
        row["mem_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        names = [nm for nm, _ in e.launches(gr.x, B)]
        wk = fused_work(e)
        it = iters_for(row["replay_us"]["mean"])
        row["kernels_graph"] = ksum_fused(kprof(gr.replay, it, os.path.join(a.out, f"trace_s55_b{B}_graph")), it, B,
                                          names)
        e.pdl = False                                   # eager launches without PDL: each kernel's own time
        e(gr.x, B)
        torch.cuda.synchronize()
        row["kernels_eager_nopdl"] = ksum_fused(kprof(lambda: e(gr.x, B), it,
                                                      os.path.join(a.out, f"trace_s55_b{B}_eager")), it, B, names)
        e.pdl = row["pdl"]
        for k in ("kernels_graph", "kernels_eager_nopdl"):
            for r in row[k]["rows"]:
                if r["name"] in wk and r["us_shot"] > 0:
                    t = r["us_shot"] * 1e-6
                    r.update(mma_tflops=wk[r["name"]]["mma"] / t / 1e12, simt_tflops=wk[r["name"]]["simt"] / t / 1e12,
                             l2_gbs=wk[r["name"]]["l2"] / t / 1e9, dram_gbs=wk[r["name"]]["dram"] / t / 1e9)
        if B == pc.VERDICT_B:
            row["ptx"] = [ptx_summary(nm, fn()) for nm, fn in e.launches(gr.x, B)]
            torch.cuda.synchronize()
        out["batches"][str(B)] = row
        dump(a.out, "s55.json", out)
        print(json.dumps({"s55": B, "dev_us": round(row["amortized_us"]["dev"], 3), "h2h_us":
                          round(row["amortized_us"]["h2h"], 3), "bitwise": row["bitwise"],
                          "labelled": row["kernels_eager_nopdl"]["labelled"]}))
        del gr, e
    out["work"] = {"engine": wk, "geometry": fused_geometry(pc_fused.Batched(m, 1))}
    dump(a.out, "s55.json", out)


def c14(a):
    import torch
    import pc_speed
    from bb import cascade_triton as ct
    dev, cd = setup()
    pr = load(a.net, dev)
    det = load_shots(a.out)[0]
    dg = torch.as_tensor(det, device=dev)
    out = {"step": "c14", "net": a.net, "sha256": pr.sig["sha256"], "env": cd, "rows": len(det), "batches": {},
           "conv_impl": pr.model.blocks[0].conv.impl, "precision": pc.PRECISION["compiled"]}
    for B in [int(b) for b in a.batches.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        cfn = torch.compile(pc_speed.make_fwd(pr), dynamic=False)        # = pc_speed.build(pr, "compiled", B)
        gr = pc_speed.Graph(cfn, B, 432, dev)
        row = {"build_s": round(time.time() - t0, 1)}
        if B == pc.VERDICT_B:                           # the frozen path's outputs (the next stage's screens)
            bits, conf = gr.run(dg)
            np.savez(os.path.join(a.out, f"ref_{a.net}.npz"), bits=bits.cpu().numpy(), conf=conf.cpu().numpy())
        w = Watch()
        row["amortized_us"] = pc_speed.amortized(gr, dg, secs=a.secs)
        row["replay_us"] = replay_us(gr, target_s=1.0)
        row["clock"] = w.stop()
        row["gpu_us_shot"] = row["replay_us"]["mean"] / B
        row["mem_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        evs = kprof(gr.replay, 2, os.path.join(a.out, f"trace_c14_b{B}_graph"))
        row["traced"] = "graph"
        if not any(x.get("cat") == "kernel" for x in evs):                # graph kernels not traced: eager calls
            evs = kprof(lambda: cfn(gr.x), 2, os.path.join(a.out, f"trace_c14_b{B}_eager"))
            row["traced"] = "eager compiled calls"
        rows, tot = ksum(evs, 2, B)
        row.update(kernels=rows[:60], kernel_names=len(rows), totals=tot, categories=categories(rows))
        out["batches"][str(B)] = row
        dump(a.out, "c14.json", out)
        print(json.dumps({"c14": B, "build_s": row["build_s"], "dev_us": round(row["amortized_us"]["dev"], 3),
                          "h2h_us": round(row["amortized_us"]["h2h"], 3)}))
        del gr, cfn
    out["gmm"] = {"autotune": {str(k): str(v) for k, v in getattr(ct._gmm, "cache", {}).items()},
                  "kernels": [ptx_summary("_gmm", k) for k in compiled_kernels(getattr(ct._gmm, "fn", None))]}
    dump(a.out, "c14.json", out)


def timeline(iv, span, N):
    """GPU segments [(kind, start us, end us)] in stream order (one stream) and the GPU span -> per shot: the busy time
    of each kind, the idle time, and the idle gap before each kind of segment (the GPU waiting for the host)."""
    busy, gaps, prev = {}, {}, 0.0
    for k, s, e in iv:
        busy[k] = busy.get(k, 0.0) + (e - s)
        gaps.setdefault(k, []).append(max(0.0, s - prev))
        prev = max(prev, e)
    tot = sum(busy.values())
    return {"gpu_us_shot": {k: v / N for k, v in busy.items()}, "gpu_busy_us_shot": tot / N,
            "gpu_span_us_shot": span / N, "idle_us_shot": (span - tot) / N, "tail_idle_us": max(0.0, span - prev),
            "gaps": {k: {"n": len(g), "us_shot": float(np.sum(g)) / N, "mean_us": float(np.mean(g)),
                         "p99_us": float(np.quantile(g, 0.99)), "max_us": float(np.max(g))} for k, g in gaps.items()}}


def pipe_once(gS, gB, H, N, tau1, dev, keep=False):
    """pc_speed.pipeline's batched loop, Relay-BP aside: S over batches of gS.B, the routed rows queued on the GPU, B
    whenever gB.B wait and once at the end (padded); with CUDA events around every GPU segment, host clocks around
    every host step and NVTX ranges (S, queue, B)."""
    import torch
    nvtx, st = torch.cuda.nvtx, torch.cuda.current_stream()
    bS, BQ, K = gS.B, gB.B, pc.K
    hc_s, hb_s = torch.empty(bS).pin_memory(), torch.empty(bS, K, dtype=torch.uint8).pin_memory()
    hc_b, hb_b = torch.empty(BQ).pin_memory(), torch.empty(BQ, K, dtype=torch.uint8).pin_memory()
    pred, route = np.zeros((N, K), np.uint8), np.zeros(N, np.int8)
    segs, host = [], dict.fromkeys(("issue_S", "wait_S", "route", "queue", "issue_B", "wait_B", "post_B"), 0.0)
    nB, q_rows, q_idx = {"full": 0, "padded": 0}, [], []

    def ev():
        x = torch.cuda.Event(enable_timing=True)
        x.record()
        return x

    def run_b(rows, idx):
        t0 = time.perf_counter()
        nvtx.range_push("B")
        e0 = ev()
        gB.x.zero_()
        gB.x[:len(idx)].copy_(rows)
        e1 = ev()
        gB.replay()
        e2 = ev()
        hb_b.copy_(gB.bits, non_blocking=True)
        hc_b.copy_(gB.conf, non_blocking=True)
        e3 = ev()
        t1 = time.perf_counter()
        st.synchronize()
        t2 = time.perf_counter()
        nvtx.range_pop()
        pred[idx] = hb_b[:len(idx)].numpy()
        segs.extend((("B_in", e0, e1), ("B", e1, e2), ("B_out", e2, e3)))
        nB["full" if len(idx) == BQ else "padded"] += 1
        host["issue_B"] += t1 - t0
        host["wait_B"] += t2 - t1
        host["post_B"] += time.perf_counter() - t2

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    e_start = ev()
    for j in range(len(H)):
        t0 = time.perf_counter()
        nvtx.range_push("S")
        c0 = ev()
        gS.x.copy_(H[j], non_blocking=True)
        c1 = ev()
        gS.replay()
        c2 = ev()
        hb_s.copy_(gS.bits, non_blocking=True)
        hc_s.copy_(gS.conf, non_blocking=True)
        c3 = ev()
        t1 = time.perf_counter()
        st.synchronize()
        t2 = time.perf_counter()
        nvtx.range_pop()
        segs.extend((("S_in", c0, c1), ("S", c1, c2), ("S_out", c2, c3)))
        lo, hi = j * bS, min(N, (j + 1) * bS)
        pred[lo:hi] = hb_s[:hi - lo].numpy()
        r = np.nonzero(hc_s[:hi - lo].numpy().astype(np.float64) < tau1)[0]
        t3 = time.perf_counter()
        host["issue_S"] += t1 - t0
        host["wait_S"] += t2 - t1
        host["route"] += t3 - t2
        if not len(r):
            continue
        nvtx.range_push("queue")
        idx = lo + r
        route[idx] = 1
        g0 = ev()
        q_rows.append(gS.x[torch.as_tensor(r, device=dev)])
        segs.append(("queue", g0, ev()))
        q_idx.append(idx)
        tb = 0.0
        while sum(len(x) for x in q_idx) >= BQ:
            g0 = ev()
            rows, idx = torch.cat(q_rows), np.concatenate(q_idx)
            segs.append(("queue", g0, ev()))
            t4 = time.perf_counter()
            run_b(rows[:BQ], idx[:BQ])
            tb += time.perf_counter() - t4
            q_rows, q_idx = [rows[BQ:]], [idx[BQ:]]
        nvtx.range_pop()
        host["queue"] += time.perf_counter() - t3 - tb
    if q_idx and sum(len(x) for x in q_idx):
        g0 = ev()
        rows, idx = torch.cat(q_rows), np.concatenate(q_idx)
        segs.append(("queue", g0, ev()))
        run_b(rows, idx)
    e_end = ev()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    iv = [(k, 1e3 * e_start.elapsed_time(x), 1e3 * e_start.elapsed_time(y)) for k, x, y in segs]
    res = {"shots": N, "wall_s": wall, "per_shot_us": 1e6 * wall / N, "per_round_us": 1e6 * wall / (N * pc.R),
           "frac_B": float(route.mean()), "B_runs": nB, "host_us_shot": {k: 1e6 * v / N for k, v in host.items()},
           **timeline(iv, 1e3 * e_start.elapsed_time(e_end), N)}
    lastB = [e - s for k, s, e in iv if k == "B"][-1:] if nB["padded"] else []
    res["padded_B_us_shot"] = lastB[0] / N if lastB else 0.0
    if keep:
        res["segments"] = [(k, round(s, 1), round(e, 1)) for k, s, e in iv]
    return res


def overlap(gS, gB, reps=3):
    """S and B replays (input resident): serial on one stream against S and B on two streams, at the ratios of RATIOS,
    and with S's stream at high priority; and whether the outputs stay bitwise."""
    import torch
    cur, s_lo, s_b, s_hi = (torch.cuda.current_stream(), torch.cuda.Stream(), torch.cuda.Stream(),
                            torch.cuda.Stream(priority=-1))

    def snap():
        torch.cuda.synchronize()
        return [t.clone() for t in (gS.bits, gS.conf, gB.bits, gB.conf)]

    def serial(nS, nB):
        for _ in range(nB):
            gB.replay()
        for _ in range(nS):
            gS.replay()

    def two(nS, nB, sS):
        s_b.wait_stream(cur)
        sS.wait_stream(cur)
        with torch.cuda.stream(s_b):
            for _ in range(nB):
                gB.replay()
        with torch.cuda.stream(sS):
            for _ in range(nS):
                gS.replay()
        cur.wait_stream(s_b)
        cur.wait_stream(sS)

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    gS.replay()
    gB.replay()
    ref, out = snap(), []
    for nS, nB in RATIOS:
        r = {"nS": nS, "nB": nB}
        for name, fn in (("serial", lambda: serial(nS, nB)), ("two_streams", lambda: two(nS, nB, s_lo)),
                         ("two_streams_S_high", lambda: two(nS, nB, s_hi))):
            fn()
            r[name + "_s"] = float(np.median([timed(fn) for _ in range(reps)]))
        r["speedup"] = r["serial_s"] / r["two_streams_s"]
        r["speedup_S_high"] = r["serial_s"] / r["two_streams_S_high_s"]
        out.append(r)
    return {"ratios": out, "bitwise": all(bool(torch.equal(x, y)) for x, y in zip(ref, snap()))}


def pipe(a):
    import torch
    import pc_speed
    dev, cd = setup()
    det = load_shots(a.out)[0]
    S, Bn = load(a.s, dev), load(a.b, dev)
    gS = pc_speed.fused_batch(S.model, pc.VERDICT_B)                     # S's frozen batched path (fused@4096)
    gB = pc_speed.Graph(torch.compile(pc_speed.make_fwd(Bn, a.score), dynamic=False), pc.VERDICT_B, 432, dev)
    _, ms = gS.run(torch.as_tensor(det, device=dev))                     # S's margins: tau1 = their q1-quantile
    ms = ms.double().cpu().numpy()
    aS, aB = replay_us(gS)["mean"] / gS.B, replay_us(gB, target_s=1.0)["mean"] / gB.B
    N = a.n
    nb = -(-N // gS.B)
    Hh = torch.zeros(nb * gS.B, det.shape[1], dtype=torch.uint8)
    Hh[:N] = torch.as_tensor(det[np.arange(N) % len(det)])
    H = Hh.reshape(nb, gS.B, -1).pin_memory()
    del Hh
    for _ in range(3):
        gS.replay()
        gB.replay()
    out = {"step": "pipe", "env": cd, "S": a.s, "B": a.b, "score": a.score, "shots": N, "unique_shots": len(det),
           "batch_S": gS.B, "batch_B": gB.B, "alone_gpu_us_shot": {"S": aS, "B": aB}, "runs": []}
    name = f"pipe{a.tag}.json"
    for q1 in [float(q) for q in a.q1.split(",")]:
        tau1 = float(np.quantile(ms, q1)) if q1 > 0 else -np.inf
        w = Watch()
        r = pipe_once(gS, gB, H, N, tau1, dev, keep=abs(q1 - 0.05) < 1e-9)
        r.update(q1=q1, tau1=tau1, clock=w.stop(), model_us_shot=aS + r["frac_B"] * aB)
        out["runs"].append(r)
        dump(a.out, name, out)
        print(json.dumps({"pipe_q1": q1, "frac_B": round(r["frac_B"], 4), "per_shot_us": round(r["per_shot_us"], 3),
                          "model_us": round(r["model_us_shot"], 3), "idle_us": round(r["idle_us_shot"], 3)}))
    if a.trace:
        q1 = 0.05
        nt = min(len(H), 25)
        evs = kprof(lambda: pipe_once(gS, gB, H[:nt], nt * gS.B, float(np.quantile(ms, q1)), dev), 1,
                    os.path.join(a.out, f"trace_pipe{a.tag}_q{q1}"))
        rows, tot = ksum(evs, 1, nt * gS.B)
        out["trace"] = {"q1": q1, "shots": nt * gS.B, "categories": categories(rows), "totals": tot}
    if a.overlap:
        out["overlap"] = overlap(gS, gB)
    dump(a.out, name, out)


def ncuwork(a):
    import torch
    import pc_speed
    dev, cd = setup()
    det = load_shots(a.out)[0]
    pr = load(a.net or ("S55" if a.what == "s55" else "C14"), dev)
    if a.what == "s55":
        gr = pc_speed.fused_batch(pr.model, a.batch)
    else:
        gr = pc_speed.Graph(torch.compile(pc_speed.make_fwd(pr), dynamic=False), a.batch, 432, dev)
    n = min(a.batch, len(det))
    gr.x[:n].copy_(torch.as_tensor(det[:n], device=dev))
    for _ in range(2):
        gr.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("prof")
    gr.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(json.dumps({"ncuwork": a.what, "batch": a.batch, "gpu": cd["gpu"], "driver": cd["driver"]}))


NCU_KEYS = {"duration": r"^gpu__time_duration\.sum$", "sm_pct": r"^sm__throughput\.avg\.pct_of_peak_sustained_elapsed$",
            "mem_pct": r"^gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed$",
            "dram_pct": r"^(gpu__)?dram__throughput\.avg\.pct_of_peak_sustained_elapsed$",
            "l2_pct": r"^lts__throughput\.avg\.pct_of_peak_sustained_elapsed$",
            "l1_pct": r"^l1tex__throughput\.avg\.pct_of_peak_sustained_active$",
            "occ_achieved": r"^sm__warps_active\.avg\.pct_of_peak_sustained_active$",
            "occ_theoretical": r"^sm__maximum_warps_per_active_cycle_pct$",
            "regs": r"^launch__registers_per_thread$", "smem_block": r"^launch__shared_mem_per_block$",
            "smem_dyn": r"^launch__shared_mem_per_block_dynamic$", "block": r"^launch__block_size$",
            "grid": r"^launch__grid_size$", "waves": r"^launch__waves_per_multiprocessor$",
            "dram_read": r"^dram__bytes_read\.sum$", "dram_write": r"^dram__bytes_write\.sum$",
            "l2_bytes": r"^lts__t_bytes\.sum$", "ipc": r"^sm__inst_executed\.avg\.per_cycle_active$",
            "issue_active": r"^sm__instruction_throughput\.avg\.pct_of_peak_sustained_active$"}


def ncu_rows(path):
    """ncu --page raw --csv: a header, a units row, then one row per profiled kernel -> [{metric: (value, unit)}]."""
    rows = list(csv.reader(open(path, newline="")))
    while rows and (len(rows[0]) < 3 or "Kernel Name" not in rows[0]):
        rows.pop(0)                                     # ncu's own messages before the table
    if not rows:
        return []
    hdr, units, data = rows[0], rows[1] if len(rows) > 1 else [], rows[2:]
    if units and units[0].strip().isdigit():            # no units row
        units, data = [""] * len(hdr), rows[1:]
    out = []
    for r in data:
        d = {}
        for h, u, v in zip(hdr, units + [""] * len(hdr), r):
            v2 = v.replace(",", "").strip()
            try:
                d[h] = (float(v2), u)
            except ValueError:
                d[h] = (v.strip(), u)
        out.append(d)
    return out


def ncuparse(a):
    out = []
    for d in ncu_rows(a.csv):
        k = {"name": str(d.get("Kernel Name", ("?", ""))[0])[:160]}
        for key, pat in NCU_KEYS.items():
            hit = [(h, v) for h, v in d.items() if re.match(pat, h)]
            if hit:
                k[key] = hit[0][1][0]
                if hit[0][1][1] and not hit[0][1][1].startswith("%"):
                    k[key + "_unit"] = hit[0][1][1]
        st = [(re.sub(r"^smsp__average_warps?_(issue_stalled|latency_issue_stalled)_|_per_issue_active\.ratio$"
                      r"|\.ratio$", "", h), v[0]) for h, v in d.items()
              if re.match(r"^smsp__average_warps?_(issue_stalled|latency_issue_stalled)_\w+", h)
              and isinstance(v[0], float) and not h.endswith("not_issued.ratio")]
        k["stalls_top"] = sorted(st, key=lambda x: -x[1])[:5]
        k["tensor"] = {h: v[0] for h, v in d.items() if "tensor" in h and isinstance(v[0], float)}
        k["flop_like"] = {h: v[0] for h, v in d.items() if re.search(r"_op_[hfd](fma|add|mul)|flop", h)
                          and isinstance(v[0], float)}
        out.append(k)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"ncu_kernels": len(out), "csv": a.csv}))


# ------------------------------------------------------------------------------------------------------------ report
def f(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.{d}f}" if abs(x) < 1e5 else f"{x:,.0f}"
    return str(x)


def table(hdr, rows):
    return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
                     + ["| " + " | ".join(f(x) if not isinstance(x, str) else x for x in r) + " |" for r in rows])


def report(a):
    def J(n):
        p = os.path.join(a.out, n)
        return json.load(open(p)) if os.path.exists(p) else None
    env_, wk, s5, c1, pp = J("env.json"), J("work.json"), J("s55.json"), J("c14.json"), J("pipe.json")
    md, sm = [f"# profile summary ({a.out})"], {}

    def section(title, fn):
        md.append(f"\n## {title}\n")
        try:
            fn()
        except Exception as ex:                         # a section that fails says so; the raw JSON stays
            md.append(f"(section failed: {ex!r})")

    ce = (env_ or {}).get("ceilings", {})
    nom = (env_ or {}).get("nominal") or {}

    def s_env():
        e = env_["env"]
        md.append(f"{e['gpu']} (sm_{e['capability'][0]}{e['capability'][1]}, {e['sms']} SMs, {e['mem_gib']} GiB, L2 "
                  f"{e['l2_mib']} MiB), driver {e['driver']}, torch {e['torch']} (CUDA {e['cuda']}), "
                  f"Triton {e['triton']}; "
                  f"max SM clock {e['smi'].get('clocks.max.sm')}, power limit {e['smi'].get('power.limit')}; host "
                  f"{e['host']} ({e['cpu']}), job {e['job']}.\n")
        md.append(table(["ceiling", "measured", "nominal"], [
            ["GEMM BF16, TFLOP/s", ce.get("gemm_bf16_tflops"), nom.get("bf16")],
            ["GEMM FP16 (FP32 acc.), TFLOP/s", ce.get("gemm_fp16_tflops"), nom.get("fp16")],
            ["GEMM FP8, TFLOP/s", ce.get("gemm_fp8_tflops"), nom.get("fp8")],
            ["GEMM FP8 fast acc., TFLOP/s", ce.get("gemm_fp8_fastacc_tflops"), None],
            ["GEMM TF32, TFLOP/s", ce.get("gemm_tf32_tflops"), nom.get("tf32")],
            ["GEMM FP32 (SIMT), TFLOP/s", ce.get("gemm_fp32_tflops"), nom.get("fp32")],
            ["DRAM copy, GB/s", ce.get("dram_copy_gbs"), nom.get("dram_gbs")],
            ["DRAM read, GB/s", ce.get("dram_read_gbs"), None],
            ["L2-resident copy, GB/s", ce.get("l2_copy_gbs"), None], ["H2D pinned, GB/s", ce.get("h2d_gbs"), None],
            ["D2H pinned, GB/s", ce.get("d2h_gbs"), None],
            ["1-kernel graph replay, us", ce.get("graph_replay_us"), None]]))
    if env_:
        section("Card and ceilings", s_env)

    def s_s55():
        rows = []
        for B, r in s5["batches"].items():
            rows.append([B, r["amortized_us"]["dev"], r["amortized_us"]["h2h"], r["gpu_us_shot"], r["bitwise"],
                         (r["clock"] or {}).get("sm_min_ratio"), r["mem_peak_gib"],
                         r["kernels_graph"].get("busy_us_iter", 0) / int(B),
                         r["kernels_eager_nopdl"].get("sum_us_iter", 0) / int(B)])
        md.append(table(["batch", "us/shot resident", "us/shot host-to-host", "us/shot GPU (events)", "bitwise",
                         "SM clock min/max", "GiB peak", "graph busy us/shot", "eager no-PDL kernels us/shot"], rows))
        B = str(pc.VERDICT_B) if str(pc.VERDICT_B) in s5["batches"] else list(s5["batches"])[-1]
        for k in ("kernels_eager_nopdl", "kernels_graph"):
            kr = s5["batches"][B][k]
            md.append(f"\nS55 at batch {B}, {k} (labelled {kr['labelled']}):\n")
            md.append(table(["kernel", "us/shot", "share", "MMA TFLOP/s", "SIMT TFLOP/s", "L2-req GB/s",
                             "DRAM-min GB/s", "regs", "smem", "occ. est. %", "grid", "block"],
                            [[r["name"], r["us_shot"], r["share"], r.get("mma_tflops"), r.get("simt_tflops"),
                              r.get("l2_gbs"), r.get("dram_gbs"), r["regs"], r["smem"], r["occ_est"], str(r["grid"]),
                              str(r["block"])] for r in kr["rows"]]))
        if "ptx" in s5["batches"][B]:
            md.append("\n" + table(["kernel", "tensor-core instructions (PTX, static)", "warps", "stages", "regs",
                                    "spills", "shared B"],
                                   [[p_["name"], ", ".join(f"{k} x{v}" for k, v in p_["mma"].items()) or "none",
                                     p_["num_warps"], p_["num_stages"], p_["regs"], p_["spills"], p_["shared_bytes"]]
                                    for p_ in s5["batches"][B]["ptx"]]))
    if s5:
        section("S55 alone (fused path)", s_s55)

    def s_c14():
        rows = [[B, r["amortized_us"]["dev"], r["amortized_us"]["h2h"], r["gpu_us_shot"], r["build_s"],
                 (r["clock"] or {}).get("sm_min_ratio"), r["mem_peak_gib"], r["kernel_names"]]
                for B, r in c1["batches"].items()]
        md.append(table(["batch", "us/shot resident", "us/shot host-to-host", "us/shot GPU (events)", "build s",
                         "SM clock min/max", "GiB peak", "kernel names"], rows))
        for B, r in c1["batches"].items():
            md.append(f"\nC14 at batch {B} by category ({r['traced']}; sum {f(r['totals'].get('sum_us_iter'), 0)} us, "
                      f"busy {f(r['totals'].get('busy_us_iter'), 0)} us per replay):\n")
            md.append(table(["category", "kernels", "us/shot", "share"],
                            [[c, v["kernels"], v["us_shot"], v["share"]] for c, v in r["categories"].items()]))
            md.append("\n" + table(["kernel (top 15)", "calls", "us/shot", "share", "tensor cores (name)", "regs",
                                    "occ. est. %"],
                                   [[x["name"][:90], x["calls"], x["us_shot"], x["share"],
                                     f"{x['tc']['tensor_cores']} {x['tc']['precision'] or ''}", x["regs"], x["occ_est"]]
                                    for x in r["kernels"][:15]]))
        g = c1.get("gmm", {})
        md.append(f"\nTriton _gmm: autotuned {g.get('autotune')}; compiled kernels: "
                  + "; ".join(f"mma {k['mma']} warps {k['num_warps']} stages {k['num_stages']} regs {k['regs']}"
                              for k in g.get("kernels", [])))
    if c1:
        section("C14 alone (compiled path)", s_c14)

    def s_roof():
        rows = []
        if wk and s5:
            w = wk["nets"][s5["net"]]
            t = w["engine_total"]
            u = 2 * w["useful_macs"]["total"]
            for B, r in s5["batches"].items():
                ts = r["gpu_us_shot"] * 1e-6
                bound = max(t["mma"] / (ce.get("gemm_fp16_tflops", np.inf) * 1e12),
                            t["simt"] / (ce.get("gemm_fp32_tflops", np.inf) * 1e12),
                            t["dram"] / (ce.get("dram_copy_gbs", np.inf) * 1e9),
                            t["l2"] / (ce.get("l2_copy_gbs", np.inf) * 1e9))
                rows.append([f"S55 fused @{B}", r["gpu_us_shot"], t["mma"] / 1e6, u / 1e6, t["mma"] / ts / 1e12,
                             100 * t["mma"] / ts / 1e12 / ce.get("gemm_fp16_tflops", np.nan), t["dram"] / 1e6,
                             t["dram"] / ts / 1e9, 100 * t["dram"] / ts / 1e9 / ce.get("dram_copy_gbs", np.nan),
                             t["l2"] / 1e6, t["l2"] / ts / 1e9,
                             100 * t["l2"] / ts / 1e9 / ce.get("l2_copy_gbs", np.nan),
                             t["mma"] / t["dram"], 1e6 * bound, ts / bound])
                sm[f"S55@{B}"] = rows[-1]
        if wk and c1:
            w = wk["nets"][c1["net"]]
            F, Bd = w["flops"]["total"], w["bytes_model"]["total"]
            for B, r in c1["batches"].items():
                ts = r["gpu_us_shot"] * 1e-6
                bound = max(F / (ce.get("gemm_bf16_tflops", np.inf) * 1e12),
                            Bd / (ce.get("dram_copy_gbs", np.inf) * 1e9))
                rows.append([f"C14 compiled @{B}", r["gpu_us_shot"], F / 1e6, 2 * w["macs_per_shot"] / 1e6,
                             F / ts / 1e12,
                             100 * F / ts / 1e12 / ce.get("gemm_bf16_tflops", np.nan), Bd / 1e6, Bd / ts / 1e9,
                             100 * Bd / ts / 1e9 / ce.get("dram_copy_gbs", np.nan), None, None, None, F / Bd,
                             1e6 * bound,
                             ts / bound])
                sm[f"C14@{B}"] = rows[-1]
        md.append(table(["path", "us/shot (GPU)", "MFLOP/shot executed", "MFLOP/shot useful", "TFLOP/s",
                         "% GEMM ceiling", "DRAM MB/shot", "DRAM GB/s", "% DRAM ceiling", "L2-req MB/shot", "L2 GB/s",
                         "% L2 ceiling", "FLOP/DRAM byte", "roofline bound us/shot", "measured / bound"], rows))
        md.append("\nS55's ceiling: the FP16 GEMM (its tl.dot inputs), the FP32 SIMT GEMM for its vote, DRAM and the "
                  "L2-resident copy; C14's: the BF16 GEMM and DRAM. C14's bytes are cascade_bytes' model.")
    section("Roofline", s_roof)

    def s_pipe():
        md.append(f"S alone {f(pp['alone_gpu_us_shot']['S'], 3)} us/shot, B alone {f(pp['alone_gpu_us_shot']['B'], 3)} "
                  f"us/shot (GPU, input resident); {pp['shots']:,} shots ({pp['unique_shots']:,} unique, cycled).\n")
        rows = []
        for r in pp["runs"]:
            g = r["gpu_us_shot"]
            cp = sum(g.get(k, 0) for k in ("S_in", "S_out", "B_in", "B_out"))
            rows.append([r["q1"], r["frac_B"], r["per_shot_us"], r["model_us_shot"], g.get("S", 0), g.get("B", 0), cp,
                         g.get("queue", 0), r["idle_us_shot"], r["padded_B_us_shot"], r["B_runs"]["full"],
                         r["B_runs"]["padded"], 100 * g.get("S", 0) / r["per_shot_us"],
                         100 * g.get("B", 0) / r["per_shot_us"]])
        md.append(table(["q1", "routed", "us/shot wall", "model aS + q1 aB", "S", "B", "copies", "queue", "GPU idle",
                         "padded last B", "B full", "B padded", "S % of wall", "B % of wall"], rows))
        r5 = next((r for r in pp["runs"] if abs(r["q1"] - 0.05) < 1e-9), None)
        if r5:
            md.append("\nIdle gaps before each kind of GPU segment at q1 = 5 % (count, us per shot, mean, p99, max):\n")
            md.append(table(["before", "n", "us/shot", "mean us", "p99 us", "max us"],
                            [[k, v["n"], v["us_shot"], v["mean_us"], v["p99_us"], v["max_us"]]
                             for k, v in r5["gaps"].items()]))
            md.append("\nHost time per shot at q1 = 5 % (us): " + ", ".join(f"{k} {f(v, 3)}"
                                                                             for k, v in r5["host_us_shot"].items()))
        if "overlap" in pp:
            o = pp["overlap"]
            md.append(f"\nS and B on two streams (input resident; outputs bitwise: {o['bitwise']}):\n")
            md.append(table(["S batches per B batch", "serial s", "two streams s", "speedup", "S high priority s",
                             "speedup"],
                            [[f"{x['nS']}:{x['nB']}", x["serial_s"], x["two_streams_s"], x["speedup"],
                              x["two_streams_S_high_s"], x["speedup_S_high"]] for x in o["ratios"]]))
    if pp:
        section("S55 -> C14 batched", s_pipe)

    for tag in ("s55", "c14"):
        nc = J(f"ncu_{tag}.json")
        if nc:
            def s_ncu(nc=nc):
                top = sorted(nc, key=lambda k: -(k.get("duration") if isinstance(k.get("duration"), float) else 0))[:15]
                md.append(table(["kernel", "duration", "SM %", "DRAM %", "L2 %", "L1 %", "occ. achieved %",
                                 "occ. theoretical %", "regs", "top stalls"],
                                [[k["name"][:70], f"{f(k.get('duration'))} {k.get('duration_unit', '')}",
                                  k.get("sm_pct"), k.get("dram_pct"), k.get("l2_pct"), k.get("l1_pct"),
                                  k.get("occ_achieved"), k.get("occ_theoretical"),
                                  k.get("regs"), ", ".join(f"{s} {v:.1f}" for s, v in k.get("stalls_top", [])[:3])]
                                 for k in top]))
            section(f"Nsight Compute, {tag}", s_ncu)
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
    dump(a.out, "summary.json", sm)
    print("\n".join(md))


# ---------------------------------------------------------------------------------------------------------- selftest
def selftest(a):
    names = ["stem", "block0", "block1", "block2", "pool", "vote", "top2"]
    fns = ["_k_stem", "_k_block", "_k_block", "_k_block", "_k_pool", "_k_vote", "_k_top2"]
    evs, t = [], 0.0
    for _ in range(3):
        for fn in fns:
            evs.append({"ph": "X", "cat": "kernel", "name": fn, "ts": t, "dur": 10.0, "args": {"grid": [16, 4096, 1]}})
            t += 9.0                                    # 1 us of PDL overlap
    r = ksum_fused(evs, 3, 4096, names)
    assert r["labelled"] and [x["name"] for x in r["rows"]] == names, r
    assert abs(r["sum_us_iter"] - 70.0) < 1e-9 and abs(r["busy_us_iter"] - (9 * 20 + 10) / 3) < 1e-9, r
    assert not ksum_fused(evs[:-1], 3, 4096, names)["labelled"]
    rows, tot = ksum(evs + [{"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD (Pinned -> Device)", "ts": 0.0,
                             "dur": 5.0}], 3, 4096)
    assert [x["cat"] for x in rows].count("copy/memset") == 1 and categories(rows)["fused engine"]["kernels"] == 5
    assert kcat("triton_poi_fused_add_0") == "inductor pointwise" and kcat("_gmm") == "bipartite conv (Triton _gmm)"
    assert tc_name("nvjet_tst_128x64_64x4")["tensor_cores"] and not tc_name("triton_poi_fused_add_0")["tensor_cores"]
    assert tc_name("sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x32_stage4_warpsize2x2x1_tensor16x8x16_"
                   "kernel") == {"tensor_cores": True, "precision": "bf16"}
    tl_ = timeline([("S_in", 1.0, 2.0), ("S", 2.0, 10.0), ("S_out", 10.0, 11.0), ("S_in", 15.0, 16.0)], 20.0, 2)
    assert abs(tl_["gpu_busy_us_shot"] - 5.5) < 1e-9 and abs(tl_["idle_us_shot"] - 4.5) < 1e-9
    assert tl_["gaps"]["S_in"]["n"] == 2 and abs(tl_["gaps"]["S_in"]["us_shot"] - 2.5) < 1e-9
    assert abs(union_us([(0, 2), (1, 3), (5, 6)]) - 4) < 1e-9
    import tempfile
    d = tempfile.mkdtemp()
    p_ = os.path.join(d, "raw.csv")
    with open(p_, "w") as fh:
        fh.write('==PROF== Disconnected\n"ID","Kernel Name","gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_'
                 'sustained_elapsed","smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",'
                 '"smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio","launch__registers_per_thread"\n'
                 '"","","usecond","%","ratio","ratio","register/thread"\n'
                 '"0","_k_block","1,234.5","41.2","3.5","0.7","128"\n')
    ncuparse(argparse.Namespace(csv=p_, out=os.path.join(d, "ncu.json")))
    k = json.load(open(os.path.join(d, "ncu.json")))[0]
    assert k["duration"] == 1234.5 and k["duration_unit"] == "usecond" and k["sm_pct"] == 41.2, k
    assert k["stalls_top"][0][0] == "long_scoreboard", k
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shots")
    s.add_argument("--p", type=float, default=0.003)
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED)
    s.add_argument("--chunks", type=int, default=10)
    s.add_argument("--procs", type=int, default=4)
    for n in ("env", "work", "s55", "c14", "pipe", "ncuwork", "report"):
        sub.add_parser(n)
    for n, p_ in sub.choices.items():
        p_.add_argument("--out", default=".", help="the output directory (shots.npz is read from it)")
    sub.choices["work"].add_argument("--s", default="S55")
    sub.choices["work"].add_argument("--b", default="C14")
    sub.choices["s55"].add_argument("--net", default="S55")
    sub.choices["s55"].add_argument("--batches", default="256,1024,4096,16384")
    sub.choices["s55"].add_argument("--secs", type=float, default=2.0)
    sub.choices["c14"].add_argument("--net", default="C14")
    sub.choices["c14"].add_argument("--batches", default="4096,8192")
    sub.choices["c14"].add_argument("--secs", type=float, default=2.0)
    s = sub.choices["pipe"]
    s.add_argument("--s", default="S55")
    s.add_argument("--b", default="C14")
    s.add_argument("--score", default="minabs")
    s.add_argument("--q1", default=",".join(str(q) for q in Q1S))
    s.add_argument("--n", type=int, default=500_000)
    s.add_argument("--tag", default="")
    s.add_argument("--no-trace", dest="trace", action="store_false")
    s.add_argument("--no-overlap", dest="overlap", action="store_false")
    s = sub.choices["ncuwork"]
    s.add_argument("--what", required=True, choices=["s55", "c14"])
    s.add_argument("--net", default="")
    s.add_argument("--batch", type=int, default=pc.VERDICT_B)
    s = sub.add_parser("ncuparse")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if getattr(a, "out", None) and a.cmd not in ("ncuparse", "selftest"):
        os.makedirs(a.out, exist_ok=True)
    {"shots": shots, "env": env, "work": work, "s55": s55, "c14": c14, "pipe": pipe, "ncuwork": ncuwork,
     "ncuparse": ncuparse, "report": report, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
