"""The joint decision of the d = 13 composition C5 (S = ConvLines d13_b5_r775, B = Cascade CS13), in the timed pipeline
(sv_speed.py pipeline --joint-w W) and its bitwise check against the reads (sv_speed.py bitwise --joint-w W).

Rule. On a shot that S routes to B (|c| below C5's tau1) and on which the two decisions differ, the answer is 1 if
c + w z > 0, 0 if it is < 0, and B's answer if it is 0; c is S's logit, z B's (its bf16 output in fp32) on the path in
use. Every other shot is answered as in C5. w is a parameter of the process.

Arithmetic, fixed: fp32 on the GPU, v = c + m with m = w32 z one IEEE fp32 multiply rounded to nearest (PTX mul.rn.f32 in
inline assembly, never fused into an FMA) and one fp32 add; w32 = float32(w), rounded to nearest. joint_np computes the
same in numpy, bit for bit.

JointB13 is B's computation (sv_common.b_fn on its path) followed by the joint step, captured in one CUDA graph on a
stream of its own, behind sv_common.Graph's interface (x, logit, dec, n, replay): logit is B's logit, dec the
composition's decision with the joint step, bdec B's own decision, flag 1 where the two decisions differed. Single mode:
the joint step reads S's own logit and decision (S's graph's static outputs) in place. Batched mode: the rows S routes
queue on the GPU with S's logit and decision, which the caller copies into the graph's c and ds before each replay."""
import hashlib
import os

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 128


@triton.jit
def _mul_rn(a, b):
    """a * b in fp32: one IEEE multiply rounded to nearest (inline PTX, never contracted into an FMA)."""
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _k_j13(c_ptr, ds_ptr, z_ptr, db_ptr, out_ptr, fl_ptr, w, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    c = tl.load(c_ptr + i, mask=m, other=0.0)
    z = tl.load(z_ptr + i, mask=m, other=0.0)
    ds = tl.load(ds_ptr + i, mask=m, other=0)
    db = tl.load(db_ptr + i, mask=m, other=0)
    v = c + _mul_rn(z, tl.zeros((BLOCK,), dtype=tl.float32) + w)
    j = tl.where(v > 0.0, 1, tl.where(v < 0.0, 0, db.to(tl.int32))).to(tl.uint8)
    dif = ds != db
    tl.store(out_ptr + i, tl.where(dif, j, db), mask=m)
    tl.store(fl_ptr + i, dif.to(tl.int8), mask=m)


def w32_of(w):
    x = np.float32(w)
    return float(x), f"0x{int(np.array([x]).view(np.uint32)[0]):08x}"


def step(c, ds, z, db, out, fl, w):
    n = out.shape[0]
    _k_j13[(triton.cdiv(n, BLOCK),)](c, ds, z, db, out, fl, w32_of(w)[0], n, BLOCK=BLOCK, num_warps=4)


class JointB13:
    """B's computation fn (x (n, D) uint8 -> (z (n,) fp32, dec (n,) uint8)) at batch n and the joint step in one CUDA
    graph (module docstring). c_src, ds_src: S's logit and decision buffers (single mode), else the graph's own."""

    def __init__(self, fn, n, Dd, dev, w, c_src=None, ds_src=None):
        self.n, self.B, self.calls = n, n, 1
        self.x = torch.zeros(n, Dd, dtype=torch.uint8, device=dev)
        self.c = c_src if c_src is not None else torch.zeros(n, dtype=torch.float32, device=dev)
        self.ds = ds_src if ds_src is not None else torch.zeros(n, dtype=torch.uint8, device=dev)
        for name, t, dt in (("c", self.c, torch.float32), ("ds", self.ds, torch.uint8)):
            if tuple(t.shape) != (n,) or t.dtype != dt or not t.is_contiguous():
                raise ValueError(f"JointB13: {name} is {tuple(t.shape)} {t.dtype}, needs contiguous ({n},) {dt}")
        self.w, self.w32_hex = w32_of(w)
        self.dec = torch.zeros(n, dtype=torch.uint8, device=dev)
        self.flag = torch.zeros(n, dtype=torch.int8, device=dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(st):
            for _ in range(3):
                z, db = fn(self.x)
                step(self.c, self.ds, z, db, self.dec, self.flag, w)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g, stream=torch.cuda.Stream()):    # a capture stream of its own
            self.logit, self.bdec = fn(self.x)
            step(self.c, self.ds, self.logit, self.bdec, self.dec, self.flag, w)

    def replay(self):
        self.g.replay()


def joint_np(c, z, w):
    """The joint answer in numpy, bit for bit _k_j13's on a shot whose decisions differ: c, z float32 -> uint8."""
    c = np.asarray(c, np.float32)
    z = np.asarray(z, np.float32)
    v = c + np.float32(w) * z
    return np.where(v > 0, 1, np.where(v < 0, 0, (z > 0).astype(np.int64))).astype(np.uint8)


def joint_gpu(c, z, ds, w):
    """_k_j13 on host arrays (every row as given: the decisions ds of S, B's decision z > 0) -> (answers, flags) numpy."""
    dev = torch.device("cuda")
    c_ = torch.as_tensor(np.ascontiguousarray(c, np.float32), device=dev)
    z_ = torch.as_tensor(np.ascontiguousarray(z, np.float32), device=dev)
    ds_ = torch.as_tensor(np.ascontiguousarray(ds, np.uint8), device=dev)
    db_ = (z_ > 0).to(torch.uint8)
    out = torch.empty_like(db_)
    fl = torch.empty(len(db_), dtype=torch.int8, device=dev)
    step(c_, ds_, z_, db_, out, fl, w)
    return out.cpu().numpy(), fl.cpu().numpy()


def assemble_joint(sc, mode, o, ch, w):
    """C5's decisions with the joint step in a mode, from the reads' outputs o (sv_common.load_read) and the thresholds
    ch: sv_common.assemble's C5 (S1, B1 single; SB, BQ5 batched), then on the routed shots whose decisions differ the
    joint answer of S's logit c and B's logit z -> (decision, stage, joint rows bool)."""
    dec, stage = sc.assemble("C5", mode, o, ch)
    S, Bn = ("S1", "B1") if mode == "single" else ("SB", "BQ5")
    c, ds = o[S]["c"], o[S]["dec"]
    bz, bd = o[Bn]["z"], o[Bn]["dec"]
    jr = (stage == 1) & (ds != bd)
    dec[jr] = joint_np(c[jr], bz[jr], w)
    return dec, stage, jr


def source():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def record(w):
    w32, hx = w32_of(w)
    return {"w": float(w), "w32": w32, "w32_hex": hx, "sv_joint_py": source(), "triton": triton.__version__,
            "rule": "1 if c + w z > 0, 0 if < 0, B's answer if 0, on routed shots whose decisions differ; fp32",
            "impl": "JointB13: B's computation + _k_j13 in one CUDA graph"}


def check(a):
    """One C5 pipeline file with the joint step (sv_speed pipeline --joint-w --save) against the reads of its shots
    (sv_read: S1, B1 in READ/single; SB, BQ5 in READ/batched), as sv_speed bitwise checks a whole stage: every decision
    and stage against assemble_joint, S's logits, and B's logits on the shots that reached B; plus the joint rows, the
    decisions the joint step changed, and _k_j13 on the read's values against joint_np."""
    import json
    from surface import sv_common as sc
    lock = json.load(open(a.lock))
    m, z = sc.load_npz(a.pipe)
    cfg, mode, p, n = m["config"], m["mode"], m["p"], m["shots"]
    w = (m.get("joint") or {}).get("w")
    if cfg != "C5" or w is None:
        raise SystemExit(f"{a.pipe}: not a C5 pipeline with the joint step")
    comps = ("S1", "B1") if mode == "single" else ("SB", "BQ5")
    o = sc.load_read(os.path.join(a.read, mode), p, comps, range(*m["chunks"]))
    if o["digests"][:len(m["digests"])] != m["digests"]:
        raise SystemExit(f"{a.pipe}: the pipeline ran other shots than the read")
    oo = {c: {k: v[:n] for k, v in o[c].items()} for c in comps}
    ch = lock["p"][str(p)]["chosen"]
    dec, stage, jr = assemble_joint(sc, mode, oo, ch, w)
    S, Bn = comps
    via = z["stage"] >= 1
    chk = {"decisions": int((z["pred"] != dec).sum()), "stages": int((z["stage"] != stage).sum()),
           "S_logits": int((z["s_log"].view(np.uint32) != oo[S]["c"].view(np.uint32)).sum()),
           "B_logits": int((z["b_log"][via].view(np.uint32) != oo[Bn]["z"][via].view(np.uint32)).sum())}
    g, _ = joint_gpu(oo[S]["c"][jr], oo[Bn]["z"][jr], oo[S]["dec"][jr], w)
    chk["gpu_vs_np"] = int((g != dec[jr]).sum())
    base = sc.assemble("C5", mode, oo, ch)[0]
    obs = o["obs"][:n].reshape(-1)
    rec = {"pipe": os.path.abspath(a.pipe), "config": cfg, "mode": mode, "p": p, "shots": n, "w": w,
           "joint": m.get("joint"), "path_B": m.get("path_B"), "routed": int(via.sum()), "joint_rows": int(jr.sum()),
           "changed_by_joint": int((dec != base).sum()), "fail_C5": int((base != obs).sum()),
           "fail_C5_joint": int((dec != obs).sum()), "fail_pipe": int((z["pred"] != obs).sum()), **chk,
           "pass": all(v == 0 for v in chk.values()),
           "timing": {"value": m["latency_us"]["mean"], "p999": m["latency_us"]["p999"]} if mode == "single" else
           {"value": m["per_shot_us"]}}
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec), flush=True)
    raise SystemExit(0 if rec["pass"] else 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=check.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check")
    s.add_argument("--lock", required=True)
    s.add_argument("--pipe", required=True)
    s.add_argument("--read", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    check(a)
