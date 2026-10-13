"""NVIDIA's Ising pre-decoders as TensorRT engines at batch 1, the way the studies built, checked and timed them. This
is our code; it runs NVIDIA's public module and checkpoints from your clone (ISING_DECODING_DIR, README section 3).

The engine is what the official code exports and builds (NVIDIA Ising-Decoding 3ce4379,
code/evaluation/logical_error_rate.py, ONNX_WORKFLOW=2): PreDecoderMemoryEvalModule of Model 1 / Model 4 with
distance = n_rounds = d (convlines.opponents.ising_module), detector bits (B, d (d^2 - 1)) uint8 -> pre-decoder frame
+ residual detectors (B, 1 + d (d^2 - 1)) uint8, so the timed call includes the module's input scatter and its
residual computation. ONNX: opset 18, dynamic batch, constant folding, dynamo off, traced on random bits (seed 0) as
the official code does. FP8: modelopt.onnx.quantization (quantize_mode "fp8", op_types_to_quantize ["Conv"],
high_precision_dtype "fp16", calibration = validation chunk 0 shots 0-255), STRONGLY_TYPED network. FP16:
BuilderFlag.FP16 on the FP32 ONNX. Departures from the official build, all for the batch-1 timing: a fixed (1, m)
profile instead of (1, B, B); an explicit 4 GiB workspace; builder optimisation level 3 (the default) and 5; DETAILED
profiling verbosity (layer and tactic names kept in the plan; it does not change tactic choice); a timing cache per
work directory, i.e. per (GPU, MIG profile, TensorRT). No ReLU substitution, no sparsity, no INT8.

A plan counts only if `check` passes it on validation chunk 0 (10,000 shots) against the official PyTorch numerics
(fp32 weights, TF32 as NVIDIA's runner enables it) followed by plain PyMatching: decision agreement at least 99.9 %
(FP16) or 99 % (FP8), the paired test not "worse", the residual detector density within 5 %, and for an FP8 plan
every convolution the recipe quantized running an FP8 (e4m3) tactic (the check's "fp8" verdict). `check` passes a
plan only when all of these hold and the plan file is the one its build recorded (same sha256); `time` times a plan
only when its check passed and the plan it loads has the checked sha256, and reports any other plan as not timed. The
studies timed the plans of each d in two processes on the timing H100, forward and reversed order, five repetitions
per plan; the registered Ising time per model and d is the fastest repetition of the fastest validated plan, plus
Ising's PyMatching share on the Grace basis (timing/grace_basis.csv).

Accuracy (`eval`). The studies read on the test sets the official PyTorch numerics (convlines.evaluate, "Ising M1 + PM"
/ "Ising M4 + PM"), every plan they timed, and the plans of a batch sweep: each model and precision built for a
static batch B of 1, 2, 4, 8, 16, 32 or 64 (TensorRT picks other tactics per B, and so other rounding), kept where
the plan's raw outputs on the 200,000 validation shots differed from those of the smaller batches. A part's name says
which plan: {M}_{prec}_B{B} a sweep plan (`build --parts`: the builder's defaults, as the sweep's: optimisation level
3, no explicit workspace, no timing cache; from prep's ONNX, where the sweep traced its own export on an all-zero shot,
with the same nodes, weights, inputs and outputs, without prep's intermediate shape annotations), {M}_{prec}_L{level}_B1
a timed plan; results/s*_test_eval.csv has one row per part and d, and `--parts registered` takes that list. `eval`
runs each plan with exactly B shots per call (the last call padded with all-zero shots, as the studies' read), plain
PyMatching on the residual detectors, flip = pre-decoder frame XOR PyMatching; with --corr also correlated PyMatching
on the same residual detectors (part {part}_corr), which study 4 read at d = 17 (results/s4_test_eval.csv). The plotted Ising accuracy is the fewest
failures over a model's parts. TensorRT's tactic choice depends on the GPU and the build, so a plan rebuilt elsewhere
can decide a few shots per million differently from the studies' plan; the registered numbers are those of the
studies' plans.

Files: ONNX = target-independent ONNX files (exported and calibrated once per (model, d)); WORK = one directory per
target GPU (plans, timing cache, reports); VAL = the validation chunks (`python -m convlines.testset val --d D
--chunks 0 --out VAL`; at d = 7-13 the studies' validation chunks were sampled on an ARM64 machine, so the FP8
calibration rows regenerated on x86-64 can differ from theirs); SHOTS = the test-set chunks (`convlines.testset regen`).
    python -m timing.ising_trt prep  --d 13 --val VAL --onnx ONNX                 # ONNX, FP8 Q/DQ and its audit
    python -m timing.ising_trt build --d 13 --onnx ONNX --work WORK               # timed plans (model x precision x level)
    python -m timing.ising_trt check --d 13 --val VAL --onnx ONNX --work WORK     # acceptance (exit 1 = FAIL)
    python -m timing.ising_trt time  --d 13 --val VAL --onnx ONNX --work WORK --order fwd --out CSV   # CUDA graphs
    python -m timing.ising_trt build --d 15 --onnx ONNX --work WORK --parts registered          # the parts read at d
    python -m timing.ising_trt eval  --d 15 --onnx ONNX --work WORK --parts registered --set T3 --shots SHOTS \\
        --chunks 0-99 --out DIR                                                   # failures per part
    python -m timing.ising_trt eval  --d 17 --onnx ONNX --work WORK --parts registered --set T5 --shots SHOTS \\
        --chunks 0-99 --corr --out DIR                                            # d = 17: + correlated PM"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np
import torch

from convlines import circuits, opponents, stats, testset

MODELS = ("M1", "M4")
PRECS = ("fp16", "fp8")
LEVELS = (3, 5)
WS_GIB = 4
CALIB = (0, 256)                     # validation chunk 0 shots [0, 256): the FP8 calibration rows
OUT_NAME = "L_and_residual_dets"
# acceptance on validation chunk 0 (10,000 shots), fixed before the first timed run
ACCEPT = dict(agree_fp16=0.999, agree_fp8=0.99, density_rel=0.05)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def val0(d, val_dir):
    """(det, obs, split) of the validation chunk 0 at d."""
    det, obs = testset.load(os.path.join(val_dir, testset.name("val", d, 0)), d)
    return det, obs, "validation"


def versions():
    v = dict(torch=torch.__version__, cuda=torch.version.cuda)
    for mod in ("tensorrt", "modelopt", "onnx", "onnxruntime"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception as e:                                   # noqa: BLE001
            v[mod] = f"unavailable ({type(e).__name__})"
    if torch.cuda.is_available():
        v["gpu"] = torch.cuda.get_device_name()
        p = torch.cuda.get_device_properties(0)
        v["sms"] = p.multi_processor_count
        v["capability"] = f"{p.major}.{p.minor}"
    try:
        v["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.split()[0]
    except Exception:                                            # noqa: BLE001
        v["driver"] = "?"
    return v


def names(onnx_dir, work, mdl, d, prec=None, level=None, batch=1):
    """The files of (model, d): ONNX and prep record; with prec, a timed plan (level) or a sweep plan (no level, batch)."""
    base = f"{mdl}_d{d}"
    r = dict(onnx16=os.path.join(onnx_dir, f"{base}.onnx"), onnx8=os.path.join(onnx_dir, f"{base}_fp8.onnx"),
             prep=os.path.join(onnx_dir, f"{base}_prep.json"))
    if work:
        r["ref"] = os.path.join(work, f"ref_{base}.npz")
        if prec:
            tag = f"L{level}_B1" if level else f"B{batch}"
            stem = os.path.join(work, "plans", f"{base}_{prec}_{tag}")
            r.update(plan=stem + ".plan", build=stem + "_build.json", check=stem + "_check.json")
    return r


PART = re.compile(r"^(M[14])_(fp16|fp8)(?:_L(\d))?_B(\d+)$")


def parts(a, d):
    """[(name, model, precision, level or None, batch)] of --parts; "registered": every TensorRT part the study read
    at d (the rows of results/s2_test_eval.csv, s3_test_eval.csv or s4_test_eval.csv)."""
    want = a.parts
    if want == ["registered"]:
        f = os.path.join(circuits.ROOT, "results", {15: "s3_test_eval.csv", 17: "s4_test_eval.csv"}.get(d, "s2_test_eval.csv"))
        want = list(dict.fromkeys(r["vs"] for r in csv.DictReader(open(f)) if int(r["d"]) == d and PART.match(r["vs"])))
    out = []
    for n in want:
        mt = PART.match(n)
        assert mt and (mt[3] is None or mt[4] == "1"), f"not a TensorRT part name: {n}"
        out.append((n, mt[1], mt[2], int(mt[3]) if mt[3] else None, int(mt[4])))
    return out


# ------------------------------------------------------------------------------------------------- prep
def export(pipe, m, path, device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    ex = torch.randint(0, 2, (1, m), dtype=torch.uint8, device=device, generator=g)
    torch.onnx.export(pipe, ex, path, opset_version=18, external_data=False, input_names=["dets"],
                      output_names=[OUT_NAME], dynamic_axes={"dets": {0: "batch"}, OUT_NAME: {0: "batch"}},
                      do_constant_folding=True, dynamo=False)


def quantize_fp8(src, dst, calib):
    import modelopt.onnx.quantization as mq
    mq.quantize(onnx_path=src, quantize_mode="fp8", calibration_data={"dets": calib}, output_path=dst,
                op_types_to_quantize=["Conv"], high_precision_dtype="fp16")


def qdq_audit(path):
    """Q/DQ placement of a quantized ONNX: for every Conv, what feeds its input and its weight (QuantizeLinear ->
    DequantizeLinear pair with its FP8 / INT8 type, a DequantizeLinear of a constant, or nothing); every Q/DQ node not
    next to a Conv; the op histogram."""
    import onnx
    from collections import Counter
    m = onnx.load(path)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    init = {i.name: i for i in g.initializer}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    tname = {v: k for k, v in onnx.TensorProto.DataType.items()}

    def qtype(qnode):
        if len(qnode.input) > 2 and qnode.input[2] in init:
            return tname.get(init[qnode.input[2]].data_type, "?")
        return "no zero point"

    def feed(inp):
        n = prod.get(inp)
        if n is None:
            return "initializer" if inp in init else "graph input"
        if n.op_type != "DequantizeLinear":
            return f"{n.op_type} (not quantized)"
        src = n.input[0]
        q = prod.get(src)
        if q is not None and q.op_type == "QuantizeLinear":
            return f"Q/DQ {qtype(q)}"
        if src in init:
            return f"DQ of constant {tname.get(init[src].data_type, '?')}"
        return f"DQ of {q.op_type if q is not None else '?'}"

    convs = [n for n in g.node if n.op_type == "Conv"]
    rows = [dict(conv=c.name, x=feed(c.input[0]), w=feed(c.input[1])) for c in convs]
    conv_inputs = {c.input[0] for c in convs} | {c.input[1] for c in convs}
    stray = []
    for n in g.node:
        if n.op_type == "DequantizeLinear" and n.output[0] not in conv_inputs:
            stray.append(dict(node=n.name, to=[u.op_type for u in users.get(n.output[0], [])]))
    ops = Counter(n.op_type for n in g.node)
    fp8 = all(r["x"].startswith("Q/DQ FLOAT8E4M3FN") and ("FLOAT8E4M3FN" in r["w"]) for r in rows)
    return dict(convs=len(convs), all_convs_fp8_qdq=bool(fp8 and convs), conv_feeds=rows, stray_dq=stray,
                ops=dict(sorted(ops.items())))


def cmd_prep(a):
    os.makedirs(a.onnx, exist_ok=True)
    for d in a.d:
        det, _, split = val0(d, a.val)
        m = det.shape[1]
        calib = np.ascontiguousarray(det[CALIB[0]:CALIB[1]])
        for mdl in a.models:
            f = names(a.onnx, "", mdl, d)
            if os.path.exists(f["prep"]) and not a.force:
                print(f"{mdl} d{d}: prepared ({f['prep']})", flush=True)
                continue
            pipe, _, cfg = opponents.ising_module(mdl, d, a.device)
            t0 = time.time()
            export(pipe, m, f["onnx16"], a.device)
            t1 = time.time()
            del pipe
            torch.cuda.empty_cache()
            err = None
            try:
                quantize_fp8(f["onnx16"], f["onnx8"], calib)
            except Exception as e:                               # noqa: BLE001  recorded; FP16 goes on
                err = repr(e)[:500]
                print(f"{mdl} d{d}: FP8 QUANTIZATION FAILED {err}", flush=True)
            t2 = time.time()
            rep = dict(model=mdl, d=d, m=m, n_rounds=int(cfg.test.n_rounds), checkpoint_sha256=opponents.ISING[mdl][2],
                       onnx16=os.path.basename(f["onnx16"]), onnx16_sha256=sha256_file(f["onnx16"]),
                       onnx8=os.path.basename(f["onnx8"]),
                       onnx8_sha256=sha256_file(f["onnx8"]) if err is None else None, fp8_error=err,
                       calib=dict(split=split, chunk=0, rows=list(CALIB),
                                  sha256=hashlib.sha256(calib.tobytes()).hexdigest()),
                       export_s=round(t1 - t0, 1), quantize_s=round(t2 - t1, 1), versions=versions(),
                       qdq=qdq_audit(f["onnx8"]) if err is None else None)
            json.dump(rep, open(f["prep"], "w"), indent=1)
            q = rep["qdq"] or dict(convs="?", all_convs_fp8_qdq=False, stray_dq=[], ops={})
            print(f"{mdl} d{d}: export {rep['export_s']} s, fp8 quantize {rep['quantize_s']} s; {q['convs']} Conv, "
                  f"all FP8 Q/DQ on input and weight: {q['all_convs_fp8_qdq']}; stray DQ {len(q['stray_dq'])}; "
                  f"ops {q['ops']}", flush=True)


# ------------------------------------------------------------------------------------------------- build
def build_plan(onnx_path, m, prec, level, cache_path):
    import tensorrt as trt
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):          # as the official code (no-op in TRT 10)
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if prec == "fp8":
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, log)
    with open(onnx_path, "rb") as fh:
        assert parser.parse(fh.read()), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    if prec == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = level
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WS_GIB << 30)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache = cfg.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    prof = builder.create_optimization_profile()
    prof.set_shape("dets", (1, m), (1, m), (1, m))
    cfg.add_optimization_profile(prof)
    t0 = time.perf_counter()
    ser = builder.build_serialized_network(net, cfg)
    t = time.perf_counter() - t0
    assert ser is not None, "TensorRT build failed"
    with open(cache_path + ".tmp", "wb") as fh:
        fh.write(memoryview(cfg.get_timing_cache().serialize()))
    os.replace(cache_path + ".tmp", cache_path)
    return bytes(ser), t, net.num_layers


def build_sweep(onnx_path, B, m, prec):
    """A plan of the studies' batch sweep: a static (B, m) profile, FP16 by BuilderFlag.FP16 or FP8 by a STRONGLY_TYPED
    network, every other builder setting TensorRT's default."""
    import tensorrt as trt
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network((1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if prec == "fp8" else 0)
    parser = trt.OnnxParser(net, log)
    assert parser.parse(open(onnx_path, "rb").read()), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    if prec == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    prof.set_shape(net.get_input(0).name, (B, m), (B, m), (B, m))
    cfg.add_optimization_profile(prof)
    t0 = time.perf_counter()
    ser = builder.build_serialized_network(net, cfg)
    t = time.perf_counter() - t0
    assert ser is not None, "TensorRT build failed"
    return bytes(ser), t, net.num_layers


def dtype_of(fmt):
    s = fmt.lower()
    for key, name in (("float8", "fp8"), ("fp8", "fp8"), ("e4m3", "fp8"), ("uint8", "uint8"), ("int8", "int8"),
                      ("bfloat16", "bf16"), ("bf16", "bf16"), ("half", "fp16"), ("fp16", "fp16"), ("bool", "bool"),
                      ("int32", "int32"), ("int64", "int64"), ("fp32", "fp32"), ("float", "fp32")):
        if key in s:
            return name
    return s or "?"


def layer_audit(ser):
    """The engine inspector's layer table (DETAILED): per layer type, tactic and the datatypes of its inputs and
    outputs; convolutions by input datatype; reformat layers."""
    import tensorrt as trt
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(ser)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    rows = []
    for l in info.get("Layers", []):
        if isinstance(l, str):
            rows.append(dict(name=l, type="", tactic="", ins=[], outs=[]))
            continue
        rows.append(dict(name=l.get("Name", ""), type=l.get("LayerType", ""), tactic=l.get("TacticName", ""),
                         ins=[dtype_of(i.get("Format/Datatype", "")) for i in l.get("Inputs", [])],
                         outs=[dtype_of(o.get("Format/Datatype", "")) for o in l.get("Outputs", [])],
                         fmt_in=[i.get("Format/Datatype", "") for i in l.get("Inputs", [])],
                         fmt_out=[o.get("Format/Datatype", "") for o in l.get("Outputs", [])]))
    conv = [r for r in rows if is_conv(r)]
    reformat = [r for r in rows if "reformat" in r["type"].lower() or "reformatting" in r["name"].lower()]
    by_in = {}
    for r in conv:                                   # convolutions by the precision they run (tactic, else input)
        k = runs(r)
        by_in[k] = by_in.get(k, 0) + 1
    return dict(layers=len(rows), conv_layers=len(conv), conv_by_input_dtype=by_in, reformat_layers=len(reformat),
                reformats=[dict(name=r["name"][:120], ins=r["ins"], outs=r["outs"]) for r in reformat],
                types={t: sum(1 for r in rows if r["type"] == t) for t in sorted({r["type"] for r in rows})},
                table=rows)


def cmd_build(a):
    import tensorrt as trt
    os.makedirs(os.path.join(a.work, "plans"), exist_ok=True)
    cache = os.path.join(a.work, "timing.cache")
    failed = []
    for d in a.d:
        m = d * (d * d - 1)
        todo = ([(f"{mdl}_{prec}_L{level}_B1", mdl, prec, level, 1) for mdl in a.models for prec in a.precisions
                 for level in a.levels] if not a.parts else parts(a, d))
        for part, mdl, prec, level, B in todo:
            f = names(a.onnx, a.work, mdl, d, prec, level, B)
            if os.path.exists(f["build"]) and os.path.exists(f["plan"]) and not a.force:
                print(f"{part} d{d}: built", flush=True)
                continue
            src = f["onnx8"] if prec == "fp8" else f["onnx16"]
            try:
                ser, t, nl = build_plan(src, m, prec, level, cache) if level else build_sweep(src, B, m, prec)
            except Exception as e:                               # noqa: BLE001  recorded, the other plans go on
                failed.append(f"{part} d{d}: {e!r}"[:300])
                print("BUILD FAILED", failed[-1], flush=True)
                continue
            with open(f["plan"], "wb") as fh:
                fh.write(ser)
            au = layer_audit(ser) if level else None                # sweep plans: no DETAILED layer table
            rep = dict(model=mdl, d=d, precision=prec, level=level, batch=B, part=part,
                       workspace_gib=WS_GIB if level else None, onnx=os.path.basename(src), onnx_sha256=sha256_file(src),
                       plan=f["plan"], plan_sha256=hashlib.sha256(ser).hexdigest(), plan_bytes=len(ser),
                       build_s=round(t, 1), network_layers=nl, timing_cache=cache if level else None,
                       trt=trt.__version__, versions=versions(), audit=au)
            json.dump(rep, open(f["build"], "w"), indent=1)
            print(f"{part} d{d}: {t:.0f} s, {len(ser) / 2**20:.1f} MiB" + (
                f", {au['layers']} engine layers, conv by input dtype {au['conv_by_input_dtype']}, "
                f"{au['reformat_layers']} reformats" if au else ""), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} builds failed: {failed}")


# ------------------------------------------------------------------------------------------------- run / check
def plan_ctx(path, m):
    import tensorrt as trt
    ser = open(path, "rb").read()
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(ser)
    assert eng is not None, f"cannot deserialize {path}"
    ctx = eng.create_execution_context()
    ctx.set_input_shape("dets", (1, m))
    x = torch.zeros((1, m), dtype=torch.uint8, device="cuda")
    out = torch.zeros((1, 1 + m), dtype=torch.uint8, device="cuda")
    ctx.set_tensor_address("dets", x.data_ptr())
    ctx.set_tensor_address(OUT_NAME, out.data_ptr())
    return eng, ctx, x, out, hashlib.sha256(ser).hexdigest()


def run_b1(ctx, x, out, det):
    """The B = 1 plan over every shot of det (n, m) uint8: (n, 1 + m) uint8."""
    dev = torch.from_numpy(det).cuda()
    res = torch.empty((len(det), out.shape[1]), dtype=torch.uint8, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    for i in range(len(det)):
        x.copy_(dev[i:i + 1])
        assert ctx.execute_async_v3(st)
        res[i].copy_(out[0])
    torch.cuda.synchronize()
    return res.cpu().numpy()


def reference(a, mdl, d, det, obs, plain):
    """The official numerics (FP32 weights, TF32 as NVIDIA's runner allows it) on validation chunk 0, on this GPU
    (cached)."""
    f = names(a.onnx, a.work, mdl, d)
    if os.path.exists(f["ref"]):
        z = np.load(f["ref"])
        return z["pre"], np.unpackbits(z["res"], axis=1)[:, :det.shape[1]], z["dec"]
    pipe, tf32, _ = opponents.ising_module(mdl, d, "cuda")
    pre, res = opponents.ising_predecode(pipe, tf32, det)
    del pipe
    torch.cuda.empty_cache()
    dec = pre ^ circuits.pm_decode(plain, res)
    np.savez_compressed(f["ref"], pre=pre, res=np.packbits(res, axis=1), dec=dec, obs=obs)
    return pre, res, dec


def cmd_check(a):
    bad = []
    for d in a.d:
        det, obs, _ = val0(d, a.val)
        m = det.shape[1]
        plain = circuits.matcher(d, False)
        for mdl in a.models:
            pre0, res0, dec0 = reference(a, mdl, d, det, obs, plain)
            f0 = dec0 != obs
            for prec in a.precisions:
                for level in a.levels:
                    f = names(a.onnx, a.work, mdl, d, prec, level)
                    if not os.path.exists(f["plan"]):
                        continue
                    try:
                        eng, ctx, x, out, sha = plan_ctx(f["plan"], m)   # keep the engine alive
                        t0 = time.time()
                        o = run_b1(ctx, x, out, det)
                    except Exception as e:                       # noqa: BLE001
                        bad.append(f"{mdl} d{d} {prec} L{level}: {e!r}"[:300])
                        print("CHECK FAILED", bad[-1], flush=True)
                        continue
                    t1 = time.time()
                    pre, res = o[:, 0], o[:, 1:]
                    dec = pre ^ circuits.pm_decode(plain, res)
                    fe = dec != obs
                    p = stats.paired(fe, f0)
                    agree = float((dec == dec0).mean())
                    dens, dens0 = float(res.mean()), float(res0.mean())
                    bld = json.load(open(f["build"])) if os.path.exists(f["build"]) else {}
                    au = bld.get("audit") or {}
                    need = ACCEPT["agree_fp16"] if prec == "fp16" else ACCEPT["agree_fp8"]
                    fp8 = None                   # FP8 execution: every Conv the recipe quantized runs e4m3
                    if prec == "fp8":
                        q = json.load(open(f["prep"]))["qdq"]
                        fp8 = fp8_verdict(q, au) if q and au else dict(ok=False, rows=[], note="no Q/DQ or layer audit")
                    ok = (agree >= need and p["verdict"] != "worse" and abs(dens / dens0 - 1) <= ACCEPT["density_rel"]
                          and (fp8 is None or fp8["ok"]) and bld.get("plan_sha256") == sha)
                    rep = dict(model=mdl, d=d, precision=prec, level=level, plan_sha256=sha, shots=len(det),
                               plan_equal_build=bld.get("plan_sha256") == sha,
                               decision_agreement=agree, pre_agreement=float((pre == pre0).mean()),
                               residual_bit_mismatch=float((res != res0).mean()), residual_density=dens,
                               residual_density_fp32=dens0, failures=int(fe.sum()), failures_fp32=int(f0.sum()),
                               paired_vs_fp32=p, fp8=fp8, conv_by_input_dtype=au.get("conv_by_input_dtype"),
                               reformat_layers=au.get("reformat_layers"), run_s=round(t1 - t0, 1),
                               test="PASS" if ok else "FAIL", accept=ACCEPT)
                    json.dump(rep, open(f["check"], "w"), indent=1)
                    print(f"{mdl} d{d} {prec} L{level}: agree {agree:.5f}, failures {rep['failures']} vs fp32 "
                          f"{rep['failures_fp32']} ({p['only_a']} / {p['only_b']}, p {p['mcnemar_p']:.2f}), density "
                          f"{dens:.5f} vs {dens0:.5f}, convs {au.get('conv_by_input_dtype')}: {rep['test']}"
                          + (f", FP8 execution {'confirmed' if fp8['ok'] else 'NOT confirmed'}" if fp8 else ""),
                          flush=True)
                    if not ok:
                        bad.append(f"{mdl} d{d} {prec} L{level}")
    if bad:
        print("FAIL:", bad, flush=True)
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------- time
def graph_reps(fn, warm=200, iters=100, reps=5):
    """The studies' protocol (NVIDIA's trtexec one): one CUDA graph of the call, 200 warm-up replays, 5 x 100 timed
    replays (CUDA events);
    returns the five means, µs per call."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms.append(e0.elapsed_time(e1) / iters * 1e3)
    return ms


def plan_list(a, d):
    out = []
    for mdl in a.models:
        for prec in a.precisions:
            for level in a.levels:
                f = names(a.onnx, a.work, mdl, d, prec, level)
                if os.path.exists(f["plan"]):
                    out.append((mdl, prec, level, f))
    return out


def cmd_time(a):
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    fields = ["d", "model", "precision", "level", "order", "position", "us_call_reps", "us_call_median", "us_call_min",
              "us_call_max", "us_per_round_median", "check", "plan_sha256", "gpu", "trt", "t"]
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    import tensorrt as trt
    gpu = torch.cuda.get_device_name()
    for d in a.d:
        m = d * (d * d - 1)
        det, _, _ = val0(d, a.val)
        todo = plan_list(a, d)
        if a.order == "rev":
            todo = todo[::-1]
        for pos, (mdl, prec, level, f) in enumerate(todo):
            chk = json.load(open(f["check"])) if os.path.exists(f["check"]) else {"test": "unchecked"}
            if chk["test"] != "PASS" or (prec == "fp8" and not (chk.get("fp8") or {}).get("ok")):
                why = f"check {chk['test']}" if chk["test"] != "PASS" else "FP8 execution not confirmed"
                print(f"d{d} {mdl} {prec} L{level}: NOT TIMED ({why})", flush=True)
                continue
            try:
                eng, ctx, x, out, sha = plan_ctx(f["plan"], m)   # keep the engine alive
                if sha != chk["plan_sha256"]:
                    print(f"d{d} {mdl} {prec} L{level}: NOT TIMED (the plan is not the checked one)", flush=True)
                    del ctx, eng
                    continue
                x.copy_(torch.from_numpy(det[:1]).cuda())
                st = torch.cuda.current_stream()
                ms = graph_reps(lambda: ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream))
            except Exception as e:                               # noqa: BLE001
                print(f"TIME FAILED d{d} {mdl} {prec} L{level}: {e!r}"[:300], flush=True)
                continue
            chk = chk["test"]
            r = dict(d=d, model=mdl, precision=prec, level=level, order=a.order, position=pos,
                     us_call_reps=json.dumps([round(v, 3) for v in ms]), us_call_median=float(np.median(ms)),
                     us_call_min=float(min(ms)), us_call_max=float(max(ms)),
                     us_per_round_median=float(np.median(ms)) / d, check=chk, plan_sha256=sha, gpu=gpu,
                     trt=trt.__version__, t=time.strftime("%H:%M:%S"))
            w.writerow(r)
            fh.flush()
            print(f"d{d} {mdl} {prec} L{level} ({a.order} #{pos}): {r['us_call_median']:.2f} us "
                  f"[{r['us_call_min']:.2f}, {r['us_call_max']:.2f}] ({chk})", flush=True)
            del ctx, st, eng
    fh.close()


# ------------------------------------------------------------------------------------------------- eval (test sets)
def run_batched(ctx, x, out, det):
    """The plan over every shot of det at its batch B = len(x): exactly B shots per call, the last call padded with
    all-zero shots; -> (n, 1 + m) uint8."""
    B, n = x.shape[0], len(det)
    xs = torch.zeros((-(-n // B) * B, det.shape[1]), dtype=torch.uint8, device="cuda")
    xs[:n] = torch.from_numpy(det).cuda()
    res = torch.empty((len(xs), out.shape[1]), dtype=torch.uint8, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    for i in range(0, len(xs), B):
        x.copy_(xs[i:i + B])
        assert ctx.execute_async_v3(st)
        res[i:i + B].copy_(out)
    torch.cuda.synchronize()
    return res[:n].cpu().numpy()


def cmd_eval(a):
    import tensorrt as trt
    os.makedirs(a.out, exist_ok=True)
    for d in a.d:
        m = d * (d * d - 1)
        plain = circuits.matcher(d, False)
        corr = circuits.matcher(d, True) if a.corr else None
        chunks = testset.parse_range(a.chunks)
        rep = {}
        for part, mdl, prec, level, B in parts(a, d):
            f = names(a.onnx, a.work, mdl, d, prec, level, B)
            ser = open(f["plan"], "rb").read()
            eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(ser)
            ctx = eng.create_execution_context()
            ctx.set_input_shape("dets", (B, m))
            x = torch.zeros((B, m), dtype=torch.uint8, device="cuda")
            out = torch.zeros((B, 1 + m), dtype=torch.uint8, device="cuda")
            ctx.set_tensor_address("dets", x.data_ptr())
            ctx.set_tensor_address(OUT_NAME, out.data_ptr())
            decs, fails = {part: []}, {part: []}
            if corr is not None:
                decs[part + "_corr"], fails[part + "_corr"] = [], []
            for j in chunks:
                det, obs = testset.load_chunks(a.set, d, [j], a.shots)
                o = run_batched(ctx, x, out, det)
                decs[part].append(o[:, 0] ^ circuits.pm_decode(plain, o[:, 1:]))
                if corr is not None:                       # the same residual detectors, correlated PyMatching
                    decs[part + "_corr"].append(o[:, 0] ^ circuits.pm_decode(corr, o[:, 1:], correlated=True))
                for k in decs:
                    fails[k].append(int((decs[k][-1] != obs).sum()))
            del ctx, eng
            for k in decs:
                rep[k] = dict(plan=os.path.basename(f["plan"]), plan_sha256=hashlib.sha256(ser).hexdigest(), batch=B,
                              set=a.set, chunks=a.chunks, shots=len(chunks) * testset.CHUNK, failures=sum(fails[k]),
                              failures_per_chunk=fails[k])
                np.savez_compressed(os.path.join(a.out, f"{k}_d{d}.npz"), decision=np.concatenate(decs[k]).astype(np.uint8))
                print(f"{k} d{d}: {sum(fails[k])} failures on {a.set} chunks {a.chunks} (plan {rep[k]['plan_sha256'][:12]})",
                      flush=True)
        json.dump(rep, open(os.path.join(a.out, f"eval_d{d}.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------- FP8 execution audit


def is_conv(r):
    t, n, tac = r.get("type", "").lower(), r.get("name", "").lower(), r.get("tactic", "").lower()
    return "conv" in t or "correlation" in t or "/conv" in n or "fprop" in tac


def norm(name):
    return re.sub(r"_myl\d+_\d+$", "", name).replace(".", "_")


def runs(r):
    """The precision a convolution layer runs, from its tactic name first (e4m3 / f16 / f32 / i8), else its inputs."""
    tac = r.get("tactic", "").lower()
    m = re.search(r"fprop_[a-z_]*?(e4m3|f16|bf16|f32|tf32|i8)", tac)
    if "e4m3" in tac:
        return "fp8"
    if m:
        return {"f16": "fp16", "bf16": "bf16", "f32": "fp32", "tf32": "tf32", "i8": "int8"}[m.group(1)]
    ins = r.get("ins") or []
    return ins[0] if ins else "?"


def conv_table(audit, qdq=None):
    """One row per ONNX Conv (from the Q/DQ audit when given, else per engine conv layer): quantized by the recipe,
    the engine layer, its tactic and the precision it runs."""
    eng = {norm(r["name"]): r for r in audit.get("table", []) if is_conv(r)}
    if qdq is None:
        return [dict(conv=k, quantized=None, layer=r["name"], tactic=r.get("tactic", "")[:90], runs=runs(r))
                for k, r in eng.items()]
    rows = []
    for c in qdq["conv_feeds"]:
        q = c["x"].startswith("Q/DQ FLOAT8E4M3FN") and "FLOAT8E4M3FN" in c["w"]
        e = eng.get(norm(c["conv"]))
        rows.append(dict(conv=c["conv"], quantized=q, feed_x=c["x"], layer=e["name"] if e else None,
                         tactic=e.get("tactic", "")[:90] if e else None, runs=runs(e) if e else "missing"))
    return rows


def fp8_verdict(qdq, audit):
    rows = conv_table(audit, qdq)
    quant = [r for r in rows if r["quantized"]]
    ok = bool(quant) and all(r["runs"] == "fp8" for r in quant)
    return dict(ok=ok, convs=len(rows), quantized=len(quant), quantized_fp8=sum(r["runs"] == "fp8" for r in quant),
                unquantized_runs=[r["runs"] for r in rows if not r["quantized"]], rows=rows)




def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "build", "check", "time", "eval"):
        s = sub.add_parser(c)
        s.add_argument("--d", type=int, nargs="+", required=True)
        s.add_argument("--models", nargs="+", default=list(MODELS))
        s.add_argument("--precisions", nargs="+", default=list(PRECS))
        s.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
        s.add_argument("--onnx", required=True, help="the ONNX directory")
        s.add_argument("--work", default="", help="the target's work directory (plans, timing cache, reports)")
        s.add_argument("--val", default="", help="the directory with validation chunk 0 of each d")
        s.add_argument("--force", action="store_true")
        s.add_argument("--device", default="cuda", help="prep only: cpu for a dry run without a GPU")
        if c in ("build", "eval"):
            s.add_argument("--parts", nargs="+", default=[] if c == "build" else ["registered"],
                           help="part names such as M4_fp16_B64 or M1_fp8_L3_B1, or registered")
        if c == "time":
            s.add_argument("--order", choices=("fwd", "rev"), default="fwd")
        if c in ("time", "eval"):
            s.add_argument("--out", required=True, help="time: the CSV; eval: a directory")
        if c == "eval":
            s.add_argument("--set", required=True, choices=sorted(testset.SETS))
            s.add_argument("--shots", required=True, help="the directory with the test-set chunk files")
            s.add_argument("--chunks", required=True)
            s.add_argument("--corr", action="store_true", help="also correlated PM on the residual detectors (_corr)")
    a = ap.parse_args()
    dict(prep=cmd_prep, build=cmd_build, check=cmd_check, time=cmd_time, eval=cmd_eval)[a.cmd](a)


if __name__ == "__main__":
    main()
