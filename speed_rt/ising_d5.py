"""D5 and review findings 2 and 3: NVIDIA's own build and NVIDIA's own timing command as further Ising candidates, and the
residual detectors of the plan whose time counts.

THE EXPERIMENT'S FROZEN SETTINGS (SPEED_RT.md, "Ising"; one pod role = one explicit list of distances, S-low d = 7 9 11 and
S-high d = 13 15 17; every list is validated against ALLOWED_D, sorted ascending and built in that order):
  plans per d    12 = 8 registered (ising_trt build: M1, M4 x FP16, FP8 x builder optimisation level 3, 5) + 4 official (this
                 module: M1, M4 x FP16, FP8). Build order per d: registered plans M1 FP16 L3, M1 FP16 L5, M1 FP8 L3, M1 FP8
                 L5, M4 ... (ising_trt cmd_build), then the official plans M1 FP16, M1 FP8, M4 FP16, M4 FP8; d ascending.
  ONNX           the same export for both builds (surface_v2/timing/ising_trt.py prep: opset 18, dynamic batch, constant
                 folding, dynamo off, traced on random bits of seed 0); FP8 by modelopt, op_types_to_quantize ["Conv"],
                 high_precision_dtype "fp16", calibration rows = validation chunk 0 shots 0-255 (256 rows, sha256 in the
                 prep record; never a timing shot).
  registered     fixed (1, m) profile, 4 GiB workspace, optimisation level 3 and 5, DETAILED verbosity, and ONE persistent
                 timing cache per work directory (WORK/timing.cache: empty on a fresh pod, loaded before and saved after
                 every registered build, ignore_mismatch False, shared by every registered plan of the pod in the build order
                 above; ising_trt.build_plan). The official builds never touch it.
  official       NVIDIA's build as Ising-Decoding 3ce4379 writes it (code/evaluation/logical_error_rate.py, ONNX_WORKFLOW=2):
                 EXPLICIT_BATCH (and STRONGLY_TYPED for FP8) network flags; BuilderFlag.FP16 for the FP16 plan on the FP32
                 ONNX (no builder precision flag for FP8); ONE optimisation profile min (1, m), opt (B, m), max (B, m) with
                 B = OFFICIAL_BATCH = 2048, NVIDIA's test batch; EVERY OTHER SETTING IS TENSORRT'S DEFAULT, stated: workspace
                 = TensorRT's default memory-pool limit (no limit set; the build record keeps the value the API reports),
                 optimisation level = the default (3; recorded), TIMING CACHE = none: no cache is created, loaded, saved or
                 passed, every plan has its own builder and builder config, so nothing carries over from one build to the
                 next. One departure, recorded: profiling verbosity DETAILED, so the plan keeps its layer and tactic names
                 for the FP8 audit; it does not change the tactic choice.
  trtexec        see below; plans, inputs, shapes, streams, warm-up, duration and statistic are fixed there.

official  NVIDIA's build applied to the ONNX files of surface_v2/timing/ising_trt.py prep. Plans:
          WORK/plans/{M}_d{d}_{prec}_official.plan with _build.json (every setting above, the default workspace and level
          the API reports, the calibration record, build_s).
check     ising_trt's acceptance (cmd_check, per plan, unchanged rules and reference) on an official plan, on the VALIDATION
          chunk 0 (--view = the validation directory, as ising_trt check's --val; no timing shot) -> _check.json. Exit 1
          when a plan fails (as ising_trt check); a plan that fails is never timed.
input     the first shot of the timing set's chunk 0 (--view = the timing view) as trtexec's input file, uint8 (1, m).
trtexec   NVIDIA's timing command for one validated plan, one repetition: trtexec --loadEngine PLAN --useCudaGraph
          --noDataTransfers --useSpinWait --loadInputs dets:INPUT (never trtexec's random inputs), --shapes dets:1xm for a
          dynamic plan, trtexec's defaults otherwise (one stream, --warmUp 200 ms, --duration 3 s, --iterations 10); the
          repetition's value is the reported GPU Compute Time mean (never the minimum single inference). It runs inside a
          speed_rt.watch window (rule floor, from trtexec's "Starting inference" line; its verdict: the clock rule and no other compute process, the host-side shares are recorded only). OUT is created when missing.
residuals the residual detectors of the timing set's chunk 0 by the plan whose GPU time counts (speed_rt/ising_select.py's
          rule), at batch 1, -> RES/res_d{d}_{M}_val000.npz (pre_L, res, obs), the file `timing.cpu_term share` reads.

    python -m speed_rt.ising_d5 official --d 7 9 11 --onnx ONNX --work WORK
    python -m speed_rt.ising_d5 check --d 7 9 11 --view VALIDATION_DIR --onnx ONNX --work WORK
    python -m speed_rt.ising_d5 input --d 7 9 11 --view TIMING_VIEW --out INPUTS
    python -m speed_rt.ising_d5 trtexec --plan PLAN --d 13 --input INPUTS/dets_d13.bin --rep 1 --core C --out DIR
    python -m speed_rt.ising_d5 residuals --d 13 --model M1 --plan PLAN --view TIMING_VIEW --out RES"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

from speed_rt.ising_select import ALLOWED_D
from timing import ising_trt as IT

# NVIDIA's public workflow (Ising-Decoding 3ce4379): conf/config_public.yaml leaves test.dataloader to the internal
# defaults of code/workflows/config_validator.py, batch_size 2048, and code/evaluation/logical_error_rate.py builds
# the profile (1, m) / (B, m) / (B, m) with B = that batch size
OFFICIAL_BATCH = 2048
TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")
TRTEXEC_ARGS = ["--useCudaGraph", "--noDataTransfers", "--useSpinWait"]


def dlist(ds):
    """The explicit distance list of a command: every d one of ALLOWED_D, no repeats, ascending (the build order)."""
    bad = [d for d in ds if d not in ALLOWED_D]
    assert not bad, f"distances {bad} are not among {ALLOWED_D}"
    assert len(set(ds)) == len(ds), f"repeated distance in {ds}"
    return sorted(ds)


def official_names(work, mdl, d, prec):
    stem = os.path.join(work, "plans", f"{mdl}_d{d}_{prec}_official")
    return dict(plan=stem + ".plan", build=stem + "_build.json", check=stem + "_check.json")


def build_official(onnx_path, m, prec):
    import tensorrt as trt
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
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
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    prof = builder.create_optimization_profile()
    prof.set_shape("dets", (1, m), (OFFICIAL_BATCH, m), (OFFICIAL_BATCH, m))
    cfg.add_optimization_profile(prof)
    info = dict(network_flags=["EXPLICIT_BATCH"] + (["STRONGLY_TYPED"] if prec == "fp8" else []),
                builder_flags=["FP16"] if prec == "fp16" else [],
                workspace_bytes_default=int(cfg.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)),
                builder_optimization_level_default=int(cfg.builder_optimization_level),
                timing_cache_object=cfg.get_timing_cache() is not None,
                disable_timing_cache_flag=bool(cfg.get_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)))
    t0 = time.perf_counter()
    ser = builder.build_serialized_network(net, cfg)
    t = time.perf_counter() - t0
    assert ser is not None, "TensorRT build failed"
    return bytes(ser), t, net.num_layers, info


def cmd_official(a):
    import tensorrt as trt
    os.makedirs(os.path.join(a.work, "plans"), exist_ok=True)
    failed, built, t_all = [], 0, time.perf_counter()
    for d in dlist(a.d):
        m = d * (d * d - 1)
        for mdl in IT.MODELS:
            f0 = IT.names(a.onnx, a.work, mdl, d)
            for prec in IT.PRECS:
                f = official_names(a.work, mdl, d, prec)
                if os.path.exists(f["build"]) and os.path.exists(f["plan"]):
                    print(f"{mdl} {prec} official d{d}: built", flush=True)
                    continue
                src = f0["onnx8"] if prec == "fp8" else f0["onnx16"]
                try:
                    ser, t, nl, info = build_official(src, m, prec)
                    prep = json.load(open(f0["prep"]))
                except Exception as e:                           # noqa: BLE001  recorded, the other plans go on
                    failed.append(f"{mdl} {prec} official d{d}: {e!r}"[:300])
                    print("BUILD FAILED", failed[-1], flush=True)
                    continue
                with open(f["plan"], "wb") as fh:
                    fh.write(ser)
                au = IT.layer_audit(ser)
                rep = dict(model=mdl, d=d, precision=prec, build="official", profile=[[1, m], [OFFICIAL_BATCH, m],
                           [OFFICIAL_BATCH, m]], workspace_gib=None, level=None, timing_cache=None,
                           timing_cache_policy="none: no cache created, loaded, saved or passed; own builder and config",
                           profiling_verbosity="DETAILED", onnx=os.path.basename(src), onnx_sha256=IT.sha256_file(src),
                           calibration=prep["calib"] if prec == "fp8" else None, **info,
                           plan=f["plan"], plan_sha256=hashlib.sha256(ser).hexdigest(), plan_bytes=len(ser),
                           build_s=round(t, 1), network_layers=nl, trt=trt.__version__, versions=IT.versions(), audit=au)
                json.dump(rep, open(f["build"], "w"), indent=1)
                built += 1
                print(f"{mdl} {prec} official d{d}: {t:.0f} s, {len(ser) / 2**20:.1f} MiB, convs {au['conv_by_input_dtype']}, "
                      f"workspace default {info['workspace_bytes_default'] / 2**30:.1f} GiB, level "
                      f"{info['builder_optimization_level_default']}, cache object {info['timing_cache_object']}", flush=True)
    print(f"OFFICIAL DONE d {sorted(a.d)}: {built} built, {len(failed)} failed, wall {time.perf_counter() - t_all:.0f} s",
          flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} builds failed: {failed}")


def check_plan(a, mdl, d, prec, f, det, obs, plain, ref):
    """ising_trt.cmd_check's per-plan body, unchanged rules, for one plan file set f."""
    from convlines import circuits, stats
    pre0, res0, dec0 = ref
    f0 = dec0 != obs
    m = det.shape[1]
    eng, ctx, x, out, sha = IT.plan_ctx(f["plan"], m)
    t0 = time.time()
    o = IT.run_b1(ctx, x, out, det)
    t1 = time.time()
    pre, res = o[:, 0], o[:, 1:]
    dec = pre ^ circuits.pm_decode(plain, res)
    fe = dec != obs
    p = stats.paired(fe, f0)
    agree = float((dec == dec0).mean())
    dens, dens0 = float(res.mean()), float(res0.mean())
    bld = json.load(open(f["build"])) if os.path.exists(f["build"]) else {}
    au = bld.get("audit") or {}
    need = IT.ACCEPT["agree_fp16"] if prec == "fp16" else IT.ACCEPT["agree_fp8"]
    fp8 = None
    if prec == "fp8":
        q = json.load(open(IT.names(a.onnx, a.work, mdl, d)["prep"]))["qdq"]
        fp8 = IT.fp8_verdict(q, au) if q and au else dict(ok=False, rows=[], note="no Q/DQ or layer audit")
    ok = (agree >= need and p["verdict"] != "worse" and abs(dens / dens0 - 1) <= IT.ACCEPT["density_rel"]
          and (fp8 is None or fp8["ok"]) and bld.get("plan_sha256") == sha)
    rep = dict(model=mdl, d=d, precision=prec, build="official", plan_sha256=sha, shots=len(det),
               plan_equal_build=bld.get("plan_sha256") == sha, decision_agreement=agree,
               pre_agreement=float((pre == pre0).mean()), residual_bit_mismatch=float((res != res0).mean()),
               residual_density=dens, residual_density_fp32=dens0, failures=int(fe.sum()), failures_fp32=int(f0.sum()),
               paired_vs_fp32=p, fp8=fp8, conv_by_input_dtype=au.get("conv_by_input_dtype"),
               reformat_layers=au.get("reformat_layers"), run_s=round(t1 - t0, 1), test="PASS" if ok else "FAIL",
               accept=IT.ACCEPT)
    json.dump(rep, open(f["check"], "w"), indent=1)
    del ctx, eng
    print(f"{mdl} d{d} {prec} official: agree {agree:.5f}, failures {rep['failures']} vs fp32 {rep['failures_fp32']}, "
          f"density {dens:.5f} vs {dens0:.5f}: {rep['test']}" + (f", FP8 {'confirmed' if fp8['ok'] else 'NOT confirmed'}"
                                                                   if fp8 else ""), flush=True)
    return ok


def cmd_check(a):
    from convlines import circuits
    bad, n, passed, t_all = [], 0, 0, time.perf_counter()
    a.val = a.view                                               # ising_trt.reference's cache lives in WORK
    for d in dlist(a.d):
        det, obs, _ = IT.val0(d, a.view)
        plain = circuits.matcher(d, False)
        for mdl in IT.MODELS:
            ref = IT.reference(a, mdl, d, det, obs, plain)
            for prec in IT.PRECS:
                f = official_names(a.work, mdl, d, prec)
                if not os.path.exists(f["plan"]):
                    bad.append(f"{mdl} d{d} {prec} official: no plan")
                    continue
                n += 1
                try:
                    ok = check_plan(a, mdl, d, prec, f, det, obs, plain, ref)
                except Exception as e:                           # noqa: BLE001
                    ok = False
                    print(f"CHECK FAILED {mdl} d{d} {prec} official: {e!r}"[:300], flush=True)
                passed += bool(ok)
                if not ok:
                    bad.append(f"{mdl} d{d} {prec} official")
    print(f"OFFICIAL CHECK DONE d {sorted(a.d)}: {passed} of {n} plans PASS, wall {time.perf_counter() - t_all:.0f} s",
          flush=True)
    if bad:
        print("FAIL:", bad, flush=True)
        raise SystemExit(1)


def cmd_input(a):
    os.makedirs(a.out, exist_ok=True)
    for d in dlist(a.d):
        det, _, _ = IT.val0(d, a.view)
        f = os.path.join(a.out, f"dets_d{d}.bin")
        raw = np.ascontiguousarray(det[:1], dtype=np.uint8).tobytes()
        if os.path.exists(f):                                    # written once; the same bytes again is a no-op
            assert open(f, "rb").read() == raw, f"{f} exists with other bytes"
        else:
            with open(f, "xb") as fh:
                fh.write(raw)
        print(f"d{d}: the first shot of the timing set's chunk 0, {det.shape[1]} bytes -> {f}")


OPTS = re.compile(r"\] (Iterations|Duration|Inference Streams|Data transfers|Spin-wait|CUDA Graph|Multithreading|Batch): (.*)")
LOADED = re.compile(r"\] (dets<-.*)")
STAT = re.compile(r"GPU Compute Time: min = ([0-9.]+) ms, max = ([0-9.]+) ms, mean = ([0-9.]+) ms, median = ([0-9.]+) ms")


def cmd_trtexec(a):
    from speed_rt import watch
    assert a.core >= 0, "--core C (the session's timed core) is required"
    os.makedirs(a.out, exist_ok=True)                            # dry 2c: the missing OUT directory stopped the step
    out = os.path.join(a.out, f"trtexec_{os.path.basename(a.plan)[:-5]}_rep{a.rep}.json")
    assert not os.path.exists(out), f"{out} exists"
    m = a.d * (a.d * a.d - 1)
    bld = json.load(open(a.plan[:-5] + "_build.json"))
    chk = json.load(open(a.plan[:-5] + "_check.json"))
    sha = IT.sha256_file(a.plan)
    assert chk["test"] == "PASS" and chk["plan_sha256"] == sha, f"{a.plan}: not the checked plan"
    cmd = [TRTEXEC, f"--loadEngine={a.plan}"] + TRTEXEC_ARGS + [f"--loadInputs=dets:{a.input}"]
    if bld.get("build") == "official":
        cmd.append(f"--shapes=dets:1x{m}")
    log = os.path.join(a.out, os.path.basename(out)[:-5] + ".log")
    w = argparse.Namespace(core=a.core, rule="floor", out=out + ".watch", marker="Starting inference")
    sys.stdout.flush()
    import contextlib
    with open(log, "x") as fh, contextlib.redirect_stdout(fh):
        rec = watch.run(w, cmd)
    text = open(log).read()
    mt = STAT.search(text)
    r = dict(item="G5-trtexec", plan=os.path.basename(a.plan), plan_sha256=sha, d=a.d, model=bld["model"],
             precision=bld["precision"], build=bld.get("build", "registered"), level=bld.get("level"), rep=a.rep,
             cmd=cmd, input_sha256=IT.sha256_file(a.input), exit=rec["exit"],
             gpu_compute_ms=dict(zip(("min", "max", "mean", "median"), map(float, mt.groups()))) if mt else None,
             us_call_mean=float(mt.group(3)) * 1e3 if mt else None, window=rec,
             trtexec_settings={k: v.strip() for k, v in OPTS.findall(text)}, loaded_inputs=LOADED.findall(text),
             accepted=bool(rec["clocks"]["ok"] and rec["gpu_alone"] and rec["exit"] == 0 and rec["marker_seen"] and mt is not None),
             log=os.path.basename(log))
    os.remove(out + ".watch")
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1)
    print(f"trtexec {r['plan']} rep {a.rep}: GPU compute mean {r['us_call_mean']} us; accepted {r['accepted']}", flush=True)


def cmd_residuals(a):
    from convlines import testset
    det, obs, _ = IT.val0(a.d, a.view)
    m = det.shape[1]
    sha = IT.sha256_file(a.plan)
    chk = json.load(open(a.plan[:-5] + "_check.json"))
    assert chk["test"] == "PASS" and chk["plan_sha256"] == sha, f"{a.plan}: not the checked plan"
    eng, ctx, x, out, _ = IT.plan_ctx(a.plan, m)
    o = IT.run_b1(ctx, x, out, det)
    pre, res = o[:, 0], np.ascontiguousarray(o[:, 1:])
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"res_d{a.d}_{a.model}_val000.npz")
    testset.write_npz(f, dict(pre_L=pre, res=res, obs=obs))
    rec = dict(d=a.d, model=a.model, plan=os.path.basename(a.plan), plan_sha256=sha, shots=len(det),
               res_content_sha256=hashlib.sha256(res.tobytes()).hexdigest(),
               pre_content_sha256=hashlib.sha256(pre.tobytes()).hexdigest(), file_sha256=IT.sha256_file(f),
               residual_density=float(res.mean()))
    json.dump(rec, open(f[:-4] + ".json", "x"), indent=1)
    print(f"d{a.d} {a.model}: residuals of {rec['plan']} -> {f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("official", "check", "input", "trtexec", "residuals"))
    ap.add_argument("--d", type=int, nargs="+", required=True)
    ap.add_argument("--onnx", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--view", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--model", choices=IT.MODELS)
    ap.add_argument("--input", default="")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--core", type=int, default=-1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.d = dlist(a.d)
    if a.what in ("trtexec", "residuals"):
        assert len(a.d) == 1, "one d"
        a.d = a.d[0]
    {"official": cmd_official, "check": cmd_check, "input": cmd_input, "trtexec": cmd_trtexec,
     "residuals": cmd_residuals}[a.what](a)


if __name__ == "__main__":
    main()
