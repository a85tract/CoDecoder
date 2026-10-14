"""G5's timing of Ising's TensorRT plans in one fresh process per (d, order): surface_v2/timing/ising_trt.py's loop
(`time`: per plan one CUDA graph of the plan's batch-1 call on the first shot of chunk 0, 200 warm-up replays, then 5 x
100 timed replays between CUDA events, `graph_reps`, unchanged), over every candidate plan of d whose check passed: the
eight registered plans (M1, M4 x FP16, FP8 x optimisation levels 3, 5; ising_trt build/check) and D5's four official
plans (speed_rt/ising_d5.py official/check), in the frozen order of PLANS (order "rev": reversed).

Around the loop, the clock and host checks of speed_rt/watch.py (review findings 4 and 5): nvidia-smi samples every 100 ms
for the whole process, and each plan's window [start, end] of its graph_reps call is judged on the samples inside it
and the nearest sample on each side (a window of 5 x 100 replays lasts tens of ms): every such sample at least 97 % of
the maximum SM clock, throttle reasons none or idle; no other compute process on the GPU before or after (the host-side
shares of the timed core and its siblings are recorded over the whole process and decide nothing: the pods time GPU work only). A plan whose window fails
is reported with accepted false; its values do not count.

The plans it times are the ones its check passed (the 8 registered and the 4 official plans of d, in PLANS order, whose builds
speed_rt/ising_d5.py's header freezes: profiles, workspace, precision flags, FP8 calibration rows, build count and order, and
the timing-cache policy of each kind: one persistent WORK/timing.cache for the registered plans, none for the official ones).
One process times one d in one order; a pod role runs it once per d of its explicit list and per order (round 1 fwd, round 2
rev). A plan that is not built, not checked, or whose check failed is never timed: its row says why and the others go on.
    python -m speed_rt.ising_time --d 13 --order fwd --view TIMING_VIEW --onnx ONNX --work WORK --core C --out DIR
-> DIR/ising_time_d13_fwd.json (written once; wall_s = the process's wall time inside main)."""
import argparse
import json
import os
import time

import numpy as np
import torch

from timing import ising_trt as IT

PLANS = [(mdl, prec, kind) for mdl in IT.MODELS for prec in IT.PRECS for kind in ("L3", "L5", "official")]


def plan_files(onnx, work, mdl, d, prec, kind):
    if kind == "official":
        from speed_rt.ising_d5 import official_names
        return official_names(work, mdl, d, prec)
    return IT.names(onnx, work, mdl, d, prec, int(kind[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--order", choices=("fwd", "rev"), required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--core", type=int, required=True)
    ap.add_argument("--rep", type=int, default=0, help="the session's round (recorded)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from speed_rt.ising_select import ALLOWED_D
    assert a.d in ALLOWED_D, f"d = {a.d} is not among {ALLOWED_D}"
    t_main = time.time()
    out = os.path.join(a.out, f"ising_time_d{a.d}_{a.order}.json")
    assert not os.path.exists(out), f"{out} exists"
    os.sched_setaffinity(0, {a.core})
    torch.set_num_threads(1)
    import tensorrt as trt
    from speed_rt import watch
    m = a.d * (a.d * a.d - 1)
    det, _, _ = IT.val0(a.d, a.view)
    todo = PLANS if a.order == "fwd" else PLANS[::-1]
    apps0 = watch.smi_apps()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    new = [x for x in watch.smi_apps() if x not in apps0]
    own = {str(os.getpid())} | ({new[0].split(",")[0].strip()} if len(new) == 1 else set())
    w = watch.Window(a.core, "floor")
    rows = []
    for pos, (mdl, prec, kind) in enumerate(todo):
        f = plan_files(a.onnx, a.work, mdl, a.d, prec, kind)
        row = dict(d=a.d, model=mdl, precision=prec, kind=kind, order=a.order, position=pos, plan=os.path.basename(f["plan"]))
        chk = json.load(open(f["check"])) if os.path.exists(f["check"]) else {"test": "unchecked"}
        if chk["test"] != "PASS" or (prec == "fp8" and not (chk.get("fp8") or {}).get("ok")):
            row.update(timed=False, why=f"check {chk['test']}" if chk["test"] != "PASS" else "FP8 execution not confirmed")
            rows.append(row)
            print(f"d{a.d} {mdl} {prec} {kind}: NOT TIMED ({row['why']})", flush=True)
            continue
        try:
            eng, ctx, x, o, sha = IT.plan_ctx(f["plan"], m)
            if sha != chk["plan_sha256"]:
                row.update(timed=False, why="the plan is not the checked one")
                rows.append(row)
                continue
            x.copy_(torch.from_numpy(det[:1]).cuda())
            t0 = time.time()
            ms = IT.graph_reps(lambda: ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream))
            t1 = time.time()
            del ctx, eng
        except Exception as e:                                   # noqa: BLE001
            row.update(timed=False, why=f"TIME FAILED {e!r}"[:300])
            rows.append(row)
            print(f"d{a.d} {mdl} {prec} {kind}: {row['why']}", flush=True)
            continue
        row.update(timed=True, plan_sha256=sha, us_call_reps=[round(v, 4) for v in ms], us_call_min=float(min(ms)),
                   us_call_median=float(np.median(ms)), t0=t0, t1=t1)
        rows.append(row)
        print(f"d{a.d} {mdl} {prec} {kind} ({a.order} #{pos}): reps {row['us_call_reps']} us", flush=True)
    win = w.stop(own_pids=own)
    samples = w.clocks.rows
    for row in rows:
        if not row.get("timed"):
            continue
        inside = [s for s in samples if row["t0"] <= s["t"] <= row["t1"]]
        before = [s for s in samples if s["t"] < row["t0"]][-1:]
        after = [s for s in samples if s["t"] > row["t1"]][:1]
        near = before + inside + after
        frac = [s["sm"] / s["sm_max"] for s in near]
        thr = sorted({s["throttle"] for s in near})
        row["clock"] = dict(samples=len(near), inside=len(inside), sm_min_frac=min(frac) if frac else None, throttle=thr)
        row["accepted"] = bool(frac and min(frac) >= watch.LIMITS["sm_floor"] and set(thr) <= watch.THROTTLE_OK and
                               win["gpu_alone"])
    rec = dict(item="G5", d=a.d, order=a.order, rep=a.rep, core=a.core, gpu=torch.cuda.get_device_name(),
               trt=trt.__version__, torch=torch.__version__, shots_set=f"the first shot of the timing set's chunk 0 at d = {a.d}",
               window=win, rows=rows, end=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=round(time.time() - t_main, 1))
    with open(out, "x") as fh:
        json.dump(rec, fh, indent=1)
    print(f"G5 d{a.d} {a.order}: {sum(r.get('accepted', False) for r in rows)} of {sum(r.get('timed', False) for r in rows)} "
          f"timed plans accepted; host {win['host']['ok']}, alone {win['gpu_alone']}", flush=True)


if __name__ == "__main__":
    main()
