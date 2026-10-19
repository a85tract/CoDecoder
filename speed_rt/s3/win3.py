"""SPEED-RT stage 3 (STAGE3_batched.md section 3): the CPU session's windows on the workstation, with
speed_rt/cpu_ws/harness.py's machinery (its Linux host mode: slots on CCD0, the reference segment before and after every
window, the fillers, the block-read check on the harness's own read counter, the clock monitor), unchanged.

    python -m speed_rt.s3.win3 --cfg speed_rt/s3/cfg_s3.json --plan PLAN.json --sessionref SREF.json --out RUN/win
                               [--budget-min 55]
Blocks run in the plan's order; a block starts only if 1.25 times its expected length fits in what remains of the budget
(counted from the first window's start), otherwise it and every later block are not timed; the same deadline holds
for every window and every repeat (a window, first or repeated, starts only if 1.25 times its expected length fits
before the deadline; otherwise it and everything after it is not timed). A rejected window runs again as a whole at
once (ids <id>_a2, <id>_a3), at most three attempts. RUN/win/run3.json records every block, window,
attempt and verdict; each window's own record is harness.py's window.json."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cpu_ws"))
import harness as H  # noqa: E402

STAGE2_REF_US = 17.306       # stage 2's session reference on this box (speed_rt/cpu_ws/README.md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--sessionref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=55.0)
    a = ap.parse_args()
    cfg = H.load_cfg(a.cfg)
    H.session_of(cfg, a.out)
    os.makedirs(a.out, exist_ok=True)
    sref = json.load(open(a.sessionref))
    ref = sref["reference_us"]
    plan = json.load(open(a.plan))
    run = dict(what="SPEED-RT stage 3 CPU session (speed_rt/s3/win3.py)", sessionref_us=ref,
               stage2_ref_us=STAGE2_REF_US, sessionref_within_1p10_of_stage2=ref <= 1.10 * STAGE2_REF_US,
               budget_min=a.budget_min, slots=cfg["slots"], blocks=[], not_timed=[])
    t0 = time.time()
    run["start"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def save():
        with open(os.path.join(a.out, "run3.json.tmp"), "w") as fh:
            json.dump(run, fh, indent=1)
        os.replace(os.path.join(a.out, "run3.json.tmp"), os.path.join(a.out, "run3.json"))

    stop = False
    for b in plan["blocks"]:
        left = a.budget_min * 60 - (time.time() - t0)
        if stop or 1.25 * b["expected_s"] > left:
            stop = True
            run["not_timed"].append(dict(block=b["block"], name=b["name"], expected_s=b["expected_s"], left_s=round(left, 1)))
            print(f"block {b['block']} not timed: expected {b['expected_s']:.0f} s, {left:.0f} s left", flush=True)
            save()
            continue
        br = dict(block=b["block"], name=b["name"], start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), windows=[])
        run["blocks"].append(br)
        for w in b["windows"]:
            if stop:
                break
            for att in (1, 2, 3):
                left = a.budget_min * 60 - (time.time() - t0)
                if 1.25 * w["expected_s"] > left:
                    stop = True
                    run["not_timed"].append(dict(block=b["block"], name=b["name"], window=w["id"], attempt=att,
                                                 expected_s=w["expected_s"], left_s=round(left, 1)))
                    print(f"window {w['id']} attempt {att} not started: {left:.0f} s left", flush=True)
                    save()
                    break
                wid = w["id"] if att == 1 else f"{w['id']}_a{att}"
                ww = dict(w, id=wid, jobs=[dict(j, attempt=att) for j in w["jobs"]])
                ts = time.time()
                rec = H.run_window(cfg, ww, a.out, ref)
                br["windows"].append(dict(id=wid, base=w["id"], attempt=att, passed=rec["passed"],
                                          seconds=round(time.time() - ts, 1), expected_s=w["expected_s"]))
                print(f"{wid}: passed {rec['passed']} in {time.time() - ts:.0f} s (expected {w['expected_s']:.0f}); "
                      f"ref {rec['check_before']} {rec['check_after']}; disk {rec['disk_delta']['passed']}", flush=True)
                save()
                if rec["passed"]:
                    break
        br["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
    run["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run["minutes"] = round((time.time() - t0) / 60, 2)
    save()
    print(f"session done in {run['minutes']} min; not timed: {[x['block'] for x in run['not_timed']]}", flush=True)


if __name__ == "__main__":
    main()
