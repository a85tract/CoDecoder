"""D9-SHALLOW's CPU plan for speed_rt/cpu_ws/harness.py (PLAN.md section 5): the speed re-test's packing (plans.py, imported
unchanged from the deployed harness) over two parts, three repetitions, rep-major, windows of at most 8, longest first, the
sides alternating:
  gref  the Grace fit's 18 reference points (plans.gref: speed_rt.gref time on the srt-ref sets), the slowest of three
  term  B1's CPU term: timing.cpu_term term on the 200,000 timing shots (the view {data}/surf_d9s/view) with the counted GPU
        session's mask ({handoff}/d9s/low/val_d9_b1w64_sh.npz), as plans.surf_term builds a term job; the slowest of three
Both hand-off inputs (the hand-off unit d9s) are imported before the run, so no job waits for one (plans.py --no-needs).
    python d9_shallow/plan_d9s.py --cpu-ws R/cpu_ws --out RUN/plan.json"""
import argparse
import json
import os
import sys

NET = "d9_b1w64_sh"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-ws", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.cpu_ws))
    import plans
    windows = []
    for rep in (1, 2, 3):
        argv = ["{py}", "-m", "timing.cpu_term", "term", "--d", "9", "--val", "{data}/surf_d9s/view", "--mask",
                f"{{handoff}}/d9s/low/val_{NET}.npz", "--rep", str(rep), "--core", "{core}", "--out", "{out}"]
        jobs = list(plans.gref(rep, False)) + [dict(id=f"term_{NET}_r{rep}", part="surf_term", side="ours", cwd="{code}",
                                                     env=plans.SURF_ENV, argv=argv, est_s=plans.EST["surf_term"][9])]
        for j in jobs:
            j["needs"] = []
        for js in plans.pack(jobs, 8):
            windows.append(dict(id=f"w{len(windows) + 1:03d}", rep=rep, jobs=js, needs=[]))
    plan = dict(what="D9-SHALLOW CPU plan (d9_shallow/plan_d9s.py over speed_rt/cpu_ws/plans.py)", parts="gref,surf_term",
                reps=3, dry=False, order="rep", slots=8, windows=windows)
    with open(a.out, "x") as fh:
        json.dump(plan, fh, indent=1)
    for w in windows:
        print(w["id"], w["rep"], " ".join(j["id"] for j in w["jobs"]))
    print(f"{a.out}: {len(windows)} windows, {sum(len(w['jobs']) for w in windows)} jobs")


if __name__ == "__main__":
    main()
