"""SPEED-RT CPU half: turns a cpu_ws run directory (harness.py's records: topo.json, sref/sessionref.json, plan.json,
win/<window>/window.json and each process's own record) into the layout each family's analysis reads (STAGE2_common
C1). It copies values and changes none; a process of a rejected window becomes a rejected attempt and does not count.
  surface   OUT/surface/records/cpu/: cpu_session.json, cpu_attempts.json, cpu/<slot>/<record>.json (STAGE2_surface 1.14;
            export_surf.py, read by analysis_s.cpu_reader)
  bb        OUT/bb/cpu/: rows.csv, accepted.json, session.json, host.json, attempt_log.json and, with --fit, grace.json
            (STAGE2_bb; export_bb.py, read by bb/analysis_bb.py --records OUT/bb)
  gs        OUT/gs/session/ and OUT/gs/prescreen/ in drive_gs's layout (STAGE2_gapsw 1.10; export_gs.py, read by
            gapsw/analysis_gs.py --session ... --prescreen ...)
Run it on the CPU host or on any copy of the run directory, with PYTHONPATH=CODE:CODE/surface_v2 (the gs part hashes the
draw's manifest and sets file the processes named).
    python export.py RUN OUT [--dry] [--fit fit_s2.json] [--only surface,bb,gs] [--max-attempts 3]
                     [--gs-manifest srt-gs.json] [--gs-sets sets_gs.json]
--dry: a dry run's export (each family's dry flag). --dry-accept-rejected (dry runs only): the records of windows that
the reference check rejected count as accepted, so that a dry run on an unsteady clock still tests every reader."""
import argparse
import glob
import json
import os
import re
import sys


def load_run(run, max_attempts=3, accept_rejected=False):
    """The run's records -> dict(run, name, topo, sessionref, plan, windows (time order), jobs {plan job id: job},
    attempts, max_attempts). One attempt per timed process (warm-ups dropped): window, kind (w or r), rep, attempt,
    slot, core, job, part, side, argv, rc, wall_s, dir (relative to RUN), files (its records), window_passed,
    accepted (window passed, exit 0, a record), plan_job."""
    run = os.path.abspath(run)

    def rd(p):
        p = os.path.join(run, p)
        return json.load(open(p)) if os.path.exists(p) else None
    plan = rd("plan.json")
    jobs = {j["id"]: j for w in plan["windows"] for j in w["jobs"]}
    windows = sorted((json.load(open(f)) for f in glob.glob(os.path.join(run, "win", "*", "window.json"))),
                     key=lambda w: (w["start"], w["window"]))
    attempts = []
    for w in windows:
        for j in sorted(w["jobs"], key=lambda j: j["slot"]):
            if (j.get("part") or "").endswith("_warm"):
                continue
            d = os.path.join("win", w["window"], os.path.basename(j["out"]))
            files = sorted(f for f in os.listdir(os.path.join(run, d)) if f not in ("stdout.txt", "stderr.txt")) \
                if os.path.isdir(os.path.join(run, d)) else []
            base = j.get("base") or re.sub(r"_a\d+$", "", j["job"])
            record = j.get("record", bool(files))
            attempts.append(dict(window=w["window"], kind=w["window"][0], rep=w.get("rep"), attempt=j.get("attempt", 1),
                                 slot=j["slot"], core=j["core"], job=j["job"], part=j.get("part"), side=j.get("side"),
                                 argv=j["argv"], rc=j["rc"], wall_s=j["wall_s"], dir=d, files=files,
                                 window_passed=bool(w["passed"]),
                                 accepted=bool((w["passed"] or accept_rejected) and j["rc"] == 0 and record),
                                 plan_job=jobs.get(base)))
    return dict(run=run, name=os.path.basename(run), topo=rd("topo.json") or {}, sessionref=rd("sref/sessionref.json"),
                plan=plan, windows=windows, jobs=jobs, attempts=attempts, max_attempts=max_attempts,
                accept_rejected=accept_rejected)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--dry-accept-rejected", action="store_true")
    ap.add_argument("--fit", default="", help="the stage's fit_s2.json (analysis_surf fit): BB's grace.json")
    ap.add_argument("--only", default="surface,bb,gs")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--gs-manifest", default=None, help="Decoder Switching's srt-gs.json (B_2's), when the run's own "
                    "path is not readable here (a fetched copy of the run)")
    ap.add_argument("--gs-sets", default=None, help="the stage's sets_gs.json (default: speed_rt.gapsw.gslib.SETS_FILE)")
    a = ap.parse_args()
    if a.dry_accept_rejected and not a.dry:
        raise SystemExit("--dry-accept-rejected is for dry runs only")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    R = load_run(a.run, a.max_attempts, a.dry_accept_rejected)
    only, res = a.only.split(","), {}
    if "surface" in only:
        from export_surf import export_surf
        res["surface"] = export_surf(R, os.path.join(a.out, "surface"), a.dry)
    if "bb" in only:
        from export_bb import export_bb
        if a.fit:
            R["fit_s2"] = a.fit
        res["bb"] = export_bb(R, os.path.join(a.out, "bb"), a.dry)
    if "gs" in only:
        from export_gs import export_gs
        res["gs"] = export_gs(R, os.path.join(a.out, "gs"), a.dry, manifest=a.gs_manifest, sets_file=a.gs_sets)
    os.makedirs(a.out, exist_ok=True)
    n = len(glob.glob(os.path.join(a.out, "export_summary*.json")))
    with open(os.path.join(a.out, f"export_summary{n or ''}.json"), "x") as fh:
        json.dump(dict(run=R["name"], dry=a.dry, accept_rejected=a.dry_accept_rejected, only=only, result=res), fh,
                  indent=1, default=str)
    for k, v in res.items():
        print(k, json.dumps({x: y for x, y in v.items() if not isinstance(y, (list, dict)) or x in ("slots",)},
                            default=str)[:400])
        for x in ("missing", "conflicts", "why_not_qualified", "not_established", "not_planned", "unplaced"):
            if v.get(x):
                print(f"  {k} {x}: {json.dumps(v[x], default=str)[:600]}")


if __name__ == "__main__":
    main()
