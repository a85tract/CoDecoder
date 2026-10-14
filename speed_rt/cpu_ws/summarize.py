"""Summary of a harness run (speed_rt/cpu_ws/harness.py run --out RUN): per window its verdict and checks, per job its
wall time and the main fields of its own record (cpu_term / gref / gs_time JSON, BB rows CSV).
    python summarize.py RUN [--json OUT.json]"""
import csv
import glob
import json
import os
import sys

KEYS = ("per_block_us", "per_shot_us", "term_us", "shots", "routed", "build_s", "loop_wall_s", "conditional_mean_us",
        "plain_in_front_us")


def job_fields(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            r = json.load(open(f))
        except ValueError:
            continue
        if isinstance(r, dict):
            out.update({k: r[k] for k in KEYS if k in r and isinstance(r[k], (int, float))})
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        for row in csv.DictReader(open(f)):
            out.update(n=row.get("n"), value_ms=row.get("value_ms"))
    return out


def main():
    run = sys.argv[1]
    rows = []
    for wf in sorted(glob.glob(os.path.join(run, "*", "window.json"))):
        w = json.load(open(wf))
        cb, ca, mon = w["check_before"], w["check_after"], w["monitor"]
        print(f"{w['window']} {'PASS' if w['passed'] else 'FAIL'}  ref {cb.get('median_us')}/{ca.get('median_us')} us, "
              f"spread {cb.get('spread')}/{ca.get('spread')}, worst {cb.get('worst_rel')}/{ca.get('worst_rel')}; disk "
              f"{w['disk_delta']['reads']} reads {w['disk_delta']['pgmajfault']} majflt; MHz clk "
              f"{mon.get('clk_mhz_mean')} win {mon.get('mhz_mean')}; slot clocks "
              f"{[r.get('clock_mhz') for r in w['ref_before']]}")
        for j in w["jobs"]:
            f = job_fields(j["out"])
            rows.append(dict(window=w["window"], job=j["job"], part=j["part"], side=j["side"], rc=j["rc"],
                             wall_s=j["wall_s"], **f))
            print(f"  {j['job']:<30} rc={j['rc']} wall={j['wall_s']:8.1f} s  "
                  + " ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in f.items()))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
