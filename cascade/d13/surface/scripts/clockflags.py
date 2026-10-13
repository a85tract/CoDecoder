"""clockflags.py CLK APPS ROWS [MARKS]: phase C's gpu_watch rule applied afterwards to each timing of a job, over that timing's
own window: accepted when the lowest SM-clock sample in the window is >= 97 % of clocks.max.sm and no process other
than the timing's own is on the card. Stdlib only (runs on an Alpine login node: python3 - CLK APPS ROWS < this).

Windows (row dates are emit times, whole seconds, written right after the timed part):
  net rows, metric "time" (sv_speed screen): from the same process's "agree" row of that path and batch (emitted just
    before the timing starts) to the time row's date + 1 s;
  pipeline rows: [date - d, date + 1 s], d = wall_s (batched) or n x mean latency (single; the 200 untimed shots and
    the per-shot Python outside the timer are left out). C3 (M alone) is CPU work: its GPU flag is not applicable.
  MARKS (winmark.py's lines, the clean repeat): a pipeline row of the same job and pid takes its exact window instead.
    MARKS hold Unix times, the samples and rows local times: run it in the cluster's zone (TZ=America/Denver, as on
    an Alpine login node), or no sample falls in an exact window.
Prints one JSON line per timing: the window, samples, lowest and 1st-percentile clock, share of samples below the
97 % line, the PIDs seen on the card, and the flag."""
import datetime as dt
import json
import sys


def ts(s):
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def rowtime(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()


clk, apps = [], []
for line in open(sys.argv[1]):
    f = line.split(",")
    try:
        clk.append((ts(f[0]), float(f[1]), float(f[2]), float(f[3])))
    except (ValueError, IndexError):
        pass
for line in open(sys.argv[2]):
    p = line.split()
    if len(p) >= 2:
        apps.append((ts(p[0] + " " + p[1]), {int(x) for x in p[2:] if x.isdigit()}))
rows = [json.loads(line) for line in open(sys.argv[3])]
marks = {}
for line in open(sys.argv[4]) if len(sys.argv) > 4 else ():
    m = json.loads(line)
    if "start" in m:
        marks[(m["job"], m["pid"])] = (m["start"], m["end"])
MAX = max(c[2] for c in clk)


def flag(t0, t1, pid):
    s = sorted(c[1] for c in clk if t0 <= c[0] <= t1)
    seen = set().union(*[a[1] for a in apps if t0 <= a[0] <= t1]) if apps else set()
    if not s:
        return {"samples": 0, "flag": "no samples"}
    low = sum(1 for x in s if x < 0.97 * MAX)
    others = sorted(seen - {pid})
    ok = s[0] >= 0.97 * MAX and not others
    return {"samples": len(s), "min_mhz": s[0], "p1_mhz": s[len(s) // 100], "max_sm_mhz": MAX,
            "min_share": round(s[0] / MAX, 4), "below_97": low, "others": others,
            "flag": "ok" if ok else ("other process" if others else "clock below 97 %")}


last_agree = {}
for r in rows:
    t = rowtime(r["date"])
    if r.get("role") == "net":
        k = (r["pid"], r.get("path"), r.get("batch"))
        if r.get("metric") == "agree":
            last_agree[k] = t
        elif r.get("metric") == "time" and k in last_agree:
            w = (last_agree[k], t + 1)
            print(json.dumps({"net": r["net"], "path": r["path"], "batch": r["batch"], "rep": r["rep"], "pid": r["pid"],
                              "win": [r["date"], round(w[1] - w[0])], **flag(*w, r["pid"])}))
    elif r.get("role") == "pipeline":
        if r.get("metric") == "batched":
            d = r["wall_s"]
        else:
            lat = r["latency_us"]
            d = lat["mean"] * r["shots"] / 1e6
        exact = (r["job"], r["pid"]) in marks
        w = marks[(r["job"], r["pid"])] if exact else (t - d, t + 1)
        out = {"config": r["config"], "p": r["p"], "mode": r["mode"], "rep": r["rep"], "pid": r["pid"],
               "win": [r["date"], round(w[1] - w[0], 3 if exact else 1)], "exact": exact, **flag(*w, r["pid"])}
        if r["config"] == "C3":
            out["flag"] = "n/a (CPU only)"
        print(json.dumps(out))
