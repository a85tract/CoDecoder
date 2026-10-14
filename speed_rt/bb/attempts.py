"""The attempt log of a BB pod session (speed_rt/bb/pod_bb.sh): every process that was started, or that could not start because
the pod's host or card failed its check just before it, with the verdicts of the module's own watch (bb.timing.gpu_watch: one
compute app, every SM clock sample at least 97 % of the maximum; GPU items only) and of speed_rt.watch (host and clocks over the
whole process). One slot = one process of one item in one round; a slot has at most three attempts (the failure rule of the BB
stage's registration). An attempt's outcome is
  accepted  the process ran to its end (exit 0), wrote its rows, and every check accepted it: the attempt counts, the slot is
            established and no further attempt is made
  rejected  a check refused it: the host check before it, the idle card before it (no process started), or the module's or the
            watch's verdict after it (the process ran and wrote its rows)
  failed    the process started and did not finish its work: a non-zero exit, no timing row, no watch record
A slot with three attempts and no accepted one is "exhausted" (not established: the session goes on, every comparison that uses
the slot is withheld) unless its third attempt failed, which is "stopped" (the session stops: a technical failure). Rejected
attempts stay in the log; analysis_bb.py takes its processes from here. Standard library only.
    python attempts.py rows-count --rows ROWS.csv                    the number of data rows (0 if the file is absent)
    python attempts.py slot --file A.json --item ITEM --round R      "<state> <attempts so far>", state open | established | exhausted | stopped
    python attempts.py record --file A.json --item ITEM --round R --attempt K --kind gpu|cpu --rows ROWS.csv --skip N
                              [--watch REC.json] [--hostcheck FILE] [--log LOG] [--cmd-text TEXT]
                              [--not-started REASON] [--waited SECONDS]
                                                                     append one attempt (the rows after the first N rows are its own);
                                                                     prints "<item> round R attempt K: OUTCOME (reason)" and exits
                                                                     0 accepted, 10 rejected, 11 failed
    python attempts.py summary --file A.json                         one line per item: slots established, exhausted, stopped"""
import argparse
import csv
import json
import os
import re
import time


def rows_of(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def _relative(text):
    """A record names files relative to the output directory or the tree, never by an absolute path (publication then changes no hash):
    any absolute directory prefix left in a command text is dropped, the file name stays."""
    return re.sub(r"(?<![\w./-])/(?:[\w.@+=-]+/)+", "", text or "")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {"format": 2, "attempts": []}


def save(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


TIMING_ROLES = {"gpu": ("gpu_term", "handoff", "opponent"), "cpu": ("opponent", "cpu_term")}


def judge(a):
    """(outcome, reasons, entry fields) of one started attempt."""
    new = rows_of(a.rows)[a.skip:]
    mine = [r for r in new if r["role"] in TIMING_ROLES[a.kind]]
    notes = [json.loads(r["note"]) if r["note"].startswith("{") else {} for r in mine]
    w = json.load(open(a.watch)) if a.watch and os.path.exists(a.watch) else None
    why = []
    failed = False
    if w is None:
        why.append("no watch record")
        failed = True
    elif w.get("exit") != 0:
        why.append(f"exit {w.get('exit')}")
        failed = True
    if not mine:
        why.append("the module wrote no timing row")
        failed = True
    if failed:
        return "failed", why, dict(processes=[], rows=[], watch=None)
    if a.kind == "gpu":
        for r, n in zip(mine, notes):
            if not n.get("accepted"):
                why.append(f"module rejected {r['role']}: clock_min_ratio {n.get('clock_min_ratio')}, other pids {n.get('other_pids')}, "
                           f"samples {n.get('samples')}, apps {n.get('apps')}")
    h = w["host"]
    if not h["ok"] and a.kind != "gpu":      # a pod times GPU work only: the window's host numbers are recorded (watch/), never judged
        why.append(f"watch: host (core steal {h.get('core_steal')}, other busy {h.get('core_other_busy')}, own CPU {h.get('own_cpu_s')} s)")
    if not w["clocks"]["ok"]:
        why.append(f"watch: clocks (min fraction {w['clocks'].get('sm_min_frac')}, throttle {w['clocks'].get('throttle_seen')}, "
                   f"samples {w['clocks'].get('samples')})")
    if not w.get("gpu_alone"):
        why.append("watch: another compute process on the GPU before or after")
    rows = [{"role": r["role"], "name": r["name"], "p": r["p"], "proc": r["proc"], "n": r["n"], "value_ms": r["value_ms"],
             "process": r["process"], "accepted": n.get("accepted")} for r, n in zip(mine, notes)]
    watch = {k: w[k] for k in ("accepted", "window_s", "exit", "gpu_alone", "rule", "child_cpu_s") if k in w}
    watch["host"] = {k: h.get(k) for k in ("core_other_busy", "core_steal", "own_cpu_s", "core_busy", "ok")}
    watch["clocks"] = {k: w["clocks"].get(k) for k in ("ok", "sm_min_frac", "samples", "throttle_seen")}
    return ("rejected" if why else "accepted"), why, dict(processes=sorted({r["process"] for r in mine}), rows=rows, watch=watch)


def record(a):
    doc = load(a.file)
    e = {"item": a.item, "round": a.round_, "attempt": a.attempt, "kind": a.kind, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "cmd": _relative(a.cmd_text), "log": os.path.join("logs", os.path.basename(a.log)) if a.log else None, "hostcheck": os.path.basename(os.path.dirname(a.hostcheck)) if a.hostcheck else None,
         "host_wait_s": a.waited}
    if a.not_started:
        e.update(started=False, outcome="rejected", reason=a.not_started, processes=[], rows=[], watch=None)
    else:
        outcome, why, extra = judge(a)
        e.update(started=True, outcome=outcome, reason="; ".join(why) or "accepted", watch_file=os.path.basename(a.watch) if a.watch else None, **extra)
    e["counted"] = e["outcome"] == "accepted"
    doc["attempts"].append(e)
    save(a.file, doc)
    print(f"{a.item} round {a.round_} attempt {a.attempt}: {e['outcome'].upper()} ({e['reason'][:160]})", flush=True)
    raise SystemExit({"accepted": 0, "rejected": 10, "failed": 11}[e["outcome"]])


def slot_state(doc, item, rnd):
    xs = [e for e in doc["attempts"] if e["item"] == item and e["round"] == rnd]
    if any(e["outcome"] == "accepted" for e in xs):
        return "established", len(xs)
    if len(xs) >= 3:
        return ("stopped" if xs[2]["outcome"] == "failed" else "exhausted"), len(xs)
    return "open", len(xs)


def summary(doc):
    out = []
    for item in sorted({e["item"] for e in doc["attempts"]}):
        rounds = sorted({e["round"] for e in doc["attempts"] if e["item"] == item})
        st = {r: slot_state(doc, item, r) for r in rounds}
        est = [r for r in rounds if st[r][0] == "established"]
        bad = {r: f"{st[r][0]} after {st[r][1]}" for r in rounds if st[r][0] != "established"}
        extra = {r: st[r][1] for r in est if st[r][1] > 1}
        out.append(f"{item}: slots {len(est)} established of {len(rounds)}; replaced attempts {extra or 'none'}; not established {bad or 'none'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("rows-count", "slot", "record", "summary"))
    ap.add_argument("--file")
    ap.add_argument("--item")
    ap.add_argument("--round", dest="round_", type=int)
    ap.add_argument("--attempt", type=int)
    ap.add_argument("--kind", choices=("gpu", "cpu"))
    ap.add_argument("--rows")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--watch")
    ap.add_argument("--hostcheck")
    ap.add_argument("--log", default="")
    ap.add_argument("--cmd-text", dest="cmd_text", default="")
    ap.add_argument("--not-started", dest="not_started", default="")
    ap.add_argument("--waited", type=float, default=0.0)
    a = ap.parse_args()
    if a.action == "rows-count":
        print(len(rows_of(a.rows)))
    elif a.action == "record":
        record(a)
    elif a.action == "slot":
        print(*slot_state(load(a.file), a.item, a.round_))
    else:
        print("\n".join(summary(load(a.file))))


if __name__ == "__main__":
    main()
