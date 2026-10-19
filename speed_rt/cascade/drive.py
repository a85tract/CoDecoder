"""drive.py ROUND --tree TREE --out OUT [--card rtx|h100]: one SPEED-RT Cascade timing job (R1-R4) with its replacement loop.

    ROUND  r1   D1-S: 36 pipelines (depspam_fm, S55'-deep -> C14' and Cascade alone, both engines, single and batched)
           r2   phase C: 36 pipelines (depspam_fm configs -1, 2, 3, 7, 8: Cascade alone, pairs, chains; single, batched)
           r3   the networks of C3 at batch 4096 (S55 both engines, S55'-deep new, C14, C14') and the pipelines of C4
                (depspam_fm configs 2 and 7 batched on both engines, Cascade alone batched)
           r4   d = 13: 48 pipelines (C1, C4, C5, C2 at p 0.006 and 0.0035, single and batched, 3 repetitions)
The plan is rep-major in the order of the original loops (d1_speed.sh, speed_pipe.sh, m2.plan, e7_pipes.sh). Every slot
is one fresh process of the launcher (srt_cas.py, or srt_d13.py for r4) with the original argv; its timed windows are
judged in-process (srtlib.Window: rejected only for outside interference, its docstring has the rule; the workload's own
power cap is recorded; SRT_BURN_S seconds of burn first, 0 in registered jobs). A process is accepted only if every window it has holds. A rejected process is
replaced by a fresh one with the same argv (--group repl1, repl2 where the harness has the option), at most twice;
after the third rejection the slot is "not established" and the analysis withholds every comparison that uses it. A crash,
a missing window, rows or shot trace, or shots outside the expected sets is not a clock rejection: it also consumes an
attempt, and if the slot's last attempt is such a failure the job stops (exit 3) with its records written.
Every attempt keeps its own directory OUT/slots/<slot>/a<k>/ (rows.jsonl, windows.jsonl, trace.jsonl, log.txt, and the
.npz files of r4); nothing is deleted. OUT/attempts.json lists every slot and attempt and is rewritten after each one.
OUT must not exist. The tree is read only; the launchers live next to this file.

Environment: SRT_SEEDS, SRT_BEACON (and SRT_DRY=1 for a dry record), as srtlib.py; the job script's environment
(PYTHONPATH, caches, ...), which every child inherits.
Card: --card rtx (default) runs the RTX PRO 6000's job as before. --card h100 is the same plan, shots, gate and statistics
on the H100 80GB HBM3 (the registered second card; each comparison's two sides in one session on one card, each card its
own verdict): every child gets SRT_CARD=h100 (the launchers then admit exactly that card and label their rows) and
SRT_PIDNS=1 (a pod's PID namespace); r4 also needs SRT_SESSION (the session's record) and the lock derived from it
(srt_d13.py h100-lock); both are copied into OUT (derived_lock.json, session.json). The card is written into attempts.json,
each slot's record and each attempt's.
Arguments: --tree (the pinned tree: D1_TREE for r1-r3, D13_TREE for r4), --reps (3), --n-single
(20000), --n-batched (500000), --burn-s (SRT_BURN_S or 0), --gate on|off (default on), --max-attempts (3), --card rtx|h100,
r2: --w4-lock (the depspam_fm lock of phase C, locked_depspam_fm.json) and --speed (speed_components.json), r4: --lock (the
d = 13 lock; on the H100 the derived one), the joint decision (phase_d/joint.py, surface/sv_joint.py): --joint-w
P=W,P=W (r1: every S -> B slot at p runs with --joint-w W and saves its shots and decisions into its attempt's pipe.npz
for joint_count.py compare) and --joint-w13 P=W,P=W (r4: every C5 slot at p runs with --joint-w W), dry runs: --only
SUBSTRING... (the slots whose id contains one), --force-bad
SUBSTRING=N,... (the first N attempts of those slots report their windows as rejected; needs SRT_DRY=1), --list (the plan
only)."""
import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srtlib  # noqa: E402

CHUNK = 10_000
FAMILIES = ("uniform", "depspam_fm")
NET_FAMILY = {"S55": "uniform", "S55p": "depspam_fm", "S55p_deep": "depspam_fm", "C14": "uniform", "C14p": "depspam_fm"}
NETS_P = (0.0055, 0.002, 0.003)                  # the networks' screen and routing shots (pc_speed.P_AGREE, RELAY_P)
NET_CHUNKS = 10                                  # N_AGREE_B // CHUNK
D13_P = (0.006, 0.0035)


def sets_by_point():
    """{(family, p): set name} and {p: set name for d = 13} from sets_cas.json."""
    bb, d13 = {}, {}
    for name, e in srtlib.set_defs().items():
        if "family" in e:
            bb[(e["family"], e["p"])] = name
        else:
            d13[e["p"]] = name
    return bb, d13


def chunks_for(n):
    return math.ceil(n / CHUNK)


def pipe_slot(rnd, script, args, rep, fam, p, cfg, mode, eng, tail_b, n, bbsets):
    sid = f"{rnd}.rep{rep}.{'dfm' if fam == 'depspam_fm' else 'uni'}.p{p:g}.c{cfg}.{mode}" + (f".{eng}" if eng else "")
    return dict(id=sid, launcher="srt_cas.py", script=script, args=args, windows=1,
                shots={bbsets[(fam, p)]: chunks_for(n)},
                meta=dict(round=rnd, role="pipeline", family=fam, config=cfg, p=p, mode=mode, engine=eng, tail_b=tail_b,
                          rep=rep, n=n))


def net_slot(rnd, script, rep, net, eng, bbsets):
    fam = NET_FAMILY[net]
    args = ["nets", "--net", net, "--engine", eng, "--rep", str(rep), "--out", "{rows}", "--group", "{group}"]
    args += ["--paths", "compiled@4096"] if eng == "cascade" else ["--batches", "4096"]
    return dict(id=f"{rnd}.rep{rep}.net.{net}.{eng}", launcher="srt_cas.py", script=script, args=args,
                windows=1, shots={bbsets[(fam, p)]: NET_CHUNKS for p in NETS_P},
                meta=dict(round=rnd, role="net", net=net, family=fam, engine=eng, rep=rep, batch=4096))


def m2_pipe(rnd, tree, rep, fam, p, cfg, mode, eng, n, bbsets):
    script = f"{tree}/cascade_compare/batchopt/m2.py"
    args = ["pipeline", "--setting", fam, "--config", str(cfg), "--p", str(p), "--mode", mode, "--engine", eng,
            "--tail-b", "0", "--rep", str(rep), "--n", str(n), "--out", "{rows}", "--group", "{group}"]
    return pipe_slot(rnd, script, args, rep, fam, p, cfg, mode, eng, 0, n, bbsets)


def weights(spec):
    """'0.002=6.31,0.003=3.76' -> {0.002: 6.31, 0.003: 3.76} (empty: {})."""
    return {float(k): float(v) for k, v in (x.split("=") for x in spec.split(",") if x)}


def plan_r1(a, bbsets, d13sets):
    script = f"{a.tree}/cascade_compare/phase_d/d1_m2.py"
    jw = weights(a.joint_w)
    out = []
    for rep in range(1, a.reps + 1):
        for p, i in ((0.002, 0), (0.003, 1)):
            for cfg, mode, eng in ((-1, "single", "base"), (-1, "batched", "base"), (i, "single", "base"),
                                   (i, "single", "new"), (i, "batched", "base"), (i, "batched", "new")):
                n = a.n_single if mode == "single" else a.n_batched
                args = ["pipeline", "--setting", "depspam_fm", "--config", str(cfg), "--p", str(p), "--mode", mode,
                        "--engine", eng, "--tail-b", "0", "--rep", str(rep), "--n", str(n), "--out", "{rows}",
                        "--group", "{group}"]
                if cfg >= 0 and p in jw:                 # the composition with the joint step
                    args += ["--joint-w", repr(jw[p]), "--save", "{savef}"]
                out.append(pipe_slot("r1", script, args, rep, "depspam_fm", p, cfg, mode, eng, 0, n, bbsets))
                out[-1]["meta"]["joint_w"] = jw.get(p) if cfg >= 0 else None
    return out


def plan_r2(a, bbsets, d13sets):
    if not (a.w4_lock and a.speed):
        raise SystemExit("r2 needs --w4-lock and --speed")
    L = json.load(open(a.w4_lock))
    for i, p, typ in ((2, 0.002, "S-B"), (3, 0.002, "S-B-R"), (7, 0.003, "S-B"), (8, 0.003, "S-B-R")):
        c = L["configs"][i]
        if c["p"] != p or c["type"] != typ:
            raise SystemExit(f"r2: config {i} of {a.w4_lock} is {c['p']} {c['type']}, not {p} {typ}")
    script = f"{a.tree}/cascade_compare/phase_c/pc_speed.py"
    out = []
    for rep in range(1, a.reps + 1):
        for p, cfgs in ((0.002, (2, 3)), (0.003, (7, 8))):
            for cfg in (-1,) + cfgs:
                for mode in ("single", "batched"):
                    n = a.n_single if mode == "single" else a.n_batched
                    args = ["pipeline", "--locked", a.w4_lock, "--config", str(cfg), "--p", str(p), "--mode", mode,
                            "--rep", str(rep), "--n", str(n), "--speed", a.speed, "--out", "{rows}"]
                    out.append(pipe_slot("r2", script, args, rep, "depspam_fm", p, cfg, mode, None, 0, n, bbsets))
    return out


def plan_r3(a, bbsets, d13sets):
    script = f"{a.tree}/cascade_compare/batchopt/m2.py"
    out = []
    for rep in range(1, a.reps + 1):
        for net, eng in (("S55", "base"), ("S55", "new"), ("S55p_deep", "new"), ("C14", "cascade"), ("C14p", "cascade")):
            out.append(net_slot("r3", script, rep, net, eng, bbsets))
    for rep in range(1, a.reps + 1):
        for p, i in ((0.002, 2), (0.003, 7)):
            for cfg, eng in ((-1, "base"), (i, "base"), (i, "new")):
                out.append(m2_pipe("r3", a.tree, rep, "depspam_fm", p, cfg, "batched", eng, a.n_batched, bbsets))
    return out


def plan_r4(a, bbsets, d13sets):
    if not a.lock:
        raise SystemExit("r4 needs --lock")
    plan = json.load(open(a.lock))["pipes"]["c7"]
    jw = weights(a.joint_w13)
    out = []
    for rep in range(1, a.reps + 1):
        for p in D13_P:
            for c in ("C1", "C4", "C5", "C2"):
                for mode in ("single", "batched"):
                    n = plan[mode]
                    args = ["pipeline", "--config", c, "--p", str(p), "--mode", mode, "--rep", str(rep), "--lock",
                            a.lock, "--stage", "c7", "--out", "{rows}", "--save", "{save}"]
                    if c == "C5" and p in jw:            # the composition with the joint step
                        args += ["--joint-w", repr(jw[p])]
                    out.append(dict(id=f"r4.rep{rep}.p{p:g}.{c}.{mode}", launcher="srt_d13.py", script=None, args=args,
                                    windows=1, shots={d13sets[p]: chunks_for(n)},
                                    meta=dict(round="r4", role="pipeline", config=c, p=p, mode=mode, rep=rep, n=n,
                                              joint_w=jw.get(p) if c == "C5" else None)))
    return out


PLANS = {"r1": plan_r1, "r2": plan_r2, "r3": plan_r3, "r4": plan_r4}


# ---------------------------------------------------------------------------------------------------------------- run
def jl(path):
    return [json.loads(x) for x in open(path) if x.strip()] if os.path.exists(path) else []


def argv_record(argv, out, tree):
    """The argv as recorded: the tree and OUT by name, any other absolute path by its file name (records carry no
    machine's paths)."""
    def one(x):
        if x.startswith(out):
            return "$OUT" + x[len(out):]
        if x.startswith(tree):
            return "$TREE" + x[len(tree):]
        return "$" + os.path.basename(x) if x.startswith("/") else x
    return [one(x) for x in argv]


def run_attempt(slot, k, a, out, bad_n):
    d = os.path.join(out, "slots", slot["id"], f"a{k}")
    os.makedirs(d)
    f = {n: os.path.join(d, n) for n in ("rows.jsonl", "windows.jsonl", "trace.jsonl", "log.txt")}
    for n in ("windows.jsonl", "trace.jsonl"):
        open(f[n], "w").close()
    group = "main" if k == 0 else f"repl{k}"
    sub = {"{rows}": f["rows.jsonl"], "{group}": group, "{save}": os.path.join(d, "pipes"),
           "{savef}": os.path.join(d, "pipe.npz")}
    args = [sub.get(x, x) for x in slot["args"]]
    cmd = [sys.executable, os.path.join(HERE, slot["launcher"])] + ([slot["script"]] if slot["script"] else []) + args
    env = dict(os.environ, SRT_WINDOWS=f["windows.jsonl"], SRT_TRACE=f["trace.jsonl"], SRT_ATTEMPT=str(k),
               SRT_BURN_S=str(a.burn_s), SRT_TAG=f"{slot['id']}.a{k}")
    if a.card == "h100":
        env.update(SRT_CARD="h100", SRT_PIDNS="1", BB_PIDNS="1")
    if k < bad_n:
        env["SRT_FORCE_BAD"] = str(bad_n)
    else:
        env.pop("SRT_FORCE_BAD", None)
    t0 = time.time()
    with open(f["log.txt"], "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    secs = time.time() - t0
    windows, rows, trace = jl(f["windows.jsonl"]), jl(f["rows.jsonl"]), jl(f["trace.jsonl"])
    hard, soft = [], []
    if rc != 0:
        hard.append(f"exit code {rc}")
    if any("error" in w for w in windows):
        hard.append("window error: " + "; ".join(str(w["error"]) for w in windows if "error" in w))
    elif len(windows) != slot["windows"]:
        hard.append(f"{len(windows)} windows, {slot['windows']} expected")
    if not rows:
        hard.append("no rows")
    got = {}
    for t in trace:
        got.setdefault(t["set"], {}).setdefault(t["chunk"], set()).add(t["sha256"])
    if set(got) != set(slot["shots"]):
        hard.append(f"shot sets {sorted(got)}, expected {sorted(slot['shots'])}")
    for name, nch in slot["shots"].items():
        have = got.get(name, {})
        if any(c not in have for c in range(nch)) or any(len(v) != 1 for v in have.values()):
            hard.append(f"{name}: chunks 0-{nch - 1} not all sampled once with one digest")
    if not hard:
        soft = [w["reason"] for w in windows if not w.get("accepted")]
    rec = dict(k=k, card=a.card, group=group, dir=os.path.relpath(d, out), rc=rc, start=t0, secs=secs, host=platform.node(),
               slurm_job=os.environ.get("SLURM_JOB_ID", ""), argv=argv_record(cmd, out, a.tree), windows=windows,
               rows=len(rows), row_pids=sorted({r.get("pid") for r in rows}), trace_lines=len(trace), hard=hard,
               soft=soft, accepted=not hard and not soft)
    return rec


def write_json(path, obj):
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", choices=sorted(PLANS))
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-single", type=int, default=20_000)
    ap.add_argument("--n-batched", type=int, default=500_000)
    ap.add_argument("--burn-s", type=float, default=float(os.environ.get("SRT_BURN_S", "0")))
    ap.add_argument("--gate", choices=("on", "off"), default=None)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--card", choices=("rtx", "h100"), default="rtx")
    ap.add_argument("--w4-lock", default="")
    ap.add_argument("--speed", default="")
    ap.add_argument("--lock", default="")
    ap.add_argument("--joint-w", default="", help="r1: the joint decision's w per p, e.g. 0.002=6.31,0.003=3.76")
    ap.add_argument("--joint-w13", default="", help="r4: C5's joint w per p, e.g. 0.006=W,0.0035=W")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--force-bad", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    gate = (a.gate or "on") == "on"
    if a.card == "rtx" and os.environ.get("SRT_CARD", "rtx").lower() != "rtx":
        raise SystemExit(f"SRT_CARD={os.environ['SRT_CARD']!r} with --card rtx")
    if a.card == "h100" and a.round == "r4" and not a.list and not os.path.isfile(os.environ.get("SRT_SESSION", "")):
        raise SystemExit("r4 on the H100 needs SRT_SESSION (the session's record, srt_d13.py h100-session)")
    a.tree = os.path.abspath(a.tree).rstrip("/")
    out = os.path.abspath(a.out)
    bbsets, d13sets = sets_by_point()
    slots = PLANS[a.round](a, bbsets, d13sets)
    if a.only:
        slots = [s for s in slots if any(x in s["id"] for x in a.only)]
    ids = [s["id"] for s in slots]
    assert len(set(ids)) == len(ids), "duplicate slot ids"
    if a.list:
        for s in slots:
            print(s["id"], s["launcher"], " ".join(s["args"]))
        print(f"{len(slots)} slots")
        return
    bad = {}
    if a.force_bad:
        if os.environ.get("SRT_DRY") != "1":
            raise SystemExit("--force-bad is for dry runs (SRT_DRY=1)")
        for x in a.force_bad.split(","):
            k, v = x.split("=")
            bad[k] = int(v)
    if not slots:
        raise SystemExit("no slots")
    srtlib.tables()                                        # the beacon, the seeds file and the sets, checked before any process
    rec, dry = srtlib.record()
    if dry and os.environ.get("SRT_DRY") != "1":
        raise SystemExit("a dry record needs SRT_DRY=1")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: a job writes into a fresh directory")
    if a.card == "h100" and a.round == "r4":               # the lock must be the one derived from the session (srt_d13.py)
        der = json.load(open(a.lock)).get("h100_derivation") or {}
        if der.get("session_sha256") != srtlib.sha256_file(os.environ["SRT_SESSION"]):
            raise SystemExit(f"{a.lock} is not derived from the session {os.environ['SRT_SESSION']}")
    os.makedirs(os.path.join(out, "slots"))
    if a.card == "h100" and a.round == "r4":               # the record carries the lock its rows name and the session's
        shutil.copyfile(a.lock, os.path.join(out, "derived_lock.json"))
        shutil.copyfile(os.environ["SRT_SESSION"], os.path.join(out, "session.json"))
    dep = os.path.join(a.tree, "DEPLOYED")
    top = dict(format="srt-cas-attempts-1", round=a.round, card=a.card, dry=dry, gate=dict(on=gate, max_attempts=a.max_attempts,
               burn_s=a.burn_s, sm_floor=srtlib.SM_FLOOR, max_gap_s=srtlib.MAX_GAP_S),
               tree_deployed=open(dep).read().strip() if os.path.exists(dep) else None,
               beacon=dict(round=rec["round"], randomness=rec["randomness"], dry=dry),
               seeds_sha256=srtlib.sha256_file(os.environ["SRT_SEEDS"]),
               sets_sha256=srtlib.sha256_file(os.path.join(HERE, "sets_cas.json")),
               launchers={n: srtlib.sha256_file(os.path.join(HERE, n)) for n in ("srt_cas.py", "srt_d13.py", "srtlib.py",
                                                                                  "drive.py")},
               args=dict(reps=a.reps, n_single=a.n_single, n_batched=a.n_batched, only=a.only, force_bad=bad),
               env={k: os.environ.get(k, "") for k in ("SLURM_JOB_ID", "M2_CARD", "BB_PIDNS", "SRT_DRY", "SRT_CARD",
                                                       "SRT_PIDNS", "SRT_KCACHE_BUILD")},
               session_sha256=srtlib.sha256_file(os.environ["SRT_SESSION"]) if os.environ.get("SRT_SESSION") else None,
               host=platform.node(), start=time.time(),
               slots=[dict(id=s["id"], card=a.card, meta=s["meta"], windows=s["windows"], shots=s["shots"], status="pending",
                           accepted_attempt=None, attempts=[]) for s in slots])
    path = os.path.join(out, "attempts.json")
    write_json(path, top)
    print(f"drive {a.round}: {len(slots)} slots, card {a.card}, gate {'on' if gate else 'off'}, burn {a.burn_s:g} s, out {out}",
          flush=True)
    stop = False
    for s, row in zip(slots, top["slots"]):
        n_bad = max([v for k, v in bad.items() if k in s["id"]] or [0])
        for k in range(a.max_attempts if gate else 1):
            r = run_attempt(s, k, a, out, n_bad)
            row["attempts"].append(r)
            if r["accepted"] or (not gate and not r["hard"]):
                row["status"] = "established" if r["accepted"] else "recorded, flagged"
                row["accepted_attempt"] = k if r["accepted"] else None
            else:
                row["status"] = "failed" if r["hard"] else "not established"
            write_json(path, top)
            w = r["windows"][0] if r["windows"] else {}
            print(f"{s['id']} a{k}: rc {r['rc']} {r['secs']:.0f} s windows {len(r['windows'])} min clock "
                  f"{w.get('clock_min_ratio')} -> {row['status']} {r['hard'] or r['soft'] or ''}", flush=True)
            if r["accepted"] or not gate:
                break
        if row["attempts"][-1]["hard"]:
            stop = True
            print(f"stop: the last attempt of {s['id']} failed ({row['attempts'][-1]['hard']})", flush=True)
            break
    top["end"] = time.time()
    top["stopped_early"] = stop
    top["summary"] = {k: sum(r["status"] == k for r in top["slots"]) for k in
                      ("established", "not established", "recorded, flagged", "failed", "pending")}
    write_json(path, top)
    print(f"drive {a.round}: {top['summary']} in {(top['end'] - top['start']) / 60:.1f} min", flush=True)
    sys.exit(3 if stop else 0)


if __name__ == "__main__":
    main()
