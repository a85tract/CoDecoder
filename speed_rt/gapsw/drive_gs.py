"""drive_gs.py: the Decoder Switching stage's session on its pod (speed_rt/gapsw; the registration is the stage's section).

    python -m speed_rt.gapsw.drive_gs prescreen --probe-file F --out DIR [--core C]
        picks the core (outside the first allowed CPU's physical core, the physical core least busy over 1 s), proves
        that /proc/stat's CPU numbers are the pinned core's (hostwin.selftest), checks the core idle over 5 s (busy share by
        others at most 5 %, steal at most 1 %), runs three fresh probe processes (plain PM per block, d = 9, three timed passes
        each after an untimed one) and sets the session's reference = the median of their nine passes; passes when the nine
        passes' largest over smallest is at most 1.06 and the container has at least 4 allowed CPUs.
        -> DIR/prescreen.json, DIR/CORE, DIR/PROBE_REF; exit 4 when the pod fails.
    python -m speed_rt.gapsw.drive_gs run --shots S --manifest M --masks K --probe-file F --probe-ref US --out OUT [--core C]
        the timed session (every process on the core picked at its start: the allowed CPU outside the driver's physical core whose
        physical core was least busy over the preceding 2 s; --core pins all of them to C, for tests): three interleaved repetitions (rep-major) of the 12
        Decoder Switching slots (the front end and correlated PM on the routed shots, d = 7 ... 17), one fresh process each
        (gs_time.py), one at a time. There are no Grace reference slots: the stage's one fit is the surface family's.
    python -m speed_rt.gapsw.drive_gs list [--reps 3]       the plan only

The failure rule (one rule, no choice):
  1. A process whose host checks reject it (exit 4: the pre-window idle or probe check, or the window's own check) or which
     fails (a check of its own, a crash, a missing record) is replaced at once by a fresh process with the same arguments, at
     most twice: a slot has at most three attempts.
  2. A slot whose third attempt is again rejected by the host checks (none of its three attempts accepted) is not established.
     The session goes on. A Decoder Switching slot not established
     withholds its d's value of that term (and its composite). The raw values of every accepted process are still published.
  3. A session that does not reach its end (the pod is lost, or the third attempt of a slot fails, not rejected, and the driver
     stops with exit 3) is a technical failure: the whole session runs once more from its first slot as a new session on a newly pre-screened
     pod (or the same pod, if it is still there), into a fresh directory, and only the new session's records count; if that also
     fails, every value of the stage is withheld.
  4. Nothing runs again because of a value it produced. Every attempt, accepted or not, and every session is kept.
OUT/attempts.json lists every slot and attempt and is rewritten after each one; OUT/session.json is written once at the start.
--only SUBSTR... and --chunks N are for dry runs (GS_FORCE_REJECT=1 in a dry session's environment rejects the first N
attempts of the slots named by --force-bad SUBSTR=N,...)."""
import argparse
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from speed_rt.gapsw import gslib as G  # noqa: E402
from speed_rt.gapsw import hostwin as H  # noqa: E402


def plan(reps=G.REPS):
    """The slots, rep-major: per repetition every d's front end and routed-shot term (12 slots; no reference slots)."""
    slots = []
    for r in range(1, reps + 1):
        for d in G.DS_DISTANCES:
            slots.append(dict(id=f"r{r}-fe-d{d}", kind="fe", rep=r, argv=["fe", "--d", str(d)]))
            slots.append(dict(id=f"r{r}-term-d{d}", kind="term", rep=r, argv=["term", "--d", str(d)]))
    return slots


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_core(secs=2.0):
    """The timed core: among the allowed CPUs outside the physical core of the lowest allowed CPU (the driver's), the one whose
    physical core (its thread and its sibling threads) was least busy over the last `secs` seconds (ties: the lowest CPU)."""
    allowed = sorted(os.sched_getaffinity(0))
    home = allowed[0]
    avoid = {home} | set(H.siblings(home))
    cand = [c for c in allowed if c not in avoid]
    s0 = H.stat_lines()
    time.sleep(secs)
    s1 = H.stat_lines()

    def busy(c):
        phys = [c] + H.siblings(c)
        tot = sum(sum(b - a for a, b in zip(s0[x], s1[x])) for x in phys if x in s0)
        bz = sum(sum(s1[x][i] - s0[x][i] for i in (0, 1, 2, 5, 6)) for x in phys if x in s0)
        return bz / tot if tot > 0 else 1.0
    best = min(cand, key=lambda c: (busy(c), c))
    return best, dict(home=home, allowed=len(allowed), candidates=len(cand), busy=busy(best), secs=secs)


def prescreen(a):
    os.makedirs(a.out, exist_ok=False)
    py = sys.executable
    core, pick = (a.core, dict(forced=True)) if a.core is not None else pick_core(1.0)
    rec = dict(item="pre-screen", core=core, pick=pick, limits=H.LIMITS, cpu=G.cpu_model(), cgroup_cpu=G.cgroup_quota(),
               cpus_allowed=len(os.sched_getaffinity(0)), cpus_online=os.cpu_count(), start=G.now(), node=os.uname().nodename)
    rec.update(getattr(a, "extra", None) or {})                # the parallel driver's: the allowed CPUs counted before it pinned itself
    st = subprocess.run([py, "-m", "speed_rt.gapsw.gs_time", "selftest", "--core", str(core)], capture_output=True, text=True)
    rec["selftest"] = json.loads(st.stdout.strip().splitlines()[-1]) if st.stdout.strip() else dict(ok=False, err=st.stderr[-300:])
    rec["idle"] = H.idle_check(core)
    vals = []
    for k in (1, 2, 3):
        subprocess.run(["taskset", "-c", str(core), py, "-m", "speed_rt.gapsw.gs_time", "probe", "--core", str(core), "--rep", str(k),
                        "--probe-file", a.probe_file, "--out", os.path.join(a.out, "probe")], check=True, stdout=subprocess.DEVNULL)
        vals += json.load(open(os.path.join(a.out, "probe", f"probe_rep{k}.json")))["probe"]["values_us"]
    ref = statistics.median(vals)
    spread = max(vals) / min(vals) - 1
    rec["probe"] = dict(values_us=vals, ref_us=ref, spread=spread, ok=bool(spread <= H.LIMITS["probe_spread"]))
    rec["cpus_ok"] = rec["cpus_allowed"] >= 4
    rec["checks"] = dict(selftest=rec["selftest"]["ok"], idle=rec["idle"]["ok"], probe=rec["probe"]["ok"], cpus=rec["cpus_ok"])
    rec["failed"] = sorted(k for k, v in rec["checks"].items() if not v)
    rec["passed"] = not rec["failed"]
    rec["end"] = G.now()
    G.write_once(os.path.join(a.out, "prescreen.json"), rec)
    if rec["passed"]:
        open(os.path.join(a.out, "CORE"), "x").write(f"{core}\n")
        open(os.path.join(a.out, "PROBE_REF"), "x").write(f"{ref!r}\n")
    log(f"PRE-SCREEN {'PASSED' if rec['passed'] else 'FAILED ' + ', '.join(rec['failed'])}: {rec['cpu']}, core {core}, "
        f"quota {rec['cgroup_cpu']}, probe reference {ref:.3f} us per shot (spread {spread:.4f})")
    raise SystemExit(0 if rec["passed"] else 4)


def classify(rc, recfile):
    """('accepted' | 'rejected' | 'failed', record or None, reasons)."""
    rec = None
    if os.path.exists(recfile):
        try:
            rec = json.load(open(recfile))
        except ValueError:
            rec = None
    if rc == 0 and rec and rec.get("accepted"):
        return "accepted", rec, []
    if rec is not None and rc == 4 and not rec.get("failed"):
        return "rejected", rec, rec.get("reasons", [])
    return "failed", rec, (rec or {}).get("reasons", []) + [f"exit {rc}" + ("" if rec else ", no record")]


def run(a):
    lock = open(os.environ.get("GS_LOCK", "/tmp/gs_timed.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another timed session holds the lock: never two timed processes at once")
    assert not os.path.exists(a.out), f"{a.out} exists: a session writes into a fresh directory"
    os.makedirs(os.path.join(a.out, "slots"))
    force = {}
    for x in (a.force_bad or "").split(","):
        if x:
            k, _, n = x.partition("=")
            force[k] = int(n or 1)
    slots = [s for s in plan(a.reps) if not a.only or any(o in s["id"] for o in a.only)]
    from convlines import provenance
    session = dict(item="Decoder Switching timing session", start=G.now(), core=a.core, core_rule=None if a.core is not None else
                   "per attempt: the allowed CPU outside the driver's physical core whose physical core was least busy over the last 2 s",
                   probe_ref_us=a.probe_ref, cpu=G.cpu_model(),
                   cgroup_cpu=G.cgroup_quota(), cpus_allowed=len(os.sched_getaffinity(0)), node=os.uname().nodename,
                   deployed=provenance.commit(G.ROOT), dry=bool(a.dry), slots=[s["id"] for s in slots], reps=a.reps,
                   max_attempts=G.MAX_ATTEMPTS, manifest_sha256=G.sha256_file(a.manifest),
                   sets_sha256=G.sha256_file(G.SETS_FILE), argv=G.rel_argv(sys.argv))
    G.write_once(os.path.join(a.out, "session.json"), session)
    attempts = dict(session=os.path.basename(os.path.abspath(a.out)), slots={})

    def save():
        tmp = os.path.join(a.out, "attempts.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(attempts, fh, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(a.out, "attempts.json"))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1")
    t_start = time.time()
    for i, s in enumerate(slots, 1):
        ent = attempts["slots"][s["id"]] = dict(kind=s["kind"], rep=s["rep"], argv=s["argv"], attempts=[], status="running")
        for k in range(1, G.MAX_ATTEMPTS + 1):
            adir = os.path.join(a.out, "slots", s["id"], f"a{k}")
            os.makedirs(adir)
            core, pick = (a.core, dict(forced=True)) if a.core is not None else pick_core()
            common = ["--rep", str(s["rep"]), "--core", str(core), "--probe-file", a.probe_file, "--probe-ref", repr(a.probe_ref),
                      "--shots", a.shots, "--manifest", a.manifest, "--out", adir]
            if s["kind"] in ("fe", "term"):
                common += ["--masks", a.masks]
            if a.chunks is not None:
                common += ["--chunks", str(a.chunks)]
            cmd = ["taskset", "-c", str(core), sys.executable, "-m", "speed_rt.gapsw.gs_time", s["argv"][0], *s["argv"][1:], *common]
            e = dict(env)
            if any(f in s["id"] and k <= n for f, n in force.items()):
                e["GS_FORCE_REJECT"] = "1"
            t0 = time.time()
            log(f"slot {i}/{len(slots)} {s['id']} attempt {k} core {core} (busy {pick.get('busy', 0):.3f})")
            with open(os.path.join(adir, "log.txt"), "w") as lg:
                rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=e, cwd=G.ROOT).returncode
            status, rec, reasons = classify(rc, os.path.join(adir, "rec.json"))
            ent["attempts"].append(dict(k=k, rc=rc, status=status, reasons=reasons, seconds=round(time.time() - t0, 1),
                                        dir=os.path.relpath(adir, a.out), core=core, pick=pick))
            log(f"  {status} rc {rc} {time.time() - t0:.0f} s {reasons}")
            if status == "accepted":
                ent["status"] = "accepted"
                break
            if status == "failed" and k == G.MAX_ATTEMPTS:
                ent["status"] = "failed"
                save()
                log("the last attempt of a slot failed: technical failure, the session stops (exit 3)")
                raise SystemExit(3)
            ent["status"] = "not established" if k == G.MAX_ATTEMPTS else "retry"
            save()
        save()
    attempts["end"] = G.now()
    attempts["seconds"] = round(time.time() - t_start, 1)
    save()
    bad = sorted(k for k, v in attempts["slots"].items() if v["status"] != "accepted")
    log(f"SESSION DONE in {attempts['seconds']:.0f} s: {len(slots) - len(bad)} of {len(slots)} slots accepted; not established: {bad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prescreen")
    p.add_argument("--probe-file", required=True)
    p.add_argument("--core", type=int, default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--shots", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--probe-file", required=True)
    p.add_argument("--core", type=int, default=None, help="pin every process to this core (default: each attempt picks the quietest)")
    p.add_argument("--probe-ref", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=G.REPS)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--force-bad", default="")
    p = sub.add_parser("list")
    p.add_argument("--reps", type=int, default=G.REPS)
    a = ap.parse_args()
    if a.cmd == "list":
        for s in plan(a.reps):
            print(s["id"], " ".join(s["argv"]))
        print(len(plan(a.reps)), "slots")
    elif a.cmd == "prescreen":
        prescreen(a)
    else:
        assert a.dry or (a.only is None and a.chunks is None and not a.force_bad and a.reps == G.REPS), "dry options in a registered session"
        run(a)


if __name__ == "__main__":
    main()
