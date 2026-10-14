"""drive_par.py: the Decoder Switching stage's session on a CPU host that runs several timed processes at once (speed_rt/gapsw;
the registered rule: at most 8 timed processes at a time, each on its own physical core with the SMT sibling left idle, the
reference segments under the same concurrency, the two sides interleaved; the stage's Grace points are the surface family's). drive_gs.py runs one process at a
time (a pod); this driver runs the same slots, the same processes (gs_time.py), the same host checks and the same failure rule
with `--workers` of them at once. The analysis (analysis_gs.py) reads either driver's records.

    python -m speed_rt.gapsw.drive_par cpus --n 8 [--from LIST]
        prints N logical CPUs of distinct physical cores (the allowed ones, outside the physical core of the lowest allowed CPU, which
        is the driver's), the lowest first
    python -m speed_rt.gapsw.drive_par prescreen --cpus LIST --probe-file F --out DIR
        the pre-screen of drive_gs.py on the first listed CPU while untimed fillers (gs_time.py load) keep every other listed CPU busy,
        so the session's plain-PM reference is measured under the concurrency of the session
    python -m speed_rt.gapsw.drive_par run --cpus LIST --shots S --manifest M --masks K --probe-file F --probe-ref US --out OUT
        the timed session: the slots of drive_gs.plan, longest first (the order only sets the wall time), each worker (one per listed
        CPU) taking the next slot and running its attempts on its own CPU; a worker with nothing left runs a filler on its CPU until
        every worker is done, so each window sees as many busy cores as there are workers. Records as drive_gs.py's (attempts.json
        adds each attempt's start and end time, its CPU and the number of busy cores at its start).
    python -m speed_rt.gapsw.drive_par list [--reps 3]       the order of the slots

The failure rule is drive_gs.py's, unchanged: an attempt rejected by the host checks or failing is replaced at once on the same CPU,
at most twice; a slot with three rejected attempts is not established and the session goes on; a slot whose last attempt fails (not
rejected) stops the session (every process is terminated, exit 3: a technical failure, the whole session runs again)."""
import argparse
import collections
import fcntl
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from speed_rt.gapsw import drive_gs as D  # noqa: E402
from speed_rt.gapsw import gslib as G  # noqa: E402
from speed_rt.gapsw import hostwin as H  # noqa: E402

MAX_WORKERS = 8
# Seconds per slot on a Xeon 8470 (the dry session of 13 October: slot id without its repetition): the scheduling order only.
EST = {"fe-d7": 24, "term-d7": 10, "fe-d9": 49, "term-d9": 12, "fe-d11": 94, "term-d11": 17, "fe-d13": 190, "term-d13": 21,
       "fe-d15": 364, "term-d15": 31, "fe-d17": 666, "term-d17": 74}


def order(slots):
    """Longest first (ties: the slot id): the schedule that ends soonest. The sides (front ends, routed terms) meet
    every other side's load because the filler holds the number of busy cores constant to the end."""
    return sorted(slots, key=lambda s: (-EST.get(s["id"].split("-", 1)[1], 0), s["id"]))


def parse_cpus(text):
    return sorted(H.cpus(text))


def physical(cpu):
    return frozenset([cpu] + H.siblings(cpu))


def distinct_cores(cpus):
    """True when no two listed CPUs share a physical core (as far as sysfs tells; without a topology the CPUs are taken as given)."""
    seen = set()
    for c in cpus:
        ph = physical(c)
        if ph & seen:
            return False
        seen |= ph
    return True


def pick_cpus(n, within=None):
    allowed = sorted(os.sched_getaffinity(0))
    home = allowed[0]
    taken = set(physical(home))
    out = []
    for c in (sorted(within) if within else allowed):
        if c in allowed and not (physical(c) & taken):
            out.append(c)
            taken |= physical(c)
        if len(out) == n:
            break
    return out


def start_filler(cpu, probe_file):
    return subprocess.Popen(["taskset", "-c", str(cpu), sys.executable, "-m", "speed_rt.gapsw.gs_time", "load", "--core", str(cpu),
                             "--probe-file", probe_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=G.ROOT,
                            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))


def stop_filler(p):
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def pin_driver(cpus):
    """The driver itself stays off the timed CPUs and their siblings."""
    allowed = set(os.sched_getaffinity(0))
    avoid = set()
    for c in cpus:
        avoid |= physical(c)
    rest = sorted(allowed - avoid)
    if rest:
        os.sched_setaffinity(0, {rest[0]})
        return rest[0]
    return None


def prescreen(a):
    cpus = parse_cpus(a.cpus)
    assert len(cpus) >= 1 and distinct_cores(cpus), "the listed CPUs share physical cores"
    fillers = [start_filler(c, a.probe_file) for c in cpus[1:]]
    time.sleep(3.0)
    a.core = cpus[0]
    allowed = len(os.sched_getaffinity(0))
    a.extra = dict(concurrency=len(cpus), cpus=cpus, fillers=len(fillers), driver_cpu=pin_driver(cpus), cpus_allowed=allowed)
    try:
        D.prescreen(a)
    finally:
        for p in fillers:
            stop_filler(p)


def run(a):
    lock = open(os.environ.get("GS_LOCK", "/tmp/gs_timed.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another timed session holds the lock")
    cpus = parse_cpus(a.cpus)
    workers = len(cpus)
    assert a.dry or workers <= MAX_WORKERS, f"at most {MAX_WORKERS} timed processes at once"
    assert distinct_cores(cpus), "the listed CPUs share physical cores"
    assert a.dry or (a.only is None and a.chunks is None and not a.force_bad and a.reps == G.REPS), "dry options in a registered session"
    assert not os.path.exists(a.out), f"{a.out} exists: a session writes into a fresh directory"
    os.makedirs(os.path.join(a.out, "slots"))
    allowed_n = len(os.sched_getaffinity(0))
    driver = pin_driver(cpus)
    force = {}
    for x in (a.force_bad or "").split(","):
        if x:
            k, _, n = x.partition("=")
            force[k] = int(n or 1)
    chosen = [s for s in D.plan(a.reps) if not a.only or any(o in s["id"] for o in a.only)]
    slots = order(chosen)
    from convlines import provenance
    session = dict(item="Decoder Switching timing session", driver="drive_par", start=G.now(), core=None,
                   core_rule="parallel: each worker keeps its own listed CPU", cpus=cpus, workers=workers, driver_cpu=driver,
                   topology=dict(distinct=True, siblings={str(c): H.siblings(c) for c in cpus}),
                   probe_ref_us=a.probe_ref, cpu=G.cpu_model(), cgroup_cpu=G.cgroup_quota(),
                   cpus_allowed=allowed_n, node=os.uname().nodename, deployed=provenance.commit(G.ROOT),
                   dry=bool(a.dry), slots=[s["id"] for s in chosen], order=[s["id"] for s in slots], reps=a.reps,
                   max_attempts=G.MAX_ATTEMPTS, manifest_sha256=G.sha256_file(a.manifest), sets_sha256=G.sha256_file(G.SETS_FILE),
                   fillers=not a.no_fillers, argv=G.rel_argv(sys.argv))
    G.write_once(os.path.join(a.out, "session.json"), session)
    attempts = dict(session=os.path.basename(os.path.abspath(a.out)), slots={}, fillers=[])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1")
    lock_a = threading.Lock()
    queue = collections.deque(slots)
    abort, done = threading.Event(), threading.Event()
    state = dict(busy=0, fillers_running=0, finished=0, procs={}, total=len(slots))
    t_start = time.time()

    def save():                                   # caller holds lock_a
        tmp = os.path.join(a.out, "attempts.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(attempts, fh, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(a.out, "attempts.json"))

    def run_slot(i, s, cpu):
        with lock_a:
            ent = attempts["slots"][s["id"]] = dict(kind=s["kind"], rep=s["rep"], argv=s["argv"], attempts=[], status="running")
        for k in range(1, G.MAX_ATTEMPTS + 1):
            if abort.is_set():
                return
            adir = os.path.join(a.out, "slots", s["id"], f"a{k}")
            os.makedirs(adir)
            common = ["--rep", str(s["rep"]), "--core", str(cpu), "--probe-file", a.probe_file, "--probe-ref", repr(a.probe_ref),
                      "--shots", a.shots, "--manifest", a.manifest, "--out", adir]
            if s["kind"] in ("fe", "term"):
                common += ["--masks", a.masks]
            if a.chunks is not None:
                common += ["--chunks", str(a.chunks)]
            cmd = ["taskset", "-c", str(cpu), sys.executable, "-m", "speed_rt.gapsw.gs_time", s["argv"][0], *s["argv"][1:], *common]
            e = dict(env)
            if any(f in s["id"] and k <= n for f, n in force.items()):
                e["GS_FORCE_REJECT"] = "1"
            with lock_a:
                state["busy"] += 1
                busy = state["busy"] + state["fillers_running"]
            t0 = time.time()
            D.log(f"slot {i}/{state['total']} {s['id']} attempt {k} cpu {cpu} (timed processes running: {busy})")
            with open(os.path.join(adir, "log.txt"), "w") as lg:
                p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=e, cwd=G.ROOT)
                with lock_a:
                    state["procs"][cpu] = p
                rc = p.wait()
                with lock_a:
                    state["procs"].pop(cpu, None)
                    state["busy"] -= 1
            if abort.is_set():
                return
            status, rec, reasons = D.classify(rc, os.path.join(adir, "rec.json"))
            with lock_a:
                ent["attempts"].append(dict(k=k, rc=rc, status=status, reasons=reasons, seconds=round(time.time() - t0, 1),
                                            dir=os.path.relpath(adir, a.out), core=cpu, pick=dict(parallel=True, worker_cpu=cpu),
                                            t_start=round(t0, 2), t_end=round(time.time(), 2), busy_at_start=busy))
            D.log(f"  {s['id']} a{k} cpu {cpu}: {status} rc {rc} {time.time() - t0:.0f} s {reasons}")
            if status == "accepted":
                with lock_a:
                    ent["status"] = "accepted"
                    save()
                return
            if status == "failed" and k == G.MAX_ATTEMPTS:
                with lock_a:
                    ent["status"] = "failed"
                    save()
                D.log("the last attempt of a slot failed: technical failure, the session stops (exit 3)")
                abort.set()
                state["exit"] = 3
                return
            with lock_a:
                ent["status"] = "not established" if k == G.MAX_ATTEMPTS else "retry"
                save()

    def worker(cpu):
        while not abort.is_set():
            with lock_a:
                if not queue:
                    break
                i = state["total"] - len(queue) + 1
                s = queue.popleft()
            run_slot(i, s, cpu)
        with lock_a:
            state["finished"] += 1
            if state["finished"] == workers:
                done.set()
        if not a.no_fillers and not abort.is_set():           # keep every window at the same number of busy cores
            f0 = time.time()
            p = start_filler(cpu, a.probe_file)
            with lock_a:
                state["fillers_running"] += 1
            done.wait()
            stop_filler(p)
            with lock_a:
                state["fillers_running"] -= 1
                attempts["fillers"].append(dict(cpu=cpu, t_start=round(f0, 2), t_end=round(time.time(), 2)))
        else:
            done.wait()

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in cpus]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(1.0)
        if abort.is_set():
            with lock_a:
                for p in list(state["procs"].values()):
                    p.terminate()
            done.set()
            for t in threads:
                t.join(timeout=30)
            break
    for t in threads:
        t.join()
    if abort.is_set():
        with lock_a:
            save()
        raise SystemExit(state.get("exit", 3))
    attempts["end"] = G.now()
    attempts["seconds"] = round(time.time() - t_start, 1)
    with lock_a:
        save()
    bad = sorted(k for k, v in attempts["slots"].items() if v["status"] != "accepted")
    D.log(f"SESSION DONE in {attempts['seconds']:.0f} s on {workers} CPUs: {len(slots) - len(bad)} of {len(slots)} slots accepted; "
          f"not established: {bad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cpus")
    p.add_argument("--n", type=int, default=MAX_WORKERS)
    p.add_argument("--from", dest="within", default=None)
    p = sub.add_parser("prescreen")
    p.add_argument("--cpus", required=True)
    p.add_argument("--probe-file", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--cpus", required=True)
    p.add_argument("--shots", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--probe-file", required=True)
    p.add_argument("--probe-ref", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=G.REPS)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--force-bad", default="")
    p.add_argument("--no-fillers", action="store_true", help="dry runs only")
    p = sub.add_parser("list")
    p.add_argument("--reps", type=int, default=G.REPS)
    a = ap.parse_args()
    if a.cmd == "cpus":
        print(",".join(map(str, pick_cpus(a.n, H.cpus(a.within) if a.within else None))))
    elif a.cmd == "list":
        for s in order(D.plan(a.reps)):
            print(s["id"], " ".join(s["argv"]))
    elif a.cmd == "prescreen":
        prescreen(a)
    else:
        assert a.dry or not a.no_fillers, "no_fillers is for dry runs"
        run(a)


if __name__ == "__main__":
    main()
