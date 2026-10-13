"""Quiet-host timing (bb/DESIGN.md 5.3-5.5, bb/PLAN.md 2.7): the composite's CPU term, the CPU opponents' batch-1 times
and the speed gate. One single-threaded decoder per process (pin it: taskset -c CORE), one shot per decode call, the
wall time of each call. Rows go to one CSV (FIELDS) shared with bb.latency (the GPU term and the handoff) and
bb.gpu_bposd time --timing-out (cudaq-qec's batch-1 call). Every row carries its session (the Slurm job), its process
(a fresh id per process) and its binding: the engine signature's digest (GPU term, handoff), the policy file's sha256
(CPU term) or the decoder configuration (cudaq-qec).
  routed    python -m bb.timing routed --policy P --split dev --chunks 0-19 --proc 1 --out T.csv [--first N]: the
            policy's fallback on its routed shots (margin < tau in the policy network's read of these shots), in split
            order, the first --first of them (default FIRST: 5,000 on BB72, 1,000 on gross); value = the routed
            fraction x their mean time (the CPU term per shot); the row records this population (population())
  opponent  python -m bb.timing opponent --code bb72 --p 0.003 --dec relay --split dev --chunks 0 --first 10000
            --proc 1 --out T.csv: the decoder's mean time per shot on these shots (its batch-1 time)
  gate      python -m bb.timing gate --rows T.csv --policy P --gpu-session J1 --cpu-session J2 [--split dev] [--out
            G.json]: only the rows of the two sessions count, each from a distinct process: composite = the slowest of
            at least 5 GPU-term processes bound to the policy's engine signature + the slowest of at least 3 CPU-term
            processes bound to the policy file and measured on the gated split's (dev or test) complete routed
            population (every chunk of the split in the shot manifest, the routed set recomputed from the policy
            network's read, the first FIRST timed) + max(1.1 us, the slowest handoff, measured in the GPU-term
            processes); limit = the fastest gating opponent, each counted at its fastest of at least 3 processes,
            cudaq-qec only in its registered configuration (CUDAQ) on the policy's GPU (the engine signature's device);
            GPU-side rows (GPU term, handoff, cudaq-qec) count only if gpu_watch accepted them, and a third rejected
            process of the network or of cudaq-qec fails the session; pass iff the session holds and composite <= limit
            (PLAN 2.6 (ii), 2.7)"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import uuid

import numpy as np

FIELDS = ["date", "host", "device", "role", "code", "p", "name", "proc", "n", "value_ms", "session", "process",
          "binding", "note"]
SESSION = os.environ.get("SLURM_JOB_ID") or os.environ.get("BB_SESSION", "")   # stage 3: a label (STAGE3_PLAN.md 4.3)
PROCESS = f"{platform.node()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"     # a fresh id in every process
FIRST = {"bb72": 5000, "bb144": 1000}      # PLAN 2.7: routed shots timed per CPU-term process, the first in split order
CUDAQ = "ms80_cs60"                        # PLAN Q1: cudaq-qec's gating configuration (bb.gpu_bposd CONFIGS)


def sig_digest(sig):
    """What a GPU-term or handoff row is bound to: the engine signature (bb.engine.Deployed.signature) as a digest."""
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def append_row(path, row):
    row = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(), "session": SESSION,
           "process": PROCESS, **row}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    print("timing", json.dumps(row), flush=True)


def _cpu():
    if not os.path.exists("/proc/cpuinfo"):                # macOS (stage 3's quiet CPU host)
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip() or "?"
    return next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")


def corecheck():
    """STAGE3_PLAN.md 0: seconds of a fixed integer loop at the start of a CPU-timing process, recorded with its row
    (macOS cannot pin a core type; an efficiency core takes about twice as long)."""
    import time
    t, s = time.perf_counter(), 0
    for i in range(3_000_000):
        s += i & 7
    return time.perf_counter() - t


def gpu_watch():
    """SETTINGS' acceptance of a GPU timing process: nvidia-smi samples this GPU's SM clock every 100 ms until the
    returned stop() is called, which gives {clock_min_ratio (smallest clock / maximum clock, None if unreadable),
    other_pids (other processes on the GPU), accepted (every sample >= 97 % and no other process)}."""
    import subprocess
    smi = ["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]]
    pidns = os.environ.get("BB_PIDNS") == "1"              # STAGE3_PLAN.md 4.1: the pod's PID namespace
    q = "clocks.sm,clocks.max.sm" + (",utilization.gpu,memory.used" if pidns else "")
    p = subprocess.Popen(smi + [f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms", "100"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def stop():
        p.terminate()
        lines = p.communicate()[0].splitlines()
        if not pidns:
            pairs = [ln.split(",") for ln in lines if ln.count(",") == 1]
            r = [int(a) / int(b) for a, b in pairs if a.strip().isdigit() and b.strip().isdigit() and int(b)]
        else:
            quads = [[x.strip() for x in ln.split(",")] for ln in lines if ln.count(",") == 3]
            quads = [x for x in quads if all(v.isdigit() for v in x) and int(x[1])]
            r = [int(x[0]) / int(x[1]) for x in quads]
        apps = subprocess.run(smi + ["--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.split()
        if pidns:                                          # host PIDs: exactly one compute app on the card
            others = [] if len(apps) == 1 else apps
        else:
            others = [int(x) for x in apps if x.isdigit() and int(x) != os.getpid()]
        clock = min(r) if r else None
        out = {"clock_min_ratio": clock, "other_pids": others,
               "accepted": clock is not None and clock >= 0.97 and not others}
        if pidns:
            out.update(samples=len(r), apps=len(apps), util_max=max((int(x[2]) for x in quads), default=None),
                       mem_max_mib=max((int(x[3]) for x in quads), default=None))
        return out
    return stop


def idle(a):
    """STAGE3_PLAN.md 4.1: just before a GPU timing process starts, two device-wide samples 100 ms apart must show
    utilization 0 % and less than 1,024 MiB used (no other context on the card); a row records them; exit 3 if not."""
    import subprocess
    import time
    smi = ["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
           "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    s = []
    for _ in range(2):
        s.append([int(x) for x in subprocess.run(smi, capture_output=True, text=True).stdout.split(",")])
        time.sleep(0.1)
    ok = all(u == 0 and m < 1024 for u, m in s)
    append_row(a.out, {"device": "gpu", "role": "idle", "name": "pre-process check", "n": 2, "value_ms": 0,
                       "note": json.dumps({"samples_util_mem": s, "ok": ok})})
    raise SystemExit(0 if ok else 3)


def _secs(dec, code, p, det):
    """Wall time of each single-shot decode call of decoder dec (bb.opp's configuration) in this process."""
    from bb import opp
    opp._init(dec, code, p)
    return opp._dec.decode(det)[1]


def population(pol, split, chunks, first):
    """The CPU term's shots (PLAN 2.7): the shots of these chunks the policy routes (margin < tau in its network's read),
    in split order, the first `first` of them timed. Returns their indices and the record a cpu_term row carries; the
    gate counts a row only if its record is the gated split's complete one (every manifest chunk, FIRST timed)."""
    from bb import route
    _, margin, sig, _ = route._load_net(pol["code"], pol["p"], split, chunks, pol["tag"])
    if sig != pol["engine"]:
        raise SystemExit("the read's engine signature differs from the policy's")
    idx = np.flatnonzero(margin < pol["policy"]["tau"])
    return idx[:first], {"split": split, "chunks": list(chunks), "shots": len(margin), "routed": len(idx),
                         "routed_sha": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()[:16],
                         "timed": len(idx[:first])}


def routed(a):
    from bb import route, shots
    pol = route.guard(a.policy) if a.split in ("dev", "test") else json.load(open(a.policy))
    code, p = pol["code"], pol["p"]
    chunks = shots.parse_range(a.chunks)
    cc = corecheck()
    idx, pop = population(pol, a.split, chunks, a.first or FIRST[code])
    frac = pop["routed"] / pop["shots"]
    det = shots.load_many(code, p, a.split, chunks)[0][idx]
    secs = _secs(pol["fallback"], code, p, det) if len(det) else np.zeros(1)
    append_row(a.out, {"device": _cpu(), "role": "cpu_term", "code": code, "p": p, "name": os.path.basename(a.policy),
                       "proc": a.proc, "n": len(det), "value_ms": frac * 1e3 * float(secs.mean()),
                       "binding": hashlib.sha256(open(a.policy, "rb").read()).hexdigest()[:16],
                       "note": json.dumps({"routed_fraction": frac, "mean_ms_routed": 1e3 * float(secs.mean()),
                                           "population": pop, "corecheck_s": cc})})


def opponent(a):
    from bb import shots
    cc = corecheck()
    det = shots.load_many(a.code, a.p, a.split, shots.parse_range(a.chunks))[0][:a.first or None]
    secs = _secs(a.dec, a.code, a.p, det)
    append_row(a.out, {"device": _cpu(), "role": "opponent", "code": a.code, "p": a.p, "name": a.dec, "proc": a.proc,
                       "n": len(det), "value_ms": 1e3 * float(secs.mean()),
                       "note": json.dumps({"median_ms": 1e3 * float(np.median(secs)),
                                           "p99_ms": 1e3 * float(np.quantile(secs, 0.99)), "split": a.split,
                                           "chunks": a.chunks, "corecheck_s": cc})})


def gate(a):
    from bb import shots
    from bb.route import GATING, HANDOFF_US
    if a.split not in ("dev", "test"):
        raise SystemExit("speed gate: --split is the gated split, dev or test")
    raw = open(a.policy, "rb").read()
    pol = json.loads(raw)
    code, p = pol["code"], pol["p"]
    # the GPU term and the handoff are per network (bb.latency times each network once, under its default p), bound to
    # the policy by the engine signature that every cell of the network shares; all other rows are per cell
    rows = [r for r in csv.DictReader(open(a.rows)) if r["code"] == code and
            (float(r["p"]) == p or r["role"] in ("gpu_term", "handoff"))]
    session = {"gpu": a.gpu_session, "cpu": a.cpu_session}
    eng = sig_digest(pol["engine"])
    full = sorted(int(r["chunk"]) for r in shots._manifest().values()
                  if (r["code"], float(r["p"]), r["split"]) == (code, p, a.split))
    if not full:
        raise SystemExit(f"speed gate: no {a.split} chunks of {code} p {p} in the shot manifest")
    pop = population(pol, a.split, full, FIRST[code])[1]
    on_pop = lambda r: r["note"].startswith("{") and json.loads(r["note"]).get("population") == pop
    on_gpu = lambda r: r["device"] == pol["engine"]["gpu"]
    # key: (role, name or None, binding or None, side, processes needed, further condition or None): the CPU term on
    # the gated split's complete routed population, cudaq-qec in its registered configuration on the policy's GPU
    spec = {"gpu_term": ("gpu_term", None, eng, "gpu", 5, None), "handoff": ("handoff", None, eng, "gpu", 5, None),
            "cpu_term": ("cpu_term", None, hashlib.sha256(raw).hexdigest()[:16], "cpu", 3, on_pop),
            "relay": ("opponent", "relay", None, "cpu", 3, None), "bposd": ("opponent", "bposd", None, "cpu", 3, None),
            "gpu_bposd": ("opponent", "gpu_bposd", CUDAQ, "gpu", 3, on_gpu)}
    need, rejected = {}, {}
    for k, (role, name, bind, side, m, cond) in spec.items():
        out = [r for r in rows if r["role"] == role and r["session"] == session[side] and
               (name is None or r["name"] == name) and (bind is None or r["binding"] == bind) and
               (cond is None or cond(r))]
        if side == "gpu":                                  # GPU-side rows count only if accepted (gpu_watch)
            ok = [r for r in out if r["note"].startswith("{") and json.loads(r["note"]).get("accepted")]
            rejected[k] = len(out) - len(ok)
            out = ok
        need[k] = out
    rejected = {k: rejected[k] for k in ("gpu_term", "gpu_bposd")}
    if max(rejected.values()) > 2:                         # PLAN 2.7: a third rejection fails the session
        out = {"policy": os.path.abspath(a.policy), "sessions": session, "rejected": rejected, "session_holds": False,
               "pass": False}
        print(json.dumps(out, indent=1), flush=True)
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)
        return
    short = {k: len(need[k]) for k, s in spec.items() if len(need[k]) < s[4]}
    twice = [k for k, v in need.items() if len({r["process"] for r in v}) != len(v)]
    hosts = {side: {r["host"] for k, s in spec.items() if s[3] == side for r in need[k]} for side in session}
    if short or twice or any(len(h) != 1 for h in hosts.values()):
        raise SystemExit(f"speed gate: too few processes {short}, one process counted twice {twice}, or a side "
                         f"from several hosts {hosts} (CPU-term rows count only on the {a.split} split's complete "
                         f"routed population: {pop['shots']} shots, {pop['routed']} routed, the first {pop['timed']} "
                         f"timed; cudaq-qec rows only {CUDAQ} on {pol['engine']['gpu']})")
    if {r["process"] for r in need["handoff"]} != {r["process"] for r in need["gpu_term"]}:
        raise SystemExit("speed gate: the handoff rows are not the GPU-term processes'")
    val = lambda k: [float(r["value_ms"]) for r in need[k]]
    terms = {"gpu_ms": max(val("gpu_term")), "cpu_ms": max(val("cpu_term")),
             "handoff_ms": max(HANDOFF_US / 1e3, max(val("handoff")))}
    fastest = {o: min(val(o)) for o in GATING}
    out = {"policy": os.path.abspath(a.policy), "sessions": session, "hosts": {k: v.pop() for k, v in hosts.items()},
           "split": a.split, "cpu_population": {k: v for k, v in pop.items() if k != "chunks"},
           "processes": {k: len(v) for k, v in need.items()}, **terms, "composite_ms": sum(terms.values()),
           "opponents_ms": fastest, "limit_ms": min(fastest.values()), "rejected": rejected, "session_holds": True}
    out["pass"] = out["composite_ms"] <= out["limit_ms"]
    print(json.dumps(out, indent=1), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["routed", "opponent", "gate", "idle"])
    ap.add_argument("--policy")
    ap.add_argument("--code")
    ap.add_argument("--p", type=float)
    ap.add_argument("--dec")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--chunks", default="0")
    ap.add_argument("--first", type=int, default=0, help="at most this many shots (0 = all; routed: FIRST)")
    ap.add_argument("--proc", type=int, default=1, help="index of this fresh process in the session")
    ap.add_argument("--rows")
    ap.add_argument("--gpu-session", help="gate: the Slurm job of the GPU-side session")
    ap.add_argument("--cpu-session", help="gate: the Slurm job of the CPU-side session")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "gate" and not (a.gpu_session and a.cpu_session):
        raise SystemExit("gate needs --gpu-session and --cpu-session")
    {"routed": routed, "opponent": opponent, "gate": gate, "idle": idle}[a.cmd](a)


if __name__ == "__main__":
    main()
