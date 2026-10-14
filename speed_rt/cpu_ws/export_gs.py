"""export_gs.py: Decoder Switching (gapsw) records of a cpu_ws run directory -> drive_gs's layout, which speed_rt/gapsw/analysis_gs.py reads
(STAGE2_gapsw 1.10). Copies values and changes none (in the window records, absolute paths are kept by file name only).

    export_gs(R, out, dry=False, manifest=None, sets_file=None)      R = export.load_run(run)
        OUT/session/     session.json, attempts.json, slots/<slot>/a<k>/{rec.json, stdout.txt, stderr.txt}, windows/<w>.json
        OUT/prescreen/   prescreen.json, PROBE_REF, sessionref.json, topo.json
    fit_from_gref(R, path, dry=False)    a TEST fit_s2 (grace_fit.fit over the run's gref records, the shape analysis_gs reads)

Slot r<rep>-<fe|term>-d<d> (drive_gs.plan); its attempts are the run's processes of that plan job, in the windows w... and their repeats r...
Status (drive_gs.classify, plus C1's window rule): accepted = window passed, exit 0, record accepted; rejected = exit 4 by the process's
own host check (its record, reasons copied), or exit 0 in a window cpu_ws rejected ("a process of a rejected window becomes a rejected
attempt"; the window's reasons are in the entry, the window record in windows/<w>.json); failed = any other exit, a missing or unreadable
record. Slot status: accepted; not established (three attempts, none accepted); incomplete (fewer, none accepted); not run.

PRESCREEN. drive_gs's pre-screen is a pod's own probe run (core pick, /proc/stat selftest, 5 s idle, three probe processes, reference =
median of their passes). cpu_ws has none of those: its reference is the session reference (3 segments of plain PM per block on every
slot, sref/sessionref.json; its median is the `--probe-ref` every process was given, to 4 decimals) and its host checks are per window
(window.json: ref_before/ref_after, disk, jobs) and per process (pre_idle, pre_probe, window in rec.json). So prescreen.json holds:
probe.values_us = the 24 reference values, probe.ref_us = the value the processes were given, spread = max/min - 1 (recorded, decides
nothing: C1 gates windows, not the reference); passed = the session reference is sound (every reference process exit 0 with a value; its
median re-derives reference_us; the processes' --probe-ref is reference_us to 4 decimals; no block read during it; the slots are topo's;
every attempt's core is one of them; at least 4 slots). selftest and idle are not run by cpu_ws: ok = None."""
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from datetime import datetime

MAX_ATTEMPTS = 3
JOB = re.compile(r"^gs_(fe|term)_d(\d+)_r(\d+)(?:_a(\d+))?$")
TFMT = "%Y-%m-%dT%H:%M:%S%z"


def _json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _put(path, obj):
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def _sha(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _arg(argv, name):
    i = argv.index(name) if name in argv else -1
    return argv[i + 1] if 0 <= i < len(argv) - 1 else None


def _num(v):
    try:
        return float(v) or None                       # "0" is the harness's "no reference"
    except (TypeError, ValueError):
        return None


def _t(s):
    return datetime.strptime(s, TFMT)


def _count(xs):
    return {x: xs.count(x) for x in sorted(set(xs))}


def _rel(x):
    """A path in a record by its file name only (records name files relatively)."""
    return os.path.basename(os.path.normpath(x)) if isinstance(x, str) and os.path.isabs(x) else x


def _slot(job):
    """A plan or attempt job id -> (slot id, kind, d, rep); None when it is not a Decoder Switching job."""
    m = JOB.match(job)
    return (f"r{m[3]}-{m[1]}-d{m[2]}", m[1], int(m[2]), int(m[3])) if m else None


def _model(topo):
    for line in (((topo or {}).get("host") or {}).get("lscpu") or "").splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return None


def _winrec(w):
    """A window record as cpu_ws wrote it, paths by file name only."""
    w = json.loads(json.dumps(w))
    for j in w.get("jobs", []):
        j["argv"] = [_rel(x) for x in j.get("argv", [])]
        j["out"] = _rel(j.get("out"))
    return w


def _why(w):
    """Why cpu_ws rejected the window (from its record alone)."""
    r = []
    dd = w.get("disk_delta") or {}
    if not dd.get("passed"):
        r.append(f"disk: reads {dd.get('reads')}, major faults {dd.get('pgmajfault')}, swap-ins {dd.get('pswpin')}")
    for k in ("check_before", "check_after"):
        c = w.get(k) or {}
        if not c.get("passed"):
            r.append(f"{k}: " + (c["why"] if "why" in c else f"worst {c.get('worst_rel')} x the session reference, spread {c.get('spread')}"))
    if not w.get("jobs_ok"):
        r.append("jobs: exit codes " + ",".join(str(j.get("rc")) for j in w.get("jobs", [])))
    return r or ["window not passed"]


def _attempt(R, sdir, sid, k, a, win, rec):
    """One attempt: slots/<sid>/a<k> (the process's own files, copied) and its attempts.json entry."""
    src, rel = os.path.join(R["run"], a["dir"]), f"slots/{sid}/a{k}"
    os.makedirs(os.path.join(sdir, rel))
    for f in sorted(set(a.get("files", [])) | {"stdout.txt", "stderr.txt"}):
        if os.path.isfile(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(sdir, rel, "rec.json.unreadable" if f == "rec.json" and rec is None else f))
    rc = a["rc"]
    if a["accepted"] and rec is not None and rec.get("accepted"):
        status, why = "accepted", []
    elif rc == 4 and rec is not None and not rec.get("failed") and not rec.get("accepted"):
        status, why = "rejected", list(rec.get("reasons", []))
    elif rc == 0 and rec is not None and rec.get("accepted") and not a["window_passed"]:
        status, why = "rejected", [f"window {a['window']} rejected by cpu_ws: " + "; ".join(_why(win))]
    else:
        status, why = "failed", list((rec or {}).get("reasons", [])) + [f"exit {rc}" + ("" if rec else ", no record")]
    return dict(k=k, rc=rc, status=status, reasons=why, seconds=a["wall_s"], dir=rel, core=a["core"],
                pick=dict(rule="cpu_ws: the vCPU of the window's slot, its SMT sibling idle", slot=a["slot"]),
                window=a["window"], window_file=f"windows/{a['window']}.json", window_passed=bool(a["window_passed"]),
                job=a["job"], attempt=a["attempt"], harness_dir=a["dir"])


def _status(ents):
    if any(e["status"] == "accepted" for e in ents):
        return "accepted"
    return "not established" if len(ents) >= MAX_ATTEMPTS else ("incomplete" if ents else "not run")


def _prescreen(R, pdir, probe_ref, cores, cpu, cgroup, limits, node):
    sref, topo = R.get("sessionref") or {}, R.get("topo") or {}
    host = topo.get("host") or {}
    cells = [x for s in sref.get("segments") or [] for x in s]
    good = [x["per_block_us"] for x in cells if x.get("per_block_us") is not None]
    ref_us, slots = sref.get("reference_us"), host.get("slots") or []
    procs = bool(cells) and len(good) == len(cells) and all(x.get("rc") == 0 for x in cells)
    med = bool(good) and ref_us is not None and math.isclose(statistics.median(good), ref_us, rel_tol=1e-9)
    same = probe_ref is not None and ref_us is not None and abs(probe_ref - ref_us) <= 5.0001e-5
    checks = dict(sessionref_processes=procs, sessionref_median=med, probe_ref_is_reference=same,
                  disk=bool((sref.get("disk_delta") or {}).get("passed")),
                  slots=bool(slots) and sref.get("slots") == slots and set(cores) <= set(slots), cpus=len(slots) >= 4)
    pre = dict(item="pre-screen", source="cpu_ws: the session reference (sessionref.json) and topo.json; windows/<w>.json hold the window verdicts",
               core=None, pick=dict(rule="per attempt: the vCPU of the window's slot (topo slots), SMT sibling idle"), limits=limits, cpu=cpu,
               cgroup_cpu=cgroup, cpus_allowed=host.get("vcpus"), cpus_online=host.get("vcpus"), start=sref.get("start"), node=node,
               selftest=dict(ok=None, run=False, note="cpu_ws runs no /proc/stat selftest"),
               idle=dict(ok=None, note="each process keeps its own 5 s idle check (pre_idle in rec.json)"),
               probe=dict(values_us=good, ref_us=probe_ref, spread=(max(good) / min(good) - 1) if good else None, ok=procs and med and same),
               session_reference_us=ref_us, cpus_ok=len(slots) >= 4, checks=checks,
               failed=sorted(k for k, v in checks.items() if not v), end=sref.get("end"))
    pre["passed"] = not pre["failed"]
    _put(os.path.join(pdir, "prescreen.json"), pre)
    if pre["passed"] and probe_ref is not None:
        with open(os.path.join(pdir, "PROBE_REF"), "x") as fh:
            fh.write(f"{probe_ref!r}\n")
    for f in ("sref/sessionref.json", "topo.json"):
        if os.path.isfile(os.path.join(R["run"], f)):
            shutil.copyfile(os.path.join(R["run"], f), os.path.join(pdir, os.path.basename(f)))
    return pre


def export_gs(R, out, dry=False, manifest=None, sets_file=None):
    """Writes OUT/session and OUT/prescreen (fresh directories) -> summary. manifest / sets_file: the draw's srt-gs.json and sets_gs.json, whose
    sha256 session.json records (default: the --manifest of the processes' argv, and speed_rt.gapsw.gslib.SETS_FILE when importable)."""
    global MAX_ATTEMPTS
    MAX_ATTEMPTS = R.get("max_attempts", MAX_ATTEMPTS)
    A = [a for a in R["attempts"] if a.get("part") in ("gs_fe", "gs_term")]
    W = {w["window"]: w for w in R["windows"]}
    missing, plan, by = [], {}, {}
    for jid in [i for i, j in R["jobs"].items() if j.get("part") in ("gs_fe", "gs_term")] + [a["job"] for a in A]:
        s = _slot(jid)
        if s:
            plan[s[0]] = (s[3], s[2], s[1] == "term")            # drive_gs.plan's order: rep-major, per d the front end, then the term
        else:
            missing.append(f"job {jid}: not a Decoder Switching job id")
    for a in A:
        s = _slot(a["job"])
        if s:
            by.setdefault(s[0], []).append(a)
    if not by:
        return dict(error="no Decoder Switching attempt in the run", missing=missing)
    for v in by.values():
        v.sort(key=lambda a: a["attempt"])
    recs = {a["dir"]: _json(os.path.join(R["run"], a["dir"], "rec.json")) for a in A}
    rl = [r for r in recs.values() if r]
    env = next((r["env"] for r in rl if isinstance(r.get("env"), dict)), {})
    cpu = env.get("cpu") or _model(R.get("topo"))
    commits = sorted({r["release_commit"] for r in rl if r.get("release_commit")})
    refs = [_num(_arg(a["argv"], "--probe-ref")) for a in A]
    probe_ref = refs[0]
    if probe_ref is None or len(set(refs)) > 1:
        missing.append(f"probe reference: {sorted(set(refs), key=str)}")
    if len(commits) > 1:
        missing.append(f"release_commit differs between records: {commits}")
    mpath = manifest or next((p for p in (_arg(a["argv"], "--manifest") for a in A) if p), None)
    msha = _sha(mpath)
    if sets_file is None:
        try:
            from speed_rt.gapsw import gslib
            sets_file = gslib.SETS_FILE
        except ImportError:
            sets_file = os.environ.get("SRT_GS_SETS")
    ssha = _sha(sets_file)
    if msha is None:
        missing.append(f"manifest_sha256: no file {mpath}")
    if ssha is None and not dry:
        missing.append(f"sets_sha256: no file {sets_file}")
    wins = sorted({a["window"] for a in A})
    sdir, pdir = os.path.join(out, "session"), os.path.join(out, "prescreen")
    os.makedirs(out, exist_ok=True)
    os.makedirs(sdir)
    os.makedirs(os.path.join(sdir, "windows"))
    os.makedirs(pdir)
    for w in wins:
        _put(os.path.join(sdir, "windows", f"{w}.json"), _winrec(W[w]))
    ids = [s for s in sorted(plan, key=plan.get) if by.get(s) or not dry]
    slots = {}
    for sid in ids:
        rep, d, term = plan[sid]
        kind = "term" if term else "fe"
        ents = [_attempt(R, sdir, sid, k, a, W[a["window"]], recs[a["dir"]]) for k, a in enumerate(by.get(sid, []), 1)]
        if any(e["status"] == "accepted" for e in ents[:-1]):
            missing.append(f"{sid}: an attempt after the accepted one")
        slots[sid] = dict(kind=kind, rep=rep, d=d, argv=[kind, "--d", str(d)], attempts=ents, status=_status(ents))
    start, end = min((W[w]["start"] for w in wins), key=_t), max((W[w]["end"] for w in wins), key=_t)
    host = (R.get("topo") or {}).get("host") or {}
    sess = dict(item="Decoder Switching timing session (cpu_ws run)", run=R["name"], start=start, end=end, core=None,
                core_rule="per attempt: the vCPU of the window's slot (cpu_ws), SMT sibling idle; the core of every attempt is in attempts.json",
                probe_ref_us=probe_ref, cpu=cpu, cgroup_cpu=env.get("cgroup_cpu"), cpus_allowed=host.get("vcpus"), node=env.get("node"),
                deployed=commits[0] if commits else None, dry=bool(dry), slots=ids, reps=max(plan[s][0] for s in ids),
                max_attempts=MAX_ATTEMPTS, manifest_sha256=msha, sets_sha256=ssha, windows=wins,
                argv=["cpu_ws harness.py run (export_gs.export_gs)"])
    _put(os.path.join(sdir, "session.json"), sess)
    _put(os.path.join(sdir, "attempts.json"), dict(session=R["name"], slots=slots, end=end, seconds=round((_t(end) - _t(start)).total_seconds(), 1)))
    pre = _prescreen(R, pdir, probe_ref, {a["core"] for a in A}, cpu, env.get("cgroup_cpu"), next((r["limits"] for r in rl if "limits" in r), None),
                     env.get("node"))
    return dict(session=sdir, prescreen=pdir, prescreen_passed=pre["passed"], prescreen_failed=pre["failed"], slots=len(ids),
                slot_status=_count([v["status"] for v in slots.values()]),
                attempts=_count([e["status"] for v in slots.values() for e in v["attempts"]]),
                windows=len(wins), not_established=[k for k, v in slots.items() if v["status"] != "accepted"], missing=missing)


def fit_from_gref(R, path, dry=False):
    """A TEST fit for analysis_gs --fit: grace_fit.fit over the slowest per_shot_us of the accepted gref records of every point (three
    accepted repetitions per point unless dry), written flat (grace_fit.fit's output plus raw_points) to `path`. The stage's fit is
    analysis_surf's fit_s2.json. Needs speed_rt on sys.path. -> dict(complete, ...)."""
    from speed_rt import grace_fit
    reps = {}
    for a in R["attempts"]:
        if a.get("part") == "gref" and a["accepted"]:
            for f in a.get("files", []):
                r = _json(os.path.join(R["run"], a["dir"], f)) if f.startswith("ref_") and f.endswith(".json") else None
                if r and r.get("what") == "ref":
                    reps.setdefault((r["decoder"], r["d"], r["p"]), []).append(r["per_shot_us"])
    pts = {k: max(v) for k, v in reps.items() if len(v) >= (1 if dry else 3)}
    missing = [list(k) for k in grace_fit.points() if k not in pts]
    if missing:
        return dict(complete=False, missing=missing)
    raw = {f"{k[0]}_d{k[1]}_p{round(k[2] * 1000)}": dict(decoder=k[0], d=k[1], p=k[2], xeon_us_per_shot=pts[k], repetitions_us_per_shot=sorted(reps[k]),
                                                         statistic="slowest of the accepted repetitions") for k in sorted(pts)}
    fit = dict(grace_fit.fit(pts), what="TEST fit from the run's gref records (export_gs.fit_from_gref): never the stage's fit_s2.json",
               dry=bool(dry), raw_points=raw)
    with open(path, "x") as fh:
        json.dump(fit, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    return dict(complete=True, points=len(pts), path=path)
