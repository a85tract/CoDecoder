"""Launcher of SPEED-RT's Cascade timing processes on the surface code at d = 13 (R4): one command of the pinned tree's
`surface` package (PYTHONPATH = the tree; its code digest is the lock's), unchanged, with the changes of srtlib.py:
beacon seeds for kind 12 at d = 13, a sha256 trace of every sampled chunk, and a burn plus an in-process clock window
around the pipeline's timed part (the tree has no sampler of its own: e7 was judged afterwards, from the job's clock
file, by surface/scripts/clockflags.py and winmark.py).

    python srt_d13.py pipeline --config C5 --p 0.006 --mode single --rep 1 --lock LOCK --stage c7 --out ROWS --save DIR
    python srt_d13.py --check                the patch installs and the seeds are the beacon's (prints the facts)

The arguments are sv_speed's. The window is placed as surface/scripts/winmark.py places its marks, by Python 3.12's
sys.monitoring LINE events at lines of surface.sv_speed.pipeline found by their exact text (the launcher refuses if the
text is not there exactly once, so it can never read another function): the burn and the window's start come just before
the timed part's first line, and its end at the line after the timed part,
  single   start: `lat = np.empty(n)` (before the loop, whose first 200 shots are untimed: the window holds them, as the
           BB harnesses' windows do); end: the first `fail = pred != obs` after the loop line `lat[i] = ...`
  batched  start: `qr, qi, qn = [], [], 0` (just before t0); end: the line after `wall = time.perf_counter() - t0`
so the burn is over before the harness takes its first clock reading. Each callback disables itself after its one call,
so the timed loops run uninstrumented. The window's verdict is one line in SRT_WINDOWS (srtlib.Window); an error inside
a callback is written there too, as a window that is not accepted and carries "error".
Card: SRT_CARD = rtx (default: nothing below applies) or h100 (drive.py --card h100). The lock pins the RTX PRO 6000, the
venv it was made in and the digest of the Triton / Inductor caches (an sm_120 build, with the cache's directory path in it),
and surface.sv_common refuses any other card. On the H100 the launcher admits exactly "NVIDIA H100 80GB HBM3" (it sets
sv_common.GPU; a MIG slice's name differs, so it is refused) and the pipeline runs against a lock DERIVED from the study's
lock and the session's own record: every field is the lock's (thresholds, frozen paths, nets, plan, the torch version and
the code digest) except the four identity fields that name the card and the venv (gpu, real_executable, prefix,
packages_sha256) and the cache digest, which are the session's (SRT_SESSION, written once on the pod by `h100-session`
after the pod's caches were built untimed). The pipeline's own check then compares the cache after its compiles with the
session's digest, and this launcher compares it again after the pipeline returns; a process that changed the caches
fails (its window line carries the error). Commands:
    python srt_d13.py h100-session --out SESSION.json          the session's record (pod, GPU, venv, code, caches)
    python srt_d13.py h100-lock --lock LOCK --session SESSION.json --out DERIVED.json
    SRT_KCACHE_BUILD=1 (with SRT_DRY=1 only): the cache build's untimed processes; the cache digest check is skipped
The joint decision (surface/sv_joint.py, drive.py r4 --joint-w13) changes the code digest and adds a kernel to the caches,
so on either card its pipelines run against a lock derived for the joint tree:
    SRT_KCACHE_BUILD=1 SRT_DRY=1 (untimed dry passes, any card)  then
    python srt_d13.py session --out SESSION.json               the session's record on this card (SRT_CARD)
    python srt_d13.py joint-lock --lock LOCK --session SESSION.json --out DERIVED.json
joint-lock replaces the lock's code digest and kernel-cache digest with the session's (on the H100 also the card and venv
fields, as h100-lock) and records joint_derivation (and, on the H100, h100_derivation for the checks below).
Why the pipeline runs as an imported module: its spawn pools (the PyMatching workers) re-run the parent's __main__ as
__mp_main__, which here is this file, so each worker patches its own surface_data before it samples."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srtlib  # noqa: E402

H100 = "NVIDIA H100 80GB HBM3"
CARD = os.environ.get("SRT_CARD", "rtx").lower()
if CARD not in ("rtx", "h100"):
    raise SystemExit(f"srt_d13: SRT_CARD={CARD!r}, not rtx or h100")
HELPER = __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("h100-session", "h100-lock", "session",
                                                                           "joint-lock")
FACTS = None if HELPER else srtlib.install_d13()   # at import: the main process and every spawned worker
SESSION_FIELDS = ("gpu", "real_executable", "prefix", "packages_sha256")      # the lock's identity keys that name card and venv


def find(src, text, lo, hi):
    """The 1-based numbers of the lines lo..hi whose stripped text is `text`."""
    return [k + 1 for k in range(lo - 1, min(hi, len(src))) if src[k].strip() == text]


def install_window(sv, mode, tag):
    """LINE callbacks at the window's two lines of sv.pipeline; -> the state dict {win, res, errors}."""
    code = sv.pipeline.__code__
    src = open(code.co_filename).read().splitlines()
    lo, hi = code.co_firstlineno, max(ln for _, _, ln in code.co_lines() if ln)
    one = lambda text, a=lo: find(src, text, a, hi)
    if mode == "single":
        loop = one("lat[i] = 1e6 * (t1 - t0)")
        start = one("lat = np.empty(n)")
        if len(loop) != 1 or len(start) != 1 or not start[0] < loop[0]:
            raise SystemExit(f"srt_d13: {code.co_filename} is not the pipeline this launcher reads (single mode)")
        stop = one("fail = pred != obs", loop[0])
        if not stop:
            raise SystemExit("srt_d13: no `fail = pred != obs` after the single-mode loop")
        want = {start[0]: "begin", stop[0]: "end"}
    else:
        start = one("qr, qi, qn = [], [], 0")
        wall = one("wall = time.perf_counter() - t0")
        if len(start) != 1 or len(wall) != 1 or src[wall[0]].strip() != "if pool is not None:":
            raise SystemExit(f"srt_d13: {code.co_filename} is not the pipeline this launcher reads (batched mode)")
        want = {start[0]: "begin", wall[0] + 1: "end"}
    M = sys.monitoring
    tool = next(k for k in range(6) if M.get_tool(k) is None)
    M.use_tool_id(tool, "srt_d13")
    st = {"win": None, "res": None, "errors": []}

    def line(_, ln):
        what = want.get(ln)
        if what is None:
            return M.DISABLE
        try:
            if what == "begin":
                st["win"] = srtlib.Window(tag).begin(srtlib.burn_seconds())
            elif st["win"] is None:
                raise RuntimeError("the window's end came before its start")
            else:
                st["res"] = st["win"].end()
        except BaseException as e:                         # never into the pipeline
            st["errors"].append(repr(e))
            srtlib.write_line("SRT_WINDOWS", dict(pid=os.getpid(), accepted=False, error=repr(e), tag=tag))
        return M.DISABLE

    M.register_callback(tool, M.events.LINE, line)
    M.set_local_events(tool, code, M.events.LINE)
    return st


def session_record(out):
    """The H100 session's record, written once: the card, driver, torch, the venv's identity, the code digest and the digest
    of the caches under TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR (the pipeline's own functions, sv_common.record)."""
    import json
    sc = importlib.import_module("surface.sv_common")
    if CARD == "h100":
        sc.GPU = H100
    _, gpu = sc.setup_gpu()
    rec = sc.record(gpu)
    keep = ("gpu", "driver", "torch", "cuda", "cudnn", "triton", "capability", "real_executable", "prefix", "packages",
            "packages_sha256", "code", "kcache", "tree", "cpu", "host", "time")
    res = {k: rec[k] for k in keep}
    res.update(card=CARD, kcache_dirs={v: os.environ.get(v, "") for v in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")})
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"srt_d13 session: {res['gpu']} driver {res['driver']} torch {res['torch']} code {res['code']['digest']} "
          f"kcache {res['kcache']} -> {out}", flush=True)


def derive_lock(lock, session, out):
    """The study's lock with the session's card, venv and cache digest in place of the lock's, written once. Refused unless
    the session's torch version and code digest are the lock's."""
    import copy
    import json
    L, S = json.load(open(lock)), json.load(open(session))
    I = L["identity"]
    if S["card"] != "h100" or S["gpu"] != H100:
        raise SystemExit(f"srt_d13: {session} is not an {H100} session")
    for k, v in (("torch", S["torch"]), ("code", S["code"])):
        if I[k] != v:
            raise SystemExit(f"srt_d13: the session's {k} {v} is not the lock's {I[k]}")
    D = copy.deepcopy(L)
    for k in SESSION_FIELDS:
        D["identity"][k] = S[k]
    D["kcache"] = dict(S["kcache"])
    D["h100_derivation"] = dict(lock_sha256=srtlib.sha256_file(lock), session_sha256=srtlib.sha256_file(session),
                                replaced=list(srtlib.D13_REPLACED), rest_sha256=srtlib.lock_rest_sha256(L), card="h100")
    assert srtlib.lock_rest_sha256(D) == D["h100_derivation"]["rest_sha256"]
    with open(out, "x") as fh:
        json.dump(D, fh, indent=1, sort_keys=True)
    print(f"srt_d13 derived lock {out} from {lock}: {SESSION_FIELDS} and kcache {S['kcache']} are the session's", flush=True)


def joint_lock(lock, session, out):
    """The study's lock for the joint decision's tree on the session's card, written once: the session's code digest and
    kernel caches (on the H100 also its card and venv fields, SESSION_FIELDS) in place of the lock's; everything else
    (thresholds, frozen paths, nets, plan, torch) is the lock's. Refused unless the session's torch is the lock's."""
    import copy
    import json
    L, S = json.load(open(lock)), json.load(open(session))
    if L["identity"]["torch"] != S["torch"]:
        raise SystemExit(f"srt_d13: the session's torch {S['torch']} is not the lock's {L['identity']['torch']}")
    D = copy.deepcopy(L)
    repl = ["identity.code", "kcache"]
    D["identity"]["code"] = S["code"]
    if S["card"] == "h100":
        for k in SESSION_FIELDS:
            D["identity"][k] = S[k]
        repl += [f"identity.{k}" for k in SESSION_FIELDS]
        D["h100_derivation"] = dict(lock_sha256=srtlib.sha256_file(lock), session_sha256=srtlib.sha256_file(session),
                                    replaced=list(srtlib.D13_REPLACED) + ["identity.code"],
                                    rest_sha256=srtlib.lock_rest_sha256(L), card="h100")
    D["kcache"] = dict(S["kcache"])
    D["joint_derivation"] = dict(lock_sha256=srtlib.sha256_file(lock), session_sha256=srtlib.sha256_file(session),
                                 replaced=sorted(repl), card=S["card"], code=S["code"])
    with open(out, "x") as fh:
        json.dump(D, fh, indent=1, sort_keys=True)
    print(f"srt_d13 joint lock {out} from {lock}: {sorted(repl)} are the session's ({S['card']})", flush=True)


def h100_start(argv, sv):
    """Before the pipeline: the H100's checks -> the session record. The lock must be the one derived from SRT_SESSION."""
    import json
    sess = os.environ.get("SRT_SESSION")
    if not sess or "--lock" not in argv:
        raise SystemExit("srt_d13: SRT_CARD=h100 needs SRT_SESSION (the session's record) and --lock (the derived lock)")
    S, L = json.load(open(sess)), json.load(open(argv[argv.index("--lock") + 1]))
    d = L.get("h100_derivation") or {}
    if d.get("session_sha256") != srtlib.sha256_file(sess) or L["identity"]["gpu"] != H100:
        raise SystemExit(f"srt_d13: the lock is not derived from the session {sess}")
    sc = sys.modules["surface.sv_common"]
    sc.GPU = H100                                          # setup_gpu admits exactly this name
    if os.environ.get("SRT_KCACHE_BUILD") == "1":          # untimed cache build: dry runs only, the digest check skipped
        if os.environ.get("SRT_DRY") != "1":
            raise SystemExit("srt_d13: SRT_KCACHE_BUILD is for dry runs (SRT_DRY=1)")
        want = L["kcache"]
        real = sc.kcache
        sc.kcache = lambda: {**want, "build": real()["digest"]}
        return S, False
    return S, True


def main(argv):
    if argv == ["--check"]:
        print(f"srt_d13: eval_seed patched for kind {FACTS['kind']} at d = 13: sets {FACTS['sets']}, card {CARD}", flush=True)
        return
    if argv and argv[0] in ("h100-session", "session"):
        return session_record(argv[argv.index("--out") + 1])
    if argv and argv[0] == "joint-lock":
        v = lambda f: argv[argv.index(f) + 1]                                          # noqa: E731
        return joint_lock(v("--lock"), v("--session"), v("--out"))
    if argv and argv[0] == "h100-lock":
        v = lambda f: argv[argv.index(f) + 1]
        return derive_lock(v("--lock"), v("--session"), v("--out"))
    if not argv or argv[0] != "pipeline" or "--mode" not in argv:
        raise SystemExit("usage: srt_d13.py pipeline --mode single|batched ... (sv_speed's arguments) | --check | "
                         "h100-session --out F | h100-lock --lock L --session S --out O")
    mode = argv[argv.index("--mode") + 1]
    if mode not in ("single", "batched"):
        raise SystemExit(f"srt_d13: --mode {mode!r}")
    print(f"srt_d13: surface.sv_speed with beacon seeds for kind {FACTS['kind']} (sets {len(FACTS['sets'])}), card {CARD}, "
          f"burn {srtlib.burn_seconds():g} s before the window", file=sys.stderr, flush=True)
    sys.argv = ["sv_speed"] + list(argv)
    sv = importlib.import_module("surface.sv_speed")
    sess, check_end = h100_start(argv, sv) if CARD == "h100" else (None, False)
    if CARD != "h100" and os.environ.get("SRT_KCACHE_BUILD") == "1":    # the joint tree's untimed cache build on the RTX
        if os.environ.get("SRT_DRY") != "1":
            raise SystemExit("srt_d13: SRT_KCACHE_BUILD is for dry runs (SRT_DRY=1)")
        import json
        sc_ = sys.modules["surface.sv_common"]
        want, real = json.load(open(argv[argv.index("--lock") + 1]))["kcache"], sc_.kcache
        sc_.kcache = lambda: {**want, "build": real()["digest"]}
    st = install_window(sv, mode, os.environ.get("SRT_TAG", "d13"))
    sv.main()
    tag = os.environ.get("SRT_TAG")
    if st["res"] is None and not st["errors"]:             # the pipeline returned without reaching the window's end
        srtlib.write_line("SRT_WINDOWS", dict(pid=os.getpid(), accepted=False, error="no window", tag=tag))
    if check_end:                                          # the caches after the pipeline: still the session's
        now = sys.modules["surface.sv_common"].kcache()
        if now["digest"] != sess["kcache"]["digest"]:
            srtlib.write_line("SRT_WINDOWS", dict(pid=os.getpid(), accepted=False, tag=tag,
                              error=f"kernel caches {now} after the pipeline, not the session's {sess['kcache']}"))


if __name__ == "__main__":
    main(sys.argv[1:])
