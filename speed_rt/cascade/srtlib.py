"""Shared code of SPEED-RT's Cascade launchers (srt_cas.py for the BB72 harnesses, srt_d13.py for the d = 13 pipeline).

A launcher changes three things in a timing process of a pinned tree, from outside the tree, and nothing else:

  seeds    the speed stream, seed kind 12 of the original studies, takes the beacon seeds of speed_rt/sets.py: chunk j of
           the stream at (family, p) or (d = 13, p) is chunk j of the matching srt-cas-* set (sets_cas.json). A kind-12
           chunk outside the sets raises; every other kind falls through to the tree's own function.
  digests  every chunk the process samples is hashed (sha256 of the bit-packed detector rows, then the observable
           bytes; its first 16 hex digits are surface.sv_common.digest's) and written to SRT_TRACE with its set, index and
           seed, so that each process proves the shots it timed.
  window   every timed window is judged in-process: nvidia-smi samples the card every 100 ms (SM clock, maximum SM
           clock, the active clock-event reasons, the enforced and the default power limit) from before the window to
           its end, and lists the compute applications every second. The window is rejected for outside interference:
           another compute application on the card, a gap of more than 1 s between samples, a clock-event reason other
           than GPU idle, applications clocks or SW power cap, an enforced power limit below the card's default, or an SM
           clock below 90 % of the maximum in a sample without the SW power cap. The workload's own power cap (SW power
           cap at the default limit) is recorded, not rejected, and so is App. D's flag (samples below 97 %). The verdict
           goes to SRT_WINDOWS (one JSON line per window) and, for the BB harnesses, into the row's "clock" field in the
           format of bb.timing.gpu_watch (clock_min_ratio, other_pids, accepted) plus the window's numbers. A burn before
           the window (SRT_BURN_S seconds of dense bf16 matrix products) exists for tests only: every registered job runs
           with 0 s.

Environment (set by the caller, usually drive.py):
  SRT_SEEDS    seeds file: the output of `python speed_rt/sets.py BEACON OUT` or of `python srtlib.py seeds ...` below
  SRT_BEACON   the beacon record SRT_SEEDS was made from (loaded by speed_rt.beacon; each seed is recomputed from it)
  SRT_DRY      1 when the record is a dry one ("dry": true): a dry record is accepted only then, and only then
  SRT_TRACE    file that receives one JSON line per sampled chunk
  SRT_WINDOWS  file that receives one JSON line per timed window
  SRT_BURN_S   seconds of burn before each window (default 0; registered jobs: 0)
  SRT_TAG      label written into every window line (drive.py: the slot and attempt)
  SRT_ATTEMPT, SRT_FORCE_BAD   dry runs only: the attempts numbered below SRT_FORCE_BAD (SRT_ATTEMPT counts from 0)
               report their windows as rejected, to exercise the replacement path
  BB_PIDNS / SRT_PIDNS   1 inside a container with its own PID namespace (a pod): "no other compute app" then means at
               most one distinct PID on the card

Standard library at import time (numpy and torch are imported inside the functions that need them). Command line, to
make the seeds file of the srt-cas sets before they are merged into config.json, and the dry lock of the d = 13 job:
    python srtlib.py seeds --beacon BEACON --out SEEDS
    python srtlib.py dry-lock --lock LOCK --out DRYLOCK --chunks 2 --single 2000 --batched 20000"""
import atexit
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # the repository root, which holds speed_rt/
KIND = 12                                              # the speed stream's seed kind in the original harnesses
SM_FLOOR = 0.90                                        # the clock rule: a sample below this share of the maximum
                                                       # rejects the window unless the card's power cap is active in it
APPX_D = 0.97                                          # App. D's flag (share of samples below it): recorded, decides nothing
POWER_CAP = 0x4                                        # NVML clock-event reason "SW power cap": the workload at the card's limit
ALLOWED = 0x1 | 0x2 | POWER_CAP                        # reasons a window may show: GPU idle, applications clocks, SW power cap;
                                                       # any other (HW slowdown, sync boost, SW/HW thermal, HW power brake,
                                                       # display clock) is outside interference and rejects the window
MAX_GAP_S = 1.0                                        # a window whose samples have a longer gap is rejected
IDENTITY_KEYS = ("kind", "code", "d", "family", "p", "chunks", "chunk_shots")


def _srt():
    """speed_rt.sets and speed_rt.beacon, imported from the repository root; the root and SURF-v2's folder (which
    speed_rt.beacon puts first on sys.path) are taken off again, so that the release's own bb/ folder cannot join the
    tree's bb package as a second namespace part (as speed_rt/bb/srt_bb.py does)."""
    sv2 = os.path.join(ROOT, "surface_v2")
    had = ROOT in sys.path
    if not had:
        sys.path.insert(0, ROOT)
    try:
        from speed_rt import beacon, sets
    finally:
        for p in (sv2, ROOT):
            while p in sys.path and not (p == ROOT and had):
                sys.path.remove(p)
    return sets, beacon


def key(p):
    return int(round(p * 1e6))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


D13_REPLACED = ("identity.gpu", "identity.real_executable", "identity.prefix", "identity.packages_sha256", "kcache")


def lock_rest_sha256(lock):
    """sha256 of a d = 13 lock (a dict) without the fields an H100 derivation replaces (D13_REPLACED) and without its
    h100_derivation record: the same value for the study's lock and for a lock derived from it (srt_d13.py h100-lock), so a
    derived lock whose rest differs from the study's is not a derivation."""
    L = json.loads(json.dumps(lock))
    L.pop("h100_derivation", None)
    L.pop("kcache", None)
    for k in ("gpu", "real_executable", "prefix", "packages_sha256"):
        L["identity"].pop(k, None)
    return hashlib.sha256(json.dumps(L, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def set_defs():
    """{name: entry} of the srt-cas sets: sets_cas.json, which config.json's copies must equal once they are merged.
    SRT_SETS_FILE names another sets file of the same sets under other names (sets_arb3.json): the seeds follow the
    same rule from the same beacon record, with each set's own TAG, so the shots are new (unregistered timings only)."""
    alt = os.environ.get("SRT_SETS_FILE")             # an unregistered timing's own sets (new names, so new TAGs)
    defs = json.load(open(alt or os.path.join(HERE, "sets_cas.json")))["sets"]
    cfg = {k: v for k, v in _srt()[0].sets().items() if k.startswith("srt-cas-")} \
        if os.path.exists(os.path.join(ROOT, "speed_rt", "config.json")) and not alt else {}
    if cfg:
        if set(cfg) != set(defs):
            raise SystemExit(f"config.json's srt-cas sets {sorted(cfg)} are not sets_cas.json's {sorted(defs)}")
        for k, v in cfg.items():
            if any(defs[k].get(f) != v.get(f) for f in IDENTITY_KEYS):
                raise SystemExit(f"{k}: config.json and sets_cas.json differ")
    return defs


def record():
    """(the beacon record named by SRT_BEACON, dry flag), checked; a dry record only with SRT_DRY=1, and then it is
    required."""
    path = os.environ.get("SRT_BEACON")
    if not path:
        raise SystemExit("srtlib: SRT_BEACON is not set")
    dry = os.environ.get("SRT_DRY") == "1"
    rec = _srt()[1].load(path, dry_ok=dry)
    if bool(rec.get("dry")) != dry:
        raise SystemExit(f"{path}: SRT_DRY={'1' if dry else 'unset'} and the record disagree (a dry record needs SRT_DRY=1, "
                         f"a timing record must not be used with it)")
    return rec, dry


def check_seeds(seeds_path, defs):
    """The seeds file against the beacon record, every seed recomputed -> {name: [seeds]} of the srt-cas sets."""
    rec, dry = record()
    f = json.load(open(seeds_path))
    if f.get("beacon_round") != rec["round"] or f.get("randomness") != rec["randomness"] or bool(f.get("dry")) != dry:
        raise SystemExit(f"{seeds_path} was not made from the beacon record {os.environ['SRT_BEACON']}")
    S = _srt()[0]
    out = {}
    for name, e in sorted(defs.items()):
        want = [S.seed(rec["randomness"], name, j) for j in range(e["chunks"])]
        if f["seeds"].get(name) != want:
            raise SystemExit(f"{seeds_path}: the seeds of {name} differ from the beacon's")
        out[name] = want
    flat = [x for v in out.values() for x in v]
    if len(set(flat)) != len(flat) or min(flat) < 2 ** 62 or max(flat) >= 2 ** 63:
        raise SystemExit("srtlib: the srt-cas seeds are not distinct values in [2^62, 2^63)")
    return out


def tables():
    """{"bb": {(family, round(1e6 p)): (set, seeds)}, "d13": {round(1e6 p): (set, seeds)}} from SRT_SEEDS / SRT_BEACON."""
    if not os.environ.get("SRT_SEEDS"):
        raise SystemExit("srtlib: SRT_SEEDS is not set (a Cascade timing process always draws beacon shots)")
    defs = set_defs()
    seeds = check_seeds(os.environ["SRT_SEEDS"], defs)
    bb, d13 = {}, {}
    for name, e in defs.items():
        if "family" in e:
            bb[(e["family"], key(e["p"]))] = (name, seeds[name])
        else:
            d13[key(e["p"])] = (name, seeds[name])
    return {"bb": bb, "d13": d13}


# ---------------------------------------------------------------------------------------------------------- digests
def chunk_digests(det, obs, n):
    """sha256 (64 hex digits) of each n-shot chunk of (det, obs): the bit-packed detector rows, then the observable
    bytes. The first 16 digits are surface.sv_common.digest's."""
    import numpy as np
    if len(det) % n or len(obs) != len(det):
        raise SystemExit(f"srtlib: {len(det)} detector rows are not whole chunks of {n}")
    out = []
    for i in range(len(det) // n):
        h = hashlib.sha256(np.packbits(np.asarray(det[i * n:(i + 1) * n], np.uint8), axis=1).tobytes())
        h.update(np.asarray(obs[i * n:(i + 1) * n], np.uint8).reshape(-1).tobytes())
        out.append(h.hexdigest())
    return out


def write_line(env_var, rec):
    path = os.environ.get(env_var)
    if not path:
        raise SystemExit(f"srtlib: {env_var} is not set")
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def trace_chunks(name, seeds, chunks, det, obs, n, extra):
    for c, dg in zip(chunks, chunk_digests(det, obs, n)):
        write_line("SRT_TRACE", dict(pid=os.getpid(), set=name, chunk=int(c), seed=seeds[int(c)], n=n, sha256=dg,
                                     det_shape=list(det.shape), obs_shape=list(obs.shape), **extra))


# -------------------------------------------------------------------------------------------------------- the window
def burn(seconds):
    """Dense bf16 matrix products on this process's device for `seconds` -> seconds spent. A local generator keeps the
    global random state of the process untouched."""
    if seconds <= 0:
        return 0.0
    import torch
    t0 = time.time()
    g = torch.Generator(device="cuda")
    g.manual_seed(20261013)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    c = torch.empty_like(a)
    end = t0 + seconds
    while time.time() < end:
        for _ in range(16):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    del a, b, c
    return time.time() - t0


def _ts(s):
    return datetime.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def burn_seconds():
    return float(os.environ.get("SRT_BURN_S", "0"))


class Window:
    """One timed window: begin() starts the samplers, burns, and marks the window's start; end() stops the samplers and
    returns the verdict {clock_min_ratio, other_pids, accepted, ...} (bb.timing.gpu_watch's three keys plus the window's
    own numbers) after writing it to SRT_WINDOWS."""

    def __init__(self, tag):
        self.tag = tag
        self.pidns = "1" in (os.environ.get("BB_PIDNS"), os.environ.get("SRT_PIDNS"), os.environ.get("G2_PIDNS"))
        self.smi = ["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]]
        self.done = False

    def begin(self, burn_s):
        self.cf, self.af = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
        self.pc = subprocess.Popen(self.smi + ["--query-gpu=timestamp,clocks.sm,clocks.max.sm,clocks_event_reasons.active,"
                                               "enforced.power.limit,power.default_limit",
                                               "--format=csv,noheader,nounits", "-lms", "100"],
                                   stdout=self.cf, stderr=subprocess.DEVNULL)
        self.pa = subprocess.Popen(self.smi + ["--query-compute-apps=timestamp,pid", "--format=csv,noheader",
                                               "-lms", "1000"], stdout=self.af, stderr=subprocess.DEVNULL)
        atexit.register(self.stop_samplers)                   # nvidia-smi -l never ends by itself
        t = time.time()
        while os.fstat(self.cf.fileno()).st_size == 0:        # the clock sampler is live before the window starts
            if time.time() - t > 15:
                raise SystemExit("srtlib: nvidia-smi gave no clock sample in 15 s")
            time.sleep(0.05)
        self.burned = burn(burn_s)
        self.t0 = time.time()
        return self

    @staticmethod
    def _hist(inw):
        h = {}
        for c in inw:
            h[str(c[1])] = h.get(str(c[1]), 0) + 1
        return h

    def stop_samplers(self):
        for p in (self.pc, self.pa):
            if p.poll() is None:
                p.terminate()

    def end(self):
        t1 = time.time()
        if self.done:
            raise SystemExit("srtlib: a window ended twice")
        self.done = True
        time.sleep(0.3)                                        # the samples up to t1 reach the file
        for p in (self.pc, self.pa):
            p.terminate()
        for p in (self.pc, self.pa):
            p.wait(timeout=15)
        self.cf.seek(0)
        self.af.seek(0)
        clk, apps = [], []
        for ln in self.cf.read().splitlines():
            f = [x.strip() for x in ln.split(",")]
            if len(f) == 6 and f[1].isdigit() and f[2].isdigit() and int(f[2]):
                try:                                   # (time, SM MHz, max SM MHz, reasons, enforced W, default W)
                    clk.append((_ts(f[0]), int(f[1]), int(f[2]), int(f[3], 16), float(f[4]), float(f[5])))
                except ValueError:
                    pass
        for ln in self.af.read().splitlines():
            f = [x.strip() for x in ln.split(",")]
            if len(f) == 2 and f[1].isdigit():
                try:
                    apps.append((_ts(f[0]), int(f[1])))
                except ValueError:
                    pass
        self.cf.close()
        self.af.close()
        now = subprocess.run(self.smi + ["--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.split()
        inw = [c for c in clk if self.t0 <= c[0] <= t1]
        pids = {p for t, p in apps if self.t0 - 1.0 <= t <= t1 + 1.0} | {int(x) for x in now if x.isdigit()}
        own = os.getpid()
        if self.pidns:                                         # host PIDs: exactly one compute app on the card
            others = sorted(pids) if len(pids) > 1 else []
        else:
            others = sorted(x for x in pids if x != own)
        ratio = min(c[1] / c[2] for c in inw) if inw else None
        ts = [c[0] for c in inw]
        gaps = ([ts[0] - self.t0] + [b - a for a, b in zip(ts, ts[1:])] + [t1 - ts[-1]]) if ts else []
        gap = max(gaps) if gaps else None
        capped = [c for c in inw if c[3] & POWER_CAP]
        low = [c for c in inw if c[1] < SM_FLOOR * c[2] and not c[3] & POWER_CAP]
        reasons = 0
        for c in inw:
            reasons |= c[3]
        why = []
        if not inw:
            why.append("no clock sample in the window")
        if low:
            why.append(f"SM clock {min(c[1] / c[2] for c in low):.4f} of the maximum without the power cap")
        if reasons & ~ALLOWED:
            why.append(f"clock-event reasons {reasons & ~ALLOWED:#x} (outside interference)")
        if any(c[4] < c[5] for c in inw):
            why.append(f"power limit {min(c[4] for c in inw):.0f} W below the default {max(c[5] for c in inw):.0f} W")
        if gap is not None and gap > MAX_GAP_S:
            why.append(f"sampler gap {gap:.2f} s")
        if others:
            why.append(f"other compute apps {others}")
        forced = os.environ.get("SRT_DRY") == "1" and bool(os.environ.get("SRT_FORCE_BAD")) and \
            int(os.environ.get("SRT_ATTEMPT", "0")) < int(os.environ["SRT_FORCE_BAD"])
        if forced:
            why.append("forced rejection (dry run)")
        res = dict(clock_min_ratio=ratio, other_pids=others, accepted=not why, reason="; ".join(why), tag=self.tag,
                   samples=len(inw), min_mhz=min((c[1] for c in inw), default=None),
                   max_mhz=max((c[2] for c in inw), default=None), gap_max_s=gap, window_s=t1 - self.t0,
                   below=sum(c[1] < APPX_D * c[2] for c in inw), power_capped=len(capped), reasons=f"{reasons:#x}",
                   power_limit_w=min((c[4] for c in inw), default=None),
                   power_default_w=max((c[5] for c in inw), default=None), low_uncapped=len(low), hist=self._hist(inw),
                   burn_s=self.burned, t0=self.t0, t1=t1, forced=forced)
        write_line("SRT_WINDOWS", dict(pid=own, **res))
        return res


# ----------------------------------------------------------------------------------------------- BB72 harness patches
def install_bb():
    """bb.cascade_data.eval_seed / eval_shots and bb.timing.gpu_watch of the tree on PYTHONPATH -> the facts of the patch.
    Run at the launcher's top level: spawned sampling workers re-run it and patch their own copy of the module."""
    from bb import cascade_data as cd
    if getattr(cd.eval_seed, "_srt", None):
        return cd.eval_seed._srt
    tb = tables()["bb"]
    orig_seed, orig_shots = cd.eval_seed, cd.eval_shots

    def eval_seed(kind, p, chunk, family="uniform"):
        base = orig_seed(kind, p, chunk, family)           # the tree's own range checks and value
        if kind != KIND:
            return base
        k = (family, key(p))
        if k not in tb or p != k[1] / 1e6 or not 0 <= chunk < len(tb[k][1]):
            raise ValueError(f"srtlib: kind {KIND} at family {family!r}, p = {p!r}, chunk {chunk!r} is not in a srt-cas "
                             f"set ({sorted(v[0] for v in tb.values())})")
        return tb[k][1][chunk]

    def eval_shots(code, p, chunks, n=10000, kind=0, procs=1, family="uniform"):
        chunks = list(chunks)
        det, obs = orig_shots(code, p, chunks, n=n, kind=kind, procs=procs, family=family)
        if kind == KIND:
            name, seeds = tb[(family, key(p))]
            trace_chunks(name, seeds, chunks, det, obs, n, dict(kind=kind, family=family, p=p, code=code))
        return det, obs

    for f, o in ((eval_seed, orig_seed), (eval_shots, orig_shots)):
        f.__module__, f.__qualname__, f.__doc__ = o.__module__, o.__qualname__, o.__doc__
    facts = dict(kind=KIND, sets=sorted(v[0] for v in tb.values()), seeds=os.environ["SRT_SEEDS"],
                 beacon=os.environ["SRT_BEACON"])
    eval_seed._srt = facts
    cd.eval_seed, cd.eval_shots = eval_seed, eval_shots
    import bb.timing as bt

    def gpu_watch():
        """The clock window of the BB harnesses: burn, then sample; the returned stop() gives the verdict (Window)."""
        return Window(os.environ.get("SRT_TAG", "bb")).begin(burn_seconds()).end

    bt.gpu_watch = gpu_watch
    return facts


# ------------------------------------------------------------------------------------------------ d = 13 harness patch
def install_d13():
    """surface.surface_data.eval_seed / eval_shots of the tree on PYTHONPATH (kind 12 at d = 13 only) -> the facts."""
    from surface import surface_data as sd
    if getattr(sd.eval_seed, "_srt", None):
        return sd.eval_seed._srt
    tb = tables()["d13"]
    orig_seed, orig_shots = sd.eval_seed, sd.eval_shots

    def eval_seed(d, kind, p, chunk):
        base = orig_seed(d, kind, p, chunk)                # the tree's own range checks and value
        if kind != KIND:
            return base
        k = key(p)
        if d != 13 or k not in tb or p != k / 1e6 or not 0 <= int(chunk) < len(tb[k][1]):
            raise ValueError(f"srtlib: kind {KIND} at d = {d}, p = {p!r}, chunk {chunk!r} is not in a srt-cas set")
        return tb[k][1][int(chunk)]

    def eval_shots(d, p, chunks, n=10000, kind=0, procs=1):
        chunks = list(chunks)
        det, obs = orig_shots(d, p, chunks, n=n, kind=kind, procs=procs)
        if kind == KIND:
            name, seeds = tb[key(p)]
            trace_chunks(name, seeds, chunks, det, obs, n, dict(kind=kind, d=d, p=p))
        return det, obs

    for f, o in ((eval_seed, orig_seed), (eval_shots, orig_shots)):
        f.__module__, f.__qualname__, f.__doc__ = o.__module__, o.__qualname__, o.__doc__
    facts = dict(kind=KIND, sets=sorted(v[0] for v in tb.values()), seeds=os.environ["SRT_SEEDS"],
                 beacon=os.environ["SRT_BEACON"])
    eval_seed._srt = facts
    sd.eval_seed, sd.eval_shots = eval_seed, eval_shots
    return facts


# ------------------------------------------------------------------------------------------------------ command line
def make_seeds(beacon_path, out):
    """The seeds file of the srt-cas sets (the format of speed_rt/sets.py's output) from a beacon record."""
    S = _srt()[0]
    b = json.load(open(beacon_path))
    r = b["randomness"]
    assert len(r) == 64, "no 32-byte randomness in the beacon record"
    defs = set_defs()
    seeds = {n: [S.seed(r, n, j) for j in range(e["chunks"])] for n, e in sorted(defs.items())}
    flat = [x for v in seeds.values() for x in v]
    assert len(set(flat)) == len(flat), "two chunks share a seed"
    rec = dict(rule=S.RULE, beacon_round=b["round"], randomness=r, dry=bool(b.get("dry")),
               tags={n: S.tag(n).decode() for n in seeds}, seeds=seeds)
    with open(out, "x") as fh:
        json.dump(rec, fh, indent=1)
    print(f"{len(flat)} seeds of {len(seeds)} sets -> {out}")


def dry_lock(lock, out, chunks, single, batched):
    """A copy of the d = 13 lock whose c7 plan is shortened (chunks 0 .. chunks-1, `single` and `batched` shots), for the
    dry runs of R4. The identity, thresholds, frozen paths and kernel-cache digest stay the lock's, so the pipeline's own
    checks run unchanged; the copy is not the lock (its sha256 differs) and the records it makes are never timing
    records."""
    L = json.load(open(lock))
    plan = L["pipes"]["c7"]
    if single > chunks * 10000 or batched > chunks * 10000:
        raise SystemExit("dry-lock: the shot counts need more chunks")
    plan.update({"chunks": [plan["chunks"][0], plan["chunks"][0] + chunks], "single": single, "batched": batched})
    with open(out, "x") as fh:
        json.dump(L, fh, indent=1, sort_keys=True)
    print(f"dry lock {out}: c7 plan {plan}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds")
    s.add_argument("--beacon", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("dry-lock")
    s.add_argument("--lock", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--chunks", type=int, default=2)
    s.add_argument("--single", type=int, default=2000)
    s.add_argument("--batched", type=int, default=20000)
    a = ap.parse_args()
    if a.cmd == "seeds":
        make_seeds(a.beacon, a.out)
    else:
        dry_lock(a.lock, a.out, a.chunks, a.single, a.batched)
