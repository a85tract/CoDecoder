"""SPEED-RT shim for the frozen BB code (retest/shim/rt_bb.py with the minimal changes). It runs one command of
bb/bb72/ising_compare (BB72: stage 3 and amendment S) or bb/gross_stage4/ising_compare (the gross code: stage 4),
whichever tree is on PYTHONPATH, and changes only these names before the command runs, each only when its variable is set:
  SRT_BEACON    (a SPEED-RT beacon record, checked by speed_rt.beacon.load) bb.shots.seed_of on the three timing sets:
                chunk j of srt-bb72-p3 (code bb72, p 0.003, split "test"), srt-bb72-p2 (bb72, 0.002, "test") and
                srt-gross-p4 (bb144, 0.004, "test4") gets speed_rt.sets.seed(R, set, j), R = the record's randomness; a chunk
                outside a set, and any other test split, is refused; every other split and code goes to the frozen function
  SRT_MANIFEST  bb.shots.MANIFEST: the draw's manifest CSV, which shots.load checks every chunk against
  SRT_DRY=1     lets SRT_BEACON name a dry record (a past round: dry runs only, never a timing seed) and then requires it;
                SRT_DRY_SHOTS=N (dry runs only) draws chunks of N shots instead of 10,000
  SRT_ORIGINAL_SEEDS=1  keeps the frozen seeds of the original test sets (the draw's host check regenerates them)
With SRT_MANIFEST set and no SRT_BEACON, a test split has no seed (sampling is refused), so a timing step can
never draw shots by mistake. The splits keep their names ("test" in bb72's tree, "test4" in gross_stage4's), so every frozen
command, record and analysis reads them unchanged; shots, predictions and outputs live in their own directories (BB_SHOTS,
BB_PREDS). With none of the variables set the shim only launches the command.
    python srt_bb.py MODULE ARGS...        e.g. srt_bb.py bb.shots --code bb72 --p 0.003 --split test --chunks 0-19 --rows M.csv
    python srt_bb.py SCRIPT.py ARGS...     e.g. srt_bb.py retest/replay/bb_replay.py shots OUT.json
    python srt_bb.py --check-beacon        the record (SRT_BEACON, SRT_DRY) loads: prints round and randomness
    python srt_bb.py --list-sets           one line per BB set: name code p split chunks (sets_bb.json)
    python srt_bb.py --env-check OUT.json  x86-64 with AVX2, Stim 1.16.0, numpy 2.2.6 (the host the sets are drawn on)
    python srt_bb.py --check-manifest M.csv... --out OUT.json
        every row's seed recomputed from the record, every file's sha256, every set complete
Standard library only except for what the command itself imports; runs under Alpine's pycpu (Python 3.12) and the pod venvs."""
import csv
import hashlib
import importlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))              # speed_rt/bb/ -> the repository root
CHUNK = 10_000
SETS_FILE = os.path.join(HERE, "sets_bb.json")                # the BB stage's own sets file (the beacon record names it)


def _srt():
    """speed_rt.sets and speed_rt.beacon, imported from the repository root; the root and SURF-v2's folder (which
    speed_rt.beacon puts first on sys.path) are taken off again, so that the release's own bb/ folder cannot join the
    tree's bb package as a second namespace part."""
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


def bb_sets():
    """{(split, code, p): set name} of the BB sets file (sets_bb.json), and each set's entry."""
    sets, _ = _srt()
    ent = {n: e for n, e in sets.sets(SETS_FILE).items() if e["kind"] == "bb"}
    return {(e["split"], e["code"], round(e["p"], 6)): n for n, e in ent.items()}, ent


def record():
    """(the beacon record, dry flag) named by SRT_BEACON, checked; a dry record only with SRT_DRY=1 and then it is required."""
    path = os.environ.get("SRT_BEACON")
    if not path:
        raise SystemExit("SRT_BEACON is not set")
    dry = os.environ.get("SRT_DRY") == "1"
    _, beacon = _srt()
    rec = beacon.load(path, dry_ok=dry)
    if bool(rec.get("dry")) != dry:
        raise SystemExit(f"{path}: SRT_DRY={'1' if dry else 'unset'} and the record disagree (a dry record needs SRT_DRY=1, "
                         f"a timing record must not be used with it)")
    return rec, dry


def setup():
    """The patches of the module docstring; returns bb.shots."""
    from bb import shots
    frozen = shots.seed_of
    if os.environ.get("SRT_BEACON"):
        sets, _ = _srt()
        key = {(e["split"], e["code"], round(e["p"], 6)): n for n, e in sets.sets(SETS_FILE).items() if e["kind"] == "bb"}
        rec, dry = record()
        R = rec["randomness"]
        table = {k: [sets.seed(R, n, j) for j in range(sets.sets(SETS_FILE)[n]["chunks"])] for k, n in key.items()}

        def seed_of(split, code, p, j):
            k = (split, code, round(p, 6))
            if k in table:
                if not 0 <= j < len(table[k]):
                    raise ValueError(f"{k}: chunk {j} is outside the SPEED-RT set {key[k]}")
                return table[k][j]
            if split in ("test", "test4"):
                raise ValueError(f"{k}: no SPEED-RT set")
            return frozen(split, code, p, j)
        shots.seed_of = seed_of
        if os.environ.get("SRT_DRY_SHOTS"):
            if not dry:
                raise SystemExit("SRT_DRY_SHOTS with a record that is not a dry one")
            shots.sample.__defaults__ = (int(os.environ["SRT_DRY_SHOTS"]),)
    else:
        if os.environ.get("SRT_DRY_SHOTS"):
            raise SystemExit("SRT_DRY_SHOTS needs SRT_BEACON (a dry record)")
        if os.environ.get("SRT_MANIFEST") and os.environ.get("SRT_ORIGINAL_SEEDS") != "1":
            def seed_of(split, code, p, j):
                if split in ("test", "test4"):
                    raise ValueError(f"{(split, code, round(p, 6))}: SRT_MANIFEST is set without SRT_BEACON, so no test seed "
                                     f"(SRT_ORIGINAL_SEEDS=1 for the original sets)")
                return frozen(split, code, p, j)
            shots.seed_of = seed_of
    if os.environ.get("SRT_MANIFEST"):
        shots.MANIFEST = os.environ["SRT_MANIFEST"]
    return shots


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def env_check(out):
    """The draw host: x86-64 with AVX2, Stim 1.16.0, numpy 2.2.6 (every earlier test set)."""
    import numpy as np
    import stim
    flags = ""
    if os.path.exists("/proc/cpuinfo"):
        flags = next((ln for ln in open("/proc/cpuinfo") if ln.startswith("flags")), "")
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    rep = dict(machine=platform.machine(), avx2=" avx2" in flags, cpu=cpu, python=sys.version.split()[0], stim=stim.__version__,
               numpy=np.__version__, host=platform.node(), stim_modules=sorted(m for m in sys.modules if m.startswith("stim._")))
    rep["ok"] = bool(rep["machine"] == "x86_64" and rep["avx2"] and rep["stim"] == "1.16.0" and rep["numpy"] == "2.2.6")
    with open(out, "x") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep), flush=True)
    raise SystemExit(0 if rep["ok"] else 1)


def check_manifest(paths, out):
    """Every manifest row of the three sets against the record: the seed recomputed (speed_rt.sets.seed), the file's sha256
    recomputed from BB_SHOTS, the shots per chunk, and every set complete (chunks 0 .. n-1, once each)."""
    sets, _ = _srt()
    rec, dry = record()
    R = rec["randomness"]
    key, ent = bb_sets()
    shots_dir = os.path.expanduser(os.environ.get("BB_SHOTS", "~/work/shots"))
    want_n = int(os.environ["SRT_DRY_SHOTS"]) if (dry and os.environ.get("SRT_DRY_SHOTS")) else CHUNK
    rep, ok, seen = {}, True, {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            k = (r["split"], r["code"], round(float(r["p"]), 6))
            name = key.get(k)
            x = rep.setdefault(name or f"UNKNOWN {k}", dict(rows=0, seed_equal=0, sha_equal=0, shots_equal=0, chunks=[]))
            x["rows"] += 1
            if name is None:
                continue
            j = int(r["chunk"])
            x["chunks"].append(j)
            x["seed_equal"] += int(r["seed"]) == sets.seed(R, name, j)
            f = os.path.join(shots_dir, r["file"])
            x["sha_equal"] += os.path.exists(f) and sha256_file(f) == r["sha256"]
            x["shots_equal"] += int(r["shots"]) == want_n
            seen.setdefault(int(r["seed"]), []).append(name)
    for name, e in sorted(ent.items()):
        x = rep.setdefault(name, dict(rows=0, seed_equal=0, sha_equal=0, shots_equal=0, chunks=[]))
        x["complete"] = sorted(x["chunks"]) == list(range(e["chunks"]))
        x["expected_chunks"] = e["chunks"]
        x["chunks"] = len(x["chunks"])
        ok &= x["complete"] and x["seed_equal"] == x["sha_equal"] == x["shots_equal"] == x["rows"] == e["chunks"]
    ok &= len(rep) == len(ent) and all(len(v) == 1 for v in seen.values())
    res = dict(ok=bool(ok), dry=dry, round=rec["round"], randomness=rec["randomness"], chunk_shots=want_n,
               record_sha256=sha256_file(os.environ["SRT_BEACON"]), sets=rep, distinct_seeds=len(seen))
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("ok", "dry", "round", "distinct_seeds")}), flush=True)
    raise SystemExit(0 if ok else 1)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--check-beacon":
        rec, dry = record()
        print(json.dumps({"round": rec["round"], "randomness": rec["randomness"], "dry": dry}), flush=True)
        return
    if sys.argv[1] == "--list-sets":
        for n, e in sorted(bb_sets()[1].items()):
            print(n, e["code"], e["p"], e["split"], e["chunks"])
        return
    if sys.argv[1] == "--env-check":
        return env_check(sys.argv[2])
    if sys.argv[1] == "--check-manifest":
        args = sys.argv[2:]
        if "--out" not in args:
            raise SystemExit("--check-manifest M.csv... --out OUT.json")
        i = args.index("--out")
        return check_manifest(args[:i], args[i + 1])
    setup()
    if sys.argv[1].endswith(".py"):                        # a script (bb_replay.py, the pack scripts), no pools
        import runpy
        sys.argv = sys.argv[1:]
        return runpy.run_path(sys.argv[0], run_name="__main__")
    mod = importlib.import_module(sys.argv[1])
    sys.argv = [sys.argv[1]] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()
