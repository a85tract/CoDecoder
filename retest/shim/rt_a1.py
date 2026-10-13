"""Re-test shim for the output-head comparison (A1; paper Tables head and headcmp). It runs one command of A1's pinned
tree (rev_a1/code_at/2e69738e...; the working directory is its ising_compare, PYTHONPATH its three code directories) and
changes only these things before the command runs, and only when RT_SEEDS is set (without it the shim is an identity
launcher: the replay of the original analysis runs that way):
  a1.sets.seed_of   surface kind "test" chunk j -> seeds.chunk_seeds("head-surface")[j] (originally 72,090,000 + j);
                    BB72 kind "a1test" native chunk j -> seeds.chunk_seeds("head-bb72")[j] (originally 2,199,523,255,552
                    + j). Native chunk numbers stay 0-49, so a1.fallback's Relay-BP decoder seeds (the shot's index in its
                    set, 10,000 c + r) stay as frozen. Every other kind (cal, dry, a1cal, a1dry, ...) falls through to the
                    original function, and a chunk number outside the new sets is an error, never an old seed.
  A1_SETS, A1_SETS_ROWS   $RT_OUT/head/sets, and the manifest rows retest/head/cal_chunks.csv (A1's 40 calibration chunks,
                    taken from A1's rows.csv; the pinned tree has no manifest of its own) followed by $RT_OUT/head/sets/rows.csv
                    (the new test chunks, written by the draw). The original calibration chunks are symlinked into the sets
                    directory by the draw job; the original cal reads are symlinked into the new reads directories
                    (jobs/head_draw.sh), sha256-checked by every job. No row of an original test chunk is ever listed.
Guards (RT_SEEDS set only; each ends the run with a message): every --out / --reads / --rows / --many / --checks path lies
under $RT_OUT/head; every --many job and every draw / fallback command is of the test kind (no calibration shot is drawn
or read); a1.analyze runs the frozen config files (sha256) unless RT_DRY; RT_DRY (a switch that shrinks a1.preflight's test
sets to chunk 0 for the dry run; the job scripts shrink the draw and fallback chunk lists) and a dry beacon (the record's
"dry" flag) go together or not at all.
Spawned worker processes (a1.fallback's pools use the "spawn" start method) re-run this file as __mp_main__: the patch is
at top level, the command only under __main__, and the target's functions are pickled by their own module names, so a
worker imports the target module itself, after this file's top level has installed the patch.
    python rt_a1.py MODULE ARGS...      e.g.  rt_a1.py a1.sets draw --code surface --kind test --chunks 0-99 ...
                                              rt_a1.py a1.read | a1.preflight | a1.fallback | a1.analyze | a1.check ..."""
import hashlib
import importlib
import importlib.util
import json
import os
import sys

RT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # <freeze>/retest
TEST_KINDS = {"surface": "test", "bb72": "a1test"}
NATIVE_CHUNKS = {("surface", "test"): ("head-surface", 100), ("bb72", "a1test"): ("head-bb72", 50)}
FROZEN_CONFIG = {"analysis_surface.json": "328e54c0b7a09154219b33650b0caca3b5eec472926731dd6273ae228a85b848",
                 "analysis_bb72.json": "652dfe390b6e3ce53b8aec4704b5a226f2b178be7c6e9a4e2cb5a49765c386c5"}
OUT_FLAGS = ("--out", "--reads", "--rows", "--many", "--checks")
ACTIVE = bool(os.environ.get("RT_SEEDS"))
DRY = bool(os.environ.get("RT_DRY"))


def _die(msg):
    raise SystemExit(f"rt_a1: {msg}")


def _install():
    for k in ("RT_BEACON", "RT_OUT"):
        if not os.environ.get(k):
            _die(f"RT_SEEDS is set, {k} is not")
    if bool(json.load(open(os.environ["RT_BEACON"])).get("dry")) != DRY:
        _die("RT_DRY and a dry beacon go together (RT_DRY only with the dry seeds, the dry seeds only with RT_DRY)")
    spec = importlib.util.spec_from_file_location("rt_seeds", os.path.join(RT_DIR, "seeds.py"))
    seeds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seeds)
    sets_dir = os.path.join(os.path.abspath(os.environ["RT_OUT"]), "head", "sets")
    rows = os.path.join(RT_DIR, "head", "cal_chunks.csv") + ":" + os.path.join(sets_dir, "rows.csv")
    for k, v in (("A1_SETS", sets_dir), ("A1_SETS_ROWS", rows)):
        if os.environ.get(k, v) != v:
            _die(f"{k} is {os.environ[k]!r}, the re-test's is {v!r}")
        os.environ[k] = v                                                  # before a1.sets reads it at import
    from a1 import seeds as a1seeds
    from a1 import sets
    sets.SETS = sets_dir
    new = {}
    for key, (name, n) in NATIVE_CHUNKS.items():
        s = [int(x) for x in seeds.chunk_seeds(name)]                      # each recomputed from the beacon record
        if len(s) != n or len(set(s)) != n or min(s) < 2 ** 62:
            _die(f"the seeds of {name}: {len(s)} of {n}, distinct {len(set(s))}, smallest {min(s)}")
        new[key] = s
    assert (len(new[("surface", "test")]), len(new[("bb72", "a1test")])) == (a1seeds.SURFACE_CHUNKS["test"],
                                                                              a1seeds.BB_CHUNKS["a1test"])
    original = sets.seed_of

    def seed_of(code, kind, j):
        s = new.get((code, kind))
        if s is None:
            return original(code, kind, j)                                 # every other kind: A1's own seeds
        if not 0 <= j < len(s):
            raise ValueError(f"chunk {j} of {code} {kind}: the re-test's set has chunks 0-{len(s) - 1}")
        return s[j]

    sets.seed_of = seed_of
    sets.rt_original_seed_of = original                                    # for the replay checks
    sets.rt_new_seeds = new


if ACTIVE:
    _install()


def _guard(name, args):
    root = os.path.join(os.path.abspath(os.environ["RT_OUT"]), "head")
    for i, x in enumerate(args):
        if x in OUT_FLAGS and i + 1 < len(args):
            p = os.path.abspath(args[i + 1])
            if not p.startswith(root + os.sep):
                _die(f"{x} {p} is not under {root}")
        if x == "--kind" and i + 1 < len(args) and args[i + 1] not in TEST_KINDS.values():
            _die(f"--kind {args[i + 1]}: the re-test draws and decodes test sets only")
        if x == "--sets" and i + 1 < len(args) and args[i + 1] != "main":
            _die("--sets dry: not in the re-test")
        if x == "--many" and i + 1 < len(args):
            for j in json.load(open(args[i + 1])):
                if j["kind"] not in TEST_KINDS.values():
                    _die(f"a read of kind {j['kind']}: the re-test reads test sets only (the cal reads are the original ones)")
                for k in ("out", "export"):
                    if j.get(k) and not os.path.abspath(j[k]).startswith(root + os.sep):
                        _die(f"a read's {k} {j[k]} is not under {root}")
    if name == "a1.analyze" and not DRY:
        cfg = args[args.index("--config") + 1]
        if hashlib.sha256(open(cfg, "rb").read()).hexdigest() != FROZEN_CONFIG.get(os.path.basename(cfg)):
            _die(f"{cfg} is not one of the frozen analysis configs")


def main():
    name, args = sys.argv[1], sys.argv[2:]
    if ACTIVE:
        _guard(name, args)
    mod = importlib.import_module(name)
    if ACTIVE and DRY and name == "a1.preflight":
        for code, kind in TEST_KINDS.items():                              # the dry run's one test chunk per code
            mod.SETS[code]["test"] = (kind, "0-0")
    sys.argv = [name] + args
    mod.main()


if __name__ == "__main__":
    main()
