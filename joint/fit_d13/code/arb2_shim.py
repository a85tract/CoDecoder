"""ARB's seed shim at d = 13: retest/shim/rt_surface.py's patch with ARB's seed sets, launched as ARB's BB72 shim
(arb_shim.py) launches. It runs one script (arb2_read.py, arb2_seedcheck.py) as __main__ and, before the script starts,
replaces exactly one name in this process and in every spawned worker:

    surface.surface_data.eval_seed(d, kind, p, chunk)

for kind 11 (the comparison's confirmation kind) at d = 13 and p = 0.006 / 0.0035: chunk j gets seeds/arb2_seeds.json's
seed j of set "arb-d13-0.006" (400 chunks) or "arb-d13-0.0035" (3,000 chunks) (arb2_seeds.chunk_seeds: every seed
recomputed from the re-test's beacon and the set's TAG, an error on any mismatch). The pinned function runs first, so
its own range checks hold; a kind-11 request for another d or p, or a chunk beyond the set, raises; every other kind
falls through to the pinned function unchanged. There is no identity mode: without ARB2_SEEDS the shim refuses to start,
so that no read can draw the comparison's own kind-11 shots by accident.

    python arb2_shim.py SCRIPT.py ARGS...

Why the patch is at the top level and the script runs by exec (rt_surface.py's and rt_d1.py's docstrings): the reads
sample in a spawn pool (surface_data.stream_shots); a spawn worker re-runs the parent's __main__ file (this file) as
__mp_main__ before it unpickles any task, so every worker patches its own surface_data before it samples. Environment:
ARB2_SEEDS (the seeds file), ARB2_RELEASE (the release clone: retest/seeds.py and retest/run/beacon.json), PYTHONPATH
with the release's cascade/d13 (the package `surface`), ARB2_TRACE=DIR (optional, an existing directory): every call for
kind 11 appends (pid, parent pid, p, chunk, seed) to DIR/trace_<pid>.tsv."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIND, D = 11, 13
SETS = {6000: "arb-d13-0.006", 3500: "arb-d13-0.0035"}       # round(1e6 p) -> ARB's seed set
CHUNKS = {6000: 400, 3500: 3000}


def _load_seeds():
    spec = importlib.util.spec_from_file_location("arb2_seeds", os.path.join(HERE, "arb2_seeds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install():
    """Patch surface.surface_data.eval_seed (once per process) -> the facts of the patch."""
    path, release = os.environ.get("ARB2_SEEDS"), os.environ.get("ARB2_RELEASE")
    if not path or not release:
        raise SystemExit("arb2_shim: ARB2_SEEDS and ARB2_RELEASE must be set (no identity mode)")
    from surface import surface_data as sd
    if getattr(sd.eval_seed, "_arb2", None):
        return sd.eval_seed._arb2
    seeds = _load_seeds()
    table = {k: seeds.chunk_seeds(name, path, release) for k, name in SETS.items()}   # raises on any mismatch
    for k, n in CHUNKS.items():
        if len(table[k]) != n:
            raise SystemExit(f"arb2_shim: {SETS[k]} has {len(table[k])} seeds, not {n}")
    every = [x for v in table.values() for x in v]
    if len(set(every)) != len(every) or min(every) < 2 ** 62 or max(every) >= 2 ** 63:
        raise SystemExit("arb2_shim: the seeds are not distinct values in [2^62, 2^63)")
    orig = sd.eval_seed
    trace = os.environ.get("ARB2_TRACE")

    def eval_seed(d, kind, p, chunk):
        base = orig(d, kind, p, chunk)                       # the pinned function's own checks and value
        if kind != KIND:
            return base
        key = int(round(p * 1e6))
        if d != D or key not in table or p != key / 1e6 or not 0 <= int(chunk) < len(table[key]):
            raise ValueError(f"arb2_shim: kind {KIND} has ARB's seeds for d = {D}, p = 0.006 (chunks 0-399) and "
                             f"p = 0.0035 (chunks 0-2999) only, not d = {d}, p = {p!r}, chunk = {chunk!r}")
        seed = table[key][int(chunk)]
        if trace:
            with open(os.path.join(trace, f"trace_{os.getpid()}.tsv"), "a") as fh:
                fh.write(f"{os.getpid()}\t{os.getppid()}\t{p!r}\t{int(chunk)}\t{seed}\n")
        return seed

    eval_seed.__module__, eval_seed.__qualname__, eval_seed.__doc__ = orig.__module__, orig.__qualname__, orig.__doc__
    eval_seed._arb2 = {"kind": KIND, "d": D, "sets": {SETS[k]: len(v) for k, v in table.items()},
                       "first_seed": {SETS[k]: v[0] for k, v in table.items()}, "seeds_file": path, "release": release}
    eval_seed._orig = orig
    sd.eval_seed = eval_seed
    return eval_seed._arb2


FACTS = install()                                                           # at import: main process and spawn workers


def main(argv):
    if not argv or not argv[0].endswith(".py") or not os.path.isfile(argv[0]):
        raise SystemExit("usage: arb2_shim.py SCRIPT.py ARGS...")
    script = argv[0]
    print(f"arb2_shim: eval_seed patched for kind {KIND} at d = {D}: sets {FACTS['sets']}; first seeds "
          f"{FACTS['first_seed']}; seeds {FACTS['seeds_file']}", file=sys.stderr, flush=True)
    sys.argv = list(argv)
    sys.path[0] = os.path.dirname(os.path.abspath(script))                  # as `python SCRIPT.py` puts it
    with open(script, "rb") as fh:
        code = compile(fh.read(), script, "exec")
    exec(code, {"__name__": "__main__", "__file__": script, "__doc__": None})


if __name__ == "__main__":
    main(sys.argv[1:])
