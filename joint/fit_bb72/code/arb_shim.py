"""ARB's seed shim: retest/shim/rt_d1.py with ARB's seed sets. It runs one script (arb_read.py, arb_seedcheck.py) as
__main__ and, before the script starts, replaces exactly one name in this process and in every spawned sampling worker:

    bb.cascade_data.eval_seed(kind, p, chunk, family)

for kind 14 (D1's kind) and family depspam_fm at p = 0.002 and 0.003: chunk c0 + j (c0 = 6,433,102, D1's block start,
kept so that the read's layout is D1's) gets seeds/arb_seeds.json's seed j of set "arb-0.002" or "arb-0.003"
(arb_seeds.chunk_seeds: every seed recomputed from the re-test's beacon and the set's TAG, an error on any mismatch).
Any other chunk of kind 14 / depspam_fm is an error; every other kind, family, p and chunk falls through to the original
function. Unlike rt_d1.py there is no identity mode: without ARB_SEEDS the shim refuses to start, so that no read can draw
D1's own shots by accident.

    python arb_shim.py SCRIPT.py ARGS...

Why the patch is at the top level and the script runs by exec (rt_d1.py's docstring): pc_eval samples in a spawn pool;
a spawn worker re-runs the parent's __main__ file (this file) as __mp_main__ before it unpickles any task, so every
worker patches its own bb.cascade_data before it samples. Environment: ARB_SEEDS (the seeds file), ARB_RELEASE (the
release clone: retest/seeds.py and retest/run/beacon.json), ARB_TRACE=DIR (optional, an existing directory): every call
for kind 14 appends (pid, parent pid, p, chunk, seed) to DIR/trace_<pid>.tsv."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
F = "310ce8e0f8bfb9d3b97b27186547a836d29fc9ca"         # D1's freeze commit (its block start c0)
KIND, FAMILY = 14, "depspam_fm"
C0 = int(F[:7], 16) % 9_000_000                        # d1_common.c0_of(F) = 6,433,102
SETS = {2000: "arb-0.002", 3000: "arb-0.003"}          # round(p * 1e6) -> ARB's seed set
CHUNKS = {2000: 1100, 3000: 1700}                      # D1's blocks (d1_locked_depspam_fm.json "chunks")


def _load_seeds():
    spec = importlib.util.spec_from_file_location("arb_seeds", os.path.join(HERE, "arb_seeds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install():
    """Patch bb.cascade_data.eval_seed (once per process) -> the facts of the patch."""
    path, release = os.environ.get("ARB_SEEDS"), os.environ.get("ARB_RELEASE")
    if not path or not release:
        raise SystemExit("arb_shim: ARB_SEEDS and ARB_RELEASE must be set (no identity mode)")
    from bb import cascade_data as cd
    if getattr(cd.eval_seed, "_arb", None):
        return cd.eval_seed._arb
    assert C0 == 6_433_102, C0
    sd = _load_seeds()
    table = {k: sd.chunk_seeds(name, path, release) for k, name in SETS.items()}   # raises on any mismatch
    for k, n in CHUNKS.items():
        if len(table[k]) != n:
            raise SystemExit(f"arb_shim: {SETS[k]} has {len(table[k])} seeds, D1's block is {n}")
    every = [x for v in table.values() for x in v]
    if len(set(every)) != len(every) or min(every) < 2 ** 62 or max(every) >= 2 ** 63:
        raise SystemExit("arb_shim: the seeds are not distinct values in [2^62, 2^63)")
    orig = cd.eval_seed
    trace = os.environ.get("ARB_TRACE")

    def eval_seed(kind, p, chunk, family="uniform"):
        if kind == KIND and family == FAMILY:                              # D1's kind: only ARB's chunks
            key = int(round(p * 1e6))
            j = chunk - C0
            if key not in table or p != key / 1e6 or not 0 <= j < len(table[key]):
                raise ValueError(f"arb_shim: kind {kind} {family} at p = {p!r}, chunk {chunk!r} is not one of ARB's "
                                 f"chunks (p 0.002: {C0} .. {C0 + CHUNKS[2000] - 1}, p 0.003: {C0} .. "
                                 f"{C0 + CHUNKS[3000] - 1})")
            seed = table[key][j]
            if trace:
                with open(os.path.join(trace, f"trace_{os.getpid()}.tsv"), "a") as fh:
                    fh.write(f"{os.getpid()}\t{os.getppid()}\t{p!r}\t{chunk}\t{seed}\n")
            return seed
        return orig(kind, p, chunk, family)

    eval_seed.__module__, eval_seed.__qualname__, eval_seed.__doc__ = orig.__module__, orig.__qualname__, orig.__doc__
    eval_seed._arb = {"kind": KIND, "family": FAMILY, "c0": C0, "sets": dict(SETS),
                      "first_seed": {SETS[k]: v[0] for k, v in table.items()}, "seeds_file": path, "release": release}
    eval_seed._orig = orig
    cd.eval_seed = eval_seed
    return eval_seed._arb


FACTS = install()                                                           # at import: main process and spawn workers


def main(argv):
    if not argv or not argv[0].endswith(".py") or not os.path.isfile(argv[0]):
        raise SystemExit("usage: arb_shim.py SCRIPT.py ARGS...")
    script = argv[0]
    print(f"arb_shim: eval_seed patched for kind {KIND} {FAMILY}: chunks {C0}.. of the sets {sorted(SETS.values())}; "
          f"first seeds {FACTS['first_seed']}; seeds {FACTS['seeds_file']}", file=sys.stderr, flush=True)
    sys.argv = list(argv)
    sys.path[0] = os.path.dirname(os.path.abspath(script))                  # as `python SCRIPT.py` puts it
    with open(script, "rb") as fh:
        code = compile(fh.read(), script, "exec")
    exec(code, {"__name__": "__main__", "__file__": script, "__doc__": None})


if __name__ == "__main__":
    main(sys.argv[1:])
