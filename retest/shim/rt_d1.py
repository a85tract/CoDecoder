"""Re-test shim for D1, the follow-up of Cascade on BB72 (RETEST.md, test D1). It runs one script of D1's deployed tree
(cascade_d1/code.20261011_152939: pc_eval.py, d1_compiled.py, d1_confirm.py, any of its scripts) as __main__, and, iff the
environment variable RT_SEEDS is set, replaces exactly one name before the script starts:

    bb.cascade_data.eval_seed(kind, p, chunk, family)

for kind 14 (D1's reserved kind) and family depspam_fm at p = 0.002 and 0.003: chunk c0 + j (c0 = 6,433,102 =
int(F[:7], 16) mod 9,000,000, F = 310ce8e0...) gets seeds.chunk_seeds("d1-0.002" or "d1-0.003")[j], the seed of beacon
index j (retest/seeds.py: each recomputed from the beacon record RT_BEACON, an error on any mismatch). Any other chunk
of kind 14 / depspam_fm (another p, a chunk outside the block) is an error. Every other kind, family, p and chunk falls
through to the original function, unchanged. Without RT_SEEDS the shim is an identity launcher.

    python rt_d1.py SCRIPT.py ARGS...        (SCRIPT.py as D1's job files call it: pc_eval.py, d1_confirm.py, ...)

Why the patch is at the top level and the script runs by exec: pc_eval.py samples in a spawn pool
(bb.cascade_eval: mp.get_context("spawn").Pool(8), workers run bb.cascade_data._eval_chunk, which calls eval_seed
from its module globals). A spawn worker starts a fresh interpreter and re-runs the parent's __main__ file as
__mp_main__ before it unpickles any task. Here __main__ is this file, so every worker executes the top level below and
patches its own copy of bb.cascade_data before it samples; the target script is only exec'd under __main__ (runpy.run_path
would swap sys.modules["__main__"] for the target's module and the workers would re-run the target without this patch).
d1_confirm.py imports eval_seed inside check_read, after this patch, so its check "every chunk's seed is eval_seed(kind,
p, j, family)" holds for the new seeds. Environment: RT_SEEDS, RT_BEACON (see retest/seeds.py). With RT_D1_TRACE=DIR (an existing directory) every call for D1's kind
appends (pid, parent pid, p, chunk, seed) to DIR/trace_<pid>.tsv in the calling process: the read job's seed check reads it to
show that the main process and the sampling workers each drew every chunk's intended seed."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RT_DIR = os.path.dirname(HERE)                         # retest/ of the freeze: seeds.py, tags.json
F = "310ce8e0f8bfb9d3b97b27186547a836d29fc9ca"         # D1's freeze commit
KIND, FAMILY = 14, "depspam_fm"
C0 = int(F[:7], 16) % 9_000_000                        # d1_common.c0_of(F) = 6,433,102
SETS = {2000: "d1-0.002", 3000: "d1-0.003"}            # round(p * 1e6) -> the seed set of retest/tags.json
CHUNKS = {2000: 1100, 3000: 1700}                      # the lock's blocks (d1_locked_depspam_fm.json "chunks")


def _load_seeds():
    spec = importlib.util.spec_from_file_location("rt_seeds", os.path.join(RT_DIR, "seeds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install():
    """Patch bb.cascade_data.eval_seed (once per process). -> the facts of the patch, None when RT_SEEDS is unset."""
    if not os.environ.get("RT_SEEDS"):
        return None
    from bb import cascade_data as cd
    if getattr(cd.eval_seed, "_rt_d1", None):
        return cd.eval_seed._rt_d1
    if not os.environ.get("RT_BEACON"):
        raise SystemExit("rt_d1: RT_SEEDS is set but RT_BEACON is not")
    assert C0 == 6_433_102, C0
    sd = _load_seeds()
    table = {k: sd.chunk_seeds(name) for k, name in SETS.items()}          # raises when a seed differs from the beacon's
    for k, n in CHUNKS.items():
        if len(table[k]) != n:
            raise SystemExit(f"rt_d1: {SETS[k]} has {len(table[k])} seeds, the lock's block is {n}")
    every = [x for v in table.values() for x in v]
    if len(set(every)) != len(every) or min(every) < 2 ** 62 or max(every) >= 2 ** 63:
        raise SystemExit("rt_d1: the seeds are not distinct values in [2^62, 2^63)")
    orig = cd.eval_seed
    trace = os.environ.get("RT_D1_TRACE")                                  # a directory (must exist) or unset

    def eval_seed(kind, p, chunk, family="uniform"):
        if kind == KIND and family == FAMILY:                               # D1's kind: only the re-test's chunks
            key = int(round(p * 1e6))
            j = chunk - C0
            if key not in table or p != key / 1e6 or not 0 <= j < len(table[key]):
                raise ValueError(f"rt_d1: kind {kind} {family} at p = {p!r}, chunk {chunk!r} is not one of the "
                                 f"re-test's chunks (p 0.002: {C0} .. {C0 + CHUNKS[2000] - 1}, p 0.003: "
                                 f"{C0} .. {C0 + CHUNKS[3000] - 1})")
            seed = table[key][j]
            if trace:                                                      # one file per process: who drew which seed
                with open(os.path.join(trace, f"trace_{os.getpid()}.tsv"), "a") as fh:
                    fh.write(f"{os.getpid()}\t{os.getppid()}\t{p!r}\t{chunk}\t{seed}\n")
            return seed
        return orig(kind, p, chunk, family)

    eval_seed.__module__, eval_seed.__qualname__, eval_seed.__doc__ = orig.__module__, orig.__qualname__, orig.__doc__
    eval_seed._rt_d1 = {"kind": KIND, "family": FAMILY, "c0": C0, "F": F, "sets": dict(SETS),
                        "first_seed": {SETS[k]: v[0] for k, v in table.items()},
                        "seeds_file": os.environ["RT_SEEDS"], "beacon": os.environ["RT_BEACON"]}
    eval_seed._orig = orig
    cd.eval_seed = eval_seed
    return eval_seed._rt_d1


FACTS = install()                                                           # at import: main process and spawn workers


def main(argv):
    if not argv or not argv[0].endswith(".py") or not os.path.isfile(argv[0]):
        raise SystemExit("usage: rt_d1.py SCRIPT.py ARGS...")
    script = argv[0]
    if FACTS:
        print(f"rt_d1: eval_seed patched for kind {KIND} {FAMILY}: chunks {C0}.. of the sets {sorted(SETS.values())}; "
              f"first seeds {FACTS['first_seed']}; seeds {FACTS['seeds_file']}; beacon {FACTS['beacon']}",
              file=sys.stderr, flush=True)
    else:
        print("rt_d1: RT_SEEDS unset, the patch is off (identity launcher)", file=sys.stderr, flush=True)
    sys.argv = list(argv)
    sys.path[0] = os.path.dirname(os.path.abspath(script))                  # as `python SCRIPT.py` puts it
    with open(script, "rb") as fh:
        code = compile(fh.read(), script, "exec")
    exec(code, {"__name__": "__main__", "__file__": script, "__doc__": None})


if __name__ == "__main__":
    main(sys.argv[1:])
