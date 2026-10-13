"""Seed checks of the BB shim with the patch on (RETEST.md section 6): every re-test chunk's seed is the seeds file's
(each recomputed from the beacon), a forked worker (bb.opp and fullread fork their pools; no BB worker draws shots,
every read loads stored chunks checked against the manifest) sees the same function, every other split, code and p
is the frozen function's, and a chunk outside a re-test set or a test split without one is refused. Run from the
tree's ising_compare with PYTHONPATH as the reads and RT_SEEDS, RT_BEACON set: python bb_seedcheck.py OUT_JSON"""
import json
import os
import sys
from multiprocessing import get_context

RT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(RT, "shim"), RT]
from bb import shots  # noqa: E402

FROZEN = shots.seed_of
import rt_bb  # noqa: E402
import seeds  # noqa: E402

rt_bb.setup()


def child(args):
    return shots.seed_of(*args)


if __name__ == "__main__":
    rep, ok = {}, True
    for (split, code, p), name in rt_bb.SETS.items():
        if split == "test4" and not hasattr(shots, "TEST4_TAG"):
            continue
        want = seeds.chunk_seeds(name)
        got = [shots.seed_of(split, code, p, j) for j in range(len(want))]
        with get_context("fork").Pool(4) as pool:
            forked = pool.map(child, [(split, code, p, j) for j in range(len(want))])
        refused = 0
        for j in (-1, len(want)):
            try:
                shots.seed_of(split, code, p, j)
            except ValueError:
                refused += 1
        r = {"chunks": len(want), "main_equal": got == want, "forked_equal": forked == want, "outside_refused": refused == 2}
        rep[name] = r
        ok &= all(v for k, v in r.items() if k != "chunks")
    same = total = 0
    for split in shots.SPLIT:
        for code in shots.CODE_I:
            for p in shots.P_I:
                for j in (0, 1, 7, 19, 29, 999, 99_999):
                    total += 1
                    same += shots.seed_of(split, code, p, j) == FROZEN(split, code, p, j)
    rep["other_splits"] = {"checked": total, "equal": same}
    refused = 0
    for code, p in (("bb72", 0.001), ("bb144", 0.004), ("bb144", 0.003)):
        try:
            shots.seed_of("test", code, p, 0)
        except ValueError:
            refused += 1
    rep["test_without_set_refused"] = refused == 3
    ok &= same == total and refused == 3
    rep["ok"] = ok
    json.dump(rep, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(rep), flush=True)
    sys.exit(0 if ok else 1)
