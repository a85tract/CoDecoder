"""D1-P (PHASE_D1_PLAN.md 2.4): the design-scenario powers of D1's four comparisons, before the freeze.

  python d1_power.py --lock LOCK --out power.json [--reps 20000 --seed 20261006]

The decision of one test is bb.route.power's: the exact two-sided McNemar p below alpha and Newcombe's interval at
alpha (paired.py's method 10) entirely below zero. For each alternative of the lock (design and observed), at each
claim point with the lock's block:
  per_test   bb.route.power itself at each Holm level (0.0125, 0.05 / 3, 0.025, 0.05)
  joint      the four-test procedure: the cells drawn at both points (multinomial, one generator with the fixed
             seed, 0.2 % first), the four comparisons' p-values under a model, pc_common.holm per repetition, and a
             comparison counted only when its Holm level exists and its interval at that level lies below zero.
             identical: each compiled comparison has the eager comparison's cells at its point (C14' decides alike on
             both paths on all but a few shots per 100,000; the intersection-union step then changes nothing).
             independent: the compiled comparisons' cells drawn afresh (a bracket, not the rule's model).
  interval   how often McNemar rejects at alpha while the interval at alpha does not lie below zero (per test).
The rule (lock power_rule): if a compiled comparison's joint power under the design alternative and the identical
model is below 0.8, that block grows to the smallest multiple of 100 chunks restoring 0.8 (reported, with the joint
power at it); otherwise the sizes stay."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase_c"))
sys.path.insert(0, HERE)
import d1_common as d1  # noqa: E402
import pc_common as pc  # noqa: E402


def draws(rng, cnt, n, reps):
    both, oa, ob, N = cnt
    return rng.multinomial(n, np.array([both, oa, ob, N - both - oa - ob]) / N, size=reps).astype(np.float64)


def decide(X, levels):
    """Per repetition: the exact McNemar p and, per level, whether Newcombe's interval at it lies below zero."""
    from scipy.stats import binom, norm
    from bb import route
    a, b, c, d = (X[:, i] for i in range(4))
    m = b + c
    pm = np.where(m > 0, np.minimum(1.0, 2 * binom.cdf(np.minimum(b, c), m, 0.5)), 1.0)
    below = {al: route._newcombe_hi(a, b, c, d, float(norm.ppf(1 - al / 2))) < 0 for al in levels}
    nc = route._newcombe()
    for i in range(3):                                     # the vectorised interval equals paired.py's
        for al in levels:
            z = float(norm.ppf(1 - al / 2))
            assert (nc(int(a[i]), int(b[i]), int(c[i]), int(d[i]), z)[2] < 0) == bool(below[al][i])
    return pm, below


def joint(L, chunks, scen, model, reps, seed):
    """The four-test procedure's power per comparison (L["comparisons"] order) and the probability that every
    comparison is rejected with its interval below zero."""
    levels = L["holm_levels"]
    rng = np.random.default_rng(seed)
    ps = sorted({c["p"] for c in L["configs"]})
    dec = {}
    for p in ps:
        dec[(p, "eager")] = decide(draws(rng, L["alternatives"][scen][str(p)], chunks[p] * pc.CHUNK, reps), levels)
    for p in ps:
        dec[(p, "compiled")] = dec[(p, "eager")] if model == "identical" else \
            decide(draws(rng, L["alternatives"][scen][str(p)], chunks[p] * pc.CHUNK, reps), levels)
    comps = L["comparisons"]
    hit = np.zeros((reps, len(comps)), bool)
    for r in range(reps):
        pv = [float(dec[(c["p"], c["path"])][0][r]) for c in comps]
        th = pc.holm(pv)
        for j, c in enumerate(comps):
            al = th[j]
            hit[r, j] = al is not None and bool(dec[(c["p"], c["path"])][1][min(levels, key=lambda x: abs(x - al))][r])
    return {c["name"]: float(hit[:, j].mean()) for j, c in enumerate(comps)} | {"all": float(hit.all(1).mean())}


def main():
    from bb import route
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", default=d1.LOCK)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20261006)
    a = ap.parse_args()
    L = d1.lock(a.lock)
    levels = L["holm_levels"]
    chunks = {c["p"]: d1.n_chunks(L, c["p"]) for c in L["configs"]}
    out = {"what": "D1-P (PHASE_D1_PLAN.md 2.4)", "lock_sha256": pc.sha256_file(a.lock), "seed": a.seed,
           "reps": a.reps, "levels": levels, "chunks": {str(p): n for p, n in chunks.items()}, "per_test": {},
           "interval_fails": {}, "joint": {}}
    for scen in ("design", "observed"):
        out["per_test"][scen], out["interval_fails"][scen] = {}, {}
        for p, n in chunks.items():
            cnt = L["alternatives"][scen][str(p)]
            out["per_test"][scen][str(p)] = {str(al): route.power(cnt, n * pc.CHUNK, alpha=al, reps=a.reps,
                                                                  seed=a.seed) for al in levels}
            pm, below = decide(draws(np.random.default_rng(a.seed), cnt, n * pc.CHUNK, a.reps), levels)
            out["interval_fails"][scen][str(p)] = {str(al): int(((pm < al) & ~below[al]).sum()) for al in levels}
        out["joint"][scen] = {m: joint(L, chunks, scen, m, a.reps, a.seed) for m in ("identical", "independent")}
    design = out["joint"]["design"]["identical"]
    low = {c["p"]: design[c["name"]] for c in L["comparisons"] if c["path"] == "compiled" and design[c["name"]] < 0.8}
    out["rule"] = L["power_rule"]
    out["sizes_kept"] = not low
    if low:
        new = dict(chunks)
        for p in low:
            for ch in range(chunks[p] + 100, 10 * chunks[p] + 1, 100):
                new[p] = ch
                j = joint(L, new, "design", "identical", a.reps, a.seed)
                if all(j[c["name"]] >= 0.8 for c in L["comparisons"] if c["path"] == "compiled" and c["p"] == p):
                    break
        out["proposed_chunks"] = {str(p): n for p, n in new.items()}
        out["joint_at_proposed"] = joint(L, new, "design", "identical", a.reps, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
