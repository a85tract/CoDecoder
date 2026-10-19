"""ARBCP counts (light numpy): per setting and path, failures of Cascade alone, the composition and the frozen joint
decision (A3), repairs and breaks, exact McNemar against the composition and against Cascade alone.

  python cp_count.py --dir OUT/bb72 --p 0.002,0.003 [--json OUT.json]

Inputs per p (cp_run.py): p<p>.json (kept-shot counts, checks) and p<p>_routed.npz (classes per routed shot).
Paths: eager (the read's C14' decisions and float16 logits; cross-check against the parallel run's counts),
single (compiled@1) and batched (compiled@4096; the composition's routed shots from the routed batches, Cascade alone
from the whole-read batches, as d1_confirm.py). Kept shots answer with S55'-deep everywhere, so the composition and the
joint decision differ from each other only on routed shots, and from Cascade alone also on kept ones.
McNemar: exact two-sided binomial test on the discordant shots (scipy's binomtest at 1/2, as bb.route.paired)."""
import argparse
import json
import math
import os

import numpy as np

PATHS = {"eager": ("eager", "eager", "eager"),                        # (composition's C14', Cascade alone, A3)
         "single": ("single", "single", "single"),
         "batched": ("batched", "batched_whole_saved", "batched")}
SAVED = {"single": ("single_saved", "single_saved", "single_saved"),
         "batched": ("batched_saved", "batched_whole_saved", "batched_saved")}


def mcnemar(b, c):
    """Exact two-sided McNemar p for b and c discordant shots (binomtest(b, b + c, 1/2), 'minlike' = doubled tail)."""
    n = b + c
    if n == 0 or b == c:
        return 1.0, 0.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    log10p = math.log10(2 * tail) - n * math.log10(2)
    return min(1.0, 2 * tail / 2 ** n), log10p


def one(dirn, p, names):
    tag = f"p{p:.5f}"
    rec = json.load(open(os.path.join(dirn, f"{tag}.json")))
    z = np.load(os.path.join(dirn, f"{tag}_routed.npz"))
    yO, yN = z["yO"], z["yN"]
    nN = yN != yO                                                    # S55'-deep fails (routed)
    kept = rec["kept"]
    out = {}
    for name, (ycomp, yalone, ya3) in names.items():
        alone_path = {"eager": "eager", "single": "single", "batched": "batched"}[name]
        al = rec["alone"][alone_path]
        fCo = z[f"y_{ycomp}"] != yO                                   # composition's C14' on routed shots
        fAl = z[f"y_{yalone}"] != yO                                  # Cascade alone on routed shots
        fJ = z[f"a3_{ya3}"] != yO                                     # joint decision on routed shots
        d = z[f"y_{ycomp}"] != yN                                     # disagreement rows
        third = d & (z[f"a3_{ya3}"] != yN) & (z[f"a3_{ya3}"] != z[f"y_{ycomp}"])
        comp = kept["S_fail"] + int(fCo.sum())
        joint = kept["S_fail"] + int(fJ.sum())
        rep, brk = int((fCo & ~fJ).sum()), int((~fCo & fJ).sum())
        both_change = int((fCo & fJ & (z[f"a3_{ya3}"] != z[f"y_{ycomp}"])).sum())
        # against Cascade alone: kept shots (S55'-deep vs Cascade alone) + routed shots (joint vs Cascade alone)
        jb = al["kept_C_only"] + int((fAl & ~fJ).sum())               # Cascade alone fails, joint right
        jc = al["kept_S_only"] + int((~fAl & fJ).sum())               # joint fails, Cascade alone right
        cb = al["kept_C_only"] + int((fAl & ~fCo).sum())
        cc = al["kept_S_only"] + int((~fAl & fCo).sum())
        pj, lj = mcnemar(rep, brk)
        pa, la = mcnemar(jb, jc)
        pc_, lc = mcnemar(cb, cc)
        g = z[f"gap_{ya3}"]
        out[name] = {"alone": al["fail"],
                     "alone_consistent": al["fail"] == al["kept_C_only"] + al["kept_both"] + int(fAl.sum()),
                     "composition": comp, "joint": joint, "disagree": int(d.sum()), "changes": int((z[f"a3_{ya3}"]
                     != z[f"y_{ycomp}"]).sum()), "repairs": rep, "breaks": brk, "both_fail_changes": both_change,
                     "third_answers": int(third.sum()), "third_repairs": int((third & fCo & ~fJ).sum()),
                     "third_breaks": int((third & ~fCo & fJ).sum()),
                     "mcnemar_vs_composition": {"b": rep, "c": brk, "p": pj, "log10p": lj},
                     "mcnemar_vs_alone": {"b": jb, "c": jc, "p": pa, "log10p": la},
                     "composition_vs_alone": {"b": cb, "c": cc, "p": pc_, "log10p": lc},
                     "rel_vs_composition": (joint - comp) / comp, "rel_vs_alone": (joint - al["fail"]) / al["fail"],
                     "near_ties_gap_lt_1e-6": int((g[d] < 1e-6).sum()) if d.any() else 0}
    return rec, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--p", default="0.002,0.003")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for p in (float(x) for x in a.p.split(",")):
        rec, out = one(a.dir, p, PATHS)
        _, saved = one(a.dir, p, SAVED)
        res[str(p)] = {"paths": out, "with_saved_decisions": saved, "check": rec.get("check"), "diag": rec.get("diag"),
                       "scores_check": rec.get("scores_check", {}).get("pass"), "routed": rec["routed"],
                       "shots": rec["shots"], "w": rec["w"], "tau1": rec["tau1"], "quick": rec.get("quick")}
        for name, r in out.items():
            print(f"p={p} {name:8s} alone {r['alone']:>7,} comp {r['composition']:>7,} joint {r['joint']:>7,} "
                  f"rep {r['repairs']:>6,} brk {r['breaks']:>6,} third {r['third_answers']:>4,} "
                  f"McN(comp) p={r['mcnemar_vs_composition']['p']:.3g} McN(alone) p={r['mcnemar_vs_alone']['p']:.3g}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
