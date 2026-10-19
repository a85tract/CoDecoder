"""ARBCP counts at d = 13 (light numpy): rule J of the d = 13 exploration (~/work/arb2, PLAN.md step 4) with its fitted w per
p, on Cascade's two compiled paths of the re-test's C5.

  python cp_count_d13.py --dir OUT/d13 --w 0.006:W,0.0035:W [--json OUT.json]

On the routed shots where the network's and Cascade's decisions differ: answer 1 iff c + w z > 0 (float64), on a tie
Cascade's answer; everything else as in the composition. single: c of S1, z and the decision of B1 (compiled@1);
batched: c of SB, z and the decision of BQ5 (compiled@16 over the routed queue), Cascade alone Bb (compiled@16 over
every shot). Inputs: cp_d13x.py's d13_<p>.npz / .json."""
import argparse
import json
import os

import numpy as np

from cp_count import mcnemar


def one(dirn, p, w):
    rec = json.load(open(os.path.join(dirn, f"d13_{p}.json")))
    z = np.load(os.path.join(dirn, f"d13_{p}.npz"))
    obs = z["obs"]
    out = {}
    for mode, (r, c, dS, zc, dC, dA) in {"single": ("r1", "c1", "d1", "zB1", "dB1", "dB1"),
                                         "batched": ("rB", "cB", "dB", "zQ", "dQ", "dBb")}.items():
        m = rec["modes"][mode]
        rt = z[r]
        cc, ds = z[c][rt].astype(np.float64), z[dS][rt]
        zz, dc, da = z[zc][rt].astype(np.float64), z[dC][rt], z[dA][rt]
        o = obs[rt]
        d = ds != dc
        v = cc + w * zz
        ans = dc.copy()
        ans[d] = np.where(v[d] > 0, 1, np.where(v[d] < 0, 0, dc[d])).astype(dc.dtype)
        fCo, fJ, fAl = dc != o, ans != o, da != o
        comp, joint = m["S_fail_kept"] + int(fCo.sum()), m["S_fail_kept"] + int(fJ.sum())
        rep, brk = int((fCo & ~fJ).sum()), int((~fCo & fJ).sum())
        jb, jc = m["kept_C_only"] + int((fAl & ~fJ).sum()), m["kept_S_only"] + int((~fAl & fJ).sum())
        cb, cc_ = m["kept_C_only"] + int((fAl & ~fCo).sum()), m["kept_S_only"] + int((~fAl & fCo).sum())
        pj, lj = mcnemar(rep, brk)
        pa, la = mcnemar(jb, jc)
        pcx, lc = mcnemar(cb, cc_)
        out[mode] = {"alone": m["alone_fail"], "alone_consistent": m["alone_fail"] == m["kept_C_only"]
                     + m["kept_both"] + int(fAl.sum()), "composition": comp, "joint": joint, "routed": int(rt.sum()),
                     "disagree": int(d.sum()), "switched_to_network": int((d & (ans == ds)).sum()), "repairs": rep,
                     "breaks": brk, "ties": int((d & (v == 0)).sum()),
                     "mcnemar_vs_composition": {"b": rep, "c": brk, "p": pj, "log10p": lj},
                     "mcnemar_vs_alone": {"b": jb, "c": jc, "p": pa, "log10p": la},
                     "composition_vs_alone": {"b": cb, "c": cc_, "p": pcx, "log10p": lc},
                     "rel_vs_composition": (joint - comp) / comp, "rel_vs_alone": (joint - m["alone_fail"])
                     / m["alone_fail"]}
    return rec, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--w", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for item in a.w.split(","):
        p, w = item.split(":")
        rec, out = one(a.dir, float(p), float(w))
        res[p] = {"w": float(w), "paths": out, "checks": rec["checks"], "tau1": rec["tau1"], "shots": rec["shots"]}
        for mode, r in out.items():
            print(f"d13 p={p} {mode:8s} alone {r['alone']:>6,} comp {r['composition']:>6,} joint {r['joint']:>6,} "
                  f"rep {r['repairs']:>5,} brk {r['breaks']:>5,} McN(comp) p={r['mcnemar_vs_composition']['p']:.3g} "
                  f"McN(alone) p={r['mcnemar_vs_alone']['p']:.3g}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
