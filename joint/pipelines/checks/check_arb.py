"""Check (a): the joint decision's fixed fp32 arithmetic against ARB's float64 decision (rule A3 at ARB's fitted w) on ARB's
saved disagreement rows (~/work/arb/out/analysis/disagree_<p>.npz: rows, z, ans_A3; ~/work/arb/out/read/p<p>_scores.npy, row k =
shot rows[k], which ARB's analysis required to be the disagreement rows).

  cpu  python check_arb.py cpu --arb ARB_OUT --p 0.002 --w W --out F.json --subset S.npz [--n-subset 64]
       joint_ref.joint_np (numpy, bit for bit joint._k_jtop) on every row against ARB's ans_A3; every differing row is
       listed; S.npz keeps a subset for the GPU (every differing row, the rows whose top-2 gap of t is smallest, and
       some at random) with their scores, logits, ARB's answer and joint_np's.
  gpu  python check_arb.py gpu --subset S.npz --out F.json
       joint._k_jtop (the GPU kernel of the timed pipelines) on the subset's saved scores: its answers against joint_np's
       and ARB's.
The GPU and joint_np agree bit for bit wherever both run (here on the subset, and on the pod's own rows in
joint_scores.py / joint_count.py compare); the Linux workstation runs only numpy (the scores are 1.7 GB; the pod's link is slow)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release", "cascade", "bb72",
                                "cascade_compare", "phase_d"))
import joint_ref  # noqa: E402


def cpu(a):
    t0 = time.time()
    tag = f"p{a.p:.5f}"
    d = np.load(os.path.join(a.arb, "analysis", f"disagree_{tag}.npz"))
    rows = np.load(os.path.join(a.arb, "read", f"{tag}_scores_rows.npy"))
    sc = np.load(os.path.join(a.arb, "read", f"{tag}_scores.npy"), mmap_mode="r")
    if not np.array_equal(rows, d["rows"]):
        raise SystemExit(f"{tag}: the scores' rows are not the disagreement rows")
    z = d["z"].astype(np.float32)
    ans_arb = d["ans_A3"].astype(np.int64)
    ans, gap = joint_ref.joint_np(sc, z, a.w, gap=True)
    diff = np.nonzero(ans != ans_arb)[0]
    rng = np.random.default_rng(20261019)
    small = np.argsort(gap, kind="stable")[:a.n_subset]
    rand = rng.choice(len(z), size=min(a.n_subset // 2, len(z)), replace=False)
    pick = np.unique(np.concatenate([diff, small, rand]))
    np.savez(a.subset, p=a.p, w=a.w, idx=pick, row=rows[pick], sc=np.asarray(sc[pick], np.float32), z=z[pick],
             ans_arb=ans_arb[pick], ans_np=ans[pick], gap=gap[pick])
    rec = {"p": a.p, "w": a.w, "w32_hex": joint_ref.w32_of(a.w)[1], "rows": int(len(z)), "differ": int(len(diff)),
           "differ_rows": [{"k": int(k), "shot": int(rows[k]), "fp32": int(ans[k]), "arb_f64": int(ans_arb[k]),
                            "yC": int(d["yC"][k]), "yN": int(d["yN"][k]), "gap_fp32": float(gap[k])} for k in diff],
           "ties_fp32": int((gap == 0).sum()), "min_gap": float(gap.min()),
           "gap_quantiles": {q: float(np.quantile(gap, q)) for q in (0.0001, 0.001, 0.01, 0.5)},
           "third_answers": int(((ans != d["yC"]) & (ans != d["yN"])).sum()), "to_S": int((ans == d["yN"]).sum()),
           "to_C": int((ans == d["yC"]).sum()), "subset_rows": int(len(pick)), "secs": round(time.time() - t0, 1)}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "differ_rows"}), flush=True)


def gpu(a):
    import joint
    S = np.load(a.subset)
    w = float(S["w"])
    g = joint.joint_scores_gpu(S["sc"], S["z"], w)
    rec = {"p": float(S["p"]), "w": w, "rows": int(len(g)), "gpu_vs_np": int((g != S["ans_np"]).sum()),
           "gpu_vs_arb": int((g != S["ans_arb"]).sum()),
           "gpu_vs_arb_rows": [int(x) for x in S["row"][g != S["ans_arb"]]], "min_gap": float(S["gap"].min())}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cpu")
    s.add_argument("--arb", required=True)
    s.add_argument("--p", type=float, required=True)
    s.add_argument("--w", type=float, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--subset", required=True)
    s.add_argument("--n-subset", type=int, default=64)
    s = sub.add_parser("gpu")
    s.add_argument("--subset", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cpu": cpu, "gpu": gpu}[a.cmd](a)
