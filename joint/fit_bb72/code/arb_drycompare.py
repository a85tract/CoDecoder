"""ARB: the dry read (the first chunks of the block, read alone) against the full read's same rows, bitwise: detector
rows, observables, both networks' outputs, the float32 logits and the exported class scores. A reproducibility check of
the read and of the export; prints one JSON line and exits 1 on any difference.
  python arb_drycompare.py --dry OUT/dry/read --full OUT/read --p 0.002"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--p", type=float, required=True)
    a = ap.parse_args()
    tag = f"p{a.p:.5f}"
    dz, fz = np.load(os.path.join(a.dry, f"{tag}.npz")), np.load(os.path.join(a.full, f"{tag}.npz"))
    n = len(dz["obs"])
    res = {"p": a.p, "rows": n, "equal": {}}
    for k in sorted(set(dz.files) - {"chunks", "models"}):
        res["equal"][k] = bool(np.array_equal(dz[k], fz[k][:n]))
    dzz = np.load(os.path.join(a.dry, f"{tag}_z32.npy"))
    fzz = np.load(os.path.join(a.full, f"{tag}_z32.npy"), mmap_mode="r")
    res["equal"]["z32"] = bool(np.array_equal(dzz.view(np.uint32), np.asarray(fzz[:n]).view(np.uint32)))
    dr, fr = np.load(os.path.join(a.dry, f"{tag}_scores_rows.npy")), np.load(os.path.join(a.full, f"{tag}_scores_rows.npy"))
    m = int((fr < n).sum())
    res["equal"]["scores_rows"] = bool(np.array_equal(dr, fr[:m]))
    if res["equal"]["scores_rows"]:
        ds = np.load(os.path.join(a.dry, f"{tag}_scores.npy"))
        fs = np.load(os.path.join(a.full, f"{tag}_scores.npy"), mmap_mode="r")
        res["equal"]["scores"] = bool(np.array_equal(ds.view(np.uint32), np.asarray(fs[:m]).view(np.uint32)))
    res["score_rows"] = m
    res["pass"] = all(res["equal"].values())
    json.dump(res, open(os.path.join(a.full, f"drycompare_{a.p}.json"), "w"), indent=1)
    print("drycompare:", json.dumps(res), flush=True)
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
