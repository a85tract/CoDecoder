"""ARB at d = 13: the dry run's chunks against the same chunks of the full read, array by array and bit for bit
(obs, packed detectors, chunk, c, sdec, z, bdec, and each chunk's seed and digest). Writes FULL/drycompare_<p>.json.

  python arb2_drycompare.py --dry OUT/dry/read --full OUT/read --p 0.006"""
import argparse
import glob
import json
import os

import numpy as np

KEYS = ("obs", "det", "chunk", "c", "sdec", "z", "bdec")


def rows_of(base, p):
    """{chunk: {key: array}} and {chunk: (seed, digest)} of a read's parts."""
    d = os.path.join(base, f"p{p:.4f}")
    out, meta = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "part_*.npz"))):
        with np.load(f) as z:
            m = json.loads(str(z["meta"]))
            arr = {k: z[k] for k in KEYS}
        s, e = m["chunks"]
        for i, j in enumerate(range(s, e)):
            sl = slice(i * 10000, (i + 1) * 10000)
            out[j] = {k: v[sl] for k, v in arr.items()}
            meta[j] = (m["seeds"][i], m["digests"][i])
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--p", type=float, required=True)
    a = ap.parse_args()
    dr, dm = rows_of(a.dry, a.p)
    fr, fm = {}, {}
    full_d = os.path.join(a.full, f"p{a.p:.4f}")
    first = sorted(glob.glob(os.path.join(full_d, "part_*.npz")))[:1]     # the dry chunks lie in the first part
    if first:
        with np.load(first[0]) as z:
            m = json.loads(str(z["meta"]))
            arr = {k: z[k] for k in KEYS}
        s, e = m["chunks"]
        for i, j in enumerate(range(s, e)):
            if j in dr:
                sl = slice(i * 10000, (i + 1) * 10000)
                fr[j] = {k: v[sl] for k, v in arr.items()}
                fm[j] = (m["seeds"][i], m["digests"][i])
    res = {"p": a.p, "chunks": sorted(dr), "per_chunk": {}}
    ok = bool(dr) and set(dr) <= set(fr)
    for j in sorted(dr):
        if j not in fr:
            res["per_chunk"][str(j)] = "missing in the full read"
            continue
        eq = {k: bool(np.array_equal(dr[j][k].view(np.uint8), fr[j][k].view(np.uint8))) for k in KEYS}
        eq["seed"], eq["digest"] = dm[j][0] == fm[j][0], dm[j][1] == fm[j][1]
        res["per_chunk"][str(j)] = eq
        ok = ok and all(eq.values())
    res["equal"] = bool(ok)
    with open(os.path.join(a.full, f"drycompare_{a.p}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
