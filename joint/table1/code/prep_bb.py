"""BB72, local and light: checks the rows that alpine_extract_bb.py selected from the re-test's eager reads and writes,
per p, a reduced read DIR/<tag>.npz (only those rows) beside the read's own DIR/<tag>.json, so that ARB's
arb_scores.py runs unchanged on it (its selection, routed and differing, then keeps every row of the reduced file).

  python3 prep_bb.py data/bb data/d1 data/pod/read"""
import json
import os
import shutil
import sys

import numpy as np

TAU1 = {"p0.00200": 7.3331888389587405, "p0.00300": 7.60439167022705}
K = 12


def main():
    src, d1, out = sys.argv[1:4]
    for tag, tau1 in TAU1.items():
        d = os.path.join(src, tag)
        s = {f[4:-4]: np.load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith("sel_")}
        info = json.load(open(os.path.join(d, "info.json")))
        n = len(s["rows"])
        assert n == info["rows"] and all(len(v) == n for v in s.values())
        assert np.all(np.diff(s["rows"]) > 0)
        assert np.all(s["conf1"].astype(np.float64) < tau1) and np.all((s["pred0"] != s["pred1"]).any(1))
        b0 = np.unpackbits(s["pred0"], axis=1, count=K)
        b1 = np.unpackbits(s["pred1"], axis=1, count=K)
        bo = np.unpackbits(s["obs"], axis=1, count=K)
        assert np.array_equal(s["fail0"], (b0 != bo).any(1)) and np.array_equal(s["fail1"], (b1 != bo).any(1))
        assert (s["fail0"] | s["fail1"]).all()
        assert np.array_equal(s["cls1"].astype(np.int64), (b1.astype(np.int64) << np.arange(K)).sum(1))
        z = s["z0"]
        sign_bad = int(((z > 0).astype(np.uint8) != b0).any(1).sum())
        conf_bad = int((np.abs(z).min(1) != s["conf0"].astype(np.float16)).sum())
        chk = {"rows": n, "z0_sign_ne_pred0_rows": sign_bad, "minabs_z0_ne_f16_conf0_rows": conf_bad,
               "chunk_range": [int(s["chunk"].min()), int(s["chunk"].max())],
               "row_in_chunk_range": [int(s["row"].min()), int(s["row"].max())]}
        print(tag, json.dumps(chk), flush=True)
        o = os.path.join(out, tag)
        os.makedirs(o, exist_ok=True)
        np.savez(os.path.join(o, f"{tag}.npz"), **s)
        shutil.copy(os.path.join(d1, tag.replace("p0.00200", "0.002").replace("p0.00300", "0.003"), "read",
                                 f"{tag}.json"), o)
        json.dump({**info, "prep_checks": chk}, open(os.path.join(o, "sel_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
