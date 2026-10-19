"""BB72 on the pod: the frozen A3 (ARB's joint score) on the re-test's routed shots whose two decisions differ.
Answer argmax_y [ s(y) + w (y . z) ] over the 4,096 classes (class bit j = observable j), s = S55'-deep's exported class
scores (arb_scores.py on the reduced read), z = C14''s 12 logits as the read stores them (float16, the only copy; ARB
fitted w on float32 logits), w frozen per p; computed by arb_analysis.a3_answers (float64 on the GPU, first argmax =
lowest class index on ties).

Float16 bound: with h_j half the float16 spacing at |z_j|, any logits that round to the stored z move every A3 score
difference by at most w sum_j h_j. A row whose A3 gap (best minus second best, as computed) exceeds that bound has the
same answer under the unrounded logits; the other rows are counted as "float16_sensitive".

  python a3_pod.py READ_ROOT  ->  READ_ROOT/<tag>/<tag>_a3.npz and a3_summary.json"""
import json
import os
import sys

import numpy as np
import torch

import arb_analysis as aa

W = {"p0.00200": 6.309573444801933, "p0.00300": 3.758374042884441}
K, NY = aa.K, aa.NY


def main():
    root = sys.argv[1]
    dev = torch.device("cuda")
    pw = (1 << np.arange(K)).astype(np.int64)
    Y = torch.as_tensor(((np.arange(NY)[:, None] >> np.arange(K)) & 1).astype(np.float64).T.copy(), device=dev)
    summ = {}
    for tag, w in W.items():
        d = os.path.join(root, tag)
        r = np.load(os.path.join(d, f"{tag}.npz"))
        srec = json.load(open(os.path.join(d, f"{tag}_scores.json")))
        rows = np.load(os.path.join(d, f"{tag}_scores_rows.npy"))
        n = len(r["rows"])
        assert np.array_equal(rows, np.arange(n)), "the score export is not on every row of the reduced read"
        sc = np.load(os.path.join(d, f"{tag}_scores.npy"), mmap_mode="r")
        z = r["z0"].astype(np.float64)
        yC = np.unpackbits(r["pred0"], axis=1, count=K) @ pw
        yN = np.unpackbits(r["pred1"], axis=1, count=K) @ pw
        yO = np.unpackbits(r["obs"], axis=1, count=K) @ pw
        yA = aa.a3_answers(sc, z, [w], yC)[0].astype(np.int64)
        h = np.spacing(np.abs(r["z0"])).astype(np.float64) / 2        # half the float16 spacing at |z|
        bound = w * h.sum(1)
        gap = np.empty(n)
        top = np.empty(n, np.int64)
        for lo in range(0, n, 16384):
            hi = min(n, lo + 16384)
            T = torch.as_tensor(np.asarray(sc[lo:hi]), device=dev).double() + w * (
                torch.as_tensor(z[lo:hi], device=dev) @ Y)
            v, i = torch.topk(T, 2, dim=1)
            gap[lo:hi] = (v[:, 0] - v[:, 1]).cpu().numpy()
            top[lo:hi] = torch.argmax(T, 1).cpu().numpy()
        assert np.array_equal(top, yA)
        fC, fN, fA = yC != yO, yN != yO, yA != yO
        third = (yA != yC) & (yA != yN)
        s = {"w": w, "rows": int(n), "scores_check_pass": srec["pass"], "gpu": srec["gpu"],
             "C14_fail": int(fC.sum()), "S55_fail": int(fN.sum()), "A3_fail": int(fA.sum()),
             "repairs_vs_composition": int((fC & ~fA).sum()), "breaks_vs_composition": int((~fC & fA).sum()),
             "A3_takes_S55": int((yA == yN).sum()), "A3_keeps_C14": int((yA == yC).sum()), "A3_third": int(third.sum()),
             "third_repairs": int((third & fC & ~fA).sum()), "ties": int((gap == 0).sum()),
             "float16_sensitive": int((gap <= bound).sum()), "min_gap_over_bound": float((gap / bound).min())}
        np.savez(os.path.join(d, f"{tag}_a3.npz"), rows=r["rows"], yC=yC, yN=yN, yO=yO, yA=yA, gap=gap, bound=bound)
        summ[tag] = s
        print(tag, json.dumps(s), flush=True)
    json.dump(summ, open(os.path.join(root, "a3_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
