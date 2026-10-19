"""BB72 on the pod, after a3_pod.py: how far the float16 storage of C14''s logits can move A3's failure count.
Any float32 logits z' that round to the stored z satisfy |z'_j - z_j| <= h_j (half the float16 spacing at |z_j|), so
for the answer y* and any class y, A3's score difference T(y*) - T(y) moves by at most w sum_j [y_j != y*_j] h_j.
D(y) = T(y*) - T(y) - w sum_j [y_j != y*_j] h_j; when D(y) > 0 for every y != y*, the answer is the same under every
such z'. Otherwise the answer may become any y with D(y) <= 0 (and only such y). Bounds on the failure count:
  may_repair: y* fails and D(true class) <= 0;  may_break: y* is right and some y != y* has D(y) <= 0.
  python a3_bound.py READ_ROOT  ->  READ_ROOT/a3_bound.json"""
import json
import os
import sys

import numpy as np
import torch

K, NY = 12, 4096
W = {"p0.00200": 6.309573444801933, "p0.00300": 3.758374042884441}


def main():
    root = sys.argv[1]
    dev = torch.device("cuda")
    Yb = torch.as_tensor(((np.arange(NY)[:, None] >> np.arange(K)) & 1).astype(np.float64), device=dev)   # (NY, K)
    out = {}
    for tag, w in W.items():
        d = os.path.join(root, tag)
        r = np.load(os.path.join(d, f"{tag}.npz"))
        a = np.load(os.path.join(d, f"{tag}_a3.npz"))
        sc = np.load(os.path.join(d, f"{tag}_scores.npy"), mmap_mode="r")
        z = r["z0"].astype(np.float64)
        h = np.spacing(np.abs(r["z0"])).astype(np.float64) / 2
        yA, yO = a["yA"], a["yO"]
        n = len(yA)
        sens = np.zeros(n, bool)
        may_rep = np.zeros(n, bool)
        may_brk = np.zeros(n, bool)
        for lo in range(0, n, 4096):
            hi = min(n, lo + 4096)
            T = torch.as_tensor(np.asarray(sc[lo:hi]), device=dev).double() + w * (
                torch.as_tensor(z[lo:hi], device=dev) @ Yb.T)
            ys = torch.as_tensor(yA[lo:hi], device=dev)
            yb = Yb[ys]                                                   # (m, K) bits of y*
            diff = (Yb[None, :, :] != yb[:, None, :]).double()            # (m, NY, K)
            pert = w * (diff @ torch.as_tensor(h[lo:hi], device=dev)[:, :, None])[:, :, 0]
            D = T.gather(1, ys[:, None]) - T - pert
            D[torch.arange(hi - lo, device=dev), ys] = float("inf")
            open_ = D <= 0
            s = open_.any(1).cpu().numpy()
            yo = torch.as_tensor(yO[lo:hi], device=dev)
            fails = (ys != yo).cpu().numpy()
            rep = open_[torch.arange(hi - lo, device=dev), yo].cpu().numpy() & fails
            sens[lo:hi], may_rep[lo:hi], may_brk[lo:hi] = s, rep, s & ~fails
        fA = yA != yO
        out[tag] = {"w": w, "rows": int(n), "A3_fail": int(fA.sum()), "answer_may_change": int(sens.sum()),
                    "may_repair": int(may_rep.sum()), "may_break": int(may_brk.sum()),
                    "A3_fail_range": [int(fA.sum() - may_rep.sum()), int(fA.sum() + may_brk.sum())]}
        np.save(os.path.join(d, f"{tag}_a3_open.npy"), np.stack([sens, may_rep, may_brk], 1))
        print(tag, json.dumps(out[tag]), flush=True)
    json.dump(out, open(os.path.join(root, "a3_bound.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
