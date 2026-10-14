"""DRY-RUN ONLY: fabricates Ising's residual detectors so that `timing.cpu_term share` runs unchanged on DRY shots.

On a real session the residuals come from the untimed GPU pass on the pod that counted Ising's plan (the hand-off's
ising/<role>/res_d<d>_<M>_val000.npz: pre_L, res, obs of the timing set's chunk 0 after the counted plan). The CPU host
has no GPU and no plan, so for a dry rehearsal each fired detector of the dry timing set's chunk 0 is kept with
probability KEEP (seeded per d and model), a stand-in for what the pre-decoder leaves; pre_L is zero. Every file holds
dry_synthetic = 1. Never use these files, or a record made from them, in a result.

    PYTHONPATH=CODE/surface_v2 python speed_rt/cpu_ws/mkres_dry.py --view VIEW --out DIR [--keep 0.25]
        -> DIR/<role>/res_d<d>_<M>_val000.npz for d = 7-17 (role low: d <= 11, high: d >= 13), M = M1, M4"""
import argparse
import os

import numpy as np
from convlines import testset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, help="the dry timing view (speed_rt.draw view), chunk 0 per d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.25)
    a = ap.parse_args()
    for d in (7, 9, 11, 13, 15, 17):
        det, obs = testset.load(os.path.join(a.view, testset.name("val", d, 0)), d)
        o = os.path.join(a.out, "low" if d <= 11 else "high")
        os.makedirs(o, exist_ok=True)
        for k, m in enumerate(("M1", "M4")):
            rng = np.random.default_rng(1000 * d + k)
            res = (np.asarray(det, dtype=bool) & (rng.random(det.shape) < a.keep)).astype(np.uint8)
            np.savez_compressed(os.path.join(o, f"res_d{d}_{m}_val000.npz"), pre_L=np.zeros(len(res), np.uint8),
                                res=res, obs=obs, dry_synthetic=np.uint8(1))
            print(f"d{d} {m}: {res.shape}, {res.sum(1).mean():.3f} residual detectors per shot "
                  f"(input {np.asarray(det).sum(1).mean():.3f})", flush=True)


if __name__ == "__main__":
    main()
