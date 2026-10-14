"""DRY-RUN ONLY: fabricates routing masks so that `timing.cpu_term term` runs unchanged on DRY shots.

On a real session each mask comes from a GPU pass (`python -m timing.deployed --d D --net NAME --val VIEW --out MASKS`:
the deployed network's logits c, routed = |c| < tau). The CPU timing host has no GPU, no network weights and no build,
so for a dry rehearsal this script writes MASKS/val_<NAME>.npz in timing.deployed's format with a made-up routing: the
shots of largest detector weight (number of fired detectors; ties: lower shot index first) are routed, as many as the
network's registered validation routing fraction says (a proxy for "hard" shots). Nothing here is a network's output.
Every file holds dry_synthetic = 1 and a note, and its logits field reads "dry_synthetic", which `term` copies into its
record as mask_kind. Never use these masks, or a record made from them, in a result.

    python speed_rt/cpu_ws/mkmask_dry.py --view VIEW --out MASKS [--nets n1,n2]      (default: the 13 networks below)
    python -m timing.cpu_term term --d D --val VIEW --mask MASKS/val_<NAME>.npz --rep R --core C --out DIR

The shots are those `term` reads (the 20 chunks of d in VIEW, testset.load, concatenated in order) and their digest is
deployed.shots_sha256's, so deployed.load accepts the mask; both functions are the repo's own (surface_v2 is put on
sys.path; Stim is used only inside testset.load, as in `term`). c is float32 with |c| = tau (rank + 1/2) / n for the
n routed shots (rank 0..n-1 by weight) and above tau for the others, so routed == (|c| < tau) holds with the stored tau
(float64, as timing.deployed stores it). Three SURF-v2 networks have tau = 0: they route nothing, and tau stays finite
because `term` rejects a NaN tau. An existing file is never overwritten."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "surface_v2"))

from convlines import testset  # noqa: E402
from timing import deployed  # noqa: E402

SHOTS = 200_000          # validation shots the routed counts below refer to
# name: (d, registered tau, routed shots of the 200,000 validation shots). Deployed d = 7-15: the studies' published
# validation masks (surface_v2/routing/manifest.json "val": 4.05, 1.03, 1.02, 2.00, 5.91 %); d17_b5_r787_q08: R17's
# validation term records (7.88 %); d17_b5_r787_q10: val_d17 (study 4's tau, 9.98 %); the SURF-v2 networks have no
# validation mask: q x 200,000 with their registered q (weights/configs_sv2.json, speed_rt/freeze_surf.json).
NETS = {
    "d7_b1w64_r740": (7, 1.3934324979782104, 8095),
    "d7_b1w64_sv2": (7, 0.90324866771698, 4000),
    "d9_b3_r764": (9, 0.6820775270462036, 2051),
    "d9_b3_sv2s1": (9, 0.0, 0),
    "d11_b5_r777": (11, 1.0014245510101318, 2040),
    "d11_b5_sv2": (11, 0.0, 0),
    "d13_b5_r775": (13, 1.8486117124557495, 4000),
    "d13_b5_sv2": (13, 0.0, 0),
    "d15_b5_r782": (15, 3.9547388553619385, 11822),
    "d15_b5_sv2": (15, 1.9195706844329834, 2000),
    "d17_b5_r787_q08": (17, 4.539173126220703, 15759),
    "d17_b5_r787_q10": (17, 5.121974468231201, 19957),
    "d17_b5_sv2": (17, 2.2122440338134766, 2000),
}
NOTE = ("DRY-RUN fabrication (speed_rt/cpu_ws/mkmask_dry.py): the routed shots are those of largest detector weight, "
        "not a network's output; never a timing input")


def shots(view, d):
    """(detector weight per shot, shots digest) of d's 20 chunks in VIEW, loaded and hashed as cpu_term and deployed do."""
    sh = [testset.load(os.path.join(view, testset.name("val", d, j)), d) for j in range(20)]
    det, obs = np.concatenate([s[0] for s in sh]), np.concatenate([s[1] for s in sh])
    return det.sum(1, dtype=np.int32), deployed.shots_sha256(det, obs)


def fabricate(w, tau, n):
    """(c float32, routed bool): the n shots of largest weight w (ties: lower index) routed, routed == (|c| < tau)."""
    order = np.argsort(-w, kind="stable")                # weight descending, ties by shot index
    rank = np.empty(len(w), np.int64)
    rank[order] = np.arange(len(w))
    c = ((rank + 0.5) / max(n, 1) * (tau if tau > 0 else 1.0)).astype(np.float32)
    routed = rank < n
    assert (routed == (np.abs(c.astype(np.float64)) < tau)).all(), "c and routed disagree"
    return c, routed


def main():
    ap = argparse.ArgumentParser(description="DRY-RUN ONLY: fabricated routing masks for timing.cpu_term term")
    ap.add_argument("--view", required=True, help="the dry timing set's VIEW directory (python -m speed_rt.draw view)")
    ap.add_argument("--out", required=True, help="MASKS directory: val_<NAME>.npz per network, never overwritten")
    ap.add_argument("--nets", default="", help="comma-separated network names (default: all 13)")
    a = ap.parse_args()
    names = [x for x in a.nets.split(",") if x] or list(NETS)
    assert not [x for x in names if x not in NETS], f"unknown network in {names}; known: {list(NETS)}"
    path = {x: os.path.join(a.out, f"val_{x}.npz") for x in names}
    for x in names:                                      # refuse before computing anything
        assert not os.path.exists(path[x]), f"{path[x]} exists (never overwritten)"
        for j in range(20):
            f = os.path.join(a.view, testset.name("val", NETS[x][0], j))
            assert os.path.exists(f), f"{f} is missing"
    first = os.path.realpath(os.path.join(a.view, testset.name("val", NETS[names[0]][0], 0)))
    if "dry" not in first.lower():
        print(f"WARNING: {first} does not look like a DRY shot file; these masks are for dry runs only", file=sys.stderr)
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    for x in names:
        d, tau, count = NETS[x]
        if d not in cache:
            cache[d] = shots(a.view, d)
        w, digest = cache[d]
        n = round(count * len(w) / SHOTS)
        c, routed = fabricate(w, tau, n)
        assert int(routed.sum()) == n
        with open(path[x], "xb") as fh:
            np.savez(fh, c=c, routed=routed.astype(np.uint8), d=d, network=x, tau=np.float64(tau),
                     logits="dry_synthetic", shots_sha256=digest, dry_synthetic=np.uint8(1), note=NOTE)
        print(f"{os.path.basename(path[x])}: d={d} routed {n} of {len(w)} = {100 * n / len(w):.4f} % "
              f"(tau {tau!r}, shots_sha256 {digest[:12]})", flush=True)


if __name__ == "__main__":
    main()
