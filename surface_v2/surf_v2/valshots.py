"""The studies' own validation shots (val_legacy chunks 0-19 at d = 7-13, V15, V17b) in this release's chunk format, for
SURF-v2's timing (the GPU term, the handoff, the CPU term and calcost's ref): each set is accepted only when its digest
(timing.deployed.shots_sha256 over chunks 0-19) equals the studies' in routing/manifest.json.

At d = 7-13 the studies sampled the shots on an ARM64 machine, so they are copied from the studies' stored files
(--src: icmp.shots' files d<d>_p0.006_X_val_legacy_<j>.npz, packed det and obs); at d = 15 and 17 x86-64 regenerates
them from their seeds (--src regen).

    python -m surf_v2.valshots --d 11 --src /scratch/alpine/$USER/ising/shots --out VAL
    python -m surf_v2.valshots --d 17 --src regen --out VAL"""
import argparse
import os

import numpy as np

from convlines import circuits, decoder, testset
from timing import deployed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--src", required=True, help="the studies' shot directory, or regen (d = 15, 17)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = circuits.circuit(a.d).num_detectors
    if a.src == "regen":
        assert a.d in (15, 17), "only V15 and V17b regenerate on x86-64"
        testset.val(a.d, range(20), a.out)
    else:
        for j in range(20):
            z = np.load(os.path.join(a.src, f"d{a.d}_p{circuits.P}_X_val_legacy_{j:03d}.npz"))
            assert z["det"].shape == (testset.CHUNK, (m + 7) // 8) and z["obs"].shape == (testset.CHUNK,)
            testset.write_npz(os.path.join(a.out, testset.name("val", a.d, j)),
                              dict(det=z["det"], obs=z["obs"].astype(np.uint8),
                                   seed=np.int64(testset.VAL_BASE[a.d] + 10_000 * a.d + j)))
    sh = [testset.load(os.path.join(a.out, testset.name("val", a.d, j)), a.d) for j in range(20)]
    det, obs = np.concatenate([s[0] for s in sh]), np.concatenate([s[1] for s in sh])
    got, want = deployed.shots_sha256(det, obs), decoder.routing_manifest()["val"][str(a.d)]["shots_sha256"]
    print(f"d{a.d}: validation chunks 0-19 in {a.out}: digest {got} {'== the studies' if got == want else '!= ' + want}")
    raise SystemExit(0 if got == want else 1)


if __name__ == "__main__":
    main()
