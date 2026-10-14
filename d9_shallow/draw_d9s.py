"""D9-SHALLOW's draw (PLAN.md section 3): the two beacon sets on the verdict circuit of d = 9 (surface_v2/circuits/
d9_legacy.stim, X basis, 9 rounds, p = 0.6 %), with the public sampler and chunk format (convlines.testset: Stim's
detector sampler, 10,000 shots a chunk, deterministic npz bytes), once, on an x86-64 CPU with AVX2.

  test    100 chunks d9_p0.006_X_d9stest_NNN.npz    (seeds: the beacon record's seeds["test"])
  timing   20 chunks d9_p0.006_X_d9stiming_NNN.npz  (seeds: the beacon record's seeds["timing"])

Every seed is re-derived from the record (beacon_d9s.check) before a chunk is drawn. The manifests
OUT/manifests/d9s_test.json and d9s_timing.json hold every chunk's set, index, seed and sha256, the digest of the whole
set in chunk order (timing.deployed.shots_sha256, the digest every mask, read and timed process records), the circuit's
sha256, the versions and the CPU. As a check of the sampler on the host, SURF-v2's dry chunk 0 of d = 9 is drawn again
and compared with its manifest (convlines.testset.regen). No path in a manifest is absolute.
    python d9_shallow/draw_d9s.py --beacon d9_shallow/testsets/beacon_d9s.json --out SHOTS"""
import argparse
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "surface_v2"), ROOT, HERE]

import numpy as np  # noqa: E402
import stim  # noqa: E402

import beacon_d9s  # noqa: E402
from convlines import circuits, testset  # noqa: E402
from timing import deployed  # noqa: E402

D = 9
SPLIT = {"test": "d9stest", "timing": "d9stiming"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def host():
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
    flags = set(next((ln for ln in open("/proc/cpuinfo") if ln.startswith("flags")), "").split())
    return dict(cpu=cpu, machine=platform.machine(), avx2="avx2" in flags, avx512f="avx512f" in flags)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beacon", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = beacon_d9s.load(a.beacon)
    h = host()
    assert h["machine"] == "x86_64" and h["avx2"], f"the draw runs on x86-64 with AVX2, not {h}"
    assert stim.__version__ == "1.16.0" and np.__version__ == "2.2.6", (stim.__version__, np.__version__)
    c = circuits.circuit(D)
    os.makedirs(os.path.join(a.out, "manifests"), exist_ok=True)
    for s in ("test", "timing"):
        d = os.path.join(a.out, SPLIT[s])
        os.makedirs(d, exist_ok=False)
        files = {}
        for j, sd in enumerate(b["seeds"][s]):
            f = testset.name(SPLIT[s], D, j)
            testset.write_npz(os.path.join(d, f), testset.sample(c, sd))
            files[f] = dict(set=s, j=j, seed=sd, sha256=sha256_file(os.path.join(d, f)))
        sh = [testset.load(os.path.join(d, testset.name(SPLIT[s], D, j)), D) for j in range(len(files))]
        det, obs = np.concatenate([x[0] for x in sh]), np.concatenate([x[1] for x in sh])
        man = dict(what=f"D9-SHALLOW {s} shots (d9_shallow/draw_d9s.py)", set=s, split=SPLIT[s], tag=beacon_d9s.TAGS[s],
                   d=D, p=circuits.P, basis="X", rounds=D, chunks=len(files), chunk_shots=testset.CHUNK, shots=int(len(obs)),
                   circuit="surface_v2/circuits/d9_legacy.stim", circuit_sha256=circuits.CIRCUIT_SHA256[D],
                   beacon=dict(round=b["round"], randomness=b["randomness"], freeze_commit=b["freeze_commit"]),
                   shots_sha256=deployed.shots_sha256(det, obs), observable_ones=int(obs.sum()),
                   list_sha256=hashlib.sha256("".join(f"{k} {v['sha256']}\n" for k, v in sorted(files.items())).encode()).hexdigest(),
                   stim=stim.__version__, numpy=np.__version__, python=sys.version.split()[0], host=h, files=files)
        with open(os.path.join(a.out, "manifests", f"d9s_{s}.json"), "x") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
        print(f"{s}: {len(files)} chunks, {len(obs)} shots, shots_sha256 {man['shots_sha256'][:16]}", flush=True)
    dry = os.path.join(a.out, "sv2dry_check")
    ok = bool(testset.regen("sv2dry", D, [0], dry))
    print(json.dumps(dict(sv2dry_chunk0_equals_manifest=ok, **h)), flush=True)
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
