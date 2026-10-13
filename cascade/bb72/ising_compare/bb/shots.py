"""Seeds and stored shots of the BB sub-route (bb/DESIGN.md section 6).

Seed namespace (fresh: every registered numeric namespace of ising_compare is below 2^33 (interp/REPORT.md's audit),
qldpc_neural's below 3.1 x 10^8, interp uses 2^40 + 10,000 d + j, T2-T4 use seeds >= 2^62):
  seed = 2^41 + 10^8 split + 10^7 code + 10^6 p_idx + j      (j < 10^6; one chunk = 10,000 shots)
  split: dev 0, cal 1, smoke 2, monitor 3 (training-time evaluation); training chunk c of run run_id at training
  noise p uses 2^42 + 10^9 (10 run_id + p_idx) + c (c < 10^9, run_id < 5,000; bb.train), so a run resumes at any
  chunk. No test split exists here: a test set's seeds come from a public
  beacon after a freeze (bb/PLAN.md, stage 3).
  code: bb72 0, bb144 1; p_idx as qldpc_neural's qnd.data.P_I.
Stim's samples depend on the host, so dev / cal / monitor chunks are sampled once (on Alpine, `python -m bb.shots`),
stored packed with their sha256 in data/bb/shot_manifest.csv and only ever read back through load(), which checks the
hash (fail closed)."""
import argparse
import csv
import hashlib
import json
import os
import platform

import numpy as np

from qnd import circuits

BASE = 2 ** 41
SPLIT = {"dev": 0, "cal": 1, "smoke": 2, "monitor": 3}
CODE_I = {"bb72": 0, "bb144": 1}
P_I = {0.003: 0, 0.002: 1, 0.001: 2, 0.004: 3, 0.005: 4, 0.0025: 5, 0.0035: 6, 0.0015: 7, 0.0045: 8, 0.006: 9}
CHUNK = 10_000
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(HERE, "data", "bb", "shot_manifest.csv")
SHOTS = os.path.expanduser(os.environ.get("BB_SHOTS", "~/work/shots"))
BEACON = os.path.join(HERE, "data", "bb", "stage3", "beacon.json")
BEACON_GROSS = os.path.join(HERE, "data", "bb", "stage3_gross", "beacon.json")    # STAGE3_GROSS_PLAN.md section 2
BEACON_GROSS4 = os.path.join(HERE, "data", "bb", "stage4_gross", "beacon.json")   # STAGE4_GROSS_PLAN.md section 2
TEST_TAG = {("bb72", 0.003): b"ising_compare bb test BB72-3\n", ("bb72", 0.002): b"ising_compare bb test BB72-2\n",
            ("bb144", 0.004): b"ising_compare bb test G-4\n"}
TEST4_TAG = {("bb144", 0.004): b"ising_compare bb test4 G-4R G-4F\n"}           # gross stage 4's fresh split "test4"
FIELDS = ["file", "sha256", "shots", "seed", "code", "p", "split", "chunk", "host", "cpu", "stim"]


def seed_of(split, code, p, j):
    if not 0 <= j < 10 ** 6:
        raise ValueError(j)
    if split == "test":
        return test_seed(code, p, j)
    if split == "test4":
        return test_seed(code, p, j, json.load(open(BEACON_GROSS4))["randomness"], TEST4_TAG[(code, round(p, 6))])
    return BASE + 10 ** 8 * SPLIT[split] + 10 ** 7 * CODE_I[code] + 10 ** 6 * P_I[round(p, 6)] + j


def test_seed(code, p, j, randomness=None, tag=None):
    """Stage 3's test chunk j (bb/STAGE3_PLAN.md section 2, PLAN 3.5): 2^62 + (the first 8 bytes, big-endian, of
    SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62; R = the drand randomness in data/bb/stage3/beacon.json
    (gross: data/bb/stage3_gross/beacon.json, STAGE3_GROSS_PLAN.md), TAG one per test split; gross stage 4's split
    "test4" passes its own beacon's randomness and TEST4_TAG (STAGE4_GROSS_PLAN.md section 2)."""
    if not 0 <= j < 2 ** 16:
        raise ValueError(j)
    r = bytes.fromhex(randomness if randomness is not None else
                      json.load(open(BEACON_GROSS if code == "bb144" else BEACON))["randomness"])
    if len(r) != 32:
        raise ValueError("the beacon randomness must be 32 bytes")
    h = hashlib.sha256(r + (tag if tag is not None else TEST_TAG[(code, round(p, 6))]) + j.to_bytes(2, "big")).digest()
    return 2 ** 62 + int.from_bytes(h[:8], "big") % 2 ** 62


def chunk_seed(run_id, p, c):
    """Seed of training chunk c of run run_id at training noise p (2^42 and up: above every namespace in the header)."""
    if not (0 <= c < 10 ** 9 and 0 <= run_id < 5000):
        raise ValueError((run_id, c))
    return 2 ** 42 + 10 ** 9 * (10 * run_id + P_I[round(p, 6)]) + c


def name(code, p, split, j):
    return f"{code}_p{p}_{split}{j:04d}.npz"


def sample(code, p, split, j, n=CHUNK):
    """Fresh (det (n, D) uint8, obs (n, k) uint8) of chunk j on this host (not the stored copy)."""
    import stim  # noqa: F401
    c = circuits.load(code, "X", p)
    det, obs = c.compile_detector_sampler(seed=seed_of(split, code, p, j)).sample(n, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def _manifest():
    if not os.path.exists(MANIFEST):
        return {}
    return {r["file"]: r for r in csv.DictReader(open(MANIFEST))}


def load(code, p, split, j):
    """The stored chunk j, sha256-checked against the manifest: (det uint8 (n, D), obs uint8 (n, k))."""
    f = os.path.join(SHOTS, name(code, p, split, j))
    r = _manifest().get(os.path.basename(f))
    if r is None:
        raise FileNotFoundError(f"{os.path.basename(f)} is not in {MANIFEST}")
    blob = open(f, "rb").read()
    if hashlib.sha256(blob).hexdigest() != r["sha256"]:
        raise ValueError(f"sha256 mismatch for {f}")
    z = np.load(f)
    n = int(z["n"])
    return (np.unpackbits(z["det"], axis=1, count=int(z["D"]))[:n],
            np.unpackbits(z["obs"], axis=1, count=int(z["k"]))[:n])


def load_many(code, p, split, chunks):
    parts = [load(code, p, split, j) for j in chunks]
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


def parse_range(s):
    out = []
    for part in str(s).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    """Sample and store chunks; prints manifest rows (append them to data/bb/shot_manifest.csv in git)."""
    import stim
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--split", required=True, choices=list(SPLIT) + ["test", "test4"])
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--rows", required=True, help="CSV file to append the manifest rows to")
    a = ap.parse_args()
    os.makedirs(SHOTS, exist_ok=True)
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    new = not os.path.exists(a.rows)
    with open(a.rows, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for j in parse_range(a.chunks):
            f = os.path.join(SHOTS, name(a.code, a.p, a.split, j))
            if os.path.exists(f):
                raise FileExistsError(f)
            det, obs = sample(a.code, a.p, a.split, j)
            np.savez(f, det=np.packbits(det, axis=1), obs=np.packbits(obs, axis=1), n=len(det), D=det.shape[1],
                     k=obs.shape[1])
            row = {"file": os.path.basename(f), "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(),
                   "shots": len(det), "seed": seed_of(a.split, a.code, a.p, j), "code": a.code, "p": a.p,
                   "split": a.split, "chunk": j, "host": platform.node(), "cpu": cpu, "stim": stim.__version__}
            w.writerow(row)
            fh.flush()
            print(row["file"], row["sha256"][:12], f"fired/shot {det.mean() * det.shape[1]:.1f}",
                  f"any-obs {obs.any(1).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
