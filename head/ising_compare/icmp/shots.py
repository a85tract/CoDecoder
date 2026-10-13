"""Stored val / test shots (SETTINGS §4). Sampled once on the canonical host (the mac-mini, ICMP_CANONICAL=1) in chunks of
CHUNK shots with Stim's detector sampler, stored as packed bits (np.packbits, big-endian, the nemu.nets.unpack layout)
with their sha256 in data/shot_manifest.csv; every host reads the stored files (Stim output differs between ARM and x86).
Seeds: val 20 000 000 + 10 000 d + chunk, test 30 000 000 + 10 000 d + chunk (the 25-parameter circuits), val_legacy
40 000 000 + ..., test_legacy 50 000 000 + ... (the legacy single-p circuits); training streams use seeds >= 10^8
(SETTINGS §4), so all ranges are disjoint. Test chunks are read only by the guarded evaluation (allow_test=True)."""
import csv
import hashlib
import os

import numpy as np

from icmp import circuits

CHUNK = 10_000
SEED_BASE = {"val": 20_000_000, "test": 30_000_000, "val_legacy": 40_000_000, "test_legacy": 50_000_000}
NOISE = {"val": "25p", "test": "25p", "val_legacy": "legacy", "test_legacy": "legacy"}   # circuit of each split
VAL, TEST = "val_legacy", "test_legacy"          # the verdict setting since ASSUMPTIONS I12
SHOT_DIR = os.environ.get("ICMP_SHOTS", os.path.expanduser("~/work/ising/shots"))
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "data", "shot_manifest.csv")


def seed_of(split, d, j):
    assert 0 <= j < 10_000
    return SEED_BASE[split] + 10_000 * d + j


def name(split, d, j, p=circuits.P_VERDICT):
    return f"d{d}_p{p}_X_{split}_{j:03d}.npz"


def circuit(split, d):
    return circuits.circuit(d, noise=NOISE[split])


def draw(split, d, j, c=None):
    """(det (CHUNK, m) uint8, obs (CHUNK,) uint8) for chunk j; deterministic given the host's Stim build."""
    c = c or circuit(split, d)
    det, obs = c.compile_detector_sampler(seed=seed_of(split, d, j)).sample(CHUNK, separate_observables=True)
    return det.astype(np.uint8), obs[:, 0].astype(np.uint8)


def manifest():
    if not os.path.exists(MANIFEST):
        return {}
    return {r["file"]: r for r in csv.DictReader(open(MANIFEST))}


def load(split, d, chunks, allow_test=False, packed=False):
    """Concatenated stored chunks, sha256-checked against the manifest (fails closed on a missing or changed file)."""
    assert not split.startswith("test") or allow_test, "test chunks are read only by the guarded evaluation"
    man = manifest()
    dets, obs = [], []
    for j in chunks:
        f = name(split, d, j)
        raw = open(os.path.join(SHOT_DIR, f), "rb").read()
        assert f in man and hashlib.sha256(raw).hexdigest() == man[f]["sha256"], f"{f}: missing or hash mismatch"
        z = np.load(os.path.join(SHOT_DIR, f))
        dets.append(z["det"])
        obs.append(z["obs"])
    det = np.concatenate(dets)
    if not packed:
        m = circuit(split, d).num_detectors
        det = np.unpackbits(det, axis=1)[:, :m]
    return det, np.concatenate(obs)
