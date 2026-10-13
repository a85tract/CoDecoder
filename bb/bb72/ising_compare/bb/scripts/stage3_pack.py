"""STAGE3_PLAN.md 5 (published): every decoder's per-shot test predictions and failures, packed, one compressed npz per
test split, plus each test chunk's array-level sha256 (np.savez stamps file times, so a regenerated chunk is checked by
its arrays). Usage (after the analysis): BB_SHOTS=... BB_PREDS=... python stage3_pack.py <repo root> <out dir>"""
import hashlib
import json
import os
import sys

import numpy as np

root, out = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(root, "ising_compare"), os.path.join(root, "qldpc_neural")]
from bb import opp, route, shots  # noqa: E402

cells = json.load(open(os.path.join(root, "ising_compare", "data", "bb", "stage3", "cells.json")))
index = {}
for s, sp in cells["splits"].items():
    code, p, chunks = sp["code"], sp["p"], shots.parse_range(sp["chunks"])
    arrays, hashes = {}, []
    for j in chunks:
        det, obs = shots.load(code, p, "test", j)
        hashes.append({"chunk": j, "seed": shots.seed_of("test", code, p, j),
                       "det_packed_sha256": hashlib.sha256(np.packbits(det, axis=1).tobytes()).hexdigest(),
                       "obs_packed_sha256": hashlib.sha256(np.packbits(obs, axis=1).tobytes()).hexdigest()})
    for d in ("relay", "bposd", "gpu_bposd", "relay_full"):
        pred, _ = opp.load_preds(code, p, "test", chunks, d)
        arrays[f"{d}_pred"] = np.packbits(pred, axis=1)
    for t in sorted({c["tag"] for c in cells["cells"].values() if c["split"] == s}):
        bits, margin, sig, _ = route._load_net(code, p, "test", chunks, t)
        arrays[f"net-{t}_pred"], arrays[f"net-{t}_margin"] = np.packbits(bits, axis=1), margin
    obs = shots.load_many(code, p, "test", chunks)[1]
    arrays["obs"] = np.packbits(obs, axis=1)
    path = os.path.join(out, f"test_outputs_{s}.npz")
    np.savez_compressed(path, **arrays)
    index[s] = {"file": os.path.basename(path), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                "shots": int(len(obs)), "arrays": sorted(arrays), "chunks": hashes,
                "unpack": "np.unpackbits(x, axis=1, count=12) for predictions and obs (k = 12 logical observables)"}
json.dump(index, open(os.path.join(out, "test_outputs_index.json"), "w"), indent=1)
print(json.dumps({s: (v["file"], v["sha256"][:16]) for s, v in index.items()}))
