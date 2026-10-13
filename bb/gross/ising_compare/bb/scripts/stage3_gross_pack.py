"""STAGE3_GROSS_PLAN.md 5 (published): every decoder's per-shot test predictions, the network's test margins and the
observables, packed in one compressed npz, each test chunk's array-level sha256, and the cal reads behind sizes.json
(the network's stage-2 H100 cal read and sector Relay-BP's cal read). Usage (after the analysis):
BB_SHOTS=... BB_PREDS=... python stage3_gross_pack.py <repo root> <out dir>"""
import hashlib
import json
import os
import sys

import numpy as np

root, out = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(root, "ising_compare"), os.path.join(root, "qldpc_neural")]
from bb import opp, route, shots  # noqa: E402

c = json.load(open(os.path.join(root, "ising_compare", "data", "bb", "stage3_gross", "cell.json")))
code, p = c["code"], c["p"]
index = {}
for split, chunks, decs in (("test", shots.parse_range(c["chunks"]), ("relay", "bposd", "gpu_bposd")),
                            ("cal", shots.parse_range("0-9"), ("relay",))):
    arrays, hashes = {}, []
    for j in chunks:
        det, obs = shots.load(code, p, split, j)
        hashes.append({"chunk": j, "seed": shots.seed_of(split, code, p, j),
                       "det_packed_sha256": hashlib.sha256(np.packbits(det, axis=1).tobytes()).hexdigest(),
                       "obs_packed_sha256": hashlib.sha256(np.packbits(obs, axis=1).tobytes()).hexdigest()})
    for d in decs:
        arrays[f"{d}_pred"] = np.packbits(opp.load_preds(code, p, split, chunks, d)[0], axis=1)
    if split == "test":
        arrays["relay_full_pred_chunk0"] = np.packbits(opp.load_preds(code, p, split, [0], "relay_full")[0], axis=1)
    bits, margin, sig, _ = route._load_net(code, p, split, chunks, c["tag"])
    arrays[f"net-{c['tag']}_pred"], arrays[f"net-{c['tag']}_margin"] = np.packbits(bits, axis=1), margin
    arrays["obs"] = np.packbits(shots.load_many(code, p, split, chunks)[1], axis=1)
    path = os.path.join(out, f"{split}_outputs_G-4.npz")
    np.savez_compressed(path, **arrays)
    index[split] = {"file": os.path.basename(path), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                    "shots": int(len(arrays["obs"])), "arrays": sorted(arrays), "chunks": hashes, "engine": sig,
                    "unpack": "np.unpackbits(x, axis=1, count=12) for predictions and obs (k = 12 logical observables)"}
json.dump(index, open(os.path.join(out, "outputs_index_G-4.json"), "w"), indent=1)
print(json.dumps({s: (v["file"], v["sha256"][:16]) for s, v in index.items()}))
