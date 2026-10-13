"""python replay_read_off.py NEW_DIR ORIG_DIR OUT.json: NEW_DIR's p0.00{2,3}00.{npz,json} (one chunk, c0) against ORIG_DIR's first
10,000 rows. Compared: every per-shot array of the npz, chunk_meta[0] (chunk, seed, det_sha256), the read's kind, family, circuit
sha256, and each model's signature (every field but the timings)."""
import json
import sys

import numpy as np

new, old, out = sys.argv[1:4]
N = 10_000
res = {"points": {}}
for tag in ("p0.00200", "p0.00300"):
    zn, zo = np.load(f"{new}/{tag}.npz"), np.load(f"{old}/{tag}.npz")
    mn, mo = json.load(open(f"{new}/{tag}.json")), json.load(open(f"{old}/{tag}.json"))
    arrays = {k: bool(np.array_equal(zn[k], zo[k][:N])) for k in zn.files if k not in ("chunks", "models")}
    skip = ("secs",)
    sig = [{k: v == mo["models"][i].get(k) for k, v in m.items() if k not in skip} for i, m in enumerate(mn["models"])]
    r = {"arrays_equal": arrays, "chunks_new": mn["chunks"], "chunk_meta_equal": mn["chunk_meta"][0] == mo["chunk_meta"][0],
         "kind_family_circuit_equal": [mn[k] == mo[k] for k in ("kind", "family", "circuit_sha256")],
         "signature_fields_equal": sig, "rows_new": int(len(zn["obs"]))}
    r["pass"] = (all(arrays.values()) and r["chunk_meta_equal"] and all(r["kind_family_circuit_equal"])
                 and all(all(s.values()) for s in sig) and r["rows_new"] == N)
    res["points"][tag] = r
res["pass"] = all(r["pass"] for r in res["points"].values())
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"pass": res["pass"], **{t: r["pass"] for t, r in res["points"].items()}}))
