"""Local test of arb_seedcheck.py (no GPU, no torch): a fake read of N chunks per p in bb.cascade_eval's save format
(det packed little-endian, chunk, chunk_meta with seeds and digests), drawn through the shim exactly as cascade_eval's
pool draws (cascade_data._eval_chunk in a spawn pool). Run through the shim with ARB_TRACE=OUT/trace.
  python arb_shim.py test_cpu_seedcheck.py OUT N P   (ARB_TRACE=OUT/trace_P)"""
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from bb import cascade_data as cdata

out, n, ps = sys.argv[1], int(sys.argv[2]), [float(sys.argv[3])]
C0 = 6433102
with mp.get_context("spawn").Pool(2) as pool:
    for p in ps:
        jobs = [("bb72", 14, p, j, 10000, "depspam_fm") for j in range(C0, C0 + n)]
        res = pool.map(cdata._eval_chunk, jobs)
        det = np.concatenate([r[0] for r in res])
        cm = [{"chunk": j, "seed": cdata.eval_seed(14, p, j, "depspam_fm"),
               "det_sha256": hashlib.sha256(np.packbits(r[0], axis=1).tobytes()).hexdigest()[:16]}
              for j, r in zip(range(C0, C0 + n), res)]
        np.savez(os.path.join(out, f"p{p:.5f}.npz"), det=np.packbits(det, axis=1, bitorder="little"),
                 chunk=np.repeat(np.arange(C0, C0 + n), 10000).astype(np.int32))
        json.dump({"p": p, "kind": 14, "family": "depspam_fm", "chunks": [C0, C0 + n], "chunk_meta": cm},
                  open(os.path.join(out, f"p{p:.5f}.json"), "w"))
print("fake reads written", flush=True)
