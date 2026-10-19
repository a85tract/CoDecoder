"""ARB's read (PLAN.md step 2): D1's eager read, phase_c/pc_eval.py unchanged (bb.cascade_eval: C14' on the evaluation
backend under bf16 autocast in batches of 8,192 and the 1,808 left of each 10,000-shot chunk, S55'-deep on the fused
engine, pc_common's GPU guard, the bounded sampling prefetch of 2 x --procs chunks), with one output added: C14''s 12
logits in float32 for every shot. cascade_eval keeps them as float16 (z<i>); the float32 copy is taken from the same
call's output, before that cast, and written to SAVE/p<p>_z32.npy in the read's row order, with SAVE/p<p>_z32.json
(its sha256 and two checks: float16 of it equals the read's z<i>, and min |logit| equals the read's conf<i> bitwise).
Run through arb_shim.py (ARB's seeds), with cascade_eval's arguments, one p per run, as D1's job ran pc_eval.py:

  python arb_shim.py arb_read.py --family depspam_fm --ckpt C14p.pt,S55p_deep.pt --prec bf16 --kind 14 --p 0.002 \
      --chunks 6433102:6434202 --procs 8 --save OUT --save-det --out OUT/rows_0.002.json"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from bb import cascade_eval as ce

import pc_eval  # noqa: F401  (patches ce.Predictor: the fused engine, the GPU guard, the driver in the signature)

Z32 = []
_call = ce.Predictor.__call__


def _call_z32(self, det, bs=None):
    out = _call(self, det, bs)
    if self.kind == "cascade":
        Z32.append(out["z"].cpu().numpy())               # float32 of the network's bf16 output, before any cast
    return out


ce.Predictor.__call__ = _call_z32


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--p", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--ckpt", required=True)
    a, _ = ap.parse_known_args()
    if "," in a.p:
        raise SystemExit("arb_read: one p per run")
    t0 = time.time()
    sys.argv[0] = ce.__file__
    ce.main()
    p = float(a.p)
    meta = json.load(open(os.path.join(a.save, f"p{p:.5f}.json")))
    kinds = [m["kind"] for m in meta["models"]]
    if kinds.count("cascade") != 1:
        raise SystemExit(f"arb_read: {kinds}: exactly one Cascade network expected")
    i = kinds.index("cascade")
    z32 = np.concatenate(Z32)
    path = os.path.join(a.save, f"p{p:.5f}_z32.npy")
    np.save(path, z32)
    zr = np.load(os.path.join(a.save, f"p{p:.5f}.npz"))
    z16, conf = zr[f"z{i}"], zr[f"conf{i}"]
    rec = {"p": p, "rows": int(len(z32)), "model_index": i, "model_sha256": meta["models"][i]["sha256"],
           "shape": list(z32.shape), "dtype": str(z32.dtype),
           "float16_equals_read_z": bool(len(z16) == len(z32) and np.array_equal(z32.astype(np.float16).view(np.uint16),
                                                                                 z16.view(np.uint16))),
           "minabs_equals_read_conf_bitwise": bool(np.array_equal(np.abs(z32).min(1).view(np.uint32),
                                                                  conf.astype(np.float32).view(np.uint32))),
           "nonfinite": int((~np.isfinite(z32)).sum()), "secs": round(time.time() - t0, 1)}
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    rec["sha256"] = h.hexdigest()
    json.dump(rec, open(os.path.join(a.save, f"p{p:.5f}_z32.json"), "w"), indent=1)
    print("arb_read:", json.dumps(rec), flush=True)
    if not (rec["float16_equals_read_z"] and rec["minabs_equals_read_conf_bitwise"] and rec["nonfinite"] == 0):
        raise SystemExit("arb_read: the float32 logits do not match the read's outputs")


if __name__ == "__main__":
    main()
