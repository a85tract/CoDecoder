"""ARB's score export (PLAN.md step 2): S55'-deep's 4,096 class scores on the routed shots of a read where the two
decisions differ, from the fused engine itself (arb_fused.Scored), run again on those shots' saved detector rows.

  python arb_scores.py --read DIR --p 0.002 --ckpt S55p_deep.pt --lock d1_locked_depspam_fm.json

Routed: S55'-deep's margin (float64 of the read's float32) below D1's frozen tau1 at p (the lock), as pc_route.masks;
differ: the read's packed decisions of C14' (model 0) and S55'-deep (model 1) differ. Checks, every one bitwise, on every
exported shot (a failed check exits 1 and nothing is used): (1) pc_fused.Batched, the read's engine, run again on these
rows gives the read's decisions and margins; (2) Scored gives the same decisions, margins and per-program top two
(m1, i1, m2) as Batched; (3) the exported scores give each program's m1, i1 and m2, the decision (first argmax over the
4,096 classes; class bit j = observable j) and the margin (top one minus top two in fp32). Writes DIR/p<p>_scores.npy
(float32, rows x 4,096, row k = shot rows[k]), DIR/p<p>_scores_rows.npy (int64 shot indices of the read) and
DIR/p<p>_scores.json (counts, checks, sha256, the engine's signature)."""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from bb import cascade_eval as ce

import arb_fused
import d1_common as d1
import pc_common as pc
import pc_eval  # noqa: F401  (ce.Predictor on the fused engine, the GPU guard)

TAU1 = {0.002: 7.3331888389587405, 0.003: 7.60439167022705}      # D1's frozen tau1 (X2's bb_d1.py), checked with the lock
NDET, K = 432, 12


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def u32(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--ckpt", required=True, help="S55'-deep's checkpoint (the read's model 1)")
    ap.add_argument("--lock", required=True)
    ap.add_argument("--chunk-rows", type=int, default=16384, help="rows per check block on the CPU")
    a = ap.parse_args()
    t0 = time.time()
    p, tag = a.p, pc.p_tag(a.p)
    L = d1.lock(a.lock)
    _, c = d1.config_at(L, p)
    tau1 = float(c["tau"]["tau1"])
    if tau1 != TAU1[p] or c["S"] != "S55p_deep" or c["B"] != "C14p":
        raise SystemExit(f"lock {a.lock}: configuration at {p} is not D1's ({c['S']}, {c['B']}, tau1 {tau1})")
    meta = json.load(open(os.path.join(a.read, f"{tag}.json")))
    kinds = [m["kind"] for m in meta["models"]]
    if kinds != ["cascade", "convlines"]:
        raise SystemExit(f"{a.read}: models {kinds}, expected C14' then S55'-deep")
    z = np.load(os.path.join(a.read, f"{tag}.npz"))
    conf1 = z["conf1"]
    routed = conf1.astype(np.float64) < tau1
    pred0, pred1 = z["pred0"], z["pred1"]
    differ = (pred0 != pred1).any(1)
    rows = np.nonzero(routed & differ)[0].astype(np.int64)
    n = len(rows)
    print(f"{tag}: {len(conf1):,} shots, routed {int(routed.sum()):,}, routed with different decisions {n:,}", flush=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False                     # bb.cascade_eval's settings
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    pr = ce.Predictor(a.ckpt, dev, "ema", "bf16")
    if pr.kind != "convlines" or pr.sig["sha256"] != meta["models"][1]["sha256"]:
        raise SystemExit(f"{a.ckpt}: not the read's S55'-deep ({pr.sig['sha256'][:16]})")
    det = np.unpackbits(z["det"][rows], axis=1, count=NDET, bitorder="little")
    dg = torch.as_tensor(det, device=dev)
    orig = arb_fused.run_with_programs(pr.fused, dg)
    eng = arb_fused.Scored(pr.model, pc.FUSED_READ_B)
    exp = arb_fused.run_with_programs(eng, dg, scores=True)
    del dg
    bits_read = np.unpackbits(pred1[rows], axis=1, count=K)
    cls_read = (bits_read.astype(np.int64) << np.arange(K)).sum(1)
    ck = {}
    ck["batched_bits_equal_read"] = bool(np.array_equal(orig["bits"].numpy(), bits_read))
    ck["batched_margin_equal_read"] = bool(np.array_equal(u32(orig["margin"].numpy()), u32(conf1[rows])))
    for k in ("bits", "margin", "m1", "i1", "m2"):
        x, y = exp[k].numpy(), orig[k].numpy()
        if x.dtype == np.float32:
            x, y = u32(x), u32(y)
        ck[f"scored_{k}_equal_batched"] = bool(np.array_equal(x, y))
    if "cls1" in z.files:
        ck["read_cls_equal_bits"] = bool(np.array_equal(z["cls1"][rows].astype(np.int64), cls_read))
    sc = exp["sc"].numpy()
    m1, i1, m2 = exp["m1"].numpy(), exp["i1"].numpy(), exp["m2"].numpy()
    bad = {"program_m1": 0, "program_i1": 0, "program_m2": 0, "argmax_cls": 0, "margin": 0}
    by = eng.NY // eng.NP
    for lo in range(0, n, a.chunk_rows):
        s = sc[lo:lo + a.chunk_rows]
        g = s.reshape(len(s), eng.NP, by)
        pm1 = g.max(2)
        pi1 = g.argmax(2)
        g2 = g.copy()
        np.put_along_axis(g2, pi1[:, :, None], -np.inf, axis=2)
        pm2 = g2.max(2)
        bad["program_m1"] += int((u32(pm1) != u32(m1[lo:lo + len(s)])).any(1).sum())
        bad["program_i1"] += int(((pi1 + np.arange(eng.NP) * by) != i1[lo:lo + len(s)]).any(1).sum())
        bad["program_m2"] += int((u32(pm2) != u32(m2[lo:lo + len(s)])).any(1).sum())
        am = s.argmax(1)
        bad["argmax_cls"] += int((am != cls_read[lo:lo + len(s)]).sum())
        top1 = s[np.arange(len(s)), am]
        s2 = s.copy()
        s2[np.arange(len(s)), am] = -np.inf
        top2 = s2.max(1)
        bad["margin"] += int((u32(top1 - top2) != u32(conf1[rows[lo:lo + len(s)]])).sum())
    ck["scores_mismatches"] = bad
    ok = all(v for k, v in ck.items() if isinstance(v, bool)) and not any(bad.values())
    out = os.path.join(a.read, f"{tag}_scores.npy")
    np.save(out, sc)
    np.save(os.path.join(a.read, f"{tag}_scores_rows.npy"), rows)
    rec = {"p": p, "tau1": tau1, "shots": int(len(conf1)), "routed": int(routed.sum()), "rows": n,
           "checks": ck, "pass": bool(ok), "ckpt": os.path.abspath(a.ckpt), "ckpt_sha256": pr.sig["sha256"],
           "engine": arb_fused.signature(pc.FUSED_READ_B), "gpu": torch.cuda.get_device_name(dev),
           "torch": torch.__version__, "scores_sha256": sha256_file(out),
           "rows_sha256": sha256_file(os.path.join(a.read, f"{tag}_scores_rows.npy")),
           "secs": round(time.time() - t0, 1)}
    json.dump(rec, open(os.path.join(a.read, f"{tag}_scores.json"), "w"), indent=1)
    print("arb_scores:", json.dumps({k: rec[k] for k in ("p", "rows", "pass", "checks", "secs")}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
