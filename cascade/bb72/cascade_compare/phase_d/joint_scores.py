"""The network's class scores for the joint decision (phase_d/joint.py): S55'-deep's 4,096 class scores on the routed shots
of a read whose two decisions differ on any of Cascade's paths, from the fused engine itself (scored.Scored, ARB's
export), run again on those shots' saved detector rows (ARB's arb_scores.py, extended to the compiled paths).

  python joint_scores.py --read DIR [--compiled CDIR] --p 0.002 --ckpt S55p_deep.pt --lock d1_locked_depspam_fm.json

Routed: S55'-deep's margin (float64 of the read's float32) below the lock's tau1 at p (pc_route.masks). The rows: the
routed shots where S55'-deep's decision differs from C14''s on the eager path (the read DIR: model 0 C14', model 1
S55'-deep) or, with --compiled (d1_compiled.py read --joint on the same read), on compiled@1 (bits1) or on compiled@4096
over the routed queue (bitsR); the union, in shot order. The scores do not depend on Cascade's path. Checks, every one
bitwise, on every exported shot (a failed check exits 1): (1) pc_fused.Batched, the read's engine, run again on these
rows gives the read's decisions and margins; (2) Scored gives the same decisions, margins and per-program top two as
Batched; (3) the exported scores give each program's m1, i1 and m2, the decision (first argmax over the 4,096 classes;
class bit j = observable j) and the margin (top one minus top two in fp32); (4) joint._k_jvote, the timed pipelines'
vote, on Scored's pool output s gives the exported scores, every bit. Writes DIR/p<p>_jscores.npy (float32, rows x
4,096), DIR/p<p>_jscores_rows.npy (int64 shot indices of the read) and DIR/p<p>_jscores.json (counts per path, checks,
sha256, the engine's signature)."""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from bb import cascade_eval as ce

import d1_common as d1
import joint
import pc_common as pc
import pc_eval  # noqa: F401  (ce.Predictor on the fused engine, the GPU guard)
import scored

NDET, K = 432, 12


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def u32(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def differ_sets(read, compiled, p, tau1):
    """-> (routed bool (n,), {path: bool (n,) routed rows whose decisions differ}, S's class per shot)."""
    tag = pc.p_tag(p)
    z = np.load(os.path.join(read, f"{tag}.npz"))
    meta = json.load(open(os.path.join(read, f"{tag}.json")))
    if [m["kind"] for m in meta["models"]] != ["cascade", "convlines"]:
        raise SystemExit(f"{read}: models are not (C14', S55'-deep)")
    routed = z["conf1"].astype(np.float64) < tau1
    bN = np.unpackbits(z["pred1"], axis=1, count=K)
    sets = {"eager": routed & (z["pred0"] != z["pred1"]).any(1)}
    if compiled:
        c = np.load(os.path.join(compiled, f"{tag}.npz"))
        ri = c["routed"]
        if not np.array_equal(ri, np.nonzero(routed)[0]):
            raise SystemExit(f"{compiled}: its routed rows are not this read's")
        b1 = np.unpackbits(c["bits1"], axis=1, count=K)
        sets["compiled@1"] = routed & (b1 != bN).any(1)
        bR = np.unpackbits(c["bitsR"], axis=1, count=K)
        dR = np.zeros(len(routed), bool)
        dR[ri] = (bR != bN[ri]).any(1)
        sets["compiled@4096"] = dR
    return routed, sets, z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--compiled", default="")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--ckpt", required=True, help="S55'-deep's checkpoint (the read's model 1)")
    ap.add_argument("--lock", default=d1.LOCK)
    ap.add_argument("--chunk-rows", type=int, default=16384, help="rows per check block on the CPU")
    a = ap.parse_args()
    t0 = time.time()
    p, tag = a.p, pc.p_tag(a.p)
    L = d1.lock(a.lock)
    _, c = d1.config_at(L, p)
    tau1 = float(c["tau"]["tau1"])
    meta = json.load(open(os.path.join(a.read, f"{tag}.json")))
    routed, sets, z = differ_sets(a.read, a.compiled, p, tau1)
    conf1, pred1 = z["conf1"], z["pred1"]
    union = np.zeros(len(routed), bool)
    for v in sets.values():
        union |= v
    rows = np.nonzero(union)[0].astype(np.int64)
    n = len(rows)
    counts = {k: int(v.sum()) for k, v in sets.items()}
    print(f"{tag}: {len(conf1):,} shots, routed {int(routed.sum()):,}, rows {n:,} {counts}", flush=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False                     # bb.cascade_eval's settings
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    pr = ce.Predictor(a.ckpt, dev, "ema", "bf16")
    if pr.kind != "convlines" or pr.sig["sha256"] != meta["models"][1]["sha256"]:
        raise SystemExit(f"{a.ckpt}: not the read's S55'-deep ({pr.sig['sha256'][:16]})")
    det = np.unpackbits(z["det"][rows], axis=1, count=NDET, bitorder="little")
    dg = torch.as_tensor(det, device=dev)
    orig = scored.run_with_programs(pr.fused, dg)
    eng = scored.Scored(pr.model, pc.FUSED_READ_B)
    exp = scored.run_with_programs(eng, dg, scores=True)
    jv_bad = 0                                                        # check 4: joint._k_jvote on Scored's s
    for lo in range(0, n, eng.nmax):
        d = dg[lo:lo + eng.nmax].contiguous()
        eng(d)
        k = len(d)
        jv = joint.scores_gpu(eng, eng.s[:k])
        jv_bad += int((u32(jv) != u32(eng.sc[:k].cpu().numpy())).any(1).sum())
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
    bad = {"program_m1": 0, "program_i1": 0, "program_m2": 0, "argmax_cls": 0, "margin": 0, "jvote_scores": jv_bad}
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
    out = os.path.join(a.read, f"{tag}_jscores.npy")
    np.save(out, sc)
    np.save(os.path.join(a.read, f"{tag}_jscores_rows.npy"), rows)
    rec = {"p": p, "tau1": tau1, "shots": int(len(conf1)), "routed": int(routed.sum()), "rows": n,
           "differ_per_path": counts, "paths": sorted(sets), "compiled": os.path.abspath(a.compiled) if a.compiled
           else None, "checks": ck, "pass": bool(ok), "ckpt": os.path.abspath(a.ckpt), "ckpt_sha256": pr.sig["sha256"],
           "engine": scored.signature(pc.FUSED_READ_B), "joint_py": joint.source(),
           "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__, "scores_sha256": sha256_file(out),
           "rows_sha256": sha256_file(os.path.join(a.read, f"{tag}_jscores_rows.npy")),
           "secs": round(time.time() - t0, 1)}
    json.dump(rec, open(os.path.join(a.read, f"{tag}_jscores.json"), "w"), indent=1)
    print("joint_scores:", json.dumps({k: rec[k] for k in ("p", "rows", "differ_per_path", "pass", "checks", "secs")}),
          flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
