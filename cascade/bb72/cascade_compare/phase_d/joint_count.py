"""The joint decision's decisions and failures per Cascade path, from the reads (phase_d/joint.py), and the bitwise check
of a timed pipeline against them.

  count    python joint_count.py count --read DIR --compiled CDIR --p P --w W [--lock L] --out F.json [--answers F.npz]
  compare  python joint_count.py compare --pipe PIPE.npz --read DIR --compiled CDIR --p P --w W [--lock L] --out F.json

Inputs: the eager read DIR (joint_read.py: p<p>.npz with the detector rows, p<p>_z32.npy; joint_scores.py:
p<p>_jscores.npy / _rows.npy, which must have passed its checks) and the compiled read CDIR of the same read
(d1_compiled.py read --joint: bits1, z1R, bits4, routed, bitsR, zR). Routed: S55'-deep's margin (float64 of the read's
float32) below the lock's tau1 at p. Per path (eager: C14' on bb.cascade_eval's backend, the read's pred0 and z32;
compiled@1: bits1, z1R; compiled@4096: the routed queue, bitsR, zR), the composition answers S55'-deep's decision, or
C14''s on a routed shot; the joint composition answers the same except on the routed shots whose two decisions differ,
where it answers joint.joint_np(scores, z, w) (the GPU's fp32 arithmetic, bit for bit). count: per path the failures of
S55'-deep alone, C14' alone (eager: pred0; compiled@1: bits1; compiled@4096: bits4, Cascade alone batched), the
composition and the joint composition, the joint step's rows, changes, repairs, breaks and third answers; halves by chunk
parity are reported too (fit = even chunks). compare: a pipeline's saved decisions (m2.py pipeline --joint-w W --save)
against the joint composition on the path its mode times (single: compiled@1; batched: compiled@4096), on the same
shots (the detector rows must be equal): every shot's decision and route; the differing shots are listed."""
import argparse
import json
import os
import time

import numpy as np

import d1_common as d1
import joint
import pc_common as pc

K = 12
PATHS = ("eager", "compiled@1", "compiled@4096")


def load(read, compiled, p, lock):
    tag = pc.p_tag(p)
    L = d1.lock(lock)
    _, c = d1.config_at(L, p)
    tau1 = float(c["tau"]["tau1"])
    z = np.load(os.path.join(read, f"{tag}.npz"))
    meta = json.load(open(os.path.join(read, f"{tag}.json")))
    if [m["kind"] for m in meta["models"]] != ["cascade", "convlines"]:
        raise SystemExit(f"{read}: models are not (C14', S55'-deep)")
    zrec = json.load(open(os.path.join(read, f"{tag}_z32.json")))
    if not (zrec["float16_equals_read_z"] and zrec["minabs_equals_read_conf_bitwise"]):
        raise SystemExit(f"{read}: the float32 logits did not pass their checks")
    srec = json.load(open(os.path.join(read, f"{tag}_jscores.json")))
    if not srec["pass"]:
        raise SystemExit(f"{read}: the score export did not pass its checks")
    o = {"tau1": tau1, "obs": z["obs"], "pred0": z["pred0"], "pred1": z["pred1"], "conf1": z["conf1"],
         "chunk": z["chunk"], "z32": np.load(os.path.join(read, f"{tag}_z32.npy"), mmap_mode="r"),
         "rows": np.load(os.path.join(read, f"{tag}_jscores_rows.npy")),
         "sc": np.load(os.path.join(read, f"{tag}_jscores.npy"), mmap_mode="r"), "det": z["det"], "meta": meta,
         "srec": srec}
    if compiled:
        cz = np.load(os.path.join(compiled, f"{tag}.npz"))
        if "zR" not in cz.files:
            raise SystemExit(f"{compiled}: not a read with the logits (d1_compiled.py read --joint)")
        o["c"] = {k: cz[k] for k in cz.files}
        o["crec"] = json.load(open(os.path.join(compiled, f"{tag}.json")))
    return o


def paths_of(o, w):
    """-> {path: dict(yB (n,) Cascade alone, comp, joint (n,) classes, d (rows of the joint step), cls (their
    answers))}, yS, yO, routed."""
    pw = 1 << np.arange(K)
    cls = lambda packed: (np.unpackbits(packed, axis=1, count=K).astype(np.int64) * pw).sum(1)   # noqa: E731
    yS, yO = cls(o["pred1"]), cls(o["obs"])
    routed = o["conf1"].astype(np.float64) < o["tau1"]
    ri = np.nonzero(routed)[0]
    pos = {int(r): k for k, r in enumerate(o["rows"])}
    src = {"eager": (cls(o["pred0"]), cls(o["pred0"])[ri], np.asarray(o["z32"][ri], np.float32))}
    if "c" in o:
        c = o["c"]
        if not np.array_equal(c["routed"], ri):
            raise SystemExit("the compiled read's routed rows are not this read's")
        src["compiled@1"] = (cls(c["bits1"]), cls(c["bits1"])[ri], c["z1R"])
        src["compiled@4096"] = (cls(c["bits4"]), cls(c["bitsR"]), c["zR"])
    out = {}
    for name, (alone, yBr, zr) in src.items():
        comp = yS.copy()
        comp[ri] = yBr
        dm = yBr != yS[ri]
        d = ri[dm]
        miss = [int(r) for r in d if int(r) not in pos]
        if miss:
            raise SystemExit(f"{name}: {len(miss)} rows of the joint step have no exported scores (first {miss[:5]})")
        sc = np.stack([o["sc"][pos[int(r)]] for r in d]) if len(d) else np.zeros((0, joint.NY), np.float32)
        ans = joint.joint_np(sc, zr[dm], w) if len(d) else np.zeros(0, np.int64)
        jc = comp.copy()
        jc[d] = ans
        out[name] = {"alone": alone, "comp": comp, "joint": jc, "d": d, "ans": ans, "yB_d": yBr[dm]}
    return out, yS, yO, routed


def tally(v, yS, yO, sel):
    fS, fB, fc, fj = (yS != yO)[sel], (v["alone"] != yO)[sel], (v["comp"] != yO)[sel], (v["joint"] != yO)[sel]
    dsel = sel[v["d"]]
    a, yB, yOd, ySd = v["ans"][dsel], v["yB_d"][dsel], yO[v["d"]][dsel], yS[v["d"]][dsel]
    return {"shots": int(sel.sum()), "fail_S": int(fS.sum()), "fail_C": int(fB.sum()), "fail_comp": int(fc.sum()),
            "fail_joint": int(fj.sum()), "joint_rows": int(dsel.sum()), "changed": int((a != yB).sum()),
            "repairs": int(((yB != yOd) & (a == yOd)).sum()), "breaks": int(((yB == yOd) & (a != yOd)).sum()),
            "to_S": int((a == ySd).sum()), "third": int(((a != ySd) & (a != yB)).sum()),
            "rel_reduction": (int(fc.sum()) - int(fj.sum())) / int(fc.sum()) if fc.sum() else 0.0}


def count(a):
    t0 = time.time()
    o = load(a.read, a.compiled, a.p, a.lock)
    res, yS, yO, routed = paths_of(o, a.w)
    fit = o["chunk"] % 2 == 0
    rec = {"p": a.p, "w": a.w, "joint": joint.record(a.w), "tau1": o["tau1"], "shots": int(len(yS)),
           "routed": int(routed.sum()), "read": os.path.abspath(a.read),
           "compiled": os.path.abspath(a.compiled) if a.compiled else None, "paths": {}}
    for name, v in res.items():
        rec["paths"][name] = {h: tally(v, yS, yO, sel) for h, sel in
                              (("all", np.ones(len(yS), bool)), ("fit", fit), ("eval", ~fit))}
    rec["secs"] = round(time.time() - t0, 1)
    json.dump(rec, open(a.out, "w"), indent=1)
    if a.answers:
        np.savez_compressed(a.answers, **{f"{k.replace('@', '_')}_{x}": v[x] for k, v in res.items()
                                          for x in ("d", "ans")})
    for name, r in rec["paths"].items():
        print(name, json.dumps(r["all"]), flush=True)


def compare(a):
    t0 = time.time()
    o = load(a.read, a.compiled, a.p, a.lock)
    res, yS, yO, routed = paths_of(o, a.w)
    P = np.load(a.pipe)
    meta = json.loads(str(P["meta"]))
    mode = meta["mode"]
    path = {"single": "compiled@1", "batched": "compiled@4096"}[mode]
    n = int(meta["n"])
    rec = {"pipe": os.path.abspath(a.pipe), "mode": mode, "path": path, "p": a.p, "w": a.w, "shots": n,
           "pipe_joint": meta.get("joint"), "pipe_path_B": meta.get("path_B"), "pipe_path_S": meta.get("path_S")}
    if meta.get("joint", {}).get("w32_hex") != joint.w32_of(a.w)[1]:
        raise SystemExit(f"{a.pipe}: the pipeline's w {meta.get('joint')} is not {a.w}")
    same_rows = len(o["det"]) >= n and np.array_equal(P["det"], o["det"][:n])
    rec["det_equal"] = bool(same_rows)
    if not same_rows:
        raise SystemExit(f"{a.pipe}: its detector rows are not the read's first {n:,}")
    pw = 1 << np.arange(K)
    yP = (np.unpackbits(P["pred"], axis=1, count=K).astype(np.int64) * pw).sum(1)
    rP = P["route"] >= 1
    v = res[path]
    rec["route_differ"] = int((rP != routed[:n]).sum())
    dd = np.nonzero(yP != v["joint"][:n])[0]
    rec["decision_differ"] = int(len(dd))
    rec["differ_rows"] = [int(x) for x in dd[:200]]
    rec["joint_rows"] = int((v["d"] < n).sum())
    rec["changed_by_joint"] = int((v["joint"][:n] != v["comp"][:n]).sum())
    rec["pipe_fail"] = int((yP != yO[:n]).sum())
    rec["read_fail_joint"] = int((v["joint"][:n] != yO[:n]).sum())
    rec["read_fail_comp"] = int((v["comp"][:n] != yO[:n]).sum())
    rec["pass"] = rec["route_differ"] == 0 and rec["decision_differ"] == 0
    rec["secs"] = round(time.time() - t0, 1)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: v_ for k, v_ in rec.items() if k != "differ_rows"}), flush=True)
    raise SystemExit(0 if rec["pass"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("count", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--read", required=True)
        s.add_argument("--compiled", default="" if name == "count" else None, required=name == "compare")
        s.add_argument("--p", type=float, required=True)
        s.add_argument("--w", type=float, required=True)
        s.add_argument("--lock", default=d1.LOCK)
        s.add_argument("--out", required=True)
        if name == "count":
            s.add_argument("--answers", default="")
        else:
            s.add_argument("--pipe", required=True)
    a = ap.parse_args()
    {"count": count, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
