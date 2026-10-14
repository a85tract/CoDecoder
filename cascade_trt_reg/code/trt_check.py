"""CASCADE-TRT step 2 (exploratory): TensorRT's decisions against the compiled path's on fresh shots (seed kind 47,
never a test set), per Cascade: 100,000 shots at each of its two headline p (200,000), at the batch its batched path
uses (BB72 4,096, d = 13 16); and the batch-1 engines on the first --n1 shots at each p. Per comparison: shots whose
decision differs, failures of each engine, the shots only one of them fails, the exact McNemar p (two-sided binomial
test on the discordant failures), and the largest confidence / logit difference.

    python trt_check.py --net bb --plans DIR --ckpt C14p.pt --out check_bb.json              (both engines, one process)
    python trt_check.py --net bb --engine compiled|trt --save E.npz ...                       (one engine per process)
    python trt_check.py --net bb --compare compiled.npz trt.npz --out check_bb.json           (no GPU)

The one-process form crashed for BB72 (segmentation fault in the compiled batch-1 graph's replay after the TensorRT
engines had run in the same process); the BB72 check therefore runs each engine in its own process on the same shots
(the shots' sha256 is saved and compared)."""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_alone as ta  # noqa: E402

PS = {"bb": (0.002, 0.003), "d13": (0.006, 0.0035)}
CHUNKS = range(10, 20)                       # 100,000 shots per p (the timing processes read chunks 0-3)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def decide(net, g, det):
    """(decision (n, k) uint8, confidence or logit (n,) float32) of graph g on det (n, D) on the GPU, g.B at a time."""
    import numpy as np
    if net == "bb":
        b, c = g.run(det)
        return b.cpu().numpy(), c.float().cpu().numpy()
    outs_l, outs_d = [], []
    for i in range(0, len(det), g.n):
        lg, dc = g.run_rows(det[i:i + g.n])
        outs_l.append(lg)
        outs_d.append(dc)
    return np.concatenate(outs_d)[:, None], np.concatenate(outs_l)


def compare(net, obs, ref, new):
    import numpy as np
    (bd, bc), (td, tc_) = ref, new
    obs = obs.reshape(len(obs), -1)
    differ = (bd != td).any(1)
    fb, ft = (bd != obs).any(1), (td != obs).any(1)
    only_b, only_t = int((fb & ~ft).sum()), int((ft & ~fb).sum())
    return {"shots": int(len(obs)), "decisions_differ": int(differ.sum()), "agreement": 1 - float(differ.mean()),
            "fail_compiled": int(fb.sum()), "fail_trt": int(ft.sum()), "only_compiled_fails": only_b,
            "only_trt_fails": only_t, "mcnemar_exact_p": mcnemar_exact(only_b, only_t),
            "max_abs_diff_" + ("conf" if net == "bb" else "logit"): float(np.abs(bc - tc_).max()),
            "mean_abs_diff": float(np.abs(bc - tc_).mean())}


def totals(res):
    tot = {"bB": [], "b1": []}
    for r in res["p"].values():
        for k in tot:
            tot[k].append(r[k])
    for k, rs in tot.items():
        s = {f: sum(x[f] for x in rs) for f in ("shots", "decisions_differ", "fail_compiled", "fail_trt",
                                                 "only_compiled_fails", "only_trt_fails")}
        s["agreement"] = 1 - s["decisions_differ"] / s["shots"]
        s["mcnemar_exact_p"] = mcnemar_exact(s["only_compiled_fails"], s["only_trt_fails"])
        res["total_" + k] = s


def one_engine(a):
    """Decisions of one engine on the check's shots -> npz (with the shots' sha256 per p)."""
    import numpy as np
    import torch
    dev = ta.setup()
    if a.only:                                       # one graph in this process (the compiled BB72 b1 graph asserted
        g1 = gB = None                               # in a process that held both compiled graphs)
        if a.engine == "compiled" and a.net == "bb":
            import types
            import pc_speed as ps
            model, sha = ta.tc.load_net(a.net, dev, a.ckpt)
            pr = types.SimpleNamespace(kind="cascade", model=model)
            g = ps.build(pr, "compiled", 1 if a.only == "b1" else ta.BATCH[a.net])
        else:
            raise SystemExit("--only: compiled BB72 only")
        g1, gB = (g, None) if a.only == "b1" else (None, g)
    else:
        g1, gB, sha = ta.paths(a.net, a.engine, dev, a.plans, a.ckpt)
    keep, meta = {}, {"net": a.net, "engine": a.engine, "ckpt_sha256": sha, "n1": a.n1, "only": a.only,
                      "trt": {"b1": getattr(g1, "info", lambda: None)(), "bB": getattr(gB, "info", lambda: None)()}}
    dets = []
    for i, p in enumerate(PS[a.net]):
        det, obs = ta.shots(a.net, p, CHUNKS)
        meta[f"shots_sha256_{i}"] = hashlib.sha256(np.ascontiguousarray(det).tobytes()).hexdigest()
        keep[f"obs{i}"] = np.asarray(obs)
        dets.append(det)
    # batch 1 first: the compiled BB72 batch-1 graph hit a device-side assert (index out of bounds in the token gather)
    # when replayed after the compiled batch-4,096 graph had run in the same process
    for i, det in enumerate(dets):
        n1 = min(a.n1, len(det))
        if g1 is not None:
            keep[f"d1{i}"], keep[f"c1{i}"] = decide(a.net, g1, torch.as_tensor(det[:n1], device=dev))
    for i, det in enumerate(dets):
        if gB is not None:
            keep[f"dB{i}"], keep[f"cB{i}"] = decide(a.net, gB, torch.as_tensor(det, device=dev))
    np.savez(a.save, meta=json.dumps(meta), **keep)
    print(json.dumps(meta), flush=True)


def from_saved(a):
    import numpy as np
    A, B = (dict(np.load(f)) for f in a.compare)
    for extra in a.merge:                            # --only files of the compiled side, merged into A
        X = dict(np.load(extra))
        mx, m0 = json.loads(str(X["meta"])), json.loads(str(A["meta"]))
        if mx["engine"] != "compiled" or any(mx[k] != m0[k] for k in m0 if k.startswith("shots_sha256")):
            raise SystemExit(f"{extra}: not the compiled side on the same shots")
        A.update({k: v for k, v in X.items() if k != "meta"})
    ma, mb = json.loads(str(A["meta"])), json.loads(str(B["meta"]))
    if ma["engine"] != "compiled" or mb["engine"] != "trt":
        raise SystemExit("--compare COMPILED.npz TRT.npz")
    res = {"net": a.net, "ckpt_sha256": ma["ckpt_sha256"], "kind": ta.KIND, "chunks": [CHUNKS.start, CHUNKS.stop],
           "batch": ta.BATCH[a.net], "n1": ma["n1"], "trt": mb["trt"], "processes": "one per engine", "p": {}}
    for i, p in enumerate(PS[a.net]):
        if ma[f"shots_sha256_{i}"] != mb[f"shots_sha256_{i}"]:
            raise SystemExit(f"p = {p}: the two processes drew different shots")
        if not np.array_equal(A[f"obs{i}"], B[f"obs{i}"]):
            raise SystemExit(f"p = {p}: observables differ")
        obs = A[f"obs{i}"]
        n1 = len(A[f"d1{i}"])
        res["p"][str(p)] = {
            "bB": compare(a.net, obs, (A[f"dB{i}"], A[f"cB{i}"]), (B[f"dB{i}"], B[f"cB{i}"])),
            "b1": compare(a.net, obs[:n1], (A[f"d1{i}"], A[f"c1{i}"]), (B[f"d1{i}"], B[f"c1{i}"]))}
        res["p"][str(p)]["shots_sha256"] = ma[f"shots_sha256_{i}"]
    totals(res)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"total_bB": res["total_bB"], "total_b1": res["total_b1"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, choices=["bb", "d13"])
    ap.add_argument("--plans", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n1", type=int, default=20_000, help="shots per p for the batch-1 engines")
    ap.add_argument("--engine", default="both", choices=["both", "compiled", "trt"])
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--only", default="", choices=["", "b1", "bB"], help="one graph per process (compiled BB72)")
    ap.add_argument("--merge", nargs="*", default=[], help="with --compare: more npz files of the compiled side")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.compare:
        return from_saved(a)
    if a.engine != "both":
        return one_engine(a)
    import numpy as np
    import torch
    dev = ta.setup()
    t0 = time.time()
    c1, cB, sha = ta.paths(a.net, "compiled", dev, a.plans, a.ckpt)
    t1, tB, _ = ta.paths(a.net, "trt", dev, a.plans, a.ckpt)
    res = {"net": a.net, "ckpt_sha256": sha, "kind": ta.KIND, "chunks": [CHUNKS.start, CHUNKS.stop],
           "batch": ta.BATCH[a.net], "n1": a.n1, "trt": {"b1": t1.info(), "bB": tB.info()},
           "processes": "one process, both engines", "p": {}}
    for p in PS[a.net]:
        det, obs = ta.shots(a.net, p, CHUNKS)
        dg = torch.as_tensor(det, device=dev)
        obs = np.asarray(obs)
        r = {"bB": compare(a.net, obs, decide(a.net, cB, dg), decide(a.net, tB, dg))}
        n1 = min(a.n1, len(det))
        r["b1"] = compare(a.net, obs[:n1], decide(a.net, c1, dg[:n1]), decide(a.net, t1, dg[:n1]))
        res["p"][str(p)] = r
        print(json.dumps({"p": p, **r}), flush=True)
    totals(res)
    res["secs"] = round(time.time() - t0, 1)
    res["torch"] = torch.__version__
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"total_bB": res["total_bB"], "total_b1": res["total_b1"]}), flush=True)


if __name__ == "__main__":
    main()
