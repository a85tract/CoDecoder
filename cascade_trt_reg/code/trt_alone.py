"""CASCADE-TRT step 3 (exploratory): Cascade alone, one engine, in one fresh process, with the re-test harness's
timing functions: at batch 1 the GPU term (gpu_us) and the host round trip (host_rt: pinned H2D copy of one shot,
replay, D2H copy of the outputs, synchronise; 10,000 calls after 200 warm-up), at the batch the batched path uses the
amortized time per shot (amortized: input resident "dev" and host to host "h2h", at least 2 s each).
BB72 uses pc_speed's functions on pc_speed.Graph (compiled) or trt_common.TRTGraphBB; d = 13 uses sv_speed's copies of
the same functions on sv_common.Graph (compiled) or trt_common.TRTGraphD13. The SM clock is sampled in every window
(bb.timing.gpu_watch; BB_PIDNS=1 on a pod).

    python trt_alone.py --net bb --engine compiled|trt --rep R --plans DIR --ckpt C14p.pt --out rows.jsonl
    python trt_alone.py --net d13 --engine compiled|trt --rep R --plans DIR --out rows.jsonl"""
import argparse
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_common as tc  # noqa: E402

BATCH = {"bb": 4096, "d13": 16}
ROUNDS = {"bb": 6, "d13": 13}
P_TIME = {"bb": 0.002, "d13": 0.006}
KIND = 47                                     # CASCADE-TRT's fresh-shot seed kind (unused elsewhere)


def setup():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    return torch.device("cuda")


def shots(net, p, chunks, kind=KIND):
    """Fresh shots (det uint8, obs) of the network's setting: BB72 depspam_fm, d = 13 surface memory."""
    if net == "bb":
        from bb import cascade_data as cd
        return cd.eval_shots("bb72", p, chunks, kind=kind, procs=4, family="depspam_fm")
    from surface import surface_data as sd
    return sd.eval_shots(13, p, chunks, n=10000, kind=kind, procs=4)


def path_one(net, engine, dev, plans, B, model=None):
    """One timed path at batch B (the model already loaded for the compiled path)."""
    if engine == "trt":
        if net == "bb":
            return tc.TRTGraphBB(tc.plan_path(plans, net, B), B, 432, dev)
        from surface import surface_layout as sl
        return tc.TRTGraphD13(tc.plan_path(plans, net, B), B, sl.Layout(13).D, dev)
    if net == "bb":
        import pc_speed as ps
        return ps.build(types.SimpleNamespace(kind="cascade", model=model), "compiled", B)
    from surface import sv_common as sc
    return sc.Graph(sc.b_fn(model, "compiled"), B, model.D, dev)


def paths(net, engine, dev, plans, ckpt):
    """(graph at batch 1, graph at the batch, sha256 of the network file or '')."""
    Bb = BATCH[net]
    if engine == "trt":
        if net == "bb":
            return (tc.TRTGraphBB(tc.plan_path(plans, net, 1), 1, 432, dev),
                    tc.TRTGraphBB(tc.plan_path(plans, net, Bb), Bb, 432, dev), "")
        from surface import surface_layout as sl
        Dd = sl.Layout(13).D
        return (tc.TRTGraphD13(tc.plan_path(plans, net, 1), 1, Dd, dev),
                tc.TRTGraphD13(tc.plan_path(plans, net, Bb), Bb, Dd, dev), "")
    model, sha = tc.load_net(net, dev, ckpt)
    if net == "bb":
        import pc_speed as ps
        pr = types.SimpleNamespace(kind="cascade", model=model)
        return ps.build(pr, "compiled", 1), ps.build(pr, "compiled", Bb), sha
    from surface import sv_common as sc
    return (sc.Graph(sc.b_fn(model, "compiled"), 1, model.D, dev),
            sc.Graph(sc.b_fn(model, "compiled"), Bb, model.D, dev), sha)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, choices=["bb", "d13"])
    ap.add_argument("--engine", required=True, choices=["compiled", "trt"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--plans", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rt", type=int, default=10_000)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--seq", action="store_true", help="one Cascade graph per phase (b1 timed and freed first)")
    a = ap.parse_args()
    import numpy as np
    import torch
    from bb.timing import gpu_watch
    if a.net == "bb":
        import pc_speed as H
    else:
        from surface import sv_speed as H
    dev = setup()
    t0 = time.time()
    if a.seq:                                    # one Cascade graph at a time: b1 built, timed, freed, then the batch
        import gc
        model, sha = (None, "") if a.engine == "trt" else tc.load_net(a.net, dev, a.ckpt)
        det, _ = shots(a.net, P_TIME[a.net], range(4))
        rows = torch.as_tensor(det) if a.net == "bb" else det
        g1 = path_one(a.net, a.engine, dev, a.plans, 1, model)
        build_s = time.time() - t0
        stop = gpu_watch()
        gu = H.gpu_us(g1)
        ts = H.host_rt(g1, rows, n=a.n_rt)
        c1 = stop()
        info1 = getattr(g1, "info", lambda: None)()
        del g1
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gB = path_one(a.net, a.engine, dev, a.plans, BATCH[a.net], model)
        stop = gpu_watch()
        am = H.amortized(gB, rows, secs=a.secs)
        cB = stop()
        g1 = types.SimpleNamespace(info=lambda: info1)
    else:
        g1, gB, sha = paths(a.net, a.engine, dev, a.plans, a.ckpt)
        build_s = time.time() - t0
        det, _ = shots(a.net, P_TIME[a.net], range(4))
        rows = torch.as_tensor(det) if a.net == "bb" else det
        stop = gpu_watch()
        gu = H.gpu_us(g1)
        ts = H.host_rt(g1, rows, n=a.n_rt)
        c1 = stop()
        stop = gpu_watch()
        am = H.amortized(gB, rows, secs=a.secs)
        cB = stop()
    q = {"mean": float(ts.mean()), "p50": float(np.quantile(ts, 0.5)), "p99": float(np.quantile(ts, 0.99)),
         "p999": float(np.quantile(ts, 0.999)), "max": float(ts.max()), "n": int(len(ts))}
    info = {k: getattr(g, "info", lambda: None)() for k, g in (("b1", g1), ("bB", gB))}
    tc.emit(a.out, {"role": "alone", "net": a.net, "engine": a.engine, "rep": a.rep, "ckpt_sha256": sha, "seq": a.seq,
                    "build_s": round(build_s, 1), "gpu_us_b1": gu, "host_rt_b1": q, "clock_b1": c1,
                    "batch": BATCH[a.net], "amortized": am, "per_round_h2h": am["h2h"] / ROUNDS[a.net],
                    "clock_bB": cB, "trt": info, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "shots": {"kind": KIND, "p": P_TIME[a.net], "chunks": [0, 4]}})


if __name__ == "__main__":
    main()
