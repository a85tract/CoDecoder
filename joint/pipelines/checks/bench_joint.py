"""The joint step's added time (smoke, not registered), on the pod's card: Cascade's frozen paths with and without the
joint step, the same inputs.

  python podrun.py bench_joint.py bb  --p 0.003 --w W --out F.json      (BB72: C14' compiled@1 and compiled@4096)
  python podrun13.py bench_joint.py d13 --p 0.006 --w W --lock DRYLOCK --out F.json   (d = 13: CS13 compiled@1, @16)

Single (batch 1), per routed shot, host-measured as the pipelines measure it: pinned row -> device, replay, decision and
confidence -> pinned host, synchronise; the plain graph (pc_speed.build / sv_common.Graph) against the graph with the
joint step (joint.JointB / sv_joint.JointB13), on rows where S's decision agrees with Cascade's and on rows where it
differs (S's buffers set so), N calls each after 200 warm-up, interleaved in blocks. Batched: CUDA-event time of one
replay at the batch (4,096 / b* = 16) with a fraction of differing rows (0, the given --frac, 1)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def quant(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "p50": float(np.quantile(x, 0.5)), "p99": float(np.quantile(x, 0.99)),
            "p999": float(np.quantile(x, 0.999))}


def host_calls(g, x_h, outs, n, warm=200):
    """n round trips of graph g on the pinned row x_h: (copy in, replay, outputs to pinned host, synchronise) -> us."""
    st = torch.cuda.current_stream()
    hb = [o.new_empty(o.shape, device="cpu").pin_memory() for o in outs]
    t = np.empty(n)
    for i in range(-warm, n):
        t0 = time.perf_counter()
        g.x.copy_(x_h, non_blocking=True)
        g.replay()
        for h, o in zip(hb, outs):
            h.copy_(o, non_blocking=True)
        st.synchronize()
        if i >= 0:
            t[i] = 1e6 * (time.perf_counter() - t0)
    return t


def event_ms(g, reps=200):
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / reps


def interleave(fns, n, blocks=10):
    """Each fn(n_block) -> us array; run in alternating blocks -> concatenated per fn."""
    out = [[] for _ in fns]
    for _ in range(blocks):
        for k, f in enumerate(fns):
            out[k].append(f(n // blocks))
    return [np.concatenate(o) for o in out]


def bb(a):
    import d1_common as d1
    import joint
    import pc_common as pc
    import pc_eval  # noqa: F401  (ce.Predictor: the ConvLines networks on the fused engine, .fused)
    import pc_speed as ps
    from bb import cascade_data as cdata
    from bb.cascade_eval import Predictor
    dev, env = ps.setup()
    L = d1.lock()
    pr = Predictor(pc.NETS[d1.B][0], dev, "ema", "bf16")
    S = Predictor(pc.NETS[d1.S][0], dev, "ema", "bf16")
    det, _ = cdata.eval_shots(pc.CODE, a.p, range(1), kind=pc.KIND_DRY, procs=2, family=d1.FAMILY)
    eng = S.fused                                                       # pc_fused.Batched: s rows of real shots
    rows = torch.as_tensor(det[:4096], device=dev)
    eng(rows.contiguous())
    s_all = eng.s[:4096].clone()
    rec = {"p": a.p, "w": a.w, "gpu": env["gpu"], "joint": joint.record(a.w)}
    # single: compiled@1
    g0 = ps.build(pr, "compiled", 1)
    s1 = torch.zeros(1, eng.QP, device=dev)
    sb1 = torch.zeros(1, 12, dtype=torch.uint8, device=dev)
    gj = joint.build_b(pr, "compiled", 1, eng, a.w, s_src=s1, sb_src=sb1)
    x_h = torch.as_tensor(det[:1]).pin_memory()
    gj.x.copy_(x_h.to(dev))
    gj.replay()
    torch.cuda.synchronize()
    s1.copy_(s_all[:1])
    agree = gj.bb.clone()
    differ = 1 - agree

    def plain(n):
        return host_calls(g0, x_h, (g0.bits, g0.conf), n)

    def jt(sb):
        def f(n):
            sb1.copy_(sb)
            return host_calls(gj, x_h, (gj.bits, gj.conf), n)
        return f
    t0_, ta, td = interleave([plain, jt(agree), jt(differ)], a.n)
    rec["single"] = {"plain_us": quant(t0_), "joint_agree_us": quant(ta), "joint_differ_us": quant(td),
                     "added_agree_us": float(ta.mean() - t0_.mean()), "added_differ_us": float(td.mean() - t0_.mean()),
                     "gpu_ms": {"plain": event_ms(g0), "joint": event_ms(gj)}}
    sb1.copy_(differ)
    gj.replay()
    torch.cuda.synchronize()
    rec["single"]["differ_flag"] = int(gj.flag[0])
    # batched: compiled@4096
    B = pc.VERDICT_B
    g4 = ps.build(pr, "compiled", B)
    gj4 = joint.build_b(pr, "compiled", B, eng, a.w)
    for g in (g4, gj4):
        g.x.copy_(rows)
    gj4.s.copy_(s_all)
    gj4.replay()
    torch.cuda.synchronize()
    bbits = gj4.bb.clone()
    res = {"plain_ms": event_ms(g4)}
    for f in (0.0, a.frac, 1.0):
        m = torch.rand(B, device=dev, generator=torch.Generator(device=dev).manual_seed(1)) < f
        gj4.sb.copy_(torch.where(m[:, None], 1 - bbits, bbits))
        res[f"joint_ms_frac{f:g}"] = event_ms(gj4)
        res[f"rows_differ_frac{f:g}"] = int(m.sum())
    res["added_us_per_routed_shot"] = {k: 1e3 * (v - res["plain_ms"]) / B for k, v in res.items()
                                       if k.startswith("joint_ms")}
    rec["batched"] = res
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec), flush=True)


def d13(a):
    from surface import sv_common as sc
    from surface import sv_joint
    lock = json.load(open(a.lock))
    dev, gpu = sc.setup_gpu()
    bmod, _ = sc.load_B(dev)
    fz = lock["frozen"]["B"]
    Dd = sc.sl.Layout(sc.D).D
    det, obs = sc.sdata.eval_shots(sc.D, a.p, range(1), n=sc.CHUNK, kind=sc.KIND_DRY, procs=1)
    rec = {"p": a.p, "w": a.w, "gpu": gpu["gpu"], "joint": sv_joint.record(a.w)}
    p1, b1 = fz["single"].split("@")
    g0 = sc.Graph(sc.b_fn(bmod, p1), int(b1), Dd, dev)
    c1 = torch.zeros(1, device=dev)
    d1_ = torch.zeros(1, dtype=torch.uint8, device=dev)
    gj = sv_joint.JointB13(sc.b_fn(bmod, p1), int(b1), Dd, dev, a.w, c_src=c1, ds_src=d1_)
    x_h = torch.as_tensor(det[:1]).pin_memory()
    gj.x.copy_(x_h.to(dev))
    gj.replay()
    torch.cuda.synchronize()
    agree = gj.bdec.clone()
    differ = 1 - agree
    c1.fill_(0.5)

    def plain(n):
        return host_calls(g0, x_h, (g0.logit, g0.dec), n)

    def jt(ds):
        def f(n):
            d1_.copy_(ds)
            return host_calls(gj, x_h, (gj.logit, gj.dec), n)
        return f
    t0_, ta, td = interleave([plain, jt(agree), jt(differ)], a.n)
    rec["single"] = {"plain_us": quant(t0_), "joint_agree_us": quant(ta), "joint_differ_us": quant(td),
                     "added_agree_us": float(ta.mean() - t0_.mean()), "added_differ_us": float(td.mean() - t0_.mean()),
                     "gpu_ms": {"plain": event_ms(g0), "joint": event_ms(gj)}}
    pb, bs = fz["amortized"].split("@")
    gb = sc.Graph(sc.b_fn(bmod, pb), int(bs), Dd, dev)
    gjb = sv_joint.JointB13(sc.b_fn(bmod, pb), int(bs), Dd, dev, a.w)
    rows = torch.as_tensor(det[:int(bs)], device=dev)
    for g in (gb, gjb):
        g.x.copy_(rows)
    res = {"batch": int(bs), "plain_ms": event_ms(gb), "joint_ms": event_ms(gjb)}
    res["added_us_per_routed_shot"] = 1e3 * (res["joint_ms"] - res["plain_ms"]) / int(bs)
    rec["batched"] = res
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("bb", "d13"))
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--lock", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    {"bb": bb, "d13": d13}[a.what](a)
