"""The GPU term (a) of the composite: one fresh process timing the deployed engine's GPU time per shot at batch 1 on
every validation shot of a distance (chunks 0-19, 200,000 shots, in order). The studies counted the slowest of five
fresh processes per network, run interleaved across the networks, on one H100 80GB HBM3 (SXM) in one session (study 4,
d = 17: its own session on one single-GPU H100 SXM pod).

Protocol (production's timing loop, below unchanged): K = 100 CUDA graphs of the engine's call, graph k captured on
input slot k of a static (K, m) buffer on the GPU (3 eager calls first); block b loads shots 100 b .. 100 b + 99 into the
slots (outside the timed region, synchronised), then replays graphs 0 .. 99 back to back between two CUDA events: the
block's per-shot mean. Two warm-up blocks, then 2,000 timed blocks; the process's value is the mean over all 200,000
shots. The input is already on the GPU (a GPU-resident-input time, not an end-to-end latency). Beside it in the same
process: the tails of 5 x 100 back-to-back replays on one static input, the 2,000 block means, and nvidia-smi every
100 ms on CPUs other than the timed core. Acceptance in the studies: no other compute process on the GPU before or
after, every SM-clock sample at least 97 % of the maximum, throttle reasons none or idle (0x1); at most two replacement
processes per network.

Differences from production: the engine is built from the released weights (convlines.net.deploy_tables on the CPU,
fp32, then moved to the GPU; equal to the deployed artifact's tables except the folded stem table, within 1-2 float32
ulps from the CPU convolution's summation order) instead of loaded from the artifact file (at d = 17 the release holds
the deployed tables themselves, run with 4 pipeline stages as study 4 ran them), and the outputs are compared bit for
bit with the deployed logits that `python -m timing.deployed` computed once on the same shots (--deployed; the first
2,000 shots at d = 9-17, every shot at d = 7) instead of with the development record. Without --deployed the outputs
are not checked and the process is not accepted.

d = 7 ran in the studies as a hand-written resident CUDA kernel with a compiled router (timing/d7, built by
timing/d7/build.sh), not this engine, and has no graph to replay: its (a) is the device-side time per request from the
kernel's detection of the request to its publication of the result, read from the kernel's own globaltimer stamps
(production's loop, below unchanged: passes of 1,000 requests whose last 63 are the targets, every validation shot a
target once).

--handoff measures the handoff (c) instead of (a), as the studies did:
  d = 7     the host round trip with the stamps off minus the device time with the stamps on, in one process, on
            validation chunks 0-1, no routing (tau < 0); the registered (c) is the largest of five accepted processes.
  d = 9-17  the estimate of the studies' router process: the compiled router (timing/d7/router.cc, built by
            timing/d7/build.sh) drives the engine as the deployed decoder over all 200,000 validation shots at the
            network's tau. Per shot the bits are copied into the graph's static input (untimed), then t0, the launch of
            one CUDA graph of the engine's call and a tail kernel that writes (launch counter << 32 | the logit's bits)
            into mapped host memory, the host's spin on that word, the branch |double(c)| < tau, a routed shot's
            correlated PyMatching (compiled into the library; t1, t2 around it), t3; 200 untimed warm-up shots first.
            Then the engine's device-only time at batch 1 (one graph of its call; every 10th of the first 10,000 shots,
            per shot the median of 5 single replays between CUDA events). The estimate is
                mean(t3 - t0) - mean device-only time - routed fraction x mean(t2 - t1) over the routed shots,
            the part of the complete path that is neither the engine's GPU work nor correlated PyMatching; the
            registered (c) is max(1.1 us, the estimate) from one such process on the session host. Its routed set
            and decisions are checked against the deployed logits (--deployed) and the PyMatching wheel. Last, the
            loop's floor: the same loop around a graph without the engine (20,000 shots), one of the host gate's
            measurements (README, "Qualifying a timing session").

Acceptance, every process ((a) and (c)): with --deployed, the outputs equal the deployed logits; no other compute
process on the GPU before or after; every clock window (both phases of a handoff process) with every SM-clock sample at
least 97 % of the maximum and throttle reasons none or idle; at d = 7's (a), every device interval shorter than its host
round trip.

    python -m timing.gpu_term --d 9 --val DIR --proc 0 --out OUT --deployed MASKS/val_d9.npz [--core C] [--limit N]
    python -m timing.gpu_term --d 7 --val DIR --proc 0 --out OUT --deployed MASKS/val_d7.npz [--build build/d7]
    python -m timing.gpu_term --d 15 --val DIR --proc 0 --out OUT --deployed MASKS/val_d15.npz --handoff
    python -m timing.gpu_term --d 11 --net d11_b5_sv2 --val DIR --proc 0 --out OUT --deployed MASKS/val_d11_b5_sv2.npz
(DIR: the validation chunks of `python -m convlines.testset val --d 9 --chunks 0-19 --out DIR`; MASKS: the output of
`python -m timing.deployed`.)"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

K = 100                    # graphs per block (one per input slot)


# ---- production helpers (timing harness), unchanged --------------------------------------------------------------------
def smi_apps():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], capture_output=True,
                       text=True)
    return [x for x in r.stdout.splitlines() if x.strip()]


def pid_of(line):
    return line.split(",")[0].strip()


def own_pids(apps0):
    """The pids under which nvidia-smi lists this process: its own and, where a container reports the host's pids
    (SURF-v2's E0 pod, 12 October, listed '<host pid>, [Not Found]'), the one line that appeared once this process took
    its CUDA context; apps0 is the list from before. Two new lines identify nothing, and the process is then not alone."""
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    new = []
    for _ in range(10):
        new = [x for x in smi_apps() if x not in apps0]
        if new:
            break
        time.sleep(0.2)
    return sorted({str(os.getpid())} | ({pid_of(new[0])} if len(new) == 1 else set()))


def cpus(text):
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def off_cores(core):
    """Every online CPU but the timed core and its hyperthread siblings (the kernel keeps those the cpuset allows)."""
    sib = cpus(open(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list").read()) | {core}
    return ",".join(str(c) for c in sorted(cpus(open("/sys/devices/system/cpu/online").read()) - sib))


class Clocks:
    """nvidia-smi every 100 ms. where "off": on CPUs other than the timed core and its siblings, started before the
    warm-up, returning after its first sample (NVML up); "timed": the child inherits the timed core and starts just
    before the timed part; "none": no sampler (diagnosis only)."""
    def __init__(self, core, where="off"):
        self.where, self.p, self.first = where, None, ""
        if where == "none":
            return
        q = "timestamp,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,power.draw,temperature.gpu"
        cmd = ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms", "100"]
        if where == "off":
            cmd = ["taskset", "-c", off_cores(core)] + cmd
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        if where == "off":
            self.first = self.p.stdout.readline()

    def stop(self):
        if self.p is None:
            return dict(samples=0, where=self.where)
        self.p.terminate()
        out = self.first + self.p.communicate(timeout=10)[0]
        rows = []
        for line in out.splitlines():
            f = [x.strip() for x in line.split(",")]
            try:
                rows.append(dict(sm=float(f[1]), sm_max=float(f[2]), throttle=f[3], power=float(f[4]), temp=float(f[5])))
            except (ValueError, IndexError):
                pass
        sm = [r["sm"] for r in rows]
        mx = max((r["sm_max"] for r in rows), default=None)
        return dict(where=self.where, samples=len(rows), sm_min=min(sm, default=None), sm_median=float(np.median(sm)) if sm else None,
                    sm_max_clock=mx, sm_min_frac=(min(sm) / mx) if (sm and mx) else None,
                    throttle_seen=sorted({r["throttle"] for r in rows}),
                    power_max=max((r["power"] for r in rows), default=None))


def stats_us(v):
    v = np.asarray(v, np.float64)
    return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.median(v)), p99=float(np.percentile(v, 99)),
                max=float(v.max()), min=float(v.min()))


def tails(v):
    v = np.asarray(v, np.float64)
    return dict(stats_us(v), p999=float(np.percentile(v, 99.9)), std=float(v.std()))


# ---- production's timing loop; the engine is passed in, the first 2,000 outputs are returned ---------------------------
def frozen(eng, pool, core, where, spans):
    n, m = pool.shape
    X = torch.empty((K, m), dtype=torch.uint8, device="cuda")
    X.copy_(pool[:K])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            eng(X[0:1])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graphs, outs = [], []
    for k in range(K):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs.append(eng(X[k:k + 1])[0])
        graphs.append(g)
    torch.cuda.synchronize()
    clk = Clocks(core, where)
    # replay tails: shot 0 in slot 0, graph 0 alone
    X.copy_(pool[:K])
    torch.cuda.synchronize()
    for _ in range(200):
        graphs[0].replay()
    torch.cuda.synchronize()
    reps = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(100):
            graphs[0].replay()
        e1.record()
        torch.cuda.synchronize()
        reps.append(e0.elapsed_time(e1) / 100 * 1e3)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2001)]
    ev[0].record()
    for i in range(2000):
        graphs[0].replay()
        ev[i + 1].record()
    torch.cuda.synchronize()
    replay_calls = [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(2000)]
    # the counted pass: every shot in val order, blocks of K back-to-back replays of K graphs
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for b in range(2):                                             # warm-up: shots 0-199
        X.copy_(pool[b * K:(b + 1) * K])
        torch.cuda.synchronize()
        for g in graphs:
            g.replay()
    torch.cuda.synchronize()
    nb = n // K
    blk = np.empty(nb)
    for b in range(nb):
        X.copy_(pool[b * K:(b + 1) * K])
        torch.cuda.synchronize()
        e0.record()
        for g in graphs:
            g.replay()
        e1.record()
        e1.synchronize()
        blk[b] = e0.elapsed_time(e1) * 1e3 / K
    shot_spans = None
    if spans:                                                      # per-shot spans (events between replays), all shots
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(K + 1)]
        shot_spans = np.empty(nb * K)
        for b in range(nb):
            X.copy_(pool[b * K:(b + 1) * K])
            torch.cuda.synchronize()
            ev[0].record()
            for k, g in enumerate(graphs):
                g.replay()
                ev[k + 1].record()
            ev[K].synchronize()
            shot_spans[b * K:(b + 1) * K] = [ev[k].elapsed_time(ev[k + 1]) * 1e3 for k in range(K)]
    c = clk.stop()
    # the outputs: the first 2,000 shots through the same graphs (untimed, one shot at a time)
    nv = min(2000, nb * K)
    got = np.empty(nv, np.float32)
    for b in range(nv // K):
        X.copy_(pool[b * K:(b + 1) * K])
        for k, g in enumerate(graphs):
            g.replay()
            got[b * K + k] = outs[k].float().item()
    res = dict(per_shot_mean=float(blk.mean()), shots=int(nb * K), blocks=int(nb), graphs=K,
               workload_block_means=tails(blk), replay_static=dict(us_call_reps=reps, per_call=tails(replay_calls)),
               shot_spans=tails(shot_spans) if spans else None, clocks=c, engine_cfg=eng.cfg)
    return res, got


# ---- d = 7: production's loops for the resident kernel (timing/d7), the engine passed in -------------------------------
PASS, TARGETS = 1000, 63   # requests per pass, target requests at its end (the ring holds 64, the stop takes one)


def resident(eng, lib, det_all, core, where):
    """(a) at d = 7, stamps on: per request the time from CTA 0's detection of the request (stamp 0) to its publication
    of the result (stamp 14), from the kernel's globaltimer stamp ring. The ring keeps 64 requests, so pass j serves
    1,000 requests ending with the target shots 63 j .. 63 j + 62 (the 937 shots before them in validation order
    first, cyclically) and reads the targets' stamps after the stop; every shot is a target exactly once. Per target
    the host round trip of the same request (packing through the result seen) is kept beside it: the device interval
    must be shorter than the host round trip. No routing (tau < 0)."""
    from timing.d7.engine import NDBG
    n = len(det_all)
    dev, host = np.full(n, np.nan), np.full(n, np.nan)
    z = np.full(n, np.nan, np.float32)
    stop_slot = PASS % NDBG
    clk = Clocks(core, where)
    eng.start()                                                    # one untimed pass first
    lib.run_persistent(eng.pers, np.ascontiguousarray(det_all[:PASS]), -1.0, None, PASS, warm=0)
    eng.stop()
    npass = -(-n // TARGETS)
    starts = []
    for j in range(npass):
        tgt = np.arange(j * TARGETS, min((j + 1) * TARGETS, n))
        idx = np.concatenate([np.arange(tgt[0] - (PASS - len(tgt)), tgt[0]) % n, tgt])
        det = np.ascontiguousarray(det_all[idx])
        st0 = eng.start()
        o = lib.run_persistent(eng.pers, det, -1.0, None, PASS, warm=0)
        sp = eng.stop()
        assert sp["served"] == PASS, sp
        st = eng.stamps().astype(np.int64)
        r = np.arange(PASS - len(tgt), PASS)                       # the targets' request indices in the pass
        sl = r % NDBG
        assert stop_slot not in set(sl.tolist())
        d0, d14 = st[sl, 0, 0], st[sl, 0, 14]
        assert (d0 != 0).all() and (d14 != 0).all() and (d14 > d0).all(), j
        dev[tgt] = (d14 - d0) / 1e3
        host[tgt] = o["t_all"][r] / 1e3
        z[tgt] = o["z"][r]
        starts.append(st0["launch_to_ready_ns"])
    c = clk.stop()
    assert not np.isnan(dev).any()
    ho = host - dev
    res = dict(per_shot_mean=float(dev.mean()), shots=int(n), passes=int(npass), requests_per_pass=PASS,
               targets_per_pass=TARGETS, device_detect_to_publish=tails(dev), host_round_trip_stamps_on=tails(host),
               handoff_per_request=tails(ho), device_within_host=int((dev < host).sum()), clocks=c,
               launch_to_ready_us=stats_us(np.asarray(starts) / 1e3), engine_cfg=eng.cfg)
    return res, z


def handoff(t, lib, ctx, cubin, det, core, where):
    """(c) at d = 7: the device time per request with stamps on (the last 63 requests of each of 25 passes of 2,000,
    the stop request's slot dropped) and, stamps off, the host round trip of 20,000 requests (packing through the
    result seen) in the same process; the handoff is the mean round trip minus the mean device time. No routing."""
    from timing.d7.engine import Engine
    eng = Engine(t, lib, ctx, cubin, body=1, fwd=1, dbg=True)
    clk = Clocks(core, where)
    body, dropped = [], []
    for _ in range(25):                                            # stamps on: detection (0) -> publication (14), CTA 0
        eng.start()
        lib.run_persistent(eng.pers, det, -1.0, None, 2000, warm=0)
        sp = eng.stop()
        assert sp["served"] == 2000, sp
        st = eng.stamps().astype(np.int64)
        ok = (st[:, 0, 0] != 0) & (st[:, 0, 14] != 0)
        ok[sp["served"] % len(st)] = False                         # the stop request's slot: its detection, no publication
        dropped.append(sp["served"] % len(st))
        body += list((st[ok, 0, 14] - st[ok, 0, 0]) / 1e3)
    c_on = clk.stop()
    cfg = eng.cfg
    del eng
    eng = Engine(t, lib, ctx, cubin, body=1, fwd=1)                # stamps off: the host round trip, same process
    clk = Clocks(core, where)
    eng.start()
    o = lib.run_persistent(eng.pers, det, -1.0, None, 20000, warm=1000)
    sp = eng.stop()
    c_off = clk.stop()
    rt = o["t_all"] / 1e3
    b = stats_us(body)
    r = stats_us(rt)
    return dict(device_detect_to_publish=b, round_trip=r, handoff_us=r["mean"] - b["mean"], clocks=c_on,
                clocks_round_trip=c_off, engine_cfg=cfg, stop=sp, stamp_slots_dropped=sorted(set(dropped)))


# ---- d = 9-17: the studies' router process (its mapped mode, the device-only time and the handoff estimate) -----------
def heat(g, n=200, seconds=1.0):
    """Replay graph g at least n times and for at least `seconds` (clock ramp-up before timing)."""
    t0, i = time.perf_counter(), 0
    while i < n or time.perf_counter() - t0 < seconds:
        for _ in range(50):
            g.replay()
        i += 50
        torch.cuda.synchronize()


def graph_time_shots(fn, pool, idx, reps=10, warm_s=1.0):
    """Per-shot batch-1 time over actual shots: one CUDA graph on a static (1, m) input buffer; for each shot of idx the
    shot is copied into the buffer (untimed), then reps single replays are timed with CUDA events and the shot's time is
    their median. Device-only. -> the per-shot µs array."""
    x = pool[idx[0]:idx[0] + 1].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        fn(x)
    heat(g, 200, warm_s)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        x.copy_(pool[i:i + 1])
        for e0, e1 in ev:
            e0.record()
            g.replay()
            e1.record()
        torch.cuda.synchronize()
        out[k] = np.median([e0.elapsed_time(e1) for e0, e1 in ev]) * 1e3
    return out


def capture_tail(f, x, tail):
    """Three eager calls of f(x) on a side stream, then one CUDA graph of f(x) followed by the tail kernel (mapped
    completion)."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), torch.inference_mode():
        for _ in range(3):
            f(x)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        out = f(x)
        assert out.dtype == torch.float32 and out.numel() == 1 and out.is_contiguous(), (out.dtype, out.shape)
        tail(torch.cuda.current_stream().cuda_stream, out.data_ptr())
    torch.cuda.synchronize()
    return g, out


def router_handoff(eng, lib, ctx, det, tau, d, core, where):
    """(c) at d = 9-17: one pass of the compiled loop over every shot of det (200 warm-up shots first), then the
    device-only time on every 10th of the first 10,000 shots; the estimate = complete-path mean - device-only mean -
    routed fraction x the routed shots' mean correlated-PM time. -> (record, the loop's per-shot outputs)."""
    from convlines import circuits
    f = lambda x: eng(x)[0]                                        # noqa: E731  the logit c, shape (1,)
    pool = torch.from_numpy(det).cuda()
    x = pool[:1].clone()
    c = circuits.circuit(d)
    pm = lib.pm(str(circuits.dem(c)), c.num_detectors)
    g, out = capture_tail(f, x, lambda st, ptr: lib.tail(ctx, st, ptr))   # out: kept alive with its graph
    clk = Clocks(core, where)
    heat(g, 200, 1.0)
    o = lib.run(ctx, g, x, pool, det, tau, pm, len(det), warm=200)
    c_pass = clk.stop()
    clk = Clocks(core, where)
    dev = graph_time_shots(f, pool, list(range(0, min(len(det), 10000), 10)), reps=5)
    c_dev = clk.stop()
    zero = torch.zeros(1, dtype=torch.float32, device="cuda")      # the floor: the same loop around a graph with no
    g0, _ = capture_tail(lambda xx: zero, x, lambda st, ptr: lib.tail(ctx, st, ptr))   # engine (the host gate's)
    heat(g0, 200, 1.0)
    floor = lib.run(ctx, g0, x, pool, det, -1.0, None, min(len(det), 20000), warm=200)["t_all"] / 1e3
    t_all, t_pm, routed = o["t_all"] / 1e3, o["t_pm"] / 1e3, o["routed"].astype(bool)
    routed_pm = float(t_pm[routed].mean()) if routed.any() else 0.0
    est = float(t_all.mean() - dev.mean() - routed.mean() * routed_pm)
    return dict(complete_path=stats_us(t_all), device_only_b1=stats_us(dev), routed=int(routed.sum()),
                routed_fraction=float(routed.mean()), routed_pm_mean_us=routed_pm, handoff_estimate_us=est,
                handoff_us=max(1.1, est), handoff_rule="max(1.1 us allowance, the estimate)", floor_mapped=stats_us(floor),
                clocks=c_pass, clocks_device_only=c_dev, engine_cfg=eng.cfg), o


def main():
    from convlines import circuits, decoder, testset
    from timing import deployed
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True, choices=sorted(decoder.NETWORKS))
    ap.add_argument("--val", required=True, help="directory with the validation chunks 0-19")
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--core", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spans", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="tests only: the first N shots")
    ap.add_argument("--clocks", default="off", choices=("off", "timed", "none"))
    ap.add_argument("--build", default="build/d7", help="timing/d7/build.sh's output directory (d = 7; --handoff)")
    ap.add_argument("--handoff", action="store_true", help="the handoff (c) instead of (a)")
    ap.add_argument("--deployed", default="", help="timing.deployed's file of this d: the outputs' bitwise check")
    ap.add_argument("--net", default="", help="another network of d (weights/configs_sv2.json; the handoff at its tau)")
    a = ap.parse_args()
    core = a.core if a.core >= 0 else max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {core})
    torch.set_num_threads(1)
    name = a.net or decoder.NETWORKS[a.d]
    assert decoder.config(name)["d"] == a.d, f"{name} is not a network of d = {a.d}"
    apps0 = smi_apps()
    from convlines import provenance
    res = dict(network=name, network_sha256=decoder.weights_sha256(name), code_digest=provenance.code_digest(),
               d=a.d, proc=a.proc, core=core, pid=os.getpid(), gpu=torch.cuda.get_device_name(),
               torch=torch.__version__, clocks_where=a.clocks, limit=a.limit or None, handoff=a.handoff,
               load_start=open("/proc/loadavg").read().split()[:3], apps_before=apps0,
               start=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    res["own_pids"] = own_pids(apps0)
    sh = [testset.load(os.path.join(a.val, testset.name("val", a.d, j)), a.d) for j in range(20)]
    det, obs = np.concatenate([s[0] for s in sh]), np.concatenate([s[1] for s in sh])
    if a.deployed:
        z = deployed.load(a.deployed, a.d, det, obs, kind="deployed")
        assert str(z.get("network", decoder.NETWORKS[a.d])) == name, f"{a.deployed} holds another network's logits"
    want = z["c"] if a.deployed else None
    if a.limit:
        det = det[:a.limit]
    det = np.ascontiguousarray(det)
    t, cfg = deployed.tables(a.d, name)                            # on the CPU in fp32, as the deployed artifact
    to_dev = lambda x: x.cuda() if torch.is_tensor(x) else ({k: to_dev(v) for k, v in x.items()} if isinstance(x, dict)  # noqa: E731
                                                          else [to_dev(v) for v in x] if isinstance(x, list) else x)
    tag, got = f"p{a.proc}", None
    if a.d == 7 or a.handoff:                                      # the compiled router (and d = 7's resident kernel)
        from timing.d7.engine import Engine, Lib
        lib = Lib(os.path.join(a.build, "libs7router.so"))
        torch.zeros(1, device="cuda")                              # the primary context, current on this thread
        ctx = lib.ctx()
        cubin = os.path.join(a.build, "kernel_sm90a.cubin")
    if a.handoff and a.d == 7:
        tag = f"handoff_p{a.proc}"
        res.update(handoff(to_dev(t), lib, ctx, cubin, det[:20000], core, a.clocks),
                   shots_set="validation chunks 0-1 of d = 7")
    elif a.handoff:
        tag = f"handoff_p{a.proc}"
        r, o = router_handoff(deployed.engine(to_dev(t), cfg), lib, ctx, det, cfg["policy"]["tau"], a.d, core, a.clocks)
        got = o["z"]
        routed = o["routed"].astype(bool)
        wheel = circuits.pm_decode(circuits.matcher(a.d, True), det[routed], correlated=True)
        res.update(r, tau=cfg["policy"]["tau"],
                   routed_is_tau=bool(np.array_equal(routed, np.abs(got.astype(np.float64)) < cfg["policy"]["tau"])),
                   unrouted_sign=bool(np.array_equal(o["dec"][~routed], (got[~routed] > 0).astype(np.uint8))),
                   routed_equal_wheel=bool(np.array_equal(o["dec"][routed], wheel)))
    elif a.d == 7:
        eng = Engine(to_dev(t), lib, ctx, cubin, body=1, fwd=1, dbg=True)
        r, got = resident(eng, lib, det, core, a.clocks)
        res.update(r)
    else:
        eng = deployed.engine(to_dev(t), cfg)
        r, got = frozen(eng, torch.from_numpy(det).cuda(), core, a.clocks, a.spans)
        res.update(r)
    if not a.handoff or a.d != 7:
        res.update(shots_set=f"validation chunks 0-19 of d = {a.d} in order" + (f" (first {a.limit})" if a.limit else ""),
                   outputs_checked=len(got) if want is not None else 0,
                   outputs_equal_deployed=bool(np.array_equal(got.view(np.uint32), want[:len(got)].view(np.uint32)))
                   if want is not None else None)
    res.update(load_end=open("/proc/loadavg").read().split()[:3], end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    torch.cuda.synchronize()
    res["apps_after"] = smi_apps()
    ok = lambda c: (c.get("sm_min_frac") is not None and c["sm_min_frac"] >= 0.97 and  # noqa: E731
                    set(c["throttle_seen"]) <= {"0x0000000000000000", "0x0000000000000001"})
    chk = dict(clocks=all(ok(res[k]) for k in ("clocks", "clocks_round_trip", "clocks_device_only") if k in res),
               gpu_alone=all(pid_of(x) in res["own_pids"] for x in apps0 + res["apps_after"]))
    if "outputs_equal_deployed" in res:
        chk["outputs_equal_deployed"] = res["outputs_equal_deployed"] is True
    if a.d == 7 and not a.handoff:
        chk["device_within_host"] = res["device_within_host"] == res["shots"]
    if a.handoff and a.d != 7:
        chk["routing"] = res["routed_is_tau"] and res["unrouted_sign"] and res["routed_equal_wheel"]
    res.update(acceptance=chk, accepted=all(chk.values()))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"{name}_{tag}.json"), "w"), indent=1, default=float)
    if a.handoff and a.d == 7:
        what = (f"handoff {res['handoff_us']:.3f} us (round trip {res['round_trip']['mean']:.3f}, device "
                f"{res['device_detect_to_publish']['mean']:.3f})")
    elif a.handoff:
        what = (f"handoff estimate {res['handoff_estimate_us']:.3f} us (complete path {res['complete_path']['mean']:.3f}, "
                f"device only {res['device_only_b1']['mean']:.3f}, routed {res['routed']} x "
                f"{res['routed_pm_mean_us']:.1f}); counted max(1.1, estimate) = {res['handoff_us']:.3f}")
    else:
        what = (f"per-shot mean {res['per_shot_mean']:.3f} us over {res['shots']} val shots")
    print(f"{name} {tag}: {what}; outputs equal to the deployed logits {res.get('outputs_equal_deployed')}; checks "
          f"{chk}; accepted {res['accepted']}", flush=True)


if __name__ == "__main__":
    main()
