"""Phase C speed (PHASE_C_PLAN.md section 5): one protocol for every network and every routed pipeline, on one GPU model.

  nets       python pc_speed.py nets --net TAG --rep I --out rows.jsonl
  summarize  python pc_speed.py summarize --rows rows.jsonl --out speed_components.json
  check      python pc_speed.py check --read DIR [--eager DIR_E] --locked locked_<setting>.json \
                 --speed speed_components.json --out check_<setting>.json
  pipeline   python pc_speed.py pipeline --locked locked_<setting>.json --config I --mode single|batched --rep I \
                 --n N --speed speed_components.json --out rows.jsonl      (--config -1 --p P: Cascade alone)

nets: one network in one fresh process (the job runs every network REPS = 3 times, interleaved). Each network is
compared with its accuracy path, the computation of its reads (pc_eval.py): for a Cascade network bb.cascade_eval's
evaluation backend (--prec bf16: the module under bf16 autocast; the bipartite convolution and scatter in bf16 Triton
kernels), for a ConvLines network the fused engine (pc_fused.Batched over FUSED_READ_B shots per call). Paths: "graph"
(the evaluation backend captured in a CUDA graph at each batch size; for ConvLines the vote and margin in fp32),
"compiled" (torch.compile, default mode, then a CUDA graph; batch 1 and the verdict's batch) and, for ConvLines, "fused"
(fp16 matmul inputs with fp32 accumulation: at batch 1 bb.engine's registered engine, at the larger batch sizes
pc_fused.Batched in a CUDA graph). A ConvLines network's frozen paths are fused paths (PHASE_C_PLAN.md 2.4: its
accuracy and its speed are one computation); its other paths are timed for description. With --quick (dry
runs) a Cascade network is timed on its graph path only and a ConvLines network on its fused paths only. Per path and
batch size B:
  gpu_us     B = 1: SETTINGS 5.1's replay protocol (200 warm-up replays, 5 x 100 timed replays, CUDA events): the mean
             time of a call with the input already on the GPU
  host_rt    B = 1: 10,000 calls of the full round trip (pinned host -> device copy of one shot, replay, device ->
             pinned host copy of the decision and the confidence, synchronise), host wall time per call: mean,
             quantiles 0.5 / 0.99 / 0.999, max
  amortized  every B: wall time over consecutive batches / (batches x B) per shot, input resident on the GPU
             ("dev") and host to host ("h2h": each batch copied in from pinned memory and its outputs copied back);
             per round = per shot / 6 (the paper's amortized latency)
  agree      the screen, against the accuracy path (run chunk by chunk as the reads run it) on the same fresh shots
             (kind 12, the network's own setting): at p = 0.55 % the failures (both failure counts, the shots only one
             of them fails); at each claim point (RELAY_P) the routing: below each quantile in ROUTE_Q of the evaluated
             confidence, the shots each one routes and those only one routes. Each count is judged by
             pc_common.agrees. Recorded beside them: the shots whose decision differs (at 0.55 % and at the claim
             points) and those whose confidence differs at all, and whether the path is bitwise the accuracy path,
             which a ConvLines network's path must be to pass (screened)
The GPU's SM clock is sampled every 100 ms while a path is timed (bb.timing.gpu_watch); every row records the result.

summarize: each network's frozen timed paths (pc_common.frozen_paths), the fastest that pass the screen at their batch
size, from the slowest of all its processes. The lock records the file's sha256; check and pipeline refuse any other
file.

check (step C4b, before any confirmation read; gate finding 1): at the claim points (RELAY_P) of a setting, on a read of
fresh shots (kind 12; bb.cascade_eval through pc_eval.py, saved with the detector rows), every network alone on its
frozen single and amortized paths, and every locked configuration on its frozen single paths (batch 1) and batched paths
(4,096) with the locked thresholds, against the evaluated computation of the same shots (the read; for a routed
configuration its routing by pc_route.masks): the failures and the shots routed to B and to R, each judged by
pc_common.agrees. A ConvLines network's read must also be bitwise its frozen paths, every decision and every margin
(the read's batch size may not change them): a network alone, or a configuration whose S is that network, fails the
check otherwise. Relay-BP decides every shot either version routes to it with a new decoder seeded by the shot's index
(pc_relay), so both versions get the same Relay-BP decision on a shot. With --eager, the depth-rule networks' compiled
read (pc_eval.py) is checked against an eager read of the same rows. A failed check removes the decoder from the
headline (pc_report); nothing falls back to another path.

pipeline: a locked configuration (pc_route) run end to end on fresh shots (kind 12) at its p with real routing
decisions, one fresh process per repetition (the job runs REPS = 3, interleaved; pc_report takes the slowest). single:
one shot at a time (S at batch 1; B at batch 1 on the shots S routes; Relay-BP in this process, single-threaded, on the
shots B routes or S routes in S-R), each network on its frozen single path; per-shot host latency from the shot in
pinned memory to the decision on the host. batched: S over batches of 4,096; routed shots queue on the GPU and B runs
whenever 4,096 are waiting (and once at the end, padded), each network on its frozen batched path; Relay-BP in
NCPU_RELAY worker processes, asynchronously, RELAY_TASK shots per task; the wall time from the first copy to the last
decision over N x 6 is the amortized latency. Relay-BP here is deployed as it would run: one decoder per process, kept
across shots (its generator carries over); the accuracy reads decide the same shots with a new seeded decoder per shot,
whose construction (measured by check) is not part of any timing. With --config -1, Cascade of the setting (the lock's
B) runs alone in the same harness on the stream at --p: single on its frozen single path, batched on its frozen
amortized path; the job times it beside the pipelines, so that pc_report compares each pipeline with a Cascade timed
in the same job(s).

Every subcommand that uses the GPU refuses any card but phase C's (pc_common.GPU, a full RTX PRO 6000) and records the
card, its driver and the venv's torch."""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

import pc_common as pc

P_AGREE = 0.0055
N_AGREE_B1, N_AGREE_B = 20_000, 100_000
COMPILED_B = (1, pc.VERDICT_B)
B_Q = pc.VERDICT_B                                 # the batched pipeline: B's queue at 4,096 (S runs at 4,096 too)
RELAY_TASK = 4                                     # routed shots per Relay-BP task (keeps all workers busy)


def setup():
    """The device, its card refused unless it is phase C's GPU, and the card's record (gpu, driver, torch)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    dev = torch.device("cuda")
    return dev, pc.gpu_env(dev)


def make_fwd(pr, score="minabs"):
    """The evaluation backend's computation on one batch: det (B, 432) uint8 -> (bits (B, 12) uint8, conf (B,) fp32)."""
    import torch

    def fwd(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if pr.kind == "cascade":
                z = pr.model(x).float()
                a = z.abs()
                conf = a.min(1).values if score == "minabs" else -torch.nn.functional.softplus(-a).sum(1)
                return (z > 0).to(torch.uint8), conf
            s = pr.model(x)[0].float()
            _, bits, margin = pr.model.decide(s)
            return bits, margin.float()
    return fwd


class Graph:
    """fn captured in a CUDA graph at batch B on a static input (three warm-up calls on a side stream first, so that
    Triton autotuning and compilation happen outside the capture)."""

    def __init__(self, fn, B, D, dev):
        import torch
        self.B = B
        self.x = torch.zeros(B, D, dtype=torch.uint8, device=dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(st):
            for _ in range(3):
                fn(self.x)
        torch.cuda.current_stream().wait_stream(st)
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g):
            self.bits, self.conf = fn(self.x)

    def replay(self):
        self.g.replay()

    def run(self, det):
        """Decisions on rows det (n, D) on the device, B at a time (the last batch padded)."""
        import torch
        n, out_b, out_c = len(det), [], []
        for i in range(0, n, self.B):
            d = det[i:i + self.B]
            self.x.zero_()
            self.x[:len(d)].copy_(d)
            self.g.replay()
            out_b.append(self.bits[:len(d)].clone())
            out_c.append(self.conf[:len(d)].clone())
        return torch.cat(out_b), torch.cat(out_c)


class FusedGraph:
    """bb.engine's registered fused engine (ConvLines, batch 1) behind the Graph interface."""

    def __init__(self, model, sha):
        from bb import engine
        self.e = engine.Deployed(model, "fused", sha)
        self.B, self.x, self.bits, self.conf = 1, self.e.x, self.e.bits, self.e.margin

    def replay(self):
        self.e.g.replay()

    def run(self, det):
        import torch
        b, m = self.e.run(det.cpu().numpy())
        return torch.as_tensor(b, device=det.device), torch.as_tensor(m, device=det.device)


def fused_batch(model, B):
    """bb.fused's kernels over B shots per call (pc_fused.Batched) in a CUDA graph: the fused path at B > 1."""
    import torch
    import pc_fused
    e = pc_fused.Batched(model, B)
    g = Graph(lambda x: e(x, B), B, model.D, torch.device("cuda"))
    g.engine = e                                            # the graph replays e's buffers
    return g


def build(pr, path, B, score="minabs"):
    """A network's timed path at batch B: the evaluation backend in a CUDA graph ("graph"), the same under torch.compile
    ("compiled"), or the fused engine ("fused", ConvLines: bb.engine's at batch 1, pc_fused.Batched above)."""
    import torch
    if path == "fused":
        return FusedGraph(pr.model, pr.sig["sha256"]) if B == 1 else fused_batch(pr.model, B)
    fn = make_fwd(pr, score)
    D = pr.model.D if pr.kind == "convlines" else 432
    return Graph(torch.compile(fn, dynamic=False) if path == "compiled" else fn, B, D, torch.device("cuda"))


def frozen_components(speed, locked):
    """speed_components.json, refused unless it is the file the lock froze (its sha256)."""
    if pc.sha256_file(speed) != locked["speed_sha256"]:
        raise SystemExit(f"{speed}: not the speed components frozen in the lock ({locked['speed_sha256'][:16]})")
    return json.load(open(speed))


def gpu_us(gr, warm=200, reps=5, calls=100):
    import torch
    for _ in range(warm):
        gr.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(calls):
            gr.replay()
        e1.record()
        e1.synchronize()
        out.append(1e3 * e0.elapsed_time(e1) / calls)
    return out


def host_rt(gr, rows, n=10_000, warm=200):
    """Per-call host wall time (us) of the single-shot round trip, cycling through pinned rows."""
    import torch
    H = rows[:max(1, min(len(rows), 4096))].cpu().pin_memory()
    hb = torch.empty(1, gr.bits.shape[1], dtype=gr.bits.dtype).pin_memory()
    hc = torch.empty(1, dtype=gr.conf.dtype).pin_memory()
    st = torch.cuda.current_stream()
    ts = np.empty(n)
    for i in range(n + warm):
        t = time.perf_counter()
        gr.x.copy_(H[i % len(H):i % len(H) + 1], non_blocking=True)
        gr.replay()
        hb.copy_(gr.bits, non_blocking=True)
        hc.copy_(gr.conf, non_blocking=True)
        st.synchronize()
        if i >= warm:
            ts[i - warm] = 1e6 * (time.perf_counter() - t)
    return ts


def amortized(gr, rows, secs=2.0, warm=10):
    """(dev, h2h): per-shot wall time (us) over consecutive batches, the input resident / copied host to host."""
    import torch
    B = gr.B
    nb = max(1, min(8, len(rows) // B))
    H = rows[:nb * B].cpu().reshape(nb, B, -1).pin_memory() if len(rows) >= B else \
        rows[:1].cpu().repeat(B, 1).reshape(1, B, -1).pin_memory()
    nb = len(H)
    hb = torch.empty_like(gr.bits, device="cpu").pin_memory()
    hc = torch.empty_like(gr.conf, device="cpu").pin_memory()
    st = torch.cuda.current_stream()
    gr.x.copy_(H[0])
    out = {}
    for mode in ("dev", "h2h"):
        for _ in range(warm):
            gr.replay()
        st.synchronize()
        m, t0 = 0, time.perf_counter()
        while True:
            if mode == "h2h":
                gr.x.copy_(H[m % nb], non_blocking=True)
            gr.replay()
            if mode == "h2h":
                hb.copy_(gr.bits, non_blocking=True)
                hc.copy_(gr.conf, non_blocking=True)
                st.synchronize()
            m += 1
            if m % 8 == 0:
                st.synchronize()
                if time.perf_counter() - t0 >= secs:
                    break
        st.synchronize()
        out[mode] = 1e6 * (time.perf_counter() - t0) / (m * B)
    return out


def quant(ts):
    return {"mean": float(ts.mean()), "p50": float(np.quantile(ts, 0.5)), "p99": float(np.quantile(ts, 0.99)),
            "p999": float(np.quantile(ts, 0.999)), "max": float(ts.max()), "n": int(len(ts))}


def emit(path, row):
    row = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(),
           "session": os.environ.get("SLURM_JOB_ID", ""), "pid": os.getpid(), **row}
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def screened(r):
    """A path passes the screen at a batch size (agreement row r) when its failures and its routing at every claim point
    and quantile agree with the accuracy path's (pc_common.agrees) and, for a ConvLines network, it is bitwise its
    accuracy path, every decision and confidence (the check requires the same of its frozen paths)."""
    return (pc.agrees(r["only_timed"], r["only_eval"], r["fail_eval"])
            and all(pc.agrees(u, v, n) for x in r["route"].values() for n, u, v in x.values())
            and (r["kind"] != "convlines" or r["bitwise"]))


def nets(a):
    import torch
    from bb import cascade_data as cdata
    from bb.cascade_eval import Predictor
    from bb.timing import gpu_watch
    dev, env = setup()
    path, fam, kind = pc.NETS[a.net]
    pr = Predictor(path, dev, "ema", "bf16")
    D = pr.model.D if kind == "convlines" else 432
    base = {"role": "net", "net": a.net, "kind": kind, "rep": a.rep, **env, "sha256": pr.sig["sha256"],
            "tree": os.environ.get("PC_TREE", "")}
    if kind == "convlines":                                 # the accuracy path: the reads' fused engine (pc_eval.py)
        import pc_fused
        acc = pc_fused.Batched(pr.model, pc.FUSED_READ_B).run
    else:                                                   # the accuracy path: bb.cascade_eval's batches
        def acc(d):
            o = pr(d)
            return o["bits"], o["conf"]

    def evaluated(d):
        """The accuracy path on rows d, chunk by chunk as the reads run it: (bits, conf) on the device."""
        out = [acc(d[i:i + pc.CHUNK]) for i in range(0, len(d), pc.CHUNK)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]).float()

    det, obs = cdata.eval_shots(pc.CODE, P_AGREE, range(N_AGREE_B // pc.CHUNK), kind=a.kind, procs=4, family=fam)
    dg, og = torch.as_tensor(det, device=dev), torch.as_tensor(obs, device=dev)
    ref, ref_c = evaluated(dg)
    fail_ref = (ref != og).any(1)
    rq = {}                                                 # the claim points: rows, evaluated bits and confidences
    for p in pc.RELAY_P:
        d = torch.as_tensor(cdata.eval_shots(pc.CODE, p, range(N_AGREE_B // pc.CHUNK), kind=a.kind, procs=4,
                                             family=fam)[0], device=dev)
        rq[p] = (d,) + evaluated(d)
    fwd = make_fwd(pr)
    batches = [int(b) for b in a.batches.split(",")]
    paths = [] if kind == "convlines" and a.quick else [("graph", batches)]
    if not a.quick:
        paths.append(("compiled", [b for b in COMPILED_B if b in batches]))
    if kind == "convlines":
        paths.append(("fused", batches))
    cfn = None
    for name, bs in paths:
        for B in bs:
            t_build = time.time()
            try:
                if name == "fused":
                    gr = build(pr, name, B)
                else:
                    if name == "compiled" and cfn is None:
                        cfn = torch.compile(fwd, dynamic=False)
                    gr = Graph(cfn if name == "compiled" else fwd, B, D, dev)
            except Exception as e:                          # recorded, the other paths go on
                emit(a.out, {**base, "path": name, "batch": B, "metric": "error", "error": repr(e)[:300]})
                continue
            row = {**base, "path": name, "batch": B, "prec": pc.precision(name, kind),
                   "build_s": round(time.time() - t_build, 1)}
            n_ag = N_AGREE_B1 if B < 64 else N_AGREE_B
            bits, conf = gr.run(dg[:n_ag])
            ft, fe = (bits != og[:n_ag]).any(1), fail_ref[:n_ag]
            only_t, only_e = int((ft & ~fe).sum()), int((~ft & fe).sum())
            differ = int((bits != ref[:n_ag]).any(1).sum())
            c_differ = int((conf.float() != ref_c[:n_ag]).sum())
            route, differ_c = {}, 0                         # routed / only timed / only evaluated, per p and quantile
            for p, (d, be, ce) in rq.items():
                bt, ct = gr.run(d[:n_ag])
                differ_c += int((bt != be[:n_ag]).any(1).sum())
                c_differ += int((ct.float() != ce[:n_ag]).sum())
                ct, ce_ = ct.double().cpu().numpy(), ce[:n_ag].double().cpu().numpy()
                route[str(p)] = {}
                for q in pc.ROUTE_Q:
                    tau = float(np.quantile(ce_, q))
                    me, mt = ce_ < tau, ct < tau
                    route[str(p)][str(q)] = [int(me.sum()), int((mt & ~me).sum()), int((~mt & me).sum())]
            ag = {**row, "metric": "agree", "p": P_AGREE, "kind_seed": a.kind, "shots": n_ag, "differ": differ,
                  "differ_claim": differ_c, "conf_differ": c_differ, "bitwise": differ == differ_c == c_differ == 0,
                  "fail_timed": int(ft.sum()), "fail_eval": int(fe.sum()), "only_timed": only_t, "only_eval": only_e,
                  "route": route}
            emit(a.out, {**ag, "ok": screened(ag)})
            stop = gpu_watch()
            res = {}
            if B == 1:
                res["gpu_us"] = gpu_us(gr)
                res["host_rt"] = quant(host_rt(gr, dg, n=a.n_rt))
            res["amortized"] = amortized(gr, dg, secs=a.secs)
            emit(a.out, {**row, "metric": "time", **res, "clock": stop()})
            del gr
            torch.cuda.empty_cache()


def relay_import():
    try:
        import relay_bp  # noqa: F401
    except ImportError:                                     # bb's CPU venv holds relay_bp (same Python, same NumPy)
        sys.path.append(pc.BB + "/venv/lib/python3.12/site-packages")
        import relay_bp  # noqa: F401


def _relay_init(family, p):
    relay_import()
    import pc_relay
    pc_relay.decoder(family, p)


def _relay_ready(_):
    time.sleep(0.2)
    return os.getpid()


def _relay_one(args):
    import pc_relay
    family, p, idx, det = args
    pred, secs = pc_relay.relay_task((family, p, None, det))
    return idx, pred, secs


def pipeline(a):
    import torch
    from bb import cascade_data as cdata
    from bb.cascade_eval import Predictor
    from bb.timing import gpu_watch
    dev, env = setup()
    L = json.load(open(a.locked))
    alone = a.config < 0                                    # Cascade alone, timed beside the pipelines in their job
    if alone and a.p not in pc.RELAY_P:
        raise SystemExit(f"--config -1 (Cascade alone) needs --p in {pc.RELAY_P}")
    c = {"type": "alone", "p": a.p, "S": L["B"]} if alone else L["configs"][a.config]
    fam, p, typ = L["family"], c["p"], c["type"]
    tau = c.get("tau", {})
    sp = frozen_components(a.speed, L)
    det, obs = cdata.eval_shots(pc.CODE, p, range((a.n + pc.CHUNK - 1) // pc.CHUNK), kind=a.kind, procs=4, family=fam)
    det, obs = det[:a.n], obs[:a.n]
    S = Predictor(pc.NETS[c["S"]][0], dev, "ema", "bf16")
    Bn = Predictor(pc.NETS[c["B"]][0], dev, "ema", "bf16") if c.get("B") else None
    for t, pr in ((c["S"], S), (c.get("B"), Bn)):
        if pr is not None and pr.sig["sha256"] != L["nets"][t]:
            raise SystemExit(f"{t}: sha256 differs from the locked network")
    use_relay = typ in ("S-B-R", "S-R")
    t_S = -np.inf if alone else (tau["tau"] if typ == "S-R" else tau["tau1"])     # alone: nothing is routed
    base = {"role": "pipeline", "config": a.config, "type": typ, "p": p, "family": fam, "S": c["S"], "B": c.get("B"),
            "q1": c.get("q1"), "r": c.get("r"), "q": c.get("q"), "score": c.get("score"), "mode": a.mode,
            "rep": a.rep, "kind_seed": a.kind, **env, "locked_sha256": pc.sha256_file(a.locked),
            "speed_sha256": L["speed_sha256"], "shots": len(det), "tree": os.environ.get("PC_TREE", "")}
    D = det.shape[1]

    def net(pr, tag, mode, score="minabs"):
        """The network on its frozen path for this mode (pc_common.frozen_paths)."""
        path, B = pc.frozen_paths(sp["nets"][tag])[mode]
        return build(pr, path, B, score), f"{path}@{B}"

    if a.mode == "single":
        gS, base["path_S"] = net(S, c["S"], "single")
        gB, base["path_B"] = net(Bn, c["B"], "single", c.get("score", "minabs")) if Bn is not None else (None, None)
        if use_relay:
            relay_import()
            import pc_relay
            dec, Lm = pc_relay.decoder(fam, p)
            for i in range(3):
                dec.decode(np.ascontiguousarray(det[i]))
        H = torch.as_tensor(det).pin_memory()
        hb, hc = torch.empty(1, pc.K, dtype=torch.uint8).pin_memory(), torch.empty(1).pin_memory()
        st = torch.cuda.current_stream()
        lat, cat, fail = np.empty(len(det)), np.zeros(len(det), np.int8), np.zeros(len(det), bool)
        stop = gpu_watch()
        for i in range(-200, len(det)):                    # 200 untimed shots first
            j = i % len(det)
            t0 = time.perf_counter()
            gS.x.copy_(H[j:j + 1], non_blocking=True)
            gS.replay()
            hb.copy_(gS.bits, non_blocking=True)
            hc.copy_(gS.conf, non_blocking=True)
            st.synchronize()
            k = 0
            if float(hc[0]) < t_S:
                if typ == "S-R":
                    k = 2
                else:
                    k = 1
                    gB.x.copy_(gS.x)
                    gB.replay()
                    hb.copy_(gB.bits, non_blocking=True)
                    hc.copy_(gB.conf, non_blocking=True)
                    st.synchronize()
                    if typ == "S-B-R" and float(hc[0]) < tau["tau2"]:
                        k = 2
            if k == 2:
                e = dec.decode(np.ascontiguousarray(det[j]))
                pred = (Lm @ e.astype(np.uint8)) % 2
            else:
                pred = hb[0].numpy()
            if i >= 0:
                lat[i] = 1e6 * (time.perf_counter() - t0)
                cat[i] = k
                fail[i] = bool((pred != obs[i]).any())
        out = {**base, "metric": "single", "latency_us": quant(lat), "clock": stop(),
               "frac_B": float((cat >= 1).mean()) if typ != "S-R" else 0.0, "frac_R": float((cat == 2).mean()),
               "fail": int(fail.sum())}
        for k, name in ((0, "S_only"), (1, "via_B"), (2, "via_R")):
            if (cat == k).any():
                out[f"latency_{name}_us"] = quant(lat[cat == k])
        emit(a.out, out)
        return
    from multiprocessing import get_context

    gS, base["path_S"] = net(S, c["S"], "amortized" if alone else "batched")   # alone: Cascade's amortized path
    bS = gS.B                                               # S's batch (Cascade alone: its amortized path's)
    gB, base["path_B"] = net(Bn, c["B"], "batched", c.get("score", "minabs")) if Bn is not None else (None, None)
    pool = get_context("spawn").Pool(pc.NCPU_RELAY, initializer=_relay_init, initargs=(fam, p)) if use_relay else None
    pids = set()
    while pool is not None and len(pids) < pc.NCPU_RELAY:  # every worker has built its decoder before the clock starts
        pids |= set(pool.map(_relay_ready, range(2 * pc.NCPU_RELAY), chunksize=1))
    nb = (len(det) + bS - 1) // bS
    H = torch.zeros(nb * bS, D, dtype=torch.uint8)
    H[:len(det)] = torch.as_tensor(det)
    H = H.reshape(nb, bS, D).pin_memory()
    hc_s = torch.empty(bS).pin_memory()
    hb_s = torch.empty(bS, pc.K, dtype=torch.uint8).pin_memory()
    hc_b = torch.empty(B_Q).pin_memory()
    hb_b = torch.empty(B_Q, pc.K, dtype=torch.uint8).pin_memory()
    pred = np.zeros((len(det), pc.K), np.uint8)
    route = np.zeros(len(det), np.int8)
    st = torch.cuda.current_stream()
    for _ in range(3):                                      # warm-up of both graphs
        gS.replay()
        if gB is not None:
            gB.replay()
    st.synchronize()
    pending, q_rows, q_idx, t_b = [], [], [], 0.0

    def run_b(rows, idx):
        nonlocal t_b
        tb = time.perf_counter()
        gB.x.zero_()
        gB.x[:len(idx)].copy_(rows)
        gB.replay()
        hb_b.copy_(gB.bits, non_blocking=True)
        hc_b.copy_(gB.conf, non_blocking=True)
        st.synchronize()
        pred[idx] = hb_b[:len(idx)].numpy()
        if typ == "S-B-R":
            sel = hc_b[:len(idx)].numpy().astype(np.float64) < tau["tau2"]
            if sel.any():
                ii = idx[sel]
                route[ii] = 2
                for k in range(0, len(ii), RELAY_TASK):
                    jj = ii[k:k + RELAY_TASK]
                    pending.append(pool.apply_async(_relay_one, ((fam, p, jj, det[jj]),)))
        t_b += time.perf_counter() - tb

    stop = gpu_watch()
    t0 = time.perf_counter()
    for j in range(nb):
        gS.x.copy_(H[j], non_blocking=True)
        gS.replay()
        hb_s.copy_(gS.bits, non_blocking=True)
        hc_s.copy_(gS.conf, non_blocking=True)
        st.synchronize()
        lo, hi = j * bS, min(len(det), (j + 1) * bS)
        pred[lo:hi] = hb_s[:hi - lo].numpy()
        r = np.nonzero(hc_s[:hi - lo].numpy().astype(np.float64) < t_S)[0]
        if not len(r):
            continue
        idx = lo + r
        if typ == "S-R":
            route[idx] = 2
            for k in range(0, len(idx), RELAY_TASK):
                jj = idx[k:k + RELAY_TASK]
                pending.append(pool.apply_async(_relay_one, ((fam, p, jj, det[jj]),)))
            continue
        route[idx] = 1
        q_rows.append(gS.x[torch.as_tensor(r, device=dev)])
        q_idx.append(idx)
        while sum(len(x) for x in q_idx) >= B_Q:
            rows, idx = torch.cat(q_rows), np.concatenate(q_idx)
            run_b(rows[:B_Q], idx[:B_Q])
            q_rows, q_idx = [rows[B_Q:]], [idx[B_Q:]]
    if q_idx and sum(len(x) for x in q_idx):
        run_b(torch.cat(q_rows), np.concatenate(q_idx))
    t_gpu = time.perf_counter() - t0
    secs = []
    for f in pending:
        idx, pr_, s_ = f.get()
        pred[idx] = pr_
        secs.append(s_)
    wall = time.perf_counter() - t0
    clock = stop()
    if pool is not None:
        pool.close()
        pool.join()
    secs = np.concatenate(secs) if secs else np.zeros(0)
    fail = (pred != obs).any(1)
    emit(a.out, {**base, "metric": "batched", "batch_S": bS, "batch_B": B_Q if gB is not None else None,
                 "relay_workers": pc.NCPU_RELAY if pool else 0, "wall_s": wall, "gpu_phase_s": t_gpu, "b_stage_s": t_b,
                 "per_shot_us": 1e6 * wall / len(det),
                 "per_round_us": 1e6 * wall / (len(det) * pc.R), "throughput_shots_s": len(det) / wall,
                 "frac_B": float((route >= 1).mean()) if typ != "S-R" else 0.0, "frac_R": float((route == 2).mean()),
                 "relay_s_mean": float(secs.mean()) if len(secs) else None, "fail": int(fail.sum()), "clock": clock})


def _cmp(t, e):
    """A timed mask t against the evaluated one e on the same shots (failures, or the shots routed to B or to R)."""
    only_t, only_e, n_e = int((t & ~e).sum()), int((~t & e).sum()), int(e.sum())
    return {"eval": n_e, "timed": int(t.sum()), "only_timed": only_t, "only_eval": only_e,
            "pass": pc.agrees(only_t, only_e, n_e)}


def check(a):
    """Step C4b (PHASE_C_PLAN.md 2.4, before any confirmation read): the frozen timed paths of one setting against the
    evaluated computation at the claim points (the module's docstring); writes check_<setting>.json."""
    import torch
    from multiprocessing import get_context
    from bb.cascade_eval import Predictor
    import pc_route
    dev, env = setup()
    L = json.load(open(a.locked))
    fam = L["family"]
    sp = frozen_components(a.speed, L)
    relay_import()
    import pc_relay
    alone = [t for t in dict.fromkeys((L["B"], pc.DEPTH[fam]) + pc.OURS) if t in sp["nets"]]
    out = {"family": fam, "locked": os.path.abspath(a.locked), "locked_sha256": pc.sha256_file(a.locked),
           "speed_sha256": L["speed_sha256"], "read": os.path.abspath(a.read),
           "eager": os.path.abspath(a.eager) if a.eager else None, **env,
           "tree": os.environ.get("PC_TREE", ""), "tolerance": {"rel": pc.AGREE_REL, "z": pc.AGREE_Z},
           "frozen": {t: {m: f"{x}@{b}" for m, (x, b) in pc.frozen_paths(v).items()} for t, v in sp["nets"].items()},
           "relay_union": {}, "relay_build_ms": {}, "secs": {}, "rows": []}
    preds, graphs = {}, {}

    def outputs(t, mode, rows, score="minabs"):
        """Network t on its frozen path for this mode over the detector rows (on the device): numpy bits, conf."""
        if t not in preds:
            preds[t] = Predictor(pc.NETS[t][0], dev, "ema", "bf16")
            if preds[t].sig["sha256"] != L["nets"].get(t, preds[t].sig["sha256"]):
                raise SystemExit(f"{t}: sha256 differs from the locked network")
        path, B = pc.frozen_paths(sp["nets"][t])[mode]
        if (t, path, B, score) not in graphs:
            graphs[(t, path, B, score)] = build(preds[t], path, B, score)
        if not len(rows):
            return np.zeros((0, pc.K), np.uint8), np.zeros(0, np.float32)
        bits, conf = graphs[(t, path, B, score)].run(rows)
        return bits.cpu().numpy(), conf.float().cpu().numpy()

    pool = get_context("spawn").Pool(pc.NCPU_RELAY, initializer=relay_import)
    for p in pc.RELAY_P:
        t0 = time.time()
        z, meta, idx = pc.load_read(a.read, p)
        if meta["family"] != fam:
            raise SystemExit(f"{a.read} at {p}: family {meta['family']}, locked {fam}")
        for t, i in idx.items():
            if t in L["nets"] and meta["models"][i]["sha256"] != L["nets"][t]:
                raise SystemExit(f"{t}: sha256 differs from the locked network")
        n = len(z["obs"])
        det = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
        dg = torch.as_tensor(det, device=dev)
        obs = np.unpackbits(z["obs"], axis=1, count=pc.K)
        full = {}

        def whole(t, mode):
            """Network t's frozen path for this mode over every row (computed once per path and batch)."""
            k = (t,) + pc.frozen_paths(sp["nets"][t])[mode]
            if k not in full:
                full[k] = outputs(t, mode, dg)
            return full[k]

        def bitwise(t, mode):
            """Network t's frozen path for this mode gives the read's every decision and confidence; a ConvLines
            network must (its read is its timed path), a Cascade network need not (pc_common.agrees judges it)."""
            if pc.NETS[t][2] != "convlines":
                return True
            bits, conf = whole(t, mode)
            return bool(np.array_equal(bits, np.unpackbits(z[f"pred{idx[t]}"], axis=1, count=pc.K))
                        and np.array_equal(conf, z[f"conf{idx[t]}"]))

        head = {"p": p, "kind_seed": meta["kind"], "shots": n}
        for t in alone:                                     # every network alone on its frozen paths
            if t not in idx:
                continue
            fe = z[f"fail{idx[t]}"].astype(bool)
            pe = np.unpackbits(z[f"pred{idx[t]}"], axis=1, count=pc.K)
            for mode in ("single", "amortized"):
                bits, conf = whole(t, mode)
                c = _cmp((bits != obs).any(1), fe)
                bw = bitwise(t, mode)
                out["rows"].append({**head, "what": "alone", "net": t, "mode": mode, "path": out["frozen"][t][mode],
                                    "differ": int((bits != pe).any(1).sum()),
                                    "conf_differ": int((conf != z[f"conf{idx[t]}"]).sum()), "bitwise_required":
                                    pc.NETS[t][2] == "convlines", "fail": c, "pass": c["pass"] and bw})
        if a.eager:                                         # gate finding 9: the compiled reads against eager ones
            ze, me, ie = pc.load_read(a.eager, p)
            if pc.det_digests(me) != pc.det_digests(meta):
                raise SystemExit(f"{a.eager} at {p}: other shots than {a.read}")
            for t in ie:
                if t in idx:
                    c = _cmp(z[f"fail{idx[t]}"].astype(bool), ze[f"fail{ie[t]}"].astype(bool))
                    out["rows"].append({**head, "what": "eval", "net": t, "mode": "compiled read against eager",
                                        "fail": c, "pass": c["pass"]})
        cfgs = [(i, c) for i, c in enumerate(L["configs"]) if c["p"] == p]
        ver = {}
        for i, c in cfgs:                                   # routing: evaluated (pc_route.masks) and timed, both modes
            ver[(i, "eval")] = pc_route.masks(z, meta, idx, c, c["tau"]) + (None,)
            for mode in ("single", "batched"):
                bS, cS = whole(c["S"], mode)
                cS = cS.astype(np.float64)                  # compared as the pipelines compare
                if c["type"] == "S-R":
                    ver[(i, mode)] = (np.zeros(n, bool), cS < c["tau"]["tau"], bS)
                    continue
                toB = cS < c["tau"]["tau1"]
                ii = np.nonzero(toB)[0]
                if mode == "single" and c.get("score", "minabs") == "minabs":     # batch 1: row by row anyway
                    bB, cB = (x[ii] for x in whole(c["B"], mode))
                else:                                       # the routed rows in order, as the pipeline queues them
                    bB, cB = outputs(c["B"], mode, dg[torch.as_tensor(ii, device=dev)], c.get("score", "minabs"))
                dec = bS.copy()
                dec[ii] = bB
                toR = np.zeros(n, bool)
                if c["type"] == "S-B-R":
                    toR[ii[cB.astype(np.float64) < c["tau"]["tau2"]]] = True
                ver[(i, mode)] = (toB, toR, dec)
        union = np.nonzero(np.any([v[1] for v in ver.values()], axis=0))[0] if ver else np.zeros(0, np.int64)
        has, rfail, rsec = np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan)
        if len(union):                                      # one seeded Relay-BP decision per shot, for every version
            res = pool.map(pc_relay.relay_task, [(fam, p, union[k:k + 10], det[union[k:k + 10]])
                                                 for k in range(0, len(union), 10)], chunksize=1)
            has[union] = True
            rfail[union] = (np.concatenate([r[0] for r in res]) != obs[union]).any(1)
            rsec[union] = np.concatenate([r[1] for r in res])
        for i, c in cfgs:
            toB_e, toR_e, _ = ver[(i, "eval")]
            fe = pc_route.config_fail(z, idx, c, toB_e, toR_e, (has, rfail, rsec))
            for mode in ("single", "batched"):
                toB, toR, dec = ver[(i, mode)]
                ft = (dec != obs).any(1)
                ft[toR] = rfail[toR]
                route = {}
                if c["type"] != "S-R":
                    route["to_B"] = _cmp(toB, toB_e)
                if c["type"] != "S-B":
                    route["to_R"] = _cmp(toR, toR_e)
                cf = _cmp(ft, fe)
                sbw = bitwise(c["S"], mode)
                out["rows"].append({**head, "what": "routed", "config": i, "type": c["type"],
                                    "decoder": pc.decoder_key(c), "S": c["S"], "B": c.get("B"), "mode": mode,
                                    "paths": {r: out["frozen"][c[r]][mode] for r in ("S", "B") if c.get(r)},
                                    "fail": cf, "routing": route, "S_bitwise": sbw,
                                    "pass": cf["pass"] and all(v["pass"] for v in route.values()) and sbw})
        out["relay_union"][str(p)] = int(len(union))
        pc_relay.decoder(fam, p)                            # the detector error model's matrices, built once
        tb = time.perf_counter()
        for k in range(20):
            pc_relay.decoder(fam, p, k)
        out["relay_build_ms"][str(p)] = 1e3 * (time.perf_counter() - tb) / 20
        out["secs"][str(p)] = round(time.time() - t0, 1)
    pool.close()
    pool.join()
    out["decoders"] = {}
    for r in out["rows"]:
        k = f"{r['net']} alone" if r["what"] == "alone" else (f"config {r['config']}" if r["what"] == "routed" else
                                                                  f"{r['net']} compiled read")
        out["decoders"][k] = out["decoders"].get(k, True) and r["pass"]
    json.dump(out, open(a.out, "w"), indent=1)
    for r in out["rows"]:
        print(json.dumps({k: r[k] for k in ("p", "what", "net", "config", "type", "mode", "pass") if k in r} |
                         {"fail_e_t": [r["fail"]["eval"], r["fail"]["timed"]],
                          "only_t_e": [r["fail"]["only_timed"], r["fail"]["only_eval"]]}))
    print(json.dumps({"decoders_failing": [k for k, v in out["decoders"].items() if not v],
                      "relay_build_ms": out["relay_build_ms"], "secs": out["secs"]}))


def summarize(a):
    """speed_components.json: per network, from the slowest of all its processes (the statistic of every network and
    every pipeline; PHASE_C_PLAN.md 5.2):
      t1_us, t1_path, p999_us  its fastest screened path at batch 1: the single-shot round trip's mean and 99.9th pct.
      a_best_us, a_best        its lowest host-to-host time per shot over the screened paths and batches <= 4,096
      a4096_us, a4096_path     the fastest screened path at batch 4,096 (the batched pipeline runs S and B there)
      processes                the fewest processes behind any of these three values (pc_report counts >= REPS)
      paths, agree, bitwise    every path's values, precision and screen per batch size; every (path, batch)'s screen
                               over its processes, and whether every one of them was bitwise the accuracy path
      gpu, torch, driver       the card, the venv's torch and the driver(s) of the rows (one card and one torch only)
    These are the network's frozen timed paths (pc_common.frozen_paths); a ConvLines network's candidates are its fused
    paths only (PHASE_C_PLAN.md 2.4). A path is screened at a batch size when every agreement row of it at that batch
    size passes (screened: failures at 0.55 %, routing at the claim points, a ConvLines path bitwise); a batch size that
    fails removes only its own values from the candidates. If a value has no screened candidate, the value of the graph
    path (of a ConvLines network, its fused path) is recorded for description and no_path_agrees is set: the network
    then enters no headline, nor does a routed configuration whose S it is (pc_report); there is no fallback."""
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    nets_ = {}
    for r in rows:
        if r.get("role") == "net":
            nets_.setdefault(r["net"], []).append(r)
    envs = sorted({(r["gpu"], r.get("torch", "")) for rs in nets_.values() for r in rs})
    if len(envs) != 1:
        raise SystemExit(f"rows from several GPU models or venvs: {envs}")
    out = {"gpu": envs[0][0], "torch": envs[0][1],
           "driver": ", ".join(sorted({r.get("driver", "") for rs in nets_.values() for r in rs})),
           "rows_sha256": pc.sha256_file(a.rows), "nets": {}}
    vb = str(pc.VERDICT_B)
    for t, rs in nets_.items():
        ok, bw = {}, {}
        for r in rs:
            if r["metric"] == "agree":
                k = (r["path"], r["batch"])
                ok[k] = ok.get(k, True) and screened(r)
                bw[k] = bw.get(k, True) and bool(r.get("bitwise"))
        tm = [r for r in rs if r["metric"] == "time"]
        paths = {}
        for path in sorted({r["path"] for r in tm}):
            pr = [r for r in tm if r["path"] == path]
            bs = sorted({r["batch"] for r in pr})
            b1 = [r for r in pr if r["batch"] == 1]
            paths[path] = {"prec": pc.precision(path, pc.NETS[t][2]),
                           "t1_us": max(r["host_rt"]["mean"] for r in b1) if b1 else None,
                           "p999_us": max(r["host_rt"]["p999"] for r in b1) if b1 else None,
                           "gpu_term_us": max(float(np.mean(r["gpu_us"])) for r in b1) if b1 else None,
                           "a_us": {str(B): max(r["amortized"]["h2h"] for r in pr if r["batch"] == B) for B in bs},
                           "a_dev_us": {str(B): max(r["amortized"]["dev"] for r in pr if r["batch"] == B) for B in bs},
                           "processes": {str(B): sum(r["batch"] == B for r in pr) for B in bs},
                           "screened": {str(B): ok.get((path, B), False) for B in bs}}
        own = "fused" if pc.NETS[t][2] == "convlines" else "graph"   # the fallback, described and flagged
        cand = [k for k in paths if own == "graph" or k == own]      # a ConvLines network: its fused paths only
        flag = False

        def pick(cands, fallback, val):
            """The fastest screened candidate (path, batch); with none, the fallback path's, flagged."""
            nonlocal flag
            if not cands:
                flag, cands = True, fallback
            return min(cands, key=val)

        def t1v(kb):
            return paths[kb[0]]["t1_us"]

        def av(kb):
            return paths[kb[0]]["a_us"][kb[1]]
        g = paths[own]["a_us"]
        b1 = pick([(k, "1") for k in cand if paths[k]["screened"].get("1")], [(own, "1")], t1v)
        ab = pick([(k, B) for k in cand for B, s in paths[k]["screened"].items() if s and int(B) <= pc.VERDICT_B],
                  [(own, B) for B in g if int(B) <= pc.VERDICT_B], av)
        a4 = pick([(k, vb) for k in cand if paths[k]["screened"].get(vb)], [(own, vb)], av)
        out["nets"][t] = {"kind": pc.NETS[t][2], "t1_us": paths[b1[0]]["t1_us"], "t1_path": b1[0],
                          "p999_us": paths[b1[0]]["p999_us"], "a_best_us": paths[ab[0]]["a_us"][ab[1]],
                          "a_best": f"{ab[0]}@{ab[1]}", "a4096_us": paths[a4[0]]["a_us"][vb], "a4096_path": a4[0],
                          "processes": min(paths[b1[0]]["processes"]["1"], paths[ab[0]]["processes"][ab[1]],
                                           paths[a4[0]]["processes"][vb]),
                          "no_path_agrees": flag, "paths": paths,
                          "agree": {f"{k[0]}@{k[1]}": v for k, v in ok.items()},
                          "bitwise": {f"{k[0]}@{k[1]}": v for k, v in bw.items()}}
    json.dump(out, open(a.out, "w"), indent=1)
    for t, v in out["nets"].items():
        print(json.dumps({"net": t, "t1_us": round(v["t1_us"], 1), "t1_path": v["t1_path"], "a_best_us":
                          round(v["a_best_us"], 3), "a_best": v["a_best"], "a4096_path": v["a4096_path"],
                          "processes": v["processes"], "no_path_agrees": v["no_path_agrees"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("nets")
    s.add_argument("--net", required=True, choices=list(pc.NETS))
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--batches", default=",".join(str(b) for b in pc.BATCHES))
    s.add_argument("--n-rt", type=int, default=10_000, help="single-shot round trips timed")
    s.add_argument("--secs", type=float, default=2.0, help="seconds per amortized measurement")
    s.add_argument("--quick", action="store_true", help="dry runs: the graph path only")
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED, help="seed kind of the screen's shots (13 in dry runs)")
    s.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--rows", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("check")
    s.add_argument("--read", required=True, help="the claim points' read (curves.sh with --save-det)")
    s.add_argument("--eager", default="", help="the same rows read eagerly by the EVAL_COMPILED networks")
    s.add_argument("--locked", required=True)
    s.add_argument("--speed", required=True, help="the speed_components.json frozen in the lock")
    s.add_argument("--out", required=True)
    s = sub.add_parser("pipeline")
    s.add_argument("--locked", required=True)
    s.add_argument("--config", type=int, required=True,
                   help="index into the locked configurations; -1: Cascade alone (the lock's B) at --p")
    s.add_argument("--p", type=float, default=None, help="with --config -1: the claim point of the shot stream")
    s.add_argument("--mode", required=True, choices=["single", "batched"])
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--n", type=int, default=20_000)
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED, help="seed kind of the shot stream (13 in dry runs)")
    s.add_argument("--speed", required=True, help="the speed_components.json frozen in the lock")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"nets": nets, "summarize": summarize, "check": check, "pipeline": pipeline}[a.cmd](a)


if __name__ == "__main__":
    main()
