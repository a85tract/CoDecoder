"""SPEED-RT stage 3 (speed_rt/STAGE3_batched.md): the networks of Figs. 3 and 4 in throughput mode on one GPU pod.

The batched path of a point: the release's engine of the point's network at batch 4,096 in CUDA graphs (surface: the
deployed Triton engine, surface_v2/timing/engine.py through timing.deployed, in sub-batches inside the graph where its
int32 offsets need it; BB72: the multi-shot engine, cascade/bb72/cascade_compare/batchopt/mfused.py; gross: bb.fused's
kernels batched by phase_c/pc_fused.Batched). One loop (`Loop.run`) is timed: per batch the detector bits are copied from
pinned host memory to the graph's input (a copy stream, two input slots), the graph runs, the per-shot float (surface:
the logit c; BB: the fp32 margin and the 12 bits) is copied to pinned host memory, and on the host the release's own
rules give the decisions and the routing flags (surface: convlines.decoder.route, |double(c)| < tau, decision c > 0; BB:
margin < tau, decision the network's bits), the routed shots' indices are collected and their detector bits are packed
into one contiguous host buffer for the fallback. Mode "alone" (CoDecoder alone) is the same loop without routing.

    python -m speed_rt.s3.gpu3 validate --point P --shots DIR --out DIR      batched path against the deployed pipeline
    python -m speed_rt.s3.gpu3 decide   --point P --shots DIR --out DIR      the decision pass on the timing shots
    python -m speed_rt.s3.gpu3 time     --point P --mode route|alone [--tau NAME] --shots DIR --decide DIR --core C --out DIR
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(os.path.dirname(HERE))
B = 4096
MIN_SECONDS = 10.0          # a timed process decodes whole passes of its timing set until at least this long has passed
WARM_PASSES = 1             # untimed passes before the timed loop (Triton's kernels compiled, clocks up)

SURF = {7: ("d7_b1w64_sv2", 0.90324866771698), 9: ("d9_b1w64_sh", 2.337686538696289), 11: ("d11_b5_sv2", 0.0),
        13: ("d13_b5_sv2", 0.0), 15: ("d15_b5_sv2", 1.9195706844329834), 17: ("d17_b5_sv2", 2.2122440338134766)}
POINTS = {f"surf-d{d}": dict(fam="surf", d=d, net=n, taus={"q": t}, set=f"s3-surf-d{d}", val_set=f"srt-surf-d{d}")
          for d, (n, t) in SURF.items()}
POINTS["bb72-p2"] = dict(fam="bb72", code="bb72", p=0.002, net="weights/bb/r12_final_ema.pt",
                         taus={"q": 2.979396343231201}, set="s3-bb72-p2", val_set="srt-bb72-p2")
POINTS["bb72-p3"] = dict(fam="bb72", code="bb72", p=0.003, net="weights/bb/r12_final_ema.pt",
                         taus={"q": 3.6329734325408936}, set="s3-bb72-p3", val_set="srt-bb72-p3")
POINTS["gross-p4"] = dict(fam="gross", code="bb144", p=0.004, net="weights/bb/g2_D12k113c10_final_ema.pt",
                          taus={"bal": 2.2055678367614746, "acc": 6.272581100463867}, set="s3-gross-p4",
                          val_set="srt-gross-p4")
TREES = {"bb72": "bb/bb72/ising_compare", "gross": "bb/gross_stage4/ising_compare"}
USE = {"gross-p4": 2000}    # the gross code's timing set: the first 2,000 shots of s3-gross-p4 chunk 0 (STAGE3 section 4)
CASC = "cascade/bb72/cascade_compare"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def paths(fam):
    os.environ.setdefault("QND_RELAY", os.path.join(os.path.dirname(REL), "relay"))   # Relay-BP's testdata at d185194
    if fam == "surf":
        sys.path.insert(0, os.path.join(REL, "surface_v2"))
    else:
        for p in (os.path.join(REL, CASC, "batchopt"), os.path.join(REL, CASC, "phase_c"),
                  os.path.join(REL, os.path.dirname(TREES[fam]), "qldpc_neural"), os.path.join(REL, TREES[fam])):
            sys.path.insert(0, p)
    sys.path.insert(0, REL)


# ---------------------------------------------------------------------------------------------------------------- shots
def load_shots(pt, d, use=True):
    """(det (n, m) uint8, obs) of a directory's chunk files of the point in chunk order (surface: testset files; BB: the
    BB trees' npz layout), and the files' names and sha256."""
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    dets, obs, rec = [], [], []
    for f in files:
        p = os.path.join(d, f)
        rec.append((f, sha256_file(p)))
        z = np.load(p)
        if pt["fam"] == "surf":
            from convlines import testset
            x, o = testset.load(p, pt["d"])
        else:
            n = int(z["n"])
            x = np.unpackbits(z["det"], axis=1, count=int(z["D"]))[:n]
            o = np.unpackbits(z["obs"], axis=1, count=int(z["k"]))[:n]
        dets.append(np.ascontiguousarray(x, np.uint8))
        obs.append(o)
    det, ob = np.concatenate(dets), np.concatenate(obs)
    if use and a_point(pt) in USE:
        det, ob = det[:USE[a_point(pt)]], ob[:USE[a_point(pt)]]
    return det, ob, rec


def a_point(pt):
    return next(k for k, v in POINTS.items() if v is pt)


# --------------------------------------------------------------------------------------------------------------- engines
class Net:
    """The batched path's device part: fn(x (n, m) uint8 on the GPU) -> output tensors (surface: (c,); BB: (bits,
    margin)), every output a static buffer, so that a CUDA graph can be captured; plus the deployed reference."""

    def __init__(self, pt, dev="cuda"):
        import torch
        self.pt, self.fam = pt, pt["fam"]
        paths(self.fam)
        if self.fam == "surf":
            from convlines import decoder
            from timing import deployed
            t, cfg = deployed.tables(pt["d"], pt["net"])
            self.t, self.cfg = deployed.to_cuda(t), cfg
            self.eng = deployed.engine(self.t, cfg)
            e = self.eng
            per = max(e.m * e.C, e.NG * e.CB, e.NT * e.DP * e.C, e.NT * e.DP)
            self.sb = B
            while self.sb * per >= 2 ** 31:                   # the engine's int32 offsets: whole sub-batches inside the graph
                self.sb //= 2
            self.m = e.m
            self.sha = decoder.weights_sha256(pt["net"])
            self.engine_name = f"surface_v2/timing/engine.py FROZEN (stages {e.cfg['stages']}, pdl {e.cfg['pdl']}), sub-batch {self.sb}"
        else:
            from bb import engine as bbe
            path = os.path.join(REL, pt["net"])
            self.sha = sha256_file(path)
            self.model, _ = bbe.load_net(path, dev)
            self.m = self.model.D
            if self.fam == "bb72":
                import mfused
                self.eng = mfused.Engine(self.model, B)
                self.engine_name = "batchopt/mfused.Engine (multi-shot)"
            else:
                import pc_fused
                self.eng = pc_fused.Batched(self.model, B)
                self.engine_name = "phase_c/pc_fused.Batched (bb.fused's kernels, batched)"
            self.sb = B
        self.torch = torch

    def outputs(self, n):
        torch = self.torch
        if self.fam == "surf":
            return (torch.empty(n, dtype=torch.float32, device="cuda"),)
        return (torch.empty(n, self.model.K, dtype=torch.uint8, device="cuda"),
                torch.empty(n, dtype=torch.float32, device="cuda"))

    def fn(self, x, outs):
        """Run the engine on x (n rows), writing into outs (static)."""
        n = x.shape[0]
        if self.fam == "surf":
            for lo in range(0, n, self.sb):
                hi = min(n, lo + self.sb)
                c, _, _ = self.eng(x[lo:hi])
                outs[0][lo:hi].copy_(c)
        else:
            bits, margin = self.eng(x, n)
            outs[0].copy_(bits)
            outs[1].copy_(margin)

    def reference(self, det):
        """The deployed reaction-mode pipeline's per-shot outputs at batch one on this GPU: surface (c,); BB (bits,
        margin). d = 7's deployed engine (the resident kernel, sm_90a) is not run here: its c comes from stage 2's record."""
        if self.fam == "surf":
            from timing import deployed
            return (deployed.Logits(self.pt["d"], name=self.pt["net"])(det),)
        from bb import engine as bbe
        dep = bbe.Deployed(self.model, "fused", self.sha)
        bits, margin = dep.run(det)
        return bits, margin


def host_rule(pt, outs_np, tau):
    """The release's rules on the host: (decision, routed) of a batch; tau None: no routing (CoDecoder alone)."""
    if pt["fam"] == "surf":
        from convlines import decoder
        c = outs_np[0]
        dec = (c > 0).astype(np.uint8)
        return dec, (decoder.route(c, tau) if tau is not None else None)
    bits, margin = outs_np
    return bits, (margin < tau if tau is not None else None)


class Loop:
    """Pinned host shots -> decisions (and routed indices, packed routed shots) on the host, B shots per graph."""

    def __init__(self, net, det):
        torch = net.torch
        self.net, self.n = net, len(det)
        self.H = torch.from_numpy(np.ascontiguousarray(det)).pin_memory()
        self.Hn = self.H.numpy()
        self.batches = [(lo, min(self.n, lo + B)) for lo in range(0, self.n, B)]
        sizes = sorted({hi - lo for lo, hi in self.batches})
        self.g = {}
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            for s in (0, 1):
                for n in sizes:
                    x = torch.zeros(n, net.m, dtype=torch.uint8, device="cuda")
                    x.copy_(self.H[:n])
                    outs = net.outputs(n)
                    with torch.cuda.stream(st):
                        for _ in range(2):
                            net.fn(x, outs)
                    torch.cuda.current_stream().wait_stream(st)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        net.fn(x, outs)
                    hout = [torch.empty(o.shape, dtype=o.dtype).pin_memory() for o in outs]
                    self.g[(s, n)] = (g, x, outs, hout, [h.numpy() for h in hout])
        torch.cuda.synchronize()
        self.cs = torch.cuda.Stream()
        self.loaded = [torch.cuda.Event() for _ in range(2)]
        self.free = [torch.cuda.Event() for _ in range(2)]
        self.done = [torch.cuda.Event() for _ in range(2)]
        m = net.m
        self.pack = np.empty((self.n, m), np.uint8)               # the routed shots' bits, packed in shot order
        self.idx = np.empty(self.n, np.int64)
        self.dec = np.empty((self.n,) + ((net.model.K,) if net.fam != "surf" else ()), np.uint8)
        self.routed = np.zeros(self.n, bool)
        self.raw = [np.empty((self.n,) + tuple(o.shape[1:]), np.float32 if o.dtype == torch.float32 else np.uint8)
                    for o in net.outputs(1)]

    def _host(self, i, s, tau, keep):
        lo, hi = self.batches[i]
        n = hi - lo
        hn = self.g[(s, n)][4]
        dec, routed = host_rule(self.net.pt, [h[:n] for h in hn], tau)
        self.dec[lo:hi] = dec
        if keep:
            for r, h in zip(self.raw, hn):
                r[lo:hi] = h[:n]
        if routed is not None:
            k = np.flatnonzero(routed)
            r = len(k)
            self.idx[self.nr:self.nr + r] = k + lo
            np.take(self.Hn[lo:hi], k, axis=0, out=self.pack[self.nr:self.nr + r])
            self.routed[lo:hi] = routed
            self.nr += r

    def run(self, tau, keep=False):
        """One pass over the shots; tau None: CoDecoder alone (no routing). -> the number of routed shots."""
        torch = self.net.torch
        ms = torch.cuda.current_stream()
        self.nr = 0
        prev = None
        for i, (lo, hi) in enumerate(self.batches):
            s, n = i % 2, hi - lo
            g, x, outs, hout, _ = self.g[(s, n)]
            with torch.cuda.stream(self.cs):
                self.cs.wait_event(self.free[s])
                x.copy_(self.H[lo:hi], non_blocking=True)
                self.loaded[s].record(self.cs)
            ms.wait_event(self.loaded[s])
            g.replay()
            self.free[s].record(ms)
            for h, o in zip(hout, outs):
                h.copy_(o, non_blocking=True)
            self.done[s].record(ms)
            if prev is not None:
                self.done[prev[1]].synchronize()
                self._host(prev[0], prev[1], tau, keep)
            prev = (i, s)
        self.done[prev[1]].synchronize()
        self._host(prev[0], prev[1], tau, keep)
        return self.nr


# ------------------------------------------------------------------------------------------------------------- commands
def env_record():
    import subprocess
    import torch
    import triton
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit,power.default_limit",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return dict(gpu=torch.cuda.get_device_name(0), smi=smi, torch=torch.__version__, triton=triton.__version__,
                numpy=np.__version__, python=sys.version.split()[0],
                code={f: sha256_file(os.path.join(REL, f)) for f in (
                    "speed_rt/s3/gpu3.py", "surface_v2/timing/engine.py", "surface_v2/timing/deployed.py",
                    f"{CASC}/batchopt/mfused.py", f"{CASC}/phase_c/pc_fused.py",
                    f"{TREES['bb72']}/bb/fused.py", f"{TREES['gross']}/bb/fused.py")})


def stage1_window(tag, out):
    """Stage 1's window, as stage 1 applied it (speed_rt/cascade/srtlib.py Window: nvidia-smi samples every 100 ms and
    the compute applications every second; rejected for another compute application, a gap over 1 s, a clock-event
    reason other than idle, applications clocks or the SW power cap, an enforced power limit below the default, or an SM
    clock below 90 % of the maximum in a sample without the SW power cap; the workload's own power cap counts)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("srtlib", os.path.join(REL, "speed_rt/cascade/srtlib.py"))
    srtlib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srtlib)
    os.makedirs(out, exist_ok=True)
    os.environ["SRT_WINDOWS"] = os.path.join(out, "windows.jsonl")
    return srtlib.Window(tag).begin(0.0)


def digest(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def cmd_validate(a):
    """The batched path against the deployed pipeline on validation shots (stage 2's timing set of the point, chunk 0,
    regenerated and sha256-checked against stage 2's manifest): decisions and every tau's routing flags must be equal shot
    for shot (pass); the per-shot floats' bitwise agreement is recorded."""
    pt = POINTS[a.point]
    net = Net(pt)
    det, _, files = load_shots(pt, a.shots, use=False)
    det = det[:a.n]
    lp = Loop(net, det)
    lp.run(None, keep=True)
    got = [r.copy() for r in lp.raw]
    if pt["fam"] == "surf" and pt["d"] == 7:
        z = np.load(os.path.join(REL, "speed_rt/s2/surface/records/low/masks/val_d7_b1w64_sv2.npz"))
        ref, refsrc = (z["c"][:a.n].astype(np.float32),), "stage 2's record of the resident kernel (H100), val_d7_b1w64_sv2.npz"
    else:
        ref, refsrc = net.reference(det), "the deployed pipeline at batch one on this GPU"
    rec = dict(point=a.point, network=pt["net"], network_sha256=net.sha, engine=net.engine_name, shots=len(det),
               files=files, reference=refsrc, env=env_record())
    ok = True
    for name, tau in pt["taus"].items():
        dg, rg = host_rule(pt, got, tau)
        dr, rr = host_rule(pt, ref, tau)
        dd = int((np.asarray(dg).reshape(len(det), -1) != np.asarray(dr).reshape(len(det), -1)).any(1).sum())
        fd = int((rg != rr).sum())
        rec[f"tau_{name}"] = dict(tau=tau, decisions_differ=dd, flags_differ=fd, routed_batched=int(rg.sum()),
                                  routed_reference=int(rr.sum()))
        ok &= dd == 0 and fd == 0
    rec["float_bitwise_equal"] = [int((np.asarray(g).view(np.uint8).reshape(len(det), -1) ==
                                       np.asarray(r, g.dtype).view(np.uint8).reshape(len(det), -1)).all(1).sum())
                                  for g, r in zip(got, ref)]
    rec["passed"] = bool(ok)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"validate_{a.point}.json"), "x") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in rec if k.startswith("tau_") or k in ("point", "passed", "float_bitwise_equal",
                                                                              "engine")}))
    return 0 if ok else 5


def cmd_decide(a):
    """The stage's decision pass (untimed): the batched path over the timing shots once; per tau the routing flags, the
    decisions and the per-shot floats -> decide_<point>.npz and its digests."""
    pt = POINTS[a.point]
    net = Net(pt)
    det, obs, files = load_shots(pt, a.shots)
    obs = np.asarray(obs, np.uint8).reshape(len(det), -1)
    lp = Loop(net, det)
    out = dict(point=a.point, network=pt["net"], network_sha256=net.sha, engine=net.engine_name, shots=len(det),
               files=files, env=env_record(), taus={})
    arrays = {}
    for name, tau in pt["taus"].items():
        nr = lp.run(tau, keep=True)
        arrays[f"routed_{name}"] = lp.routed.astype(np.uint8).copy()
        fail = (lp.dec.reshape(len(det), -1) != obs).any(1)                  # descriptive only
        out["taus"][name] = dict(tau=tau, routed=int(nr), digest=digest(lp.dec, lp.routed),
                                 network_failures_all=int(fail.sum()), network_failures_unrouted=int((fail & ~lp.routed).sum()))
    arrays["dec"] = lp.dec.copy()
    for k, r in enumerate(lp.raw):
        arrays[f"raw{k}"] = r.copy()
    out["dec_digest"] = digest(lp.dec)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f"decide_{a.point}.npz"), **arrays)
    out["npz_sha256"] = sha256_file(os.path.join(a.out, f"decide_{a.point}.npz"))
    with open(os.path.join(a.out, f"decide_{a.point}.json"), "x") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("point", "shots", "taus", "engine")}))
    return 0


def cmd_time(a):
    """One fresh timed process: untimed warm pass(es), then whole passes for at least MIN_SECONDS between two clock
    readings, under speed_rt.watch's window (rule floor); its outputs (decisions and flags of its last pass) must equal the
    decision pass's. -> time_<point>_<mode>[_<tau>]_<rep>.json"""
    pt = POINTS[a.point]
    os.sched_setaffinity(0, {a.core})
    net = Net(pt)
    det, _, files = load_shots(pt, a.shots)
    ref = json.load(open(os.path.join(a.decide, f"decide_{a.point}.json")))
    tau = None if a.mode == "alone" else pt["taus"][a.tau]
    lp = Loop(net, det)
    for _ in range(WARM_PASSES):
        lp.run(tau)
    w = stage1_window(f"s3 {a.point} {a.mode} {a.tau} rep {a.rep} attempt {a.attempt}", a.out)
    t0 = time.perf_counter()
    passes = 0
    while True:
        nr = lp.run(tau)
        passes += 1
        t1 = time.perf_counter()
        if t1 - t0 >= MIN_SECONDS:
            break
    win = w.end()
    wall = t1 - t0
    if tau is None:
        same = digest(lp.dec) == ref["dec_digest"]
    else:
        same = digest(lp.dec, lp.routed) == ref["taus"][a.tau]["digest"] and nr == ref["taus"][a.tau]["routed"]
    rec = dict(point=a.point, mode=a.mode, tau_name=a.tau if tau is not None else None, tau=tau, rep=a.rep, attempt=a.attempt,
               network=pt["net"], network_sha256=net.sha, engine=net.engine_name, batch=B, shots=len(det),
               files=files, passes=passes, wall_s=wall, per_shot_us=1e6 * wall / (passes * len(det)),
               routed_per_pass=int(nr) if tau is not None else 0, outputs_equal_decision_pass=bool(same),
               window=win, env=env_record(), core=a.core)
    os.makedirs(a.out, exist_ok=True)
    name = f"time_{a.point}_{a.mode}" + (f"_{a.tau}" if tau is not None else "") + f"_{a.rep}" + \
        (f"_a{a.attempt}" if a.attempt > 1 else "") + ".json"
    with open(os.path.join(a.out, name), "x") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("point", "mode", "tau_name", "rep", "passes", "per_shot_us",
                                          "outputs_equal_decision_pass")}))
    return 0 if same and win["accepted"] else 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("validate", "decide", "time"))
    ap.add_argument("--point", required=True, choices=sorted(POINTS))
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--mode", choices=("route", "alone"), default="route")
    ap.add_argument("--tau", default="q")
    ap.add_argument("--decide", default="")
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--attempt", type=int, default=1)
    a = ap.parse_args()
    sys.exit({"validate": cmd_validate, "decide": cmd_decide, "time": cmd_time}[a.cmd](a))


if __name__ == "__main__":
    main()
