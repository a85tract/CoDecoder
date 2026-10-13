"""Faster batch-1 engines for gross's frozen network g2_D12k113c10 (bb/GROSS_FAST_ENGINE.md). bb.engine, bb.net and
bb.fused are not changed (their sha256 are in the frozen engine signatures); this module adds candidate kinds beside
bb.engine's and compares each with the frozen graph32 read. Every kind runs one shot alone in one captured CUDA graph:
  graph32    bb.engine's frozen rung, the reference: the fp32 module
  graph      bb.engine's bf16-autocast rung
  graph16    the same module under fp16 autocast (the vote and the margin in fp32, as in graph)
  compile32  torch.compile(mode="max-autotune-no-cudagraphs", inductor freezing) of the fp32 call; the CUDA graph is
             captured here as bb.engine does for its inductor rung (max-autotune's own CUDA graphs cannot run inside a
             captured graph)
  compile16  the same for the fp16-autocast call
  trt16      TensorRT with FP16 kernels (LayerNorm kept in fp32) built from the ONNX export of the trunk and the
             read-out (detector bits -> translate logits z); the detector parity, the vote and the margin in fp32
             PyTorch as bb.net computes them; TensorRT's enqueue and those ops in one CUDA graph
  fused      bb.fused's Triton engine (fp16 matmul inputs, fp32 accumulation, fp32 residual stream and vote), its
             frozen launch settings (8 taps per iteration, 2 pipeline stages, programmatic dependent launch on Hopper)
  fusedt     the same engine and source with the settings tuned for gross on the H100 (FUSEDT: 16 taps per iteration,
             4 stages, no programmatic dependent launch; a sweep of 24 settings, GROSS_FAST_ENGINE.md)
Subcommands:
  eq   python -m bb.fast eq --ckpt final_ema.pt --kind fused --ref g2_D12k113c10 --label a100 --out E.json
       the 10 CAL chunks read at batch 1 through the kind (bb.route's npz per chunk in BB_PREDS, tag RUN_KIND_LABEL),
       compared with the reference read: decisions that differ, shots whose routing at tau changes, the failures of the
       network alone and of the hybrid (sector Relay-BP on the routed shots) for both, exact McNemar on the discordant
       shots
  lat  python -m bb.fast lat --ckpt final_ema.pt --kind fused --proc 1 --out T.csv
       bb.latency's protocol and rows for any kind (200 warm-up replays, 5 x 100 timed replays with CUDA events, the
       handoff, nvidia-smi's acceptance); one fresh process per call, the slowest of 3 counts"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from bb import engine, net, opp, route, shots
from bb.timing import append_row, gpu_watch, sig_digest

KINDS = ("graph32", "graph", "graph16", "compile32", "compile16", "trt16", "fused", "fusedt")
FUSEDT = {"tpi": 16, "stages": 4, "pdl": False}
TAU = 0.5629201531410217           # the frozen policy's routing threshold (q = 0.5 %, STAGE3_GROSS_PLAN.md)
HERE = os.path.dirname(os.path.abspath(__file__))
PRECISION = {"graph16": "fp16 autocast, fp32 vote and margin",
             "compile32": "fp32 (TF32 off), inductor max-autotune kernels",
             "compile16": "fp16 autocast, fp32 vote and margin, inductor max-autotune kernels",
             "trt16": "TensorRT FP16 kernels (LayerNorm fp32) for the trunk and read-out; fp32 parity, vote and margin"}


class Fast(engine.Deployed):
    """bb.engine.Deployed for every kind above: the same fixed settings, input buffer, run(), time_calls() and
    handoff_us(); the frozen kinds (graph32, graph, fused) are bb.engine's own, signature included."""

    def __init__(self, model, kind="fused", net_sha256="", cache=""):
        if kind in ("graph32", "graph", "fused"):
            super().__init__(model, kind, net_sha256)
            return
        if kind not in KINDS:
            raise ValueError(kind)
        super().__init__(model, "fp32", net_sha256)        # the fixed settings only (fp32 captures nothing)
        self.kind = kind
        if kind == "graph16":
            fn = self._call16
        elif kind == "fusedt":
            from bb import fused
            fn = self.fused = fused.Engine(model, **FUSEDT)
        elif kind in ("compile32", "compile16"):
            import torch._inductor.config as icfg
            icfg.freezing = True                           # weights, masks and position gates folded as constants
            fn = torch.compile(self._call16 if kind == "compile16" else self._call32,
                               mode="max-autotune-no-cudagraphs", dynamic=False)
        else:
            fn = self.trt = Trt(model, cache)
        self.x = torch.zeros(1, model.D, dtype=torch.uint8, device=self.dev)
        with torch.no_grad():
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    fn(self.x)
            torch.cuda.current_stream().wait_stream(st)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.bits, self.margin = fn(self.x)

    def _call16(self, x):
        with torch.autocast("cuda", dtype=torch.float16):
            s, _, _ = self.model(x)
        _, bits, mg = self.model.decide(s)
        return bits, mg.float()

    def _call32(self, x):
        s, _, _ = self.model(x)
        _, bits, mg = self.model.decide(s)
        return bits, mg.float()

    def signature(self):
        sig = super().signature()
        if self.kind in PRECISION:
            sig["precision"] = PRECISION[self.kind]
            sig["src"]["fast.py"] = engine.sha256_file(os.path.join(HERE, "fast.py"))[:16]
        if self.kind == "fusedt":
            from bb import fused
            import triton
            sig.update(precision=fused.PRECISION, triton=triton.__version__, launch=FUSEDT)
            sig["src"]["fused.py"] = engine.sha256_file(os.path.join(HERE, "fused.py"))[:16]
            sig["src"]["fast.py"] = engine.sha256_file(os.path.join(HERE, "fast.py"))[:16]
        if self.kind.startswith("compile"):
            import triton
            sig["triton"] = triton.__version__
        if self.kind == "trt16":
            import tensorrt
            sig["tensorrt"] = tensorrt.__version__
        return sig


class _Trunk(torch.nn.Module):
    """bb.net's features + read-out, detector bits (1, D) float -> z (1, K, G), in ops that ONNX and TensorRT take:
    the circular pads as concatenations, each tap convolution's masked weight and the position gates P as constants,
    the read-out's einsums as matmuls (the query-key product folded per head, as bb.fused does)."""

    def __init__(self, model):
        super().__init__()
        if model.cfg.get("tblocks") or model.cfg.get("rbins"):
            raise ValueError("trt16 has neither time blocks nor the read-out branch R")
        self.m = model
        hd = model.head
        H, dh, C = hd.h, hd.dh, hd.ln.normalized_shape[0]
        with torch.no_grad():
            self.register_buffer("wg", (hd.key.weight.view(H, dh, C) * hd.q0.view(H, dh, 1)).sum(1).t() * dh ** -0.5)
            self.register_buffer("P", hd.position().float().contiguous())                  # (H, K G, S)
            self.register_buffer("u0", (hd.q0 + hd.e).repeat_interleave(hd.G, 0) + hd.o.bias)
            self.wm = [(c.weight * c.mask).detach() for c in [model.stem] + [b.conv for b in model.blocks]]

    @staticmethod
    def _wrap(x, lo, hi, dim):                             # circular pad: lo entries before and hi after along dim
        assert lo >= 0 and hi >= 0, (lo, hi)
        n = x.shape[dim]
        parts = ([x.narrow(dim, n - lo, lo)] if lo else []) + [x] + ([x.narrow(dim, 0, hi)] if hi else [])
        return torch.cat(parts, dim) if len(parts) > 1 else x

    def _conv(self, x, c, w):                              # bb.net.TapConv: F.pad(..., mode="circular") + conv3d
        x = self._wrap(self._wrap(x, -c.blo, c.bhi, -1), -c.alo, c.ahi, -2)
        return F.conv3d(x, w, c.bias, padding=(c.dt, 0, 0))

    def forward(self, det):
        m, hd = self.m, self.m.head
        T, l, mm = m.T, m.l, m.m
        pad = torch.cat([det, det.new_zeros(1, 1)], 1)
        x = pad[:, m.gidx].view(1, T, 2, l, mm).transpose(1, 2)
        x = self._conv(x, m.stem, self.wm[0]).permute(0, 2, 3, 4, 1) + m.tbias[None, :, None, None, :]
        for blk, w in zip(m.blocks, self.wm[1:]):
            u = F.silu(blk.down(blk.ln(x)))
            v = F.silu(self._conv(u.permute(0, 4, 1, 2, 3), blk.conv, w).permute(0, 2, 3, 4, 1))
            x = x + blk.up(v)
        f = x.reshape(1, T * m.n2, -1)[0]                  # (S, C)
        y = hd.ln(f)
        gate = torch.sigmoid(y @ self.wg + hd.bc)          # (S, H)
        U = (gate.unsqueeze(-1) * hd.v(y).view(-1, hd.h, hd.dh)).transpose(0, 1)        # (H, S, dh)
        a = torch.matmul(self.P, U).transpose(0, 1).reshape(-1, hd.h * hd.dh)            # (K G, C)
        u = self.u0 + a @ hd.o.weight.t()
        return hd.mlp(u).view(1, m.K, m.G)


class Trt:
    """TensorRT FP16 engine of _Trunk (built once per process from its ONNX export) followed by bb.net's detector
    parity, vote and decision in fp32: (1, D) uint8 on the GPU -> (bits (1, K) uint8, margin (1,) fp32)."""

    def __init__(self, model, cache=""):
        import tensorrt as trt
        self.model = model
        dev = next(model.parameters()).device
        d = cache or f"/tmp/bb_fast_trt_{os.getpid()}"
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "trunk.onnx")
        self.xf = torch.zeros(1, model.D, dtype=torch.float32, device=dev)
        self.z = torch.zeros(1, model.K, model.G, dtype=torch.float32, device=dev)
        trunk = _Trunk(model).eval()
        with torch.no_grad():
            err = (trunk(self.xf + 1) - model.head(model.features(self.xf + 1))).abs().max().item()
            if err > 1e-3:
                raise RuntimeError(f"_Trunk differs from bb.net by {err}")
            torch.onnx.export(trunk, (self.xf,), path, input_names=["det"], output_names=["z"], opset_version=18,
                              dynamo=False)
        log = trt.Logger(trt.Logger.WARNING)
        b = trt.Builder(log)
        nw = b.create_network(0)
        ps = trt.OnnxParser(nw, log)
        if not ps.parse_from_file(path):
            raise RuntimeError("; ".join(str(ps.get_error(i)) for i in range(ps.num_errors)))
        cfg = b.create_builder_config()
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(nw.num_layers):
            ly = nw.get_layer(i)
            if ly.type == trt.LayerType.NORMALIZATION:
                ly.precision = trt.float32
                ly.set_output_type(0, trt.float32)
        plan = b.build_serialized_network(nw, cfg)
        if plan is None:
            raise RuntimeError("TensorRT build failed")
        self.rt = trt.Runtime(log)
        self.eng = self.rt.deserialize_cuda_engine(plan)
        self.ctx = self.eng.create_execution_context()
        self.ctx.set_tensor_address("det", self.xf.data_ptr())
        self.ctx.set_tensor_address("z", self.z.data_ptr())

    def __call__(self, x):
        self.xf.copy_(x)
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        _, bits, mg = self.model.decide(self.model.vote(self.z, self.model.detparity(x)))
        return bits, mg.float()


def mcnemar(a, b):
    """Exact two-sided McNemar on paired failures: shots where only a fails / only b fails, and the p-value."""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return {"only_candidate": n01, "only_reference": n10, "p": p}


def compare(obs, rb, rm, cb, cm, fb, tau=TAU):
    """Candidate (cb, cm) against the reference read (rb, rm) on the same shots; fb = the fallback's decisions."""
    nr, nc, ff = (rb != obs).any(1), (cb != obs).any(1), (fb != obs).any(1)
    rr, rc = rm < tau, cm < tau
    hr, hc = np.where(rr, ff, nr), np.where(rc, ff, nc)
    d = np.abs(cm.astype(np.float64) - rm)
    return {"shots": len(obs), "tau": tau, "decisions_differ": int((cb != rb).any(1).sum()),
            "routing_changed": int((rr != rc).sum()), "routed": {"reference": int(rr.sum()), "candidate": int(rc.sum()),
                                                                 "only_candidate": int((rc & ~rr).sum()),
                                                                 "only_reference": int((rr & ~rc).sum())},
            "margin_absdiff": {"max": float(d.max()), "p99.9": float(np.quantile(d, 0.999)),
                               "median": float(np.median(d))},
            "net_failures": {"reference": int(nr.sum()), "candidate": int(nc.sum()), **mcnemar(nc, nr)},
            "hybrid_failures": {"reference": int(hr.sum()), "candidate": int(hc.sum()), **mcnemar(hc, hr)},
            "fallback_failures": int(ff.sum())}


def _engine(a):
    dev = torch.device("cuda")
    model, ck = engine.load_net(a.ckpt, dev)
    sha = engine.sha256_file(a.ckpt)
    t = time.time()
    eng = Fast(model, a.kind, sha, a.cache)
    return model, ck, sha, eng, time.time() - t


def eq(a):
    model, ck, sha, eng, build_s = _engine(a)
    code, p = ck["cfg"]["code"], ck["cfg"]["p"]
    sig = eng.signature()
    meta = {"ckpt": os.path.abspath(a.ckpt), "run": ck["hp"]["run"], "run_id": ck["hp"]["run_id"],
            "samples": ck.get("samples"), "status": ck.get("status")}
    tag = f"{ck['hp']['run']}_{a.kind}_{a.label}"
    chunks = shots.parse_range(a.chunks)
    os.makedirs(opp.PREDS, exist_ok=True)
    t, O = time.time(), []
    for j in chunks:
        det, obs = shots.load(code, p, "cal", j)
        bits, margin = eng.run(det)
        np.savez(route.net_path(code, p, "cal", j, tag), pred=bits, margin=margin, fail=(bits != obs).any(1),
                 sig=json.dumps(sig, sort_keys=True), meta=json.dumps(meta))
        O.append(obs)
    read_s = time.time() - t
    obs = np.concatenate(O)
    cb, cm, _, _ = route._load_net(code, p, "cal", chunks, tag)
    rb, rm, rsig, _ = route._load_net(code, p, "cal", chunks, a.ref)
    fb, _ = opp.load_preds(code, p, "cal", chunks, a.fallback)
    out = {"kind": a.kind, "tag": tag, "chunks": a.chunks, "engine": sig, "binding": sig_digest(sig),
           "reference": {"tag": a.ref, "engine": rsig}, "fallback": a.fallback, "build_s": build_s, "read_s": read_s,
           **compare(obs, rb, rm, cb, cm, fb, a.tau)}
    print("eq", json.dumps(out), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def lat(a):
    model, ck, sha, eng, build_s = _engine(a)
    stop = gpu_watch()
    reps = eng.time_calls()
    ho = eng.handoff_us()                                  # inside the watched window too
    acc = stop()
    sig = eng.signature()
    base = {"device": sig["gpu"], "code": ck["cfg"]["code"], "p": ck["cfg"]["p"], "name": f"{a.kind}:{sha[:12]}",
            "proc": a.proc, "binding": sig_digest(sig)}
    append_row(a.out, dict(base, role="gpu_term", n=500, value_ms=float(np.mean(reps)) / 1e3,
                           note=json.dumps({"rep_us": reps, "params": net.n_params(model), "build_s": build_s, **acc})))
    append_row(a.out, dict(base, role="handoff", n=1000, value_ms=ho / 1e3, note=json.dumps(acc)))
    print("latency", json.dumps({"engine": sig, "rep_us": reps, "mean_us": float(np.mean(reps)), "handoff_us": ho,
                                 **acc}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eq", "lat"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", default="fused", choices=KINDS)
    ap.add_argument("--cache", default="", help="trt16: directory for the ONNX export")
    ap.add_argument("--chunks", default="0-9")
    ap.add_argument("--ref", default="g2_D12k113c10", help="the reference read's tag (the frozen graph32 H100 read)")
    ap.add_argument("--fallback", default="relay")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--label", default="", help="eq: the read's tag is RUN_KIND_LABEL (e.g. the GPU)")
    ap.add_argument("--proc", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    eq(a) if a.cmd == "eq" else lat(a)


if __name__ == "__main__":
    main()
