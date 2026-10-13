"""The deployed inference path of ConvLines-BB (bb/PLAN.md 2.4): one fixed computation per network, used for every
calibration, dev and test read and for the batch-1 timing, so the margins that set tau are the margins the deployed
decoder produces (bb/DESIGN.md 5.3). Each shot runs alone (batch 1, so no padding) through one captured CUDA graph:
  graph     the PyTorch module: bf16 autocast for the trunk and the read-out, the vote and the margin in fp32
  graph32   the same graph entirely in fp32 (TF32 off): the rung for a network whose bf16 rungs fail validation
  inductor  the same computation compiled by torch.compile (default mode: heuristic kernels, no autotuning)
  fused     the engine port (PLAN 2.4, rung 2): the same network in a few Triton kernels (bb.fused; its precision is
            fused.PRECISION and its source's sha256 is part of the signature)
  fp32      the eager module in fp32 (batches of 2,048): the validation reference, never deployed
Fixed settings: cuDNN deterministic without benchmarking, TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8. signature() names
everything a read depends on (kind, batch, precision, GPU model, software versions, the network's and this code's
sha256); bb.route refuses to mix reads whose signatures differ."""
import hashlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bb import net  # noqa: E402

KINDS = ("graph", "graph32", "inductor", "fused", "fp32")
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_net(path, dev, allow_incomplete=False):
    """(model in eval mode, checkpoint dict) of a final_ema.pt; anything but a completed run needs allow_incomplete."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    if not ck.get("completed") and not allow_incomplete:
        raise SystemExit(f"{path}: not a completed run's network (status {ck.get('status', 'stage-1 checkpoint')})")
    model = net.build(ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["state"])
    return model, ck


class Deployed:
    def __init__(self, model, kind="graph", net_sha256=""):
        if kind not in KINDS:
            raise ValueError(kind)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model, self.kind, self.net_sha256 = model.eval(), kind, net_sha256
        self.dev = next(model.parameters()).device
        if kind == "fp32":
            return
        if kind == "fused":
            from bb import fused
            fn = self.fused = fused.Engine(model)          # kept: the captured graph uses its buffers and tables
        else:
            fn = torch.compile(self._call, dynamic=False) if kind == "inductor" else self._call
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

    def _call(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.kind != "graph32"):
            s, _, _ = self.model(x)
        _, bits, mg = self.model.decide(s)
        return bits, mg.float()

    @torch.no_grad()
    def run(self, det):
        """det (n, D) uint8 -> (bits (n, K) uint8, margin (n,) float32) as numpy."""
        n = len(det)
        X = torch.as_tensor(np.ascontiguousarray(det), device=self.dev)
        if self.kind == "fp32":
            out = [self.model.decide(self.model(X[i:i + 2048])[0]) for i in range(0, n, 2048)]
            return torch.cat([o[1] for o in out]).cpu().numpy(), torch.cat([o[2] for o in out]).float().cpu().numpy()
        B = torch.empty((n, self.model.K), dtype=torch.uint8, device=self.dev)
        M = torch.empty(n, dtype=torch.float32, device=self.dev)
        for i in range(n):
            self.x.copy_(X[i:i + 1])
            self.g.replay()
            B[i:i + 1].copy_(self.bits)
            M[i:i + 1].copy_(self.margin)
        return B.cpu().numpy(), M.cpu().numpy()

    def time_calls(self, warm=200, reps=5, calls=100):
        """SETTINGS 5.1's replay protocol on the captured graph: warm-up replays, then reps x calls timed replays (CUDA
        events) -> the mean of each repetition in us."""
        for _ in range(warm):
            self.g.replay()
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(calls):
                self.g.replay()
            e1.record()
            e1.synchronize()
            out.append(1e3 * e0.elapsed_time(e1) / calls)
        return out

    def handoff_us(self, n=1000):
        """The measured part of DESIGN 5.3 (c): mean host wall time of one shot's transfers around the graph (pinned
        host -> device copy of the detector bits, device -> pinned host copy of the decision and margin, then a
        synchronisation), without the graph itself; 50 untimed rounds first."""
        import time
        hx = torch.zeros(1, self.model.D, dtype=torch.uint8).pin_memory()
        hb = torch.empty(1, self.model.K, dtype=torch.uint8).pin_memory()
        hm = torch.empty(1, dtype=torch.float32).pin_memory()
        ts = []
        torch.cuda.synchronize()
        for i in range(n + 50):
            t = time.perf_counter()
            self.x.copy_(hx, non_blocking=True)
            hb.copy_(self.bits, non_blocking=True)
            hm.copy_(self.margin, non_blocking=True)
            torch.cuda.synchronize()
            if i >= 50:
                ts.append(1e6 * (time.perf_counter() - t))
        return float(np.mean(ts))

    def signature(self):
        sig = {"kind": self.kind, "batch": 2048 if self.kind == "fp32" else 1,
               "precision": "fp32" if self.kind in ("fp32", "graph32") else "bf16 autocast, fp32 vote and margin",
               "gpu": torch.cuda.get_device_name(self.dev), "torch": torch.__version__, "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "net_sha256": self.net_sha256,
               "src": {f: sha256_file(os.path.join(HERE, f))[:16] for f in ("engine.py", "net.py")}}
        if self.kind in ("inductor", "fused"):
            import triton
            sig["triton"] = triton.__version__
        if self.kind == "fused":
            from bb import fused
            sig["precision"] = fused.PRECISION
            sig["src"]["fused.py"] = sha256_file(os.path.join(HERE, "fused.py"))[:16]
        return sig
