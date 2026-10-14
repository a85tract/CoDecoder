"""CASCADE-TRT (exploratory, not registered): shared pieces. The Cascade networks' evaluation computation as an exportable
module, the TensorRT engine behind the timing harnesses' graph interfaces, and the loaders.

  bb   C14p (bb.cascade_net, 14 blocks, width 512) as pc_speed.make_fwd computes it:
       det (B, 432) uint8 -> bits (B, 12) (z > 0), conf (B,) fp32 (min |z|)
  d13  CS13 (surface.surface_net, 13 blocks, width 256) as surface.sv_common.b_fn computes it:
       det (B, Dd) uint8 -> logit (B,) fp32, dec (B,) (z > 0)

The exported graph is the module's own forward (embedding, blocks, read-out) in fp32 with the token step written with
ONNX-friendly ops (the same values) and, for BB72, the bipartite convolution on its PyTorch path (gather + GEMM, the
arithmetic the Triton kernels implement); TensorRT gets the BF16 flag and picks its kernels."""
import json
import os
import time



OUT_NAMES = {"bb": ("bits", "conf"), "d13": ("logit", "dec")}


def load_net(net, dev, ckpt=None):
    """The Cascade network as the harness loads it, in eval mode on dev -> (module, sha256 of the file read)."""
    if net == "bb":
        from bb.cascade_eval import Predictor
        pr = Predictor(ckpt, dev, "ema", "bf16")
        return pr.model, pr.sig["sha256"]
    from surface import sv_common as sc
    m, rec = sc.load_B(dev)
    return m, rec["sha256"]


def export_module(net, model):
    import torch
    import torch.nn.functional as F

    if net == "bb":
        from bb import cascade_net as cnet

        def gather_mm_plain(x, w, idx, inv):              # _GatherMM.forward without the autograd.Function wrapper
            B, N, C = x.shape
            xp = torch.cat([x, x.new_zeros(B, 1, C)], 1)
            return xp.index_select(1, idx.reshape(-1)).view(B, idx.shape[0], -1) @ w.t()
        cnet.gather_mm = gather_mm_plain
        for blk in model.blocks:
            blk.conv.impl = "torch"

    class Export(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("gi", m.gidx.clone())
            self.register_buffer("tb", m.tbase.float())

        def forward(self, det):
            x = det.float()
            x = torch.cat([x, torch.zeros_like(x[:, :1])], 1)            # the pad column (index D): bit 0
            tok = (torch.index_select(x, 1, self.gi) + self.tb).to(torch.int32)   # tbase + bit, as m.tokens
            z = self.m.readout(self.m.backbone(tok))
            if net == "bb":
                return z > 0, z.abs().amin(1)
            z = z[:, 0]
            return z, z > 0

    return Export(model).eval()


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_TD = None


def _torch_dtype(trt, dt):
    import torch
    global _TD
    if _TD is None:
        _TD = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.INT32: torch.int32,
               trt.DataType.INT8: torch.int8, trt.DataType.BOOL: torch.bool, trt.DataType.UINT8: torch.uint8}
        for name, tt in (("BF16", torch.bfloat16), ("INT64", torch.int64)):
            if hasattr(trt.DataType, name):
                _TD[getattr(trt.DataType, name)] = tt
    return _TD[dt]


class TRTEngine:
    """A TensorRT plan with its I/O tensors as static torch tensors on dev; the enqueue captured in a CUDA graph when
    possible (graph=True), else enqueued directly on the current stream. replay() runs one batch."""

    def __init__(self, plan, dev, graph=True):
        import tensorrt as trt
        import torch
        self.trt_version = trt.__version__
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.rt = trt.Runtime(self.logger)
        with open(plan, "rb") as fh:
            self.engine = self.rt.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise SystemExit(f"{plan}: TensorRT could not deserialize the plan")
        self.ctx = self.engine.create_execution_context()
        self.t = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            self.t[n] = torch.zeros(shape, dtype=_torch_dtype(trt, self.engine.get_tensor_dtype(n)), device=dev)
            if not self.ctx.set_tensor_address(n, self.t[n].data_ptr()):
                raise SystemExit(f"{plan}: set_tensor_address({n}) failed")
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):                              # deferred initialisation outside any capture
                if not self.ctx.execute_async_v3(st.cuda_stream):
                    raise SystemExit(f"{plan}: execute_async_v3 failed")
        st.synchronize()
        torch.cuda.current_stream().wait_stream(st)
        self.g, self.graph_error, self.capture = None, None, os.environ.get("CTRT_CAPTURE", "torch")
        self.gx = None
        if graph:
            try:
                if self.capture == "cudart":                # the CUDA runtime's own capture (TensorRT's samples)
                    self._capture_cudart()
                else:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=torch.cuda.Stream(), capture_error_mode="thread_local"):
                        ok = self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                    if not ok:
                        raise RuntimeError("execute_async_v3 returned False under capture")
                    g.replay()
                    torch.cuda.synchronize()
                    self.g = g
            except Exception as e:                          # noqa: BLE001  recorded in every row
                self.graph_error = repr(e)[:300]
                self.g = self.gx = None
                torch.cuda.synchronize()

    def _capture_cudart(self):
        """cudaStreamBeginCapture / enqueueV3 / cudaStreamEndCapture / cudaGraphInstantiate on a stream of our own, no
        torch graph machinery (torch.cuda.graph synchronises, empties the caching allocator's cache and clears cuBLAS
        workspaces on entry)."""
        import torch
        from cuda.bindings import runtime as cr

        def ck(r):
            err = r[0] if isinstance(r, tuple) else r
            if err != cr.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudart: {err}")
            return r[1] if isinstance(r, tuple) and len(r) == 2 else r
        self._st = torch.cuda.Stream()
        s = self._st.cuda_stream
        ck(cr.cudaStreamBeginCapture(s, cr.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
        ok = self.ctx.execute_async_v3(s)
        graph = ck(cr.cudaStreamEndCapture(s))
        if not ok:
            raise RuntimeError("execute_async_v3 returned False under capture")
        self._cr, self._graph = cr, graph
        self.gx = ck(cr.cudaGraphInstantiate(graph, 0))
        ck(cr.cudaGraphLaunch(self.gx, s))
        ck(cr.cudaStreamSynchronize(s))

    def replay(self):
        import torch
        if self.g is not None:
            self.g.replay()
        elif self.gx is not None:
            self._cr.cudaGraphLaunch(self.gx, torch.cuda.current_stream().cuda_stream)
        else:
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    def info(self):
        return {"trt": self.trt_version, "cuda_graph": self.g is not None or self.gx is not None,
                "capture": self.capture, "graph_error": self.graph_error}


class TRTGraphBB(TRTEngine):
    """pc_speed.Graph's interface (x, bits, conf, B, replay, run) on a TensorRT plan of the BB72 Cascade."""

    def __init__(self, plan, B, D, dev, graph=True):
        import torch
        super().__init__(plan, dev, graph)
        self.B = B
        self.x = self.t["det"]
        if tuple(self.x.shape) != (B, D):
            raise SystemExit(f"{plan}: input {tuple(self.x.shape)}, not ({B}, {D})")
        b = self.t["bits"]
        self.bits = b.view(torch.uint8) if b.dtype != torch.uint8 else b     # bool and uint8: one byte each
        self.conf = self.t["conf"]

    def run(self, det):
        import torch
        n, out_b, out_c = len(det), [], []
        for i in range(0, n, self.B):
            d = det[i:i + self.B]
            self.x.zero_()
            self.x[:len(d)].copy_(d)
            self.replay()
            out_b.append(self.bits[:len(d)].clone())
            out_c.append(self.conf[:len(d)].clone())
        return torch.cat(out_b), torch.cat(out_c)


class TRTGraphD13(TRTEngine):
    """surface.sv_common.Graph's interface (x, logit, dec, n, B, calls, replay, run_rows) on a TensorRT plan of CS13."""

    def __init__(self, plan, B, Dd, dev, graph=True):
        import torch
        super().__init__(plan, dev, graph)
        self.B, self.calls, self.n = B, 1, B
        self.x = self.t["det"]
        if tuple(self.x.shape) != (B, Dd):
            raise SystemExit(f"{plan}: input {tuple(self.x.shape)}, not ({B}, {Dd})")
        self.logit = self.t["logit"]
        d = self.t["dec"]
        self.dec = d.view(torch.uint8) if d.dtype != torch.uint8 else d

    def run_rows(self, rows):
        m = len(rows)
        if m < self.n:
            self.x.zero_()
        self.x[:m].copy_(rows)
        self.replay()
        return self.logit[:m].float().cpu().numpy(), self.dec[:m].cpu().numpy()


def plan_path(root, net, B):
    return os.path.join(root, f"{net}_b{B}.plan")


def emit(path, row):
    row = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), **row}
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
