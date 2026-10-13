"""Cascade-surface's evaluation at d = 13 (cascade_compare/SURFACE_PLAN.md sections 4, 5, 7, 9 and 10, with the items of
the independent check folded in; surface/EVAL_READY.md): the fixed inputs, the decoders' computations, the records every
output carries, and the statistics. sv_read.py (the streamed reads), sv_speed.py (screen, pipelines, bitwise check) and
sv_route.py (rule, lock, confirmation analysis, report) import it.

Decoders (SURFACE_PLAN 4.1). S = ConvLines d13_b5_r775 on the release's deployed Triton engine (timing/engine.py,
FROZEN; fp16 matmul inputs, fp32 accumulation): confidence |c|, decision c > 0. B = Cascade-surface, the EMA network of
CS13 (final_ema.pt) in eval mode under bf16 autocast, on the path "graph" (the module in a CUDA graph) or "compiled"
(torch.compile in its default mode, then a CUDA graph): confidence |z|, decision z > 0. M = correlated PyMatching 2.4.0
through the release's matcher, decode (one shot per call) or decode_batch. A shot goes to the next stage iff its float32
confidence, compared in float64, lies strictly below the threshold.

Computations (SURFACE_PLAN 4.4; item 9 of the check: the chains' queue runs at B's amortized batch b*, as Bb does):
  S1 S at batch 1 | SB S at 4,096 | B1 B at batch 1 | Bb B at b* | BQ1, BQ5 B at b* on the queue of the shots SB routes
  under C1's (C5's) tau1 | MS M's decode | MB M's decode_batch.
GPU steps run on a full RTX PRO 6000 (GPU) with TF32 off and cuDNN's autotuning off; the kernel caches are the job's
TRITON_CACHE_DIR and TORCHINDUCTOR_CACHE_DIR, one directory for the whole evaluation, whose digest every record carries
(pinned caches, item 8). Every record also carries the environment's identity: the resolved interpreter, the venv and a
hash of its package list (env(), item 10), and the code's (code(): the deployed commit and a digest of every file of the
evaluation code and the release, finding 3 of the stage gate of 2026-10-12). Outputs are published once and never
overwritten (publish(), finding 7)."""
import hashlib
import io
import json
import math
import os
import platform
import sys
import time

import numpy as np

from surface import surface_data as sdata
from surface import surface_layout as sl

R = os.environ.get("CASCADE_ROOT", "runs/cascade_d13")      # the run directory: ckpt/CS13/final_ema.pt and the outputs
CKPT = {"final": "ckpt/CS13/final_ema.pt", "budget": "ckpt/CS13/budget_ema.pt"}          # under R
CKPT_SHA256 = {"final": "5351b27b449145ae6c461260b6320925cebba56d2b0449f5a23884d9f787a659",
               "budget": "8ef1c794eeed1625902d8644f92add167afc49c9050a8b87543f31bc23cda4d0"}
S_NAME = "d13_b5_r775"
S_SHA256 = "cbd2b84a47dd6bf71c62eb0cae4d660e2811db5efb9b55b695dd9337d141f12a"
D = 13
P_HIGH, P_LOW = 0.006, 0.0035
PS = (P_HIGH, P_LOW)
CHUNK, PART = 10_000, 100                 # shots per chunk; chunks per output file
KIND_MONITOR, KIND_DEV, KIND_CONF, KIND_SPEED, KIND_DRY = 1, 10, 11, 12, 13
DEV_CHUNKS = {P_HIGH: 100, P_LOW: 1000}   # development blocks: chunks 0 .. n-1 of kind 10
CONF_CAP = {P_HIGH: 400, P_LOW: 4000}     # the confirmation's cap in chunks (4e6 and 4e7 shots)
GRID = (10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000)  # phase C's grid, 3,000 and 4,000
Q1 = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60)
R2 = (0.001, 0.0025, 0.005, 0.01, 0.02)
TAU_C4 = 1.8486117124557495               # the registered routing of S to correlated PM (C4)
SPEED_MARGIN, W_M = 0.9, 16               # feasibility: predicted <= 0.9 x Cascade alone; M's workers when batched
DELTA_REL, POWER, ALPHA, M_FAMILY = 0.10, 0.90, 0.05, 40
SEED_POWER, REPS_POWER = 20261006, 4000   # phase C's bb.route.power: fixed seed; size_comparison's repetitions
BATCHES = (1, 16, 64, 256, 1024, 4096)
S_BATCH = 4096                            # S's batched path (SB) and the batched pipelines' S
REPS = 3
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
COMPS = ("S1", "SB", "B1", "Bb", "BQ1", "BQ5", "MS", "MB", "MP")     # MP: plain PM, context only
CONFIGS = ("C1", "C2", "C3", "C4", "C5")
PIPE_CHUNKS = (0, 50)                     # kind 12 at each p: the pipelines' stream (single mode: its first 20,000)
GATE_CHUNKS = (200, 250)                  # kind 12: the bitwise gate before the lock (item 8)
PIPE_CONFIGS = ("C1", "C2", "C4", "C5")   # the configurations each pipeline stage runs at both p in both modes
PIPES = {"gate": {"chunks": GATE_CHUNKS, "single": 20_000, "batched": 500_000, "reps": (1,)},
         "c7": {"chunks": PIPE_CHUNKS, "single": 20_000, "batched": 500_000, "reps": (1, 2, 3)}}
DRY_PIPES = {"gate": {"chunks": (200, 205), "single": 2_000, "batched": 50_000, "reps": (1,)},
             "c7": {"chunks": (0, 5), "single": 2_000, "batched": 50_000, "reps": (1, 2, 3)}}
SCREEN_CHUNKS = (1000, 1010)              # kind 12: the speed screen's agreement shots


# ---------------------------------------------------------------------------------------------------------- records

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def digest(det, obs):
    """A chunk's shots: sha256 of the bit-packed detector rows and the observable bytes (16 hex digits)."""
    h = hashlib.sha256(np.packbits(np.asarray(det, np.uint8), axis=1).tobytes())
    h.update(np.asarray(obs, np.uint8).reshape(-1).tobytes())
    return h.hexdigest()[:16]


def env():
    """The environment's identity (item 10): the interpreter as invoked and resolved, the venv (sys.prefix), where torch
    was loaded from, and the sha256 of the sorted name==version list of every installed distribution. Refused when
    torch does not come from the venv itself (the pilot's torchrun ran another venv's interpreter)."""
    import importlib.metadata as md
    import torch
    pk = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in md.distributions()})
    out = {"python": platform.python_version(), "executable": sys.executable,
           "real_executable": os.path.realpath(sys.executable), "prefix": sys.prefix,
           "torch_file": os.path.realpath(torch.__file__), "packages": len(pk),
           "packages_sha256": hashlib.sha256("\n".join(pk).encode()).hexdigest(),
           "container": os.environ.get("APPTAINER_CONTAINER", os.environ.get("SINGULARITY_CONTAINER", "")),
           "venv_staged": os.environ.get("SV_VENV_STAGED", "")}    # eval/jobs/common.sh: a local copy at the same path
    if not out["torch_file"].startswith(os.path.realpath(sys.prefix) + os.sep):
        raise SystemExit(f"torch from {out['torch_file']}, not from the venv {sys.prefix}")
    return out


def code(root=None, release=None):
    """The code's identity (finding 3 of the stage gate): the deployed tree's DEPLOYED line (commit and release commit,
    each with '+dirty' when its checkout had uncommitted changes) and a digest over every file of the code that runs,
    by relative path and content sha256: surface/'s *.py, *.sh and *.sbatch (job files included) in the tree this module
    was imported from, and every file of the release directory the loaders read (CONVLINES_RELEASE). Kernel caches say
    nothing about the Python code; this does."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    release = os.path.realpath(release or sl.release_root())
    lines = []
    for tag, base, ext in (("surface", os.path.join(root, "surface"), (".py", ".sh", ".sbatch")),
                           ("release", release, "")):
        for dp, dns, fs in os.walk(base):
            dns[:] = [x for x in dns if x != "__pycache__"]
            for f in fs:
                if f.endswith(ext):
                    p = os.path.join(dp, f)
                    lines.append(f"{tag}/{os.path.relpath(p, base)} {sha256_file(p)}")
    dep = os.path.join(root, "DEPLOYED")
    tag = open(dep).read().split() if os.path.isfile(dep) else []
    return {"commit": tag[0] if tag else "", "release": tag[2] if len(tag) > 2 else "",
            "clean": len(tag) > 2 and not any(t.endswith("+dirty") for t in tag), "files": len(lines),
            "digest": hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:16]}


IDENTITY = ("gpu", "torch", "real_executable", "prefix", "packages_sha256", "code")   # what a read shares with the lock
ANALYSIS = ("real_executable", "prefix", "packages_sha256", "code")                   # and an analysis without a GPU


def kcache():
    """Digest of the kernel caches (TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR): every file's relative path and size,
    and the content of autotuning results (*best_config*, *.json under 64 kB). A process that compiles or tunes
    anything new changes it."""
    h, n = hashlib.sha256(), 0
    for var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        root = os.environ.get(var, "")
        h.update(f"{var}={root}\n".encode())
        if not root or not os.path.isdir(root):
            continue
        for dp, _, fs in sorted(os.walk(root)):
            for f in sorted(fs):
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(p, root)} {st.st_size}\n".encode())
                if ("best_config" in f or f.endswith(".json")) and st.st_size < 65536:
                    with open(p, "rb") as fh:
                        h.update(fh.read())
                n += 1
    return {"digest": h.hexdigest()[:16], "files": n}


def setup_gpu():
    """The device, refused unless it is a full RTX PRO 6000 (a MIG slice's name differs), with TF32 off and cuDNN's
    autotuning off (SURFACE_PLAN 4.4, 5), and its record: card, driver (the job exports SV_GPU_DRIVER), torch, CUDA,
    cuDNN, Triton."""
    import torch
    import triton
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(dev)
    if name != GPU and not (os.environ.get("SV_DEBUG_MIG") and name.startswith(GPU)):
        raise SystemExit(f"GPU {name!r}: every GPU step of the evaluation runs on a full {GPU}")
    # SV_DEBUG_MIG: code checks on a MIG slice of the same card; its name differs from GPU, so no lock, read or
    # timing of the evaluation accepts its records (IDENTITY)
    return dev, {"gpu": name, "driver": os.environ.get("SV_GPU_DRIVER", ""), "torch": torch.__version__,
                 "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "triton": triton.__version__,
                 "capability": list(torch.cuda.get_device_capability(dev))}


def tree():
    """The deployed tree (its DEPLOYED file: commit and release) the job runs."""
    for d in os.environ.get("PYTHONPATH", "").split(os.pathsep) + [os.getcwd()]:
        f = os.path.join(d, "DEPLOYED")
        if d and os.path.isfile(f):
            return open(f).read().strip()
    return ""


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            return next((ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")), "")
    except OSError:
        return ""


def record(gpu=None):
    """What every output carries: the environment's identity, the card, the kernel caches, the tree, job, host and the
    host's CPU (M's decodes and t_M run on it)."""
    r = {**env(), **(gpu or {}), "code": code(), "kcache": kcache(), "tree": tree(),
         "job": os.environ.get("SLURM_JOB_ID", ""), "host": platform.node(), "pid": os.getpid(), "cpu": cpu_model(),
         "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return r


def same_identity(a, b, what, keys=IDENTITY):
    """Refuse a record whose GPU model, torch, venv or code differs from the reference's (the lock's)."""
    bad = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
    if bad:
        raise SystemExit(f"{what}: environment or code differs from the reference: {bad}")


def check_analysis(ref, what):
    """An analysis (summarize, select, bitwise, gate, confirm, report) runs only under the reference's venv and code
    (finding 3 of the stage gate)."""
    same_identity({**env(), "code": code()}, ref, what, ANALYSIS)


def publish(path, data):
    """data (bytes) as the file path, published once (finding 7 of the stage gate): written and synced under a name of
    this process's own, then hard-linked to path, which fails when path exists. Concurrent writers can neither
    truncate each other's file nor replace a published one, and a reader never sees a partial file.
    -> True if this call published path, False if it existed already."""
    tmp = f"{path}.tmp.{platform.node()}.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.unlink(tmp)


def save_npz(path, meta, shared=False, **arrays):
    """Arrays and a JSON record in one npz, published once (publish) -> its sha256. An existing file is refused, or
    with shared=True (an observable file, which every read of the same chunks writes) left for the caller to compare:
    None."""
    buf = io.BytesIO()
    np.savez(buf, meta=np.array(json.dumps(meta)), **arrays)
    if not publish(path, buf.getvalue()):
        if shared:
            return None
        raise SystemExit(f"{path} exists: an output is never overwritten")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def write_json(path, obj):
    """A JSON output (speed summary, lock, bitwise check, gate file, confirmation, read summary), published once."""
    if not publish(path, json.dumps(obj, indent=1, default=list).encode()):
        raise SystemExit(f"{path} exists: an output is never overwritten")


def load_npz(path, sha=False):
    """-> (record, arrays), and the file's sha256 with sha=True (from the same bytes)."""
    raw = open(path, "rb").read()
    z = np.load(io.BytesIO(raw))
    out = json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}
    return (*out, hashlib.sha256(raw).hexdigest()) if sha else out


def committed_pair(lock_path, gate_path, receipt=""):
    """The committed lock and gate (finding 2 of the stage gate), required before any confirmation shot is drawn and by
    confirm: the gate file passed for this lock, and scripts/lockpair.sh's receipt (lock_commit.json beside the lock,
    written only after it committed both files, pushed the commit to the hub and found the hub's branch at that commit
    with these bytes) names both files' sha256 -> (gate, receipt)."""
    lock_sha, gate_sha = sha256_file(lock_path), sha256_file(gate_path)
    G = json.load(open(gate_path))
    if G.get("lock_sha256") != lock_sha or not G.get("all_pass"):
        raise SystemExit(f"{gate_path}: no passed bitwise gate for this lock (item 8)")
    rp = receipt or os.path.join(os.path.dirname(os.path.abspath(lock_path)), "lock_commit.json")
    if not os.path.isfile(rp):
        raise SystemExit(f"{rp} missing: commit and push the lock and gate first (scripts/lockpair.sh)")
    Rc = json.load(open(rp))
    f = Rc.get("files", {})
    if f.get("locked_d13.json", {}).get("sha256") != lock_sha or f.get("gate_d13.json", {}).get("sha256") != gate_sha:
        raise SystemExit(f"{rp}: the committed lock and gate are other files than {lock_path} and {gate_path}")
    return G, Rc


def completed(base, kc):
    """The files that the reads in base list (read_*.json) as theirs, from reads that ended complete with the kernel
    caches still at digest kc (finding 2: checked again at completion) -> {name: sha256}."""
    ok = {}
    for f in sorted(os.listdir(base)):
        if not (f.startswith("read_") and f.endswith(".json")):
            continue
        s = json.load(open(os.path.join(base, f)))
        if s.get("complete") and s.get("kcache_end", {}).get("digest") == kc:
            for name, sha in s["files"].items():
                if ok.setdefault(name, sha) != sha:
                    raise SystemExit(f"{base}: two complete reads list {name} with different sha256")
    return ok


def p_tag(p):
    return f"p{p:.4f}"


def chunk_range(s):
    """'a-b' -> range(a, b) (b exclusive)."""
    a, b = (int(x) for x in s.split("-"))
    return range(a, b)


def kind_of(stage, dry):
    """The seed kind of a stage; every dry run reads kind 13 only."""
    if dry:
        return KIND_DRY
    return {"monitor": KIND_MONITOR, "dev": KIND_DEV, "conf": KIND_CONF, "speed": KIND_SPEED}[stage]


# --------------------------------------------------------------------------------------------------------- decoders

def read_safetensors(path):
    """A safetensors file -> {name: CPU tensor}. bb's venv has no safetensors package; the format is an 8-byte
    little-endian header length, a JSON header (dtype, shape, byte offsets per tensor) and the raw little-endian data."""
    import torch
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    dt = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
          "BOOL": torch.bool}
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        t = torch.frombuffer(bytearray(raw[8 + n + a:8 + n + b]), dtype=dt[v["dtype"]]) if b > a else \
            torch.empty(0, dtype=dt[v["dtype"]])
        out[k] = t.reshape(v["shape"])
    return out


def load_S():
    """S: ConvLines d13_b5_r775 from the release (sha256 checked), its deploy tables folded on the CPU in fp32 as the
    release does (convlines.net.deploy_tables) -> (fp32 module, tables, config, record)."""
    import torch
    sl.release_root()
    from convlines import circuits as rc, decoder
    from convlines.net import ConvLines, deploy_tables
    cfg = decoder.config(S_NAME)
    path = os.path.join(rc.WEIGHTS, cfg["file"])
    sha = sha256_file(path)
    if sha != S_SHA256:
        raise SystemExit(f"{path}: sha256 {sha}, not {S_SHA256}")
    model = ConvLines(cfg["d"], os.path.join(rc.WEIGHTS, os.path.basename(cfg["grid"]["geometry"])), **cfg["arch"])
    model.load_state_dict(read_safetensors(path))
    model.eval()
    with torch.no_grad():
        t = deploy_tables(model)
    return model, t, cfg, {"net": "S", "name": S_NAME, "sha256": sha, "engine": "timing.engine FROZEN"}


def s_fn(t, cfg):
    """A new instance of S's engine (its own buffers) as fn(x (n, 2184) uint8 on the GPU) -> (c (n,) fp32, dec (n,))."""
    from timing import deployed
    eng = deployed.engine(deployed.to_cuda(t), cfg)

    def fn(x):
        c, _, dec = eng(x)
        return c, dec
    fn.engine = eng
    return fn


def load_B(dev, which="final"):
    """B: CS13's EMA network (final_ema.pt for every read; budget_ema.pt only described) in eval mode on dev, the file
    checked by sha256 -> (module, record)."""
    import torch
    from surface import surface_net as snet
    path = os.path.join(R, CKPT[which])
    st = torch.load(path, map_location="cpu", weights_only=False)
    # The published file (weights/cascade_d13/CS13/final_ema.pt) is an inference copy of the checkpoint the comparison
    # read: the same tensors, with that checkpoint's sha256 as source_sha256.
    sha = st.get("source_sha256") or sha256_file(path)
    if sha != CKPT_SHA256[which]:
        raise SystemExit(f"{path}: sha256 {sha}, not {CKPT_SHA256[which]}")
    m = snet.build(st["cfg"], sl.Layout(st["cfg"]["d"]))
    m.load_state_dict(st["ema"])
    m = m.to(dev).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, {"net": "B", "ckpt": which, "file": path, "sha256": sha, "step": st["step"], "status": st["status"],
               "cfg": st["cfg"], "params": sum(q.numel() for q in m.parameters())}


def b_fn(model, path):
    """B's computation on a path ("graph": the module; "compiled": torch.compile in its default mode) as fn(x (n, D)
    uint8) -> (z (n,) fp32, dec (n,) uint8) under bf16 autocast. z is the bf16 logit in fp32."""
    import torch

    def fwd(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x).float()[:, 0]
        return z, (z > 0).to(torch.uint8)
    if path == "graph":
        return fwd
    if path == "compiled":
        return torch.compile(fwd, dynamic=False)
    raise ValueError(f"B's paths are graph and compiled, not {path!r}")


class Graph:
    """fn captured in one CUDA graph on a static input x of `calls` consecutive batches of B shots (three warm-up passes
    on a side stream first, so that compilation and Triton's JIT happen outside the capture). calls = 1 is a timed path
    (its outputs are fn's own tensors); calls > 1 serves the batch-1 reads (Bulk1): each call gets its own input tensor
    of shape (B, D) at offset 0, as the timed path's (copied from x inside the graph), so that every call runs the
    timed path's kernels with the same specialisations, and its outputs are copied into the graph's output slots."""

    def __init__(self, fn, B, Dd, dev, calls=1):
        import torch
        self.fn, self.B, self.calls, self.n = fn, B, calls, B * calls
        self.x = torch.zeros(self.n, Dd, dtype=torch.uint8, device=dev)
        xs = [self.x] if calls == 1 else [torch.zeros(B, Dd, dtype=torch.uint8, device=dev) for _ in range(calls)]
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(st):
            for _ in range(3):
                fn(xs[0])
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        if calls > 1:
            self.logit = torch.empty(self.n, dtype=torch.float32, device=dev)
            self.dec = torch.empty(self.n, dtype=torch.uint8, device=dev)
        self.g = torch.cuda.CUDAGraph()
        # each graph is captured on a stream of its own: graphs captured on one stream share cuBLAS's workspace (kept
        # per stream), and replaying such graphs concurrently corrupted B's batch-1 reads (Bulk1, 2026-10-12: 2,298 of
        # 4,096 logits differed from the timed path, 257 NaN; with a capture stream per graph none differed)
        with torch.no_grad(), torch.cuda.graph(self.g, stream=torch.cuda.Stream()):
            if calls == 1:
                self.logit, self.dec = fn(self.x)
            else:
                for k in range(calls):
                    xs[k].copy_(self.x[k * B:(k + 1) * B])
                    lg, dc = fn(xs[k])
                    self.logit[k * B:(k + 1) * B].copy_(lg)
                    self.dec[k * B:(k + 1) * B].copy_(dc)
        self.xs = xs

    def replay(self):
        self.g.replay()

    def run_rows(self, rows):
        """rows (m <= n, D) on the GPU -> (logit, dec) numpy for those rows, the rest of the input zero (padding)."""
        m = len(rows)
        if m < self.n:
            self.x.zero_()
        self.x[:m].copy_(rows)
        self.g.replay()
        return self.logit[:m].float().cpu().numpy(), self.dec[:m].cpu().numpy()


class Bulk1:
    """A batch-1 computation over many shots: G graphs of `calls` consecutive batch-1 calls, each with its own fn instance
    (make_fn(); S's engine buffers are per instance), each captured on its own stream (Graph), replayed round robin on G
    CUDA streams. Per shot the kernels are those of the timed batch-1 path; sv_read's dry run and the bitwise gate check
    that the outputs are equal."""

    def __init__(self, make_fn, Dd, dev, calls=32, G=4):
        import torch
        self.calls, self.G = calls, G
        self.gs = [Graph(make_fn(), 1, Dd, dev, calls) for _ in range(G)]
        self.st = [torch.cuda.Stream() for _ in range(G)]
        self.hin = [torch.zeros(calls, Dd, dtype=torch.uint8).pin_memory() for _ in range(G)]
        self.hl = [torch.empty(calls, dtype=torch.float32).pin_memory() for _ in range(G)]
        self.hd = [torch.empty(calls, dtype=torch.uint8).pin_memory() for _ in range(G)]

    def run(self, det):
        """det (n, D) uint8 numpy -> (logit (n,) fp32, dec (n,) uint8) numpy."""
        import torch
        n, K = len(det), self.calls
        lg, dc = np.empty(n, np.float32), np.empty(n, np.uint8)
        pend = [None] * self.G

        def harvest(s):
            ev, lo, hi = pend[s]
            ev.synchronize()
            lg[lo:hi] = self.hl[s][:hi - lo].numpy()
            dc[lo:hi] = self.hd[s][:hi - lo].numpy()
            pend[s] = None
        for gi, lo in enumerate(range(0, n, K)):
            s, hi = gi % self.G, min(lo + K, n)
            if pend[s] is not None:
                harvest(s)
            h = self.hin[s].numpy()
            h[:hi - lo] = det[lo:hi]
            if hi - lo < K:
                h[hi - lo:] = 0
            g = self.gs[s]
            with torch.cuda.stream(self.st[s]):
                g.x.copy_(self.hin[s], non_blocking=True)
                g.replay()
                self.hl[s].copy_(g.logit, non_blocking=True)
                self.hd[s].copy_(g.dec, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
            pend[s] = (ev, lo, hi)
        for s in range(self.G):
            if pend[s] is not None:
                harvest(s)
        return lg, dc


class Queue:
    """A path at batch B over a stream of rows in order (SB, Bb, BQ1, BQ5 and the batched pipelines): rows wait on the
    GPU; each full batch of B runs at once and the remainder at flush, zero-padded as the pipelines pad it.
    add / flush -> [(idx, logit, dec, rows, batch number)] in order; watermark(fed) = the first index not yet decided.
    A resumed read (sv_read --resume) sets `batches` to the number of the batch it rebuilds."""

    def __init__(self, graph):
        self.g, self.B = graph, graph.n
        self.rows, self.idx, self.n = [], [], 0
        self.batches = 0

    def add(self, rows, idx):
        if len(idx):
            self.rows.append(rows)
            self.idx.append(np.asarray(idx, np.int64))
            self.n += len(idx)
        out = []
        while self.n >= self.B:
            out.append(self._run(self.B))
        return out

    def flush(self):
        return [self._run(self.n)] if self.n else []

    def _run(self, m):
        import torch
        rows, idx = torch.cat(self.rows), np.concatenate(self.idx)
        lg, dc = self.g.run_rows(rows[:m])
        self.n -= m
        self.rows, self.idx = ([rows[m:]], [idx[m:]]) if self.n else ([], [])
        self.batches += 1
        return idx[:m], lg, dc, rows[:m], self.batches - 1

    def watermark(self, fed):
        return int(self.idx[0][0]) if self.n else fed


def m_matcher(p, correlated=True):
    """Correlated PyMatching at p through the release's matcher (DEM decomposed, approximate disjoint errors); plain
    PyMatching (the same DEM without correlations, context only) with correlated=False."""
    sl.release_root()
    from convlines import circuits
    return circuits.matcher(D, correlated, sl.circuit_at(D, p))


def m_single(m, det):
    """MS: decode one shot per call, each timed -> (pred (n,) uint8, us (n,) float32)."""
    pred, us = np.empty(len(det), np.uint8), np.empty(len(det), np.float32)
    for i in range(len(det)):
        t = time.perf_counter_ns()
        pred[i] = m.decode(det[i], enable_correlations=True)[0]
        us[i] = (time.perf_counter_ns() - t) / 1e3
    return pred, us


def m_batch(m, det, correlated=True):
    """MB: decode_batch -> pred (n,) uint8 (MP: a plain matcher, correlated=False)."""
    if not len(det):
        return np.zeros(0, np.uint8)
    out = m.decode_batch(det, enable_correlations=True) if correlated else m.decode_batch(det)
    return np.asarray(out, np.uint8).reshape(len(det), -1)[:, 0]


def shots(p, chunks, kind, procs=2):
    """The shots of chunks of `kind` at p, streamed (surface_data.stream_shots): (chunk, det, obs)."""
    return sdata.stream_shots(D, p, chunks, n=CHUNK, kind=kind, procs=procs)


def routed(conf, tau):
    """The routing rule: float32 confidence compared in float64, strictly below tau (a shot exactly at tau stays)."""
    return np.abs(np.asarray(conf, np.float32)).astype(np.float64) < tau


# ------------------------------------------------------------------------------------------------------- statistics

def wilson(f, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    ph = f / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def p_block_to_pl(pb, R=13):
    """The paper's Eq. (1) with k = 1 and R = 13: P_L = (1 - (1 - 2 P_block)^(1/R)) / 2 (rates only, never differences)."""
    x = 1 - 2 * pb
    return (1 - x ** (1 / R)) / 2 if x > 0 else float("nan")


def point(f, n):
    lo, hi = wilson(f, n)
    return {"shots": int(n), "fail": int(f), "p_block": f / n if n else float("nan"), "pb_lo": lo, "pb_hi": hi,
            "pl": p_block_to_pl(f / n) if n else float("nan"), "pl_lo": p_block_to_pl(lo), "pl_hi": p_block_to_pl(hi)}


def newcombe10(a, b, c, d, z):
    """Newcombe's matched-pair interval (method 10) for p1 - p2, p1 = (a + b) / n, p2 = (a + c) / n (a both fail, b only
    1, c only 2, d neither): phase C's qldpc_neural/bench/paired.py newcombe10 -> (diff, lo, hi)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if den == 0 else max(0.0, a * d - b * c) / math.sqrt(den)
    dt = p1 - p2
    lo = dt - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = dt + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return dt, lo, hi


def newcombe_hi(a, b, c, d, z):
    """newcombe10's upper end, vectorised over arrays (phase C's bb.route._newcombe_hi)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n

    def wil(k):
        p = k / n
        den = 1 + z * z / n
        ctr = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return ctr - h, ctr + h
    _, u1 = wil(a + b)
    l2, _ = wil(a + c)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = np.where(den > 0, np.maximum(0.0, a * d - b * c) / np.sqrt(np.maximum(den, 1)), 0.0)
    return p1 - p2 + np.sqrt(np.maximum(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))


def mcnemar_p(b, c):
    """The exact two-sided McNemar test on the discordant pairs (phase C's binomtest(b, b + c, 0.5))."""
    from scipy.stats import binomtest
    return float(binomtest(int(b), int(b + c), 0.5).pvalue) if b + c else 1.0


def ni_p(a, b, c, d, delta):
    """Non-inferiority of 1 against 2 at margin delta (H0: p1 - p2 >= delta): the one-sided level at which the upper end
    of newcombe10 equals delta, by bisection on z (the upper end rises with z), so that p < alpha exactly when the upper
    end at the one-sided level alpha (z = Phi^-1(1 - alpha)) lies below delta. 1.0 when the estimate is >= delta."""
    from scipy.stats import norm
    if newcombe10(a, b, c, d, 0.0)[2] >= delta:
        return 1.0
    lo, hi = 0.0, 1.0
    while newcombe10(a, b, c, d, hi)[2] < delta:
        hi *= 2
        if hi > 64:
            return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if newcombe10(a, b, c, d, mid)[2] < delta:
            lo = mid
        else:
            hi = mid
    return float(norm.sf(lo))


def holm(pvals, alpha=ALPHA):
    """Holm's step-down procedure: index -> the level at which that hypothesis is rejected, or None (not rejected; once
    a test fails, it and every later one are not rejected)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, m, go = {}, len(pvals), True
    for rank, i in enumerate(order):
        a = alpha / (m - rank)
        if not go or pvals[i] >= a:
            go = False
            out[i] = None
        else:
            out[i] = a
    return out


def counts(fx, fy):
    """Paired failures of X and Y on the same shots."""
    fx, fy = np.asarray(fx, bool), np.asarray(fy, bool)
    return {"both": int((fx & fy).sum()), "only_x": int((fx & ~fy).sum()), "only_y": int((~fx & fy).sum()),
            "shots": int(len(fx))}


def abcd(cnt):
    a, b, c = cnt["both"], cnt["only_x"], cnt["only_y"]
    return a, b, c, cnt["shots"] - a - b - c


def _draws(probs, n, reps=REPS_POWER, seed=SEED_POWER):
    X = np.random.default_rng(seed).multinomial(n, probs, size=reps)
    return (X[:, i].astype(np.float64) for i in range(4))


def ni_power(cnt, n, delta, alpha_one):
    """Planned power of the non-inferiority test (SURFACE_PLAN 4.4): a true difference of zero at the development
    block's discordant rate (each direction half of it, both-fail at its development rate), the decision 'newcombe10's
    upper end at the one-sided level alpha_one lies below delta', simulated with phase C's fixed seed."""
    from scipy.stats import norm
    a, b, c, d = abcd(cnt)
    N = cnt["shots"]
    dis = (b + c) / N
    probs = np.array([a / N, dis / 2, dis / 2, 1 - a / N - dis])
    A, Bb, Cc, Dd = _draws(probs, n)
    hi = newcombe_hi(A, Bb, Cc, Dd, float(norm.ppf(1 - alpha_one)))
    return float(np.mean(hi < delta))


def mcnemar_power(cnt, n, alpha):
    """Power of the exact two-sided McNemar test at alpha if the development block's paired rates were the truth."""
    from scipy.stats import binom
    a, b, c, d = abcd(cnt)
    N = cnt["shots"]
    if b + c == 0:
        return 0.0
    _, B_, C_, _ = _draws(np.array([a, b, c, d]) / N, n)
    m = B_ + C_
    pm = np.where(m > 0, np.minimum(1.0, 2 * binom.cdf(np.minimum(B_, C_), m, 0.5)), 1.0)
    return float(np.mean(pm < alpha))


# ------------------------------------------------------------------------------------- outputs: loading, assembling

def load_read(d, p, comps, chunks=None, want=None, complete=None):
    """A read's outputs at p (sv_read: d/p<p>/<comp>_<c0>-<c1>.npz and obs_*.npz) for the computations comps, over the
    chunk range `chunks` (a range; None = every part present, which must be contiguous): every part's digests must be
    the observable file's, so that every computation decided the same shots; parts that overlap the range are cut to it.
    BQ1 / BQ5 are expanded to every shot (z NaN and dec 255 where a shot was not queued). want: {key: value} each part's
    record must carry (identity, sha256, paths, thresholds); a mismatch is refused. complete: a kernel-cache digest;
    every file must then be one that a complete read listed with that sha256 and ended with those caches (completed).
    -> {"obs", "digests", "chunks", "meta": {comp: first part's record}, "jobs": every part's job,
    comp: {array: values}}."""
    base = os.path.join(d, p_tag(p))
    done = completed(base, complete) if complete is not None else None

    def get(name):
        if done is None:
            return load_npz(os.path.join(base, name))
        m, arr, sha = load_npz(os.path.join(base, name), sha=True)
        if done.get(name) != sha:
            raise SystemExit(f"{base}/{name}: not a file of a complete read whose kernel caches stayed at {complete}")
        return m, arr
    parts = sorted(f[4:-4] for f in os.listdir(base) if f.startswith("obs_") and f.endswith(".npz"))
    rng = [tuple(int(x) for x in s.split("-")) for s in parts]
    if chunks is not None:
        rng = [r for r in rng if r[0] < chunks.stop and chunks.start < r[1]]
    if not rng:
        raise SystemExit(f"{base}: no read parts for chunks {chunks}")
    for (_, e0), (s1, _) in zip(rng, rng[1:]):
        if e0 != s1:
            raise SystemExit(f"{base}: the parts are not contiguous ({e0} -> {s1})")
    lo, hi = (chunks.start, chunks.stop) if chunks is not None else (rng[0][0], rng[-1][1])
    if rng[0][0] > lo or rng[-1][1] < hi:
        raise SystemExit(f"{base}: parts cover {rng[0][0]}-{rng[-1][1]}, not {lo}-{hi}")
    cut = slice((lo - rng[0][0]) * CHUNK, (hi - rng[0][0]) * CHUNK)
    out = {"obs": [], "digests": [], "chunks": (lo, hi), "meta": {}, "jobs": set()}
    acc = {c: {} for c in comps}
    for s, e in rng:
        mo, ao = get(f"obs_{s:05d}-{e:05d}.npz")
        out["obs"].append(ao["obs"])
        out["digests"] += mo["digests"]
        for c in comps:
            m, arr = get(f"{c}_{s:05d}-{e:05d}.npz")
            out["jobs"].add(m.get("record", {}).get("job", ""))
            if m["digests"] != mo["digests"] or m["p"] != p:
                raise SystemExit(f"{base}: {c} {s}-{e} decided other shots than the observable file's")
            for k, v in (want or {}).items():
                got = m
                for kk in k.split("."):
                    got = got.get(kk) if isinstance(got, dict) else None
                if callable(v) and not v(c, got) or not callable(v) and got != v:
                    raise SystemExit(f"{base}: {c} {s}-{e}: {k} = {got!r}, not the reference's")
            out["meta"].setdefault(c, m)
            if c in ("BQ1", "BQ5"):
                n = (e - s) * CHUNK
                z, dec = np.full(n, np.nan, np.float32), np.full(n, 255, np.uint8)
                loc = arr["shot"] - s * CHUNK
                z[loc], dec[loc] = arr["z"], arr["dec"]
                arr = {"z": z, "dec": dec}
            for k, v in arr.items():
                acc[c].setdefault(k, []).append(v)
    out["obs"] = np.concatenate(out["obs"])[cut]
    out["digests"] = out["digests"][lo - rng[0][0]:hi - rng[0][0]]
    for c in comps:
        out[c] = {k: np.concatenate(v)[cut] for k, v in acc[c].items()}
    return out


def assemble(cfg, mode, o, ch):
    """A configuration's decisions in a mode as its pipeline makes them (SURFACE_PLAN 4.4), from the outputs o of the
    computations on the same shots (load_read) and the thresholds ch at p ({"C1": {"tau1", "tau2"}, "C5": {"tau1"}}):
      single:  C1 S1, B1 on the shots S1 routes, MS on the shots B1 routes | C2 B1 | C3 MS | C4 S1, MS | C5 S1, B1
      batched: C1 SB, BQ1, MB on the shots BQ1 routes | C2 Bb | C3 MB | C4 SB, MB | C5 SB, BQ5
    -> (decision (n,) uint8, stage (n,) int8: 0 decided first, 1 by B, 2 by M)."""
    single = mode == "single"
    S, Bn = ("S1", "B1") if single else ("SB", {"C1": "BQ1", "C5": "BQ5"}.get(cfg))
    M = "MS" if single else "MB"
    if cfg == "C2":
        b = o["B1" if single else "Bb"]["dec"]
        return b.copy(), np.ones(len(b), np.int8)
    if cfg == "C3":
        return o[M]["pred"].copy(), np.full(len(o[M]["pred"]), 2, np.int8)
    c = o[S]["c"]
    dec, stage = o[S]["dec"].copy(), np.zeros(len(c), np.int8)
    if cfg == "C4":
        toM = routed(c, TAU_C4)
        dec[toM], stage[toM] = o[M]["pred"][toM], 2
        return dec, stage
    toB = routed(c, ch[cfg]["tau1"])
    bz, bd = o[Bn]["z"], o[Bn]["dec"]
    if (bd[toB] == 255).any():
        raise SystemExit(f"{cfg} {mode}: a shot routed to B has no output of {Bn}")
    dec[toB], stage[toB] = bd[toB], 1
    if cfg == "C1":
        toM = toB & routed(bz, ch["C1"]["tau2"])
        dec[toM], stage[toM] = o[M]["pred"][toM], 2
    return dec, stage
