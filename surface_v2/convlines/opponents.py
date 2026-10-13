"""The six opponents and the two post hoc controls, each a function from detector bits (n, m) uint8 to the predicted
logical flip (n,) uint8.

  plain PM, correlated PM   PyMatching 2.4.0 on the verdict circuit's DEM (convlines.circuits)
  Ising M1 + PM, M4 + PM    NVIDIA's public pre-decoders "Ising-Decoder-SurfaceCode-1-Fast" (M1, receptive field 9)
                            and "-Accurate" (M4, receptive field 13) through NVIDIA's own inference module
                            (PreDecoderMemoryEvalModule: detectors -> the pre-decoder's logical frame pre_L and the
                            residual detectors), then plain PyMatching on the residual; flip = pre_L XOR PM(residual).
                            The official numerics: fp32 weights, TF32 as NVIDIA's runner enables it. NVIDIA's code and
                            checkpoints are not part of this repository: clone NVIDIA/Ising-Decoding at commit
                            3ce4379fa6cf475789007bde4b6a8206e8622992 (tag 0.1.0), fetch its two checkpoints
                            (git lfs pull), install its inference requirements and set ISING_DECODING_DIR to the clone.
                            The studies also read every TensorRT FP16 / FP8 plan of Ising they timed on the H100; those
                            rows are in results/ (README), and timing/ising_trt.py builds the plans.
  Ising M1 / M4 + correlated PM  the same, with correlated PyMatching on the residual (the post hoc controls of the
                            README at d = 7-15, not registered opponents there; registered opponents of study 4 at
                            d = 17)
  min-LLR BP-OSD, BP 60     NVIDIA CUDA-Q QEC's nv-qldpc-decoder (cudaq-qec 0.8.0, needs an NVIDIA GPU) on the joint DEM
                            (both detector types, decompose_errors=False, identical columns merged), given the
                            observable matrix: use_sparsity, use_osd, osd_method 3 (OSD-CS), osd_order 10,
                            bp_batch_size 2048; min-LLR: osd_init_method "min_llr", max_iterations 10; BP 60:
                            "final_llr", max_iterations 60."""
import hashlib
import os
import types

import numpy as np

from convlines import circuits

ISING = {"M1": ("Ising-Decoder-SurfaceCode-1-Fast.pt", 1, "08ce4b1e09e3f396401d606b7c476225101012a6206bc7877efe153daae7b8c4"),
         "M4": ("Ising-Decoder-SurfaceCode-1-Accurate.pt", 4, "f49df7fc70d2ca890d5155982dae79e531f012d584203308617e9f6316b19fa9")}
BPOSD = dict(use_sparsity=True, use_osd=True, osd_method=3, osd_order=10, max_iterations=10, bp_batch_size=2048)
BPOSD_ARMS = {"minllr": dict(osd_init_method="min_llr"), "bp60": dict(osd_init_method="final_llr", max_iterations=60)}


def pymatching(d, correlated):
    m = circuits.matcher(d, correlated)
    return lambda det: circuits.pm_decode(m, det, correlated)


# ---- Ising (NVIDIA's code, imported from ISING_DECODING_DIR, unmodified) ----------------------------------------------
def ising_module(model, d, device="cuda"):
    """(NVIDIA's PreDecoderMemoryEvalModule of model M1 / M4 at distance d with d rounds, on device; the TF32 settings
    of NVIDIA's runner (matmul, cuDNN); NVIDIA's merged config)."""
    import sys
    import torch
    root = os.environ["ISING_DECODING_DIR"]
    if os.path.join(root, "code") not in sys.path:
        sys.path.insert(0, os.path.join(root, "code"))
    from omegaconf import OmegaConf
    from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
    from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
    from workflows.run import _load_model
    fname, mid, sha = ISING[model]
    ckpt = os.path.join(root, "models", fname)
    assert hashlib.sha256(open(ckpt, "rb").read()).hexdigest() == sha, f"{ckpt}: not the released checkpoint"
    cfg = OmegaConf.load(os.path.join(root, "conf", "config_public.yaml"))
    cfg.model_id, cfg.distance, cfg.n_rounds = mid, d, d
    cfg.workflow.task = "inference"
    cfg = apply_public_defaults_and_model(cfg, validate_public_config(cfg))
    cfg.model_checkpoint_file = ckpt
    cfg.test.meas_basis_test = "X"
    assert cfg.data.code_rotation == "XV" and cfg.test.n_rounds == d and not cfg.enable_fp16
    tf32 = bool(cfg.enable_matmul_tf32), bool(cfg.enable_cudnn_tf32)      # NVIDIA's runner settings, set per call
    dev = torch.device(device)
    net = _load_model(cfg, types.SimpleNamespace(rank=0, device=dev)).eval()
    pipe = PreDecoderMemoryEvalModule(net, cfg, _build_stab_maps(d, "XV"), dev).to(dev).eval()
    return pipe, tf32, cfg


def ising_predecode(pipe, tf32, det, device="cuda", batch=512):
    """The official numerics: (the pre-decoder's logical frame pre_L (n,) uint8, the residual detectors (n, m) uint8)."""
    import torch
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    try:
        pre, res = [], []
        with torch.no_grad():
            for i in range(0, len(det), batch):
                o = pipe(torch.from_numpy(np.ascontiguousarray(det[i:i + batch])).to(device)).cpu().numpy()
                pre.append(o[:, 0])
                res.append(o[:, 1:])
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    return np.concatenate(pre).astype(np.uint8), np.concatenate(res).astype(np.uint8)


def ising(model, d, device="cuda", batch=512, correlated=False):
    pipe, tf32, _ = ising_module(model, d, device)
    pm = circuits.matcher(d, correlated)

    def run(det):
        pre, res = ising_predecode(pipe, tf32, det, device, batch)
        return circuits.pm_decode(pm, res, correlated) ^ pre
    return run


# ---- BP-OSD (CUDA-Q QEC) ------------------------------------------------------------------------------------------------
def check_matrices(dem):
    """(H (detectors x columns) uint8, O (observables x columns) uint8, priors (columns,) float64) of a DEM; columns
    with identical (detectors, observables) merged as independent flips."""
    cols = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        if not dets and not obs:
            continue
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        q = cols.get(key, 0.0)
        cols[key] = p * (1 - q) + q * (1 - p)
    keys = list(cols)
    H = np.zeros((dem.num_detectors, len(keys)), np.uint8)
    O = np.zeros((dem.num_observables, len(keys)), np.uint8)
    for j, (ds, os_) in enumerate(keys):
        H[list(ds), j] = 1
        O[list(os_), j] = 1
    return H, O, np.array([cols[k] for k in keys])


def bposd(d, arm, chunk=2048):
    import cudaq_qec as qec
    H, O, p = check_matrices(circuits.dem(circuits.circuit(d), decompose=False))
    dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=p.tolist(), O=O, **{**BPOSD, **BPOSD_ARMS[arm]})

    def run(det):
        out = np.empty(len(det), np.uint8)
        for i in range(0, len(det), chunk):
            res = dec.decode_batch(det[i:i + chunk].astype(np.float64))
            out[i:i + chunk] = [r.result[0] > 0.5 for r in res]
        return out
    return run


def make(name, d, device="cuda"):
    if name == "plain PM":
        return pymatching(d, False)
    if name == "correlated PM":
        return pymatching(d, True)
    if name in ("Ising M1 + PM", "Ising M4 + PM"):
        return ising(name.split()[1], d, device)
    if name in ("Ising M1 + correlated PM", "Ising M4 + correlated PM"):
        return ising(name.split()[1], d, device, correlated=True)
    if name == "min-LLR BP-OSD":
        return bposd(d, "minllr")
    if name == "BP 60 BP-OSD":
        return bposd(d, "bp60")
    raise ValueError(name)


NAMES = ("Ising M1 + PM", "Ising M4 + PM", "plain PM", "correlated PM", "min-LLR BP-OSD", "BP 60 BP-OSD")
CONTROLS = ("Ising M1 + correlated PM", "Ising M4 + correlated PM")      # post hoc, after the studies (README)
