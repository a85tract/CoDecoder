"""A trained front end (convlines_train.train's final_ema.pt): load it and run it.

The file holds the EMA weights, the network's configuration, what the network was built from (the surface code's
geometry, lines and parity map, or the BB code and its circuit) and the training record; torch.load(...,
weights_only=True) reads it. load() rebuilds the network; infer() returns, per shot, the decided observables and the
confidence the routing compares with tau: |c| for a surface-code network (decision c > 0), the margin between the
two best of the 2^k logical classes for a BB network. Inference runs in fp32 (no autocast, TF32 off)."""
import hashlib
import io

import numpy as np
import torch

FORMAT = "convlines_train/1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def to_saved(spec):
    """numpy arrays -> tensors (weights_only loading); everything else as is."""
    return {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in spec.items()}


def from_saved(spec):
    return {k: v.numpy() if torch.is_tensor(v) else v for k, v in spec.items()}


def build(kind, spec, arch, cfg=None):
    """The untrained network of a front end (surface: convlines.net.ConvLines; bb: bb.net.ConvLinesBB)."""
    if kind == "surface":
        from convlines.net import ConvLines
        from convlines_train import surface
        model = ConvLines(spec["d"], io.BytesIO(surface.geometry_npz(spec)), **arch)
        model.A.copy_(torch.as_tensor(spec["A"], dtype=torch.float32))
        return model
    from convlines_train import bbcode
    import stim
    name = bbcode.register(spec["params"], stim.Circuit(spec["circuit"]))
    if cfg is None:
        stem, blk = bbcode.frozen()[3].taps_from_design(spec["design"])
        return bbcode.frozen()[3].ConvLinesBB(name, 0.0, stem, blk, arch["width"], arch["bott"], arch["blocks"],
                                              arch["heads"])
    assert cfg["code"] == name, "the front end's code name is not its circuit's"
    return bbcode.frozen()[3].build(cfg)


class FrontEnd:
    def __init__(self, ck, device="cpu"):
        import stim
        self.kind, self.spec, self.arch = ck["kind"], from_saved(ck["spec"]), ck["arch"]
        self.model = build(self.kind, self.spec, self.arch, ck.get("cfg"))
        self.model.load_state_dict(ck["state"])
        self.model.to(device).eval()
        self.device = device
        self.circuit = stim.Circuit(self.spec["circuit"])
        self.num_detectors = self.circuit.num_detectors
        self.train = ck.get("train", {})

    @torch.no_grad()
    def infer(self, det, batch=4096):
        """(decided observables (n, k) uint8, confidence (n,) float64) for detector bits det (n, m)."""
        old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        preds, confs = [], []
        try:
            for i in range(0, len(det), batch):
                x = torch.as_tensor(np.ascontiguousarray(det[i:i + batch]), dtype=torch.uint8, device=self.device)
                if self.kind == "surface":
                    c = self.model(x)[0].float()
                    preds.append((c > 0).to(torch.uint8)[:, None].cpu())
                    confs.append(c.abs().double().cpu())
                else:
                    _, bits, margin = self.model.decide(self.model(x)[0].float())
                    preds.append(bits.cpu())
                    confs.append(margin.double().cpu())
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
        return torch.cat(preds).numpy().astype(np.uint8), torch.cat(confs).numpy()


class Released(FrontEnd):
    """A network of the paper's surface-code circuits (circuits/d*_legacy.stim, grid weights/surface/grid_d*.npz): a released
    one by name (weights/surface/configs.json, d = 7-15) or a retrained one, training/surface's arch.train final_ema.pt
    ({'state', 'cfg'}); the circuit is the verdict circuit of its distance."""

    def __init__(self, state, arch, d, device="cpu"):
        import os
        from convlines import circuits
        from convlines.net import ConvLines
        self.kind, self.arch, self.train = "surface", arch, {}
        grid = os.path.join(circuits.WEIGHTS, f"grid_d{d}.npz")
        self.model = ConvLines(d, grid, **arch)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device
        self.circuit = circuits.circuit(d)
        self.num_detectors = self.circuit.num_detectors
        self.spec = dict(d=d, circuit=str(self.circuit))


class TrainedBB(FrontEnd):
    """A network of training/bb's bb.train ({'state', 'cfg', ...}: the released r11, r12, g2_D12k113c10 and S55, or one
    retrained by training/TRAINING.md) with Relay-BP's circuit of its code at its p (uniform circuit noise; QND_RELAY).
    The networks trained under the fitted depspam_fm noise (S55', S55'-deep) need that family's circuit (tune --circuit)."""

    def __init__(self, ck, device="cpu"):
        from convlines_train import bbcode
        _, _, bcode, bnet, _ = bbcode.frozen()
        cfg = ck["cfg"]
        self.kind, self.arch, self.train = "bb", {}, {}
        self.model = bnet.build(cfg)
        self.model.load_state_dict(ck["state"])
        self.model.to(device).eval()
        self.device = device
        self.circuit = bcode.load_circuit(cfg["code"], cfg["p"])
        self.num_detectors = self.circuit.num_detectors
        self.spec = dict(code=cfg["code"], p=cfg["p"], circuit=str(self.circuit))


def load(path, device="cpu"):
    """A front end: convlines_train.train's final_ema.pt, training/surface's arch.train final_ema.pt, training/bb's
    bb.train network (TrainedBB) or the name of a released surface-code network (d7_b1w64_r740, ..., d15_b5_r782). Its
    .file is the weights file read."""
    import os
    if not os.path.exists(path):
        from safetensors.torch import load_file
        from convlines import decoder
        cfg = decoder.config(path)
        assert "file" in cfg, f"{path}: only its deploy tables are released (no fp32 network to run here)"
        file = os.path.join(decoder.circuits.WEIGHTS, cfg["file"])
        fe = Released(load_file(file), cfg["arch"], cfg["d"], device)
        fe.file = file
        return fe
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:                                    # arch.train's record holds more than tensors: your own file
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") == FORMAT:
        fe = FrontEnd(ck, device)
    elif ck.get("cfg", {}).get("arch") == "convlines":
        cfg = ck["cfg"]
        arch = {k: cfg[k] for k in ("width", "bott", "blocks", "heads", "stem_k", "block_k", "mlp")}
        fe = Released(ck["state"], arch, cfg["d"], device)
    else:
        cfg = ck.get("cfg", {})
        assert "code" in cfg and "L" not in cfg and "state" in ck, \
            f"{path}: neither a {FORMAT} front end nor an arch.train or bb.train checkpoint"
        fe = TrainedBB(ck, device)
    fe.file = path
    return fe
