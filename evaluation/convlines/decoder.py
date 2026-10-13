"""The PyTorch reference of the decoder: a ConvLines network plus confidence routing to correlated PyMatching.

For each shot the network gives one logit c for the logical flip. A shot with |c| < tau (c as float32, the comparison
in double precision against the registered full-precision tau) is routed: correlated PyMatching decodes it. Every
other shot is decided by the network alone, flip = (c > 0). tau was fitted on a separate calibration set of 200,000
shots as the q-quantile of |c| (q in weights/surface/configs.json), before any test shot existed.

The deployed executables (timing/engine.py's Triton engine at d = 9-17, timing/d7's CUDA kernel at d = 7) compute
c with fp16 matrix inputs and fp32 accumulation; this reference runs the trained fp32 module, so a shot whose c lies
within rounding of 0 or of +-tau can be decided differently (README, "Verification"). decode() takes the deployed c
(timing.deployed.Logits) or the studies' routing mask (routing/, published_mask) instead where given.

d = 17 (study 4): this release holds the network in its deployed form only, the deploy tables the deployed engine ran
(weights/surface/configs.json "tables", read by load_tables); its fp32 module is not included, so at d = 17 c comes from the
deployed engine on an NVIDIA GPU (convlines.evaluate --routing deployed).

    python -m convlines.decoder --d 9 --chunk DIR/d9_p0.006_X_test2_000.npz [--device cuda]"""
import argparse
import hashlib
import io
import json
import os

import numpy as np

from convlines import circuits

ROOT = circuits.ROOT
NETWORKS = {7: "d7_b1w64_r740", 9: "d9_b3_r764", 11: "d11_b5_r777", 13: "d13_b5_r775", 15: "d15_b5_r782",
            17: "d17_b5_r787"}
# The reverse post hoc control (README, "The two post hoc controls"): the same networks routing |c| < tau to plain PM,
# tau refitted for plain PM on the calibration sets by the registered rule (q = 2, 1, 0, 1 and 1 % at d = 7-15).
TAU_PLAIN = {7: 0.7138206958770752, 9: 0.6820775270462036, 11: 0.0, 13: 1.0115736722946167, 15: 1.0091289281845093}


def config(name):
    """A network's config: weights/surface/configs.json (the registered studies' six), else
    weights/surface/configs_sv2.json (SURF-v2's networks; an entry with "base" is that registered network with the entry's fields, e.g. another policy)."""
    nets = json.load(open(os.path.join(circuits.WEIGHTS, "configs.json")))["networks"]
    if name in nets:
        return nets[name]
    e = dict(json.load(open(os.path.join(circuits.WEIGHTS, "configs_sv2.json")))["networks"][name])
    base = e.pop("base", None)
    return dict(nets[base], **e, name=name) if base else dict(e, name=name)


def check_file(cfg, path):
    """A SURF-v2 network's weights against the sha256 its config holds once frozen (no check before the freeze)."""
    if cfg.get("sha256"):
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert sha == cfg["sha256"], f"{path}: sha256 {sha} is not the frozen {cfg['sha256']}"
    return path


def weights_sha256(name):
    """The sha256 of a network's weight file (its deploy tables where the release holds only those)."""
    cfg = config(name)
    return hashlib.sha256(open(os.path.join(circuits.WEIGHTS, cfg.get("file") or cfg["tables"]), "rb").read()).hexdigest()


def load_network(name, device="cpu"):
    """(model in eval mode on device, its config)."""
    from safetensors.torch import load_file
    from convlines.net import ConvLines
    cfg = config(name)
    assert "file" in cfg, (f"{name}: only its deploy tables (weights/surface/{cfg.get('tables')}) are in this repository, not the "
                           "fp32 network; decode it with the deployed engine (--routing deployed, an NVIDIA GPU)")
    model = ConvLines(cfg["d"], os.path.join(circuits.WEIGHTS, os.path.basename(cfg["grid"]["geometry"])), **cfg["arch"])
    model.load_state_dict(load_file(check_file(cfg, os.path.join(circuits.WEIGHTS, cfg["file"]))))
    return model.to(device).eval(), cfg


def read_tables(path):
    """Deploy tables in convlines.net.deploy_tables's form (fp32 on the CPU) from a safetensors file: the tensors under
    their dotted names (P, blocks.0.Wd, head.Wsc, ...), every other entry in the header's "tables" JSON."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, "pt") as f:
        meta = json.loads(f.metadata()["tables"])
    t = {k: v for k, v in meta.items() if k not in ("blocks", "head")}
    t["shape"], t["blocks"], t["head"] = tuple(t["shape"]), [dict(b) for b in meta["blocks"]], dict(meta["head"])
    for k, v in load_file(path).items():
        part = k.split(".")
        if part[0] == "blocks":
            t["blocks"][int(part[1])][part[2]] = v
        elif part[0] == "head":
            t["head"][part[1]] = v
        else:
            t[k] = v
    return t


def load_tables(name):
    """(deploy tables, config) of a network published in its deployed form (configs.json "tables"; d = 17)."""
    cfg = config(name)
    return read_tables(check_file(cfg, os.path.join(circuits.WEIGHTS, cfg["tables"]))), cfg


def logits(model, det, device="cpu", batch=1000):
    """(n,) float32 logits c for detector bits det (n, m) uint8, in full fp32: TF32 is switched off for the call
    (PyTorch allows it in cuDNN convolutions by default, and NVIDIA's Ising loader enables it for matmuls too)."""
    import torch                                        # imported here: the opponents' CPU and CUDA-Q venvs lack it
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        out = []
        with torch.no_grad():
            for i in range(0, len(det), batch):
                c, _ = model(torch.from_numpy(np.ascontiguousarray(det[i:i + batch])).to(device))
                out.append(c.float().cpu())
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    return torch.cat(out).numpy().astype(np.float32)


def route(c, tau):
    return np.abs(np.asarray(c, np.float32).astype(np.float64)) < tau


def routing_manifest():
    return json.load(open(os.path.join(ROOT, "routing", "manifest.json")))


def published_mask(entry):
    """The bool mask of a routing/manifest.json entry (test: every shot of the set in chunk order; val: validation
    chunks 0-19), its file checked against the entry's sha256."""
    raw = open(os.path.join(ROOT, "routing", entry["file"]), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"], f"routing/{entry['file']}: sha256 differs from the manifest"
    return np.load(io.BytesIO(raw))["routed"].astype(bool)


def decode(model, cfg, det, device="cpu", matcher=None, c=None, routed=None, correlated=True):
    """(c, routed, decision) of every shot: c from the fp32 reference unless given (the deployed executable's c), routed
    = |c| < tau unless given (a published mask); any other shot is decided by c > 0. correlated=False sends the routed
    shots to plain PM (the reverse control, with its tau in cfg)."""
    c = logits(model, det, device) if c is None else np.asarray(c, np.float32)
    routed = route(c, cfg["policy"]["tau"]) if routed is None else np.asarray(routed, bool)
    dec = (c > 0).astype(np.uint8)
    if routed.any():
        m = matcher or circuits.matcher(cfg["d"], correlated=correlated)
        dec[routed] = circuits.pm_decode(m, det[routed], correlated=correlated)
    return c, routed, dec


def main():
    import torch
    from convlines import testset
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True, choices=sorted(NETWORKS))
    ap.add_argument("--chunk", required=True, help="a chunk file (test or validation format)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, cfg = load_network(NETWORKS[a.d], a.device)
    det, obs = testset.load(a.chunk, a.d)
    c, routed, dec = decode(model, cfg, det, a.device)
    print(json.dumps(dict(network=NETWORKS[a.d], chunk=os.path.basename(a.chunk), shots=len(obs),
                          routed=int(routed.sum()), failures=int((dec != obs).sum()),
                          network_alone_failures=int(((c > 0).astype(np.uint8) != obs).sum()))))


if __name__ == "__main__":
    main()
