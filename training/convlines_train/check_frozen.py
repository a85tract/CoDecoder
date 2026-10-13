"""Check that the frozen training trees (training/surface, training/bb) rebuild the released networks.

surface  for each distance: the frozen icmp.circuits builds NVIDIA's circuit with the registered sha256
         (convlines/circuits.py); the frozen arch.net.ConvLines (geometry from NVIDIA's code, A fitted by icmp.lines.gmap
         on fresh shots) has the released grid (weights/surface/grid_d*.npz), the released A and parameter count, and loads the
         released state dict (d = 7-15; at d = 17 the deploy tables' A, in engine order). Needs NVIDIA's
         Ising-Decoding at 3ce4379 (ISING_DECODING_DIR, as scripts/make_circuits.py).
bb       the frozen bb.net (and bb.cascade_net) build every released BB network from its own configuration and load its
         weights strictly: r11, r12 (weights/bb), g2_D12k113c10 (weights/bb), S55, S55', S55'-deep, C14, C14', C6,
         C6' (weights/cascade_bb72), with their parameter counts; bb.design recomputes BB72's design (training/bb's
         data/bb/design_bb72.json) from the pinned circuit. Needs Relay-BP's circuits (QND_RELAY, README "BB72").
The training records inside the released BB checkpoints (bb.train's arguments and sample count) are printed.

    ISING_DECODING_DIR=... QND_RELAY=... python -m convlines_train.check_frozen [--surface 7,9,11,13,15,17] [--bb]"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root
SURF = os.path.join(ROOT, "training", "surface", "ising_compare")
BBI, BBQ = (os.path.join(ROOT, "training", "bb", p) for p in ("ising_compare", "qldpc_neural"))
BB_NETS = {"r11": "weights/bb/r11_final_ema.pt", "r12": "weights/bb/r12_final_ema.pt",
           "g2_D12k113c10": "weights/bb/g2_D12k113c10_final_ema.pt", "S55": "weights/cascade_bb72/S55.pt",
           "S55'": "weights/cascade_bb72/S55p.pt", "S55'-deep": "weights/cascade_bb72/S55p_deep.pt",
           "C14": "weights/cascade_bb72/C14.pt", "C14'": "weights/cascade_bb72/C14p.pt",
           "C6": "weights/cascade_bb72/C6.pt", "C6'": "weights/cascade_bb72/C6p.pt"}


def surface(ds):
    os.environ.setdefault("ICMP_NV", os.path.join(os.environ["ISING_DECODING_DIR"], "code"))
    sys.path.insert(0, SURF)
    from safetensors.torch import load_file
    from arch import net as anet
    from icmp import circuits as ic
    from convlines import circuits as rc, decoder
    ok = True
    for d in ds:
        text = str(ic.circuit(d, noise=ic.NOISE_VERDICT))
        name = decoder.NETWORKS[d]
        cfg = decoder.config(name)
        torch.manual_seed(0)
        model = anet.ConvLines(d, **cfg["arch"])
        g = np.load(os.path.join(rc.WEIGHTS, os.path.basename(cfg["grid"]["geometry"])))
        lay = model.lay
        row = dict(d=d, network=name,
                   circuit_sha256_equal=hashlib.sha256(text.encode()).hexdigest() == rc.CIRCUIT_SHA256[d],
                   grid_equal=bool(np.array_equal(lay.idx, g["idx"]) and np.array_equal(lay.is_x, g["is_x"].astype(bool))
                                   and np.array_equal(lay.off, g["line_off"])),
                   parameters=sum(p.numel() for p in model.parameters()), parameters_released=cfg["parameters"])
        if "file" in cfg:
            state = load_file(os.path.join(rc.WEIGHTS, cfg["file"]))
            row["A_equal"] = bool(torch.equal(model.A.to(torch.float32), state["A"].to(torch.float32)))
            model.load_state_dict(state, strict=True)
            row["state_dict_loads"] = True
        else:
            t, _ = decoder.load_tables(name)
            order, _ = lay.site_order()
            row["A_equal"] = bool(torch.equal(model.A[torch.as_tensor(order)].to(torch.float32),
                                              t["A"].to(torch.float32)))
        row["ok"] = all(row[k] for k in ("circuit_sha256_equal", "grid_equal", "A_equal")) and \
            row["parameters"] == row["parameters_released"]
        ok &= row["ok"]
        print(json.dumps(row), flush=True)
    return ok


def bb():
    sys.path[:0] = [BBI, BBQ]
    from bb import cascade_net, design, net as bnet
    ok = True
    for name, rel in BB_NETS.items():
        ck = torch.load(os.path.join(ROOT, rel), map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if "L" in cfg:                                       # a Cascade network
            model, state = cascade_net.build(cfg), ck["ema"]
        else:
            model, state = bnet.build(cfg), ck["state"]
        model.load_state_dict(state, strict=True)
        row = dict(network=name, file=rel, parameters=sum(p.numel() for p in model.parameters()), loads=True,
                   blocks=cfg.get("blocks", cfg.get("L")), p=cfg.get("p"))
        hp = ck.get("hp")
        if hp:                                               # bb.train's record in the original checkpoint
            row["recorded"] = {k: hp.get(k) for k in ("run", "run_id", "code", "p", "samples", "lr", "warmup",
                                                      "decay_start", "p_ramp", "blocks", "init") if k in hp}
            row["recorded"]["samples_done"] = ck.get("samples")
        print(json.dumps(row, default=str), flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "design_bb72.json")
        argv = sys.argv
        sys.argv = ["bb.design", "--code", "bb72", "--p", "0.003", "--out", out]
        try:
            design.main()
        finally:
            sys.argv = argv
        new, old = json.load(open(out)), json.load(open(os.path.join(BBI, "data", "bb", "design_bb72.json")))
        same = new["taps"]["stem"] == old["taps"]["stem"] and new["taps"]["tanner"] == old["taps"]["tanner"]
        print(json.dumps(dict(design="bb72", taps_equal=same, stem_taps=len(new["taps"]["stem"]))), flush=True)
        ok &= same
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="7,9,11,13,15,17", help="distances ('' for none)")
    ap.add_argument("--bb", action="store_true")
    a = ap.parse_args()
    ok = True
    if a.surface:
        ok &= surface([int(x) for x in a.surface.split(",")])
    if a.bb:
        ok &= bb()
    print("frozen trees:", "every check passed" if ok else "A CHECK FAILED", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
