#!/usr/bin/env python3
"""Check the released networks: every file's sha256 against MANIFEST.sha256, then each network of networks.csv built
with this repository's code, its weights loaded strictly, and its parameter count compared with the table.

    python weights/check_weights.py                     # every family, on the CPU (a few minutes)
    python weights/check_weights.py --family head       # one family: surface, bb, cascade_bb72, cascade_d13, head

Each family runs in its own process, because training/bb and head/ hold different versions of the same packages (bb,
arch). The output-head comparison's surface arms build their geometry with NVIDIA's Ising-Decoding code at 3ce4379
(ICMP_NV=<your clone>/code); without it those nine rows report the import error and the check fails. Prints one JSON
line per file and per network, and exits 1 if anything fails. The d = 17 surface network is released as its deploy
tables only, so its row builds the module without weights and checks the parameter count alone."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAMILIES = ("surface", "bb", "cascade_bb72", "cascade_d13", "head")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ckpt(path):
    """torch.load, without unpickling code where the file allows it; -> (checkpoint, 'weights_only' or 'full')."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False), "full"


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def surface(rows):
    sys.path.insert(0, os.path.join(REPO, "evaluation"))
    from convlines import circuits, decoder, net
    for r in rows:
        cfg = decoder.config(r["network"])
        if "file" in cfg:
            model, _ = decoder.load_network(r["network"])
            yield r, n_params(model), "state dict loaded strictly", "safetensors"
        else:
            grid = os.path.join(circuits.WEIGHTS, os.path.basename(cfg["grid"]["geometry"]))
            model = net.ConvLines(cfg["d"], grid, **cfg["arch"])
            decoder.load_tables(r["network"])
            yield r, n_params(model), "deploy tables read; module built without weights", "safetensors"


def bb(rows):
    sys.path[:0] = [os.path.join(REPO, "training", "bb", p) for p in ("ising_compare", "qldpc_neural")]
    from bb import cascade_net, net
    for r in rows:
        ck, mode = load_ckpt(os.path.join(HERE, r["file"]))
        cfg = ck["cfg"]
        if "L" in cfg:                                       # a Cascade network
            model, state = cascade_net.build(cfg), ck["ema"]
        else:
            model, state = net.build(cfg), ck["state"]
        model.load_state_dict(state, strict=True)
        yield r, n_params(model), "state dict loaded strictly", mode


def cascade_d13(rows):
    sys.path.insert(0, os.path.join(REPO, "cascade", "d13"))
    from surface import surface_layout, surface_net
    for r in rows:
        ck, mode = load_ckpt(os.path.join(HERE, r["file"]))
        model = surface_net.build(ck["cfg"], surface_layout.Layout(ck["cfg"]["d"]))
        model.load_state_dict(ck["ema"], strict=True)
        yield r, n_params(model), f"EMA loaded strictly; step {ck['step']}, status {ck['status']}", mode


def head(rows):
    sys.path[:0] = [os.path.join(REPO, "head", p) for p in ("ising_compare", "qldpc_neural")]
    for r in rows:
        ck, mode = load_ckpt(os.path.join(HERE, r["file"]))
        if r["file"].startswith("head/surface/"):
            from arch import net
            ex = net.export_decision(ck)
            model = net.net_from_cfg(ex["cfg"])
        else:
            from bb import net
            ex = net.export_decision(ck)
            model = net.build(ex["cfg"])
        model.load_state_dict(ex["state"], strict=True)
        yield r, n_params(model), "decision path loaded strictly (training-only read-out dropped)", mode


def run_family(fam):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "networks.csv"))) if r["family"] == fam]
    ok, done = True, set()
    try:
        for r, n, how, mode in {"surface": surface, "bb": bb, "cascade_bb72": bb, "cascade_d13": cascade_d13,
                                "head": head}[fam](rows):
            good = n == int(r["parameters"])
            ok &= good
            done.add(r["network"])
            print(json.dumps({"network": r["network"], "file": r["file"], "parameters": n,
                              "parameters_expected": int(r["parameters"]), "ok": good, "load": how, "torch_load": mode}),
                  flush=True)
    except Exception as e:                                   # report the rest of the family as failed
        for r in rows:
            if r["network"] not in done:
                print(json.dumps({"network": r["network"], "file": r["file"], "ok": False,
                                  "error": f"{type(e).__name__}: {e}"}), flush=True)
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=FAMILIES, help="check one family's networks only (no sha256 pass)")
    a = ap.parse_args()
    if a.family:
        sys.exit(0 if run_family(a.family) else 1)
    ok = True
    for line in open(os.path.join(HERE, "MANIFEST.sha256")):
        if not line.strip():
            continue
        want, rel = line.split(maxsplit=1)
        rel = rel.strip()
        path = os.path.join(HERE, rel)
        got = sha256_file(path) if os.path.exists(path) else None
        ok &= got == want
        print(json.dumps({"file": rel, "sha256_ok": got == want, "bytes": os.path.getsize(path) if got else None}),
              flush=True)
    for fam in FAMILIES:
        ok &= subprocess.run([sys.executable, os.path.abspath(__file__), "--family", fam]).returncode == 0
    print("all networks check" if ok else "CHECK FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
