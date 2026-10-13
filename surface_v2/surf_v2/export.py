"""A SURF-v2 stage's final_ema.pt as this release's weight file: the checkpoint's state without the training-only
read-out (aux.*, as arch.net.export_decision drops it), tensors unchanged, as weights/<name>.safetensors, loaded
strictly into convlines.net.ConvLines with the entry's arch (weights/configs_sv2.json). Written once, before the
calibration reads, so that the calibration reads, the timing, the freeze and the test reads all use one file.

    python -m surf_v2.export --ckpt CKPT/final_ema.pt --name d11_b5_sv2 --out weights
    [--check-arch ISING_COMPARE --shots DRY]   the research code's network (arch.net) on the first 2,000 sv2dry shots:
                                              fp32 logits equal to this release's module's, bit for bit
The record (<name>.export.json) goes into the public freeze commit, so it names no account: the account's home is
written ~ and its name $USER (/scratch/alpine/$USER/..., as the jobs write it), in the checkpoint's path and in its
hp, which hold the trainer's arguments (surf_v2/public.sh refuses any file that names an account; found by E0)."""
import argparse
import getpass
import hashlib
import json
import os
import sys

import numpy as np
import torch

from convlines import decoder, testset


def portable(rec):
    """rec as plain JSON with the account's home written ~ and the account's name $USER."""
    u, home = getpass.getuser(), os.path.expanduser("~")

    def fix(o):
        if isinstance(o, str):
            o = o.replace(home + "/", "~/") if home not in ("", "/") else o
            return o.replace(u, "$USER") if u and u != "root" else o
        if isinstance(o, dict):
            return {fix(k): fix(v) for k, v in o.items()}
        return [fix(v) for v in o] if isinstance(o, list) else o
    return fix(json.loads(json.dumps(rec, default=str)))


def main():
    from safetensors.torch import save_file
    from convlines.net import ConvLines
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="the network's name in weights/configs_sv2.json")
    ap.add_argument("--out", required=True, help="the directory of the weight file (the release's weights/ on a pod)")
    ap.add_argument("--check-arch", default="", help="the research repository's ising_compare directory")
    ap.add_argument("--shots", default="", help="--check-arch: the sv2dry chunk files")
    a = ap.parse_args()
    e = decoder.config(a.name)
    raw = open(a.ckpt, "rb").read()
    sha_ck = hashlib.sha256(raw).hexdigest()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["d"] == e["d"] and cfg.get("head", "vote") == "vote", f"{a.ckpt}: not a vote network of d = {e['d']}"
    assert all(cfg[k] == (list(v) if isinstance(v, list) else v) for k, v in e["arch"].items()), \
        f"{a.ckpt}: its size {({k: cfg[k] for k in e['arch']})} is not {e['arch']}"
    drop = sorted(k for k in ck["state"] if k.startswith("aux."))
    assert bool(drop) == bool(cfg.get("aux_lines", False)), "aux weights and aux_lines disagree"
    state = {k: v.contiguous() for k, v in ck["state"].items() if not k.startswith("aux.")}
    model = ConvLines(e["d"], os.path.join(decoder.ROOT, e["grid"]["geometry"]), **e["arch"])
    model.load_state_dict(state, strict=True)
    n_par = sum(p.numel() for p in model.parameters())
    assert n_par == e["parameters"], f"{n_par} parameters, not {e['parameters']}"
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, e["file"])
    assert not os.path.exists(f), f"{f} exists"
    # one metadata key: safetensors writes its metadata map in hash order, so two keys give either of two files
    meta = json.dumps({"final_ema_sha256": sha_ck, "network": a.name}, sort_keys=True)
    save_file(state, f, metadata={"surf_v2_export": meta})
    rec = dict(network=a.name, d=e["d"], file=e["file"], sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(),
               final_ema=os.path.abspath(a.ckpt), final_ema_sha256=sha_ck, dropped=drop, parameters=n_par,
               hp=ck.get("hp"), row=ck.get("row"))
    if a.check_arch:
        sys.path.insert(0, a.check_arch)
        from arch import net as anet
        ex = anet.export_decision(ck)
        ref = anet.net_from_cfg(ex["cfg"])
        ref.load_state_dict(ex["state"])
        det, _ = testset.load_chunks("sv2dry", e["d"], [0], a.shots)
        det = det[:2000]
        c1 = decoder.logits(model.eval(), det)
        c2 = decoder.logits(ref.eval(), det)
        rec["check_arch"] = dict(shots=len(det), equal=bool(np.array_equal(c1.view(np.uint32), c2.view(np.uint32))),
                                 max_abs_diff=float(np.abs(c1.astype(np.float64) - c2).max()),
                                 decisions_differ=int(((c1 > 0) != (c2 > 0)).sum()))
        assert rec["check_arch"]["decisions_differ"] == 0, rec["check_arch"]
    with open(f[:-len(".safetensors")] + ".export.json", "x") as fh:
        json.dump(portable(rec), fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("network", "sha256", "final_ema_sha256", "dropped", "parameters")} |
                     ({"check_arch": rec["check_arch"]} if "check_arch" in rec else {})), flush=True)


if __name__ == "__main__":
    main()
