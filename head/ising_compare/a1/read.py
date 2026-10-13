"""A1's reads (revision/REVISION_PLAN.md 2.1, "Evaluation", "Inference"): an arm's final EMA network on an A1 set. The
training-only read-out is dropped first (net.export_decision; --export saves that decision file, the graph the A3
session times, with its sha256), the network is rebuilt from the exported configuration and loaded strictly, and it
runs in fp32 without autocast and with TF32 off, at batch --batch; the first --b1 shots run again at batch one (the
large batches checked against batch one: any decision that differs fails the read: it is kept as <out>.failed.npz,
never under --out). An arm's checkpoint (seed > 0) must be the arm's completed run (a1.podval.check_identity). Test
reads take their job list from a1.preflight, the stage barrier (a1/jobs/args/reads_main.txt).
  surface  c (float32), pred = c > 0, conf = |c|
  bb72     joint head: pred = the argmax class's 12 bits, conf = the margin (top-1 minus top-2 score); pooled head:
           pred = logit > 0, conf = min_k |logit_k|
Output npz: pred, conf, c (surface), and meta (JSON: arm, seed, the checkpoint's and the decision file's sha256, the
set's kind, chunks and digest, the batch-one check, parameters and read-out multiply-adds, device, torch).
    python -m a1.read --code surface --ckpt CKPT --arm S-VA --seed 1 --kind dry --chunks 0-9 --out FILE [--export F]"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from a1 import sets


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_decision(code, ckpt, dev):
    """(model in eval mode without the training-only read-out, exported checkpoint dict, net module)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if code == "surface":
        from arch import net
        ex = net.export_decision(ck)
        model = net.net_from_cfg(ex["cfg"])
    else:
        from bb import net
        ex = net.export_decision(ck)
        model = net.build(ex["cfg"])
    model.load_state_dict(ex["state"])                      # strict: exactly the decision path's weights
    return model.to(dev).eval(), ex, net


@torch.no_grad()
def run(code, model, det, dev, bs):
    """-> (pred, conf, c or None) as numpy, fp32."""
    preds, confs, cs = [], [], []
    for i in range(0, len(det), bs):
        x = torch.as_tensor(det[i:i + bs], device=dev)
        if code == "surface":
            c, _ = model(x)
            c = c.float()
            preds.append((c > 0).to(torch.uint8).cpu())
            confs.append(c.abs().cpu())
            cs.append(c.cpu())
        else:
            s, _, _ = model(x)
            _, bits, mg = model.decide(s.float())
            preds.append(bits.cpu())
            confs.append(mg.float().cpu())
    cat = lambda v: torch.cat(v).numpy() if v else None
    return cat(preds), cat(confs), cat(cs)


def read_one(code, ckpt, arm, seed, kind, chunks, out, export="", batch=4096, b1=2000, cache=None):
    """One read (see the module doc); returns the batch-one check's number of differing decisions."""
    dev = torch.device("cuda")
    if seed:                                 # an arm's run: its checkpoint must be the arm's (gate finding 6)
        from a1 import podval
        rname = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))   # the run's directory = its --run
        if not (rname.endswith(f"_{arm}_s{seed}") and rname.split("_")[0] in ("a1", "a1ramp", "a1dry")):
            raise SystemExit(f"{ckpt}: not a checkpoint of {arm} seed {seed}")
        podval.check_identity(code, torch.load(ckpt, map_location="cpu", weights_only=False), rname, arm, seed,
                              samples=None if rname.startswith("a1dry") else podval.SAMPLES)
    model, ex, net = load_decision(code, ckpt, dev)
    if export:
        torch.save(dict(ex), export)
    key = (kind, chunks)
    if cache is None or key not in cache:
        got = sets.load(code, kind, sets.parse_range(chunks))
        if cache is not None:
            cache[key] = got
    else:
        got = cache[key]
    det, obs, digest = got
    pred, conf, c = run(code, model, det, dev, batch)
    p1, f1, _ = run(code, model, det[:b1], dev, 1)
    differ = int((p1 != pred[:b1]).reshape(len(p1), -1).any(1).sum())
    b1c = {"shots": int(len(p1)), "decisions_differ": differ,
           "max_abs_conf_diff": float(np.abs(f1.astype(np.float64) - conf[:b1]).max())}
    if code == "surface":
        hm = net.head_macs(ex["cfg"], model.m)
        size = {"params": net.n_params(model), "readout_macs": hm["inference"]["total"], "head_macs": hm}
    else:
        size = {"params": net.n_params(model), "readout_macs": net.readout_macs(model), "macs": net.macs(model)}
    meta = {"code": code, "arm": arm, "seed": seed, "ckpt": os.path.abspath(ckpt), "ckpt_sha256": sha_file(ckpt),
            "decision_sha256": sha_file(export) if export else "", "dropped": ex["dropped"],
            "cfg": {k: v for k, v in ex["cfg"].items() if "taps" not in k}, "kind": kind, "chunks": chunks,
            "shots": int(len(det)), "set_digest": digest, "b1_check": b1c, **size,
            "device": torch.cuda.get_device_name(), "torch": torch.__version__}
    pk = pred.reshape(len(pred), -1)
    meta["failures"] = int((pk != obs[:, :pk.shape[1]]).any(1).sum())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    assert out.endswith(".npz")
    tmp = out[:-4] + ".tmp.npz"              # renamed into place only when the batch-one check passes (finding 6)
    np.savez(tmp, pred=pred, conf=conf, **({"c": c} if c is not None else {}), meta=json.dumps(meta))
    os.replace(tmp, out[:-4] + ".failed.npz" if differ else out)
    print("READ", json.dumps({k: meta[k] for k in ("arm", "seed", "kind", "chunks", "shots", "failures", "b1_check",
                                                  "params", "readout_macs")}), flush=True)
    return differ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=["surface", "bb72"], required=True)
    ap.add_argument("--many", default="", help="a JSON list of reads {ckpt, arm, seed, kind, chunks, out[, export]}")
    ap.add_argument("--ckpt")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0, help="the arm's training seed (0: a reference network)")
    ap.add_argument("--kind")
    ap.add_argument("--chunks")
    ap.add_argument("--out")
    ap.add_argument("--export", default="", help="save the decision file (no training-only read-out) here")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--b1", type=int, default=2000, help="shots also run at batch one")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jobs = json.load(open(a.many)) if a.many else [dict(ckpt=a.ckpt, arm=a.arm, seed=a.seed, kind=a.kind,
                                                         chunks=a.chunks, out=a.out, export=a.export)]
    cache, bad = {}, []
    for j in jobs:
        if read_one(a.code, j["ckpt"], j["arm"], j["seed"], j["kind"], j["chunks"], j["out"], j.get("export", ""),
                    a.batch, a.b1, cache):
            bad.append(j["out"])
    if bad:
        raise SystemExit(f"batch one and batch {a.batch} differ in {bad}")


if __name__ == "__main__":
    main()
