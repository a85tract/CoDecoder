"""Evaluate decoders on fresh shots (bb.cascade_data.eval_shots: seeds eval_seed(kind, p, chunk), Stim's circuit sampler
at circuit_at(code, p)). Every model decodes the same rows, so the comparisons are paired. --family picks the circuit
family (cascade_data: uniform, the default, or the noise-model fit's depspam_fm, with its own seeds).

  python -m bb.cascade_eval --ckpt A.pt[,B.pt,...] --p 0.001,0.002,... --chunks 0:100 [--kind 0] [--save DIR]

Prediction adapter (stage 2): a checkpoint is a Cascade network (bb.cascade_net,
12 logits) or a ConvLines-BB network (bb.net, the r11 / r12 / S55 files of bb.train: 4,096 class scores), told apart by
its cfg. Each becomes, per shot, a decision (12 bits in Stim's observable order) and a confidence:
  cascade    bits = (logit > 0); confidence = min over the observables of |logit|; the logits are kept (float16)
  convlines  class = argmax of the scores and bits = Ybits[class] (bit j of the class index is observable j: the
             registered decoder's conversion, bb.net.ConvLinesBB.decide, used as is); confidence = the top-1 minus
             top-2 score (the registered margin); the class index is kept
Precision (--prec): auto = the cascade networks under bf16 autocast (their training precision) and the ConvLines
networks in fp32 with TF32 off and cuDNN deterministic, in batches of 2,048 (bb.engine's fp32 path, the reference of
the earlier curve study); bf16 or fp32 force one for all. A shot fails when any of its 12 bits differs from the observables.

Per model and p: shots, block failures, P_block with its 95 % Wilson interval, P_L per logical qubit per round by the
paper's Eq. (1) (k = 12, R = 6) at the estimate and at both interval ends, and the signature (checkpoint sha256, kind,
weights, precision, circuit sha256, GPU, versions, code tree). --save DIR writes per p DIR/p<p>.npz: the shot ids
(chunk, row), obs (packed), and per model i pred<i> (packed bits), conf<i> (float32), fail<i>, plus z<i> (cascade
logits, float16) or cls<i> (ConvLines class, uint16); and DIR/p<p>.json: the signatures, the chunks' seeds and the
sha256 of their packed detector rows. --save-det also keeps the packed detector rows (det, numpy's packbits with
bitorder "little"), so that a later CPU decoder reads the same rows (Stim's samples for a seed depend on the host's
SIMD width). Shots are drawn in chunks of 10,000 by --procs processes, at most 2 --procs chunks ahead of the GPU;
each chunk goes to the device once for all models."""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from bb import cascade_data as cdata
from bb import cascade_net as cnet
from bb.cascade_stats import p_block_to_pl, wilson


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(path, dev, which="ema"):
    """A Cascade checkpoint -> (CascadeBB in eval mode, the checkpoint dict) (kept for stage-1 callers)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = cnet.build(ck["cfg"])
    m.load_state_dict(ck[which] if which in ck else ck["ema"])
    return m.to(dev).eval(), ck


class Predictor:
    """One decoder: __call__(det on the device, uint8 (n, 432)) -> dict(bits (n, 12) uint8, conf (n,) float32, and z
    or cls), all on the device."""

    def __init__(self, path, dev, which="ema", prec="auto"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        self.path, self.dev = path, dev
        self.kind = "convlines" if cfg.get("arch") == "convlines_bb" else "cascade"
        if self.kind == "convlines":
            from bb import net
            self.model = net.build(cfg)
            if "ema" in ck:                              # a bb.train last.pt: "ema" and the raw "state"
                key = "ema" if which == "ema" else "state"
            else:                                        # final_ema.pt / budget_ema.pt: "state" is the EMA network
                key = "state"
            self.prec = "fp32" if prec == "auto" else prec
            self.bs = 2048
        else:
            self.model = cnet.build(cfg)
            key = which if which in ck else "ema"
            self.prec = "bf16" if prec == "auto" else prec
            self.bs = 8192
        self.model.load_state_dict(ck[key])
        self.model = self.model.to(dev).eval()
        self.weights = key
        self.ck = ck
        self.sig = {"path": os.path.abspath(path), "sha256": sha256_file(path), "kind": self.kind, "weights": key,
                    "prec": self.prec, "batch": self.bs, "cfg": {k: v for k, v in cfg.items() if "taps" not in k},
                    "step": ck.get("step"), "samples": ck.get("samples"), "status": ck.get("status")}

    @torch.no_grad()
    def __call__(self, det, bs=None):
        bs = bs or self.bs
        outs = []
        for i in range(0, len(det), bs):
            d = det[i:i + bs]
            with torch.autocast(d.device.type, dtype=torch.bfloat16, enabled=self.prec == "bf16" and d.is_cuda):
                if self.kind == "cascade":
                    z = self.model(d).float()
                    outs.append({"bits": (z > 0).to(torch.uint8), "conf": z.abs().min(1).values, "z": z})
                else:
                    s = self.model(d)[0].float()
                    cls, bits, margin = self.model.decide(s)
                    outs.append({"bits": bits, "conf": margin.float(), "cls": cls})
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def row(fail, n, k=12, R=6):
    lo, hi = wilson(fail, n)
    return {"shots": n, "fail": fail, "p_block": fail / n, "pb_lo": lo, "pb_hi": hi,
            "pl": p_block_to_pl(fail / n, k, R), "pl_lo": p_block_to_pl(lo, k, R), "pl_hi": p_block_to_pl(hi, k, R)}


def chunks_in_order(pool, jobs, ahead):
    """pool.apply_async over jobs with at most `ahead` outstanding, yielded in order (bounded prefetch)."""
    pend = collections.deque()
    it = iter(jobs)
    for j in it:
        pend.append(pool.apply_async(cdata._eval_chunk, (j,)))
        if len(pend) >= ahead:
            break
    while pend:
        out = pend.popleft().get()
        nxt = next(it, None)
        if nxt is not None:
            pend.append(pool.apply_async(cdata._eval_chunk, (nxt,)))
        yield out


def env_sig(dev):
    tree = ""
    for f in ("../DEPLOYED", "DEPLOYED"):
        if os.path.exists(f):
            tree = open(f).read().strip()
    return {"gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
            "cuda": torch.version.cuda, "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "tree": tree}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="comma-separated checkpoints (Cascade or ConvLines-BB)")
    ap.add_argument("--which", default="ema", choices=["ema", "model"], help="weights of a last.pt (EMA or raw)")
    ap.add_argument("--prec", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--code", default="bb72")
    ap.add_argument("--family", default="uniform", choices=list(cdata.FAMILIES), help="circuit family (cascade_data)")
    ap.add_argument("--p", required=True)
    ap.add_argument("--chunks", default="0:10", help="a:b = chunks a .. b-1 of 10,000 shots")
    ap.add_argument("--kind", type=int, default=0, help="eval_seed kind (0 evaluation, 2 smoke, 5 pilots, 6 checks)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=0, help="batch size (0 = 8,192 for Cascade, 2,048 for ConvLines)")
    ap.add_argument("--bn-recal", type=int, default=0, help="recompute the Cascade networks' BN statistics on this "
                    "many fresh shots at --bn-recal-p (seed kind 4) before evaluating (0 = the checkpoint's)")
    ap.add_argument("--bn-recal-p", type=float, default=0.0055)
    ap.add_argument("--save", default="")
    ap.add_argument("--save-det", action="store_true", help="with --save: keep the packed detector rows too")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False         # bb.engine's settings (the registered fp32 reference)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    paths = a.ckpt.split(",")
    preds = [Predictor(pth, dev, a.which, a.prec) for pth in paths]
    if a.bn_recal:
        from bb.cascade_optim import recalibrate_bn
        det, _ = cdata.eval_shots(a.code, a.bn_recal_p, range((a.bn_recal + 9999) // 10000), kind=4, procs=a.procs,
                                  family=a.family)
        batches = [torch.as_tensor(det[i:i + 3328], device=dev) for i in range(0, a.bn_recal, 3328)]
        for pr in preds:
            if pr.kind == "cascade":
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    recalibrate_bn(pr.model, batches)
                pr.sig["bn_recal"] = [a.bn_recal, a.bn_recal_p]
    c0, c1 = (int(x) for x in a.chunks.split(":"))
    env = env_sig(dev)
    res = []
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for p in (float(x) for x in a.p.split(",")):
            t0 = time.time()
            circ_sha = hashlib.sha256(str(cdata.circuit_at(a.code, p, a.family)).encode()).hexdigest()
            fails = [0] * len(preds)
            secs = [0.0] * len(preds)
            n = 0
            keep = collections.defaultdict(list)
            chunk_meta = []
            jobs = [(a.code, a.kind, p, j, 10000, a.family) for j in range(c0, c1)]
            for j, (det, obs) in zip(range(c0, c1), chunks_in_order(pool, jobs, 2 * a.procs)):
                n += len(det)
                dg = torch.as_tensor(det, device=dev)       # once per chunk, for every model
                og = torch.as_tensor(obs, device=dev)
                for i, pr in enumerate(preds):
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    ts = time.time()
                    out = pr(dg, a.bs or None)
                    fail = (out["bits"] != og).any(1)
                    fails[i] += int(fail.sum())
                    secs[i] += time.time() - ts
                    if a.save:
                        keep[f"pred{i}"].append(np.packbits(out["bits"].cpu().numpy(), axis=1))
                        keep[f"conf{i}"].append(out["conf"].float().cpu().numpy())
                        keep[f"fail{i}"].append(fail.cpu().numpy())
                        if "z" in out:
                            keep[f"z{i}"].append(out["z"].cpu().numpy().astype(np.float16))
                        else:
                            keep[f"cls{i}"].append(out["cls"].cpu().numpy().astype(np.uint16))
                if a.save:
                    keep["obs"].append(np.packbits(obs, axis=1))
                    if a.save_det:
                        keep["det"].append(np.packbits(det, axis=1, bitorder="little"))
                    keep["chunk"].append(np.full(len(det), j, np.int32))
                    keep["row"].append(np.arange(len(det), dtype=np.int32))
                    chunk_meta.append({"chunk": j, "seed": cdata.eval_seed(a.kind, p, j, a.family),
                                       "det_sha256": hashlib.sha256(np.packbits(det, axis=1).tobytes()).hexdigest()[:16]})
            sigs = []
            for i, pr in enumerate(preds):
                sig = dict(pr.sig, circuit_sha256=circ_sha, **env)
                sigs.append(sig)
                r = {"model": pr.path, "which": pr.weights, "kind": pr.kind, "prec": pr.prec, "sha256": sig["sha256"],
                     "bn_recal": a.bn_recal if pr.kind == "cascade" else 0, "p": p, "family": a.family,
                     "kind_seed": a.kind,
                     "chunks": [c0, c1], **row(fails[i], n), "secs_model": round(secs[i], 1),
                     "secs": round(time.time() - t0, 1)}
                res.append(r)
                print(json.dumps(r), flush=True)
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                np.savez(os.path.join(a.save, f"p{p:.5f}.npz"), chunks=np.array([c0, c1]), models=np.array(paths),
                         **{k: np.concatenate(v) for k, v in keep.items()})
                with open(os.path.join(a.save, f"p{p:.5f}.json"), "w") as fh:
                    json.dump({"p": p, "kind": a.kind, "chunks": [c0, c1], "code": a.code, "family": a.family,
                               "circuit_sha256": circ_sha,
                               "models": sigs, "chunk_meta": chunk_meta}, fh, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
