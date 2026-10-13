"""A1 stage 1 on the pods (revision/A1_POD_RUN.md): the validation record of one run, what the stage-1 decision rule
(a1.rule) reads. The read is a1.read's (the training-only read-out dropped, strict reload of the decision path, fp32
without autocast and with TF32 off, batch 4,096, the first 2,000 shots again at batch one), on validation or
development shots, never on test shots:
  surface  val_legacy d = 9 chunks 0-19: 200,000 shots of the verdict circuit (icmp.shots, every chunk sha-checked
           against data/shot_manifest.csv). Chunk 0 is also the trainer's monitor set, which selects nothing (the
           record reads final_ema.pt).
  bb72     dev p = 0.3 % chunks 0-19: 200,000 shots of the registered BB72 circuit (bb.shots, sha-checked against
           data/bb/shot_manifest.csv).
Before reading, the checkpoint must be the arm's (gate finding 6): its head and auxiliary-loss weight are the arm's flags
in a1/jobs/args/common.sh (ARMS below, tested against it), its seed and run name match, it completed its 2e8 samples,
and its trunk start is the seed's registered trunk file (data/a1/trunks.txt). The record is written under a temporary
name and renamed to --out only when the batch-one check passes (no differing decision); otherwise it is kept as
<out>.failed.npz and the exit status is 1, so a failed record is never read as a good one.
Output npz: pred, conf (a1.read's confidence: |c| on the surface; on BB72 the joint margin, or min_k |logit_k| for the
pooled head), fail (uint8, 1 where the decision differs from the observables), c (surface), meta (JSON: a1.read's
fields, the set and its digest, the run's not-trained check and its stream record). Every arm reads the same shots in
the same order, so the records pair shot by shot.
    python -m a1.podval --code surface --ckpt CKPT --run a1_S-VA_s1 --arm S-VA --seed 1 --out FILE.npz"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from a1 import read

CHUNKS = range(20)
SAMPLES = 2e8
# arm -> (code, --head, auxiliary-loss weight: --line-weight on the surface, --translate-weight on BB72), common.sh's
ARMS = {"S-VA": ("surface", "vote", 1.0), "S-V": ("surface", "vote", 0.0), "S-L": ("surface", "line", 0.0),
        "S-P": ("surface", "pooled", 0.0), "S-PA": ("surface", "pooled", 1.0),
        "B-JT": ("bb72", "joint", 1.0), "B-J": ("bb72", "joint", 0.0), "B-P": ("bb72", "pooled", 0.0),
        "B-PA": ("bb72", "pooled", 1.0)}


def registered_trunk(code, seed, path=None):
    path = path or os.path.join(os.path.dirname(__file__), "..", "data", "a1", "trunks.txt")
    for line in open(path):
        f = line.split()
        if len(f) == 3 and f[0] == code and int(f[1]) == seed:
            return f[2]
    raise SystemExit(f"no registered trunk for {code} seed {seed}")


def check_identity(code, ck, run, arm, seed, trunks=None, samples=SAMPLES):
    """The checkpoint is arm `arm`'s completed run `run` of seed `seed` from the registered trunk, of `samples` samples
    (None: any, the dry runs); returns its facts."""
    want_code, head, w = ARMS[arm]
    if code != want_code:
        raise SystemExit(f"{arm} is a {want_code} arm, not {code}")
    hp = ck["hp"]
    aux = hp["line_weight"] if code == "surface" else hp["translate_weight"]
    init_sha = hp.get("init_sha256") if code == "surface" else hp.get("init_sha")
    done = ck["row"].get("completed") if code == "surface" else ck.get("completed")
    got = {"run": hp["run"], "head": hp["head"], "aux_weight": float(aux), "seed": hp["seed"],
           "samples": float(hp["samples"]), "completed": bool(done), "init_sha256": init_sha}
    want = {"run": run, "head": head, "aux_weight": w, "seed": seed, "samples": samples, "completed": True,
            "init_sha256": registered_trunk(code, seed, trunks)}
    if samples is None:
        want["samples"] = got["samples"]
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise SystemExit(f"{run}: the checkpoint is not the arm's completed run (got, want): {bad}")
    return got


def load_set(code):
    """(det (n, m) uint8, obs (n, k) uint8, description with the digest = sha256 of the chunk files' sha256 in order)."""
    if code == "surface":
        from icmp import shots
        det, obs = shots.load(shots.VAL, 9, CHUNKS)
        files = [shots.name(shots.VAL, 9, j) for j in CHUNKS]
        man = shots.manifest()
        obs = obs.reshape(len(obs), 1)
        desc = {"split": shots.VAL, "d": 9, "p": 0.006}
    else:
        from bb import shots
        det, obs = shots.load_many("bb72", 0.003, "dev", CHUNKS)
        files = [shots.name("bb72", 0.003, "dev", j) for j in CHUNKS]
        man = shots._manifest()
        desc = {"split": "dev", "code": "bb72", "p": 0.003}
    digest = hashlib.sha256(",".join(man[f]["sha256"] for f in files).encode()).hexdigest()
    return det, obs, {**desc, "chunks": f"{CHUNKS[0]}-{CHUNKS[-1]}", "shots": int(len(det)), "digest": digest}


def run_facts(code, run_dir, ck):
    """The run's not-trained check and stream record as its trainer wrote them."""
    if code == "surface":
        f = os.path.join(run_dir, "train_check.json")
        return (json.load(open(f)) if os.path.exists(f) else None), ck["row"].get("stream_chain")
    f = os.path.join(run_dir, "log.json")
    lj = json.load(open(f)) if os.path.exists(f) else {}
    return lj.get("train_check"), lj.get("stream_chain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=["surface", "bb72"], required=True)
    ap.add_argument("--ckpt", required=True, help="the run's final_ema.pt")
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="FILE.npz")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--b1", type=int, default=2000)
    a = ap.parse_args()
    assert a.out.endswith(".npz")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ident = check_identity(a.code, ck, a.run, a.arm, a.seed)
    tc, chain = run_facts(a.code, os.path.dirname(os.path.abspath(a.ckpt)), ck)
    if tc is None:
        raise SystemExit(f"{a.run}: no not-trained check in the run's records")
    det, obs, desc = load_set(a.code)
    model, ex, net = read.load_decision(a.code, a.ckpt, dev)
    pred, conf, c = read.run(a.code, model, det, dev, a.batch)
    p1, f1, _ = read.run(a.code, model, det[:a.b1], dev, 1)
    differ = int((p1 != pred[:a.b1]).reshape(len(p1), -1).any(1).sum())
    pk = pred.reshape(len(pred), -1)
    fail = (pk != obs[:, :pk.shape[1]]).any(1).astype(np.uint8)
    if a.code == "surface":
        size = {"params": net.n_params(model), "readout_macs": net.head_macs(ex["cfg"], model.m)["inference"]["total"]}
    else:
        size = {"params": net.n_params(model), "readout_macs": net.readout_macs(model)}
    meta = {"code": a.code, "run": a.run, "arm": a.arm, "seed": a.seed, "ckpt": os.path.abspath(a.ckpt),
            "ckpt_sha256": read.sha_file(a.ckpt), "identity": ident, "dropped": ex["dropped"], "set": desc,
            "failures": int(fail.sum()),
            "b1_check": {"shots": int(len(p1)), "decisions_differ": differ,
                         "max_abs_conf_diff": float(np.abs(f1.astype(np.float64) - conf[:a.b1]).max())},
            "train_check": tc, "stream_chain": chain, **size,
            "device": torch.cuda.get_device_name(), "torch": torch.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out[:-4] + ".tmp.npz"
    np.savez(tmp, pred=pred, conf=conf, fail=fail, **({"c": c} if c is not None else {}), meta=json.dumps(meta))
    print("VALREC", json.dumps({k: meta[k] for k in ("run", "failures", "b1_check", "params", "readout_macs")}),
          flush=True)
    if differ:
        os.replace(tmp, a.out[:-4] + ".failed.npz")
        raise SystemExit(f"batch one and batch {a.batch} differ on {differ} shots: the record is kept as .failed.npz")
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
