"""Batch-1 GPU term and handoff of ConvLines-BB (bb/DESIGN.md 5.3 (a) and (c)) in one process: the deployed engine's
call (bb.engine: detector bits on the GPU -> decision and margin on the GPU, one CUDA graph) under SETTINGS 5.1's replay
protocol (200 warm-up replays, then 5 repetitions of 100 timed replays, CUDA events; the process's value = the mean of
the 500 calls), and the host round trip of one shot's transfers. The protocols run it in several fresh processes and
count the slowest (bb/PLAN.md 2.7). Acceptance (SETTINGS' rule): nvidia-smi samples the SM clock every 100 ms while
the replays and the handoff run, and both rows are accepted iff every sample is at least 97 % of the maximum clock and
no other process uses the GPU; bb.timing gate counts accepted rows only. Both rows are bound to the engine signature
(bb.timing.sig_digest), the session and this process. With --design instead of --ckpt the weights are random (the
computation does not depend on them): the engine port's early timing check. Rows go to bb.timing's CSV.
  python -m bb.latency --ckpt final_ema.pt --engine graph --proc 1 --out T.csv
  python -m bb.latency --code bb72 --design design_bb72.json --engine inductor --proc 1 --out T.csv"""
import argparse
import json

import numpy as np
import torch

from bb import engine, net
from bb.timing import append_row, gpu_watch, sig_digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--code")
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--design")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bott", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--tblocks", default="", help="gross2: time-block dilations (random weights only)")
    ap.add_argument("--rbins", default="", help="gross2: read-out branch R time-bin edges (random weights only)")
    ap.add_argument("--engine", default="graph", choices=["graph", "graph32", "inductor", "fused"])
    ap.add_argument("--proc", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.ckpt:
        model, ck = engine.load_net(a.ckpt, dev, a.allow_incomplete)
        sha, code, p = engine.sha256_file(a.ckpt), ck["cfg"]["code"], a.p
    else:
        stem, blk = net.taps_from_design(a.design)
        model = net.ConvLinesBB(a.code, a.p, stem, blk, a.width, a.bott, a.blocks,
                                tblocks=[int(x) for x in a.tblocks.split(",") if x],
                                rbins=[int(x) for x in a.rbins.split(",") if x]).to(dev).eval()
        sha, code, p = "random" + "0" * 6, a.code, a.p
    eng = engine.Deployed(model, a.engine, sha)
    stop = gpu_watch()
    reps = eng.time_calls()
    ho = eng.handoff_us()                                  # inside the watched window too
    acc = stop()
    sig = eng.signature()
    base = {"device": sig["gpu"], "code": code, "p": p, "name": f"{a.engine}:{sha[:12]}", "proc": a.proc,
            "binding": sig_digest(sig)}
    append_row(a.out, dict(base, role="gpu_term", n=500, value_ms=float(np.mean(reps)) / 1e3,
                           note=json.dumps({"rep_us": reps, "params": net.n_params(model), **acc})))
    append_row(a.out, dict(base, role="handoff", n=1000, value_ms=ho / 1e3, note=json.dumps(acc)))
    print("latency", json.dumps({"engine": sig, "rep_us": reps, "mean_us": float(np.mean(reps)), "handoff_us": ho,
                                 **acc}), flush=True)


if __name__ == "__main__":
    main()
