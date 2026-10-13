"""STAGE4_GROSS_PLAN.md section 2: PLAN 3.3's size rule (bb.route.power, the full registered decision) for cells G-4R
and G-4F on CAL's paired counts of the fusedt read (tau = the q-quantile of its CAL margins), with worst cases where CAL
has no full read of an opponent: no shared failure for the two BP-OSD opponents (stage-1 dev rates), and for full-DEM
Relay-BP the hybrid's worst case of GROSS_FULLDEM_CAL.md (full-DEM Relay-BP counted right on every CAL shot the full-DEM CAL
study did not decode). Per comparison the target power is 1 - 0.10 / (the cell's comparisons); n_cell = the largest of its
comparisons' smallest passing n (multiples of 10,000 from 100,000 to 300,000).
Usage: python stage4_gross_sizes.py <repo root> <preds dir> <tag> <out.json>
(preds: bb144_p0.004_cal000j_net-<tag>.npz and bb144_p0.004_cal000j_relay.npz, j = 0..9)"""
import hashlib
import json
import os
import sys

import numpy as np

root, preds, tag, out_path = sys.argv[1:5]
sys.path[:0] = [os.path.join(root, "ising_compare"), os.path.join(root, "qldpc_neural")]
from bb import route  # noqa: E402

inputs, nf, rf, mg = {}, [], [], []
for j in range(10):
    for name in (f"bb144_p0.004_cal{j:04d}_net-{tag}.npz", f"bb144_p0.004_cal{j:04d}_relay.npz"):
        path = os.path.join(preds, name)
        inputs[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        z = np.load(path)
        (rf if name.endswith("_relay.npz") else nf).append(z["fail"])
        if "margin" in z.files:
            mg.append(z["margin"])
nf, rf, mg = np.concatenate(nf), np.concatenate(rf), np.concatenate(mg)
N = len(nf)
full = np.load(os.path.join(root, "ising_compare", "data", "bb", "fullroute_gross", "bb144_p0.004_cal_pick_relay_full.npz"))
decoded = np.zeros(N, bool)
decoded[full["idx"]] = True
ffail = np.zeros(N, bool)                         # full-DEM Relay-BP's failures; undecoded shots count as right
ffail[full["idx"]] = full["fail"]


def counts(h, o):
    return int((h & o).sum()), int((h & ~o).sum()), int((o & ~h).sum()), N


def need(cnt, target):
    return next((n for n in range(100_000, 300_001, 10_000) if route.power(cnt, n) >= target), None)


cal, scen = {}, {}
for cell, q in (("G-4R", 0.01), ("G-4F", 0.02)):
    tau = float(np.quantile(mg, q))
    r = mg < tau
    if cell == "G-4R":
        hyb = np.where(r, rf, nf)
    else:
        assert decoded[r].all(), "a routed CAL shot outside the full-DEM CAL study's read"
        hyb = np.where(r, ffail, nf)
    cal[cell] = {"q": q, "tau": tau, "routed": int(r.sum()), "hybrid": int(hyb.sum()), "net_alone": int(nf.sum()),
                 "relay": int(rf.sum()), "vs_relay": counts(hyb, rf)}
    a, b, c, _ = counts(hyb, rf)
    shift = int(round((b + c) ** 0.5 / 2))        # half the standard error of the discordant difference c - b
    s = {"relay (CAL paired counts)": (a, b, c, N),
         "relay, difference one SE smaller": (a, b + shift, c - shift, N),
         "relay, worst case (no shared failure)": (0, int(hyb.sum()), int(rf.sum()), N),
         "bposd, worst case (stage-1 dev 523 / 20,000)": (0, int(hyb.sum()), 2615, N),
         "gpu_bposd, worst case (stage-1 dev 515 / 20,000)": (0, int(hyb.sum()), 2575, N)}
    if cell == "G-4F":
        s = {"relay_full, worst case on CAL (GROSS_FULLDEM_CAL.md 5)": counts(hyb, ffail), **s}
        cal[cell]["vs_relay_full_worst"] = counts(hyb, ffail)
    k = 4 if cell == "G-4F" else 3
    target = 1 - 0.10 / k
    scen[cell] = {"per_comparison_power": target, "comparisons": {}}
    for name, cnt in s.items():
        scen[cell]["comparisons"][name] = {"counts_both_onlyhybrid_onlyopp_shots": list(cnt), "n": need(cnt, target),
                                          "power_at_150000": route.power(cnt, 150_000),
                                          "power_at_300000": route.power(cnt, 300_000)}
    ns = [v["n"] for v in scen[cell]["comparisons"].values()]
    scen[cell]["n_rule"] = None if None in ns else max(ns)
n = max(150_000, scen["G-4R"]["n_rule"] or 300_000)
res = {"rule": "bb.route.power: exact two-sided McNemar p < 0.05/3 and the paired Newcombe (method 10) (1 - 0.05/3) "
               "interval below 0; 20,000 draws, seed 20261006; target 1 - 0.10 / comparisons per comparison",
       "tag": tag, "cal": cal, "cells": scen, "n": n,
       "decision": f"{n:,} shots: 150,000 (GROSS_FULLDEM_CAL.md 5) unless G-4R's rule needs more", "inputs": inputs}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps({"cal": cal, "n": {k: v["n_rule"] for k, v in scen.items()}, "test_n": n,
                  "power150": {k: {c: round(v["power_at_150000"], 4) for c, v in s["comparisons"].items()}
                               for k, s in scen.items()}}, indent=1))
