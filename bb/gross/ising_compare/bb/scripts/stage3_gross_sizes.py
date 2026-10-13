"""STAGE3_GROSS_PLAN.md section 2: PLAN 3.3's size rule (bb.route.power, the full registered decision) for cell G-4,
on CAL's paired counts of the frozen hybrid (g2_D12k113c10, tau of q = 0.5 %) against sector Relay-BP, the only
opponent read on CAL, and on worst cases (no shared failure) for the two BP-OSD opponents from their stage-1 dev rates.
Usage: python stage3_gross_sizes.py <repo root> <net preds dir> <relay preds dir>
(net: bb144_p0.004_{cal0000..0009,monitor0010..0014}_net-g2_D12k113c10.npz; relay: bb144_p0.004_cal*_relay.npz)"""
import hashlib
import json
import os
import sys

import numpy as np

root, net_dir, relay_dir = sys.argv[1:4]
sys.path[:0] = [os.path.join(root, "ising_compare"), os.path.join(root, "qldpc_neural")]
from bb import route  # noqa: E402

TAU = 0.5629201531410217                       # frontierf_g2_D12k113c10.json, policy q = 0.005
NET = "bb144_p0.004_{}_net-g2_D12k113c10.npz"


def load(path, keys):
    z = np.load(path)
    return [z[k] for k in keys], hashlib.sha256(open(path, "rb").read()).hexdigest()


inputs, nf, rf, mg = {}, [], [], []
for j in range(10):
    (f, m), h = load(os.path.join(net_dir, NET.format(f"cal{j:04d}")), ("fail", "margin"))
    (r,), hr = load(os.path.join(relay_dir, f"bb144_p0.004_cal{j:04d}_relay.npz"), ("fail",))
    nf.append(f), mg.append(m), rf.append(r)
    inputs[NET.format(f"cal{j:04d}")], inputs[f"bb144_p0.004_cal{j:04d}_relay.npz"] = h, hr
nf, rf, mg = np.concatenate(nf), np.concatenate(rf), np.concatenate(mg)
routed = mg < TAU
hyb = np.where(routed, rf, nf)
N = len(hyb)
cal = {"shots": N, "routed": int(routed.sum()), "net_alone": int(nf.sum()), "relay": int(rf.sum()),
       "hybrid": int(hyb.sum()), "both": int((hyb & rf).sum()), "only_hybrid": int((hyb & ~rf).sum()),
       "only_relay": int((rf & ~hyb).sum()), "net_fail_not_routed": int((nf & ~routed).sum())}

df, dm = [], []                                   # D-bank: out of sample for tau (no opponent read there)
for j in range(10, 15):
    (f, m), h = load(os.path.join(net_dir, NET.format(f"monitor{j:04d}")), ("fail", "margin"))
    df.append(f), dm.append(m)
    inputs[NET.format(f"monitor{j:04d}")] = h
df, dm = np.concatenate(df), np.concatenate(dm)
dbank = {"shots": len(df), "net_alone": int(df.sum()), "routed": int((dm < TAU).sum()),
         "net_fail_not_routed": int((df & (dm >= TAU)).sum())}

b, c, a = cal["only_hybrid"], cal["only_relay"], cal["both"]
se = (b + c) ** 0.5                               # one standard error of the discordant difference c - b on CAL
shift = int(round(se / 2))
SCEN = {"relay (CAL paired counts)": (a, b, c, N),
        "relay, difference one SE smaller": (a, b + shift, c - shift, N),
        "relay, worst case (no shared failure)": (0, cal["hybrid"], cal["relay"], N),
        "bposd, worst case (stage-1 dev 523 / 20,000)": (0, cal["hybrid"], 2615, N),
        "gpu_bposd, worst case (stage-1 dev 515 / 20,000)": (0, cal["hybrid"], 2575, N)}
cell = ("bb144", 0.004)
target = 1 - (1 - route.POWER) / len(route.GATING)
grid = list(range(route.FLOOR[cell], route.CAP["bb144"] + 1, 10_000))
out = {"rule": "bb.route.power: exact two-sided McNemar p < 0.05/3 and the paired Newcombe (method 10) (1 - 0.05/3) "
               "interval entirely below 0; 20,000 multinomial draws, seed 20261006",
       "tau": TAU, "per_comparison_power_target": target, "floor": grid[0], "cap": grid[-1],
       "cal": cal, "dbank": dbank, "inputs_sha256": inputs, "scenarios": {}}
for name, cnt in SCEN.items():
    pw = {n: route.power(cnt, n) for n in grid}
    out["scenarios"][name] = {"counts_per_N": list(cnt), "n_rule": next((n for n in grid if pw[n] >= target), None),
                              "power": {str(n): round(v, 4) for n, v in pw.items()}}
print(json.dumps(out, indent=1))
