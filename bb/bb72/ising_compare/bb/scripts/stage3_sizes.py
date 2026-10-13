"""STAGE3_PLAN.md section 2: PLAN 3.3's size rule (bb.route.power, the full registered decision) on the worst case of the
published dev counts (no shared failure), at each registered cell's floor. Usage: python stage3_sizes.py <repo root>"""
import json
import os
import sys

root = sys.argv[1]
sys.path[:0] = [os.path.join(root, "ising_compare"), os.path.join(root, "qldpc_neural")]
from bb import route  # noqa: E402

DEV = {"BB72-3": (("bb72", 0.003), 1327, {"relay": 2747, "bposd": 3308, "gpu_bposd": 3366}),   # r11, dev failures
       "BB72-2": (("bb72", 0.002), 269, {"relay": 773, "bposd": 911, "gpu_bposd": 895})}       # r12, dev failures
target = 1 - (1 - route.POWER) / len(route.GATING)
out = {"rule": "bb.route.power: exact two-sided McNemar p < 0.05/3 and the paired Newcombe (method 10) (1 - 0.05/3) "
               "interval entirely below 0; 20,000 multinomial draws, seed 20261006",
       "worst_case": "both fail = 0, only ours = ours' dev failures, only the opponent's = its dev failures",
       "per_comparison_power_target": target, "cells": {}}
for cell, (key, ours, opps) in DEV.items():
    floor = route.FLOOR[key]
    res = {o: {"dev_counts": [0, ours, f, floor], "power_at_floor": route.power((0, ours, f, floor), floor)}
           for o, f in opps.items()}
    out["cells"][cell] = {"floor": floor, "opponents": res,
                          "n": floor if all(r["power_at_floor"] >= target for r in res.values()) else None}
print(json.dumps(out, indent=1))
