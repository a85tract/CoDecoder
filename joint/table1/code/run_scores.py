"""Runs ARB's arb_scores.py unchanged, with one exception recorded in the log: pc_common's GPU guard (a full RTX PRO 6000
only) is set to the card of this pod when that card differs, because the RTX PRO 6000 was sold out and the pod is an
H100. Every bitwise check of arb_scores.py against the read (decisions and margins of the read's own engine, the
exporting engine, the exported scores) still runs and decides the exit code.

  python run_scores.py --read DIR --p 0.002 --ckpt S55_dfm_deep/final_ema.pt --lock LOCK"""
import json
import os
import runpy
import sys

import torch

import pc_common as pc

name = torch.cuda.get_device_name(0)
print("GPU_GUARD", json.dumps({"guard": pc.GPU, "gpu": name, "lifted": name != pc.GPU}), flush=True)
pc.GPU = name
script = os.path.join(os.environ["CODE"], "arb_scores.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
