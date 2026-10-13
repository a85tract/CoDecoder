"""Inference throughput of Cascade-surface with random weights in eval mode (bf16 autocast, optionally compiled):
shots/s at each batch size, timed by wall clock over --secs after 5 warm-up passes, shots resident on the GPU. The
compute estimate's evaluation term (cascade_compare/SURFACE_PLAN.md section 6); the comparison's speed protocol
(phase C's: CUDA graphs, host round trips, three processes) is a separate step.
    python -m surface.bench_infer --d 13 --L 13 --H 256 [--compile] [--batches 1,256,1024,4096]"""
import argparse
import json
import time

import torch

from surface import surface_data as sdata
from surface import surface_layout as sl
from surface import surface_net as snet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=13)
    ap.add_argument("--L", type=int, default=13)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--batches", default="1,256,1024,4096")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--secs", type=float, default=10.0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    lay = sl.Layout(a.d)
    m = snet.build({"d": a.d, "R": lay.R, "H": a.H, "L": a.L}, lay).to(dev).eval()
    f = torch.compile(m) if a.compile else m
    bs = [int(b) for b in a.batches.split(",")]
    det, _ = sdata.sample(a.d, 0.006, max(bs), sdata.eval_seed(a.d, 2, 0.006, 10))
    x_all = torch.as_tensor(det, device=dev)
    for B in bs:
        x = x_all[:B]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                f(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            n, t0 = 0, time.time()
            while time.time() - t0 < a.secs:
                f(x)
                n += 1
            torch.cuda.synchronize()
            dt = time.time() - t0
        print("INFER " + json.dumps({"d": a.d, "L": a.L, "H": a.H, "batch": B, "compile": a.compile,
                                     "device": torch.cuda.get_device_name(dev), "passes": n,
                                     "shots_per_s": round(n * B / dt, 1), "us_per_shot": round(1e6 * dt / (n * B), 2),
                                     "ms_per_pass": round(1e3 * dt / n, 3),
                                     "max_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)}),
              flush=True)


if __name__ == "__main__":
    main()
