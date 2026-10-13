"""NVIDIA's GPU BP-OSD (cudaq-qec nv-qldpc-decoder) on stored BB shots (bb/DESIGN.md 5.2, 5.5; qldpc_neural
bench/gpu_bposd.py on this route's shots). Configuration ms80_cs60 = min-sum (scale 1.0), 80 iterations, OSD-CS
(osd_method 3) order 60, X sector: the GPU twin of the CPU accuracy reference. Two separate commands (the cudaq venv:
Alpine ising/venv_cudaq, gpu.sbatch's $PYQ):
  read  python -m bb.gpu_bposd read --code bb72 --p 0.003 --split dev --chunks 0-19 --out rows.csv: the predictions of
        every chunk (BB_PREDS, name gpu_bposd, for the paired tests), written only when the whole read has finished,
        every chunk file carrying the read's id; bb.opp.load_preds refuses chunks of different reads, so a rerun
        replaces the whole read (cudaq-qec's decisions vary between processes). Dev and test reads are blind (no
        failure count printed or recorded, as bb.opp).
  time  python -m bb.gpu_bposd time --code bb72 --p 0.003 --split dev --chunks 0 --batches 1,16,256,4096 --out rows.csv
        [--timing-out T.csv --proc K]: host-to-host time per call by batch size on these shots; writes no predictions.
        --timing-out adds the batch-1 mean per call (--reps1 calls, watched by bb.timing.gpu_watch) as an opponent row
        of bb.timing's CSV."""
import argparse
import csv
import datetime
import json
import os
import platform
import time
import uuid

import numpy as np

from bb import opp, shots
from qnd import circuits

CONFIGS = {"ms80_cs60": dict(bp_method=1, scale_factor=1.0, max_iterations=80, use_osd=True, osd_method=3, osd_order=60)}
FIELDS = ["date", "host", "gpu", "code", "p", "decoder", "params", "what", "read", "split", "chunks", "batch", "shots",
          "block_failures", "ler", "mean_call_ms", "p50_call_ms", "p99_call_ms", "us_per_shot", "shots_per_s", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "time"])
    ap.add_argument("--code", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--first", type=int, default=0, help="only the first N shots of each chunk (read) or in all (time)")
    ap.add_argument("--config", default="ms80_cs60")
    ap.add_argument("--batches", default="1,16,256,4096")
    ap.add_argument("--reps1", type=int, default=300, help="calls timed at batch 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing-out", default="")
    ap.add_argument("--proc", type=int, default=1)
    a = ap.parse_args()
    gpu = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().split("\n")[0].strip()
    import cudaq_qec as qec
    H, L, q, keep = circuits.problem(a.code, "X", a.p, "basis")
    Hd = H.toarray().astype(np.uint8)
    Lr = L.tocsr()
    base = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(), "gpu": gpu,
            "code": a.code, "p": a.p, "decoder": f"nv-qldpc-decoder {a.config}", "params": str(CONFIGS[a.config])}

    def make(bs):
        return qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=list(map(float, q)), use_sparsity=True,
                               bp_batch_size=bs, **CONFIGS[a.config])

    def decode(dec, s):
        res = dec.decode_batch(np.ascontiguousarray(s, dtype=np.float32))
        e = np.array([np.asarray(r.result) for r in res]) > 0.5
        return ((Lr @ e.T.astype(np.uint8)).T % 2).astype(np.uint8)

    rows = []
    if a.cmd == "read":
        blind = a.split in ("dev", "test", "test4")
        rid = f"{os.environ.get('SLURM_JOB_ID', '')}:{platform.node()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        suffix = f"_first{a.first}" if a.first else ""
        dec = make(1000)
        done, fails, n = [], 0, 0
        for j in shots.parse_range(a.chunks):
            det, obs = shots.load(a.code, a.p, a.split, j)
            det, obs = det[:a.first or None], obs[:a.first or None]
            s = det[:, keep]
            pred = np.concatenate([decode(dec, s[i:i + 1000]) for i in range(0, len(s), 1000)])
            fail = (pred != obs).any(1)
            path = opp.pred_path(a.code, a.p, a.split, j, "gpu_bposd" + suffix)
            np.savez(path + ".part.npz", pred=pred, fail=fail, secs=np.zeros(len(fail), np.float32),
                     conv=np.ones(len(fail), bool), read=rid, chunks=a.chunks)
            done.append(path)
            fails += int(fail.sum())
            n += len(fail)
            print(f"chunk {j}: {'blind' if blind else f'{int(fail.sum())} / {len(fail)} failures'}", flush=True)
        for path in done:                                  # the whole read replaces the stored one
            os.replace(path + ".part.npz", path)
        rows.append({**base, "what": "accuracy", "read": rid, "split": a.split, "chunks": a.chunks, "batch": 1000,
                     "shots": n, "block_failures": "" if blind else fails, "ler": "" if blind else fails / n})
        print(rows[-1], flush=True)
    else:
        det_t = shots.load_many(a.code, a.p, a.split, shots.parse_range(a.chunks))[0][:a.first or None][:, keep]
        for B in [int(b) for b in a.batches.split(",")]:
            if B > len(det_t):
                rows.append({**base, "what": "time", "batch": B, "error": f"batch {B} > {len(det_t)} shots"})
                continue
            dec = make(B)
            decode(dec, det_t[:B])
            reps = a.reps1 if B == 1 else max(5, 4096 // B)
            lat = []
            watch = B == 1 and a.timing_out
            if watch:
                from bb.timing import append_row, gpu_watch
                stop = gpu_watch()
            for r in range(reps):
                i = (r * B) % max(1, len(det_t) - B)
                t = time.perf_counter()
                decode(dec, det_t[i:i + B])
                lat.append(time.perf_counter() - t)
            acc = stop() if watch else {}
            lat = np.array(lat)
            rows.append({**base, "what": "time (host to host, numpy in)", "split": a.split, "chunks": a.chunks,
                         "batch": B, "shots": B * reps, "mean_call_ms": 1e3 * lat.mean(),
                         "p50_call_ms": 1e3 * np.median(lat), "p99_call_ms": 1e3 * np.quantile(lat, 0.99),
                         "us_per_shot": 1e6 * lat.mean() / B, "shots_per_s": B / lat.mean()})
            if watch:
                append_row(a.timing_out, {"device": gpu, "role": "opponent", "code": a.code, "p": a.p,
                                          "name": "gpu_bposd", "proc": a.proc, "n": reps, "value_ms": 1e3 * lat.mean(),
                                          "binding": a.config,
                                          "note": json.dumps({"call": "host to host, numpy in", "split": a.split,
                                                              "chunks": a.chunks, **acc})})
            print(rows[-1], flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
