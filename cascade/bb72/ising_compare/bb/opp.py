"""CPU opponents and fallbacks on stored BB shots (bb/DESIGN.md section 5): per-shot predictions for the paired tests.
  relay       IBM's Relay-BP (relay-bp 0.2.2, RelayDecoderF32) on the X-detector sector with the authors' tuned
              gross-code parameters (qldpc_neural qnd.teachers 'relay': pre_iter 80, 300 sets x 60 iterations, stop
              after 5 solutions) -- the closed route's "sector Relay-BP"
  relay_full  the same decoder and parameters on the full DEM (both detector sectors)
  bposd       ldpc 2.4.1 BP-OSD, min-sum (scale 1.0), 80 iterations, OSD-CS order 60, X sector (the closed route's
              accuracy reference 'bposd_ms80_cs60')
Every decoder runs single-threaded per process; P processes share the shots. Per chunk an npz in BB_PREDS:
pred (n, 12) uint8, fail (n,) bool, secs (n,) float32 (wall time of each decode call; a loaded node, so not a
timing-protocol number), conv (n,) bool. One CSV row per (decoder, chunk set). Dev and test reads are blind: they
store the predictions but print and record no failure count (bb/PLAN.md 2.0: no dev failure is seen before the
policies are committed; bb.route apply reads them after).
  python -m bb.opp --code bb72 --p 0.003 --split dev --chunks 0-1 --dec relay,bposd --procs 32 --out rows.csv"""
import argparse
import csv
import datetime
import os
import platform
import time
from multiprocessing import get_context

import numpy as np

from bb import shots

PREDS = os.path.expanduser(os.environ.get("BB_PREDS", "~/work/preds"))
DECODERS = {"relay": ("relay", "basis"), "relay_full": ("relay", "all"), "bposd": ("bposd_ms80_cs60", "basis")}
_dec = None


def _init(dec, code, p):
    global _dec
    from qnd import teachers
    name, sector = DECODERS[dec]
    _dec = teachers.Decoder(name, code, "X", p, sector)


def _run(det):
    return _dec.decode(det)


def pred_path(code, p, split, j, dec):
    return os.path.join(PREDS, f"{code}_p{p}_{split}{j:04d}_{dec}.npz")


def load_preds(code, p, split, chunks, dec):
    """(pred, secs) of a stored read; cudaq-qec's chunks must come from one whole read (bb.gpu_bposd read stores its id
    in every chunk), so a partly replaced read is refused until the whole read is made again (PLAN 2.8)."""
    z = [np.load(pred_path(code, p, split, j, dec)) for j in chunks]
    reads = {str(x["read"]) if "read" in x.files else "" for x in z}
    if dec.startswith("gpu_bposd") and (len(reads) != 1 or "" in reads):
        raise SystemExit(f"{dec} {code} p {p} {split} chunks {chunks}: not one whole read {sorted(reads)}; make the "
                         f"whole read again")
    return np.concatenate([x["pred"] for x in z]), np.concatenate([x["secs"] for x in z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--dec", required=True, help="comma list of " + ",".join(DECODERS))
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--first", type=int, default=0, help="only the first N shots of each chunk (0 = all)")
    ap.add_argument("--slice", type=int, default=50, help="shots per task")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blind = a.split in ("dev", "test", "test4")
    os.makedirs(PREDS, exist_ok=True)
    from bb.timing import _cpu
    cpu = _cpu()
    for dec in a.dec.split(","):
        _init(dec, a.code, a.p)                           # fail fast here: a failing pool initializer respawns forever
        t0 = time.time()
        fails = n = 0
        secs_all = []
        with get_context("fork").Pool(a.procs, initializer=_init, initargs=(dec, a.code, a.p)) as pool:
            for j in shots.parse_range(a.chunks):
                det, obs = shots.load(a.code, a.p, a.split, j)
                if a.first:
                    det, obs = det[:a.first], obs[:a.first]
                parts = pool.map(_run, [det[i:i + a.slice] for i in range(0, len(det), a.slice)])
                pred = np.concatenate([x[0] for x in parts]).astype(np.uint8)
                secs = np.concatenate([x[1] for x in parts]).astype(np.float32)
                conv = np.concatenate([x[2] for x in parts])
                fail = (pred != obs).any(1)
                suffix = "" if not a.first else f"_first{a.first}"
                np.savez(pred_path(a.code, a.p, a.split, j, dec + suffix), pred=pred, fail=fail, secs=secs, conv=conv)
                fails += int(fail.sum())
                n += len(fail)
                secs_all.append(secs)
                seen = "blind" if blind else f"{int(fail.sum())} / {len(fail)} failures"
                print(f"{dec} chunk {j}: {seen}, mean {1e3 * secs.mean():.2f} ms per shot, converged "
                      f"{conv.mean():.4f}", flush=True)
        s = np.concatenate(secs_all)
        row = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(), "cpu": cpu,
               "procs": a.procs, "code": a.code, "p": a.p, "split": a.split, "chunks": a.chunks, "first": a.first,
               "decoder": dec, "shots": n, "failures": "" if blind else fails, "ler": "" if blind else fails / n,
               "mean_ms": 1e3 * s.mean(),
               "p50_ms": 1e3 * np.median(s), "p99_ms": 1e3 * np.quantile(s, 0.99), "max_ms": 1e3 * s.max(),
               "wall_s": time.time() - t0}
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print("row", row, flush=True)


if __name__ == "__main__":
    main()
