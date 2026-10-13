"""Paired comparison of decoders on the same val / test shots from saved per-shot predictions (npz with 'fail' and
'pred'). python -m bench.paired --dir PREDS --code bb72 --p 0.003 --split val --chunks 2-4 --a net-RUN-TAG \
    --b relay_basis,bposd_gu_basis,... [--teacher relay_basis] --out data/stage2_paired.csv
Per pair: failures of a and b, discordant shots, exact two-sided McNemar p, the LER difference a - b with Newcombe's
paired 95 % interval (method 10); with --teacher also the teacher's errors a corrects and the new errors a makes."""
import argparse
import csv
import glob
import math
import os

import numpy as np
from scipy.stats import binomtest

from qnd.train import parse_range


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe10(a, b, c, d, z=1.959964):
    """Paired difference p1 - p2 with p1 = (a+b)/n, p2 = (a+c)/n (a: both fail, b: only 1, c: only 2, d: neither)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if denom == 0 else max(0.0, a * d - b * c) / math.sqrt(denom)
    dt = p1 - p2
    lo = dt - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = dt + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return dt, lo, hi


def load(d, code, p, split, chunks, name, basis="X"):
    fails, preds = [], []
    for j in chunks:
        f = glob.glob(os.path.join(d, f"{code}_{basis}_{p}_{split}{j}_{name}.npz"))
        if len(f) != 1:
            raise FileNotFoundError(f"{code}_{basis}_{p}_{split}{j}_{name}.npz in {d}: {f}")
        z = np.load(f[0])
        fails.append(z["fail"].astype(bool)); preds.append(z["pred"])
    return np.concatenate(fails), np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--basis", default="X")
    ap.add_argument("--split", default="val")
    ap.add_argument("--chunks", default="2-4")
    ap.add_argument("--a", required=True, help="comma list")
    ap.add_argument("--b", required=True, help="comma list")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    ch = parse_range(x.chunks)
    rows = []
    for an in x.a.split(","):
        fa, pa = load(x.dir, x.code, x.p, x.split, ch, an, x.basis)
        extra = {}
        if x.teacher:
            ft, pt = load(x.dir, x.code, x.p, x.split, ch, x.teacher, x.basis)
            extra = {"teacher": x.teacher, "teacher_failures": int(ft.sum()),
                     "teacher_errors_corrected": int((ft & ~fa).sum()), "new_errors": int((~ft & fa).sum()),
                     "agreement_with_teacher": float((pa == pt).all(1).mean())}
        for bn in x.b.split(","):
            fb, _ = load(x.dir, x.code, x.p, x.split, ch, bn, x.basis)
            n = len(fa)
            both, only_a, only_b = int((fa & fb).sum()), int((fa & ~fb).sum()), int((~fa & fb).sum())
            disc = only_a + only_b
            pmc = binomtest(only_a, disc, 0.5).pvalue if disc else 1.0
            dt, lo, hi = newcombe10(both, only_a, only_b, n - both - only_a - only_b)
            rows.append({"code": x.code, "p": x.p, "split": x.split, "chunks": x.chunks, "shots": n, "a": an, "b": bn,
                         "fail_a": int(fa.sum()), "fail_b": int(fb.sum()), "ratio_a_b": fa.sum() / max(1, fb.sum()),
                         "only_a": only_a, "only_b": only_b, "mcnemar_p": pmc, "diff_ler": dt, "diff_lo95": lo,
                         "diff_hi95": hi, **extra})
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    new = not os.path.exists(x.out)
    with open(x.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
