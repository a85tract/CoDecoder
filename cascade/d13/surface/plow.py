"""The choice of the lower test p (cascade_compare/SURFACE_PLAN.md section 2.2): correlated PyMatching's block failure
rate r_M on 1,000,000 fresh shots at each candidate p (seed kind 6, chunks 0-99 of 10,000), plain PyMatching beside it
as context, and the rule p_low = the smallest candidate with RATIO x r_M x CAP >= NEED (0.6 x r_M x 4e7 >= 1,200).
No network output is involved. Decoding times are this CPU's, for the compute estimate only.
    python -m surface.plow --d 13 --procs 16 --out OUT.json"""
import argparse
import json
import multiprocessing as mp
import time

from surface import surface_data as sdata
from surface import surface_layout as sl

CANDIDATES = (0.002, 0.0025, 0.003, 0.0035, 0.004)
RATIO, CAP, NEED = 0.6, 4e7, 1200
_M = {}


def _chunk(args):
    d, p, j, n = args
    sl.release_root()
    from convlines import circuits
    if (d, p) not in _M:
        c = sl.circuit_at(d, p)
        _M[d, p] = (circuits.matcher(d, True, c), circuits.matcher(d, False, c))
    det, obs = sdata.sample(d, p, n, sdata.eval_seed(d, 6, p, j))
    mc, mpl = _M[d, p]
    t = time.time()
    fc = int((circuits.pm_decode(mc, det, True) != obs[:, 0]).sum())
    tc = time.time() - t
    t = time.time()
    fp = int((circuits.pm_decode(mpl, det, False) != obs[:, 0]).sum())
    tp = time.time() - t
    return p, fc, fp, tc, tp, float(det.mean()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=13)
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from surface.surface_train import wilson
    jobs = [(a.d, p, j, 10000) for p in CANDIDATES for j in range(a.chunks)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(_chunk, jobs, chunksize=max(1, len(jobs) // (4 * a.procs)))
    rows = []
    for p in CANDIDATES:
        rr = [r for r in res if r[0] == p]
        n = sum(r[6] for r in rr)
        fc, fp = sum(r[1] for r in rr), sum(r[2] for r in rr)
        lo, hi = wilson(fc, n)
        rows.append({"p": p, "shots": n, "corr_pm_fail": fc, "r_M": fc / n, "r_M_lo": lo, "r_M_hi": hi,
                     "plain_pm_fail": fp, "det_rate": round(sum(r[5] for r in rr) / len(rr), 6),
                     "corr_pm_us_per_shot": round(1e6 * sum(r[3] for r in rr) / n, 2),
                     "plain_pm_us_per_shot": round(1e6 * sum(r[4] for r in rr) / n, 2),
                     "expected_cascade_fail_at_cap": round(RATIO * fc / n * CAP, 1),
                     "meets_rule": RATIO * fc / n * CAP >= NEED})
        print(json.dumps(rows[-1]), flush=True)
    ok = [r["p"] for r in rows if r["meets_rule"]]
    out = {"d": a.d, "seed_kind": 6, "chunks": a.chunks, "rule": f"smallest p with {RATIO} x r_M x {CAP:g} >= {NEED}",
           "candidates": list(CANDIDATES), "p_low": min(ok) if ok else None, "rows": rows,
           "circuit_sha256": sl.check_circuit_at(a.d), "env": sdata.env_record(a.d),
           "minutes": round((time.time() - t0) / 60, 2)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"p_low": out["p_low"], "minutes": out["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
