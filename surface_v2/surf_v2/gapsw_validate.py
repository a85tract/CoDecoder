"""The complementary gap of surf_v2/gapsw.py against an exhaustive search, on circuits of the verdict family: NVIDIA's
MemoryCircuit (Ising-Decoding 3ce4379, the arguments of scripts/make_circuits.py: p = 0.006, d rounds, X basis, code
rotation XV, all detectors and the boundary detectors) at d = 3 and 5, and the verdict circuit itself at d = 7, whose Stim
text the same generator must reproduce (its sha256 is convlines.circuits.CIRCUIT_SHA256[7]).

The search shares nothing with PyMatching's solver or with the split graph. It takes plain PM's matching graph as
Matching.edges() lists it (the same edges, float weights and observable bits; the boundary is one node B) and, per shot:
Dijkstra on the graph's two-sheeted cover (a node and the parity of the observable so far) gives the shortest path of
each parity between any two defects and from each defect to B; a dynamic programme over the subsets of each connected
component's defects then tries every pairing (each defect with another defect or with B, each path of either parity),
and the components combine by parity; last, a loop from B to B of odd parity may be added (the only way to change the
class when no path ends at B). That is the least weight of a correction (an edge set with the shot's syndrome) of each
observable value, because the least correction of a class splits into such paths and cycles, and a cycle between
detectors never flips the observable. The front end's numbers must agree with it: plain PM's weight with the lesser of
the two, its answer with the class of the lesser one (unless the two are within the tolerance), the gap with their
difference, and the split graph's weight of each class with the search's. A component with more than --cap defects is
skipped and counted.

    ISING_DECODING_DIR=... python -m surf_v2.gapsw_validate --d 3 5 7 --shots 5000 5000 5000 --out OUT"""
import argparse
import hashlib
import json
import os
import sys
import time
from functools import lru_cache

import numpy as np

from convlines import circuits, provenance
from surf_v2 import gapsw

INF = float("inf")


def nvidia_circuit(d, p=circuits.P):
    sys.path.insert(0, os.path.join(os.environ["ISING_DECODING_DIR"], "code"))
    from qec.surface_code.memory_circuit import MemoryCircuit
    mc = MemoryCircuit(distance=d, idle_error=p, sqgate_error=p, tqgate_error=p, spam_error=(2.0 / 3.0) * p, n_rounds=d,
                       basis="X", code_rotation="XV", get_all_detectors=True, noise_model=None, add_boundary_detectors=True)
    mc.set_error_rates()
    return mc.stim_circuit


def pairings(k, pair, bnd):
    """(least weight with parity 0, with parity 1) over every way to match k defects among themselves or to B;
    pair[i][j] and bnd[i] are (shortest even, shortest odd) path lengths."""
    @lru_cache(maxsize=None)
    def f(mask):
        if mask == 0:
            return 0.0, INF
        i = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << i)
        best = [INF, INF]
        opts = [(bnd[i], rest)]
        j_left = rest
        while j_left:
            j = (j_left & -j_left).bit_length() - 1
            j_left ^= 1 << j
            opts.append((pair[i][j], rest ^ (1 << j)))
        for c, sub in opts:
            r = f(sub)
            for p in (0, 1):
                for q in (0, 1):
                    v = c[p] + r[q]
                    if v < best[p ^ q]:
                        best[p ^ q] = v
        return best[0], best[1]
    return f((1 << k) - 1)


class Exhaustive:
    def __init__(self, m):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        n = self.n = m.num_detectors
        rows, cols, vals, ir, ic = [], [], [], [], []
        for u, v, a in m.edges():
            f = int(0 in a["fault_ids"])
            t = n if v is None else v
            for s in (0, 1):
                rows += [2 * u + s, 2 * t + (s ^ f)]
                cols += [2 * t + (s ^ f), 2 * u + s]
                vals += [a["weight"], a["weight"]]
            if v is not None:
                ir += [u, v]
                ic += [v, u]
        self.G = csr_matrix((vals, (rows, cols)), shape=(2 * n + 2, 2 * n + 2))
        assert self.G.nnz == len(vals), "parallel edges in the cover"
        _, self.comp = connected_components(csr_matrix((np.ones(len(ir)), (ir, ic)), shape=(n, n)), directed=False)

    def classes(self, x, cap):
        """(least weight with observable 0, with observable 1) of syndrome x, or None (a component over cap)."""
        from scipy.sparse.csgraph import dijkstra
        T = [int(t) for t in np.flatnonzero(x)]
        groups = {}
        for t in T:
            groups.setdefault(int(self.comp[t]), []).append(t)
        if any(len(g) > cap for g in groups.values()):
            return None
        n = self.n
        D = dijkstra(self.G, directed=False, indices=[2 * t for t in T] + [2 * n])
        row = {t: i for i, t in enumerate(T)}
        tot = (0.0, INF)
        for g in groups.values():
            pair = [[(D[row[a], 2 * b], D[row[a], 2 * b + 1]) for b in g] for a in g]
            bnd = [(D[row[a], 2 * n], D[row[a], 2 * n + 1]) for a in g]
            c = pairings(len(g), pair, bnd)
            tot = (min(tot[0] + c[0], tot[1] + c[1]), min(tot[0] + c[1], tot[1] + c[0]))
        loop = D[len(T), 2 * n + 1]
        return min(tot[0], tot[1] + loop), min(tot[1], tot[0] + loop)


def run(d, shots, seed, cap, tol):
    c = nvidia_circuit(d)
    text = str(c)
    sha = hashlib.sha256(text.encode()).hexdigest()
    if d in circuits.CIRCUIT_SHA256:
        assert sha == circuits.CIRCUIT_SHA256[d], f"d = {d}: NVIDIA's generator does not give the verdict circuit"
        c = circuits.circuit(d)
    det, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    det, obs = det.astype(np.uint8), obs[:, 0].astype(np.uint8)
    fe = gapsw.FrontEnd(circuit=c)
    dec, w, gap = fe.batch(det)
    w0, w1 = fe.least(det, np.zeros(shots, np.uint8)), fe.least(det, np.ones(shots, np.uint8))
    ex = Exhaustive(fe.m)
    t0 = time.time()
    b = np.full((shots, 2), np.nan)
    for i in range(shots):
        r = ex.classes(det[i], cap)
        if r is not None:
            b[i] = r
    ok = ~np.isnan(b[:, 0])
    i = np.arange(shots)
    best, other = b[i, dec], b[i, 1 - dec]
    sep = np.abs(b[:, 0] - b[:, 1]) > tol
    res = dict(d=d, circuit_sha256=sha, verdict_circuit=d in circuits.CIRCUIT_SHA256, seed=seed, shots=shots, cap=cap,
               tol=tol, checked=int(ok.sum()), skipped=int((~ok).sum()), search_s=time.time() - t0,
               detectors=fe.n, split_to_b=fe.to_b, split_colour_ones=int(len(fe.S)),
               plain_failures=int((dec != obs).sum()), gap_zero=int((gap == 0).sum()),
               max_dev_weight=float(np.max(np.abs(w - np.minimum(b[:, 0], b[:, 1]))[ok])),
               max_dev_gap=float(np.max(np.abs(gap - (other - best))[ok])),
               max_dev_class0=float(np.max(np.abs(w0 - b[:, 0])[ok])),
               max_dev_class1=float(np.max(np.abs(w1 - b[:, 1])[ok])),
               separated=int((ok & sep).sum()), ties=int((ok & ~sep).sum()),
               class_disagree=int((ok & sep & (np.argmin(np.nan_to_num(b, nan=INF), 1) != dec)).sum()),
               split_least_equals_plain=bool(np.array_equal(np.where(dec == 1, w1, w0), w)), gap_nonneg=bool((gap >= 0).all()))
    res["passed"] = bool(res["max_dev_weight"] <= tol and res["max_dev_gap"] <= tol and res["max_dev_class0"] <= tol
                         and res["max_dev_class1"] <= tol and res["class_disagree"] == 0
                         and res["split_least_equals_plain"] and res["gap_nonneg"] and res["checked"] > 0)
    return res, dict(det=np.packbits(det, axis=1), obs=obs, dec=dec, weight=w, gap=gap, w0=w0, w1=w1, search=b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--shots", type=int, nargs="+", default=[5000, 5000, 5000])
    ap.add_argument("--seed", type=int, default=7_100_000, help="Stim's seed at d is seed + d")
    ap.add_argument("--cap", type=int, default=14)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres = dict(what="the complementary gap against an exhaustive search (surf_v2/gapsw_validate.py)",
                  gapsw_sha256=gapsw.gapsw_sha256(), validate_sha256=provenance.sha_file(os.path.abspath(__file__)),
                  code_digest=provenance.code_digest(), release_commit=provenance.commit(), host=provenance.host(),
                  versions=provenance.versions(), distances={})
    for d, n in zip(a.d, a.shots):
        res, arrs = run(d, n, a.seed + d, a.cap, a.tol)
        gapsw.save(os.path.join(a.out, f"validate_d{d}"), arrs, dict(res, what="per-shot values of the validation"))
        allres["distances"][str(d)] = res
        print(json.dumps(res, sort_keys=True), flush=True)
    allres["passed"] = all(r["passed"] for r in allres["distances"].values())
    with open(os.path.join(a.out, "validate.json"), "x") as fh:
        json.dump(allres, fh, indent=1, sort_keys=True)
    print("VALIDATION " + ("PASSED" if allres["passed"] else "FAILED"), flush=True)
    raise SystemExit(0 if allres["passed"] else 1)


if __name__ == "__main__":
    main()
