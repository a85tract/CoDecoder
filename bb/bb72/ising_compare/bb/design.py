"""Design facts and checks for ConvLines-BB (bb/DESIGN.md): one CPU job per code.
  python -m bb.design --code bb72 --p 0.003 --out OUT.json [--label-shots 2000]
  python -m bb.design --rf-update OUT.json      (recompute only the receptive field of an existing OUT.json)
Computes and checks
  layout   every detector's (t, type, site); per-layer counts; the X stabilisers read off the final layer equal the
           polynomial construction; the 12 observables commute with the Z checks
  dem      the undecomposed DEM is invariant under the translations x and y (every fault's translated detector set is a
           fault with the same prior); the co-firing offsets (dt, da, db) of every pair of detectors inside one fault,
           with their prior mass: the stem's taps; the Tanner-graph offsets (checks sharing a data qubit) must lie
           inside their spatial projection; the blocks' taps = the Tanner offsets x dt in {-1, 0, 1} (as
           net.taps_from_design); the receptive field of the stem and of each further block, from these taps
  logicals the translation action M_g on the 12 logicals, the subgroup acting trivially, each logical's own
           stabiliser subgroup, the distinct functionals among the k G translates
  labels   on fresh smoke shots: the circuit with one extra observable per translate gives the original detectors and
           observables, and every extra observable equals (y Mcol + det Gdet) mod 2"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np

from bb import code as bcode
from bb import shots
from qnd import bb as qbb
from qnd import circuits


def centred(v, n):
    v = np.asarray(v) % n
    return np.where(v > n // 2, v - n, v)


def cofire(lay, H, q):
    """{(dt, da, db, type1, type2): [mass, count]} over ordered pairs of distinct detectors inside one fault."""
    l, m = lay.l, lay.m
    a, b = np.divmod(lay.site, m)
    out = defaultdict(lambda: [0.0, 0])
    H = H.tocsc()
    for f in range(H.shape[1]):
        d = H.indices[H.indptr[f]:H.indptr[f + 1]]
        if len(d) < 2:
            continue
        dt = lay.t[d][None, :] - lay.t[d][:, None]
        da = centred(a[d][None, :] - a[d][:, None], l)
        db = centred(b[d][None, :] - b[d][:, None], m)
        for i in range(len(d)):
            for j in range(len(d)):
                if i != j:
                    e = out[(int(dt[i, j]), int(da[i, j]), int(db[i, j]), int(lay.typ[d[i]]), int(lay.typ[d[j]]))]
                    e[0] += q[f]
                    e[1] += 1
    return out


def block_taps(tanner):
    """One block's (dt, da, db) taps: the Tanner offsets (X-X, X -> Z, Z -> X) x dt in {-1, 0, 1}, as net.taps_from_design."""
    sp = sorted({tuple(x) for x in tanner["xx"] + tanner["xz"] + tanner["zx"]})
    return [(dt, a, b) for a, b in sp for dt in (-1, 0, 1)]


def receptive_field(stem, blk, l, m, nblocks=6):
    """The detectors one output site depends on after the stem and nb blocks, from the taps the network uses (stem: the
    co-firing (dt, da, db), blk: one block's taps); offsets add, mod the torus in space, unwrapped in time. Per nb: the
    time reach (largest |dt|), the share of the torus reached at dt = 0 and at |dt| = the reach (the smaller side), and
    full_window = the largest w such that every site is reached at every |dt| <= w (-1: not even at dt = 0)."""
    R = {(int(t), int(x) % l, int(y) % m) for t, x, y in stem}
    rows = []
    for nb in range(nblocks + 1):
        reach = max(abs(t) for t, _, _ in R)
        share = {t: len({(x, y) for u, x, y in R if u == t}) / (l * m) for t in range(-reach, reach + 1)}
        full = -1
        while full < reach and min(share[full + 1], share[-full - 1]) == 1:
            full += 1
        rows.append({"blocks": nb, "time_reach": reach, "share_dt0": share[0],
                     "share_at_reach": min(share[reach], share[-reach]), "full_window": full})
        R = {(t + u, (x + v) % l, (y + w) % m) for t, x, y in R for u, v, w in blk}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code")
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--out")
    ap.add_argument("--label-shots", type=int, default=2000)
    ap.add_argument("--rf-update", default="", help="recompute the receptive field of this design JSON in place")
    a = ap.parse_args()
    if a.rf_update:
        rep = json.load(open(a.rf_update))
        rep["receptive_field"] = receptive_field([t[:3] for t in rep["taps"]["stem"]], block_taps(rep["taps"]["tanner"]),
                                                 rep["layout"]["l"], rep["layout"]["m"])
        json.dump(rep, open(a.rf_update, "w"), indent=1)
        print("rf", rep["code"], json.dumps(rep["receptive_field"]), flush=True)
        return
    t0 = time.time()
    rep = {"code": a.code, "p": a.p}
    lay = bcode.Layout(a.code, a.p)
    l, m, n2, T = lay.l, lay.m, lay.n2, lay.T
    hx_poly, hz_poly, terms_A, terms_B = qbb.matrices(a.code)
    layers = [{"t": t, "X": int(((lay.t == t) & (lay.typ == 0)).sum()), "Z": int(((lay.t == t) & (lay.typ == 1)).sum())}
              for t in range(T)]
    rep["layout"] = {"l": l, "m": m, "rounds": lay.R, "T": T, "detectors": lay.D, "layers": layers,
                     "hx_equals_polynomials": bool((lay.hx == hx_poly).all()),
                     "logicals_commute_with_Z": bool(((hz_poly.astype(np.int64) @ lay.logicals.T.astype(np.int64)) % 2 == 0).all()),
                     "logical_weights": lay.logicals.sum(1).tolist()}
    print("layout", json.dumps(rep["layout"]), flush=True)

    # ---- DEM: translation invariance and co-firing offsets
    H, L, q, _ = circuits.problem(a.code, "X", a.p, "all")
    H = H.tocsc()
    faults = {}
    for f in range(H.shape[1]):
        key = tuple(sorted(H.indices[H.indptr[f]:H.indptr[f + 1]].tolist()))
        faults[key] = faults.get(key, 0.0) + q[f]
    inv = {}
    for g in [(1, 0), (0, 1)]:
        perm = lay.detector_perm(g)
        missing, worst = 0, 0.0
        for key, pr in faults.items():
            k2 = tuple(sorted(perm[list(key)].tolist()))
            if k2 not in faults:
                missing += 1
            else:
                worst = max(worst, abs(faults[k2] - pr) / pr)
        inv[f"x^{g[0]} y^{g[1]}"] = {"missing": missing, "max_rel_prior_diff": worst}
    rep["dem"] = {"faults": int(H.shape[1]), "distinct_detector_sets": len(faults), "translation_invariance": inv}
    print("dem", json.dumps(rep["dem"]), flush=True)
    cf = cofire(lay, H, q)
    total = sum(v[0] for v in cf.values())
    stem = defaultdict(float)
    for (dt, da, db, _, _), (mass, _) in cf.items():
        stem[(dt, da, db)] += mass
    stem[(0, 0, 0)] += 0.0
    stem_taps = sorted(stem, key=lambda k: (-stem[k], k))
    spatial = sorted({(da, db) for _, da, db in stem_taps})
    dts = sorted({dt for dt, _, _ in stem_taps})
    # Tanner-graph offsets: X-X / Z-Z via one shared data qubit, X -> Z via one shared data qubit
    mono =lambda M: tuple(divmod(int(np.nonzero(M[0])[0][0]), m))   # monomial offset of a shift matrix (row 0)
    A = [mono(t) for t in terms_A]
    B = [mono(t) for t in terms_B]
    sub = lambda u, v: (int(centred(u[0] - v[0], l)), int(centred(u[1] - v[1], m)))
    add = lambda u, v: (int(centred(u[0] + v[0], l)), int(centred(u[1] + v[1], m)))
    tanner_xx = sorted({sub(u, v) for P in (A, B) for u in P for v in P})
    tanner_xz = sorted({add(u, v) for u in A for v in B})
    tanner_zx = sorted({sub((0, 0), add(u, v)) for u in A for v in B})
    sp = set(spatial)
    xx_seen = {(da, db) for (dt, da, db, t1, t2) in cf if t1 == t2 == 0}
    xz_seen = {(da, db) for (dt, da, db, t1, t2) in cf if t1 == 0 and t2 == 1}
    rep["taps"] = {
        "stem": [[int(x) for x in k] + [stem[k] / total] for k in stem_taps],
        "stem_count": len(stem_taps), "stem_dt": dts, "block_spatial": [list(s) for s in spatial],
        "block_spatial_count": len(spatial),
        "tanner": {"A": A, "B": B, "xx": tanner_xx, "xz": tanner_xz, "zx": tanner_zx,
                   "xx_inside_block_taps": set(tanner_xx) <= sp, "xz_inside": set(tanner_xz) <= sp,
                   "zx_inside": set(tanner_zx) <= sp,
                   "xx_cofire_minus_tanner": sorted(xx_seen - set(tanner_xx)),
                   "tanner_xx_minus_cofire": sorted(set(tanner_xx) - xx_seen),
                   "xz_cofire_minus_tanner": sorted(xz_seen - set(tanner_xz)),
                   "tanner_xz_minus_cofire": sorted(set(tanner_xz) - xz_seen)},
        "box": {"dt": [min(dts), max(dts)], "da": [min(s[0] for s in spatial), max(s[0] for s in spatial)],
                "db": [min(s[1] for s in spatial), max(s[1] for s in spatial)]},
    }
    by_type = defaultdict(lambda: [0.0, 0])
    for (dt, da, db, t1, t2), (mass, cnt) in cf.items():
        by_type[f"{'XZ'[t1]}{'XZ'[t2]}"][0] += mass / total
        by_type[f"{'XZ'[t1]}{'XZ'[t2]}"][1] += 1
    rep["taps"]["pair_mass_by_type"] = dict(by_type)
    rf = receptive_field(stem_taps, block_taps(rep["taps"]["tanner"]), l, m)
    rep["receptive_field"] = rf
    print("taps", json.dumps({k: v for k, v in rep["taps"].items() if k != "stem"}), flush=True)
    print("rf", json.dumps(rf), flush=True)

    # ---- logicals under translation
    tr = bcode.translates(a.code, a.p)
    M = tr["M"]
    G, k = M.shape[0], M.shape[1]
    I = np.eye(k, dtype=np.uint8)
    triv = [g for g in range(G) if (M[g] == I).all()]
    stab = [[g for g in range(G) if (M[g, kk] == I[kk]).all()] for kk in range(k)]
    funcs = {tuple(M[g, kk]) for g in range(G) for kk in range(k)}
    rep["logicals"] = {
        "trivial_translations": [list(divmod(g, m)) for g in triv],
        "stabiliser_sizes": [len(s) for s in stab],
        "stabilisers": [[list(divmod(g, m)) for g in s] for s in stab],
        "distinct_functionals": len(funcs),
        "translates": int(k * G),
        "rank_of_all_translate_functionals": int(qbb.gf2_rank(np.array(sorted(funcs), np.uint8))),
        "stabiliser_weight_per_translate_max": int(tr["C"].sum(-1).max()),
    }
    print("logicals", json.dumps({kk: v for kk, v in rep["logicals"].items() if kk != "stabilisers"}), flush=True)

    # ---- labels on fresh smoke shots (chunk 999 of the smoke split, never stored)
    if a.label_shots:
        aug, extra = bcode.translate_circuit(a.code, a.p)
        seed = shots.seed_of("smoke", a.code, a.p, 999)
        det0, obs0 = lay.circuit.compile_detector_sampler(seed=seed).sample(a.label_shots, separate_observables=True)
        det1, obs1 = aug.compile_detector_sampler(seed=seed).sample(a.label_shots, separate_observables=True)
        same_det = bool((det0 == det1).all()) and bool((obs0 == obs1[:, :k]).all())
        y_all = (obs1[:, :k].astype(np.int64) @ tr["Mcol"] + det1.astype(np.int64) @ tr["Gdet"]) % 2
        cols = [kk * G + g for kk, g in extra]
        ok = (y_all[:, cols] == obs1[:, k:]).all(0)
        base_ok = all((y_all[:, kk * G] == obs1[:, kk]).all() for kk in range(k))
        rep["labels"] = {"shots": a.label_shots, "seed": seed, "detectors_and_observables_unchanged": same_det,
                         "translates_checked": len(cols), "translates_equal": int(ok.sum()), "base_columns_equal": base_ok,
                         "mean_translate_flip_rate": float(obs1[:, k:].mean()), "block_flip_rate": float(obs0.any(1).mean())}
        print("labels", json.dumps(rep["labels"]), flush=True)
    rep["seconds"] = time.time() - t0
    json.dump(rep, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("written", a.out, f"{rep['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
