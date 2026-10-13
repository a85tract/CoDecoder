"""Stage G1 (arch/README.md): the dense detector grid at distance d and the offsets of the DEM's error mechanisms in it,
i.e. which relations a local convolution kernel has to cover.

Grid: every detector gets (t, i, j) = (its round index from circuits.geometry, its ancilla coordinates divided by their
spacing). Offsets are taken between the detectors of one decomposed component (PyMatching's edges) and, in the
undecomposed DEM, between the X- and Z-type detectors of one mechanism (the X-Z correlations correlated matching uses).
    python -m arch.geom --d 9 [--out FILE.json]"""
import argparse
import collections
import json

import numpy as np

from icmp import circuits


def layout(d):
    """(mc, g, idx): g = circuits.geometry rows (x, y, t, is_x, boundary); idx (m, 3) int = (t, i, j)."""
    mc = circuits.memory_circuit(d, noise=circuits.NOISE_VERDICT)
    g = circuits.geometry(mc)
    idx = np.zeros((len(g), 3), np.int64)
    idx[:, 0] = g[:, 2] - g[:, 2].min()
    for a, col in ((1, 0), (2, 1)):
        v = np.unique(g[:, col])
        s = int(np.min(np.diff(v)))
        assert np.all((v - v.min()) % s == 0), f"coordinate {col} is not on a lattice of spacing {s}"
        idx[:, a] = (g[:, col] - v.min()) // s
    assert len({tuple(r) for r in idx}) == len(idx), "two detectors share a grid site"
    return mc, g, idx


def components(dem):
    """[(p, [detector lists of the components], [observable flags of the components])] of a DEM's error instructions."""
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps, obs, cur, o = [], [], [], 0
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append(cur), obs.append(o)
                cur, o = [], 0
            elif t.is_relative_detector_id():
                cur.append(t.val)
            elif t.is_logical_observable_id():
                o ^= 1
        comps.append(cur), obs.append(o)
        out.append((inst.args_copy()[0], comps, obs))
    return out


def canon(v):
    v = tuple(int(x) for x in v)
    return v if v >= tuple(-x for x in v) else tuple(-x for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=9)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = a.d
    mc, g, idx = layout(d)
    c = mc.stim_circuit
    m = len(g)
    is_x = g[:, 3].astype(bool)
    rep = {"d": d, "detectors": m, "grid_shape": [int(idx[:, k].max()) + 1 for k in range(3)]}
    occ = {}
    for t in range(rep["grid_shape"][0]):
        sel = idx[:, 0] == t
        occ[t] = {"X": int((sel & is_x).sum()), "Z": int((sel & ~is_x).sum()), "boundary": int(g[sel, 4].sum())}
    rep["per_layer"] = occ
    par = (idx[:, 1] + idx[:, 2]) % 2
    rep["checkerboard"] = {"X_parity": sorted({int(p) for p in par[is_x]}), "Z_parity": sorted({int(p) for p in par[~is_x]})}
    sites = {(int(i), int(j)): int(x) for (t, i, j), x in zip(idx, is_x)}
    H, W = rep["grid_shape"][1:]
    rep["site_map"] = ["".join("X" if sites.get((i, j)) == 1 else "Z" if (i, j) in sites else "." for j in range(W))
                       for i in range(H)]
    # decomposed DEM: PyMatching's edges
    dm = circuits.dem(c)
    pair, single, nsplit = collections.Counter(), collections.Counter(), collections.Counter()
    obs_edges = []
    for p, comps, obs in components(dm):
        nsplit[len(comps)] += p
        for cd, o in zip(comps, obs):
            if len(cd) == 2:
                u, v = cd
                key = ("XZ"[int(not is_x[u])] + "XZ"[int(not is_x[v])],) + canon(idx[v] - idx[u])
                pair[key] += p
                if o:
                    obs_edges.append(("pair", [int(x) for x in idx[u]], [int(x) for x in idx[v]]))
            elif len(cd) == 1:
                u = cd[0]
                single["XZ"[int(not is_x[u])]] += p
                if o:
                    obs_edges.append(("boundary", [int(x) for x in idx[u]]))
            else:
                single[f"{len(cd)}-detector component"] += p
    tot = sum(pair.values())
    rows = sorted(pair.items(), key=lambda kv: -kv[1])
    rep["edge_offsets"] = [{"types": k[0], "dt_di_dj": list(k[1:]), "p_sum": v, "share": v / tot} for k, v in rows]
    rep["boundary_edges_p"] = dict(single)
    rep["components_per_mechanism_p"] = {str(k): v for k, v in sorted(nsplit.items())}
    bb = np.array([r["dt_di_dj"] for r in rep["edge_offsets"]])
    rep["edge_offset_extent"] = {"max_abs_dt": int(np.abs(bb[:, 0]).max()), "max_abs_di": int(np.abs(bb[:, 1]).max()),
                                 "max_abs_dj": int(np.abs(bb[:, 2]).max())}
    for name, (kt, ki, kj) in {"3x3x3": (1, 1, 1), "3x5x5": (1, 2, 2), "3x3x5": (1, 1, 2), "5x5x5": (2, 2, 2)}.items():
        cov = sum(r["p_sum"] for r in rep["edge_offsets"] if abs(r["dt_di_dj"][0]) <= kt and abs(r["dt_di_dj"][1]) <= ki
                  and abs(r["dt_di_dj"][2]) <= kj)
        rep.setdefault("edge_mass_covered", {})[name] = cov / tot
    rep["observable_edges"] = {"count": len(obs_edges), "kinds": dict(collections.Counter(e[0] for e in obs_edges)),
                               "sites": sorted({tuple(e[1][1:]) for e in obs_edges if e[0] == "boundary"})}
    # undecomposed DEM: X-Z offsets inside one mechanism
    du = circuits.dem(c, decompose=False)
    xz = collections.Counter()
    nxz = 0.0
    for p, comps, _ in components(du):
        dets = [u for cd in comps for u in cd]
        xs, zs = [u for u in dets if is_x[u]], [u for u in dets if not is_x[u]]
        if xs and zs:
            nxz += p
            for u in xs:
                for v in zs:
                    xz[tuple(int(x) for x in idx[v] - idx[u])] += p
    tz = sum(xz.values())
    rep["xz_mechanism_p"] = nxz
    rep["xz_offsets"] = [{"z_minus_x_dt_di_dj": list(k), "p_sum": v, "share": v / tz}
                         for k, v in sorted(xz.items(), key=lambda kv: -kv[1])]
    xb = np.array([r["z_minus_x_dt_di_dj"] for r in rep["xz_offsets"]])
    rep["xz_offset_extent"] = {"max_abs_dt": int(np.abs(xb[:, 0]).max()), "max_abs_di": int(np.abs(xb[:, 1]).max()),
                               "max_abs_dj": int(np.abs(xb[:, 2]).max())}
    # logical lines on the grid
    from icmp import lines
    off = lines.line_offsets(d)
    data = mc.code.code_dict["data"]
    co = np.array([data[q]["coord"] for q in sorted(data)], float)
    rep["data_coords_first_line"] = co[:d].tolist()
    rep["line_offset_range"] = [int(off.min()), int(off.max())]
    rep["line_near_rows_cols"] = [{"line": j, "grid_i": sorted({int(x) for x in idx[np.abs(off[:, j]) <= 1, 1]}),
                                   "grid_j": sorted({int(x) for x in idx[np.abs(off[:, j]) <= 1, 2]})} for j in range(d)]
    A, rank = lines.gmap(d)
    _, aug = lines.line_circuit(d)
    det, ob = aug.compile_detector_sampler(seed=31337).sample(10_000, separate_observables=True)
    gg = (det.astype(np.int64) @ A.astype(np.int64)) & 1
    rep["gmap"] = {"rank": int(rank), "errors_on_10000_fresh": int((gg != (ob ^ ob[:, :1])).sum()),
                   "detectors_per_line": [int(A[:, j].sum()) for j in range(d)],
                   "line_flip_rate": [float(x) for x in ob.mean(0)]}
    fired = det.sum(1)
    rep["fired_per_shot"] = {"mean": float(fired.mean()), "median": float(np.median(fired)), "p99": float(np.quantile(fired, 0.99)),
                             "max": int(fired.max())}
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
