"""X2 surface: join the per-chunk files of surface_chunk.py, check the registered counts, and write the tables and the
figure data: $X2_OUT/surface/summary.json, $X2_OUT/surface/fig_d{d}.npz.

  python surface_join.py --d 13 17

Cells (the network alone, sign(c), against correlated PM, on every test shot): both right, only the network fails
(NF), only correlated PM fails (MF), both fail (BF). |c| percentile = 100 (rank - 0.5) / n of |c| among the test shots
(average ranks), so routed shots (|c| < tau) are the lowest percentiles. Signed gap = W(wrong class) - W(true class)
of plain matching on the split graph, in nats: plain PM fails exactly where it is negative (up to ties)."""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

import x2lib as X
from surface_chunk import STRUCT_NAMES

CELLS = ("both right", "only the network fails", "only correlated PM fails", "both fail")
FRACS = (0.01, 0.02, 0.05, 0.08, 0.10)


def load(d):
    files = sorted(glob.glob(os.path.join(X.OUT, "surface", f"d{d}", "c[0-9][0-9][0-9].npz")))
    assert len(files) == 100, f"d = {d}: {len(files)} chunk files"
    keys = ("obs", "c", "routed", "net", "cpm", "pm", "pair", "n_det", "n_x", "pairs", "bd_hist",
            "cw_true", "cw_wrong", "pw_true", "pw_wrong", "wc", "wp")
    acc = {k: [] for k in keys}
    sel, feats, checks = [], [], {}
    for f in files:
        z = np.load(f)
        j = int(z["j"])
        assert os.path.basename(f) == f"c{j:03d}.npz"
        for k in keys:
            acc[k].append(z[k])
        sel.append(j * X.CHUNK + z["sel"].astype(np.int64))
        feats.append(z["feats"])
        assert tuple(z["feat_names"]) == STRUCT_NAMES
        for name, v in zip(z["check_names"], z["check_vals"]):
            checks.setdefault(str(name), []).append(float(v))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["sel"], out["feats"] = np.concatenate(sel), np.concatenate(feats)
    return out, checks


def q(a, ps=(25, 50, 75)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, ps)] if len(a) else [None] * len(ps)


def describe(z, idx, sel_pos, F):
    """Descriptors of the shots idx (all shots of a cell); the structure ones on the selected shots among them."""
    n_x = z["n_x"][idx]
    pairs = z["pairs"][idx]
    per100 = lambda k: 100.0 * pairs[:, k] / np.maximum(n_x, 1)
    bd = z["bd_hist"][idx].astype(float)
    nb = bd.shape[1]
    bdm = (bd * np.arange(1, nb + 1)).sum(1) / np.maximum(bd.sum(1), 1)
    gap = z["pw_wrong"][idx] - z["pw_true"][idx]
    cgap = z["cw_wrong"][idx] - z["cw_true"][idx]
    pos = sel_pos[idx]
    have = pos >= 0
    f = F[pos[have]]
    col = {n: f[:, i] for i, n in enumerate(STRUCT_NAMES)}
    nonb = col["D_time"] + col["D_space"] + col["D_spacetime"]
    st_nonb = col["st_time"] + col["st_space"] + col["st_spacetime"]
    return dict(
        shots=int(len(idx)), routed=int(z["routed"][idx].sum()),
        n_det=q(z["n_det"][idx]), n_x=q(n_x),
        time_pairs_per100x=q(per100(0)), space_pairs_per100x=q(per100(1)), spacetime_pairs_per100x=q(per100(2)),
        x_event_bdist_mean=q(bdm), pct_abs_c=q(z["pct"][idx], (10, 25, 50, 75, 90)),
        plain_gap=q(gap, (10, 25, 50, 75, 90)), plain_gap_negative=int((gap < 0).sum()),
        plain_gap_abs_below_2=int((np.abs(gap) < 2).sum()), corr_gap=q(cgap, (10, 25, 50, 75, 90)),
        corr_gap_positive=int((cgap > 0).sum()), corr_gap_abs_below_2=int((np.abs(cgap) < 2).sum()),
        plain_pm_fails=int((z["pm"][idx] != z["obs"][idx]).sum()),
        structure_shots=int(have.sum()),
        D_time=q(col["D_time"]), D_space=q(col["D_space"]), D_spacetime=q(col["D_spacetime"]),
        D_time_share=q(col["D_time"] / np.maximum(nonb, 1)), D_text=q(col["D_text"]), D_maxrun=q(col["D_maxrun"]),
        D_has_time_edge=int((col["D_time"] > 0).sum()), D_maxrun_ge3=int((col["D_maxrun"] >= 3).sum()),
        D_maxrun_ge2=int((col["D_maxrun"] >= 2).sum()), D_maxrun_ge4=int((col["D_maxrun"] >= 4).sum()),
        dec_n=q(col["dec_n"]), dec_bdist_mean=q(col["dec_bdist_mean"]), dec_bdist_min=q(col["dec_bdist_min"]),
        dec_bdist_max=q(col["dec_bdist_max"]), dec_text=q(col["dec_text"]),
        dec_bdist_min_eq1=int((col["dec_bdist_min"] == 1).sum()),
        st_time_share=q(col["st_time"] / np.maximum(st_nonb, 1)), st_boundary=q(col["st_boundary"]),
        st_maxrun=q(col["st_maxrun"]))


def capture(z, fracs, k_reg):
    """For the least confident fraction f of the test shots: the network's failures among them, correlated PM's
    failures there, and the pair's failures if exactly those shots were routed. The row with registered = True routes
    the k_reg least confident shots, which are the shots |c| < tau routes (no ties at tau)."""
    n = len(z["obs"])
    order = np.argsort(np.abs(z["c"].astype(np.float64)), kind="stable")
    fn, fc = z["fail_net"][order], z["fail_cpm"][order]
    cfn, cfc = np.concatenate([[0], np.cumsum(fn)]), np.concatenate([[0], np.cumsum(fc)])
    tot_n, tot_c = int(fn.sum()), int(fc.sum())
    rows = []
    for f, reg in sorted([(f, False) for f in fracs] + [(k_reg / n, True)]):
        k = k_reg if reg else int(round(f * n))
        rows.append(dict(frac=f, registered=reg, routed=k, net_fail_routed=int(cfn[k]), net_fail_share=cfn[k] / tot_n,
                         net_rate_routed=cfn[k] / max(k, 1), cpm_fail_routed=int(cfc[k]),
                         cpm_rate_routed=cfc[k] / max(k, 1), cpm_fail_share=cfc[k] / tot_c,
                         pair_fail=int(tot_n - cfn[k] + cfc[k]), pair_minus_cpm=int(tot_n - cfn[k] + cfc[k] - tot_c)))
    grid = np.unique(np.round(np.logspace(np.log10(n * 1e-4), np.log10(n * 0.4), 200)).astype(np.int64))
    curve = dict(k=grid, net_cap=cfn[grid] / tot_n, cpm_cap=cfc[grid] / tot_c, cpm_rate=cfc[grid] / grid,
                 pair=tot_n - cfn[grid] + cfc[grid])
    return rows, curve


def one(d):
    z, checks = load(d)
    t = X.TESTS[d]
    n = len(z["obs"])
    obs = z["obs"]
    z["fail_net"], z["fail_cpm"] = z["net"] != obs, z["cpm"] != obs
    fail_pair, fail_pm = z["pair"] != obs, z["pm"] != obs
    reg = t["registered"]
    got = dict(failures=int(fail_pair.sum()), routed=int(z["routed"].sum()), cpm=int(z["fail_cpm"].sum()))
    assert got == reg, f"d = {d}: {got} differs from the registered {reg}"
    a = np.abs(z["c"].astype(np.float64))
    z["pct"] = 100.0 * (rankdata(a, method="average") - 0.5) / n
    cell = z["fail_net"].astype(np.int8) + 2 * z["fail_cpm"].astype(np.int8)
    sel_pos = np.full(n, -1, np.int64)
    sel_pos[z["sel"]] = np.arange(len(z["sel"]))
    ref = np.zeros(n, bool)
    ref[np.arange(0, n, X.CHUNK)[:, None] + np.arange(0, X.CHUNK, X.REF_EVERY)[None, :]] = True
    routed, kept = z["routed"], ~z["routed"]
    groups = {"all shots (structure: every 32nd shot)": np.nonzero(np.ones(n, bool))[0]}
    for k, name in enumerate(CELLS):
        groups[name] = np.nonzero(cell == k)[0]
        if k in (1, 2):
            groups[name + ", kept"] = np.nonzero((cell == k) & kept)[0]
            groups[name + ", routed"] = np.nonzero((cell == k) & routed)[0]
    desc = {}
    for name, idx in groups.items():
        if name.startswith("all shots"):
            dd = describe(z, idx, np.where(ref, sel_pos, -1), z["feats"])
        else:
            dd = describe(z, idx, sel_pos, z["feats"])
        desc[name] = dd
    rows, curve = capture(z, FRACS, int(routed.sum()))
    assert [r for r in rows if r["registered"]][0]["pair_fail"] == got["failures"], "registered row"
    nf_k, mf_k = int(((cell == 1) & kept).sum()), int(((cell == 2) & kept).sum())
    check_tot = {k: (float(np.sum(v)) if k != "plain_min_dev" else float(np.max(v))) for k, v in checks.items()}
    summary = dict(
        d=d, test_set=t["set"], shots=n, tau=t["tau"], q=t["q"], network=t["network"],
        registered=reg, reproduced=got, checks=check_tot,
        network_alone_failures=int(z["fail_net"].sum()), correlated_pm_failures=int(z["fail_cpm"].sum()),
        plain_pm_failures=int(fail_pm.sum()), pair_failures=int(fail_pair.sum()),
        cells={name: int((cell == k).sum()) for k, name in enumerate(CELLS)},
        cells_routed={name: int(((cell == k) & routed).sum()) for k, name in enumerate(CELLS)},
        kept_nf=nf_k, kept_mf=mf_k, retained_excess=nf_k - mf_k,
        pair_decomposition=dict(nf_kept=nf_k, mf_routed=int(((cell == 2) & routed).sum()),
                                bf=int((cell == 3).sum())),
        routed_pct_threshold=float(100 * routed.mean()), describe=desc, capture=rows)
    figdata = dict(cell=cell[z["sel"]], pct_sel=z["pct"][z["sel"]], routed_sel=routed[z["sel"]],
                   gap_sel=(z["pw_wrong"] - z["pw_true"])[z["sel"]], cgap_sel=(z["cw_wrong"] - z["cw_true"])[z["sel"]],
                   ref_sel=ref[z["sel"]], feats=z["feats"], feat_names=np.array(STRUCT_NAMES),
                   n_det_sel=z["n_det"][z["sel"]], n_x_sel=z["n_x"][z["sel"]],
                   pct_nf=z["pct"][cell == 1], pct_mf=z["pct"][cell == 2],
                   pct_bf=z["pct"][cell == 3], tau_pct=100 * routed.mean(), **{f"curve_{k}": v for k, v in curve.items()},
                   n_det_hist=np.bincount(z["n_det"]), totals=np.array([n, int(z["fail_net"].sum()),
                                                                       int(z["fail_cpm"].sum()), int(fail_pair.sum())]))
    np.savez_compressed(os.path.join(X.OUT, "surface", f"fig_d{d}.npz"), **figdata)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[13, 17])
    a = ap.parse_args()
    res = {str(d): one(d) for d in a.d}
    path = os.path.join(X.OUT, "surface", "summary.json")
    json.dump(res, open(path, "w"), indent=1)
    for d, s in res.items():
        print(f"d = {d}: cells {s['cells']}; routed {s['cells_routed']}; excess {s['retained_excess']}; checks "
              f"{s['checks']}")


if __name__ == "__main__":
    main()
