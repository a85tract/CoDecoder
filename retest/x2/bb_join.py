"""X2 BB72: tables from bb_d1.py's per-shot files -> $X2_OUT/bb/summary.json and $X2_OUT/bb/fig.npz.

S = S55'-deep (the six-block ConvLines-BB front end), C = C14' (our Cascade re-implementation), eager path. A shot is
routed to C when S's margin is below tau1 and kept otherwise. Cells over every shot: S only fails, C only fails, both
fail; among the kept shots these are D1's breaks, repairs and both-fail shots. Percentiles are 100 (rank - 0.5) / n
over all shots of the read (average ranks); low = unsure."""
import json
import os

import numpy as np
from scipy.stats import rankdata

OUT = os.path.join(os.environ.get("X2_OUT", "out"), "bb")
FRACS = (0.01, 0.02, 0.05, 0.10, 0.20)


def q(a, ps=(25, 50, 75)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, ps)] if len(a) else [None] * len(ps)


def wrong_counts(w):
    """[shots with 0, 1, 2, >= 3 logicals wrong]."""
    b = np.bincount(w.astype(np.int64), minlength=4)
    return [int(b[0]), int(b[1]), int(b[2]), int(b[3:].sum())]


def one(p):
    z = np.load(f"{OUT}/p{p:.5f}.npz")
    n = int(z["n"])
    routed, fS, fC = z["routed"], z["fS"], z["fC"]
    kept = ~routed
    pS = 100.0 * (rankdata(z["confS"].astype(np.float64), method="average") - 0.5) / n
    pC = 100.0 * (rankdata(z["confC"].astype(np.float64), method="average") - 0.5) / n
    sel = z["sel"]
    pos = np.full(n, -1, np.int64)
    pos[sel] = np.arange(len(sel))
    ref = np.zeros(n, bool)
    ref[::1000] = True
    nr = z["per_round"].shape[1]
    groups = {
        "all shots (per-layer features: every 1000th shot)": np.ones(n, bool),
        "S only fails, kept (breaks)": fS & ~fC & kept,
        "C only fails, kept (repairs)": ~fS & fC & kept,
        "both fail, kept": fS & fC & kept,
        "S only fails, routed": fS & ~fC & routed,
        "C only fails, routed": ~fS & fC & routed,
        "both fail, routed": fS & fC & routed,
    }
    desc = {}
    for name, m in groups.items():
        idx = np.nonzero(m)[0]
        ps = pos[idx[ref[idx]]] if name.startswith("all shots") else pos[idx]
        ps = ps[ps >= 0]
        pr = z["per_round"][ps].astype(float)
        ev = np.maximum(pr.sum(1), 1)
        zc = np.abs(z["zC"][ps].astype(np.float64))
        errc = z["errC"][ps].astype(bool)
        wrong_abs = np.where(errc, zc, np.nan)
        right_abs = np.where(~errc, zc, np.nan)
        desc[name] = dict(
            shots=int(len(idx)), n_det=q(z["n_det"][idx]), S_margin_pct=q(pS[idx], (10, 25, 50, 75, 90)),
            C_conf_pct=q(pC[idx], (10, 25, 50, 75, 90)),
            C_logicals_wrong=wrong_counts(z["wrongC"][idx]), S_logicals_wrong=wrong_counts(z["wrongS"][idx]),
            true_logicals_flipped=q(z["n_flip"][idx]),
            layer_shots=int(len(ps)), time_pairs_per100=q(100 * z["tpairs"][ps] / ev),
            first_layer_share=q(100 * pr[:, 0] / ev), last_layer_share=q(100 * pr[:, nr - 1] / ev),
            mean_layer=q((pr * np.arange(nr)).sum(1) / ev),
            C_abs_logit_wrong_min=q(np.nanmin(np.where(errc.any(1)[:, None], wrong_abs, np.inf), 1)
                                    if errc.any() else []),
            C_abs_logit_right_min=q(np.nanmin(np.where(np.isfinite(right_abs), right_abs, np.inf), 1)))
    # capture by the least sure fraction of S's margin
    order = np.argsort(z["confS"].astype(np.float64), kind="stable")
    cfs = np.concatenate([[0], np.cumsum(fS[order])])
    cfc = np.concatenate([[0], np.cumsum(fC[order])])
    totS, totC = int(fS.sum()), int(fC.sum())
    cap = []
    k_reg = int(routed.sum())
    for f, reg in sorted([(f, False) for f in FRACS] + [(k_reg / n, True)]):
        k = k_reg if reg else int(round(f * n))
        cap.append(dict(frac=f, registered=reg, routed=k, S_fail_routed=int(cfs[k]), S_fail_share=cfs[k] / totS,
                        C_fail_routed=int(cfc[k]), C_rate_routed=cfc[k] / max(k, 1),
                        pair_fail=int(totS - cfs[k] + cfc[k]), pair_minus_C=int(totS - cfs[k] + cfc[k] - totC)))
    grid = np.unique(np.round(np.logspace(np.log10(n * 1e-4), np.log10(n * 0.4), 200)).astype(np.int64))
    summary = dict(p=p, shots=n, routed=int(routed.sum()), S=totS, C=totC,
                   D=int((fS & kept).sum() + (fC & routed).sum()),
                   cells={k: int(v.sum()) for k, v in groups.items() if not k.startswith("all")},
                   compiled=json.loads(str(z["comp"])), counts=json.loads(str(z["counts"])), describe=desc,
                   capture=cap,
                   C_kept_share_of_C=float((fC & kept).sum() / totC), S_routed_share_of_S=float((fS & routed).sum() / totS))
    fig = dict(pS_Sonly=pS[fS & ~fC], pS_Conly=pS[~fS & fC], pS_both=pS[fS & fC], pS_breaks=pS[fS & ~fC & kept], pS_repairs=pS[~fS & fC & kept], pC_repairs=pC[~fS & fC & kept],
               pC_breaks=pC[fS & ~fC & kept], pS_S=pS[fS], pS_C=pS[fC], pC_C=pC[fC],
               tau_pct=100 * routed.mean(), k=grid, S_cap=cfs[grid] / totS, C_cap=cfc[grid] / totC,
               C_rate=cfc[grid] / grid, pair=totS - cfs[grid] + cfc[grid], totals=np.array([n, totS, totC]))
    return summary, fig


def main():
    res, figs = {}, {}
    for p in (0.002, 0.003):
        s, f = one(p)
        res[str(p)] = s
        figs.update({f"p{int(p * 1000)}_{k}": v for k, v in f.items()})
        print(f"p = {p}: cells {s['cells']}; C kept share {s['C_kept_share_of_C']:.3f}; S routed share "
              f"{s['S_routed_share_of_S']:.3f}")
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/fig.npz", **figs)


if __name__ == "__main__":
    main()
