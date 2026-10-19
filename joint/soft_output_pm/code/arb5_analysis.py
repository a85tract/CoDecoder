"""ARB5 analysis (exploratory): the joint decision c + w s on the routed shots where the network and correlated PM differ,
from arb5_alpine.py's records. w is fitted on the even chunks (fit half) and every reported count is from the odd chunks
(evaluation half). s is the signed gap: +gap when the matcher's answer is 1, -gap when it is 0 (plain: plain PM's gap and
answer; corr: the correlated-PM gap of arb5_alpine.py and correlated PM's answer). The answer is 1 when c + w s > 0, 0 when
it is < 0, and correlated PM's on a tie. w runs over {0} and 10^(k/40), k = -120..120, plus infinity (the composition), as
ARB's A3; the fit-half net gain picks w, ties keep the larger w.

    python arb5_analysis.py --out OUT --release RELEASE      (writes OUT/arb5_results.json and OUT/arb5_tables.md)"""
import argparse
import json
import math

import numpy as np
from scipy.stats import binomtest

CHUNK = 10_000
WS = [0.0] + [10 ** (k / 40) for k in range(-120, 121)] + [math.inf]
QREG = {7: 0.02, 9: 0.06, 11: 0.0, 13: 0.0, 15: 0.01, 17: 0.01}
QREAD = {7: [0.02, 0.05], 9: [0.06], 11: [0.02, 0.05], 13: [0.02, 0.05], 15: [0.01, 0.02, 0.05], 17: [0.01, 0.02, 0.05]}
TAU_D9 = 2.337686538696289
NETWORK = {7: "d7_b1w64_sv2", 9: "d9_b1w64_sh", 11: "d11_b5_sv2", 13: "d13_b5_sv2", 15: "d15_b5_sv2", 17: "d17_b5_sv2"}


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def newcombe_paired(a, b, c, d):
    """Newcombe's method 10: 95 % interval of p1 - p2 for paired proportions; a = both fail, b = only the first fails,
    c = only the second fails, d = neither."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return p1 - p2 - dl, p1 - p2 + du


def joint(c, s, cpm, w):
    if math.isinf(w):
        return cpm.copy()
    v = c.astype(np.float64) + w * s
    return np.where(v > 0, 1, np.where(v < 0, 0, cpm)).astype(np.uint8)


def gain(y, cpm, obs, m):
    rep = int(((cpm != obs) & (y == obs) & m).sum())
    brk = int(((cpm == obs) & (y != obs) & m).sum())
    return rep, brk


def load(d, out, release):
    """-> per-shot arrays of the routed shots at the largest q read (idx, c, obs, cpm, pm, gap_pm, gap_cpm, times) and
    the per-chunk counts of every shot (net_fail, cpm_fail), with the record."""
    z = dict(np.load(f"{out}/arb5_d{d}.npz"))
    rec = json.load(open(f"{out}/arb5_d{d}.json"))
    if d == 9:
        R = f"{release}/d9_shallow/records/reads/"
        zn, zb = np.load(R + "d9stest_d9_b1w64_sh.npz"), np.load(R + "d9stest_base.npz")
        c, obs, cpm, pm = zn["c"], zb["obs"].astype(np.uint8), zb["cpm"].astype(np.uint8), zb["pm"].astype(np.uint8)
        idx = z["idx"]
        assert np.array_equal(idx, np.flatnonzero(np.abs(c) < TAU_D9))
        chunk = np.arange(len(c)) // CHUNK
        nch = len(c) // CHUNK
        z.update(c=c[idx], obs=obs[idx], cpm_read=cpm[idx], pm_read=pm[idx])
        z["count_net_fail"] = np.bincount(chunk[(c > 0).astype(np.uint8) != obs], minlength=nch)
        z["count_cpm_fail"] = np.bincount(chunk[cpm != obs], minlength=nch)
        rec["tau"] = {"0.06": TAU_D9}
    assert np.array_equal(z["pm"], z["pm_read"]) and np.array_equal(z["cpm"], z["cpm_read"]), "recomputed answers differ"
    return z, rec


def analyse(d, z, rec):
    idx, c, obs, cpm, pm = z["idx"], z["c"], z["obs"], z["cpm_read"], z["pm_read"]
    chunk = idx // CHUNK
    nch = len(z["count_net_fail"])
    even_ch = np.arange(nch) % 2 == 0
    netd = (c > 0).astype(np.uint8)
    S = {"plain": z["gap_pm"] * (2.0 * pm - 1), "corr": z["gap_cpm"] * (2.0 * cpm - 1)}
    res = dict(d=d, network=NETWORK[d], q_registered=QREG[d], shots=nch * CHUNK, rows=[])
    for h, hm in (("fit", even_ch), ("eval", ~even_ch)):
        res[f"{h}_shots"] = int(hm.sum()) * CHUNK
        res[f"{h}_cpm_alone"] = int(z["count_cpm_fail"][hm].sum())
        res[f"{h}_net_alone"] = int(z["count_net_fail"][hm].sum())
    half = chunk % 2 == 0
    for q in QREAD[d]:
        tau = rec["tau"][str(q)]
        r = np.abs(c) < tau
        D = r & (netd != cpm)
        row = dict(q=q, tau=tau, registered=q == QREG[d])
        for h, hm in (("fit", half), ("eval", ~half)):
            hc = even_ch if h == "fit" else ~even_ch
            comp = int(z["count_net_fail"][hc].sum()) - int(((netd != obs) & r & hm).sum()) + int(((cpm != obs) & r & hm).sum())
            row[h] = dict(routed=int((r & hm).sum()), disagree=int((D & hm).sum()), composition=comp,
                          oracle=comp - int((D & hm & (cpm != obs)).sum()),
                          only_cpm_fails=int((D & hm & (cpm != obs)).sum()), only_net_fails=int((D & hm & (cpm == obs)).sum()))
        cD, cpmD, obsD, hD = c[D], cpm[D], obs[D], half[D]
        for v, s in S.items():
            sD = s[D]
            fits = [gain(joint(cD, sD, cpmD, w), cpmD, obsD, hD) for w in WS]
            nets = [a - b for a, b in fits]
            best = max(nets)
            k = max(i for i, x in enumerate(nets) if x == best)            # ties keep the larger w
            w = WS[k]
            rep, brk = gain(joint(cD, sD, cpmD, w), cpmD, obsD, ~hD)
            e = row["eval"]
            n = res["eval_shots"]
            comp = e["composition"]
            lo, hi = newcombe_paired(comp - rep, rep, brk, n - comp - brk)
            ev_nets = [a - b for a, b in (gain(joint(cD, sD, cpmD, ww), cpmD, obsD, ~hD) for ww in WS)]
            row[v] = dict(w=w, fit_repairs=fits[k][0], fit_breaks=fits[k][1], fit_net=best,
                          fit_rel=best / row["fit"]["composition"] if row["fit"]["composition"] else None,
                          eval_failures=comp - rep + brk, repairs=rep, breaks=brk, net=rep - brk,
                          rel=(rep - brk) / comp if comp else None,
                          rel_ci=[lo * n / comp, hi * n / comp] if comp else None,
                          mcnemar_p=binomtest(rep, rep + brk, 0.5).pvalue if rep + brk else 1.0,
                          share_of_oracle=(rep - brk) / e["only_cpm_fails"] if e["only_cpm_fails"] else None,
                          eval_best_net=max(ev_nets), eval_net_w1=ev_nets[WS.index(1.0)])
        # against the paper's row (the registered composition), shot by shot on the evaluation half: the two differ only
        # on the shots read here (|c| < tau at q_max), and both take the network's answer elsewhere
        treg = rec["registered"]["tau"] if d != 9 else TAU_D9
        reg_ans = np.where(np.abs(c) < treg, cpm, netd)
        nrd_fail = int(z["count_net_fail"][~even_ch].sum()) - int(((netd != obs) & ~half).sum())   # elsewhere: both fail
        n = res["eval_shots"]
        for v, s in S.items():
            y = np.where(D, joint(c, s, cpm, row[v]["w"]), np.where(r, cpm, netd))
            fr, fj = (reg_ans != obs) & ~half, (y != obs) & ~half
            a_, b_, c_ = nrd_fail + int((fr & fj).sum()), int((fr & ~fj).sum()), int((~fr & fj).sum())
            regf = a_ + b_
            lo, hi = newcombe_paired(a_, b_, c_, n - a_ - b_ - c_)
            row[v]["vs_registered"] = dict(registered=regf, joint=a_ + c_, only_registered_fails=b_, only_joint_fails=c_,
                                           rel=(b_ - c_) / regf, rel_ci=[lo * n / regf, hi * n / regf],
                                           mcnemar_p=binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0)
            assert a_ + c_ == row[v]["eval_failures"], (d, q, v)
        # where the correlated gap is negative (the other class's correlated decode found a lighter correction than
        # correlated PM's own), s has the network's sign, so the joint decision takes the network's answer at any finite w
        neg = (z["gap_cpm"] < 0)
        yk = joint(c, S["corr"], cpm, row["corr"]["w"])
        for nm, m in (("corr_gap_negative", D & ~half & neg), ("corr_gap_nonnegative", D & ~half & ~neg)):
            rep, brk = gain(yk, cpm, obs, m)
            row["corr"][nm] = dict(rows=int(m.sum()), repairs=rep, breaks=brk, cpm_wrong=int((m & (cpm != obs)).sum()))
        # time per routed shot (us, one pinned core of the job): plain gap = plain PM + its split decode; correlated gap
        # = the split correlated decode (correlated PM's own weight comes with its decode); on every routed shot, and
        # amortized when it runs only on the disagreement rows
        t = {k: z[k][r] for k in ("t_pm", "t_split", "t_cpm", "t_c_other")}
        Dr = (netd != cpm)[r]
        row["time_us"] = dict(routed=int(r.sum()), disagree_fraction=float(Dr.mean()),
                              cpm=float(t["t_cpm"].mean()), plain_gap=float((t["t_pm"] + t["t_split"]).mean()),
                              corr_gap=float(t["t_c_other"].mean()),
                              plain_gap_amortized=float(((t["t_pm"] + t["t_split"]) * Dr).mean()),
                              corr_gap_amortized=float((t["t_c_other"] * Dr).mean()))
        rt = rec.get("registered_times", {}).get(str(q))
        if rt:
            row["time_us"]["registered_front_us"] = rt["front_us"]
            row["time_us"]["registered_split_us"] = rt["split_us"]
            row["time_us"]["registered_ratio_to_grace"] = rec["registered_times"]["all"]["ratio_to_grace"]
        res["rows"].append(row)
    res["checks"] = rec["checks"]
    res["host"] = dict(host=rec["host"], cpu=rec["cpu"], core=rec["core"], job=rec["job"])
    return res


def fmt(x):
    return f"{x:,}" if isinstance(x, int) else x


def tables(all_res):
    L = []
    L.append("| d | network | q | routed | disagreement rows | correlated PM alone | network alone | composition | joint (plain gap) | "
             "joint (corr. gap) | routed oracle | repairs / breaks (plain) | repairs / breaks (corr.) | relative reduction, plain [95 % CI] | "
             "relative reduction, corr. [95 % CI] | w plain / corr. |")
    L.append("|" + "---|" * 16)
    for r in all_res:
        for row in r["rows"]:
            e, p, k = row["eval"], row["plain"], row["corr"]
            ci = lambda x: (f"{100 * x['rel']:.1f} % [{100 * x['rel_ci'][0]:.1f}, {100 * x['rel_ci'][1]:.1f}]"
                            if x["rel"] is not None else "-")
            qs = f"{100 * row['q']:.0f} %" + (" (registered)" if row["registered"] else "")
            L.append(f"| {r['d']} | {r['network']} | {qs} | {e['routed']:,} | {e['disagree']:,} | {r['eval_cpm_alone']:,} | "
                     f"{r['eval_net_alone']:,} | {e['composition']:,} | {p['eval_failures']:,} | {k['eval_failures']:,} | "
                     f"{e['oracle']:,} | {p['repairs']:,} / {p['breaks']:,} | {k['repairs']:,} / {k['breaks']:,} | {ci(p)} | {ci(k)} | "
                     f"{p['w']:.3g} / {k['w']:.3g} |")
    return "\n".join(L)


def timing_7713(all_res, out):
    """The gaps' time per routed shot on the paper's CPU (arb5_alpine.py time: EPYC 7713, one pinned core, the first
    10,000 routed shots at the registered q, q = 2 % at d = 11 and 13) -> rows and a table."""
    tj = json.load(open(f"{out}/t7713/time.json"))
    rows = []
    for r in all_res:
        d = r["d"]
        t = dict(np.load(f"{out}/t7713/time_d{d}.npz"))
        z = np.load(f"{out}/arb5_d{d}.npz")
        if d == 9:
            R = r["_release"] + "/d9_shallow/records/reads/"
            c_all, cpm_all = np.load(R + "d9stest_d9_b1w64_sh.npz")["c"], np.load(R + "d9stest_base.npz")["cpm"]
            c, cpm = c_all[t["idx"]], cpm_all[t["idx"]]
        else:
            sel = np.searchsorted(z["idx"], t["idx"])
            c, cpm = z["c"][sel], z["cpm_read"][sel]
        D = (c > 0).astype(np.uint8) != cpm
        plain, corr = t["t_pm"] + t["t_split"], t["t_c_other"]
        x = tj["results"][str(d)]
        reg = [row["time_us"] for row in r["rows"] if row["q"] == x["q"]]
        rows.append(dict(d=d, q=x["q"], shots=x["shots"], same_as_main_job=x["same_as_main_job"],
                         cpm_us=float(t["t_cpm"].mean()), plain_gap_us=float(plain.mean()), corr_gap_us=float(corr.mean()),
                         plain_over_cpm=float(plain.mean() / t["t_cpm"].mean()), corr_over_cpm=float(corr.mean() / t["t_cpm"].mean()),
                         disagree_fraction=float(D.mean()), plain_gap_amortized_us=float((plain * D).mean()),
                         corr_gap_amortized_us=float((corr * D).mean()),
                         registered_front_us=reg[0].get("registered_front_us") if reg else None))
    L = ["| d | q | routed shots timed | correlated PM (us) | plain gap (us) | correlated gap (us) | plain gap / correlated PM | "
         "correlated gap / correlated PM | share of routed shots where the two answers differ | plain gap, only on those (us per routed shot) | "
         "correlated gap, only on those (us per routed shot) | plain gap, registered gap-switching calibration (us) |", "|" + "---|" * 12]
    for x in rows:
        rf = f"{x['registered_front_us']:.0f}" if x["registered_front_us"] else "-"
        L.append(f"| {x['d']} | {100 * x['q']:.0f} % | {x['shots']:,} | {x['cpm_us']:.0f} | {x['plain_gap_us']:.0f} | {x['corr_gap_us']:.0f} | "
                 f"{x['plain_over_cpm']:.2f} | {x['corr_over_cpm']:.2f} | {100 * x['disagree_fraction']:.0f} % | "
                 f"{x['plain_gap_amortized_us']:.0f} | {x['corr_gap_amortized_us']:.0f} | {rf} |")
    return rows, "\n".join(L), dict(host=tj["host"], cpu=tj["cpu"], core=tj["core"], job=tj["job"])


def registered_table(all_res):
    L = ["| d | q | the paper's row: registered composition | joint (corr. gap) at q | difference [95 % CI] | exact McNemar p | "
         "joint (plain gap) at q | difference [95 % CI] |", "|" + "---|" * 8]
    for r in all_res:
        for row in r["rows"]:
            k, p = row["corr"]["vs_registered"], row["plain"]["vs_registered"]
            f = lambda x: f"{-100 * x['rel']:+.1f} % [{-100 * x['rel_ci'][1]:+.1f}, {-100 * x['rel_ci'][0]:+.1f}]"
            L.append(f"| {r['d']} | {100 * row['q']:.0f} %{' (registered)' if row['registered'] else ''} | {k['registered']:,} | "
                     f"{k['joint']:,} | {f(k)} | {k['mcnemar_p']:.2g} | {p['joint']:,} | {f(p)} |")
    return "\n".join(L)


def fit_table(all_res):
    L = ["| d | q | fit-half disagreement rows | fit-half composition | w (plain) | fit-half net (plain) | w (corr.) | fit-half net (corr.) | "
         "eval half: net at the fitted w / at the eval half's own best w / at w = 1 (corr.) |", "|" + "---|" * 9]
    for r in all_res:
        for row in r["rows"]:
            f, p, k = row["fit"], row["plain"], row["corr"]
            L.append(f"| {r['d']} | {100 * row['q']:.0f} % | {f['disagree']:,} | {f['composition']:,} | {p['w']:.3g} | {p['fit_net']:+,} | "
                     f"{k['w']:.3g} | {k['fit_net']:+,} | {k['net']:+,} / {k['eval_best_net']:+,} / {k['eval_net_w1']:+,} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--d", type=int, nargs="*", default=[7, 9, 11, 13, 15, 17])
    a = ap.parse_args()
    res = []
    for d in a.d:
        z, rec = load(d, a.out, a.release)
        res.append(analyse(d, z, rec))
    t = tables(res) + "\n\n" + registered_table(res) + "\n\n" + fit_table(res)
    try:
        for r in res:
            r["_release"] = a.release
        trows, tt, thost = timing_7713(res, a.out)
        t += "\n\n" + tt
    except FileNotFoundError as e:
        trows, thost = None, None
        print(f"no EPYC 7713 timing yet ({e})")
    for r in res:
        r.pop("_release", None)
    json.dump(dict(distances=res, timing_7713=trows, timing_host=thost), open(f"{a.out}/arb5_results.json", "w"), indent=1)
    open(f"{a.out}/arb5_tables.md", "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
