"""ARB at d = 13: the tables of RESULTS.md from arb2_analysis.py's summary.json (formatting only, no new numbers).

  python arb2_tables.py OUT/analysis/summary.json > OUT/analysis/tables.md"""
import json
import math
import sys

NAMES = {"J": "J (joint decision)", "P": "P (preference)", "T": "T (two thresholds)"}


def n(x):
    return f"{x:,}"


def g(x, d=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{d}g}"


def pct(x, d=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}"


def pval(x):
    return "< 1e-300" if x == 0 else f"{x:.3g}"           # scipy's binomtest underflows to 0


def params(nm, q):
    if nm == "J":
        return f"w = {g(q['w'])} (step [{g(q['step'][0], 7)}, {g(q['step'][1], 7)}))"
    if nm == "P":
        return f"a = {g(q['a'])}, b = {g(q['b'])}"
    return f"t_C = {g(q['t_C'])}, t_S = {g(q['t_S'])}"


def row(name, e, fitted=""):
    return (f"| {name} | {n(e['failures'])} | {n(e['repairs'])} | {n(e['breaks'])} | {e['net_per_1e4']:+.3g} "
            f"[{e['net_per_1e4_lo']:+.3g}, {e['net_per_1e4_hi']:+.3g}] | {pct(e['rel_reduction'])} % "
            f"[{pct(e['rel_lo'])}, {pct(e['rel_hi'])}] | {pct(e['oracle_share'], 1)} % | {pval(e['mcnemar_p'])} | "
            f"{fitted} |")


def main(path):
    s = json.load(open(path))
    out = []
    for p, r in s["p"].items():
        ev, ft, al = r["counts"]["eval"], r["counts"]["fit"], r["counts"]["all"]
        out.append(f"### p = {100 * float(p):g} % (evaluation half: odd chunks, {n(r['eval_shots'])} shots; "
                   f"tau1 = {r['tau1']})\n")
        out.append("| decoder | failures | repairs | breaks | net (repairs - breaks) per 10^4 shots, 95 % CI | "
                   "relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition "
                   "| fitted on the fit half (fit-half net) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        out.append(f"| Cascade alone | {n(ev['C'])} | | | | | | | |")
        out.append(f"| network alone | {n(ev['S'])} | | | | | | | |")
        out.append(f"| composition (the comparison's C5) | {n(ev['composition'])} | | | | | | | |")
        for nm in ("J", "P", "T"):
            q = r["rules"][nm]
            out.append(row(NAMES[nm], q["eval"], f"{params(nm, q['params'])} ({q['fit']['gain']:+,})"))
        out.append(row("reference: always the network on the disagreement rows",
                       r["reference"]["always_network_on_disagreement"]))
        out.append(row("reference: routed oracle", r["reference"]["routed_oracle"]))
        out.append("")
        fit = "; ".join(f"{nm} {q['fit']['repairs']:,} repairs, {q['fit']['breaks']:,} breaks, net "
                        f"{q['fit']['gain']:+,} ({pct(q['fit']['rel_reduction'])} %)" for nm, q in r["rules"].items())
        out.append(f"Fit half (even chunks, {n(r['fit_shots'])} shots; composition {n(ft['composition'])} failures, "
                   f"routed oracle repairs {n(ft['oracle_repairs'])}): {fit}.\n")
        dz = r["disagreement"]
        out.append(f"Evaluation half: routed {n(ev['routed'])} ({pct(ev['routed_share'])} %), disagreement rows "
                   f"{n(dz['eval_rows'])} (only Cascade fails {n(dz['eval_B_wrong'])}, only the network fails "
                   f"{n(dz['eval_S_wrong'])}); routed oracle failures {n(ev['oracle_fail'])}. Fit half: disagreement "
                   f"rows {n(dz['fit_rows'])} (only Cascade fails {n(dz['fit_B_wrong'])}, only the network fails "
                   f"{n(dz['fit_S_wrong'])}).\n")
        jb = r["j_eval_own_best"]
        out.append(f"J diagnostics: the evaluation half's own best w = {g(jb['w'])} (step [{g(jb['step'][0], 7)}, "
                   f"{g(jb['step'][1], 7)}), evaluation net {jb['eval_gain']:+,}); the fitted w's evaluation net "
                   f"{jb['fitted_w_eval_gain']:+,}.\n")
        out.append(f"Per-category counts of the whole read ({n(r['shots'])} shots) beside the re-test (paper Table I, "
                   f"single mode) and the comparison's original single-mode counts (rates per 10^6 shots; z of the "
                   f"difference, independent samples):\n")
        out.append("| count | ARB | ARB rate | re-test | re-test rate | z | original | original rate | z |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        lab = {"C2": "Cascade alone (C2)", "C5": "composition (C5)", "only_pair": "only the composition fails",
               "only_C2": "only Cascade alone fails"}
        vr = r["vs_reference"]
        for k in ("C2", "C5", "only_pair", "only_C2"):
            a_, b_ = vr["retest_single"][k], vr["original_single"][k]
            out.append(f"| {lab[k]} | {n(a_['arb'])} | {a_['arb_per_1e6']:.1f} | {n(a_['ref'])} | "
                       f"{a_['ref_per_1e6']:.1f} | {g(a_['z'], 3)} | {n(b_['ref'])} | {b_['ref_per_1e6']:.1f} | "
                       f"{g(b_['z'], 3)} |")
        rs = vr["retest_single"].get("routed_share")
        if rs:
            out.append(f"| routed share | {pct(rs['arb'], 3)} % | | {pct(rs['ref'], 2)} % | | | "
                       f"{pct(vr['original_single']['routed_share']['ref'], 2)} % | | |")
        out.append("")
        out.append(f"Whole read: network alone {n(al['S'])}, only the network fails {n(al['only_S'])}, only Cascade "
                   f"fails {n(al['only_C'])}, both {n(al['both'])}, routed {n(al['routed'])}, routed oracle failures "
                   f"{n(al['oracle_fail'])}.\n")
    if "choice" in s:
        ch = s["choice"]
        out.append(f"Carried forward (fit half only): {ch['carried_forward']}; mean fit-half relative reduction over "
                   f"the two p: " + ", ".join(f"{k} {pct(v)} %" for k, v in ch["fit_scores"].items()) + ".\n")
        cw = s["j_common_w"]
        per = "; ".join(f"p = {100 * float(p):g} %: fit net {v['fit_gain']:+,}, evaluation net {v['eval']['net']:+,} "
                        f"({pct(v['eval']['rel_reduction'])} %)" for p, v in cw["per_p"].items())
        out.append(f"One w for both p (diagnostic): w = {g(cw['w'])}, mean fit-half relative reduction "
                   f"{pct(cw['fit_mean_rel_reduction'])} %; {per}.\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
