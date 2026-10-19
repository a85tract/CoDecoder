"""RESULTS.md's tables from out/analysis/summary.json (formatting only; every number is the pod's analysis output).
  python3 results_tables.py ~/work/arb/out/analysis/summary.json > tables_for_results.md"""
import json
import sys

ARBS = ("A1m", "A1s", "A2", "A3")
LABEL = {"A1m": "A1m (min abs logit)", "A1s": "A1s (softplus score)", "A2": "A2 (preference)", "A3": "A3 (joint score)"}


def pv(x):
    if x == 0:
        return "< 1e-300"
    return f"{x:.1e}"


def ci(lo, hi, scale=1e4):
    return f"[{lo * scale:.2f}, {hi * scale:.2f}]"


def params(nm, t):
    if nm.startswith("A1"):
        return f"t_C = {t['t_C']:.4g}, t_S = {t['t_S']:.2g}"
    if nm == "A2":
        return f"a = {t['a']:.4g}, b = {t['b']:.4g}"
    return f"w = {t['w']:.4g}"


def main():
    r = json.load(open(sys.argv[1]))
    out = []
    for p, x in r["points"].items():
        e, n = x["counts"]["eval"], x["eval_shots"]
        q = {"0.002": "2 %", "0.003": "10 %"}[p]
        out.append(f"### p = {float(p) * 100:.1f} %: evaluation half (odd chunks), {n:,} shots\n")
        out.append("| decoder | failures | repairs | breaks | net | net per 10^4 shots [95 % CI] | relative failure "
                   "reduction [approx. 95 % CI] | share of the routed oracle | exact McNemar p vs composition |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        out.append(f"| Cascade alone (C14') | {e['C']:,} | | | | | | | |")
        out.append(f"| composition (S55'-deep, {q} routed to C14') | {e['composition']:,} | | | | | | | |")
        for nm in ARBS:
            v = x["arbiters"][nm]["eval"]
            out.append(f"| {LABEL[nm]} | {v['failures']:,} | {v['repairs']:,} | {v['breaks']:,} | {v['net']:+,} | "
                       f"{v['net_per_shot'] * 1e4:.2f} {ci(v['net_lo'], v['net_hi'])} | {v['rel_reduction'] * 100:.1f} % "
                       f"[{v['rel_lo'] * 100:.1f}, {v['rel_hi'] * 100:.1f}] | {v['oracle_share'] * 100:.1f} % | "
                       f"{pv(v['mcnemar_p'])} |")
        for nm, lab in (("routed_oracle", "routed oracle (reference)"),
                        ("always_S_on_disagreement", "always S55'-deep on the disagreement rows (reference)")):
            v = x["reference"][nm]
            out.append(f"| {lab} | {v['failures']:,} | {v['repairs']:,} | {v['breaks']:,} | {v['net']:+,} | "
                       f"{v['net_per_shot'] * 1e4:.2f} {ci(v['net_lo'], v['net_hi'])} | {v['rel_reduction'] * 100:.1f} % | "
                       f"{v['oracle_share'] * 100:.1f} % | {pv(v['mcnemar_p'])} |")
        t = x["arbiters"]["A3"]["eval"]["third"]
        out.append("")
        out.append(f"A3's third-answer changes (answers that are neither network's): {t['changes']:,}, of which "
                   f"{t['repairs']:,} repairs, {t['breaks']:,} breaks and {t['neutral']:,} neutral (both answers and A3's "
                   f"wrong). Its two-answer changes: {x['arbiters']['A3']['eval']['repairs'] - t['repairs']:,} repairs, "
                   f"{x['arbiters']['A3']['eval']['breaks'] - t['breaks']:,} breaks. No arbiter is without breaks, so the "
                   f"3/N bound does not apply (3/N_eval would be {3 / n:.1e}).")
        out.append("")
        out.append(f"Disagreement rows of the evaluation half: {e['routed_differ']:,} of {e['routed']:,} routed shots "
                   f"(only C14' fails {e['oracle_repairs']:,}, only S55'-deep fails {e['routed_only_S']:,}, both fail "
                   f"{e['routed_differ'] - e['oracle_repairs'] - e['routed_only_S']:,}); the routed oracle keeps "
                   f"{e['oracle_fail']:,} failures.\n")
        f = x["counts"]["fit"]
        out.append(f"Fitted on the fit half (even chunks, {x['fit_shots']:,} shots, composition {f['composition']:,} "
                   f"failures, {x['features']['fit_rows']:,} disagreement rows):\n")
        out.append("| arbiter | parameters | fit-half repairs | breaks | net | relative reduction |")
        out.append("|---|---|---|---|---|---|")
        for nm in ARBS:
            a = x["arbiters"][nm]
            out.append(f"| {nm} | {params(nm, a['params'])} | {a['fit']['repairs']:,} | {a['fit']['breaks']:,} | "
                       f"{a['fit']['gain']:+,} | {a['fit']['rel_reduction'] * 100:.2f} % |")
        out.append("")
    out.append("### Per-category counts of the whole reads beside D1's original reads\n")
    out.append("| count | ARB 0.2 % | D1 0.2 % | z | ARB 0.3 % | D1 0.3 % | z |")
    out.append("|---|---|---|---|---|---|---|")
    a, b = r["points"]["0.002"]["vs_d1"], r["points"]["0.003"]["vs_d1"]
    for k in a:
        out.append(f"| {k} | {a[k]['arb']:,} | {a[k]['d1']:,} | {a[k]['z']:+.2f} | {b[k]['arb']:,} | {b[k]['d1']:,} | "
                   f"{b[k]['z']:+.2f} |")
    out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
