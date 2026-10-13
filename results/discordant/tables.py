"""X2: the markdown tables of X2_RESULT.md from the joins' summaries -> $X2_OUT/tables.md.

  python tables.py"""
import json
import os

OUT = os.environ.get("X2_OUT", "out")


def i(x):
    return f"{int(x):,}"


def pc(a, b, nd=1):
    return f"{100.0 * a / b:.{nd}f} %" if b else "-"


def med(v, nd=1):
    """median [IQR] from a [25, 50, 75] list."""
    if v[1] is None:
        return "-"
    return f"{v[1]:.{nd}f} [{v[0]:.{nd}f}, {v[2]:.{nd}f}]"


def med5(v, nd=1):
    """median [10th, 90th] from a [10, 25, 50, 75, 90] list."""
    if v[2] is None:
        return "-"
    return f"{v[2]:.{nd}f} [{v[0]:.{nd}f}, {v[4]:.{nd}f}]"


def surface(S, lines):
    for d in ("13", "17"):
        s = S[d]
        lines += [f"### d = {d} ({s['test_set']}, {i(s['shots'])} shots, q = {100 * s['q']:.0f} %, tau = {s['tau']})", ""]
        lines += ["| cell | shots | routed to correlated PM | kept |", "|---|---|---|---|"]
        for name, n in s["cells"].items():
            r = s["cells_routed"][name]
            lines.append(f"| {name} | {i(n)} | {i(r)} ({pc(r, n)}) | {i(n - r)} ({pc(n - r, n)}) |")
        dec = s["pair_decomposition"]
        lines += ["", f"Network alone {i(s['network_alone_failures'])}, correlated PM {i(s['correlated_pm_failures'])}, "
                      f"plain PM {i(s['plain_pm_failures'])}, pair {i(s['pair_failures'])} = kept shots only the network "
                      f"fails {i(dec['nf_kept'])} + routed shots only correlated PM fails {i(dec['mf_routed'])} + both "
                      f"fail {i(dec['bf'])}. Retained-shot excess (kept: only the network fails - only correlated PM "
                      f"fails) = {i(s['kept_nf'])} - {i(s['kept_mf'])} = {s['retained_excess']:+,}.", ""]
        D = s["describe"]
        lines += ["| shots | n | routed | detection events | X-type events | time-like pairs / 100 X events | "
                  "space-like pairs / 100 X | space-time pairs / 100 X | rows to nearer X boundary, mean per shot | "
                  "\\|c\\| percentile, median [10th, 90th] |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, x in D.items():
            lines.append(f"| {name} | {i(x['shots'])} | {pc(x['routed'], x['shots'])} | {med(x['n_det'], 0)} | "
                         f"{med(x['n_x'], 0)} | {med(x['time_pairs_per100x'])} | {med(x['space_pairs_per100x'])} | "
                         f"{med(x['spacetime_pairs_per100x'])} | {med(x['x_event_bdist_mean'], 2)} | "
                         f"{med5(x['pct_abs_c'])} |")
        lines += ["", "| shots | n | plain PM fails | plain class gap, nats: median [10th, 90th] | \\|plain gap\\| "
                  "< 2 nats | correlated class gap (approximate), nats: median [10th, 90th] | correlated gap > 0 | "
                  "class difference D: time-like edges | D: a run of >= 3 time-like edges at one site | "
                  "D: rounds spanned |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, x in D.items():
            ns = x["structure_shots"]
            lines.append(f"| {name} | {i(x['shots'])} | {pc(x['plain_pm_fails'], x['shots'])} | "
                         f"{med5(x['plain_gap'])} | {pc(x['plain_gap_abs_below_2'], x['shots'])} | "
                         f"{med5(x['corr_gap'])} | {pc(x['corr_gap_positive'], x['shots'])} | "
                         f"{med(x['D_time'], 0)} | {pc(x['D_maxrun_ge3'], ns)} of {i(ns)} | {med(x['D_text'], 0)} |")
        a, b = D["only the network fails"], D["only correlated PM fails"]
        p1, p2 = a["D_maxrun_ge3"] / a["structure_shots"], b["D_maxrun_ge3"] / b["structure_shots"]
        se = (p1 * (1 - p1) / a["structure_shots"] + p2 * (1 - p2) / b["structure_shots"]) ** 0.5
        lines += ["", f"A run of >= 3 time-like edges at one site in D: only the network fails {100 * p1:.1f} %, only "
                      f"correlated PM fails {100 * p2:.1f} %, difference {100 * (p1 - p2):+.1f} points "
                      f"[{100 * (p1 - p2 - 1.96 * se):+.1f}, {100 * (p1 - p2 + 1.96 * se):+.1f}] (95 %, normal "
                      f"approximation, independent cells; descriptive).", ""]
        lines += ["", "| least confident fraction routed | shots | network failures among them (share of all) | "
                  "network failure rate there | correlated PM failures there (rate) | pair failures | pair - correlated "
                  "PM |", "|---|---|---|---|---|---|---|"]
        for r in s["capture"]:
            lab = f"registered tau ({100 * r['frac']:.2f} %)" if r["registered"] else f"{100 * r['frac']:.0f} %"
            lines.append(f"| {lab} | {i(r['routed'])} | {i(r['net_fail_routed'])} "
                         f"({100 * r['net_fail_share']:.1f} %) | {100 * r['net_rate_routed']:.1f} % | "
                         f"{i(r['cpm_fail_routed'])} ({100 * r['cpm_rate_routed']:.1f} %) | {i(r['pair_fail'])} | "
                         f"{r['pair_minus_cpm']:+,} |")
        lines.append("")
    return lines


def bb(B, lines):
    for p in ("0.002", "0.003"):
        s = B[p]
        lines += [f"### BB72, p = {100 * float(p):.1f} % ({i(s['shots'])} shots, routed {i(s['routed'])} = "
                  f"{pc(s['routed'], s['shots'], 2)})", ""]
        lines += [f"S55'-deep alone {i(s['S'])}, C14' alone {i(s['C'])}, pair {i(s['D'])} failures. C14''s failures "
                  f"on kept shots: {pc(s['C_kept_share_of_C'], 1)} of its failures; S55'-deep's failures routed: "
                  f"{pc(s['S_routed_share_of_S'], 1)}. Compiled C14' paths (kept repairs / breaks): single "
                  f"{s['compiled']['single']['repairs']} / {s['compiled']['single']['breaks']}, batched "
                  f"{s['compiled']['batched']['repairs']} / {s['compiled']['batched']['breaks']}.", ""]
        lines += ["| shots | n | detection events | S55'-deep margin percentile, median [10th, 90th] | C14' confidence "
                  "percentile, median [10th, 90th] | C14' logicals wrong: 1 / 2 / >= 3 | C14' \\|logit\\| of its wrong "
                  "logical, median [IQR] | time-like pairs / 100 events |", "|---|---|---|---|---|---|---|---|"]
        for name, x in s["describe"].items():
            w = x["C_logicals_wrong"]
            zw = "-" if name.startswith("all shots") else med(x["C_abs_logit_wrong_min"], 2)
            lines.append(f"| {name} | {i(x['shots'])} | {med(x['n_det'], 0)} | {med5(x['S_margin_pct'])} | "
                         f"{med5(x['C_conf_pct'], 2)} | {w[1]:,} / {w[2]:,} / {w[3]:,} | {zw} | "
                         f"{med(x['time_pairs_per100'])} |")
        lines += ["", "| least sure fraction routed | shots | S55'-deep failures among them (share) | C14' failures "
                  "there (rate) | pair failures | pair - C14' |", "|---|---|---|---|---|---|"]
        for r in s["capture"]:
            lab = f"registered tau1 ({100 * r['frac']:.2f} %)" if r["registered"] else f"{100 * r['frac']:.0f} %"
            lines.append(f"| {lab} | {i(r['routed'])} | {i(r['S_fail_routed'])} "
                         f"({100 * r['S_fail_share']:.1f} %) | {i(r['C_fail_routed'])} ({100 * r['C_rate_routed']:.1f} %)"
                         f" | {i(r['pair_fail'])} | {r['pair_minus_C']:+,} |")
        lines.append("")
    return lines


def main():
    lines = ["# X2 tables (generated by tables.py)", ""]
    lines = surface(json.load(open(os.path.join(OUT, "surface", "summary.json"))), lines)
    bpath = os.path.join(OUT, "bb", "summary.json")
    if os.path.exists(bpath):
        lines = bb(json.load(open(bpath)), lines)
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} lines -> {OUT}/tables.md")


if __name__ == "__main__":
    main()
