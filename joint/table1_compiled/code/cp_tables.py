"""ARBCP: the tables of RESULTS.md from cp_count.py's and cp_count_d13.py's JSON (formatting only, no new numbers).

  python cp_tables.py --bb72 out/bb72_counts.json [--d13 out/d13_counts.json] > out/tables.md"""
import argparse
import json


def pfmt(m):
    if m["p"] >= 1e-3:
        return f"{m['p']:.3f}"
    if m["p"] > 1e-300:
        return f"{m['p']:.1e}"
    return f"1e{m['log10p']:.0f}"


def pct(x):
    return f"{100 * x:+.1f} %"


def rows(setting, paths, order):
    out = []
    for name in order:
        r = paths[name]
        mc, ma = r["mcnemar_vs_composition"], r["mcnemar_vs_alone"]
        out.append(f"| {setting} | {name} | {r['alone']:,} | {r['composition']:,} | {r['joint']:,} | "
                   f"{r['repairs']:,} / {r['breaks']:,} | {pct(r['rel_vs_composition'])} | {pfmt(mc)} | "
                   f"{pct(r['rel_vs_alone'])} | {pfmt(ma)} ({ma['b']:,} / {ma['c']:,}) |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bb72", required=True)
    ap.add_argument("--d13")
    a = ap.parse_args()
    L = ["| setting | path | Cascade alone | composition | joint decision | repairs / breaks | joint vs composition | "
         "McNemar p | joint vs Cascade alone | McNemar p (only Cascade fails / only joint fails) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    bb = json.load(open(a.bb72))
    for p, lab in (("0.002", "BB72 0.2 %"), ("0.003", "BB72 0.3 %")):
        if p in bb:
            L += rows(lab, bb[p]["paths"], ("single", "batched", "eager"))
    if a.d13:
        d = json.load(open(a.d13))
        for p, lab in (("0.006", "d = 13, 0.6 %"), ("0.0035", "d = 13, 0.35 %")):
            if p in d:
                L += rows(lab, d[p]["paths"], ("single", "batched"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
