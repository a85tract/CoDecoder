"""check_paper_numbers.py: every number that the paper prints about the joint decision, recomputed from the records
in joint/ and compared with the paper's text.

    python3 joint/check_paper_numbers.py          (standard library only; well under a second)

Each line gives where the paper prints the number, the paper's value, the value from the records (rounded as the paper
rounds) and the record it comes from. A line ends in "ok" or "MISMATCH"; the script exits 1 if any line is a mismatch.
The paper is "CoDecoder: Make Quantum Decoders Up to 100x Faster and Even More Accurate", in its version of
19 October 2026. Table I is its table of the Cascade comparisons; Section III B states the rule.
"""
import glob
import json
import math
import os
import sys

J = os.path.dirname(os.path.abspath(__file__))
BAD, NOTES = [], []


def load(rel):
    return json.load(open(os.path.join(J, rel)))


def show(where, paper, ours, src, ok=None, note=""):
    ok = (paper == ours) if ok is None else ok
    if not ok:
        BAD.append(where)
    elif note:
        NOTES.append(f"{where}: {note}")
    flag = "MISMATCH" if not ok else ("ok, see note" if note else "ok")
    print(f"{where:<58} paper {paper:<14} records {ours:<14} {flag}   [{src}]")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def span(vals, nd=0, sep="-"):
    lo, hi = f"{min(vals):.{nd}f}", f"{max(vals):.{nd}f}"
    return lo if lo == hi else f"{lo}{sep}{hi}"


def log10_mcnemar(b, c):
    """log10 of the exact two-sided McNemar p-value, 2 * P(X <= min(b, c)) for X ~ Bin(b + c, 1/2), capped at 1."""
    n, k = b + c, min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(0.0, math.log10(2 * tail) - n * math.log10(2))


CP = "table1_compiled/out"
bb = load(f"{CP}/bb72_counts.json")
d13 = load(f"{CP}/d13_counts.json")
SET = [("BB72, p = 0.2 %", bb["0.002"]), ("BB72, p = 0.3 %", bb["0.003"]),
       ("d = 13, p = 0.6 %", d13["0.006"]), ("d = 13, p = 0.35 %", d13["0.0035"])]

print("== Table I: the joint decision's failures on Table I's test shots (Cascade's compiled paths)")
PAPER_FAIL = {"BB72, p = 0.2 %": ("6,597", "6,588"), "BB72, p = 0.3 %": ("65,189", "65,185"),
              "d = 13, p = 0.6 %": ("4,886", "4,884"), "d = 13, p = 0.35 %": ("2,306", "2,308")}
PAPER_BASE = {"BB72, p = 0.2 %": ("8,074", "8,073", "7,986", "7,984"),
              "BB72, p = 0.3 %": ("75,979", "75,973", "75,740", "75,735"),
              "d = 13, p = 0.6 %": ("4,936", "4,947", "4,917", "4,927"),
              "d = 13, p = 0.35 %": ("2,354", "2,352", "2,325", "2,322")}
PAPER_ROUTED = {"BB72, p = 0.2 %": "2", "BB72, p = 0.3 %": "10", "d = 13, p = 0.6 %": "10", "d = 13, p = 0.35 %": "1"}
for name, S in SET:
    P = S["paths"]
    for k, mode in enumerate(("single", "batched")):
        show(f"Table I {name}, joint decision, {mode}", PAPER_FAIL[name][k], f"{P[mode]['joint']:,}",
             f"{CP}/{'bb72' if 'BB72' in name else 'd13'}_counts.json")
    a1, ab, c1, cb = PAPER_BASE[name]
    same = (f"{P['single']['alone']:,}", f"{P['batched']['alone']:,}", f"{P['single']['composition']:,}",
            f"{P['batched']['composition']:,}") == (a1, ab, c1, cb)
    show(f"  same shots as Table I's Cascade alone and composition", f"{a1}/{c1}",
         f"{P['single']['alone']:,}/{P['single']['composition']:,}", "the same files", same)
    routed = S.get("routed", P["single"].get("routed"))
    show(f"Table I {name}, routed (%)", PAPER_ROUTED[name], f"{100 * routed / S['shots']:.0f}",
         f"routed {routed:,} of {S['shots']:,}")

print("\n== Table I 'better' and Appendix E: one Holm procedure over the eight exact McNemar tests against Cascade alone")
tests = []
for name, S in SET:
    for mode in ("single", "batched"):
        m = S["paths"][mode]["mcnemar_vs_alone"]
        lp = log10_mcnemar(m["b"], m["c"])
        if abs(lp - m["log10p"]) > 1e-6:
            BAD.append(f"recorded McNemar p of {name} {mode}")
        tests.append((lp, name, mode, m["b"], m["c"]))
tests.sort()
holm = all(lp <= math.log10(0.05 / (len(tests) - i)) for i, (lp, *_) in enumerate(tests))
rej = sum(1 for _, _, _, b, c in tests if b > c) if holm else 0
show("Holm at alpha = 0.05 rejects all eight ('better')", "8 of 8", f"{rej} of {len(tests)}",
     "mcnemar_vs_alone (b, c) of both files, p recomputed exactly")
lp, name, mode, b, c = tests[-1]
show("App. E: the largest p-value", "0.033", f"{10 ** lp:.3f}", f"{name}, {mode}")
show("App. E: its setting", "d=13 0.6% single", "d=13 0.6% single" if (name, mode) == ("d = 13, p = 0.6 %", "single")
     else f"{name} {mode}", "")
show("App. E: shots failing only with Cascade alone / only joint", "289/239", f"{b}/{c}", "mcnemar_vs_alone")

print("\n== Section IV D, Section VI, abstract, conclusion, related work: fewer failures than Cascade alone")
rel_bb = [1 - S["paths"][m]["joint"] / S["paths"][m]["alone"] for _, S in SET[:2] for m in ("single", "batched")]
rel_13 = [1 - S["paths"][m]["joint"] / S["paths"][m]["alone"] for _, S in SET[2:] for m in ("single", "batched")]
show("Sec. IV D: BB72, p = 0.2 %, vs Cascade alone (%)", "18", span([100 * x for x in rel_bb[:2]]), "both modes")
show("Sec. IV D: BB72, p = 0.3 %, vs Cascade alone (%)", "14", span([100 * x for x in rel_bb[2:]]), "both modes")
show("Sec. IV D: d = 13, vs Cascade alone (%)", "1.0-2.0", span([100 * x for x in rel_13], 1), "both p, both modes")
show("Sec. VI, abstract, conclusion: BB72 (%)", "14-18", span([100 * x for x in rel_bb]), "both p, both modes")
rel_c = [1 - S["paths"][m]["joint"] / S["paths"][m]["composition"] for _, S in SET[:2] for m in ("single", "batched")]
show("Related work: 'removed a further 14-18 %', vs Cascade", "14-18", span([100 * x for x in rel_bb]),
     f"vs the composition instead: {span([100 * x for x in rel_c], 1)} %",
     note=f"against Cascade alone's failures; against the composition's the range is {span([100 * x for x in rel_c], 1)} %"
          f" ({span([100 * x for x in rel_c])} %)")

print("\n== Section VI and Appendix E: repairs and breaks against the composition")
ratios = [S["paths"][m]["repairs"] / S["paths"][m]["breaks"] for _, S in SET[:2] for m in ("single", "batched")]
show("Sec. VI: BB72 repairs per break", "14 to 49", span(ratios, sep=" to "), "repairs / breaks, both p and modes")
thirds = [S["paths"][m]["third_repairs"] / S["paths"][m]["repairs"] for _, S in SET[:2] for m in ("single", "batched")]
show("Sec. VI, App. E: repairs with a third answer", "about a third", span([100 * x for x in thirds]) + " %",
     "third_repairs / repairs", all(abs(x - 1 / 3) < 0.05 for x in thirds))
s6 = d13["0.006"]["paths"]["single"]
show("Sec. VI: d = 13, single, p = 0.6 %: repairs / breaks", "249/218", f"{s6['repairs']}/{s6['breaks']}", f"{CP}/d13_counts.json")

print("\n== Appendix E: the shots the joint decision can change (routed, the two decisions differ)")
for (name, S), paper in zip(SET, ("0.08", "0.56", "0.62", "0.01")):
    show(f"App. E: {name}, share of all shots (%)", paper, f"{100 * S['paths']['single']['disagree'] / S['shots']:.2f}",
         f"disagree {S['paths']['single']['disagree']:,} of {S['shots']:,} (single path)")

print("\n== Appendix E and Section III B: the fit of w on separate fresh shots, frozen for Table I and the timing")
fb = load("fit_bb72/out/analysis/summary.json")["points"]
fd = load("fit_d13/out/analysis/summary.json")["p"]
W = {"0.002": fb["0.002"]["arbiters"]["A3"]["params"]["w"], "0.003": fb["0.003"]["arbiters"]["A3"]["params"]["w"],
     "0.006": fd["0.006"]["rules"]["J"]["params"]["w"], "0.0035": fd["0.0035"]["rules"]["J"]["params"]["w"]}
show("App. E: BB72 fit shots", "5,500,000/8,500,000",
     f"{fb['0.002']['fit_shots']:,}/{fb['0.003']['fit_shots']:,}", "fit_bb72 summary.json (even chunks)")
show("App. E: BB72 w (log grid 10^(k/40))", "6.31/3.76", f"{W['0.002']:.2f}/{W['0.003']:.2f}",
     f"k = {40 * math.log10(W['0.002']):.3f}, {40 * math.log10(W['0.003']):.3f}")
show("App. E: d = 13 fit shots", "2,000,000/15,000,000",
     f"{fd['0.006']['fit_shots']:,}/{fd['0.0035']['fit_shots']:,}", "fit_d13 summary.json (even chunks)")
show("App. E: d = 13 w (every breakpoint)", "7.87/13.84", f"{W['0.006']:.2f}/{W['0.0035']:.2f}", "rule J")
frozen = {"0.002": bb["0.002"]["w"], "0.003": bb["0.003"]["w"], "0.006": d13["0.006"]["w"], "0.0035": d13["0.0035"]["w"]}
timed = {}
for f in glob.glob(os.path.join(J, "timing/records/*/out/**/slots/*joint*/a*/rows.jsonl"), recursive=True):
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            timed.setdefault(str(r["p"]), set()).add((r.get("joint") or {}).get("w", r.get("joint_w")))
same_w = all(frozen[p] == W[p] and timed.get(p) == {W[p]} for p in W)
show("Sec. III B: the same w frozen for Table I and the timing", "frozen", "frozen" if same_w else "differs",
     "Table I counts and every joint timing process", same_w)
share = {"0.002": fb["0.002"]["arbiters"]["A3"]["eval"]["oracle_share"],
         "0.003": fb["0.003"]["arbiters"]["A3"]["eval"]["oracle_share"],
         "0.006": fd["0.006"]["rules"]["J"]["eval"]["oracle_share"],
         "0.0035": fd["0.0035"]["rules"]["J"]["eval"]["oracle_share"]}
show("App. E: held-out share of a perfect choice, BB72 (%)", "59-61", span([100 * share[p] for p in ("0.002", "0.003")]),
     "evaluation halves (odd chunks)")
show("App. E: held-out share of a perfect choice, d = 13 (%)", "2-4", span([100 * share[p] for p in ("0.006", "0.0035")]),
     "evaluation halves (odd chunks)")
nw = [fd[p]["reference"]["always_network_on_disagreement"] for p in ("0.006", "0.0035")]
x = [r["breaks"] / r["repairs"] for r in nw]
show("App. E: d = 13, network wrong / Cascade wrong (times)", "three", "three" if all(round(v) == 3 for v in x) else
     span(x, 2), "eval disagreement rows: " + ", ".join(f"{r['breaks']:,}/{r['repairs']:,}" for r in nw),
     all(round(v) == 3 for v in x))

print("\n== Table I (rows a): the timing of the joint decision, slowest of three against Cascade alone's fastest")
T = {(c["card"], c["code"], c["p"], c["mode"]): c for c in load("timing/timing.json")["cells"]}
S1 = {(c["card"], c["code"], c["p"], c["mode"]): c for c in load("timing/stage1.json")["cells"]}
TAB = {("bb72", 0.002): ("174.9", "1.356", "195.8", "1.707", "10.08", "8.85", "9.68", "4.03"),
       ("bb72", 0.003): ("315.7", "2.334", "351.8", "2.252", "5.61", "5.14", "5.37", "3.05"),
       ("d13", 0.006): ("108.1", "2.34", "127.2", "2.15", "3.56", "4.45", "3.30", "4.23"),
       ("d13", 0.0035): ("69.4", "1.36", "91.3", "1.29", "5.54", "7.65", "4.64", "7.06")}
for (code, p), vals in TAB.items():
    nd_b = 3 if code == "bb72" else 2
    k = 0
    for card in ("RTX", "H100"):
        for mode, stat, nd in (("single", "mean", 1), ("batched", "per_round", nd_b)):
            c = T[(card, code, p, mode)]["decoders"]["joint"]
            show(f"Table I {code} p={p} {card} {mode}: time", vals[k], f"{c['value'][stat]:.{nd}f}", "timing/timing.json")
            show(f"Table I {code} p={p} {card} {mode}: speed-up", vals[4 + k],
                 f"{c['speedup'][stat]:.2f}", "Cascade alone's fastest / joint's slowest")
            k += 1

print("\n== Section IV D and Appendix E: the joint decision against the composition in the same session")
diffs = []
for key, c in T.items():
    stat = "mean" if key[3] == "single" else "per_round"
    diffs.append((c["decoders"]["joint"]["value"][stat] / c["decoders"]["comp"]["value"][stat] - 1, key))
diffs.sort()
worst, wkey = diffs[-1]
show("Sec. IV D: at most 2.4 % longer in 15 of the 16 (%)", "2.4", pct(diffs[-2][0], 1),
     f"the 15 range from {pct(diffs[0][0], 1)} to {pct(diffs[-2][0], 1)} %")
show("App. E: as long within 2.4 % in 15 of the 16 (%)", "2.4", pct(max(abs(d) for d, _ in diffs[:-1]), 1),
     "largest absolute difference of the 15")
show("App. E: the 16th timing", "RTX bb72 0.003 single", " ".join(map(str, (wkey[0], wkey[1], wkey[2], wkey[3]))), "")
show("App. E: the 16th took longer (%)", "6.7", pct(worst, 1), "timing/timing.json")


def rows(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(J, pattern))):
        r = [json.loads(x) for x in open(f) if x.strip()][-1]
        out.append((f, r))
    return out


slot = "timing/records/RTX/out/bb_rtx_AJ52/slots/r1.rep*.dfm.p0.003.c1.single.base{}/a0/rows.jsonl"
mhz = lambda r: max(r["clock"]["hist"].items(), key=lambda kv: kv[1])[0]
cj = {mhz(r) for _, r in rows(slot.format(".joint"))}
cc = [int(mhz(r)) for _, r in rows(slot.format(""))]
low = 1 - int(min(cj)) / max(cc)
show("App. E: the 16th's joint processes: lower SM clock (%)", "5", pct(low), f"joint {sorted(cj)} MHz, composition {cc}")
order = {}
for f, r in rows("timing/records/RTX/out/order_rtx_AJ56/*/slots/*/a0/rows.jsonl"):
    order.setdefault((r["p"], "joint" if ".joint/" in f else "comp"), []).append((r["latency_us"]["mean"], mhz(r)))
for p, paper in ((0.002, "0.1-0.3"), (0.003, "0.2")):
    comp = order[(p, "comp")][0][0]
    inc = [100 * (m / comp - 1) for m, _ in order[(p, "joint")]]
    clocks = {c for _, c in order[(p, "joint")] + order[(p, "comp")]}
    each, mean = span(inc, 1), f"{sum(inc) / len(inc):.1f}"
    show(f"App. E, Sec. IV D: equal-clock run, p = {p} (% longer)", paper, each if paper == each else mean,
         f"order test, all at {sorted(clocks)} MHz; the two joint processes " + " and ".join(f"{v:.3f}" for v in inc) + " %",
         paper in (each, mean) and len(clocks) == 1,
         "" if paper == each else f"the paper rounds the mean of the two joint processes ({mean} %); each rounds to {each} %")
up = []
for p in (0.002, 0.003):
    now = T[("H100", "bb72", p, "single")]["decoders"]["alone"]["value"]["mean"]
    then = S1[("H100", "bb72", p, "single")]["decoders"]["alone"]["value"]["mean"]
    up.append(100 * (now / then - 1))
show("App. E: H100 Cascade alone, single BB72, longer (%)", "27-28", span(up), "timing.json against stage1.json")
exceed = all(T[("H100", "bb72", p, "single")]["decoders"]["joint"]["speedup"]["mean"]
             > S1[("H100", "bb72", p, "single")]["decoders"]["comp"]["speedup"]["mean"] for p in (0.002, 0.003))
show("App. E: H100 BB72 single speed-ups exceed Table I's", "exceed", "exceed" if exceed else "do not",
     "joint (this session) against the composition (stage 1)", exceed)

print("\n== Discussion: 'Training the network for its fallback' and 'Better scores for the joint decision' (exploratory)")
orc = [fd[p]["counts"]["all"] for p in ("0.006", "0.0035")]
x = [1 - a["oracle_fail"] / a["composition"] for a in orc]
xe = [fd[p]["reference"]["routed_oracle"]["rel_reduction"] for p in ("0.006", "0.0035")]
show("Discussion: d = 13, a perfect choice removes (%)", "30", span([100 * v for v in x]),
     f"whole fresh reads, {span([100 * v for v in x], 1)} %",
     note=f"on the whole fresh reads ({', '.join(f'{100 * v:.1f}' for v in x)} % at p = 0.6 and 0.35 %); "
          f"on their evaluation halves {', '.join(f'{100 * v:.1f}' for v in xe)} %")
pm = {D["d"]: {r["q"]: r for r in D["rows"]} for D in load("soft_output_pm/out/arb5_results.json")["distances"]}
red = [pm[d][0.02]["corr"]["vs_registered"]["rel"] for d in (11, 13, 15, 17)]
show("Discussion: q = 2 %, fewer failures than Table III's row (%)", "5-18", span([100 * v for v in red]),
     "d = 11-17: " + ", ".join(f"{100 * v:.1f}" for v in red))
dis = [pm[d][0.02]["eval"]["disagree"] / pm[d][0.02]["eval"]["routed"] for d in (11, 13, 15, 17)]
show("Discussion: the second decode runs on (% of routed)", "17-27", span([100 * v for v in dis]),
     "d = 11-17: " + ", ".join(f"{100 * v:.1f}" for v in dis))

for n in NOTES:
    print(f"\nnote: {n}")
print(f"\n{'all numbers match the paper' if not BAD else 'MISMATCH: ' + '; '.join(BAD)}")
sys.exit(1 if BAD else 0)
