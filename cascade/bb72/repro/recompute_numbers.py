#!/usr/bin/env python3
"""Recompute the numbers that the paper reports from the Cascade comparison, from the records in this tree.

No GPU and no shots: the inputs are the committed counts and timing rows. The released code does the arithmetic
(pc_report.headline, paired_counts and pipes, pc_common.holm and point, d1_confirm.speed, m2.py report,
noise_fit/table.py and the curve sets' bb/cascade/curves.py); this script feeds it the records and, for each number,
puts side by side the value recomputed here, the value that the records state and the value printed in the paper
(convlines_paper main.tex at d477757).
Run from anywhere, in the release's Python environment (README, "Reproducing the Cascade comparison"):

    python cascade/repro/recompute_numbers.py --out /tmp/cascade_numbers

It writes numbers.json and numbers.md to --out and exits 1 if a recomputed value differs from its record. A printed
value of the paper that the recomputed value does not give at the paper's precision is listed at the end; it does not
change the exit status."""
import argparse
import contextlib
import copy
import csv
import glob
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CAS = os.path.dirname(HERE)
CC = os.path.join(CAS, "cascade_compare")
PC, PD, BO = (os.path.join(CC, d) for d in ("phase_c", "phase_d", "batchopt"))
NF = os.path.join(CAS, "noise_fit")
sys.path[:0] = [os.path.join(CAS, "ising_compare"), os.path.join(CAS, "qldpc_neural"), PC, PD, BO, NF]

import pc_common as pc  # noqa: E402

pc.W4 = PC   # phase C's run directory on the cluster; in this tree its records are under phase_c/ (out/, out_h100/)
import pc_report  # noqa: E402

R = pc.R     # syndrome rounds per BB72 shot: "per round" is per shot / R
MISSING = object()
ROWS = []


def load(path):
    with open(path) as fh:
        return json.load(fh)


def jl(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def rel(path):
    return os.path.relpath(path, os.path.dirname(CAS))


def close(a, b):
    """Equal, floats to 1e-9 relative (the same arithmetic on the same inputs)."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def differences(a, b, path="", out=None):
    """The paths at which two JSON values differ (at most 10)."""
    out = [] if out is None else out
    if len(out) >= 10:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'the record' if k in b else 'the recomputation'}")
            else:
                differences(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, f"{path}[{i}]", out)
    elif not close(a, b):
        out.append(f"{path}: {str(a)[:60]} against {str(b)[:60]}")
    return out


def show(fmt, x):
    try:
        return fmt(x)
    except (TypeError, ValueError):
        return str(x)


def check(sec, what, got, record=MISSING, paper=None, fmt=str, src="", paper_ok=None):
    """One number: the recomputed value, its record (if the records state it) and the paper's printed value (if any)."""
    r = {"section": sec, "what": what, "value": show(fmt, got), "source": src}
    if record is not MISSING:
        r["record"] = show(fmt, record)
        r["record_ok"] = bool(close(got, record))
    if paper is not None:
        r["paper"] = paper
        r["paper_ok"] = bool(r["value"] == paper) if paper_ok is None else bool(paper_ok)
    ROWS.append(r)
    return got


def f0(x):
    return f"{x:,.0f}"


def f1(x):
    return f"{x:,.1f}"


def f2(x):
    return f"{x:,.2f}"


def f3(x):
    return f"{x:,.3f}"


def e2(x):
    return f"{x:.2e}"


def pct1(x):
    return f"{100 * x:.1f}"


def rng(fmt, sep="–"):
    return lambda t: f"{fmt(min(t))}{sep}{fmt(max(t))}"


def flag(fmt, flagged):
    return lambda x: fmt(x) + ("^c" if flagged else "")


# ----------------------------------------------------------------------------------------------------------- phase C
PCO = os.path.join(PC, "out")
SUM_F = os.path.join(PCO, "report", "phase_c_summary.json")


def phase_c():
    S = "phase C accuracy"
    SUM = load(SUM_F)
    CONF = {f: load(os.path.join(PCO, f"confirm_{f}.json")) for f in pc.SETTINGS}
    src_c = rel(os.path.join(PCO, "confirm_<setting>.json"))
    # every paired result's statistics from its counts alone
    members = [r["stats"] for f in pc.SETTINGS for r in SUM["curves"][f]["vs_cascade"]] + \
              [r["vs_B_95"] for f in pc.SETTINGS for r in CONF[f]["rows"]]
    bad = 0
    for s in members:
        t = pc_report.paired_counts(s, s.get("alpha", pc.ALPHA))
        bad += any(not close(t[k], s[k]) for k in ("mcnemar_p", "diff_ler", "diff_lo", "diff_hi", "verdict") if k in s)
    check(S, f"paired results whose McNemar p, Newcombe interval or verdict at 0.05 differ when recomputed from their "
             f"counts (of {len(members)})", bad, 0, fmt=f0, src=rel(SUM_F) + ", " + src_c)
    same = all(close(a["vs_B_95"], b["vs_B_95"]) and a["fail"] == b["fail"]
               for f in pc.SETTINGS for a, b in zip(CONF[f]["rows"],
                                                    next(c for c in SUM["confirm"] if c["family"] == f)["rows"]))
    check(S, "confirmation counts in the confirmation files and in the report", same, True, src=src_c)
    # one Holm procedure over the headline family of both settings
    comps = [copy.deepcopy(SUM["curves"][f]["vs_cascade"]) for f in pc.SETTINGS]
    conf = [copy.deepcopy(CONF[f]) for f in pc.SETTINGS]
    m = pc_report.headline(comps, conf)
    check(S, "comparisons under one Holm procedure at 0.05", m, SUM["m_headline"], "132", f0, rel(SUM_F))
    check(S, "of which development-curve comparisons and confirmation rows",
          (sum(r["ours"] for c in comps for r in c), sum(len(c["rows"]) for c in conf)),
          fmt=lambda t: f"{t[0]} + {t[1]}", src=rel(SUM_F))
    diff = 0
    for f, c in zip(pc.SETTINGS, comps):
        for r, r0 in zip(c, SUM["curves"][f]["vs_cascade"]):
            if r["ours"]:
                diff += r["verdict"] != r0.get("verdict") or not close(r["holm_alpha"], r0.get("holm_alpha"))
    for cf in conf:
        c0 = next(c for c in SUM["confirm"] if c["family"] == cf["family"])
        for r, r0 in zip(cf["rows"], c0["rows"]):
            diff += r["verdict"] != r0.get("verdict") or not close(r["holm_alpha"], r0.get("holm_alpha"))
    check(S, "Holm verdicts and levels that differ from the report's (of 132)", diff, 0, fmt=f0, src=rel(SUM_F))

    def crow(f, p, typ):
        return next(r for c in conf if c["family"] == f for r in c["rows"] if r["p"] == p and r["type"] == typ
                    and r["S"] in pc.SMALL[f])

    fam = "depspam_fm"
    paper = ((0.002, "5,000,000", "3,858", ("1", "3,896", "level"), ("2, 0.1", "3,596", "better")),
             (0.003, "1,000,000", "4,459", ("5", "4,550", "worse"), ("5, 0.25", "3,957", "better")))
    for p, shots, cas, pair, chain in paper:
        pr, ch = crow(fam, p, "S-B"), crow(fam, p, "S-B-R")
        at = f"fitted, p = {100 * p:g} %"
        check(S, f"{at}: confirmation shots", pr["shots"], pr["vs_B_95"]["shots"], shots, f0, src_c)
        check(S, f"{at}: Cascade alone, failures", pr["B_alone"]["fail"], pr["vs_B_95"]["fail_b"], cas, f0, src_c)
        check(S, f"{at}: pair, routed (%)", 100 * pr["q1"], paper=pair[0], fmt=lambda x: f"{x:g}", src=src_c)
        check(S, f"{at}: pair, failures", pr["fail"], pr["vs_B_95"]["fail_a"], pair[1], f0, src_c)
        check(S, f"{at}: pair, verdict (Holm)", pr["verdict"], paper=pair[2], src=src_c)
        check(S, f"{at}: chain, routed to Cascade, then to Relay-BP (%)", (100 * ch["q1"], 100 * ch["r"]),
              paper=chain[0], fmt=lambda t: f"{t[0]:g}, {t[1]:g}", src=src_c)
        check(S, f"{at}: chain, failures", ch["fail"], ch["vs_B_95"]["fail_a"], chain[1], f0, src_c)
        check(S, f"{at}: chain, verdict (Holm)", ch["verdict"], paper=chain[2], src=src_c)
        # the plan's secondary statement: non-inferiority within 10 % of Cascade's development rate
        dev = SUM["curves"][fam]["points"][str(p)][pc.BIG[fam]]
        margin = pc.DELTA_REL * dev["fail"] / dev["shots"]
        check(S, f"{at}: pair, non-inferiority margin (10 % of Cascade's development rate)", margin, pr["ni_margin"],
              fmt=e2, src=src_c)
        a_ni = pr["vs_B_ni"]["alpha"]
        check(S, f"{at}: pair, two-sided level of the non-inferiority interval (one-sided 0.05/32)", a_ni,
              2 * pc.ALPHA / 32, fmt=lambda x: f"{x:g}", src=src_c)
        hi = pc_report.paired_counts(pr["vs_B_95"], a_ni)["diff_hi"]
        check(S, f"{at}: pair, upper end of the excess failure rate's interval", hi, pr["vs_B_ni"]["diff_hi"], fmt=e2,
              src=src_c)
        check(S, f"{at}: pair, excess below the margin", hi < margin, pr["noninferior"], "True", src=src_c)
    pr = crow(fam, 0.003, "S-B")
    check(S, "fitted, p = 0.3 %: pair, Cascade's correct decisions broken on kept shots", pr["kept_regressions_vs_B"],
          paper="155", fmt=f0, src=src_c)
    check(S, "fitted, p = 0.3 %: pair, Cascade's failures repaired on kept shots", pr["kept_repairs_vs_B"], paper="64",
          fmt=f0, src=src_c)
    check(S, "fitted, p = 0.3 %: pair minus Cascade failures = broken minus repaired (Eq. identity)",
          pr["fail"] - pr["B_alone"]["fail"], pr["kept_regressions_vs_B"] - pr["kept_repairs_vs_B"], fmt=f0, src=src_c)
    fam = "uniform"
    for p, pair, chain in ((0.002, "level", "worse"), (0.003, "worse", "better")):
        check(S, f"uniform, p = {100 * p:g} %: pair, verdict (Holm)", crow(fam, p, "S-B")["verdict"], paper=pair,
              src=src_c)
        check(S, f"uniform, p = {100 * p:g} %: chain, verdict (Holm)", crow(fam, p, "S-B-R")["verdict"], paper=chain,
              src=src_c)
    sr = [r for c in conf for r in c["rows"] if r["type"] == "S-R" and r["p"] in pc.RELAY_P]
    n = sum(r["fail"] > r["B_alone"]["fail"] for r in sr)
    check(S, "pairs with full-DEM Relay-BP as the fallback that fail more often than Cascade alone (both points, both "
             "settings)", f"{n} of {len(sr)}", paper=f"{len(sr)} of {len(sr)}", src=src_c)
    acc = {}
    for r in comps[pc.SETTINGS.index(fam)]:
        if r["ours"] and r["p"] in pc.RELAY_P:
            acc.setdefault(f"{r['net']} alone", {})[r["p"]] = r["verdict"]
    for r in conf[pc.SETTINGS.index(fam)]["rows"]:
        if r["p"] in pc.RELAY_P:
            acc.setdefault(r["decoder"], {})[r["p"]] = r["verdict"]
    check(S, f"uniform: decoders better than Cascade at both points (of {len(acc)})",
          sum(all(v.get(p) == "better" for p in pc.RELAY_P) for v in acc.values()), paper="0", fmt=f0, src=rel(SUM_F))
    # the development curves (kind 10)
    S = "phase C development curves"
    pts = SUM["curves"]["depspam_fm"]["points"]["0.003"]
    comp = {r["net"]: r["stats"] for r in SUM["curves"]["depspam_fm"]["vs_cascade"] if r["p"] == 0.003}
    for t, v in (("S55p", "4,931"), ("S55p_deep", "3,374")):
        check(S, f"fitted, p = 0.3 %: {t} alone, failures", pts[t]["fail"], comp[t]["fail_a"], v, f0, rel(SUM_F))
    check(S, "fitted, p = 0.3 %: C14p (Cascade), failures", pts["C14p"]["fail"], comp["S55p"]["fail_b"], "2,305", f0,
          rel(SUM_F))
    check(S, "fitted, p = 0.3 %: development shots", pts["C14p"]["shots"], comp["S55p"]["shots"], "500,000", f0,
          rel(SUM_F))
    for p in (0.001, 0.002, 0.003, 0.004):
        pt = SUM["curves"]["depspam_fm"]["points"][str(p)]["C14p"]
        q = pc.point(pt["fail"], pt["shots"])
        check(S, f"C14p, p = {100 * p:g} %: P_L 95 % interval (Eq. 1, Wilson)", [q["pl_lo"], q["pl_hi"]],
              [pt["pl_lo"], pt["pl_hi"]], fmt=lambda t: f"[{t[0]:.3g}, {t[1]:.3g}]", src=rel(SUM_F))
        v = pc_report.PAPER["Cascade"][p]
        where = "inside" if q["pl_lo"] <= v <= q["pl_hi"] else ("below" if v < q["pl_lo"] else "above")
        check(S, f"published Cascade P_L at p = {100 * p:g} % ({v:.3g}) against that interval", where,
              paper="inside" if p < 0.004 else "below", src="cascade_compare/phase_c/pc_report.py (PAPER)")


def phase_c_times():
    S = "phase C timing (RTX PRO 6000)"
    SUM = load(SUM_F)
    CONF = {f: load(os.path.join(PCO, f"confirm_{f}.json")) for f in pc.SETTINGS}
    rf = os.path.join(PCO, "speed", "rows.jsonl")
    rows = jl(rf)
    src = rel(rf)
    for fam in pc.SETTINGS:
        agg = pc_report.pipes(rows, fam, CONF[fam]["locked_sha256"])
        dec = {d["decoder"]: d for d in SUM["verdicts"][fam]["decoders"]}
        bad = n = 0
        for r in CONF[fam]["rows"]:
            if r["p"] not in pc.RELAY_P:
                continue
            x = next(y for y in dec[r["decoder"]]["by_p"] if y["p"] == r["p"])
            s1, bt = agg[(r["i"], r["p"], "single")], agg[(r["i"], r["p"], "batched")]
            c1, cb = agg[(-1, r["p"], "single")], agg[(-1, r["p"], "batched")]
            got = (s1["mean_us"], s1["p999_us"], bt["per_shot_us"], c1["mean_us"], c1["p999_us"], cb["per_shot_us"])
            want = (x["single_mean_us"], x["single_p999_us"], x["amortized_us"], x["cascade_single_mean_us"],
                    x["cascade_single_p999_us"], x["cascade_amortized_us"])
            bad += not close(got, want)
            n += 1
        check(S, f"{fam}: timed configurations whose slowest-process times differ from the report's (of {n})", bad, 0,
              fmt=f0, src=src + ", " + rel(SUM_F))
    fam = "depspam_fm"
    sha = CONF[fam]["locked_sha256"]
    agg = pc_report.pipes(rows, fam, sha)

    def procs(i, p, m):
        return [x for x in rows if x.get("role") == "pipeline" and x["family"] == fam and x["locked_sha256"] == sha
                and x["config"] == i and x["p"] == p and x["metric"] == m]

    def flagged(i, p, m):
        return any(not (x.get("clock") or {}).get("accepted", True) for x in procs(i, p, m))

    cfg = {(r["p"], r["type"]): r["i"] for r in CONF[fam]["rows"] if r["S"] in pc.SMALL[fam]}
    table = {0.002: {"alone": ("1,696.5", "11.98^c"), "S-B": ("105.5", "0.89"), "S-B-R": ("254.3", "1.70")},
             0.003: {"alone": ("1,700.8", "11.98"), "S-B": ("178.2", "1.38"), "S-B-R": ("882.4", "10.66")}}
    for p, t in table.items():
        for typ, (single, batched) in t.items():
            i = -1 if typ == "alone" else cfg[(p, typ)]
            name = {"alone": "Cascade alone", "S-B": "pair", "S-B-R": "chain"}[typ]
            check(S, f"fitted, p = {100 * p:g} %: {name}, mean single-shot latency (us)",
                  agg[(i, p, "single")]["mean_us"], paper=single, fmt=flag(f1, flagged(i, p, "single")), src=src)
            check(S, f"fitted, p = {100 * p:g} %: {name}, batched cost (us per round)",
                  agg[(i, p, "batched")]["per_shot_us"] / R, paper=batched, fmt=flag(f2, flagged(i, p, "batched")),
                  src=src)
    sp = [agg[(-1, p, "single")]["mean_us"] / agg[(cfg[(p, "S-B")], p, "single")]["mean_us"] for p in pc.RELAY_P]
    bp = [agg[(-1, p, "batched")]["per_shot_us"] / agg[(cfg[(p, "S-B")], p, "batched")]["per_shot_us"]
          for p in pc.RELAY_P]
    check(S, "fitted pair: single-shot speed-up over Cascade alone", sp, paper="9.5–16.1", fmt=rng(f1), src=src)
    check(S, "fitted pair: batched speed-up over Cascade alone", bp, paper="8.7–13.5", fmt=rng(f1), src=src)
    check(S, "fitted pair: higher 99.9th percentile than Cascade alone at both points",
          all(agg[(cfg[(p, "S-B")], p, "single")]["p999_us"] > agg[(-1, p, "single")]["p999_us"] for p in pc.RELAY_P),
          paper="True", src=src)
    ch = [(agg[(cfg[(p, "S-B-R")], p, m)], agg[(-1, p, m)]) for p in pc.RELAY_P for m in ("single", "batched")]
    check(S, "fitted chain: faster than Cascade alone on average (single mean and batched, both points)",
          all(a.get("mean_us", a.get("per_shot_us")) < c.get("mean_us", c.get("per_shot_us")) for a, c in ch),
          paper="True", src=src)
    i8 = cfg[(0.003, "S-B-R")]
    check(S, "fitted chain, p = 0.3 %: 99.9th-percentile single-shot latency (ms)",
          agg[(i8, 0.003, "single")]["p999_us"] / 1000, paper="129.6", fmt=f1, src=src)
    check(S, "fitted, p = 0.3 %: Cascade alone, 99.9th-percentile single-shot latency (ms)",
          agg[(-1, 0.003, "single")]["p999_us"] / 1000, paper="1.72", fmt=f2, src=src)
    # the timing protocol (Appendix, timing of the Cascade comparison)
    pr_ = [x for x in rows if x.get("role") == "pipeline"] + jl(os.path.join(PD, "out", "speed", "rows.jsonl"))
    check(S, "shots per process, single and batched mode (every phase C and D1 pipeline process)",
          (sorted({x["shots"] for x in pr_ if x["metric"] == "single"}),
           sorted({x["shots"] for x in pr_ if x["metric"] == "batched"})), paper="20,000; 500,000",
          fmt=lambda t: "; ".join(", ".join(f"{v:,}" for v in u) for u in t), src=src)
    check(S, "Relay-BP worker processes in batched mode", pc.NCPU_RELAY, paper="16", fmt=f0,
          src="cascade_compare/phase_c/pc_common.py")
    # the clock rule (Appendix, timing of the Cascade comparison)
    al = {f: [x for x in rows if x.get("role") == "pipeline" and x["family"] == f
              and x["locked_sha256"] == CONF[f]["locked_sha256"] and x["config"] == -1 and x["metric"] == "batched"
              and x["p"] == 0.002] for f in pc.SETTINGS}
    fl = [x for f in pc.SETTINGS for x in al[f] if not x["clock"]["accepted"]]
    check(S, "Cascade alone, batched, p = 0.2 %: processes below the 97 % clock rule (both settings)", len(fl),
          paper="3", fmt=f0, src=src)
    check(S, "their lowest SM clock (% of the maximum)", [x["clock"]["clock_min_ratio"] for x in fl],
          paper="96.3–96.6", fmt=rng(pct1), src=src)
    check(S, "accepted processes left (uniform, fitted)",
          tuple(sum(bool(x["clock"]["accepted"]) for x in al[f]) for f in pc.SETTINGS), paper="1, 2",
          fmt=lambda t: f"{t[0]}, {t[1]}", src=src)
    check(S, "the reported (slowest) process of each is accepted",
          all(max(al[f], key=lambda x: x["per_shot_us"])["clock"]["accepted"] for f in pc.SETTINGS), paper="True",
          src=src)
    flagged_other = [(x["family"], x["config"], x["type"], x["metric"]) for x in rows if x.get("role") == "pipeline"
                     and x["locked_sha256"] in (CONF[f]["locked_sha256"] for f in pc.SETTINGS)
                     and not (x.get("clock") or {}).get("accepted", True) and x not in fl]
    check(S, "other timed processes below the clock rule, by type and mode (the paper reports none of their times)",
          sorted({f"{t[2]} {t[3]}" for t in flagged_other} | set()), fmt=lambda t: f"{len(flagged_other)}: " +
          ", ".join(t), src=src)


# ------------------------------------------------------------------------------------------------- exploratory look
def explore():
    S = "exploratory look (not registered)"
    xf = os.path.join(PC, "explore", "explore_deep.json")
    X = load(xf)
    src = rel(xf)
    check(S, "configurations evaluated on phase C's confirmation reads", X["meta"]["looks"], paper="52", fmt=f0,
          src=src)
    cf = {(x["setting"], x["p"], x["S"], x["q1"]): x for x in X["configs"]}
    for p, q1, fail, cas, pv, pfmt in ((0.002, 0.02, "3,822", "3,858", "2.1e-07", lambda x: f"{x:.1e}"),
                                       (0.003, 0.10, "4,452", "4,459", "0.12", f2)):
        x = cf[("depspam_fm", p, "S55p_deep", q1)]
        c = x["conf"]
        at = f"six-block pair, p = {100 * p:g} %, {100 * q1:g} % routed"
        check(S, f"{at}: the first lock's pick (before the amendment)", "lock1" in x["labels"], paper="True", src=src)
        check(S, f"{at}: failures", c["fail"], paper=fail, fmt=f0, src=src)
        check(S, f"{at}: Cascade alone, failures", c["B_fail"], paper=cas, fmt=f0, src=src)
        s = pc_report.paired_counts({"shots": c["shots"], "fail_a": c["fail"], "fail_b": c["B_fail"],
                                     "only_a": c["only_pair"], "only_b": c["only_B"]}, pc.ALPHA)
        check(S, f"{at}: exact McNemar p", s["mcnemar_p"], c["mcnemar_p"], pv, pfmt, src)
    ratios = [cf[("depspam_fm", 0.003, "S55p", q)]["conf"]["kept_breaks"] /
              cf[("depspam_fm", 0.003, "S55p_deep", q)]["conf"]["kept_breaks"] for q in (0.02, 0.05, 0.10)]
    check(S, "p = 0.3 %, 2-10 % routed: kept shots broken, three-block over six-block", ratios, paper="3.5 to 11",
          fmt=lambda t: f"{min(t):.1f} to {max(t):.0f}", src=src)
    L = load(os.path.join(PD, "d1_locked_depspam_fm.json"))
    same = all(any(close((c["S"], c["q1"], c["tau"]["tau1"]), (x["S"], x["q1"], x["tau1"])) for x in X["configs"]
                   if x["setting"] == "depspam_fm" and x["p"] == c["p"]) for c in L["configs"])
    check(S, "D1's locked configurations are the look's (network, q1 and threshold)", same, True,
          src="cascade_compare/phase_d/d1_locked_depspam_fm.json")


# --------------------------------------------------------------------------------------------------------------- D1
def d1():
    S = "D1 accuracy"
    df = os.path.join(PD, "out", "confirm", "run_1", "d1_result.json")
    D = load(df)
    L = load(os.path.join(PD, "d1_locked_depspam_fm.json"))
    src = rel(df)
    pts = D["points"]
    shots = {p: pts[p]["shots"] for p in pts}
    check(S, "shots at p = 0.2 % and 0.3 %", (shots["0.002"], shots["0.003"]), paper="11,000,000 and 17,000,000",
          fmt=lambda t: f"{t[0]:,} and {t[1]:,}", src=src)
    check(S, "fresh shots in all", sum(shots.values()) / 1e6, paper="28", fmt=lambda x: f"{x:g}", src=src)
    bad = n = 0
    for p in pts:
        for m in ("eager", "single", "batched"):
            s = pts[p]["paths"][m]["stats_95"]
            t = pc_report.paired_counts(s, s["alpha"])
            bad += any(not close(t[k], s[k]) for k in ("mcnemar_p", "diff_ler", "diff_lo", "diff_hi", "verdict"))
            n += 1
    check(S, f"paired results whose statistics differ when recomputed from their counts (of {n})", bad, 0, fmt=f0,
          src=src)
    comps = []
    for cp in L["comparisons"]:
        r = pts[str(cp["p"])]
        if cp["path"] == "eager":
            pv = r["paths"]["eager"]["stats_95"]["mcnemar_p"]
        else:
            ps_, pb_ = (r["paths"][m]["stats_95"]["mcnemar_p"] for m in ("single", "batched"))
            pv = ps_ if r["compiled_coincide"] else max(ps_, pb_)     # the intersection-union test
        comps.append({**cp, "mcnemar_p": pv})
    th = pc.holm([x["mcnemar_p"] for x in comps])
    for j, x in enumerate(comps):
        r = pts[str(x["p"])]
        parts = ("eager",) if x["path"] == "eager" else ("single", "batched")
        if th[j] is None:
            v = "level"
        else:
            vs = {pc_report.paired_counts(r["paths"][m]["stats_95"], th[j])["verdict"] for m in parts}
            v = vs.pop() if len(vs) == 1 else "level"
        rec = next(y for y in D["comparisons"] if y["name"] == x["name"])
        check(S, f"{x['name']} (p = {100 * x['p']:g} %, {x['path']}): p-value, Holm level", (x["mcnemar_p"], th[j]),
              (rec["mcnemar_p"], rec["holm_alpha"]), fmt=lambda t: f"{t[0]:.3g}, {t[1]}", src=src)
        check(S, f"{x['name']} (p = {100 * x['p']:g} %, {x['path']}): verdict", v, rec["verdict"], "better", src=src)
    check(S, "comparisons under the Holm procedure", len(comps), D["family_size"], "4", f0, src)
    tab = {"0.002": {"single": ("7,853", "7,912"), "batched": ("7,858", "7,918")},
           "0.003": {"single": ("76,409", "76,676"), "batched": ("76,445", "76,711")}}
    for p, t in tab.items():
        for m, (dv, cv) in t.items():
            x = pts[p]["paths"][m]
            at = f"p = {100 * float(p):g} %, compiled {m} path"
            check(S, f"{at}: pair, failures", x["D"]["fail"], x["stats_95"]["fail_a"], dv, f0, src)
            check(S, f"{at}: Cascade alone, failures", x["C"]["fail"], x["stats_95"]["fail_b"], cv, f0, src)
    for p, d, iv, sc in (("0.002", "5.36e-06", "[3.59, 7.16]", 1e-6), ("0.003", "1.57e-05", "[1.33, 1.81]", 1e-5)):
        s = pts[p]["paths"]["single"]["stats_95"]
        at = f"p = {100 * float(p):g} %, single path"
        check(S, f"{at}: Cascade's failure rate minus the pair's", -s["diff_ler"], paper=d, fmt=e2, src=src)
        check(S, f"{at}: its 95 % interval (x {sc:g})", (-s["diff_hi"] / sc, -s["diff_lo"] / sc), paper=iv,
              fmt=lambda t: f"[{t[0]:.2f}, {t[1]:.2f}]", src=src)
    for p, about in (("0.002", 0.75), ("0.003", 0.35)):
        red = []
        for m in ("single", "batched"):
            x = pts[p]["paths"][m]
            red.append((x["C"]["fail"] - x["D"]["fail"]) / x["C"]["fail"])
            check(S, f"p = {100 * float(p):g} %, {m} path: relative reduction", red[-1], -x["diff_share_of_C"],
                  fmt=lambda v: f"{100 * v:.3f} %", src=src)
        check(S, f"p = {100 * float(p):g} %: relative reduction on both compiled paths (%)", red,
              paper=f"about {about}", fmt=rng(lambda v: f"{100 * v:.2f}"), src=src,
              paper_ok=all(abs(100 * v - about) < 0.0125 for v in red))
    for p, kept in (("0.002", "4.0"), ("0.003", "3.0")):
        x = pts[p]["paths"]["single"]
        check(S, f"p = {100 * float(p):g} %, single path: Cascade's failures on kept shots (% of its failures)",
              x["C_fail_kept"] / x["C"]["fail"], paper=kept, fmt=pct1, src=src)
        sh = [pts[p]["paths"][m]["C_fail_kept"] / pts[p]["paths"][m]["C"]["fail"]
              for m in ("eager", "single", "batched")]
        check(S, f"p = {100 * float(p):g} %: the same on the eager, single and batched paths", sh,
              fmt=rng(lambda v: f"{100 * v:.2f}"), src=src)
    sh = [pts["0.003"]["paths"][m]["C_fail_routed"] / pts["0.003"]["paths"][m]["C"]["fail"]
          for m in ("eager", "single", "batched")]
    check(S, "p = 0.3 %: Cascade's failures on routed shots (% of its failures; three paths)", sh, paper="97",
          fmt=lambda t: f"{100 * min(t):.0f}" if f"{100 * min(t):.0f}" == f"{100 * max(t):.0f}" else rng(pct1)(t),
          src=src)
    check(S, "p = 0.3 %: shots routed (%)", 100 * pts["0.003"]["q1"], paper="10", fmt=lambda x: f"{x:g}", src=src)

    S = "D1 timing (RTX PRO 6000)"
    import d1_confirm
    rf = os.path.join(PD, "out", "speed", "rows.jsonl")
    rows = jl(rf)
    sp = d1_confirm.speed(L, rows, D["lock_sha256"])
    dif = differences(sp, D["speed"])
    check(S, f"speed aggregates recomputed from the {len(rows)} process rows: differing entries", len(dif), 0, fmt=f0,
          src=rel(rf) + " -> " + src)
    for d_ in dif:
        print("  differs:", d_)
    P = sp["points"]

    def fl(p, eng, cfg_, mode):
        return any(c["accepted"] is False for c in P[p][eng]["clock"] if c["config"] == cfg_ and c["mode"] == mode)

    ci = {p: d1_common_config(L, float(p)) for p in P}
    table = {"0.002": {"alone": ("1,759.8^c", "12.00^c"), "base": ("163.2", "1.355"), "new": ("138.8^c", "0.734")},
             "0.003": {"alone": ("1,763.8^c", "12.01"), "base": ("300.5", "2.332"), "new": ("276.6", "1.710")}}
    for p, t in table.items():
        at = f"p = {100 * float(p):g} %"
        e = P[p]["base"]
        check(S, f"{at}: Cascade alone, mean single-shot latency (us)", e["alone_single"]["mean_us"],
              P[p]["new"]["alone_single"]["mean_us"], t["alone"][0], flag(f1, fl(p, "base", -1, "single")), src)
        check(S, f"{at}: Cascade alone, batched (us per round)", e["alone_batched"]["per_shot_us"] / R,
              P[p]["new"]["alone_batched"]["per_shot_us"] / R, t["alone"][1], flag(f2, fl(p, "base", -1, "batched")),
              src)
        for eng in ("base", "new"):
            e = P[p][eng]
            name = {"base": "pair (BB engine)", "new": "pair (multi-shot engine)"}[eng]
            check(S, f"{at}: {name}, mean single-shot latency (us)", e["D_single"]["mean_us"], paper=t[eng][0],
                  fmt=flag(f1, fl(p, eng, ci[p], "single")), src=src)
            check(S, f"{at}: {name}, batched (us per round)", e["D_batched"]["per_shot_us"] / R, paper=t[eng][1],
                  fmt=flag(f3, fl(p, eng, ci[p], "batched")), src=src)
    s1 = [P[p][e]["speedup_single_mean"] for p in P for e in ("base", "new")]
    b1 = [P[p][e]["speedup_batched"] for p in P for e in ("base", "new")]
    check(S, "single-shot speed-up over Cascade alone (two engines, two points)", s1, paper="5.9–12.7", fmt=rng(f1),
          src=src)
    check(S, "batched speed-up over Cascade alone (two engines, two points)", b1, paper="5.2–16.4", fmt=rng(f1),
          src=src)
    check(S, "the lowest batched speed-up, unrounded", min(b1), fmt=lambda x: f"{x:.4f}", src=src)
    check(S, "multi-shot engine: batched speed-up at p = 0.2 % and 0.3 %",
          (P["0.002"]["new"]["speedup_batched"], P["0.003"]["new"]["speedup_batched"]), paper="16.4 and 7.0",
          fmt=lambda t: f"{t[0]:.1f} and {t[1]:.1f}", src=src)
    clean = {"single ratios involve a flagged Cascade process": all(fl(p, "base", -1, "single") for p in P),
             "batched ratios at 0.2 % involve one": fl("0.002", "base", -1, "batched"),
             "batched ratios at 0.3 % are clean": not fl("0.003", "base", -1, "batched")
             and not any(fl("0.003", e, ci["0.003"], "batched") for e in ("base", "new"))}
    for k, v in clean.items():
        check(S, k, v, paper="True", src=src)
    tail = [P[p][e]["D_single"]["p999_us"] / P[p][e]["alone_single"]["p999_us"] - 1 for p in P for e in ("base", "new")]
    check(S, "pair's 99.9th percentile above Cascade's (%)", tail, paper="3.5–7.7", fmt=rng(pct1), src=src)
    rows_fl = [r for r in rows if (r.get("clock") or {}).get("accepted") is False]
    check(S, "processes below the 97 % clock rule (of 36)", len(rows_fl), paper="8", fmt=f0, src=rel(rf))
    check(S, "their lowest SM clock (% of the maximum)", [r["clock"]["clock_min_ratio"] for r in rows_fl],
          paper="94.1–94.7", fmt=rng(pct1), src=rel(rf))
    # D1 plan 4.3: predictions from the component times (the plan's printed values and its formula)
    meas = {(p, e): (P[p][e]["D_single"]["mean_us"], P[p][e]["D_batched"]["per_shot_us"]) for p in P
            for e in ("base", "new")}
    printed = {("0.002", "base"): (166.5, 8.2), ("0.002", "new"): (136, 4.6), ("0.003", "base"): (309, 13.9),
               ("0.003", "new"): (279, 10.4)}
    s_single, s_batch, q1 = {"base": 130.9, "new": 100.5}, {"base": 6.26, "new": 2.69}, {"0.002": 0.02, "0.003": 0.10}
    formula = {(p, e): (s_single[e] + q1[p] * 1782, s_batch[e] + q1[p] * 71.6 + 0.5) for p, e in printed}
    for name, pred in (("printed in the plan", printed), ("from the plan's formula, unrounded", formula)):
        dev = [abs(meas[k][j] / pred[k][j] - 1) for k in pred for j in (0, 1)]
        check(S, f"largest deviation of D's measured times from the predictions {name} (%)", max(dev),
              paper="4.3" if name.startswith("printed") else None, fmt=pct1,
              src="cascade_compare/phase_d/PHASE_D1_PLAN.md 4.3")


def d1_common_config(L, p):
    import d1_common
    return d1_common.config_at(L, p)[0]


# --------------------------------------------------------------------------------------------- multi-shot engine, H100
def m2_regen(dirs, out):
    import argparse as ap
    import m2
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.log"), "w") as fh, contextlib.redirect_stdout(fh):
        m2.report(ap.Namespace(dirs=",".join(dirs), out=out))
    return load(os.path.join(out, "summary.json"))


def without_dirs(d):
    return {k: v for k, v in d.items() if k != "dirs"}


def multishot(out):
    S = "multi-shot engine (RTX PRO 6000)"
    base = os.path.join(BO, "out_m2")
    dirs = sorted(glob.glob(os.path.join(base, "rows", "m2_*")))
    new = m2_regen(dirs, os.path.join(out, "m2_rtx"))
    old = load(os.path.join(base, "summary.json"))
    dif = differences(without_dirs(new), without_dirs(old))
    src = rel(os.path.join(base, "summary.json"))
    check(S, f"summary rebuilt by m2.py report from the {len(dirs)} job directories: differing entries", len(dif), 0,
          fmt=f0, src=rel(os.path.join(base, "rows")) + " -> " + src)
    for d_ in dif:
        print("  differs:", d_)
    sc = new["screen"]["new"]
    rows = [r for d in dirs if os.path.exists(os.path.join(d, "screen.jsonl"))
            for r in jl(os.path.join(d, "screen.jsonl")) if "batch" in r and "sets" in r and r["cfg"] == "new"
            and not r.get("control")]
    check(S, "screen: shot evaluations", sc["rows"], paper="88,750,000", fmt=f0, src=src)
    check(S, "screen: decisions and FP32 margin bit patterns differing from the BB engine",
          (sc["differ"], sc["margin_bits_differ"]), paper="0 and 0", fmt=lambda t: f"{t[0]} and {t[1]}", src=src)
    check(S, "screen: networks, batch sizes, smallest and largest batch",
          (len({r["net"] for r in rows}), len({r["batch"] for r in rows}), min(r["batch"] for r in rows),
           max(r["batch"] for r in rows)), paper="5, 14, 1, 4,100",
          fmt=lambda t: f"{t[0]}, {t[1]}, {t[2]:,}, {t[3]:,}", src=src)
    nets = new["nets"]

    def h2h(k):
        return nets[k]["h2h"]

    s55, s55b, deep = h2h("S55|new|all|fused@4096"), h2h("S55|base|all|fused@4096"), h2h("S55p_deep|new|all|fused@4096")
    check(S, "three-block network at 4,096, host to host (us per shot)", s55["value"], paper="1.883", fmt=f3, src=src)
    check(S, "the same, per round", s55["value"] / R, paper="0.314", fmt=f3, src=src)
    check(S, "BB engine over multi-shot engine, same batches, same job", s55b["value"] / s55["value"], paper="2.20",
          fmt=f2, src=src)
    check(S, "six-block network at 4,096 (us per round; flagged by the clock rule)",
          (deep["value"] / R, deep["flagged"]), paper="0.448, True", fmt=lambda t: f"{t[0]:.3f}, {t[1]}", src=src)
    c14 = [h2h(k) for k in ("C14|cascade|all|compiled@4096", "C14p|cascade|all|compiled@4096")]
    check(S, "Cascade alone at 4,096 (us per round; C14, C14p)", [c["value"] / R for c in c14], paper="11.96",
          fmt=lambda t: f2(t[0]) if f2(t[0]) == f2(t[1]) else rng(f2)(t), src=src)
    r = deep["value"] / c14[1]["value"]
    check(S, "six-block network over Cascade alone, per shot at 4,096 (C14p)", r, paper="0.04", fmt=f2, src=src)
    est = [1 / (0.04 + q) for q in (0.02, 0.10)]
    check(S, "Eq. (speedup) with kappa = 1 and 0.04, at 2 % and 10 % routed", est, paper="about 17 and 7",
          fmt=lambda t: f"{t[0]:.1f} and {t[1]:.1f}", src="paper, Eq. (speedup)",
          paper_ok=[round(x) for x in est] == [17, 7])
    pipe = new["pipe"]
    flagged_ratio = []
    for p, cfg, v, rb, rc in ((0.002, 2, "0.510", "1.74", "23.5"), (0.003, 7, "0.999", "1.38", "12.0")):
        for fam in ("depspam_fm", "uniform"):         # the paper's values are the fitted setting's (Table, RQ1)
            k = f"{fam}|{p}|{cfg}|S-B|%s|0|all"
            n_, b_ = pipe[k % "new"], pipe[k % "base"]
            at = f"{fam}, p = {100 * p:g} %"
            fit = fam == "depspam_fm"
            check(S, f"{at}: phase C pair on the multi-shot engine (us per round)", n_["batched"]["value"] / R,
                  paper=v if fit else None, fmt=flag(f3, n_["batched"]["flagged"]), src=src)
            check(S, f"{at}: BB engine over multi-shot engine", b_["batched"]["value"] / n_["batched"]["value"],
                  paper=rb if fit else None, fmt=f2, src=src)
            check(S, f"{at}: Cascade alone over the pair", n_["cascade_batched"]["value"] / n_["batched"]["value"],
                  paper=rc if fit else None, fmt=f1, src=src)
            if n_["cascade_batched"]["flagged"] or n_["batched"]["flagged"]:
                flagged_ratio.append(f"{fam} {100 * p:g} %")
    check(S, "phase C pair ratios that involve a process below the clock rule", ", ".join(flagged_ratio) or "none",
          paper="depspam_fm 0.2 %", src=src)

    S = "H100 (secondary)"
    hd = os.path.join(BO, "out_h100", "20261011_204455")
    hn = m2_regen([hd], os.path.join(out, "m2_h100"))
    dif = differences(without_dirs(hn), without_dirs(load(os.path.join(hd, "summary.json"))))
    src = rel(os.path.join(hd, "summary.json"))
    check(S, "summary rebuilt by m2.py report from the session's rows: differing entries", len(dif), 0, fmt=f0,
          src=rel(hd))
    for d_ in dif:
        print("  differs:", d_)
    hp = hn["pipe"]
    alone = [hp[f"{f}|{p}|-1|alone|None|0|all"]["batched"] for f in pc.SETTINGS for p in pc.RELAY_P]
    check(S, "Cascade alone, batched (us per shot)", [a["value"] for a in alone], paper="about 40", fmt=rng(f1),
          src=src, paper_ok=all(39.5 <= a["value"] < 41.5 for a in alone))
    check(S, "three-block network at 4,096: H100 against RTX PRO 6000 (us per shot)",
          (hn["nets"]["S55|new|all|fused@4096"]["h2h"]["value"], s55["value"]), paper="2.82 against 1.88",
          fmt=lambda t: f"{t[0]:.2f} against {t[1]:.2f}", src=src)
    ratio = [hp[f"{f}|{p}|{c}|S-B|new|0|all"]["cascade_batched"]["value"] /
             hp[f"{f}|{p}|{c}|S-B|new|0|all"]["batched"]["value"]
             for f in pc.SETTINGS for p, c in ((0.002, 2), (0.003, 7))]
    check(S, "phase C pair (multi-shot engine): batched speed-up over Cascade alone", ratio, paper="7.6–11.4",
          fmt=rng(f1), src=src)
    slower = [hp[f"{f}|0.003|8|S-B-R|{e}|0|all"]["batched"]["value"] >
              hp[f"{f}|0.003|8|S-B-R|{e}|0|all"]["cascade_batched"]["value"]
              for f in pc.SETTINGS for e in ("base", "new")]
    check(S, "phase C chains at p = 0.3 % slower than Cascade alone in batches (both settings, both engines)",
          f"{sum(slower)} of {len(slower)}", paper=f"{len(slower)} of {len(slower)}", src=src)
    vals = [v for k, e in hp.items() if k.endswith("|0|all") and ("|S-B|" in k or "|S-B-R|" in k or "|alone|" in k)
            for m, v in e.items() if m in ("batched", "cascade_batched") and v]
    check(S, "batched values of the pairs and of Cascade alone that involve a flagged process",
          f"{sum(bool(v['flagged']) for v in vals)} of {len(vals)}", paper=f"{len(vals)} of {len(vals)}", src=src)
    cs = os.path.join(hd, "env", "clock_samples.csv")
    with open(cs) as fh:
        rd = csv.reader(fh, skipinitialspace=True)
        head = next(rd)
        ir, ip = head.index("clocks_event_reasons.active"), head.index("enforced.power.limit [W]")
        reasons, limits = set(), set()
        for row in rd:
            if len(row) > ir:
                reasons.add(int(row[ir], 16))
                limits.add(row[ip])
    bits = sorted({b for r in reasons for b in range(16) if r >> b & 1})
    names = {0: "idle", 2: "SW power cap"}
    check(S, "throttle reasons in the clock samples, and the enforced power limit",
          (", ".join(names.get(b, f"bit {b}") for b in bits), ", ".join(sorted(limits))),
          paper="idle, SW power cap; 700.00 W", fmt=lambda t: f"{t[0]}; {t[1]}", src=rel(cs))


# -------------------------------------------------------------------------------------------------- noise-model fit
def noise_fit():
    S = "noise-model fit"
    import table
    paths = sorted(glob.glob(os.path.join(NF, "runs", "mac", "*.jsonl"))) + \
        sorted(glob.glob(os.path.join(NF, "runs", "alpine", "*.jsonl")))
    rows, heads = table.load(paths)
    summ = table.summarize(rows, heads)
    cf = os.path.join(NF, "cascade_cal_counts.json")
    old = load(cf)["summary"]
    keys = ("point", "model", "p", "dec", "shots", "failures", "p_block", "p_block_ci", "p_l", "p_l_ci", "ratio")
    bad = sum(not close({k: a[k] for k in keys}, {k: b[k] for k in keys}) for a, b in zip(summ, old)) + \
        abs(len(summ) - len(old))
    check(S, f"(point, decoder) rows of the fit's table that differ when rebuilt from the {len(paths)} run files",
          bad, 0, fmt=f0, src=rel(cf))
    fit = [r for r in summ if r["model"] == "depspam_fm"]
    check(S, "fitted setting: Tesseract failures at p = 0.2 % and 0.3 %",
          [r["failures"] for r in fit if r["dec"] == "tesseract"], paper="17–20", fmt=rng(f0), src=rel(cf))
    inside = [r["p_l_ci"][0] <= r["paper_p_l"] <= r["p_l_ci"][1] for r in fit if r["dec"] in
              ("tesseract", "bposd", "relay_full")]
    check(S, "fitted setting: published Tesseract, BP-OSD and Relay-BP rates inside our 95 % intervals",
          f"{sum(inside)} of {len(inside)}", paper=f"{len(inside)} of {len(inside)}", src=rel(cf))



# ------------------------------------------------------------------------------------------------------- curve sets
def curve_sets():
    S = "curve sets (exploratory)"
    from bb.cascade import curves as cv
    cf = os.path.join(CAS, "ising_compare", "bb", "cascade", "curves_counts.json")
    rec = load(cf)
    taus = load(os.path.join(CAS, "ising_compare", "data", "bb", "cascade", "taus.json"))
    check(S, "taus in the counts that differ from data/bb/cascade/taus.json", len(differences(rec["taus"], taus)), 0,
          fmt=f0, src=rel(cf))
    bad = ident = n = 0
    for pp, pt in rec["points"].items():                 # each (p, decoder) entry again from its failures and shots
        new = {}
        for k, v in pt.items():
            x = cv.point(v["failures"], v["shots"])
            x.update({e: v[e] for e in v if e not in x and not e.startswith("ratio_to_")})
            if "routed" in v:
                x["routed_fraction"] = v["routed"] / v["shots"]
                ident += v["failures"] != pt["r12"]["failures"] - v["r12_fail_in_routed"] + v["relay_fail_in_routed"]
            new[k] = x
        cv._ratios(new, float(pp))
        bad += sum(not close(new[k], pt[k]) for k in pt)
        n += len(pt)
    check(S, f"(p, decoder) entries of the counts that differ when recomputed from their failures and shots "
             f"(Eq. (1), Wilson intervals, ratios to the published rates; {n} entries)", bad, 0, fmt=f0, src=rel(cf))
    check(S, "routed entries whose failures are not the network's minus its routed failures plus Relay-BP's",
          ident, 0, fmt=f0, src=rel(cf))
    reg = {str(pp): cv._ratios({k: cv.point(*c) for k, c in cs.items()}, pp) for pp, cs in cv.REGISTERED.items()}
    check(S, "registered cells' entries that differ when recomputed", len(differences(reg, rec["registered"])), 0,
          fmt=f0, src=rel(cf))
    q = rec["points"]["0.1"]
    r12, hy = q["r12"], q["r12+relay q=0.0025"]
    check(S, "p = 0.1 %: shots", r12["shots"], paper="10,000,000", fmt=f0, src=rel(cf))
    check(S, "p = 0.1 %, q = 0.25 %: share of the shots routed to full-DEM Relay-BP, %", hy["routed"] / hy["shots"],
          hy["routed_fraction"], paper="0.29", fmt=lambda x: f"{100 * x:.2f}", src=rel(cf))
    check(S, "p = 0.1 %, q = 0.25 %: the network's failures on the routed shots, of all its failures",
          f"{hy['r12_fail_in_routed']} of {r12['failures']}", paper="344 of 348", src=rel(cf))
    check(S, "p = 0.1 %, q = 0.25 %: full-DEM Relay-BP's failures on the routed shots", hy["relay_fail_in_routed"],
          paper="469", fmt=f0, src=rel(cf))
    check(S, "p = 0.1 %, q = 0.25 %: failures after routing", r12["failures"] - hy["r12_fail_in_routed"] +
          hy["relay_fail_in_routed"], hy["failures"], paper="473", fmt=f0, src=rel(cf))

# ----------------------------------------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for numbers.json, numbers.md and the rebuilt summaries")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    errors = []
    for fn in (phase_c, phase_c_times, explore, d1, lambda: multishot(a.out), noise_fit, curve_sets):
        try:
            fn()
        except Exception:                                          # report every section, then fail
            errors.append(traceback.format_exc())
            print(errors[-1], file=sys.stderr)
    rec_bad = [r for r in ROWS if r.get("record_ok") is False]
    pap_bad = [r for r in ROWS if r.get("paper_ok") is False]
    md = ["| section | quantity | recomputed | record | paper | |", "|---|---|---|---|---|---|"]
    for r in ROWS:
        st = "RECORD DIFFERS" if r.get("record_ok") is False else ("paper differs" if r.get("paper_ok") is False
                                                                     else "ok")
        md.append(f"| {r['section']} | {r['what']} | {r['value']} | {r.get('record', '')} | {r.get('paper', '')} | "
                  f"{st} |")
    md += ["", f"{len(ROWS)} numbers; {sum('record' in r for r in ROWS)} compared with their records, "
               f"{len(rec_bad)} differing; {sum('paper' in r for r in ROWS)} compared with the paper, {len(pap_bad)} "
               f"not as printed; {len(errors)} sections failed to run."]
    with open(os.path.join(a.out, "numbers.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "numbers.json"), "w") as fh:
        json.dump({"rows": ROWS, "errors": errors}, fh, indent=1, default=str)
    for r in pap_bad:
        print(f"paper differs: {r['section']}: {r['what']}: recomputed {r['value']}, paper {r['paper']}")
    for r in rec_bad:
        print(f"RECORD DIFFERS: {r['section']}: {r['what']}: recomputed {r['value']}, record {r['record']}")
    print(md[-1])
    sys.exit(1 if rec_bad or errors else 0)


if __name__ == "__main__":
    main()
