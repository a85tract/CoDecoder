"""The surface rows of the surface stage's results table (the stage's section, "Verdicts and wording"): every surface speed
number the paper prints (inventory rows S1-S11 and E1-E10), old -> new, and every surface speed comparison with its verdict,
plus the Grace fit's rows (F1). Two network sets (D3 (A)), each against its own old values:
  sv2       SURF-v2's six networks: the revision's surface rows (SURF-v2's frozen rule: the paper's surface rows are the
            new networks' pairs); old = SURF-v2's registered values (surface_v2/surf_v2/analysis/analysis.json)
  deployed  the six deployed networks (d = 17: R17's q = 8 %): the rows of the paper as it stands; old = its printed
            values
The baselines are shared by both sets; old = the paper's printed values (BP-OSD at d = 9-15: the registered records,
surface_v2/results). Bases: grace (registered: every CPU time of a pod x the pod's fitted factor of its decoder, speed_rt/
grace_fit.py; pod S-low d = 7-11, pod S-high d = 13-17) and xeon (the measured-CPU disclosure, D4: the Linux workstation's own Ryzen times,
unconverted; it decides nothing).
A comparison holds when ours is strictly below the baseline on the basis named; a side that is not established
withholds it (holds = None, with the reason). Printed ranges of speed-ups round down, of time fractions up (the
direction that weakens the claim). Pure functions: no clock, no console output."""
import math

DS = (7, 9, 11, 13, 15, 17)
BASE = (("pm", "PM"), ("corr_pm", "correlated PM"), ("ising_M1_pm", "Ising-Fast + PM"),
        ("ising_M4_pm", "Ising-Accurate + PM"), ("bposd_minllr", "BP-OSD (min-LLR)"), ("bposd_bp60", "BP-OSD (BP 60)"))
CTRL = (("ising_M1_corr_pm", "Ising-Fast + correlated PM"), ("ising_M4_corr_pm", "Ising-Accurate + correlated PM"))
SETS = (("sv2", "SURF-v2's networks (the revision's rows)"), ("deployed", "deployed networks (the paper's rows)"))

# The old values. deployed and the baselines: the paper as printed (inventory rows S1-S11); sv2: SURF-v2's registered
# analysis; controls at d = 7-15 and BP-OSD at d = 9-15: the registered records (results/controls_speed.json,
# s2_composite.json), which the paper prints only as ranges.
OLD = {
    "deployed": dict(ours=[12.924, 30.049, 43.925, 57.957, 86.080, 135.019], a=[8.4, 27.9, 40.6, 49.2, 50.9, 66.7],
                     b=[2.2, 1.0, 2.2, 7.7, 33.9, 66.8], c=[2.2, 1.1, 1.1, 1.1, 1.3, 1.5],
                     routed_pct=[4.05, 1.03, 1.02, 2.00, 5.91, 7.88], cpu_share_pct=[17, 3, 5, 13, 39, 49]),
    "sv2": dict(ours=[11.793, 29.049, 41.644, 49.910, 56.544, 76.766], a=[8.466, 27.949, 40.544, 48.810, 49.558, 66.990],
                b=[1.111, 0.0, 0.0, 0.0, 5.886, 8.676], c=[2.216, 1.1, 1.1, 1.1, 1.1, 1.1],
                routed_pct=[2.0009, 0.0, 0.0, 0.0, 0.9588, 1.0004], cpu_share_pct=[9.4, 0.0, 0.0, 0.0, 10.4, 11.3]),
    "base": dict(pm=[17.16, 36.99, 72.02, 129.61, 208.53, 325.23], corr_pm=[35.71, 67.59, 155.61, 279.63, 464.95, 714.05],
                 ising_M1_pm=[48.756, 56.401, 68.633, 90.262, 122.846, 161.330],
                 ising_M4_pm=[61.410, 68.083, 78.666, 98.342, 129.631, 165.970],
                 bposd_minllr=[2864.20, 9353.56, 25202.85, 68177.06, None, 444500.0],
                 bposd_bp60=[5668.34, 18211.25, 46430.19, 115302.02, None, 619200.0],
                 ising_M1_corr_pm=[53.34, 68.45, 97.05, 144.02, 215.90, 297.508],
                 ising_M4_corr_pm=[65.07, 78.23, 100.97, 139.84, 203.29, 275.794]),
    "s2_ratio": [2.76, 2.25, 3.54, 4.82, 5.40, 5.29], "s4_fraction": [0.753, 0.812, 0.640, 0.642, 0.701, 0.837],
    "s7": {"7": ("PM", 4.2), "9": ("PM", 6.9), "11": ("Ising-Fast + PM", 24.7), "13": ("Ising-Fast + PM", 32.3),
           "15": ("Ising-Fast + PM", 36.8), "17": ("Ising-Fast + PM", 26.3)},
    "s8": dict(ours=148.7, corr_pm=860.7, ising_M1_pm=170.5, margin=21.7, b=80.5, band=(712.0, 840.0)),
    "s9": {"7": (11.69, 6.68), "13": (53.50, 49.47), "15": (66.63, 81.60), "17": (104.15, 124.58)},
    "s11": (1.2, 1.6), "e1": 8.44, "e2": (17.2, 17.16), "e8": (14.89, 16.27),
}
GRACE = ("Grace basis (an estimate, no Grace hardware): ours = H100 GPU term + handoff + routed correlated PM timed on one core "
         "of the Linux workstation's Ryzen and converted by the fitted factor; baselines = the Linux workstation's Ryzen times of PM and correlated PM "
         "converted by the fitted factors, Ising = its TensorRT plan on the H100 + its PM share converted, BP-OSD = CUDA-Q "
         "on the H100 (pod host CPU), unconverted; one pod per distance (S-low d = 7-11, S-high d = 13-17), each comparison "
         "with both sides from its pod")
XEON_SENS = ("sensitivity check (STAGE2_common C7): the 48 comparisons with every CPU part as timed on one core of the Linux workstation's Ryzen, unconverted "
             "(ours: GPU term + handoff + routed correlated PM; baselines: PM and Ising's PM shares on the Linux workstation's Ryzen; BP-OSD as on the Grace basis); "
             "reported beside the Grace-basis verdicts, which decide")
XEON = ("measured CPU (D4): every CPU part as timed on one core of the Linux workstation's Ryzen (S-low d = 7-11, S-high d = 13-17), "
        "unconverted; old = the Apple M4 basis; decides nothing")


def floor_to(x, nd):
    return None if x is None else math.floor(x * 10 ** nd + 1e-9) / 10 ** nd


def ceil_to(x, nd):
    return None if x is None else math.ceil(x * 10 ** nd - 1e-9) / 10 ** nd


def _div(x, y):
    return None if x is None or y is None or y == 0 else x / y


def _sub(x, y):
    return None if x is None or y is None else x - y


def num(row, item, unit, old, new, basis, why=None, **kw):
    return dict(row=row, item=item, kind="number", unit=unit, old=old, new=new,
                why=None if new is not None else (why or "an input is not established"), basis=basis, **kw)


def cmp_(row, item, unit, ours, theirs, ours_old, theirs_old, basis, relation="below", why=None, **kw):
    ok = None if ours is None or theirs is None else bool(ours < theirs if relation == "below" else ours > theirs)
    return dict(row=row, item=item, kind="comparison", unit=unit, relation=relation, ours_old=ours_old,
                theirs_old=theirs_old, ratio_old=_div(theirs_old, ours_old), ours=ours, theirs=theirs,
                ratio=_div(theirs, ours), holds=ok, why=None if ok is not None else (why or "a side is not established"),
                basis=basis, **kw)


def label(row, item, why):
    return dict(row=row, item=item, kind="label", unit=None, old=None, new=None, why=why, basis=None)


def table(comp, parts, base, Sq, N, nets, why, fits):
    """comp, parts, base: analysis_s.composites and baselines merged over the pods; Sq, N: the pods' GPU and the CPU worker's
    quantities (merged); nets: {set: {d: network}}; why: {quantity: reason} for the quantities that are not established;
    fits: {role: grace_fit.fit's result or None} for the pods "low" and "high"."""
    T = []
    g, ep = base["grace"], base["xeon"]

    def bval(b, k, d):
        return b.get((k, d))
    # S1: the baselines (shared), then ours per set
    for k, name in BASE + CTRL:
        for i, d in enumerate(DS):
            T.append(num("S1", f"{name} at d = {d}", "us", OLD["base"][k][i], bval(g, k, d), GRACE))
    for s, sname in SETS:
        for i, d in enumerate(DS):
            n = nets[s][d]
            T.append(num("S1", f"ours, {sname}, d = {d} ({n})", "us", OLD[s]["ours"][i], comp["grace"].get(n), GRACE,
                         why=why.get(n)))
    for s, sname in SETS:
        r_old, r_new = [], []
        for i, d in enumerate(DS):
            n = nets[s][d]
            o = OLD["s2_ratio"][i] if s == "deployed" else _div(OLD["base"]["corr_pm"][i], OLD[s]["ours"][i])
            v = _div(bval(g, "corr_pm", d), comp["grace"].get(n))
            r_old.append(o)
            r_new.append(v)
            T.append(num("S2", f"correlated PM / ours, {sname}, d = {d}", "x", o, v, GRACE))
        lo = None if None in r_new else floor_to(min(r_new), 1)
        hi = None if None in r_new else floor_to(max(r_new), 1)
        T.append(num("S2", f"range 'takes LO-HI times less composite time than correlated PM', {sname}: LO", "x",
                     floor_to(min(r_old), 1), lo, GRACE, headline="H-matching-lo"))
        T.append(num("S2", f"range 'takes LO-HI times less composite time than correlated PM', {sname}: HI", "x",
                     floor_to(max(r_old), 1), hi, GRACE, headline="H-matching-hi"))
    # S3, S4, S10: the comparisons on the Grace basis: 48 speed comparisons per network set = 8 opponents at each of the six distances
    # (the six baselines and the two Ising + correlated-PM controls); the controls at d = 7-15 are the paper's S10 rows (post hoc), counted in the 48
    CK = {k for k, _ in CTRL}
    for s, sname in SETS:
        held, n_all, withheld, fr_old, fr_new = 0, 0, 0, [], []
        for i, d in enumerate(DS):
            n = nets[s][d]
            ours, ours_old = comp["grace"].get(n), OLD[s]["ours"][i]
            for k, name in BASE + CTRL:
                row = "S10" if k in CK and d != 17 else "S4"
                c = cmp_(row, f"ours vs {name}{' (control)' if row == 'S10' else ''}, {sname}, d = {d}", "us", ours, bval(g, k, d),
                         ours_old, OLD["base"][k][i], GRACE, counted_in="S4")
                T.append(c)
                n_all += 1
                held += c["holds"] is True
                withheld += c["holds"] is None
            others = [bval(g, k, d) for k, _ in BASE]
            fr_new.append(None if ours is None or None in others else ours / min(others))
            fr_old.append(OLD["s4_fraction"][i] if s == "deployed" else
                          _div(ours_old, min(x for x in (OLD["base"][k][i] for k, _ in BASE) if x is not None)))
            T.append(num("S4", f"ours / the fastest baseline, {sname}, d = {d}", "x", fr_old[-1], fr_new[-1], GRACE))
        T.append(num("S4", f"range 'ours takes LO-HI of the time of the fastest other decoder', {sname}: LO", "x",
                     ceil_to(min(fr_old), 2), None if None in fr_new else ceil_to(min(fr_new), 2), GRACE))
        T.append(num("S4", f"range 'ours takes LO-HI of the time of the fastest other decoder', {sname}: HI", "x",
                     ceil_to(max(fr_old), 2), None if None in fr_new else ceil_to(max(fr_new), 2), GRACE))
        T.append(num("S4", f"speed comparisons that hold of {n_all} ('all 48 hold'), {sname}", "count", n_all, held, GRACE))
        T.append(num("S4", f"speed comparisons withheld (a side not established), {sname}", "count", 0, withheld, GRACE))
        n17 = nets[s][17]
        T.append(cmp_("S3", f"ours vs correlated PM at d = 17, {sname}", "us", comp["grace"].get(n17),
                      bval(g, "corr_pm", 17), OLD[s]["ours"][5], OLD["base"]["corr_pm"][5], GRACE))
        # the sensitivity check (UPDATE 27 item 8): the same 48 comparisons on the raw Ryzen basis, nothing converted; descriptive
        xh, xn = 0, 0
        for i, d in enumerate(DS):
            for k, name in BASE + CTRL:
                c = cmp_("S4x", f"ours vs {name}, raw Ryzen basis, {sname}, d = {d}", "us", comp["xeon"].get(nets[s][d]), bval(ep, k, d),
                         None, None, XEON_SENS, descriptive=True)
                T.append(c)
                xn += 1
                xh += c["holds"] is True
        T.append(num("S4x", f"speed comparisons that hold of {xn} on the raw Ryzen basis (sensitivity check), {sname}", "count", None,
                     xh, XEON_SENS, descriptive=True))
    T.append(label("S5", "PM 129.6 us and correlated PM 279.6 us at d = 13 (2.2 times)",
                   "NVIDIA's own Grace values (Ising paper Tables VII and IX, p = 0.6 %): two of the Grace fit's 18 points, "
                   "not re-measured; the CPU worker's times of the same decoders and circuits are rows F1"))
    # S6 (Table V): the terms; S11: a routed shot's correlated PM over the node's per-block mean
    for s, sname in SETS:
        fr_old, fr_new, s11 = [], [], []
        for i, d in enumerate(DS):
            n = nets[s][d]
            p = parts.get(f"grace.{n}", {})
            tot = comp["grace"].get(n)
            rf = N.get(f"N.routed_fraction.{n}")
            for key, unit, o, v in (("routed share", "%", OLD[s]["routed_pct"][i], None if rf is None else 100 * rf),
                                    ("network term a", "us", OLD[s]["a"][i], p.get("a")),
                                    ("routed fallback b", "us", OLD[s]["b"][i], p.get("b")),
                                    ("handoff c", "us", OLD[s]["c"][i], p.get("c")),
                                    ("total", "us", OLD[s]["ours"][i], tot),
                                    ("CPU share b / total", "%", OLD[s]["cpu_share_pct"][i],
                                     None if p.get("b") is None or not tot else 100 * p["b"] / tot)):
                T.append(num("S6", f"{key}, {sname}, d = {d} ({n})", unit, o, v, GRACE))
            if d <= 15:
                fr_old.append(_div(OLD[s]["a"][i], OLD[s]["ours"][i]))
                fr_new.append(_div(p.get("a"), tot))
            cond, ref = N.get(f"N.cond.{n}"), N.get(f"N.ref_corr.{d}")
            if cond is not None and ref:
                s11.append(cond / ref)
            T.append(num("S11", f"a routed shot's correlated PM / its mean per shot, {sname}, d = {d}", "x", None,
                         _div(cond, ref), "one core of the Linux workstation's Ryzen (CPU worker)",
                         why="no routed shots" if N.get(f"N.term.{n}") == 0 else None))
        T.append(num("S6", f"range 'the network term takes LO-HI % up to d = 15', {sname}: LO", "%",
                     math.floor(100 * min(fr_old)), None if None in fr_new else math.floor(100 * min(fr_new)), GRACE))
        T.append(num("S6", f"range 'the network term takes LO-HI % up to d = 15', {sname}: HI", "%",
                     math.ceil(100 * max(fr_old)), None if None in fr_new else math.ceil(100 * max(fr_new)), GRACE))
        T.append(num("S11", f"range 'LO-HI times its mean time per shot', {sname}: LO", "x",
                     OLD["s11"][0] if s == "deployed" else None, floor_to(min(s11), 1) if s11 else None,
                     "one core of the Linux workstation's Ryzen", why="no network of the set routes a shot"))
        T.append(num("S11", f"range 'LO-HI times its mean time per shot', {sname}: HI", "x",
                     OLD["s11"][1] if s == "deployed" else None, ceil_to(max(s11), 1) if s11 else None,
                     "one core of the Linux workstation's Ryzen", why="no network of the set routes a shot"))
    # S7: the fastest baseline and the margin (Grace basis)
    for s, sname in SETS:
        for i, d in enumerate(DS):
            n = nets[s][d]
            vals = [(bval(g, k, d), name) for k, name in BASE]
            ok = all(v is not None for v, _ in vals)
            best = min(vals) if ok else (None, None)
            o = OLD["s7"][str(d)] if s == "deployed" else (None, None)
            T.append(num("S7", f"fastest baseline at d = {d} and its margin over ours, {sname}", "us", o[1],
                         _sub(best[0], comp["grace"].get(n)), GRACE, fastest_old=o[0], fastest=best[1]))
    # S8: d = 17 on the registered (fitted) basis; the separate d = 17 basis of the earlier plan is gone
    for s, sname in SETS:
        n = nets[s][17]
        T.append(num("S8", f"routed CPU term b at d = 17, {sname} ({n})", "us", OLD["s8"]["b"] if s == "deployed" else None,
                     parts.get(f"grace.{n}", {}).get("b"), GRACE))
        ours = comp["grace"].get(n)
        cs = [None if ours is None or bval(g, k, 17) is None else ours < bval(g, k, 17) for k, _ in BASE + CTRL]
        T.append(num("S8", f"comparisons at d = 17 that hold ('all eight speed comparisons at d = 17 still hold'), {sname}",
                     "count", 8, None if None in cs else sum(cs), GRACE, why="a comparison at d = 17 is withheld"))
    f_hi = fits.get("high")
    f_lo = fits.get("low")
    for fit, d, o_old, nv in ((f_hi, 17, OLD["s8"]["band"][0], 860.71), (f_lo, 9, None, 67.59)):
        pt = None if fit is None else [x for x in fit["points"] if (x["decoder"], x["d"], x["p"]) == ("corr", d, 0.006)][0]
        T.append(num("S8", f"correlated PM at d = {d}, p = 0.6 %, on the fitted conversion (NVIDIA's measured "
                     f"{nv} us)", "us", o_old, None if pt is None else pt["xeon_us_per_shot"] * fit["applied"]["corr"],
                     GRACE + "; the residual of this fit point, decides nothing"))
    T.append(label("S8", "the margin with E6's 67.20 us network term (25.8 us)",
                   "E6's persistent engine is not released and not re-measured (D7): labelled as the earlier development "
                   "measurement"))
    # S9: the measured-CPU disclosure on the CPU worker's host (plain PM against ours with b timed there)
    for s, sname in SETS:
        for i, d in enumerate(DS):
            n = nets[s][d]
            o = OLD["s9"].get(str(d), (None, None)) if s == "deployed" else (None, None)
            T.append(cmp_("S9", f"ours vs PM, measured CPU, {sname}, d = {d}", "us", comp["xeon"].get(n),
                          bval(ep, "pm", d), o[0] if d != 17 else None, o[1] if d != 17 else None, XEON,
                          descriptive=True))
    T.append(cmp_("S9", "ours (deployed, d = 17 at study 4's q = 10 %) vs PM, measured CPU", "us",
                  comp["xeon"].get("d17_b5_r787_q10"), bval(ep, "pm", 17), OLD["s9"]["17"][0], OLD["s9"]["17"][1], XEON,
                  descriptive=True))
    # E rows
    T.append(num("E1", "the resident d = 7 kernel's GPU term (deployed d7_b1w64_r740, slowest of five)", "us", OLD["e1"],
                 Sq.get("S.d7_b1w64_r740.a"), "H100 device time per shot", headline="H4-kernel"))
    T.append(num("E1", "beside it: SURF-v2's d = 7 network on the same kernel (d7_b1w64_sv2)", "us", 8.466,
                 Sq.get("S.d7_b1w64_sv2.a"), "H100 device time per shot"))
    T.append(cmp_("E2", "the fused Triton engine at d = 7 vs PM's Grace value at d = 7 (the engine is the slower)", "us",
                  Sq.get("S.triton7.a"), bval(g, "pm", 7), OLD["e2"][0], OLD["e2"][1],
                  "Triton engine on the H100 (slowest of five, now from the release) against PM on the Grace basis",
                  relation="above"))
    T.append(num("E8", "CPU time saved at d = 17 routing 8 instead of 10 % (Grace basis)", "us", OLD["e8"][0],
                 _sub(parts.get("grace.d17_b5_r787_q10", {}).get("b"), parts.get("grace.d17_b5_r787_q08", {}).get("b")),
                 GRACE))
    T.append(num("E8", "the same in the Linux workstation's own Ryzen time", "us", OLD["e8"][1],
                 _sub(N.get("N.term.d17_b5_r787_q10"), N.get("N.term.d17_b5_r787_q08")), "one core of the Linux workstation's Ryzen"))
    for r, item, why_ in (
            ("E3", "a two-block d = 9 network on development GPU slices (26.6-27.5 us against 241-283 us)",
             "not released, MIG slices (D7): labelled as an earlier exploratory development measurement"),
            ("E4", "about 40 small kernels per shot at batch 1", "a kernel count, not a time"),
            ("E5", "7 of the H100's 132 multiprocessors and one spinning host core", "a resource statement, not a time"),
            ("E6", "a persistent engine at d = 17: 74.57 us against 67.20 us",
             "the engine is not released (D7): labelled as the earlier development measurement"),
            ("E7", "the eight-block network's 25 us; the projected 19 us",
             "the network is not released (D7): labelled as an earlier exploratory measurement and a projection"),
            ("E9", "the routing rule's speed limit and projected times", "inputs of the q rule, fixed on calibration "
             "shots before each test: not re-measured"),
            ("E10", "the d = 13 threshold selection's predicted single-shot times", "inputs of a frozen selection "
             "rule: not re-measured")):
        T.append(label(r, item, why_))
    # F1: the Grace fit of each pod (the stage's section, "The Grace conversion"): raw Ryzen times, every point's ratio, the
    # factors with their spread, the rule's decision and the residuals; the per-point ratios are a diagnostic
    for role, rname in (("low", "pod S-low"), ("high", "pod S-high")):
        fit = fits.get(role)
        basis = f"{rname}: NVIDIA's Grace time over the Linux workstation's Ryzen time, per shot (a shot is d rounds)"
        why_ = why.get(f"FIT.{role}") or "the fit is not established"
        for key, name in (("plain", "plain PM"), ("corr", "correlated PM"), ("common", "both decoders")):
            T.append(num("F1", f"{rname}: fitted factor, {name} (geometric mean of the ratios)", "x", None,
                         None if fit is None else fit["factors"][key], basis, why=why_))
        for key, name in (("plain", "plain PM"), ("corr", "correlated PM")):
            T.append(num("F1", f"{rname}: spread of ln(ratio), {name} (standard deviation, n - 1)", "ln", None,
                         None if fit is None else fit["spread"][key], basis, why=why_))
        for key, name in (("plain", "plain PM terms"), ("corr", "correlated PM terms")):
            T.append(num("F1", f"{rname}: factor applied to {name}" + ("" if fit is None else
                         (" (one common factor)" if fit["common"] else " (its own factor)")), "x", None,
                         None if fit is None else fit["applied"][key], basis, why=why_))
        if fit is not None:
            for x in fit["points"]:
                T.append(num("F1", f"{rname}: {x['decoder']} PM, d = {x['d']}, p = {x['p'] * 100:g} %: NVIDIA "
                             f"{x['grace_us_per_shot']:.2f} us, Ryzen {x['xeon_us_per_shot']:.2f} us; ratio, then residual (ln)",
                             "x", None, x["ratio"], basis, residual_ln=x["residual_ln"], xeon_us=x["xeon_us_per_shot"],
                             grace_us=x["grace_us_per_shot"]))
    return T


def fmt(x, unit):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, str):
        return x
    if unit == "count":
        return f"{x:d}" if isinstance(x, int) else f"{x}"
    return f"{x:.3f}" if abs(x) < 1000 else f"{x:.1f}"


STD = ("row", "item", "kind", "unit", "old", "new", "why", "basis", "relation", "ours_old", "theirs_old", "ratio_old", "ours",
       "theirs", "ratio", "holds")


def render(T):
    """table.md and rows.csv (text) of the rows, in order; the csv's last column holds a row's extra fields as JSON."""
    import csv
    import io
    import json
    md = ["| row | item | unit | old | new | verdict | note |", "|---|---|---|---|---|---|---|"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["row", "kind", "item", "unit", "old", "new", "ours_old", "theirs_old", "ours", "theirs", "ratio_old",
                "ratio", "holds", "why", "basis", "extra"])
    for r in T:
        if r["kind"] == "comparison":
            old = f"{fmt(r['ours_old'], r['unit'])} vs {fmt(r['theirs_old'], r['unit'])}"
            new = f"{fmt(r['ours'], r['unit'])} vs {fmt(r['theirs'], r['unit'])}"
            verdict = ("withheld" if r["holds"] is None else ("holds" if r["holds"] else "does not hold")) + \
                (" (descriptive)" if r.get("descriptive") else "")
        else:
            old, new, verdict = fmt(r.get("old"), r["unit"]), fmt(r.get("new"), r["unit"]), ""
        md.append(f"| {r['row']} | {r['item']} | {r['unit'] or ''} | {old} | {new} | {verdict} | {r.get('why') or ''} |")
        extra = {k: v for k, v in r.items() if k not in STD}
        w.writerow([r["row"], r["kind"], r["item"], r["unit"], r.get("old"), r.get("new"), r.get("ours_old"),
                    r.get("theirs_old"), r.get("ours"), r.get("theirs"), r.get("ratio_old"), r.get("ratio"),
                    r.get("holds"), r.get("why"), r.get("basis"), json.dumps(extra, sort_keys=True) if extra else ""])
    return "\n".join(md) + "\n", buf.getvalue()
