"""ARB analysis (PLAN.md steps 3-5): arbiters on the routed shots of CoDecoder + Cascade where the two decisions differ.

  python arb_analysis.py --read OUT/read --lock d1_locked_depspam_fm.json --out OUT/analysis [--p 0.002,0.003]

Inputs per p (arb_read.py, arb_scores.py): the read p<p>.npz/.json (model 0 C14', model 1 S55'-deep), the float32 logits
p<p>_z32.npy, and S55'-deep's class scores p<p>_scores.npy on the rows p<p>_scores_rows.npy (their check must have
passed). Detector rows are never read.

Definitions. routed = S55'-deep's margin < D1's tau1 (the lock); the composition answers with C14' on routed shots and
S55'-deep elsewhere; the disagreement rows are the routed shots where the two decisions differ, and every arbiter acts on
them only (everything else as in the composition). Halves: fit = even chunks, evaluation = odd chunks. On a disagreement
row a choice of S55'-deep's answer repairs (+1) when only C14' fails, breaks (-1) when only S55'-deep fails, and counts
zero when both fail. Fitting objective: the composition's failures on the fit half, i.e. the largest repairs - breaks;
"always C14'" (gain 0) is always available, so a fitted gain is never negative. Features, computed once per p on the
disagreement rows (z = C14''s 12 logits, s = S55'-deep's 4,096 class scores, class bit j = observable j):
  A1m  switch to S55'-deep iff min |z| < t_C and margin > t_S
  A1s  the same with -sum softplus(-|z|) (pc_common.confidence's "softplus" score) for min |z|
  A2   switch iff dC < a + b dN; dC = sum over the differing bits of |z| = log P_C(C14''s answer) - log P_C(S55'-deep's),
       dN = s(S55'-deep's answer) - s(C14''s answer)
  A3   answer argmax_y [ s(y) + w (y . z) ] (log P_C(y) = y . z + const); may be a third answer
Sweeps (cumulative counts over sorted values; thresholds at midpoints between neighbouring fit-half values):
  A1   exact over both thresholds: t_S descends through every distinct fit-half margin, and for each the best t_C is
       read off the cumulative gain over min |z| (or the softplus score) groups; ties keep the first found (the highest
       t_S, then the smallest t_C), i.e. the fewest switches
  A2   b in {0} and 10^(k/40), k = -120..120; for each b the best a exactly by cumulative counts over sorted dC - b dN;
       ties keep the smaller b, then the smaller a
  A3   w in {0}, 10^(k/40) for k = -120..120 and infinity (= the composition); ties keep the larger w
Carried forward (fixed before any read): the arbiter with the largest mean over the two p of its fit-half relative
failure reduction (net gain / the composition's fit-half failures); ties go to fewer parameters (A3 has one), then to
the order A1m, A1s, A2, A3; none when the best mean is not positive. Reports on the evaluation half (every arbiter):
failures of C14' alone, the composition and the arbiter's composition; repairs, breaks and A3's third-answer changes;
the net improvement (repairs - breaks) / N_eval with Newcombe's 95 % interval and the exact two-sided McNemar p against
the composition (bb.route.paired); the relative failure reduction; the recovered share of the routed oracle (net / the
evaluation half's routed shots where only C14' fails); with no break, the one-sided 95 % upper bound 3 / N_eval. And
the per-category counts of the whole read beside D1's original reads (PLAN.md's table; X2's bb_d1.py REG)."""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

K, NY = 12, 4096
GRID = [10 ** (k / 40) for k in range(-120, 121)]
ARBS = ("A1m", "A1s", "A2", "A3")
NPAR = {"A1m": 2, "A1s": 2, "A2": 2, "A3": 1}
TAU1 = {0.002: 7.3331888389587405, 0.003: 7.60439167022705}
# D1's original reads (PLAN.md's table; X2's bb_d1.py REG): all shots, eager path
D1 = {0.002: dict(shots=11_000_000, C=7922, S=10348, only_S=4735, only_C=2309, both=5613, routed=220058,
                  kept_breaks=19, kept_repairs=86, kept_both=238),
      0.003: dict(shots=17_000_000, C=76730, S=112636, only_S=54491, only_C=18585, both=58145, routed=1701544,
                  kept_breaks=81, kept_repairs=342, kept_both=1977)}
for _v in D1.values():
    _v["C_fail_routed"] = _v["C"] - _v["kept_repairs"] - _v["kept_both"]
    _v["composition"] = _v["kept_breaks"] + _v["kept_both"] + _v["C_fail_routed"]
    _v["oracle_fail"] = _v["both"] + _v["kept_breaks"]
    _v["oracle_repairs"] = _v["composition"] - _v["oracle_fail"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cut_below(lo, hi):
    """A threshold t with lo < t <= hi, so that x < t keeps lo and drops hi (hi = inf: t = inf)."""
    if not np.isfinite(hi):
        return math.inf
    m = lo + (hi - lo) / 2
    return m if lo < m <= hi else hi


def cut_above(lo, hi):
    """A threshold t with lo <= t < hi, so that y > t keeps hi and drops lo (lo = -inf: t = -inf)."""
    if not np.isfinite(lo):
        return -math.inf
    m = lo + (hi - lo) / 2
    return m if lo <= m < hi else lo


# ---------------------------------------------------------------------------------------------------------- fitting
def fit_a1(x, y, g):
    """max over (t_C, t_S) of sum g[x < t_C and y > t_S] -> dict(gain, t_C, t_S)."""
    best = {"gain": 0, "t_C": -math.inf, "t_S": math.inf}
    if len(x) == 0:
        return best
    ux, gx = np.unique(x, return_inverse=True)
    o = np.argsort(-y, kind="stable")
    ys, gs, xs = y[o], g[o].astype(np.int64), gx[o]
    starts = np.r_[0, np.flatnonzero(ys[1:] != ys[:-1]) + 1]
    ends = np.r_[starts[1:], len(ys)]
    leaf = np.zeros(len(ux), np.int64)
    for i, j in zip(starts, ends):
        np.add.at(leaf, xs[i:j], gs[i:j])
        cs = np.cumsum(leaf)
        k = int(np.argmax(cs))
        if cs[k] > best["gain"]:
            best = {"gain": int(cs[k]), "t_C": cut_below(ux[k], ux[k + 1] if k + 1 < len(ux) else math.inf),
                    "t_S": cut_above(ys[j] if j < len(ys) else -math.inf, ys[i])}
    return best


def a1_switch(x, y, t):
    return (x < t["t_C"]) & (y > t["t_S"])


def fit_a2(dC, dN, g):
    best = {"gain": 0, "a": -math.inf, "b": 0.0}
    if len(dC) == 0:
        return best
    for b in [0.0] + GRID:
        u = dC - b * dN
        o = np.argsort(u, kind="stable")
        us, cs = u[o], np.cumsum(g[o].astype(np.int64))
        last = np.r_[us[1:] != us[:-1], True]
        cand = np.where(last, cs, np.iinfo(np.int64).min)
        k = int(np.argmax(cand))
        if cand[k] > best["gain"]:
            best = {"gain": int(cand[k]), "a": cut_below(us[k], us[k + 1] if k + 1 < len(us) else math.inf), "b": b}
    return best


def a2_switch(dC, dN, t):
    return (dC - t["b"] * dN) < t["a"]


def a3_answers(sc, z, ws, yC, block=16384):
    """argmax_y [ s(y) + w (y . z) ] for every w (inf: C14''s answer) -> (len(ws), n) int16 classes."""
    n = len(z)
    out = np.empty((len(ws), n), np.int16)
    Y = ((np.arange(NY)[:, None] >> np.arange(K)) & 1).astype(np.float64)
    try:
        import torch
        dev = torch.device("cuda") if torch.cuda.is_available() else None
    except ImportError:
        dev = None
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        if dev is not None:
            S = torch.as_tensor(np.asarray(sc[lo:hi]), device=dev).double()
            Yz = torch.as_tensor(z[lo:hi], device=dev) @ torch.as_tensor(Y.T.copy(), device=dev)
            for k, w in enumerate(ws):
                out[k, lo:hi] = yC[lo:hi] if math.isinf(w) else torch.argmax(S + w * Yz, 1).cpu().numpy()
        else:
            S = np.asarray(sc[lo:hi]).astype(np.float64)
            Yz = z[lo:hi] @ Y.T
            for k, w in enumerate(ws):
                out[k, lo:hi] = yC[lo:hi] if math.isinf(w) else np.argmax(S + w * Yz, 1)
    return out


# --------------------------------------------------------------------------------------------------------- counting
def cats(fC, fN, routed, differ, comp):
    """Per-category counts on a set of shots."""
    keep = ~routed
    r = {"shots": int(len(fC)), "C": int(fC.sum()), "S": int(fN.sum()), "only_S": int((fN & ~fC).sum()),
         "only_C": int((fC & ~fN).sum()), "both": int((fC & fN).sum()), "routed": int(routed.sum()),
         "kept_breaks": int((fN & ~fC & keep).sum()), "kept_repairs": int((~fN & fC & keep).sum()),
         "kept_both": int((fN & fC & keep).sum()), "C_fail_routed": int((fC & routed).sum()),
         "composition": int(comp.sum()), "routed_differ": int((routed & differ).sum()),
         "routed_both_fail": int((routed & fC & fN).sum()), "routed_both_fail_same_answer":
         int((routed & fC & fN & ~differ).sum()), "routed_only_S": int((routed & fN & ~fC).sum())}
    r["oracle_repairs"] = int((routed & fC & ~fN).sum())
    r["oracle_fail"] = r["composition"] - r["oracle_repairs"]
    return r


def two_rate_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else float("nan")
    return (k1 / n1 - k2 / n2) / se if se == se and se > 0 else float("nan")


def evaluate(name, ans, d, yC, yN, yO, comp, fC, ev, oracle_rep, paired):
    """An arbiter's composition on the evaluation half: ans = its class on every disagreement row d (both halves)."""
    arbf = comp.copy()
    arbf[d] = ans != yO
    e = ev[d]
    a, c = ans[e], yC[e]
    ch = a != c
    rep = ch & (a == yO[e])                                      # the composition failed (c != yO since a == yO != c)
    brk = ch & (c == yO[e])
    third = ch & (a != yN[e])
    n_ev = int(ev.sum())
    st = paired(arbf[ev], comp[ev], 0.05)                        # LER(arbiter) - LER(composition)
    vsC = paired(arbf[ev], fC[ev], 0.05)
    net = int(rep.sum() - brk.sum())
    assert net == st["only_b"] - st["only_a"], (name, net, st)
    comp_ev = int(comp[ev].sum())
    r = {"failures": int(arbf[ev].sum()), "changed": int(ch.sum()), "repairs": int(rep.sum()), "breaks": int(brk.sum()),
         "third": {"changes": int(third.sum()), "repairs": int((third & rep).sum()), "breaks": int((third & brk).sum()),
                   "neutral": int((third & ~rep & ~brk).sum())},
         "net": net, "net_per_shot": net / n_ev, "net_lo": -st["diff_hi"], "net_hi": -st["diff_lo"],
         "mcnemar_p": st["mcnemar_p"], "rel_reduction": net / comp_ev if comp_ev else float("nan"),
         "rel_lo": -st["diff_hi"] * n_ev / comp_ev if comp_ev else float("nan"),
         "rel_hi": -st["diff_lo"] * n_ev / comp_ev if comp_ev else float("nan"),
         "oracle_share": net / oracle_rep if oracle_rep else float("nan"),
         "upper_break_rate_3_over_N": 3 / n_ev if int(brk.sum()) == 0 else None,
         "vs_cascade": {"diff_ler": vsC["diff_ler"], "lo": vsC["diff_lo"], "hi": vsC["diff_hi"],
                        "mcnemar_p": vsC["mcnemar_p"]}}
    return r, arbf


def fit_gain(ans, yC, yO, fit_rows, detail=False):
    a, c, o = ans[fit_rows], yC[fit_rows], yO[fit_rows]
    rep, brk = int(((a == o) & (c != o)).sum()), int(((a != o) & (c == o)).sum())
    if detail:
        return rep - brk, int((a != c).sum()), rep, brk
    return rep - brk, int((a != c).sum())


# ------------------------------------------------------------------------------------------------------------- main
def analyse_p(read, p, tau1, paired, out_dir):
    t0 = time.time()
    tag = f"p{p:.5f}"
    meta = json.load(open(os.path.join(read, f"{tag}.json")))
    if [m["kind"] for m in meta["models"]] != ["cascade", "convlines"]:
        raise SystemExit(f"{read}/{tag}: models are not (C14', S55'-deep)")
    z = np.load(os.path.join(read, f"{tag}.npz"))
    fC, fN = z["fail0"].astype(bool), z["fail1"].astype(bool)
    pred0, pred1, obs, chunk = z["pred0"], z["pred1"], z["obs"], z["chunk"]
    conf0, conf1 = z["conf0"], z["conf1"]
    margin = conf1.astype(np.float64)
    routed = margin < tau1
    differ = (pred0 != pred1).any(1)
    comp = np.where(routed, fC, fN)
    fit = chunk % 2 == 0
    ev = ~fit
    rec = {"p": p, "tau1": tau1, "chunks": meta["chunks"], "shots": int(len(fC)), "fit_shots": int(fit.sum()),
           "eval_shots": int(ev.sum())}
    rec["counts"] = {"all": cats(fC, fN, routed, differ, comp),
                     "fit": cats(fC[fit], fN[fit], routed[fit], differ[fit], comp[fit]),
                     "eval": cats(fC[ev], fN[ev], routed[ev], differ[ev], comp[ev])}
    a_ = rec["counts"]["all"]
    rec["vs_d1"] = {k: {"arb": a_[k], "d1": D1[p][k], "arb_rate": a_[k] / a_["shots"], "d1_rate": D1[p][k] / D1[p]["shots"],
                        "z": two_rate_z(a_[k], a_["shots"], D1[p][k], D1[p]["shots"])}
                    for k in ("C", "S", "only_S", "only_C", "both", "routed", "kept_breaks", "kept_repairs", "kept_both",
                              "C_fail_routed", "composition", "oracle_fail", "oracle_repairs")}
    # ---- features, once, on the disagreement rows
    d = np.nonzero(routed & differ)[0]
    srec = json.load(open(os.path.join(read, f"{tag}_scores.json")))
    rows = np.load(os.path.join(read, f"{tag}_scores_rows.npy"))
    if not srec["pass"] or not np.array_equal(rows, d):
        raise SystemExit(f"{tag}: the score export did not pass its checks or is not on the disagreement rows")
    sc = np.load(os.path.join(read, f"{tag}_scores.npy"), mmap_mode="r")
    zrec = json.load(open(os.path.join(read, f"{tag}_z32.json")))
    if not (zrec["float16_equals_read_z"] and zrec["minabs_equals_read_conf_bitwise"]):
        raise SystemExit(f"{tag}: the float32 logits did not pass their checks")
    z32 = np.load(os.path.join(read, f"{tag}_z32.npy"), mmap_mode="r")
    n = len(d)
    pw = (1 << np.arange(K)).astype(np.int64)
    bC = np.unpackbits(pred0[d], axis=1, count=K)
    bN = np.unpackbits(pred1[d], axis=1, count=K)
    bO = np.unpackbits(obs[d], axis=1, count=K)
    yC, yN, yO = bC @ pw, bN @ pw, bO @ pw
    zz = np.asarray(z32[d]).astype(np.float64)
    assert np.array_equal((zz > 0).astype(np.uint8), bC), "C14''s decision is not z > 0"
    assert np.array_equal(fC[d], yC != yO) and np.array_equal(fN[d], yN != yO)
    g = np.where(fC[d] & ~fN[d], 1, np.where(~fC[d] & fN[d], -1, 0)).astype(np.int64)
    assert not (~fC[d] & ~fN[d]).any()
    confm = conf0[d].astype(np.float64)
    confs = -np.logaddexp(0.0, -np.abs(zz)).sum(1)
    dC = (np.abs(zz) * (bC != bN)).sum(1)
    S = np.asarray(sc)
    sN = S[np.arange(n), yN].astype(np.float64)
    sC = S[np.arange(n), yC].astype(np.float64)
    dN = sN - sC
    mN = margin[d]
    assert (dN >= 0).all(), "S55'-deep's answer is not its top score"
    h = fit[d]
    oracle_rep = rec["counts"]["eval"]["oracle_repairs"]
    # ---- fits (fit half only) and answers on every disagreement row
    fits, answers = {}, {}
    for nm, x in (("A1m", confm), ("A1s", confs)):
        t = fit_a1(x[h], mN[h], g[h])
        fits[nm] = t
        answers[nm] = np.where(a1_switch(x, mN, t), yN, yC)
    t = fit_a2(dC[h], dN[h], g[h])
    fits["A2"] = t
    answers["A2"] = np.where(a2_switch(dC, dN, t), yN, yC)
    ws = [0.0] + GRID + [math.inf]
    A3 = a3_answers(S, zz, ws, yC)
    gains = [fit_gain(A3[k], yC, yO, h)[0] for k in range(len(ws))]
    kbest = max(range(len(ws)), key=lambda k: (gains[k], ws[k]))   # ties: the larger w
    fits["A3"] = {"gain": gains[kbest], "w": ws[kbest]}
    answers["A3"] = A3[kbest].astype(np.int64)
    a3_curve = {"w": ws, "fit_gain": gains, "eval_gain": [fit_gain(A3[k], yC, yO, ~h)[0] for k in range(len(ws))]}
    assert np.array_equal(A3[0].astype(np.int64), yN), "A3 at w = 0 is not S55'-deep's answer"
    # ---- per-arbiter fit-half record and evaluation-half report
    comp_fit = rec["counts"]["fit"]["composition"]
    rec["arbiters"] = {}
    for nm in ARBS:
        fg, fch, frep, fbrk = fit_gain(answers[nm], yC, yO, h, detail=True)
        assert fg == fits[nm]["gain"], (nm, fg, fits[nm])
        r, _ = evaluate(nm, answers[nm], d, yC, yN, yO, comp, fC, ev, oracle_rep, paired)
        rec["arbiters"][nm] = {"params": {k: v for k, v in fits[nm].items() if k != "gain"},
                               "fit": {"gain": fg, "changed": fch, "repairs": frep, "breaks": fbrk,
                                       "rel_reduction": fg / comp_fit if comp_fit else 0.0},
                               "eval": r}
    # references on the evaluation half: always S55'-deep on the disagreement rows (A3 at w = 0) and the routed oracle
    r, _ = evaluate("always_S", yN, d, yC, yN, yO, comp, fC, ev, oracle_rep, paired)
    rec["reference"] = {"always_S_on_disagreement": r}
    oracle = np.where(fC[d] & ~fN[d], yN, yC)
    r, _ = evaluate("oracle", oracle, d, yC, yN, yO, comp, fC, ev, oracle_rep, paired)
    rec["reference"]["routed_oracle"] = r
    rec["a3_curve"] = a3_curve
    rec["features"] = {"rows": n, "fit_rows": int(h.sum()), "eval_rows": int((~h).sum()),
                       "gain_counts_fit": {str(k): int((g[h] == k).sum()) for k in (-1, 0, 1)},
                       "gain_counts_eval": {str(k): int((g[~h] == k).sum()) for k in (-1, 0, 1)}}
    np.savez_compressed(os.path.join(out_dir, f"disagree_{tag}.npz"), rows=d, fit=h, yC=yC.astype(np.int16),
                        yN=yN.astype(np.int16), yO=yO.astype(np.int16), gain=g.astype(np.int8),
                        z=zz.astype(np.float32), min_abs_z=confm, softplus_score=confs, margin=mN, dC=dC, dN=dN,
                        **{f"ans_{k}": v.astype(np.int16) for k, v in answers.items()})
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def choose(per_p):
    """The arbiter carried forward: the largest mean fit-half relative reduction over p (module docstring)."""
    score = {nm: float(np.mean([r["arbiters"][nm]["fit"]["rel_reduction"] for r in per_p.values()])) for nm in ARBS}
    best = sorted(ARBS, key=lambda nm: (-score[nm], NPAR[nm], ARBS.index(nm)))[0]
    return {"rule": "largest mean over p of the fit-half relative failure reduction; ties: fewer parameters, then "
                    "A1m, A1s, A2, A3; none if not positive",
            "fit_scores": score, "carried_forward": best if score[best] > 0 else None}


def fmt_p(x):
    return "n/a" if x is None or x != x else (f"{x:.2e}" if x < 1e-3 else f"{x:.3f}")


def tables(res):
    L = []
    for key, r in res["points"].items():
        e = r["counts"]["eval"]
        n = r["eval_shots"]
        L.append(f"### p = {float(key) * 100:.1f} % (evaluation half: odd chunks, {n:,} shots; tau1 = {r['tau1']})\n")
        L.append("| decoder | failures | repairs | breaks | third-answer changes (repairs / breaks / neutral) | net "
                 "(repairs - breaks) / N_eval, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | "
                 "McNemar p vs composition | fitted on the fit half (fit-half net) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        L.append(f"| Cascade alone (C14') | {e['C']:,} | | | | | | | | |")
        L.append(f"| composition (D1's rule) | {e['composition']:,} | | | | | | | | |")
        for nm in ARBS:
            a = r["arbiters"][nm]
            v = a["eval"]
            t = v["third"]
            third = f"{t['changes']:,} ({t['repairs']:,} / {t['breaks']:,} / {t['neutral']:,})" if nm == "A3" else "-"
            ub = f" (<= {v['upper_break_rate_3_over_N']:.1e} per shot)" if v["upper_break_rate_3_over_N"] else ""
            par = ", ".join(f"{k} = {x:.4g}" for k, x in a["params"].items())
            L.append(f"| {nm} | {v['failures']:,} | {v['repairs']:,} | {v['breaks']:,}{ub} | {third} | "
                     f"{v['net_per_shot']:.2e} [{v['net_lo']:.2e}, {v['net_hi']:.2e}] | {v['rel_reduction'] * 100:.2f} % "
                     f"[{v['rel_lo'] * 100:.2f}, {v['rel_hi'] * 100:.2f}] | "
                     f"{v['oracle_share'] * 100:.1f} % | {fmt_p(v['mcnemar_p'])} | {par} ({a['fit']['gain']:+,}) |")
        for nm, lab in (("always_S_on_disagreement", "reference: always CoDecoder on the disagreement rows"),
                        ("routed_oracle", "reference: routed oracle (two answers)")):
            v = r["reference"][nm]
            L.append(f"| {lab} | {v['failures']:,} | {v['repairs']:,} | {v['breaks']:,} | - | "
                     f"{v['net_per_shot']:.2e} [{v['net_lo']:.2e}, {v['net_hi']:.2e}] | {v['rel_reduction'] * 100:.2f} % | "
                     f"{v['oracle_share'] * 100:.1f} % | {fmt_p(v['mcnemar_p'])} | |")
        L.append("")
        f_ = r["counts"]["fit"]
        L.append(f"Fit half (even chunks, {r['fit_shots']:,} shots; composition {f_['composition']:,} failures, routed "
                 f"oracle repairs {f_['oracle_repairs']:,}): " + "; ".join(
                     f"{nm} {r['arbiters'][nm]['fit']['repairs']:,} repairs, {r['arbiters'][nm]['fit']['breaks']:,} "
                     f"breaks, net {r['arbiters'][nm]['fit']['gain']:+,} ({r['arbiters'][nm]['fit']['rel_reduction'] * 100:.2f} %)"
                     for nm in ARBS) + ".\n")
        L.append(f"Evaluation half: routed {e['routed']:,}, disagreement rows {e['routed_differ']:,} (only C14' fails "
                 f"{e['oracle_repairs']:,}, only CoDecoder fails {e['routed_only_S']:,}, both fail with different answers "
                 f"{e['routed_differ'] - e['oracle_repairs'] - e['routed_only_S']:,}); routed oracle failures "
                 f"{e['oracle_fail']:,}.\n")
        L.append(f"Per-category counts of the whole read ({r['shots']:,} shots) beside D1's original reads "
                 f"(rates per 10^6 shots; z of the difference, independent samples):\n")
        L.append("| count | ARB | D1 | ARB rate | D1 rate | z |")
        L.append("|---|---|---|---|---|---|")
        for k, v in r["vs_d1"].items():
            L.append(f"| {k} | {v['arb']:,} | {v['d1']:,} | {v['arb_rate'] * 1e6:.1f} | {v['d1_rate'] * 1e6:.1f} | "
                     f"{v['z']:+.2f} |")
        L.append("")
    c = res["choice"]
    L.append(f"Carried forward (fit half only; {c['rule']}): **{c['carried_forward']}**; mean fit-half relative "
             f"reduction " + ", ".join(f"{k} {v * 100:.2f} %" for k, v in c["fit_scores"].items()) + ".\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--lock", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", default="0.002,0.003")
    a = ap.parse_args()
    from bb.route import paired
    os.makedirs(a.out, exist_ok=True)
    tau = dict(TAU1)
    if a.lock:
        L = json.load(open(a.lock))
        for c in L["configs"]:
            if c["p"] in tau and float(c["tau"]["tau1"]) != tau[c["p"]]:
                raise SystemExit(f"the lock's tau1 at {c['p']} is not D1's")
    res = {"points": {}, "inputs": {}}
    for p in (float(x) for x in a.p.split(",")):
        res["points"][str(p)] = analyse_p(a.read, p, tau[p], paired, a.out)
        tag = f"p{p:.5f}"
        res["inputs"][str(p)] = {"z32_sha256": json.load(open(os.path.join(a.read, f"{tag}_z32.json")))["sha256"],
                                 "scores_sha256": json.load(open(os.path.join(a.read, f"{tag}_scores.json")))["scores_sha256"]}
        print(json.dumps({"p": p, "secs": res["points"][str(p)]["secs"],
                          "eval": {nm: {k: res["points"][str(p)]["arbiters"][nm]["eval"][k]
                                        for k in ("repairs", "breaks", "net")}
                                   for nm in ARBS}}), flush=True)
    res["choice"] = choose(res["points"]) if len(res["points"]) == 2 else None
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    if res["choice"]:
        open(os.path.join(a.out, "tables.md"), "w").write(tables(res))
    print(json.dumps({"choice": res["choice"]}), flush=True)


if __name__ == "__main__":
    main()
