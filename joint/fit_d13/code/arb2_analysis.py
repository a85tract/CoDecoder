"""ARB at d = 13, the analysis (PLAN.md steps 3-5): rules on the routed shots of the composition (ConvLines d13_b5_r775
-> Cascade CS13, the comparison's C5) where the two decisions differ. One logical bit.

  python arb2_analysis.py --read OUT/read --lock RELEASE/cascade/d13/out/locked_d13.json --out OUT/analysis \
      [--p 0.006,0.0035]

Inputs per p: arb2_read.py's parts OUT/read/p<p>/part_*.npz (obs, chunk, c, sdec, z, bdec; the detector rows are not
read). Definitions:
  S = the network's decision c > 0 (the engine's own decision, checked equal), B = Cascade's z > 0 (checked equal);
  routed = |c| (float32, compared in float64) strictly below the lock's tau1 of C5 (sv_common.routed); the composition
  answers B on routed shots and S elsewhere; the disagreement rows are the routed shots with S != B. With one bit,
  exactly one of the two is right on such a row: taking S's answer repairs (+1) when B is wrong and breaks (-1) when S is
  wrong. Every rule acts on the disagreement rows only; everywhere else the composition's answer stands.
  Halves: fit = even chunks, evaluation = odd chunks. Every parameter is fitted on the fit half, and the rule carried
  forward is chosen there; everything is reported on the evaluation half.
Rules (on a disagreement row; "switch" = take the network's answer):
  J  joint decision: answer 1 iff c + w z > 0, Cascade's answer on a tie (c + w z = 0); w = inf is the composition.
     On a disagreement row this switches iff |c| > w |z|, so the fit-half gain is a step function of w with steps at
     the ratios |c| / |z| of the fit-half rows: the sweep is exact over every step. The fitted w is the geometric
     midpoint of the best step (the first step [0, r1): r1 / 2; the last finite step [rm, inf): 2 rm); ties keep the
     larger w. The answers are then recomputed from c + w z itself (float64) and the fit-half gain is asserted equal.
  P  preference: switch iff |z| < a + b |c|; b in {0} and 10^(k/40), k = -120..120; for each b the best a exactly by
     cumulative counts over sorted |z| - b |c| (a at the midpoint between neighbouring values); ties keep the smaller
     b, then the smaller a (ARB's A2).
  T  two thresholds: switch iff |z| < t_C and |c| > t_S, exact over both thresholds (ARB's A1 sweep); ties keep the
     first found (the highest t_S, then the smallest t_C), i.e. the fewest switches.
Objective on the fit half: the composition's failures, i.e. the largest repairs - breaks ("never switch" = 0 is always
available). Carried forward (fixed before any read): the rule with the largest mean over the two p of its fit-half
relative failure reduction (gain / the composition's fit-half failures); ties go to fewer parameters (J has one), then
to the order J, P, T; none when the best mean is not positive. Reported per p on the evaluation half: the failures of
Cascade alone, the network alone, the composition and each rule's composition; repairs and breaks; the net (repairs -
breaks) per 10^4 shots with Newcombe's 95 % interval (method 10, sv_common.newcombe10, as bb.route.paired) and the exact
two-sided McNemar p against the composition (sv_common.mcnemar_p); the relative reduction (net / the composition's
failures, its interval the net's divided by that count, approximate); the routed oracle (right whenever one of the two
is right: on the disagreement rows always right) and the share of its repairs each rule recovers. Also: J's gain over a
grid of w on both halves, the evaluation half's own best w (diagnostic), one w for both p (diagnostic: the grid value
with the largest mean fit-half relative reduction), and the per-category counts of the whole read beside the
re-test's (paper Table I) and the comparison's original counts."""
import argparse
import glob
import hashlib
import json
import math
import os
import time

import numpy as np

from surface import sv_common as sc

GRID = [10 ** (k / 40) for k in range(-120, 121)]
RULES = ("J", "P", "T")
NPAR = {"J": 1, "P": 2, "T": 2}
Z95 = 1.959964
TAU1 = {0.006: 5.242445659637451, 0.0035: 8.111059150695802}     # C5's tau1 (RETEST.md section 3.3, the lock)
SHOTS = {0.006: 4_000_000, 0.0035: 30_000_000}
# Counts per configuration (RETEST_RESULTS.md, "Failures per decoder" and "The pair against Cascade alone"): C2 =
# Cascade alone, C5 = the composition (the pair), only_pair = C5 fails and C2 does not, only_C2 = the reverse,
# routed = C5's routed share (rounded in the source).
REF = {"retest_single": {0.006: dict(shots=1_000_000, C2=4936, C5=4917, only_pair=21, only_C2=40, routed=0.0998),
                         0.0035: dict(shots=30_000_000, C2=2354, C5=2325, only_pair=0, only_C2=29, routed=0.0100)},
       "retest_batched": {0.006: dict(shots=1_000_000, C2=4947, C5=4927, only_pair=21, only_C2=41),
                          0.0035: dict(shots=30_000_000, C2=2352, C5=2322, only_pair=0, only_C2=30)},
       "original_single": {0.006: dict(shots=1_000_000, C2=4963, C5=4936, only_pair=13, only_C2=40, routed=0.0996),
                           0.0035: dict(shots=30_000_000, C2=2406, C5=2373, only_pair=0, only_C2=33, routed=0.0100)},
       "original_batched": {0.006: dict(shots=1_000_000, C2=4953, C5=4926, only_pair=13, only_C2=40),
                            0.0035: dict(shots=30_000_000, C2=2411, C5=2376, only_pair=0, only_C2=35)}}


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


# ---------------------------------------------------------------------------------------------------------- rules
def j_switch(c, z, w):
    """J on disagreement rows: True where its answer (1 iff c + w z > 0; a tie keeps Cascade's) is the network's."""
    if math.isinf(w):
        return np.zeros(len(c), bool)
    v = c + w * z
    B = z > 0
    ans = np.where(v > 0, True, np.where(v < 0, False, B))
    return ans != B


def fit_j(c, z, g):
    """Exact sweep of J's w on rows (c, z, gain g) -> dict(gain, w, step=[lo, hi], switches)."""
    ac, az = np.abs(c), np.abs(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(az > 0, ac / az, np.where(ac > 0, np.inf, 0.0))
    fin = np.unique(r[np.isfinite(r) & (r > 0)])
    # gain on the step [fin[k-1], fin[k]) (fin[-1] := 0) = sum of g over rows with r >= fin[k] (r > w for w in it)
    best = {"gain": 0, "w": math.inf, "step": [math.inf, math.inf]}
    o = np.argsort(-r, kind="stable")
    rs, gs = r[o], np.cumsum(g[o].astype(np.int64))
    # rows with r > w for w on step k: the ones with r >= fin[k]; their count is searchsorted on -rs
    edges = [0.0] + list(fin)
    for k in range(len(edges) - 1, -1, -1):          # from the largest w down, so that ties keep the larger w
        lo = edges[k]
        hi = edges[k + 1] if k + 1 < len(edges) else math.inf
        n_sw = int(np.searchsorted(-rs, -lo, side="left"))          # rows with r > lo
        gain = int(gs[n_sw - 1]) if n_sw else 0
        if gain > best["gain"]:
            if k == 0:
                w = hi / 2 if np.isfinite(hi) else 1.0
            elif not np.isfinite(hi):
                w = 2 * lo
            else:
                w = math.sqrt(lo * hi)
            best = {"gain": gain, "w": float(w), "step": [float(lo), float(hi)], "switches": n_sw}
    return best


def fit_t(x, y, g):
    """max over (t_C, t_S) of sum g[x < t_C and y > t_S] -> dict(gain, t_C, t_S) (ARB's fit_a1)."""
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


def t_switch(x, y, t):
    return (x < t["t_C"]) & (y > t["t_S"])


def fit_p(x, y, g):
    """max over (a, b) of sum g[x < a + b y] (ARB's fit_a2 with dC = |z|, dN = |c|) -> dict(gain, a, b)."""
    best = {"gain": 0, "a": -math.inf, "b": 0.0}
    if len(x) == 0:
        return best
    for b in [0.0] + GRID:
        u = x - b * y
        o = np.argsort(u, kind="stable")
        us, cs = u[o], np.cumsum(g[o].astype(np.int64))
        last = np.r_[us[1:] != us[:-1], True]
        cand = np.where(last, cs, np.iinfo(np.int64).min)
        k = int(np.argmax(cand))
        if cand[k] > best["gain"]:
            best = {"gain": int(cand[k]), "a": cut_below(us[k], us[k + 1] if k + 1 < len(us) else math.inf), "b": b}
    return best


def p_switch(x, y, t):
    return (x - t["b"] * y) < t["a"]


# --------------------------------------------------------------------------------------------------------- counting
def cats(fB, fS, routed, differ, comp):
    """Per-category counts on a set of shots (one bit: on routed rows that differ exactly one decoder fails)."""
    keep = ~routed
    r = {"shots": int(len(fB)), "C": int(fB.sum()), "S": int(fS.sum()), "only_S": int((fS & ~fB).sum()),
         "only_C": int((fB & ~fS).sum()), "both": int((fB & fS).sum()), "routed": int(routed.sum()),
         "kept_breaks": int((fS & ~fB & keep).sum()), "kept_repairs": int((~fS & fB & keep).sum()),
         "kept_both": int((fS & fB & keep).sum()), "C_fail_routed": int((fB & routed).sum()),
         "composition": int(comp.sum()), "routed_differ": int((routed & differ).sum()),
         "routed_both_fail": int((routed & fB & fS).sum()), "routed_only_S": int((routed & fS & ~fB).sum())}
    r["oracle_repairs"] = int((routed & fB & ~fS).sum())
    r["oracle_fail"] = r["composition"] - r["oracle_repairs"]
    r["routed_share"] = r["routed"] / r["shots"] if r["shots"] else float("nan")
    return r


def two_rate_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else float("nan")
    return (k1 / n1 - k2 / n2) / se if se == se and se > 0 else float("nan")


def paired(fa, fb):
    """P(a fails) - P(b fails) on the same shots: Newcombe's 95 % interval (method 10) and the exact two-sided
    McNemar p (sv_common.newcombe10 / mcnemar_p, phase C's bb.route.paired)."""
    fa, fb = np.asarray(fa, bool), np.asarray(fb, bool)
    n = len(fa)
    both, oa, ob = int((fa & fb).sum()), int((fa & ~fb).sum()), int((~fa & fb).sum())
    dt, lo, hi = sc.newcombe10(both, oa, ob, n - both - oa - ob, Z95)
    return {"shots": n, "fail_a": int(fa.sum()), "fail_b": int(fb.sum()), "only_a": oa, "only_b": ob,
            "diff": dt, "lo": lo, "hi": hi, "mcnemar_p": sc.mcnemar_p(oa, ob)}


def evaluate(sw, d, fB, fS, comp, ev, oracle_rep):
    """A rule's composition on the evaluation half; sw = its switches on every disagreement row d (both halves)."""
    arbf = comp.copy()
    arbf[d] = np.where(sw, fS[d], fB[d])
    e = ev[d]
    rep = sw[e] & fB[d][e]
    brk = sw[e] & fS[d][e]
    n_ev = int(ev.sum())
    st = paired(arbf[ev], comp[ev])
    vsC = paired(arbf[ev], fB[ev])
    net = int(rep.sum() - brk.sum())
    assert net == st["only_b"] - st["only_a"], (net, st)
    comp_ev = int(comp[ev].sum())
    r = {"failures": int(arbf[ev].sum()), "switches": int(sw[e].sum()), "repairs": int(rep.sum()),
         "breaks": int(brk.sum()), "net": net, "net_per_1e4": 1e4 * net / n_ev, "net_per_1e4_lo": -1e4 * st["hi"],
         "net_per_1e4_hi": -1e4 * st["lo"], "mcnemar_p": st["mcnemar_p"],
         "rel_reduction": net / comp_ev if comp_ev else float("nan"),
         "rel_lo": -st["hi"] * n_ev / comp_ev if comp_ev else float("nan"),
         "rel_hi": -st["lo"] * n_ev / comp_ev if comp_ev else float("nan"),
         "oracle_share": net / oracle_rep if oracle_rep else float("nan"),
         "upper_break_rate_3_over_N": 3 / n_ev if int(brk.sum()) == 0 else None,
         "vs_cascade_alone": {"diff": vsC["diff"], "lo": vsC["lo"], "hi": vsC["hi"], "mcnemar_p": vsC["mcnemar_p"],
                              "only_rule": vsC["only_a"], "only_cascade": vsC["only_b"]}}
    return r


def gain_of(sw, g, rows):
    return int(g[rows][sw[rows]].sum())


# ------------------------------------------------------------------------------------------------------------- main
def load_read(read, p):
    d = os.path.join(read, sc.p_tag(p))
    parts = sorted(glob.glob(os.path.join(d, "part_*.npz")))
    keys = ("obs", "chunk", "c", "sdec", "z", "bdec")
    arr = {k: [] for k in keys}
    files = {}
    seen = []
    for f in parts:
        with np.load(f) as z:
            m = json.loads(str(z["meta"]))
            for k in keys:
                arr[k].append(z[k])
        seen.append(m["chunks"])
        files[os.path.relpath(f, read)] = sha256_file(f)
    out = {k: np.concatenate(v) for k, v in arr.items()}
    return out, files, seen


def analyse_p(read, p, tau1, out_dir):
    t0 = time.time()
    x, files, seen = load_read(read, p)
    n = len(x["obs"])
    obs, chunk, c32, z32 = x["obs"].astype(bool), x["chunk"].astype(np.int64), x["c"], x["z"]
    nch = n // sc.CHUNK
    checks = {"shots": n, "chunks_0_to_n": bool(np.array_equal(np.unique(chunk), np.arange(nch))
                                                and np.array_equal(np.bincount(chunk), np.full(nch, sc.CHUNK))),
              "parts_contiguous": [s for s, _ in seen] == sorted(s for s, _ in seen)
              and all(seen[i][1] == seen[i + 1][0] for i in range(len(seen) - 1)) and (not seen or seen[0][0] == 0),
              "finite": bool(np.isfinite(c32).all() and np.isfinite(z32).all()),
              "S_decision_is_c_gt_0": bool(np.array_equal(x["sdec"], (c32 > 0).astype(np.uint8))),
              "B_decision_is_z_gt_0": bool(np.array_equal(x["bdec"], (z32 > 0).astype(np.uint8))),
              "full_block": n == SHOTS[p]}
    if not all(v for k, v in checks.items() if isinstance(v, bool) and k != "full_block"):
        raise SystemExit(f"p {p}: input checks failed {checks}")
    c, z = c32.astype(np.float64), z32.astype(np.float64)
    S, B = x["sdec"].astype(bool), x["bdec"].astype(bool)
    fS, fB = S != obs, B != obs
    routed = sc.routed(c32, tau1)
    differ = S != B
    comp = np.where(routed, fB, fS)
    fit = chunk % 2 == 0
    ev = ~fit
    rec = {"p": p, "tau1": tau1, "shots": n, "fit_shots": int(fit.sum()), "eval_shots": int(ev.sum()),
           "input_checks": checks, "files": files}
    rec["counts"] = {"all": cats(fB, fS, routed, differ, comp), "fit": cats(fB[fit], fS[fit], routed[fit],
                                                                             differ[fit], comp[fit]),
                     "eval": cats(fB[ev], fS[ev], routed[ev], differ[ev], comp[ev])}
    a_ = rec["counts"]["all"]
    mine = {"C2": a_["C"], "C5": a_["composition"], "only_pair": a_["kept_breaks"], "only_C2": a_["kept_repairs"]}
    rec["vs_reference"] = {}
    for src, ref in REF.items():
        rr = ref[p]
        rec["vs_reference"][src] = {k: {"arb": mine[k], "ref": rr[k], "arb_per_1e6": 1e6 * mine[k] / n,
                                        "ref_per_1e6": 1e6 * rr[k] / rr["shots"],
                                        "z": two_rate_z(mine[k], n, rr[k], rr["shots"])} for k in mine}
        if "routed" in rr:
            rec["vs_reference"][src]["routed_share"] = {"arb": a_["routed_share"], "ref": rr["routed"]}
    # ---- the disagreement rows
    d = np.nonzero(routed & differ)[0]
    assert not (fB[d] & fS[d]).any() and (fB[d] | fS[d]).all(), "one bit: exactly one decoder fails on such a row"
    g = np.where(fB[d], 1, -1).astype(np.int64)
    cd, zd = c[d], z[d]
    ac, az = np.abs(cd), np.abs(zd)
    h = fit[d]
    oracle_rep = rec["counts"]["eval"]["oracle_repairs"]
    comp_fit = rec["counts"]["fit"]["composition"]
    fits, sw = {}, {}
    tj = fit_j(cd[h], zd[h], g[h])
    sw["J"] = j_switch(cd, zd, tj["w"])
    gj = gain_of(sw["J"], g, h)
    assert gj == tj["gain"], ("J: the exact rule's fit-half gain differs from the sweep's", gj, tj)
    fits["J"] = tj
    tp = fit_p(az[h], ac[h], g[h])
    fits["P"], sw["P"] = tp, p_switch(az, ac, tp)
    tt = fit_t(az[h], ac[h], g[h])
    fits["T"], sw["T"] = tt, t_switch(az, ac, tt)
    rec["rules"] = {}
    for nm in RULES:
        fg = gain_of(sw[nm], g, h)
        assert fg == fits[nm]["gain"], (nm, fg, fits[nm])
        fr = sw[nm] & h
        rec["rules"][nm] = {"params": {k: v for k, v in fits[nm].items() if k != "gain"},
                            "fit": {"gain": fg, "switches": int(fr.sum()), "repairs": int((fr & fB[d]).sum()),
                                    "breaks": int((fr & fS[d]).sum()),
                                    "rel_reduction": fg / comp_fit if comp_fit else 0.0},
                            "eval": evaluate(sw[nm], d, fB, fS, comp, ev, oracle_rep)}
    rec["reference"] = {"always_network_on_disagreement": evaluate(np.ones(len(d), bool), d, fB, fS, comp, ev,
                                                                   oracle_rep),
                        "routed_oracle": evaluate(fB[d].copy(), d, fB, fS, comp, ev, oracle_rep)}
    ws = [0.0] + GRID + [math.inf]
    curve = {"w": ws, "fit_gain": [], "eval_gain": []}
    for w in ws:
        s_ = j_switch(cd, zd, w)
        curve["fit_gain"].append(gain_of(s_, g, h))
        curve["eval_gain"].append(gain_of(s_, g, ~h))
    rec["j_curve"] = curve
    te = fit_j(cd[~h], zd[~h], g[~h])
    rec["j_eval_own_best"] = {"w": te["w"], "step": te["step"], "eval_gain": te["gain"],
                              "fitted_w_eval_gain": rec["rules"]["J"]["eval"]["net"]}
    rec["disagreement"] = {"rows": int(len(d)), "fit_rows": int(h.sum()), "eval_rows": int((~h).sum()),
                           "fit_B_wrong": int((g[h] == 1).sum()), "fit_S_wrong": int((g[h] == -1).sum()),
                           "eval_B_wrong": int((g[~h] == 1).sum()), "eval_S_wrong": int((g[~h] == -1).sum())}
    np.savez_compressed(os.path.join(out_dir, f"disagree_{sc.p_tag(p)}.npz"), rows=d, chunk=chunk[d].astype(np.int32),
                        fit=h, S=S[d], B=B[d], obs=obs[d], gain=g.astype(np.int8), c=c32[d], z=z32[d],
                        **{f"switch_{k}": v for k, v in sw.items()})
    rec["secs"] = round(time.time() - t0, 1)
    rec["_cache"] = {"cd": cd, "zd": zd, "g": g, "h": h, "d": d, "fB": fB, "fS": fS, "comp": comp, "ev": ev}
    return rec


def choose(per_p):
    score = {nm: float(np.mean([r["rules"][nm]["fit"]["rel_reduction"] for r in per_p.values()])) for nm in RULES}
    best = sorted(RULES, key=lambda nm: (-score[nm], NPAR[nm], RULES.index(nm)))[0]
    return {"rule": "largest mean over p of the fit-half relative failure reduction; ties: fewer parameters, then "
                    "J, P, T; none if not positive",
            "fit_scores": score, "carried_forward": best if score[best] > 0 else None}


def common_w(per_p):
    """Diagnostic: one w for both p, the grid value (or a per-p fitted value) with the largest mean fit-half relative
    reduction; its evaluation-half result at each p."""
    cands = sorted(set([0.0] + GRID + [r["rules"]["J"]["params"]["w"] for r in per_p.values()]))
    best = None
    for w in cands:
        m = np.mean([gain_of(j_switch(r["_cache"]["cd"], r["_cache"]["zd"], w), r["_cache"]["g"], r["_cache"]["h"])
                     / r["counts"]["fit"]["composition"] if r["counts"]["fit"]["composition"] else 0.0
                     for r in per_p.values()])                  # 0 without fit-half failures, as rel_reduction
        if best is None or m > best[1] or (m == best[1] and w > best[0]):
            best = (w, float(m))
    out = {"w": best[0], "fit_mean_rel_reduction": best[1], "per_p": {}}
    for p, r in per_p.items():
        cc = r["_cache"]
        s_ = j_switch(cc["cd"], cc["zd"], best[0])
        e = evaluate(s_, cc["d"], cc["fB"], cc["fS"], cc["comp"], cc["ev"], r["counts"]["eval"]["oracle_repairs"])
        out["per_p"][str(p)] = {"fit_gain": gain_of(s_, cc["g"], cc["h"]), "eval": e}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", default="0.006,0.0035")
    ap.add_argument("--allow-partial", action="store_true", help="dry runs: blocks shorter than the plan's")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lock = json.load(open(a.lock))
    ps = [float(x) for x in a.p.split(",")]
    per_p = {}
    for p in ps:
        tau1 = lock["p"][str(p)]["chosen"]["C5"]["tau1"]
        if tau1 != TAU1[p]:
            raise SystemExit(f"the lock's tau1 {tau1} at p = {p} is not the comparison's {TAU1[p]}")
        r = analyse_p(a.read, p, tau1, a.out)
        if not r["input_checks"]["full_block"] and not a.allow_partial:
            raise SystemExit(f"p {p}: {r['shots']} shots, not the plan's {SHOTS[p]}")
        per_p[p] = r
    summ = {"what": "ARB at d = 13 (exploratory): rules on the routed disagreement rows of the composition",
            "lock": {"file": a.lock, "sha256": sha256_file(a.lock), "tau1_C5": {str(p): TAU1[p] for p in ps}},
            "p": {}}
    if len(ps) == 2:
        summ["choice"] = choose(per_p)
        summ["j_common_w"] = common_w(per_p)
    for p, r in per_p.items():
        r.pop("_cache")
        summ["p"][str(p)] = r
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    for p, r in summ["p"].items():
        e = r["counts"]["eval"]
        print(f"p {p}: eval shots {e['shots']}, Cascade alone {e['C']}, network alone {e['S']}, composition "
              f"{e['composition']}, routed oracle {e['oracle_fail']}; disagreement rows {r['disagreement']}", flush=True)
        for nm in RULES:
            q = r["rules"][nm]
            print(f"  {nm} {q['params']}: fit gain {q['fit']['gain']} ({100 * q['fit']['rel_reduction']:.2f} %); "
                  f"eval failures {q['eval']['failures']}, repairs {q['eval']['repairs']}, breaks {q['eval']['breaks']}"
                  f", net {q['eval']['net']} ({100 * q['eval']['rel_reduction']:.2f} %), McNemar p "
                  f"{q['eval']['mcnemar_p']:.3g}", flush=True)
    if "choice" in summ:
        print("choice:", json.dumps(summ["choice"]), flush=True)
        print("one w for both p:", json.dumps({k: v for k, v in summ["j_common_w"].items() if k != "per_p"}), flush=True)


if __name__ == "__main__":
    main()
