"""Exploratory and post hoc (EXPLORE_DEEP.md): S -> B on phase C's confirmation blocks (seed kind 11) with the thresholds
the locks computed on the development block. Nothing here belongs to phase C's registered analysis or enters a verdict.

A configuration is S -> B with B = the setting's Cascade: S decides, and the shots whose S margin is below tau1 go to B,
which decides them. tau1 = pc_route's q1-quantile of S's development margin, taken from the lock (its candidates, or
its configs) and never refitted on the confirmation block. Which configurations:
  lock1  the first lock's S -> B pick at every p of both settings (1e8d1c6d, superseded by Amendment 1, never read on
         the confirmation block by phase C)
  grid   at 0.2 % and 0.3 %: every small network of the setting at every q1 of pc_common.Q1
  lock2  Amendment 1's S -> B picks (629ca1c4), recomputed only to check this script against confirm_<setting>.json
For every configuration, on the development block (kind 10, the lock's read) and on the confirmation block (kind 11,
phase C's read), the script reports the failures of the pair, of Cascade alone and of S alone on the same shots, the
paired counts against Cascade, and bb.route.paired (exact McNemar, Newcombe's 95 % interval, as in pc_route.confirm).
On the shots S keeps it counts the repairs (Cascade fails, S does not) and the breaks (S fails, Cascade does not). Every
shot on which the pair and Cascade differ is a kept shot, so only_pair = breaks and only_B = repairs (checked per row).

  python explore_deep.py --lock1 DIR --lock2 DIR --confirm DIR --out OUT.json --tables OUT.md [--procs 16]
DIR: the locked_<setting>.json of the first lock and of Amendment 1, and the registered confirm_<setting>.json. The
reads (each lock's locked["dev"] and pc_common.W4/out/conf/<setting>) are only read: each npz is opened once and only
the arrays used are loaded."""
import argparse
import json
import math
import os
import time
from multiprocessing import get_context

import numpy as np
from scipy.stats import binomtest

import pc_common as pc
from bb.route import paired

CLAIM = (0.002, 0.003)                                   # the p of the q1 grids (phase C's claim points)
CONF = os.path.join(pc.W4, "out", "conf")
LOCKKEYS = ("family", "B", "dev", "nets", "read_prec", "gpu", "torch", "conf_chunks")
NAME = {"S55p": "S55'", "S55p_deep": "S55'-deep", "C14p": "C14'", "S55": "S55", "C14": "C14"}


def sb_taus(lock):
    """(p, S, q1) -> the S -> B candidate of a lock (pc_route.dev_taus on its development read)."""
    return {(c["p"], c["S"], c["q1"]): c for c in lock["candidates"] if c["type"] == "S-B"}


def plan(L1, L2):
    """setting -> {(p, S, q1): {labels, tau1}}; both locks rest on one development read, so their candidates'
    thresholds must agree."""
    out = {}
    for s in pc.SETTINGS:
        if any(L1[s][k] != L2[s][k] for k in LOCKKEYS):
            raise SystemExit(f"{s}: the two locks differ in {[k for k in LOCKKEYS if L1[s][k] != L2[s][k]]}")
        c1, c2 = sb_taus(L1[s]), sb_taus(L2[s])
        if {k: v["tau"] for k, v in c1.items()} != {k: v["tau"] for k, v in c2.items()}:
            raise SystemExit(f"{s}: the locks' S -> B thresholds differ")
        want = {}
        for lab, L in (("lock1", L1), ("lock2", L2)):
            for c in L[s]["configs"]:
                if c["type"] == "S-B":
                    k = (c["p"], c["S"], c["q1"])
                    if c["tau"] != c1[k]["tau"]:
                        raise SystemExit(f"{s} {k}: the config's threshold is not its candidate's")
                    want.setdefault(k, set()).add(lab)
        for p in CLAIM:
            for S in pc.SMALL[s]:
                for q1 in pc.Q1:
                    want.setdefault((p, S, q1), set()).add("grid")
        out[s] = {k: {"labels": sorted(v), "tau1": c1[k]["tau"]["tau1"]} for k, v in want.items()}
    return out


def block(task):
    """One read (setting, block, p): every planned configuration on it (a failed check raises RuntimeError: a
    SystemExit would stop the pool's worker and leave the pool waiting)."""
    try:
        return _block(*task)
    except SystemExit as e:
        raise RuntimeError(str(e)) from None


def _block(s, blk, d, p, cfgs, lk):
    t0 = time.time()
    z, meta, idx = pc.load_read(d, p)
    B = lk["B"]
    if meta["family"] != lk["family"]:
        raise SystemExit(f"{d} at {p}: family {meta['family']}, locked {lk['family']}")
    for t in {B} | {S for S, _ in cfgs}:
        m = meta["models"][idx[t]]
        if (m["sha256"], m["prec"], m["gpu"], m["torch"]) != (lk["nets"][t], lk["read_prec"][t], lk["gpu"], lk["torch"]):
            raise SystemExit(f"{d} at {p}: {t} is not the locked network, precision, card or torch")
    fb = z[f"fail{idx[B]}"].astype(bool)
    n = len(fb)
    if n != (lk["conf_chunks"][str(p)] if blk == "conf" else pc.DEV_CHUNKS[p]) * pc.CHUNK:
        raise SystemExit(f"{d} at {p}: {n} shots, not the block's")
    rows = []
    for (S, q1), c in sorted(cfgs.items()):
        cs = pc.confidence(z, meta, idx, S)               # float64 margins, as pc_route
        fs = z[f"fail{idx[S]}"].astype(bool)
        toB = cs < c["tau1"]
        keep = ~toB
        fd = np.where(toB, fb, fs)                        # pc_route.config_fail for S -> B
        st = paired(fd, fb, 0.05)
        rep, brk = int((fb & ~fs & keep).sum()), int((~fb & fs & keep).sum())
        r = {"setting": s, "block": blk, "p": p, "S": S, "q1": q1, "shots": n, "routed": int(toB.sum()),
             "to_B": float(toB.mean()), "fail": int(fd.sum()), "B_fail": int(fb.sum()), "S_fail": int(fs.sum()),
             "both": int((fd & fb).sum()), "only_pair": st["only_a"], "only_B": st["only_b"],
             "mcnemar_p": st["mcnemar_p"], "diff": st["diff_ler"], "diff_lo": st["diff_lo"], "diff_hi": st["diff_hi"],
             "verdict_95": st["verdict"], "kept_repairs": rep, "kept_breaks": brk,
             "kept_both": int((fb & fs & keep).sum()), "S_only": int((fs & ~fb).sum()), "B_only": int((~fs & fb).sum())}
        r["routed_breaks"], r["routed_repairs"] = r["S_only"] - brk, r["B_only"] - rep
        if not (r["only_pair"] == brk and r["only_B"] == rep and r["fail"] - r["B_fail"] == brk - rep):
            raise SystemExit(f"{s} {blk} {p} {S} {q1}: the identity fails")
        if blk == "dev":                                  # pc_route.dev_taus, recomputed on the development margins
            r["tau1_refit"] = float(np.quantile(cs, q1))
        rows.append(r)
    return {"rows": rows, "secs": round(time.time() - t0, 2), "setting": s, "block": blk, "p": p, "shots": n}


def need(r, b, fB, pool, alpha=0.05):
    """Kept-shot changes that would let the pair beat Cascade on a block of this size, from r repairs and b breaks on the
    kept shots, Cascade's fB failures and the pool of kept shots both fail (where new repairs would come from):
      parity  net gain (added repairs + removed breaks) for strictly fewer failures than Cascade
      add     repairs to add, breaks fixed, for the exact McNemar test at two-sided alpha (None: not within the pool)
      remove  breaks to remove, repairs fixed, for the same (None: not reachable with r repairs)
      ten     net gain for 10 % fewer failures than Cascade (pc_common.DELTA_REL, the sizing's alternative)"""
    def sig(x, y):
        return x > y and binomtest(x, x + y, 0.5).pvalue < alpha
    add = next((a for a in range(max(0, b - r + 1), pool + 1) if sig(r + a, b)), None)
    rem = next((x for x in range(max(0, b - r + 1), b + 1) if sig(r, b - x)), None)
    return {"alpha": alpha, "parity": max(0, b - r + 1), "add_repairs": add, "remove_breaks": rem,
            "ten_pct": max(0, math.ceil(pc.DELTA_REL * fB - 1e-9) - (r - b)), "pool_kept_both": pool}


def validate(cfg, res, L1, L2, reg):
    """The script against phase C: thresholds refitted on the development margins equal the locks'; development counts
    equal the locks' records; Amendment 1's S -> B rows on the confirmation block equal confirm_<setting>.json."""
    bad, n = [], 0

    def chk(what, got, want):
        nonlocal n
        n += 1
        if got != want:
            bad.append({"check": what, "got": got, "want": want})

    for s in pc.SETTINGS:
        cand = sb_taus(L2[s])
        for (p, S, q1), c in cfg[s].items():
            dv, cf = res[(s, p, S, q1)]["dev"], res[(s, p, S, q1)]["conf"]
            k = f"{s} {p} {S} {q1}"
            chk(f"{k}: tau1 refitted on dev", dv["tau1_refit"], c["tau1"])
            chk(f"{k}: dev failures", dv["fail"], cand[(p, S, q1)]["fail"])
            chk(f"{k}: dev routed fraction", dv["to_B"], cand[(p, S, q1)]["to_B"])
            for lab, L in (("lock1", L1), ("lock2", L2)):
                for lc in L[s]["configs"]:
                    if lc["type"] == "S-B" and (lc["p"], lc["S"], lc["q1"]) == (p, S, q1):
                        chk(f"{k}: {lab} dev vs_B", [dv["both"], dv["only_pair"], dv["only_B"], dv["shots"]],
                            [lc["vs_B"][x] for x in ("both", "only_D", "only_B", "shots")])
            if "lock2" in c["labels"]:
                rr = [x for x in reg[s]["rows"] if x["type"] == "S-B" and (x["p"], x["S"], x["q1"]) == (p, S, q1)]
                chk(f"{k}: one registered row", len(rr), 1)
                if len(rr) == 1:
                    x, v = rr[0], rr[0]["vs_B_95"]
                    chk(f"{k}: conf vs registered",
                        [cf["fail"], cf["shots"], cf["B_fail"], cf["S_fail"], cf["to_B"], cf["kept_repairs"],
                         cf["kept_breaks"], cf["only_pair"], cf["only_B"], cf["mcnemar_p"], cf["diff"], cf["diff_lo"],
                         cf["diff_hi"]],
                        [x["fail"], x["shots"], x["B_alone"]["fail"], x["S_alone"]["fail"], x["to_B"],
                         x["kept_repairs_vs_B"], x["kept_regressions_vs_B"], v["only_a"], v["only_b"], v["mcnemar_p"],
                         v["diff_ler"], v["diff_lo"], v["diff_hi"]])
    return {"checks": n, "failed": len(bad), "bad": bad}


# tables (Markdown) ------------------------------------------------------------------------------------------------
def pct(x):
    return f"{100 * x:g} %"


def num(x):
    return f"{x:,}"


def pval(p):
    if p == 0:
        return "0 (underflow)"
    if p >= 0.1:
        return "1.0" if p >= 0.995 else f"{p:.2f}"
    return f"{p:.1e}"


def diff(c):
    d = "0.0" if c["diff"] == 0 else f"{c['diff']:.2e}"
    return f"{d} [{c['diff_lo']:.1e}, {c['diff_hi']:.1e}]"


def rel(c):
    return f"{100 * (c['fail'] / c['B_fail'] - 1):+.1f} %"


def mark(c):
    return ("†" if "lock1" in c["labels"] else "") + ("‡" if "lock2" in c["labels"] else "")


def tables(out):
    rows, reg = out["configs"], out["registered_sb"]
    L = ["# Exploratory, post hoc: S -> B on the confirmation blocks with the development thresholds", "",
         f"Generated by explore_deep.py (job {out['meta']['job']}, {out['meta']['secs']} s). † the first lock's pick "
         "(1e8d1c6d), ‡ Amendment 1's pick (registered, recomputed as a check). Pair = S -> B; breaks / repairs = the "
         "kept shots on which only S fails / only Cascade fails, equal by construction to the paired counts only the "
         "pair fails / only Cascade fails. Differences are the pair's block failure rate minus Cascade's, with "
         "Newcombe's 95 % interval; McNemar p is exact and two-sided; nothing is adjusted for multiplicity.", ""]
    L += ["## The first lock's S -> B at every p", "",
          "| setting | p | S, q1 | dev: pair / Cascade | routed (conf) | conf: pair / Cascade (shots) | pair - Cascade "
          "| only pair / only Cascade (= kept breaks / repairs) | difference [95 %] | McNemar p | Amendment 1's pick: "
          "failures |", "|" + "---|" * 11]
    for c in sorted((c for c in rows if "lock1" in c["labels"]), key=lambda c: (c["setting"] != "depspam_fm", c["p"])):
        d, f = c["dev"], c["conf"]
        g = reg[c["setting"]][str(c["p"])]
        L.append(f"| `{c['setting']}` | {pct(c['p'])} | {NAME[c['S']]}, {pct(c['q1'])} | {num(d['fail'])} / "
                 f"{num(d['B_fail'])} | {100 * f['to_B']:.1f} % | {num(f['fail'])} / {num(f['B_fail'])} "
                 f"({num(f['shots'])}) | {f['fail'] - f['B_fail']:+,} ({rel(f)}) | {num(f['only_pair'])} / "
                 f"{num(f['only_B'])} | {diff(f)} | {pval(f['mcnemar_p'])} | {NAME[g['S']]}, {pct(g['q1'])}: "
                 f"{num(g['fail'])} |")
    for s in ("depspam_fm", "uniform"):
        for p in CLAIM:
            L += ["", f"## `{s}` at {pct(p)}: every small network at every q1", "",
                  "| S | q1 | tau1 | dev: pair / Cascade (only pair / only Cascade) | routed (conf) | conf: pair / "
                  "Cascade | pair - Cascade | only pair / only Cascade (= kept breaks / repairs) | kept shots both "
                  "fail | difference [95 %] | McNemar p |", "|" + "---|" * 11]
            for S in pc.SMALL[s]:
                cs = sorted((c for c in rows if (c["setting"], c["p"], c["S"]) == (s, p, S) and "grid" in c["labels"]),
                            key=lambda c: c["q1"])
                f0 = cs[0]["conf"]
                L.append(f"| {NAME[S]} alone | - | - | - | 0 % | {num(f0['S_fail'])} / {num(f0['B_fail'])} | "
                         f"{f0['S_fail'] - f0['B_fail']:+,} | {num(f0['S_only'])} / {num(f0['B_only'])} | - | - | - |")
                for c in cs:
                    d, f = c["dev"], c["conf"]
                    L.append(f"| {NAME[S]}{mark(c)} | {pct(c['q1'])} | {c['tau1']:.3f} | {num(d['fail'])} / "
                             f"{num(d['B_fail'])} ({d['only_pair']} / {d['only_B']}) | {100 * f['to_B']:.2f} % | "
                             f"{num(f['fail'])} / {num(f['B_fail'])} | {f['fail'] - f['B_fail']:+,} ({rel(f)}) | "
                             f"{num(f['only_pair'])} / {num(f['only_B'])} | {num(f['kept_both'])} | {diff(f)} | "
                             f"{pval(f['mcnemar_p'])} |")
    for p in CLAIM:
        L += ["", f"## What the network would need on the kept shots at {pct(p)} (confirmation block)", "",
              "| setting | S | q1 | kept breaks / repairs | kept shots both fail | net gain for parity | repairs to "
              "add for p < 0.05 | breaks to remove for p < 0.05 | net gain for 10 % fewer failures |", "|" + "---|" * 9]
        for c in sorted((c for c in rows if c["p"] == p and "grid" in c["labels"]),
                        key=lambda c: (c["setting"] != "depspam_fm", c["S"], c["q1"])):
            f, nd = c["conf"], c["need"]
            L.append(f"| `{c['setting']}` | {NAME[c['S']]}{mark(c)} | {pct(c['q1'])} | {num(f['kept_breaks'])} / "
                     f"{num(f['kept_repairs'])} | {num(f['kept_both'])} | {nd['parity']} | "
                     f"{'-' if nd['add_repairs'] is None else nd['add_repairs']} | "
                     f"{'-' if nd['remove_breaks'] is None else nd['remove_breaks']} | {nd['ten_pct']} |")
    v = out["validation"]
    L += ["", f"Validation: {v['checks']} checks, {v['failed']} failed."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock1", required=True)
    ap.add_argument("--lock2", required=True)
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    lp = {f"{k}_{s}": os.path.join(d, f"{'confirm' if k == 'confirm' else 'locked'}_{s}.json")
          for k, d in (("lock1", a.lock1), ("lock2", a.lock2), ("confirm", a.confirm)) for s in pc.SETTINGS}
    L1, L2, reg = ({s: json.load(open(lp[f"{k}_{s}"])) for s in pc.SETTINGS} for k in ("lock1", "lock2", "confirm"))
    for s in pc.SETTINGS:
        if reg[s]["locked_sha256"] != pc.sha256_file(lp[f"lock2_{s}"]):
            raise SystemExit(f"{s}: confirm_{s}.json was not made from this lock")
    cfg = plan(L1, L2)
    tasks = []
    for s in pc.SETTINGS:
        lk = {k: L2[s][k] for k in LOCKKEYS}
        by_p = {}
        for (p, S, q1), c in cfg[s].items():
            by_p.setdefault(p, {})[(S, q1)] = c
        for p, cs in by_p.items():
            tasks += [(s, "dev", lk["dev"], p, cs, lk), (s, "conf", os.path.join(CONF, s), p, cs, lk)]
    size = {t[:4]: (t[5]["conf_chunks"][str(t[3])] if t[1] == "conf" else pc.DEV_CHUNKS[t[3]]) for t in tasks}
    tasks.sort(key=lambda t: -size[t[:4]])                # the largest reads first
    with get_context("fork").Pool(min(a.procs, len(tasks))) as pool:
        done = pool.map(block, tasks, chunksize=1)
    res = {}
    for b in done:
        for r in b["rows"]:
            res.setdefault((r["setting"], r["p"], r["S"], r["q1"]), {})[r["block"]] = r
    configs = []
    for s in pc.SETTINGS:
        for (p, S, q1), c in sorted(cfg[s].items()):
            f = res[(s, p, S, q1)]["conf"]
            configs.append({"setting": s, "p": p, "S": S, "B": L2[s]["B"], "q1": q1, "tau1": c["tau1"],
                            "labels": c["labels"], "dev": res[(s, p, S, q1)]["dev"], "conf": f,
                            "need": need(f["kept_repairs"], f["kept_breaks"], f["B_fail"], f["kept_both"])})
    looks = [c for c in configs if {"lock1", "grid"} & set(c["labels"])]
    out = {"meta": {"what": "exploratory, post hoc; not part of phase C's registered analysis",
                    "job": os.environ.get("SLURM_JOB_ID", ""), "node": os.uname().nodename,
                    "inputs": {k: {"path": os.path.abspath(v), "sha256": pc.sha256_file(v)} for k, v in lp.items()},
                    "conf_dir": CONF, "dev_dir": {s: L2[s]["dev"] for s in pc.SETTINGS}, "claim_p": CLAIM,
                    "q1": pc.Q1, "configs": len(configs), "looks": len(looks),
                    "bonferroni_005": 0.05 / len(looks), "procs": a.procs,
                    "reads": [{k: b[k] for k in ("setting", "block", "p", "shots", "secs")} for b in done]},
           "validation": validate(cfg, res, L1, L2, reg),
           "registered_sb": {s: {str(x["p"]): {k: x[k] for k in ("S", "q1", "fail", "shots")}
                                 for x in reg[s]["rows"] if x["type"] == "S-B"} for s in pc.SETTINGS},
           "configs": configs}
    out["meta"]["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)
    open(a.tables, "w").write(tables(out))
    v = out["validation"]
    print(json.dumps({"validation": {k: v[k] for k in ("checks", "failed")}, "bad": v["bad"][:5],
                      "configs": len(configs), "looks": len(looks), "secs": out["meta"]["secs"]}))
    for c in configs:
        if c["p"] == 0.003 or "lock1" in c["labels"]:
            f = c["conf"]
            print(f"{c['setting']} {c['p']} {c['S']} q1={c['q1']} {'/'.join(c['labels'])}: pair {f['fail']} B "
                  f"{f['B_fail']} breaks {f['kept_breaks']} repairs {f['kept_repairs']} p {f['mcnemar_p']:.2g}")


if __name__ == "__main__":
    main()
