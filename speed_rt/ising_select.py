"""The frozen rule that picks the Ising plan whose GPU time counts, per model and d (SPEED_RT.md, "Ising"; review findings
2 and 3). Candidates are the accepted repetitions of every validated plan in the session's counted attempts
(attempts.json): the five graph repetitions of each of the two ising_time processes (fwd, rev; speed_rt/ising_time.py)
and the trtexec repetitions (GPU Compute Time mean per
repetition; speed_rt/ising_d5.py). Ising's time = the smallest candidate value (the studies' "fastest repetition of the
fastest validated plan"); the counted plan is that candidate's plan. Ties go to the first plan in ising_time.PLANS order
(model, FP16 before FP8, level 3, level 5, official), then graph before trtexec, then fwd before rev, then the earlier
repetition. The residual detectors whose PyMatching share is timed (P3) are this plan's (ising_d5 residuals), and the
analysis uses the same function, so the pod and the analysis cannot disagree.

A pod role names its distances explicitly (S-low 7 9 11, S-high 13 15 17): with --d, only those distances are selected and every
(model, d) of the list without an accepted candidate is reported as missing (its comparison has no Ising time: no verdict).
Without --d every distance present in the session is selected and nothing is reported missing.
    python -m speed_rt.ising_select --session DIR [--d 7 9 11] [--json]
(prints, per model and d, the counted plan and value; --json: also the "missing" list)"""
import argparse
import json
import os

KINDS = ("L3", "L5", "official")
ALLOWED_D = (7, 9, 11, 13, 15, 17)                       # the surface distances; a pod role's list is a subset


def plan_key(model, prec, kind):
    return (("M1", "M4").index(model), ("fp16", "fp8").index(prec), KINDS.index(kind))


def counted(session, item):
    """The record files of SESSION's counted attempts of ITEM (attempts.json: accepted by the watch and the module; a
    replaced attempt never counts), in attempt order."""
    att = json.load(open(os.path.join(session, "attempts.json")))
    return [os.path.join(session, x["record"]) for x in att if x["item"] == item and x["accepted"] and x["record"]]


def candidates(session):
    """[(value_us, tie-break key, model, d, plan file name, source)] over the accepted repetitions of the counted
    attempts in SESSION."""
    out = []
    for f in counted(session, "G5-graph"):
        r = json.load(open(f))
        for row in r["rows"]:
            if not row.get("accepted"):
                continue
            for k, v in enumerate(row["us_call_reps"]):
                key = plan_key(row["model"], row["precision"], row["kind"]) + (0, ("fwd", "rev").index(r["order"]), k)
                out.append((float(v), key, row["model"], r["d"], row["plan"], f"graph {r['order']} rep {k + 1}"))
    for f in counted(session, "G5-trtexec"):
        r = json.load(open(f))
        if not r.get("accepted"):
            continue
        kind = "official" if r["build"] == "official" else f"L{r['level']}"
        key = plan_key(r["model"], r["precision"], kind) + (1, 0, r["rep"])
        out.append((float(r["us_call_mean"]), key, r["model"], r["d"], r["plan"].rsplit(".", 1)[0] + ".plan",
                    f"trtexec rep {r['rep']}"))
    return out


def select(session, ds=None):
    """{(model, d): dict(value_us, plan, source, candidates)}; a (model, d) without an accepted candidate is absent. DS
    (a pod role's explicit list) keeps only those distances."""
    best = {}
    for c in candidates(session):
        if ds is not None and c[3] not in ds:
            continue
        k = (c[2], c[3])
        if k not in best or (c[0], c[1]) < (best[k][0], best[k][1]):
            best[k] = c
    n = {}
    for c in candidates(session):
        if ds is None or c[3] in ds:
            n[(c[2], c[3])] = n.get((c[2], c[3]), 0) + 1
    return {k: dict(value_us=v[0], plan=v[4], source=v[5], candidates=n[k]) for k, v in sorted(best.items())}


def missing(sel, ds):
    """["M1 d7", ...]: the (model, d) of the explicit list DS that have no counted plan in SEL."""
    return [f"{m} d{d}" for d in ds for m in ("M1", "M4") if (m, d) not in sel]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--d", type=int, nargs="+", default=None, choices=ALLOWED_D, help="the pod role's explicit list")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    s = select(a.session, a.d)
    miss = missing(s, a.d) if a.d else []
    if a.json:
        print(json.dumps(dict({f"{m} d{d}": v for (m, d), v in s.items()}, **({"missing": miss} if a.d else {})),
                         indent=1, sort_keys=True))
        return
    for (m, d), v in s.items():
        print(f"{m} d{d} {v['plan']} {v['value_us']:.3f} us ({v['source']}; {v['candidates']} candidates)")
    if miss:
        print("missing (no accepted candidate; the comparison has no Ising time):", ", ".join(miss))


if __name__ == "__main__":
    main()
