"""A1's stage-1 continuation rule (revision/A1_POD_RUN.md, section 2), frozen before any stage-1 training: does the
logical-line output head (surface: the line vote; BB72: the joint translate score) beat a pooled head of matched size
under matched supervision? Computed on the stage-1 runs' validation records (a1.podval: d = 9 val_legacy chunks 0-19 and
BB72 dev chunks 0-19, 200,000 shots each), never on test shots. Seed 1 only.

Per code, the two readout contrasts at matched supervision decide (line head L against pooled head P):
  surface  S-VA against S-PA (line loss on), S-V against S-P (no line loss)
  bb72     B-JT against B-PA (translate loss on), B-J against B-P (no translate loss)
At each routed fraction f in {0, 1, 2, 5} %, each arm routes its round(f N) least confident validation shots (confidence
ascending in float64, ties by shot index) and keeps the rest; K_f(arm) = its failures on the shots it keeps (the routed
shots count as decided correctly: an ideal fallback, since the fallbacks' validation answers are not part of stage 1).
Per contrast and f: the paired kept-failure indicators of L and P on the same shots, the exact two-sided McNemar test on
the discordant shots (a1.analyze.mcnemar), and the relative difference rel = (K_f(P) - K_f(L)) / K_f(P). Holm runs over
the code's 8 tests (2 contrasts x 4 fractions). At f the line head
  wins   when rel >= MARGIN = 3 % and the Holm-adjusted p < 0.05,
  loses  when rel <= -3 % and the Holm-adjusted p < 0.05, and otherwise ties (within the validation noise).
The test takes the shot noise; the 3 % margin is above the training-run noise seen so far: two replicas of the same d = 9
network (r710, r711, 200,000 shots) differed by 0.5 % (learned vote) and 1.2 % (mean of lines).
A contrast favours the line head when it wins at one f or more and loses at none. A contrast with an arm that failed its
not-trained check (train_check.trained false) is undecided: the plan's ramp reruns come first.
Decision per code: CONTINUE when a contrast favours the line head; otherwise RAMP_FIRST when a contrast is undecided
(stage 2 for that code starts with the seed-1 ramp reruns of the failed arms and the deployed arm, and this rule is
applied again to the ramp runs' records); otherwise STOP. The supervision contrasts (S-VA against S-V, S-PA against
S-P; B-JT against B-J, B-PA against B-P) are reported beside them with the same statistics and decide nothing.
Inputs are refused unless every record is the arm's (arm, seed, run name, completed, registered trunk: a1.podval's
identity), passed its batch-one check, read the same validation set (one digest), and every run of the code read the
same training stream (equal stream_chain). With --ckpt-dir each record's checkpoint sha256 must equal the file's.
The second application (--ramp, after RAMP_FIRST; fixed on 2026-10-12 before any ramp rerun started): the records of
seed 1's ramp reruns a1ramp_<arm>_s1 (the arms that failed their check and the deployed arm; at most one each). The
decisive contrasts are those above whose two arms both have a ramp record (surface after S-P and S-PA failed: S-VA
against S-PA), with the same statistics (Holm over their tests). A ramp contrast favours the line head when it wins at
one fraction or more and loses at none, or when its line arm passed its not-trained check and its pooled arm failed it
even with the ramp (the plan runs nothing further: the matched pooled head does not train where the line head does).
Decision: CONTINUE when a ramp contrast favours the line head, otherwise STOP.
    python -m a1.rule --records DIR [--ckpt-dir DIR] --code surface --out DIR      (records DIR/a1_<arm>_s1.npz)
    python -m a1.rule --ramp --records DIR [--ckpt-dir DIR] --code surface --out DIR   (DIR/a1ramp_<arm>_s1.npz)"""
import argparse
import hashlib
import json
import os

import numpy as np

from a1.analyze import holm, mcnemar

RULE = "A1 stage-1 continuation rule v1 (2026-10-12)"
FRACTIONS = (0.0, 0.01, 0.02, 0.05)
MARGIN = 0.03
ALPHA = 0.05
SEED = 1
DECIDE = {"surface": [("S-VA", "S-PA", "line loss on"), ("S-V", "S-P", "no line loss")],
          "bb72": [("B-JT", "B-PA", "translate loss on"), ("B-J", "B-P", "no translate loss")]}
DESCRIBE = {"surface": [("S-VA", "S-V", "vote: line loss"), ("S-PA", "S-P", "pooled: line loss")],
            "bb72": [("B-JT", "B-J", "joint: translate loss"), ("B-PA", "B-P", "pooled: translate loss")]}
DEPLOYED = {"surface": "S-VA", "bb72": "B-JT"}


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_records(code, rdir, ckpt_dir="", prefix="a1"):
    arms = sorted({x for c in DECIDE[code] for x in c[:2]})
    if prefix == "a1ramp":                   # the ramp reruns that exist (the failed arms and the deployed arm)
        arms = [x for x in arms if os.path.exists(os.path.join(rdir, f"a1ramp_{x}_s{SEED}.npz"))]
        if DEPLOYED[code] not in arms or len(arms) < 2:
            raise SystemExit(f"the {code} ramp records need the deployed arm and a failed arm: found {arms}")
    R = {}
    for arm in arms:
        run = f"{prefix}_{arm}_s{SEED}"
        z = np.load(os.path.join(rdir, run + ".npz"))
        meta = json.loads(str(z["meta"]))
        bad = [k for k, v in (("code", code), ("arm", arm), ("seed", SEED), ("run", run)) if meta.get(k) != v]
        if bad or not meta.get("identity", {}).get("completed"):
            raise SystemExit(f"{run}: not the arm's completed run ({bad or 'not completed'})")
        if meta["b1_check"]["decisions_differ"]:
            raise SystemExit(f"{run}: failed its batch-one check")
        if ckpt_dir and sha_file(os.path.join(ckpt_dir, run, "final_ema.pt")) != meta["ckpt_sha256"]:
            raise SystemExit(f"{run}: the record is not of the checkpoint in {ckpt_dir}")
        fail, conf = z["fail"].astype(bool), z["conf"].astype(np.float64)
        if conf.ndim != 1 or len(fail) != len(conf) or len(fail) != meta["set"]["shots"]:
            raise SystemExit(f"{run}: malformed record")
        R[arm] = {"fail": fail, "conf": conf, "meta": meta, "sha256": sha_file(os.path.join(rdir, run + ".npz"))}
    for key, what in ((lambda m: m["set"]["digest"], "validation set"), (lambda m: m["stream_chain"], "training stream")):
        vals = {arm: key(r["meta"]) for arm, r in R.items()}
        if len(set(vals.values())) != 1 or None in vals.values():
            raise SystemExit(f"the {code} records differ in their {what}: {vals}")
    return R


def kept_failures(fail, conf, f):
    """Failure indicators on the kept shots when the round(f N) least confident shots are routed."""
    n = int(round(f * len(conf)))
    k = fail.copy()
    if n:
        k[np.argsort(conf, kind="stable")[:n]] = False
    return k


def contrast_rows(R, pairs, decisive):
    rows = []
    for x, y, label in pairs:
        for f in FRACTIONS:
            a, b = kept_failures(R[x]["fail"], R[x]["conf"], f), kept_failures(R[y]["fail"], R[y]["conf"], f)
            n10, n01, p = mcnemar(a, b)
            kx, ky = int(a.sum()), int(b.sum())
            rows.append({"x": x, "y": y, "label": label, "decisive": decisive, "f": f, "fail_x": kx, "fail_y": ky,
                         "only_x": n10, "only_y": n01, "p": p, "rel": (ky - kx) / ky if ky else 0.0})
    return rows


def decide(code, R, ramp=False):
    pairs = [c for c in DECIDE[code] if c[0] in R and c[1] in R]
    rows = contrast_rows(R, pairs, True)
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(p)
        r["outcome"] = ("win" if r["rel"] >= MARGIN else "loss" if r["rel"] <= -MARGIN else "tie") \
            if p < ALPHA else "tie"
    trained = {arm: bool((r["meta"]["train_check"] or {}).get("trained")) for arm, r in R.items()}
    contrasts = []
    for x, y, label in pairs:
        mine = [r for r in rows if r["x"] == x and r["y"] == y]
        wins = [r["f"] for r in mine if r["outcome"] == "win"]
        losses = [r["f"] for r in mine if r["outcome"] == "loss"]
        if ramp and trained[x] and not trained[y]:
            state = "favours the line head (the pooled head does not train even with the ramp)"
        elif ramp and not trained[x]:
            state = "does not favour the line head (its line arm does not train even with the ramp)"
        elif not (trained[x] and trained[y]):
            state = "undecided (not trained: " + ", ".join(a for a in (x, y) if not trained[a]) + ")"
        else:
            state = "favours the line head" if wins and not losses else "does not favour the line head"
        contrasts.append({"x": x, "y": y, "label": label, "wins_at": wins, "losses_at": losses, "state": state})
    if any(c["state"].startswith("favours the line head") for c in contrasts):
        decision = "CONTINUE"
    elif any(c["state"].startswith("undecided") for c in contrasts):
        decision = "RAMP_FIRST"
    else:
        decision = "STOP"
    desc = contrast_rows(R, [c for c in DESCRIBE[code] if c[0] in R and c[1] in R], False)
    for r in desc:
        r["p_holm"], r["outcome"] = None, "descriptive"
    return {"rule": RULE, "application": "ramp reruns" if ramp else "stage 1", "code": code, "seed": SEED,
            "fractions": list(FRACTIONS), "margin": MARGIN, "alpha": ALPHA,
            "decision": decision, "contrasts": contrasts, "tests": rows, "descriptive": desc,
            "trained": trained, "set": R[DEPLOYED[code]]["meta"]["set"],
            "stream_chain": R[DEPLOYED[code]]["meta"]["stream_chain"],
            "records": {arm: {"sha256": r["sha256"], "ckpt_sha256": r["meta"]["ckpt_sha256"],
                              "failures": r["meta"]["failures"], "device": r["meta"]["device"]}
                        for arm, r in R.items()}}


def markdown(res):
    out = [f"# {res['rule']}, {res['application']}: {res['code']} -> **{res['decision']}**", "",
           f"Validation set {res['set']['split']} chunks {res['set']['chunks']} ({res['set']['shots']} shots, digest "
           f"{res['set']['digest'][:12]}); training stream chain {res['stream_chain'][:16]}; margin "
           f"{100 * res['margin']:.0f} %, Holm over the {len(res['tests'])} decisive tests, alpha {res['alpha']}.", "",
           "| x | y | held | f | failures x | failures y | only x | only y | rel (y - x) / y | p | p Holm | outcome |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["tests"] + res["descriptive"]:
        ph = "" if r["p_holm"] is None else f"{r['p_holm']:.3g}"
        out.append(f"| {r['x']} | {r['y']} | {r['label']} | {100 * r['f']:g} % | {r['fail_x']} | {r['fail_y']} | "
                   f"{r['only_x']} | {r['only_y']} | {100 * r['rel']:+.1f} % | {r['p']:.3g} | {ph} | {r['outcome']} |")
    out.append("")
    for c in res["contrasts"]:
        out.append(f"- {c['x']} against {c['y']} ({c['label']}): {c['state']}; wins at f = "
                   f"{[100 * f for f in c['wins_at']]} %, losses at {[100 * f for f in c['losses_at']]} %")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--code", choices=sorted(DECIDE), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ramp", action="store_true", help="the second application, on the ramp reruns' records")
    a = ap.parse_args()
    res = decide(a.code, load_records(a.code, a.records, a.ckpt_dir, "a1ramp" if a.ramp else "a1"), a.ramp)
    os.makedirs(a.out, exist_ok=True)
    name = f"rule_{a.code}" + ("_ramp" if a.ramp else "")
    with open(os.path.join(a.out, name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out, name + ".md"), "w") as fh:
        fh.write(markdown(res))
    print(f"RULE {a.code} {res['decision']}", json.dumps(res["contrasts"]), flush=True)


if __name__ == "__main__":
    main()
