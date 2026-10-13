"""A1's stage barrier before any test read (revision/A1_POD_RUN.md, gate finding 3). The reads of the calibration and
test shots (a1.read --many) take their job list from here only, and it is written only when everything they need is in
place, so no test shot is read for a stage that may still change:
  1. the continuation rule has been applied (a1.rule on the stage-1 validation records): DECISIONS/rule_<code>.json
     with decision CONTINUE (stage 2 done: every arm of the code, seeds 1 and 2) or STOP (the stage-1 arms of seed 1
     only, with seed 1's ramp reruns if any ran); after RAMP_FIRST the rule's second application on the ramp reruns
     (DECISIONS/rule_<code>_ramp.json, a1.rule --ramp) gives CONTINUE or STOP; a missing or unknown decision, or
     RAMP_FIRST without that second result: refused;
  2. every required run is complete: <ckpt-root>/<run>/final_ema.pt, identity-checked (a1.podval.check_identity: the
     arm's head and loss weight, seed, run name, 2e8 samples completed, the seed's registered trunk) and trained
     without a noise ramp;
  3. every required run has its not-trained check, and it is the run's (run name, head, auxiliary weight, seed, noise
     ramp, taken at 2e7 samples, a trained flag); under CONTINUE each run that failed it has its ramp rerun
     a1ramp_<arm>_s<seed> complete, and so has its seed's deployed arm (the plan's ramp dispositions); under STOP
     nothing more is trained: a run that failed its check is read as it is, beside seed 1's ramp reruns if they ran
     (under RAMP_FIRST). A ramp rerun passes the same checks as an arm's run (gate finding 4 on
     revision/A1_READ_PLAN.md): check_identity under its own run name, trained with the ramp (p_ramp RAMP), and its
     own not-trained check;
  4. the reference row's network (REFS) is the registered one before any shot loads: the file exists, its sha256 and
     run name are the registered ones (REF_REG), and it completed its 2e8 samples. A reference has no A1 not-trained
     check (an earlier study trained and registered it); a missing or other reference is refused, not left pending;
  5. completed reads are kept: a read whose output exists, is of the same checkpoint (sha256) and arm and seed on the
     same set (kind and chunks), and passed its batch-one check is not repeated (a restart reads only what is missing).
Writes --out (the pending jobs, a1.read --many's format) and --out's .preflight.json (what was checked); exit 1 with
the list of what is missing otherwise.
    python -m a1.preflight --code surface --decisions DIR --ckpt-root DIR --reads DIR --out FILE"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from a1 import podval

ARMS = {"surface": ("S-VA", "S-V", "S-L", "S-P", "S-PA"), "bb72": ("B-JT", "B-J", "B-P", "B-PA")}
STAGE1 = {"surface": ("S-VA", "S-V", "S-P", "S-PA"), "bb72": ("B-JT", "B-J", "B-P", "B-PA")}
DEPLOYED = {"surface": "S-VA", "bb72": "B-JT"}
SETS = {"surface": {"cal": ("cal", "0-19"), "test": ("test", "0-99")},
        "bb72": {"cal": ("a1cal", "0-19"), "test": ("a1test", "0-49")}}
# --sets dry: the dry runs' sets (a1/jobs/args/dry_*.txt) in both roles, for a dry run of this path that reads no
# calibration or test shot (a1/jobs/args/reads_dry_pods.txt)
SETS_DRY = {"surface": {"cal": ("dry", "0-3"), "test": ("dry", "4-9")},
            "bb72": {"cal": ("a1dry", "0-3"), "test": ("a1dry", "4-9")}}
REFS = {"surface": ("r764", "<cluster-scratch>/ising_arch/ckpt/d9_b3t_r764/final_ema.pt"),
        "bb72": ("r12", "<cluster-scratch>/bb/ckpt/r12/final_ema.pt")}
# the references' registered networks: (run name, sha256 of final_ema.pt, where it is registered)
REF_REG = {"surface": ("d9_b3t_r764", "462ffb294ab2022bc1d4b276227a62270e6e4dde4fab186958c11881fcfffb40",
                       "interp/common.py MODELS[9]; data/study2/candidates/selection.json"),
           "bb72": ("r12", "f2821c327b4e233a4dd1a75c465ceb261448d981c2b55166862fc34a99ec3e96",
                    "ising-bb: data/bb/policies/bb72_p0.003_rep2.json net_sha256")}
RAMP = "0.001:0.25"                 # the ramp reruns' --p-ramp (a1/jobs/args/train_surface.txt, train_bb72.txt)


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def train_check(code, run_dir):
    if code == "surface":
        f = os.path.join(run_dir, "train_check.json")
        return json.load(open(f)) if os.path.exists(f) else None
    f = os.path.join(run_dir, "log.json")
    return json.load(open(f)).get("train_check") if os.path.exists(f) else None


def check_run(code, ckpt_root, run, arm, seed):
    """-> (problems, the not-trained check): run `run` (a1_* or its ramp rerun a1ramp_*) is arm `arm`'s completed run of
    seed `seed` from the registered trunk, trained with the noise ramp exactly when it is a ramp rerun, and its
    not-trained check is its own, taken at 2e7 samples."""
    ramp = RAMP if run.startswith("a1ramp_") else ""
    ck_path = os.path.join(ckpt_root, run, "final_ema.pt")
    if not os.path.exists(ck_path):
        return [f"{run}: the ramp rerun its seed needs is not complete" if ramp else f"{run}: no final_ema.pt"], None
    try:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        podval.check_identity(code, ck, run, arm, seed)
    except SystemExit as e:
        return [str(e)], None
    problems = []
    if (ck["hp"].get("p_ramp") or "") != ramp:
        problems.append(f"{run}: trained with p_ramp {ck['hp'].get('p_ramp')!r}, not {ramp!r}")
    tc = train_check(code, os.path.join(ckpt_root, run))
    if tc is None:
        return problems + [f"{run}: no not-trained check"], None
    _, head, w = podval.ARMS[arm]
    want = {"run": run, "head": head, "line_weight" if code == "surface" else "translate_weight": w, "seed": seed,
            "p_ramp": ramp}
    got = {k: tc.get(k) for k in want}
    got["p_ramp"] = got["p_ramp"] or ""
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if not isinstance(tc.get("trained"), bool):
        bad["trained"] = (tc.get("trained"), "true or false")
    if not 1.9e7 <= float(tc.get("samples") or 0) <= 2.1e7:
        bad["samples"] = (tc.get("samples"), 2e7)
    if bad:
        problems.append(f"{run}: the not-trained check is not this run's at 2e7 samples (got, want): {bad}")
    return problems, tc


def check_ref(code):
    """-> problems: the reference row's network is the registered one (sha256, run name) and completed 2e8 samples."""
    name, path = REFS[code]
    run, sha, where = REF_REG[code]
    if not os.path.exists(path):
        return [f"reference {name}: no {path}"]
    got = sha_file(path)
    if got != sha:
        return [f"reference {name}: {path} has sha256 {got}, not the registered {sha} ({where})"]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    done = ck.get("row", {}).get("completed") if code == "surface" else ck.get("completed")
    facts = {"run": ck["hp"].get("run"), "samples": float(ck["hp"].get("samples") or 0), "completed": done is True}
    bad = {k: (facts[k], v) for k, v in {"run": run, "samples": 2e8, "completed": True}.items() if facts[k] != v}
    return [f"reference {name}: not the registered completed run (got, want): {bad}"] if bad else []


def done_read(out, ckpt_sha, arm, seed, kind, chunks):
    """The read at `out` exists, is of this checkpoint, arm and seed on this set, and passed its batch-one check."""
    if not os.path.exists(out):
        return False
    meta = json.loads(str(np.load(out)["meta"]))
    return (meta.get("ckpt_sha256") == ckpt_sha and meta.get("arm") == arm and meta.get("seed") == seed
            and (meta.get("kind"), meta.get("chunks")) == (kind, chunks) and meta["b1_check"]["decisions_differ"] == 0)


def plan(code, decisions, ckpt_root, reads, refs=True, sets=None):
    """-> (pending jobs, record of the checks, problems)."""
    sets = sets or SETS[code]
    problems, rec = [], {"code": code}
    f = os.path.join(decisions, f"rule_{code}.json")
    dec = json.load(open(f))["decision"] if os.path.exists(f) else None
    if dec == "RAMP_FIRST":                      # the second application, on seed 1's ramp reruns
        f = os.path.join(decisions, f"rule_{code}_ramp.json")
        dec = json.load(open(f))["decision"] if os.path.exists(f) else "RAMP_FIRST"
    rec["decision"], rec["decision_file"] = dec, f
    if dec not in ("CONTINUE", "STOP"):
        return [], rec, [f"{f}: decision {dec!r}: the reads wait for CONTINUE (stage 2 done) or STOP"]
    arms, seeds = (ARMS[code], (1, 2)) if dec == "CONTINUE" else (STAGE1[code], (1,))
    runs = []
    for arm in arms:
        for s in seeds:
            runs.append((f"a1_{arm}_s{s}", arm, s))
    checks, need_ramp = {}, []
    for run, arm, s in runs:
        probs, tc = check_run(code, ckpt_root, run, arm, s)
        problems += probs
        if tc is None:
            continue
        checks[run] = tc
        if not tc.get("trained") and (dec == "CONTINUE" or s == 1):
            need_ramp += [(arm, s), (DEPLOYED[code], s)]     # STOP: seed 1's reruns, which ran under RAMP_FIRST
    if dec == "STOP":                            # ... are read only if they ran (a plain STOP trains nothing more)
        need_ramp = [(a_, s_) for a_, s_ in need_ramp
                     if os.path.exists(os.path.join(ckpt_root, f"a1ramp_{a_}_s{s_}", "final_ema.pt"))]
    ramps = []
    for arm, s in sorted(set(need_ramp)):
        run = f"a1ramp_{arm}_s{s}"
        probs, tc = check_run(code, ckpt_root, run, arm, s)
        problems += probs
        if tc is not None:
            checks[run] = tc
        if not probs:
            ramps.append((run, arm, s))
    if refs:
        problems += check_ref(code)
    rec.update({"runs": [r[0] for r in runs], "train_checks": checks, "ramp_runs": [r[0] for r in ramps],
                "ref": dict(zip(("run", "sha256", "registered"), REF_REG[code])) if refs else None})
    if problems:
        return [], rec, problems
    jobs, kept = [], []
    for run, arm, s in runs + ramps:
        ck_path = os.path.join(ckpt_root, run, "final_ema.pt")
        csha = sha_file(ck_path)
        pre = "ramp_" if run.startswith("a1ramp_") else ""
        for role, (kind, ch) in sets.items():
            out = os.path.join(reads, f"{pre}{arm}_s{s}_{role}.npz")
            j = dict(ckpt=ck_path, arm=arm, seed=s, kind=kind, chunks=ch, out=out)
            if not pre:
                j["export"] = os.path.join(reads, f"{arm}_s{s}_decision.pt")
            (kept if done_read(out, csha, arm, s, kind, ch) else jobs).append(j)
    if refs:
        name, path = REFS[code]
        for role, (kind, ch) in sets.items():
            out = os.path.join(reads, f"ref_{name}_{role}.npz")
            j = dict(ckpt=path, arm=name, seed=0, kind=kind, chunks=ch, out=out)
            (kept if done_read(out, REF_REG[code][1], name, 0, kind, ch) else jobs).append(j)
    rec.update({"pending": [j["out"] for j in jobs], "kept": [j["out"] for j in kept]})
    return jobs, rec, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=sorted(ARMS), required=True)
    ap.add_argument("--decisions", required=True, help="the directory of a1.rule's rule_<code>.json")
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-refs", action="store_true", help="leave out the reference rows (dry runs)")
    ap.add_argument("--sets", choices=["main", "dry"], default="main", help="dry: the dry sets in both roles")
    a = ap.parse_args()
    jobs, rec, problems = plan(a.code, a.decisions, a.ckpt_root, a.reads, not a.no_refs,
                               SETS_DRY[a.code] if a.sets == "dry" else None)
    rec["sets"] = a.sets
    rec["problems"] = problems
    with open(a.out + ".preflight.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    if problems:
        raise SystemExit("PREFLIGHT REFUSED: no test read until these are resolved:\n  " + "\n  ".join(problems))
    with open(a.out, "w") as fh:
        json.dump(jobs, fh, indent=1)
    print(f"PREFLIGHT OK {a.code}: decision {rec['decision']}, {len(rec['runs'])} runs, {len(rec['ramp_runs'])} ramp "
          f"runs, {len(jobs)} reads pending, {len(rec['kept'])} kept", flush=True)


if __name__ == "__main__":
    main()
