"""The rule, the lock, the one confirmation analysis and the report (cascade_compare/SURFACE_PLAN.md 4.3-4.5 and 7, with
item 2 of the independent check: "better" and "worse" are an exact-McNemar rejection by Holm plus the sign of the
discordant counts).

  select   python -m surface.sv_route select --dev DIR --speed speed.json --out locked_d13.json
  gate     python -m surface.sv_route gate --lock locked_d13.json --check bitwise_gate.json --out gate_d13.json
  confirm  python -m surface.sv_route confirm --lock locked_d13.json --gate gate_d13.json --conf DIR --out confirm.json
  report   python -m surface.sv_route report --lock L --confirm C --speed S --pipes rows.jsonl --bitwise B --out MD

select (4.3, on the development block of each p: SB, Bb, MS of kind 10). delta_p = 10 % of C2's (Bb's) block failure
rate. C5 candidates: tau1(q1) = numpy.quantile(|c|, q1) (linear, in float64) for q1 in Q1. C1 candidates: tau1(q1) and
for each r in R2 below q1, tau2 = the quantile of |z| over the shots tau1 routes at which a fraction r of all shots
reaches M. Routed fractions f_B, f_M are measured (ties count as routed); t_M(X) is MS's mean decode time over the shots
X routes to M (the development read timed each decode on the node's CPU). Predicted cost from the screen's components
(the slowest process): L1 = t1(S) + f_B t1(B) + f_M t_M(X); A = a(S) + f_B a(B) + f_M t_M(X) / 16, a(S) at 4,096 and
a(B) at b*. Feasible: L1 <= 0.9 t1(B) and A <= 0.9 a(B). Choice per chain: the fewest development failures among the
feasible, ties to the lower L1; none feasible: the fewest failures, flagged "predicted slower than Cascade". Sizing
(4.4): the smallest grid size between the development size and the cap at which the non-inferiority tests of C1 and C5
against C2 (one-sided at alpha / 40) have power >= 0.9 for a true difference of zero at the development discordance
(sv_common.ni_power); a chain below 0.9 at the cap is cap-limited. Planned powers: N1, N2 at that size; S1-S8 the exact
McNemar's at alpha / 40 under the development rates. Joint power of a chain: the product over both p. The lock names
the identity of the environment and of the code (commit and digest; a real lock only from a clean commit, finding 3 of
the stage gate of 2026-10-12), the pinned kernel caches (unchanged since the screen's summary), the frozen paths, every
network's and input's sha256, the development outputs' sha256, the speed summary, the pipelines' plan (gate and C7:
chunks, shots, repetitions) and the blocks to read (conf: kind 11, chunks 0 .. n-1). select refuses a screen summary
that froze nothing reproducible (finding 4) and never overwrites a lock.

gate (item 8, finding 5): the bitwise check of the pipelines against the read computations on kind-12 shots
(GATE_CHUNKS), with this lock's thresholds, written beside the lock (gate_d13.json: the lock's sha256, the check's, the
result). It passes only if both modes pass and the check holds every output of the lock's gate plan, each on the plan's
chunks and seed kind and made with this lock, the pipelines all in one job and the reads in other jobs. The lock and
the gate file are then committed and pushed together (scripts/lockpair.sh, which writes the receipt beside the lock).

confirm (4.4-4.5): refused without the committed lock and gate (sv_common.committed_pair, finding 2); every
computation of the confirmation block, each part refused unless its environment, code, kernel caches, network, path,
thresholds, lock and lock commit are the lock's and a complete read lists it; the decisions per configuration and mode
(sv_common.assemble); per p and mode the ten hypotheses (N1, N2 non-inferiority by the inverted Newcombe interval;
S1-S8 exact McNemar), forty in one Holm family at alpha 0.05; labels: "worse" = S rejected and the X-only failures
exceed the Y-only ones, "better" = S rejected and fewer; "non-inferior" = N rejected and not "worse"; "inconclusive" =
a cap-limited chain-vs-C2 comparison with nothing rejected; "level" otherwise.

report (7): SURFACE_RESULT tables and the verdicts: a chain is "faster within the predefined accuracy tolerance" at p
when in each mode it is non-inferior or better against C2, its pipeline is faster than C2 timed in the same job (the
slowest of three processes), and that mode's bitwise check passed; "faster and more accurate" with "better" in both.
The timings come from the C7 check (bitwise --expect c7), which carries every one of the 48 pipeline outputs of the
lock's plan with the timing saved beside its outputs; each must match its row in the pipelines' log (finding 5)."""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from surface import sv_common as sc

S_PAIRS = (("C1", "C2"), ("C5", "C2"), ("C4", "C2"), ("C3", "C2"), ("C1", "C4"), ("C5", "C4"), ("C1", "C5"),
           ("C4", "C3"))
N_PAIRS = (("C1", "C2"), ("C5", "C2"))
MODES = ("single", "batched")


def grid(p, floor, cap):
    return sorted({c for c in sc.GRID if floor <= c <= cap} | {floor, cap})


def parse_pmap(s, cast=int):
    """'0.006:a-b,0.0035:c-d' or '0.006:10,0.0035:20' -> {p: value}."""
    out = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[float(k)] = sc.chunk_range(v) if "-" in v else cast(v)
    return out


# ------------------------------------------------------------------------------------------------------------ select

def select(a):
    t0 = time.time()
    sp = json.load(open(a.speed))
    if sp.get("flags") or set(sp.get("frozen", {})) != {"S", "B"}:
        raise SystemExit(f"{a.speed}: no reproducible frozen paths ({sp.get('flags')}): no lock (finding 4)")
    ident = sp["identity"]
    sc.check_analysis(ident, "select")
    if not a.dry and not ident["code"]["clean"]:
        raise SystemExit(f"code {ident['code']}: the evaluation runs a clean commit (DEPLOYED without +dirty); no lock")
    kc = sc.kcache()
    if kc["digest"] != sp["kcache"]["digest"]:
        raise SystemExit(f"kernel caches {kc} changed since the screen's summary ({sp['kcache']}): no lock")
    kinds = {"dev": sc.kind_of("dev", a.dry), "conf": sc.kind_of("conf", a.dry), "speed": sc.kind_of("speed", a.dry)}
    dev_ch = parse_pmap(a.dev_chunks) if a.dev_chunks else {p: range(0, sc.DEV_CHUNKS[p]) for p in sc.PS}
    if not a.dry and any(dev_ch[p] != range(0, sc.DEV_CHUNKS[p]) for p in sc.PS):
        raise SystemExit("the development blocks are chunks 0 .. n-1 of kind 10 (SURFACE_PLAN 4.4)")
    comp = sp["components"]
    t1S, aS, t1B, aB = comp["S"]["t1_us"], comp["S"]["a_us"], comp["B"]["t1_us"], comp["B"]["a_us"]
    alpha = sc.ALPHA / sc.M_FAMILY
    lay = sc.sl.Layout(sc.D)
    L = {"dry": a.dry, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "identity": ident, "code": ident["code"],
         "drivers": sp["drivers"], "kcache": kc, "kcache_screen": sp["kcache"], "frozen": sp["frozen"],
         "b_star": comp["B"]["b_star"], "components": comp, "pipes": sc.DRY_PIPES if a.dry else sc.PIPES,
         "speed": {"file": os.path.abspath(a.speed), "sha256": sc.sha256_file(a.speed),
                   "rows_sha256": sp["rows_sha256"]},
         "nets": {"S": {"name": sc.S_NAME, "sha256": sc.S_SHA256},
                  "B": {"file": os.path.join(sc.R, sc.CKPT["final"]), "sha256": sc.CKPT_SHA256["final"]}},
         "inputs": {"circuit_sha256": lay.circuit_sha256, "grid_sha256": lay.grid_sha256},
         "kinds": kinds, "alpha_size": alpha, "m_family": sc.M_FAMILY, "w_m": sc.W_M,
         "pipe_chunks": list(sc.PIPE_CHUNKS), "gate_chunks": list(sc.GATE_CHUNKS), "dev": {}, "p": {},
         "computations": {"S1": "S on the engine at batch 1", "SB": f"S on the engine in batches of {sc.S_BATCH}",
                          "B1": f"B on {sp['frozen']['B']['single']}",
                          "Bb": f"B on {sp['frozen']['B']['amortized']} in batches of b*",
                          "BQ1": "B at b* on the queue of the shots SB routes under C1's tau1 (item 9)",
                          "BQ5": "B at b* on the queue of the shots SB routes under C5's tau1 (item 9)",
                          "MS": "correlated PM decode, one shot per call", "MB": "correlated PM decode_batch",
                          "settings": "TF32 off, cudnn.benchmark False, cudnn.deterministic True, torch.compile "
                                      "default mode, bf16 autocast for B, the engine's FROZEN for S"}}
    want = {f"record.{k}": ident[k] for k in sc.IDENTITY}
    want["kind"] = kinds["dev"]
    want["record.kcache.digest"] = sp["kcache"]["digest"]
    want["path"] = lambda c, got: got == {"SB": sp["frozen"]["S"]["batched"], "Bb": sp["frozen"]["B"]["amortized"],
                                          "MS": "decode@1"}[c]
    want["net.sha256"] = lambda c, got: got == {"S": sc.S_SHA256, "B": sc.CKPT_SHA256["final"]}.get(c[0], got)
    for p in sc.PS:
        o = sc.load_read(a.dev, p, ["SB", "Bb", "MS"], dev_ch[p], want, complete=sp["kcache"]["digest"])
        L["dev"][str(p)] = {"chunks": [dev_ch[p].start, dev_ch[p].stop], "digests_sha256": hashlib.sha256(
            "".join(o["digests"]).encode()).hexdigest()[:16],
            "files": sorted(f for f in os.listdir(os.path.join(a.dev, sc.p_tag(p))) if f.endswith(".npz"))}
        obs, N = o["obs"], len(o["obs"])
        c, sdec = o["SB"]["c"], o["SB"]["dec"]
        z, bdec = o["Bb"]["z"], o["Bb"]["dec"]
        mpred, mus = o["MS"]["pred"], o["MS"]["us"]
        fC2 = bdec != obs
        delta = sc.DELTA_REL * float(fC2.mean())
        ac, az = np.abs(c).astype(np.float64), np.abs(z).astype(np.float64)
        rows = []
        for q1 in sc.Q1:
            tau1 = float(np.quantile(ac, q1))
            toB = sc.routed(c, tau1)
            d5 = sdec.copy()
            d5[toB] = bdec[toB]
            fB = float(toB.mean())
            rows.append({"config": "C5", "q1": q1, "tau1": tau1, "fail": int((d5 != obs).sum()), "f_B": fB,
                         "f_M": 0.0, "t_M_us": 0.0, "L1_us": t1S + fB * t1B, "A_us": aS + fB * aB,
                         "vs_C2": sc.counts(d5 != obs, fC2)})
            nB = int(toB.sum())
            for r in sc.R2:
                if r >= q1:
                    continue
                tau2 = float(np.quantile(az[toB], min(1.0, r * N / max(1, nB)))) if nB else float("-inf")
                toM = toB & sc.routed(z, tau2)
                d1 = d5.copy()
                d1[toM] = mpred[toM]
                fM = float(toM.mean())
                tM = float(mus[toM].mean()) if toM.any() else 0.0
                rows.append({"config": "C1", "q1": q1, "r": r, "tau1": tau1, "tau2": tau2,
                             "fail": int((d1 != obs).sum()), "f_B": fB, "f_M": fM, "t_M_us": tM,
                             "L1_us": t1S + fB * t1B + fM * tM, "A_us": aS + fB * aB + fM * tM / sc.W_M,
                             "vs_C2": sc.counts(d1 != obs, fC2)})
        for row in rows:
            row["feasible"] = row["L1_us"] <= sc.SPEED_MARGIN * t1B and row["A_us"] <= sc.SPEED_MARGIN * aB
        chosen = {}
        for cfg in ("C1", "C5"):
            rs = [r for r in rows if r["config"] == cfg]
            feas = [r for r in rs if r["feasible"]]
            pick = min(feas or rs, key=lambda r: (r["fail"], r["L1_us"]))
            chosen[cfg] = {k: pick[k] for k in ("q1", "r", "tau1", "tau2") if k in pick}
            chosen[cfg]["rule"] = ("the fewest development failures among the speed-feasible" if feas else
                                   "none speed-feasible: the fewest development failures; predicted slower than "
                                   "Cascade")
            chosen[cfg]["predicted_slower"] = not feas
        chosen["C4"] = {"tau1": sc.TAU_C4, "rule": "the registered threshold"}
        # the development decisions of every configuration (planned powers of S1-S8)
        dd = {"C2": bdec, "C3": mpred}
        d4 = sdec.copy()
        t4 = sc.routed(c, sc.TAU_C4)
        d4[t4] = mpred[t4]
        dd["C4"] = d4
        for cfg in ("C1", "C5"):
            toB = sc.routed(c, chosen[cfg]["tau1"])
            dx = sdec.copy()
            dx[toB] = bdec[toB]
            if cfg == "C1":
                toM = toB & sc.routed(z, chosen["C1"]["tau2"])
                dx[toM] = mpred[toM]
            dd[cfg] = dx
        fails = {k: v != obs for k, v in dd.items()}
        cnt = {f"{x}-{y}": sc.counts(fails[x], fails[y]) for x, y in S_PAIRS}
        floor, cap = (dev_ch[p].stop - dev_ch[p].start), sc.CONF_CAP[p]
        size, pw_at = None, {}
        for nch in grid(p, min(floor, cap), cap):
            pw = {x: sc.ni_power(cnt[f"{x}-C2"], nch * sc.CHUNK, delta, alpha) for x, _ in N_PAIRS}
            pw_at[nch] = pw
            if all(v >= sc.POWER for v in pw.values()):
                size = nch
                break
        cap_limited = size is None
        size = size or cap
        pw = pw_at[size]
        planned = {f"N:{x}-C2": pw[x] for x, _ in N_PAIRS}
        planned.update({f"S:{x}-{y}": sc.mcnemar_power(cnt[f"{x}-{y}"], size * sc.CHUNK, alpha) for x, y in S_PAIRS})
        conf_n = parse_pmap(a.dry_conf)[p] if a.dry else size
        c0 = a.dry_conf_start if a.dry else 0
        L["p"][str(p)] = {"delta": delta, "C2_dev_fail": int(fC2.sum()), "dev_shots": N, "chosen": chosen,
                          "conf_chunks": [c0, c0 + conf_n], "planned_chunks": size,
                          "cap_limited": {x: pw[x] < sc.POWER for x, _ in N_PAIRS} if cap_limited else
                          {x: False for x, _ in N_PAIRS},
                          "planned_power": planned, "power_by_size": {str(k): v for k, v in pw_at.items()},
                          "dev_counts": cnt, "candidates": rows,
                          "dev_fail": {k: int(v.sum()) for k, v in fails.items()}}
    L["joint_power"] = {x: float(np.prod([L["p"][str(p)]["planned_power"][f"N:{x}-C2"] for p in sc.PS]))
                        for x, _ in N_PAIRS}
    L["secs"] = round(time.time() - t0, 1)
    sc.write_json(a.out, L)
    for p in sc.PS:
        q = L["p"][str(p)]
        print(json.dumps({"p": p, "delta": q["delta"], "chosen": q["chosen"], "planned_chunks": q["planned_chunks"],
                          "conf_chunks": q["conf_chunks"], "cap_limited": q["cap_limited"],
                          "planned_power": {k: round(v, 3) for k, v in q["planned_power"].items()},
                          "dev_fail": q["dev_fail"]}), flush=True)
    print(json.dumps({"joint_power": L["joint_power"], "secs": L["secs"]}), flush=True)


def gate(a):
    """The bitwise gate (item 8) as a file beside the lock (the lock itself never changes after select, so that every
    read and pipeline names one lock sha256): both modes must pass on the gate's kind-12 shots, every configuration at
    both p, with the lock's thresholds."""
    L = json.load(open(a.lock))
    lock_sha = sc.sha256_file(a.lock)
    sc.check_analysis(L["identity"], "gate")
    chk = json.load(open(a.check))
    if chk.get("expect") != "gate" or chk["lock_sha256"] != lock_sha:
        raise SystemExit("the check is not the gate's check (bitwise --expect gate) against this lock")
    plan = L["pipes"]["gate"]
    need = {(c, m, p, r) for c in sc.PIPE_CONFIGS for m in MODES for p in sc.PS for r in plan["reps"]}
    cover = {(r["config"], r["mode"], r["p"], r["rep"]) for r in chk["rows"]}
    span = {m: [plan["chunks"][0], plan["chunks"][0] + -(-plan[m] // sc.CHUNK)] for m in MODES}
    off = [r["file"] for r in chk["rows"] if r["kind"] != L["kinds"]["speed"] or r["chunks"] != span[r["mode"]] or
           r["shots"] != plan[r["mode"]] or r["lock_sha256"] != lock_sha or r["job"] in r["read_jobs"]]
    jobs = sorted({r["job"] for r in chk["rows"]})
    g = {"lock_sha256": lock_sha, "check": os.path.abspath(a.check), "check_sha256": sc.sha256_file(a.check),
         "pass": chk["pass"], "missing": sorted(map(list, need - cover)), "off_plan": off, "pipeline_jobs": jobs,
         "read_jobs": sorted({j for r in chk["rows"] for j in r["read_jobs"]}), "rows": len(chk["rows"]),
         "all_pass": (all(chk["pass"].get(m, False) for m in MODES) and need == cover and
                      len(chk["rows"]) == len(need) and not off and len(jobs) == 1)}
    sc.write_json(a.out, g)
    print(json.dumps(g), flush=True)
    if not g["all_pass"]:
        raise SystemExit("the bitwise gate failed: no lock commit, no confirmation read")


# ----------------------------------------------------------------------------------------------------------- confirm

def label(rej_s, bx, by, rej_n=None, cap_limited=False):
    """Item 2: worse / better from the exact McNemar test's rejection and the sign of the discordant counts."""
    if rej_s and bx > by:
        return "worse"
    if rej_s and bx < by:
        return "better"
    if rej_n:
        return "non-inferior"
    if cap_limited and not rej_s and not rej_n:
        return "inconclusive"
    return "level"


def confirm(a):
    from scipy.stats import norm
    t0 = time.time()
    L = json.load(open(a.lock))
    lock_sha = sc.sha256_file(a.lock)
    sc.check_analysis(L["identity"], "confirm")
    _, Rc = sc.committed_pair(a.lock, a.gate, a.receipt)        # finding 2: the committed lock and its passed gate
    fz = L["frozen"]
    paths = {"S1": fz["S"]["single"], "SB": fz["S"]["batched"], "B1": fz["B"]["single"], "Bb": fz["B"]["amortized"],
             "BQ1": fz["B"]["amortized"], "BQ5": fz["B"]["amortized"], "MS": "decode@1", "MB": "decode_batch",
             "MP": "plain decode_batch"}
    out = {"lock": os.path.abspath(a.lock), "lock_sha256": lock_sha, "gate_sha256": sc.sha256_file(a.gate),
           "lock_commit": Rc["commit"], "conf": os.path.abspath(a.conf), "p": {}, "hypotheses": []}
    hyp = []
    for p in sc.PS:
        q = L["p"][str(p)]
        ch = q["chosen"]
        want = {f"record.{k}": L["identity"][k] for k in sc.IDENTITY}
        want.update({"record.kcache.digest": L["kcache"]["digest"], "kind": L["kinds"]["conf"],
                     "lock_sha256": lock_sha, "lock_commit": Rc["commit"],
                     "path": lambda c, got: got == paths[c],
                     "net.sha256": lambda c, got: got == {"S": L["nets"]["S"]["sha256"],
                                                          "B": L["nets"]["B"]["sha256"]}.get(c[0], got),
                     "tau1": lambda c, got: got == {"BQ1": ch["C1"]["tau1"], "BQ5": ch["C5"]["tau1"]}.get(c)})
        o = sc.load_read(a.conf, p, list(sc.COMPS), range(*q["conf_chunks"]), want, complete=L["kcache"]["digest"])
        obs, N = o["obs"], len(o["obs"])
        res = {"shots": N, "chunks": q["conf_chunks"], "modes": {},
               "plain_pm": sc.point(int((o["MP"]["pred"] != obs).sum()), N)}          # context only
        for mode in MODES:
            dec, stg, fl = {}, {}, {}
            for cfg in sc.CONFIGS:
                dec[cfg], stg[cfg] = sc.assemble(cfg, mode, o, ch)
                fl[cfg] = dec[cfg] != obs
            Bx = o["B1" if mode == "single" else "Bb"]["dec"] != obs
            Sx = o["S1" if mode == "single" else "SB"]["dec"] != obs
            rows = {}
            for cfg in sc.CONFIGS:
                r = {**sc.point(int(fl[cfg].sum()), N), "f_B": float((stg[cfg] >= 1).mean()) if cfg in ("C1", "C5")
                     else float(cfg == "C2"), "f_M": float((stg[cfg] == 2).mean())}
                if cfg in ("C1", "C4", "C5"):
                    keep = stg[cfg] == 0
                    r["kept"] = {"B_fails_S_repairs": int((Bx & ~Sx & keep).sum()),
                                 "S_fails_B_would_not": int((Sx & ~Bx & keep).sum())}
                if cfg in ("C1", "C4"):
                    via = stg[cfg] == 2
                    prior = Bx if cfg == "C1" else Sx
                    r["M_decides"] = {"repairs": int((prior & ~fl[cfg] & via).sum()),
                                      "regressions": int((~prior & fl[cfg] & via).sum())}
                rows[cfg] = r
            for x, y in S_PAIRS:
                cn = sc.counts(fl[x], fl[y])
                hyp.append({"p": p, "mode": mode, "test": "S", "x": x, "y": y, **cn,
                            "pvalue": sc.mcnemar_p(cn["only_x"], cn["only_y"])})
            for x, y in N_PAIRS:
                cn = sc.counts(fl[x], fl[y])
                hyp.append({"p": p, "mode": mode, "test": "N", "x": x, "y": y, **cn, "delta": q["delta"],
                            "pvalue": sc.ni_p(*sc.abcd(cn), q["delta"])})
            agree = {"S1_SB_logits_differ": int((o["S1"]["c"].view(np.uint32) != o["SB"]["c"].view(np.uint32)).sum()),
                     "MS_MB_differ": int((o["MS"]["pred"] != o["MB"]["pred"]).sum()),
                     "B1_Bb_decisions_differ": int((o["B1"]["dec"] != o["Bb"]["dec"]).sum())}
            for bq in ("BQ1", "BQ5"):
                m_ = o[bq]["dec"] != 255
                agree[f"{bq}_Bb_logits_differ"] = int((o[bq]["z"][m_].view(np.uint32) !=
                                                       o["Bb"]["z"][m_].view(np.uint32)).sum())
            res["modes"][mode] = {"configs": rows, "agree": agree}
        out["p"][str(p)] = res
    if len(hyp) != sc.M_FAMILY:
        raise SystemExit(f"{len(hyp)} hypotheses, the family has {sc.M_FAMILY}")
    lv = sc.holm([h["pvalue"] for h in hyp], sc.ALPHA)
    for i, h in enumerate(hyp):
        h["holm_level"] = lv[i]
        h["rejected"] = lv[i] is not None
        a_, b_, c_, d_ = sc.abcd(h)
        z95 = float(norm.ppf(0.975))
        h["diff"], h["lo95"], h["hi95"] = sc.newcombe10(a_, b_, c_, d_, z95)
        if h["rejected"]:
            zh = float(norm.ppf(1 - lv[i] / (2 if h["test"] == "S" else 1)))
            _, h["lo_holm"], h["hi_holm"] = sc.newcombe10(a_, b_, c_, d_, zh)
            if h["test"] == "N" and not h["hi_holm"] < h["delta"]:
                raise SystemExit(f"non-inferiority p-value and interval disagree: {h}")
    out["hypotheses"] = hyp
    labels = {}
    for p in sc.PS:
        for mode in MODES:
            hs = {(h["test"], h["x"], h["y"]): h for h in hyp if h["p"] == p and h["mode"] == mode}
            for x, y in S_PAIRS:
                s = hs[("S", x, y)]
                nn = hs.get(("N", x, y))
                cap = L["p"][str(p)]["cap_limited"].get(x, False) if y == "C2" else False
                labels[f"{p}|{mode}|{x}-{y}"] = label(s["rejected"], s["only_x"], s["only_y"],
                                                      nn["rejected"] if nn else None, cap)
    out["labels"] = labels
    out["secs"] = round(time.time() - t0, 1)
    sc.write_json(a.out, out)
    for k, v in labels.items():
        print(json.dumps({"comparison": k, "label": v}), flush=True)
    for p in sc.PS:
        for mode in MODES:
            print(json.dumps({"p": p, "mode": mode, "fail": {c: r["fail"] for c, r in
                                                             out["p"][str(p)]["modes"][mode]["configs"].items()},
                              "agree": out["p"][str(p)]["modes"][mode]["agree"]}), flush=True)


# ------------------------------------------------------------------------------------------------------------ report

def verdicts(L, C, bw):
    """Section 7 per chain and p, the headline and the d = 17 conditions (P1, P2); the timings are the C7 check's rows
    (each pipeline output's own record)."""
    slow = {}
    for r in bw["rows"]:
        k = (r["p"], r["mode"], r["config"])
        s = slow.setdefault(k, {"value": 0.0, "p999": 0.0, "processes": 0, "jobs": set()})
        s["value"] = max(s["value"], r["timing"]["value"])
        s["p999"] = max(s["p999"], r["timing"].get("p999") or 0.0)
        s["processes"] += 1
        s["jobs"].add(r["job"])
    lab = C["labels"]
    out = {}
    for p in sc.PS:
        for x in ("C1", "C5"):
            per = {}
            for mode in MODES:
                acc = lab[f"{p}|{mode}|{x}-C2"]
                sx, s2 = slow.get((p, mode, x)), slow.get((p, mode, "C2"))
                same_job = bool(sx and s2 and sx["jobs"] == s2["jobs"])
                fast = bool(sx and s2 and same_job and sx["processes"] >= sc.REPS and s2["processes"] >= sc.REPS
                            and sx["value"] < s2["value"])
                per[mode] = {"accuracy": acc, "accuracy_ok": acc in ("non-inferior", "better"),
                             "speed_us": sx and sx["value"], "c2_us": s2 and s2["value"], "faster": fast,
                             "same_job": same_job, "bitwise": bool(bw["pass"].get(mode, False))}
                if mode == "single" and sx and s2:
                    per[mode]["p999_us"], per[mode]["c2_p999_us"] = sx["p999"], s2["p999"]
                    per[mode]["tail"] = "faster on average, slower in the tail" if sx["p999"] > s2["p999"] else ""
            ok = all(v["accuracy_ok"] and v["faster"] and v["bitwise"] for v in per.values())
            better = ok and all(v["accuracy"] == "better" for v in per.values())
            out[f"{p}|{x}"] = {"modes": per, "faster_within_tolerance": ok, "faster_and_more_accurate": better}
    head = [x for x in ("C1", "C5") if all(out[f"{p}|{x}"]["faster_within_tolerance"] for p in sc.PS)]
    head_b = [x for x in head if all(out[f"{p}|{x}"]["faster_and_more_accurate"] for p in sc.PS)]
    P1 = all(lab[f"{p}|{m}|C3-C2"] == "worse" for p in sc.PS for m in MODES)       # C2 better than C3
    P2 = [x for x in ("C1", "C5") if out[f"{sc.P_HIGH}|{x}"]["faster_within_tolerance"] and
          all(lab[f"{sc.P_LOW}|{m}|{x}-C2"] in ("better", "non-inferior") for m in MODES)]
    return out, {"headline_chains": head, "headline_more_accurate": head_b, "P1": P1, "P2_chains": P2,
                 "d17": "propose" if P1 and P2 else ("examine the reproduction" if not P1 else "not proposed")}


def fmt(x, d=3):
    """d significant digits (P_block near 0.02 and its interval stay distinguishable)."""
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{d}g}"
    return str(x)


def report(a):
    L, C, sp = json.load(open(a.lock)), json.load(open(a.confirm)), json.load(open(a.speed))
    lock_sha = sc.sha256_file(a.lock)
    sc.check_analysis(L["identity"], "report")
    if C["lock_sha256"] != lock_sha or sc.sha256_file(a.speed) != L["speed"]["sha256"]:
        raise SystemExit("the confirmation or the speed summary belongs to another lock")
    bw = json.load(open(a.bitwise))           # finding 5: every C7 output of the lock's plan, bound to its timing
    plan = L["pipes"]["c7"]
    need = {(c, m, p, r) for c in sc.PIPE_CONFIGS for m in MODES for p in sc.PS for r in plan["reps"]}
    got = [(r["config"], r["mode"], r["p"], r["rep"]) for r in bw["rows"]]
    if bw.get("expect") != "c7" or bw["lock_sha256"] != lock_sha or set(got) != need or len(got) != len(need) or \
            any(r["lock_sha256"] != lock_sha for r in bw["rows"]):
        raise SystemExit(f"{a.bitwise}: not the C7 check of this lock with all {len(need)} outputs of its plan")
    log = {}
    for x in open(a.pipes):
        if x.strip():
            r = json.loads(x)
            if r.get("role") == "pipeline" and r.get("saved"):
                log.setdefault((r["saved"], r["job"], r["pid"]), []).append(r)
    for r in bw["rows"]:
        t = log.get((r["file"], r["job"], r["pid"]), [])
        v = [x["latency_us"]["mean"] if x["mode"] == "single" else x["per_shot_us"] for x in t]
        if len(t) != 1 or t[0].get("lock_sha256") != lock_sha or v[0] != r["timing"]["value"]:
            raise SystemExit(f"{r['file']}: no single matching timing record in {a.pipes}")
    ver, head = verdicts(L, C, bw)
    md = [f"# Cascade-surface at d = 13: the confirmation{' (DRY RUN, kind 13 shots, not a result)' if L['dry'] else ''}",
          "", f"Lock sha256 {C['lock_sha256'][:16]}; GPU {L['identity']['gpu']}, driver {', '.join(L['drivers'])}, "
          f"torch {L['identity']['torch']}, venv {L['identity']['prefix']} (packages {L['identity']['packages_sha256'][:16]}). "
          "B is Cascade-surface at H = 256, an inferred reproduction (SURFACE_PLAN 3.2); bf16 for B, fp16 matmul "
          "inputs with fp32 accumulation for S's engine, PyMatching on the node's CPU.", ""]
    for p in sc.PS:
        q, cp = L["p"][str(p)], C["p"][str(p)]
        md += [f"## p = {p * 100:g} %", "", f"Confirmation block: {cp['shots']:,} shots (chunks {cp['chunks'][0]}-"
               f"{cp['chunks'][1] - 1}); delta_p = {fmt(q['delta'])} (10 % of C2's development failure rate). "
               f"Thresholds: C1 tau1 {q['chosen']['C1']['tau1']:.6g}, tau2 {q['chosen']['C1']['tau2']:.6g} "
               f"(q1 {q['chosen']['C1']['q1']}, r {q['chosen']['C1']['r']}); C5 tau1 {q['chosen']['C5']['tau1']:.6g} "
               f"(q1 {q['chosen']['C5']['q1']}); C4 tau {sc.TAU_C4}.", ""]
        for mode in MODES:
            md += [f"### {mode}", "", "| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |",
                   "|---|---|---|---|---|---|"]
            for cfg, r in cp["modes"][mode]["configs"].items():
                md.append(f"| {cfg} | {r['fail']:,} | {fmt(r['p_block'])} ({fmt(r['pb_lo'])}, {fmt(r['pb_hi'])}) | "
                          f"{fmt(r['pl'])} | {r['f_B']:.4f} | {r['f_M']:.4f} |")
            md += ["", "| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |",
                   "|---|---|---|---|---|---|---|---|"]
            for h in C["hypotheses"]:
                if h["p"] == p and h["mode"] == mode:
                    key = f"{p}|{mode}|{h['x']}-{h['y']}"
                    lb = C["labels"][key] if h["test"] == "S" else ("rejected" if h["rejected"] else "-")
                    md.append(f"| {h['x']} vs {h['y']} | {h['test']} | {h['only_x']} | {h['only_y']} | "
                              f"{fmt(h['pvalue'])} | {fmt(h['holm_level'])} | ({fmt(h['lo95'])}, {fmt(h['hi95'])}) | "
                              f"{lb} |")
            md += ["", f"Bitwise notes: {json.dumps(cp['modes'][mode]['agree'])}", ""]
    md += ["## Speed (one GPU model; slowest of three processes)", "",
           f"Decoders alone: S t1 {fmt(sp['components']['S']['t1_us'])} us, a(4096) {fmt(sp['components']['S']['a_us'])} us; "
           f"B t1 {fmt(sp['components']['B']['t1_us'])} us ({sp['frozen']['B']['single']}), a {fmt(sp['components']['B']['a_us'])} us "
           f"({sp['frozen']['B']['amortized']}); M {json.dumps(sp['M'])}.", "",
           "| p | chain | mode | accuracy | pipeline us/shot | C2 us/shot | faster | bitwise | tail |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in ver.items():
        p, x = k.split("|")
        for mode, m in v["modes"].items():
            md.append(f"| {p} | {x} | {mode} | {m['accuracy']} | {fmt(m['speed_us'])} | {fmt(m['c2_us'])} | "
                      f"{m['faster']} | {m['bitwise']} | {m.get('tail', '')} |")
    md += ["", "## Verdicts", ""]
    for k, v in ver.items():
        p, x = k.split("|")
        word = ("faster and more accurate" if v["faster_and_more_accurate"] else
                "faster within the predefined accuracy tolerance (up to 10 % more failures than Cascade alone)"
                if v["faster_within_tolerance"] else "no claim")
        md.append(f"- p = {float(p) * 100:g} %, {x}: {word}")
    md += ["", f"Headline chains: {head['headline_chains'] or 'none'} (joint planned power "
           f"{json.dumps({x: round(v, 3) for x, v in L['joint_power'].items()})}); more accurate: "
           f"{head['headline_more_accurate'] or 'none'}. P1 (C2 better than C3 at both p, both modes): {head['P1']}; "
           f"P2 chains: {head['P2_chains'] or 'none'}; d = 17: {head['d17']}.", ""]
    open(a.out, "w").write("\n".join(md))
    json.dump({"lock_sha256": lock_sha, "verdicts": ver, "headline": head}, open(a.out + ".json", "w"), indent=1,
              default=list)
    print(json.dumps(head), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--dev", required=True)
    s.add_argument("--speed", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dry", action="store_true")
    s.add_argument("--dev-chunks", default="", help="dry runs: 0.006:a-b,0.0035:c-d")
    s.add_argument("--dry-conf", default="0.006:5,0.0035:10", help="dry runs: confirmation chunks per p")
    s.add_argument("--dry-conf-start", type=int, default=4000)
    s = sub.add_parser("gate")
    s.add_argument("--lock", required=True)
    s.add_argument("--check", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("confirm")
    s.add_argument("--lock", required=True)
    s.add_argument("--gate", required=True)
    s.add_argument("--receipt", default="", help="scripts/lockpair.sh's receipt (default: lock_commit.json beside the "
                   "lock)")
    s.add_argument("--conf", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("report")
    for k in ("lock", "confirm", "speed", "pipes", "bitwise", "out"):
        s.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    {"select": select, "gate": gate, "confirm": confirm, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
