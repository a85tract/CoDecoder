"""Phase C routing (PHASE_C_PLAN.md section 4): the rule on the development block, the lock, and the one confirmation read.

Configurations (S = a small network, B = Cascade of the setting, R = full-DEM Relay-BP; a shot is routed when its
confidence is strictly below the threshold):
  S-B    S decides; shots with S's margin < tau1 go to B, which decides them
  S-B-R  as S-B; of the shots S routed, those with B's confidence < tau2 go to R, which decides them
  S-R    S decides; shots with S's margin < tau go to R (the known baseline; S-R with S = r12 is question 1's accurate
         mode)
Thresholds on the development block (dev_taus): tau1 = the q1-quantile of S's margin; tau2 = the value below which B's
confidence falls on r x N of the S-routed shots (so a fraction r of all shots reaches R); tau = the q-quantile of S's
margin (numpy's default interpolation). On the confirmation block they are used as locked.

  python pc_route.py select  --dev DIR --relay DIR_R --speed speed_components.json --out locked_<setting>.json
  python pc_route.py confirm --conf DIR --relay DIR_R --locked locked_<setting>.json --out confirm_<setting>.json
Reads are pc_eval.py's (bb.cascade_eval --save outputs, pc_common.load_read): a ConvLines network's confidence is the
margin of its fused engine, the path it is timed on, and the lock records each network's read precision, which the
confirmation read must repeat. The lock also names the card, its driver and the torch build of the timings (one GPU
model, one venv: PHASE_C_PLAN.md 5.1); select and confirm refuse a read made on another card or torch. Relay results
are pc_relay.py's. Every locked configuration's comparison with Cascade belongs to the headline family
(pc_common.M_HEADLINE), which pc_report judges with one Holm procedure over both settings; confirm reports the
statistics, not the verdicts."""
import argparse
import json
import os
import time

import numpy as np

import pc_common as pc


def dev_candidates(family, p, idx, relay_only=False):
    """The rule's candidates at p in a setting (only networks present in the read)."""
    B = pc.BIG[family]
    smalls = [s for s in pc.SMALL[family] if s in idx]
    out = []
    if not relay_only:
        out += [{"type": "S-B", "p": p, "S": s, "B": B, "q1": q1} for s in smalls for q1 in pc.Q1]
    if p in pc.RELAY_P:
        out += [{"type": "S-B-R", "p": p, "S": s, "B": B, "q1": q1, "r": r, "score": sc}
                for s in smalls for q1 in pc.Q1 for sc in pc.SCORES for r in pc.R2 if r < q1]
        out += [{"type": "S-R", "p": p, "S": s, "q": pc.BASE_Q[p]} for s in smalls + [pc.REGISTERED] if s in idx]
    return out


def dev_taus(z, meta, idx, cfgs):
    """Each candidate's thresholds, estimated on this read."""
    out = []

    def conf(tag, score="minabs"):
        return pc.confidence(z, meta, idx, tag, score)

    for c in cfgs:
        if c["type"] == "S-R":
            out.append({"tau": float(np.quantile(conf(c["S"]), c["q"]))})
            continue
        cs = conf(c["S"])
        t1 = float(np.quantile(cs, c["q1"]))
        if c["type"] == "S-B":
            out.append({"tau1": t1})
            continue
        routed = cs < t1
        cb = conf(c["B"], c["score"])[routed]
        frac = c["r"] * len(cs) / max(1, len(cb))
        out.append({"tau1": t1, "tau2": float(np.quantile(cb, min(1.0, frac))) if len(cb) else float("-inf")})
    return out


def masks(z, meta, idx, c, tau):
    """(to B, to R) boolean masks of a configuration with thresholds tau on a read."""
    n = len(z["obs"])
    if c["type"] == "S-R":
        return np.zeros(n, bool), pc.confidence(z, meta, idx, c["S"]) < tau["tau"]
    toB = pc.confidence(z, meta, idx, c["S"]) < tau["tau1"]
    if c["type"] == "S-B":
        return toB, np.zeros(n, bool)
    return toB, toB & (pc.confidence(z, meta, idx, c["B"], c["score"]) < tau["tau2"])


def load_relay(d, p, n):
    """pc_relay's result at p -> (has (n,) bool, fail (n,) bool) over the read's shots."""
    r = np.load(os.path.join(d, f"relay_{pc.p_tag(p)}.npz"))
    has, fail = np.zeros(n, bool), np.zeros(n, bool)
    has[r["idx"]] = True
    fail[r["idx"]] = r["fail"]
    secs = np.full(n, np.nan)
    secs[r["idx"]] = r["secs"]
    return has, fail, secs


def config_fail(z, idx, c, toB, toR, relay=None):
    """Per-shot failures of a configuration."""
    f = z[f"fail{idx[c['S']]}"].astype(bool).copy()
    if c["type"] != "S-R":
        fb = z[f"fail{idx[c['B']]}"].astype(bool)
        f[toB] = fb[toB]
    if toR.any():
        has, rf, _ = relay
        if not has[toR].all():
            raise SystemExit(f"{c}: {int((~has[toR]).sum())} shots routed to Relay-BP were not decoded")
        f[toR] = rf[toR]
    return f


def counts(fd, fb):
    return {"both": int((fd & fb).sum()), "only_D": int((fd & ~fb).sum()), "only_B": int((~fd & fb).sum()),
            "shots": int(len(fd))}


def alternative(cnt):
    """The alternative of the sizing (finding 8): D fails 10 % less often than B (delta = DELTA_REL x B's development
    failure rate) at the development block's discordant rate d, the discordances at (d - delta) / 2 for D and (d + delta)
    / 2 for B; None when d < delta (D and B differ on too few shots to show it)."""
    N = cnt["shots"]
    fB = (cnt["both"] + cnt["only_B"]) / N
    d = (cnt["only_D"] + cnt["only_B"]) / N
    delta = pc.DELTA_REL * fB
    if delta <= 0 or d < delta:
        return None, d, delta
    pb = (d + delta) / 2
    return (cnt["both"] / N, pb - delta, pb, 1.0), d, delta


def size_comparison(cnt, cap_chunks, floor_chunks, alpha):
    """Confirmation size for D against B (finding 8): the smallest grid size at which the exact McNemar test and the
    Newcombe interval at alpha (ALPHA / M_HEADLINE) detect a 10 % reduction of B's failures with probability >= POWER,
    under the development block's discordant rate (bb.route.power, simulated with its fixed seed)."""
    from bb import route
    alt, d, delta = alternative(cnt)
    if alt is None:
        return {"sized": False, "why": "the development block's discordant rate is below 10 % of B's failure rate",
                "d": d, "delta": delta}
    grid = sorted({c for c in (10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, cap_chunks)
                   if floor_chunks <= c <= cap_chunks} | {floor_chunks})
    for ch in grid:
        pw = route.power(alt, ch * pc.CHUNK, alpha=alpha, reps=4000)
        if pw >= pc.POWER:
            return {"sized": True, "chunks": ch, "power": pw, "d": d, "delta": delta}
    return {"sized": False, "why": "underpowered at the cap", "d": d, "delta": delta}


def planned_power(cnt, chunks, alpha):
    """The power of D's comparison with B on a confirmation block of `chunks` chunks (gate finding 3: every locked
    comparison is judged, so each carries its planned power); None when d < delta."""
    from bb import route
    alt = alternative(cnt)[0]
    return None if alt is None else route.power(alt, chunks * pc.CHUNK, alpha=alpha, reps=4000)


def predicted(c, sp, t_r, to_b, to_r):
    """Predicted mean single-shot latency L1 and amortized latency A per shot (us) of a configuration from the speed
    job's components (speed_components.json): t1 = single-shot host round trip on the network's frozen single path,
    a = host-to-host time per shot at batch 4,096 on its frozen batched path (the batched pipeline runs S and B's queue
    at 4,096); t_r = Relay-BP's mean time per routed shot (us); to_b, to_r = the fractions of the development block the
    candidate routes to B and to R (measured with its thresholds, ties included: gate finding 6)."""
    n = sp["nets"]
    L1, A = n[c["S"]]["t1_us"], n[c["S"]]["a4096_us"]
    if c["type"] in ("S-B", "S-B-R"):
        L1 += to_b * n[c["B"]]["t1_us"]
        A += to_b * n[c["B"]]["a4096_us"]
    if to_r:
        L1 += to_r * t_r
        A += to_r * t_r / pc.NCPU_RELAY
    return L1, A


def select(a):
    t0 = time.time()
    sp = json.load(open(a.speed))
    ps = [float(x) for x in a.p.split(",")] if a.p else list(pc.PS)
    fam = json.load(open(os.path.join(a.dev, f"{pc.p_tag(ps[0])}.json")))["family"]
    B = pc.BIG[fam]
    LC, AC = sp["nets"][B]["t1_us"], sp["nets"][B]["a_best_us"]           # Cascade on its frozen paths
    alpha = pc.ALPHA / pc.M_HEADLINE                                      # the sizing's share of the one alpha
    locked = {"family": fam, "B": B, "dev": os.path.abspath(a.dev), "relay_dev": os.path.abspath(a.relay),
              "speed": os.path.abspath(a.speed), "speed_sha256": pc.sha256_file(a.speed),
              "gpu": sp["gpu"], "driver": sp["driver"], "torch": sp["torch"],    # the one GPU model and venv (plan 5.1)
              "frozen_paths": {t: {m: f"{x}@{b}" for m, (x, b) in pc.frozen_paths(v).items()}
                               for t, v in sp["nets"].items()},
              "cascade_t1_us": LC, "cascade_a_us": AC, "alpha_size": alpha, "m_headline": pc.M_HEADLINE,
              "ni_margin": {}, "configs": [], "candidates": [], "nets": {}, "read_prec": {}}
    for p in ps:
        z, meta, idx = pc.load_read(a.dev, p)
        if meta["family"] != fam:
            raise SystemExit(f"{a.dev}: mixed families")
        locked["nets"].update({t: meta["models"][i]["sha256"] for t, i in idx.items()})
        locked["read_prec"].update({t: meta["models"][i]["prec"] for t, i in idx.items()})   # ConvLines: "fused"
        for t, i in idx.items():                                 # read on the card and the torch of the timings
            m = meta["models"][i]
            if (m["gpu"], m["torch"]) != (sp["gpu"], sp["torch"]):
                raise SystemExit(f"{a.dev} at {p}: {t} read on {m['gpu']} (torch {m['torch']}), timed on {sp['gpu']} "
                                 f"(torch {sp['torch']})")
        cands = dev_candidates(fam, p, idx)
        taus = dev_taus(z, meta, idx, cands)
        relay = load_relay(a.relay, p, len(z["obs"])) if p in pc.RELAY_P else None
        fb = z[f"fail{idx[B]}"].astype(bool)
        locked["ni_margin"][str(p)] = pc.DELTA_REL * float(fb.mean())        # gate finding 5: frozen before the read
        rows = []
        for c, t in zip(cands, taus):
            toB, toR = masks(z, meta, idx, c, t)
            fd = config_fail(z, idx, c, toB, toR, relay)
            t_r = float(np.nanmean(relay[2][toR])) if toR.any() else 0.0      # seconds per routed shot
            L1, A = predicted(c, sp, 1e6 * t_r, float(toB.mean()), float(toR.mean()))
            rows.append({**c, "tau": t, "fail": int(fd.sum()), "shots": int(len(fd)), "to_B": float(toB.mean()),
                         "to_R": float(toR.mean()), "relay_s_mean": t_r, "L1_us": L1, "A_us": A,
                         "feasible": L1 <= pc.SPEED_MARGIN * LC and A <= pc.SPEED_MARGIN * AC,
                         "vs_B": counts(fd, fb)})
        for typ in ("S-B", "S-B-R"):
            rs = [r for r in rows if r["type"] == typ]
            if not rs:
                continue
            feas = [r for r in rs if r["feasible"]]
            F = min(r["fail"] for r in feas or rs)               # Amendment 1 (plan 4.3, step 3): the candidates
            bound = F + F ** 0.5                                 # within one standard error of the fewest failures
            band = [r for r in feas or rs if r["fail"] <= bound]  # count as tied; the lowest predicted L1 is taken
            pick = min(band, key=lambda r: (r["L1_us"], r["fail"]))
            rule = "the lowest predicted L1 within F + sqrt(F) development failures, F the fewest (Amendment 1)"
            locked["configs"].append({**pick, "rule": f"among the speed-feasible, {rule}" if feas else
                                      f"none speed-feasible: {rule}", "F": F, "band_bound": bound,
                                      "tied": [{k: r[k] for k in ("S", "q1", "r", "score", "fail", "L1_us", "A_us")
                                                if k in r} for r in band]})
        s_pick = next((c["S"] for c in locked["configs"] if c["type"] == "S-B" and c["p"] == p), None)
        for r in rows:
            if r["type"] == "S-R" and r["S"] in (s_pick, pc.REGISTERED):
                locked["configs"].append({**r, "rule": "a priori (the registered accurate mode's q)"})
        locked["candidates"] += [{k: v for k, v in r.items() if k != "vs_B"} for r in rows]
    locked["sweep_secs"] = round(time.time() - t0, 1)            # the candidates over every p (arrays read once)
    for c in locked["configs"]:
        c["sizing"] = size_comparison(c["vs_B"], pc.CONF_CAP[c["p"]], min(pc.DEV_CHUNKS[c["p"]], pc.CONF_CAP[c["p"]]),
                                      alpha)
    locked["conf_chunks"] = {}
    for p in ps:
        sz = [c["sizing"]["chunks"] for c in locked["configs"] if c["p"] == p and c["sizing"]["sized"]]
        locked["conf_chunks"][str(p)] = min(pc.CONF_CAP[p], max([pc.DEV_CHUNKS[p]] + sz))
    for c in locked["configs"]:                              # every locked comparison is judged (gate finding 3)
        c["planned_power"] = planned_power(c["vs_B"], locked["conf_chunks"][str(c["p"])], alpha)
    locked["joint_power"] = {}                               # 'better' at both claim points (gate finding 5)
    for key in sorted({pc.decoder_key(c) for c in locked["configs"]}):
        pw = [c["planned_power"] for c in locked["configs"] if pc.decoder_key(c) == key and c["p"] in pc.RELAY_P]
        locked["joint_power"][key] = (float(np.prod(pw)) if len(pw) == len(pc.RELAY_P) and None not in pw
                                      else None)
    locked["secs"] = round(time.time() - t0, 1)
    json.dump(locked, open(a.out, "w"), indent=1)
    for c in locked["configs"]:
        print(json.dumps({k: c[k] for k in ("type", "p", "S", "fail", "shots", "to_B", "to_R", "L1_us", "A_us",
                                             "feasible", "rule", "planned_power") if k in c}))
    print(json.dumps({"conf_chunks": locked["conf_chunks"], "joint_power": locked["joint_power"],
                      "alpha_size": alpha, "sweep_secs": locked["sweep_secs"], "secs": locked["secs"]}))


def confirm(a):
    """The one analysis of a confirmation block: per locked configuration its accuracy, routed fractions, and paired
    statistics against Cascade alone and against S alone. The verdicts against Cascade come from pc_report's one Holm
    procedure over the headline family. Secondary (gate finding 5): D is 'not worse than Cascade by more than the
    margin' when the upper end of Newcombe's interval for P(D) - P(B), at the one-sided level ALPHA / M_SECONDARY
    (Bonferroni over every locked configuration of both settings), lies below the margin frozen in the lock (DELTA_REL
    x Cascade's development failure rate at p)."""
    from bb.route import paired
    locked = json.load(open(a.locked))
    fam, B = locked["family"], locked["B"]
    a_ni = 2 * pc.ALPHA / pc.M_SECONDARY                     # two-sided level of the one-sided Bonferroni bound
    res = []
    for p in sorted({c["p"] for c in locked["configs"]}):
        z, meta, idx = pc.load_read(a.conf, p)
        if meta["family"] != fam:
            raise SystemExit(f"{a.conf} at {p}: family {meta['family']}, locked {fam}")
        for t, i in idx.items():
            if t in locked["nets"] and meta["models"][i]["sha256"] != locked["nets"][t]:
                raise SystemExit(f"{t}: sha256 differs from the locked network")
            if meta["models"][i]["prec"] != locked["read_prec"].get(t, meta["models"][i]["prec"]):
                raise SystemExit(f"{t}: read in {meta['models'][i]['prec']}, locked in {locked['read_prec'][t]}")
            if (meta["models"][i]["gpu"], meta["models"][i]["torch"]) != (locked["gpu"], locked["torch"]):
                raise SystemExit(f"{t}: read on {meta['models'][i]['gpu']} (torch {meta['models'][i]['torch']}), "
                                 f"locked on {locked['gpu']} (torch {locked['torch']})")
        relay = load_relay(a.relay, p, len(z["obs"])) if p in pc.RELAY_P else None
        fb = z[f"fail{idx[B]}"].astype(bool)
        for ci, c in enumerate(locked["configs"]):
            if c["p"] != p:
                continue
            toB, toR = masks(z, meta, idx, c, c["tau"])
            fd = config_fail(z, idx, c, toB, toR, relay)
            fs = z[f"fail{idx[c['S']]}"].astype(bool)
            keep = ~toB & ~toR                               # shots S decides: D = S there
            row = {"i": ci, "type": c["type"], "decoder": pc.decoder_key(c), "p": p, "S": c["S"], "q1": c.get("q1"),
                   "r": c.get("r"), "q": c.get("q"), "score": c.get("score"), "planned_power": c.get("planned_power"),
                   **pc.point(int(fd.sum()), len(fd)),
                   "B_alone": pc.point(int(fb.sum()), len(fb)), "S_alone": pc.point(int(fs.sum()), len(fs)),
                   "to_B": float(toB.mean()), "to_R": float(toR.mean()),
                   "kept_repairs_vs_B": int((fb & ~fs & keep).sum()), "kept_regressions_vs_B": int((~fb & fs & keep).sum()),
                   "vs_B_95": paired(fd, fb, 0.05), "vs_S_95": paired(fd, fs, 0.05),
                   "ni_margin": locked["ni_margin"][str(p)], "vs_B_ni": paired(fd, fb, a_ni)}
            row["noninferior"] = bool(row["vs_B_ni"]["diff_hi"] < row["ni_margin"])
            if toR.any():
                rf = relay[1]
                bf = fb if c["type"] == "S-B-R" else fs
                row["relay_repairs"] = int((bf & ~rf & toR).sum())
                row["relay_regressions"] = int((~bf & rf & toR).sum())
                row["relay_s_mean"] = float(np.nanmean(relay[2][toR]))
            res.append(row)
    out = {"family": fam, "B": B, "locked": os.path.abspath(a.locked), "locked_sha256": pc.sha256_file(a.locked),
           "conf": os.path.abspath(a.conf), "alpha_ni_one_sided": a_ni / 2, "rows": res}
    json.dump(out, open(a.out, "w"), indent=1)
    for r in res:
        v = r["vs_B_95"]
        print(json.dumps({k: r[k] for k in ("i", "type", "p", "S", "fail", "shots", "to_B", "to_R", "noninferior")} |
                         {"B_fail": r["B_alone"]["fail"], "mcnemar_p": v["mcnemar_p"],
                          "diff_95": [v["diff_lo"], v["diff_hi"]]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--dev", required=True)
    s.add_argument("--relay", required=True)
    s.add_argument("--speed", required=True)
    s.add_argument("--p", default="", help="dry runs only: a subset of PS")
    s.add_argument("--out", required=True)
    s = sub.add_parser("confirm")
    s.add_argument("--conf", required=True)
    s.add_argument("--relay", required=True)
    s.add_argument("--locked", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": select, "confirm": confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
