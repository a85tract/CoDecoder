"""summarize.py: ARB3's timing of the joint decision (TIMING.md) from drive.py's records.

    python summarize.py --rec CARD=DIR ... --out timing.json [--md timing_tables.md]

Each DIR is one drive.py OUT (attempts.json + slots/), of round r1 (BB72) or r4 (d = 13), run with --joint-both. Per
setting (code, p), card and mode, three decoders: Cascade alone (r1 config -1, r4 C2), the composition (r1 config >= 0
on the BB engine, r4 C5) and the composition with the joint step (slot id + '.joint'). For every slot, the established
attempt's row is used (a slot that was not established is reported and leaves its cell empty).
  single   mean and 99.9th percentile of the per-shot latency (us), each process's own over its 20,000 timed shots
  batched  wall time per shot and per syndrome round (us; per round = per shot / 6 on BB72, / 13 at d = 13)
A composition's value is the slowest of its three processes and Cascade alone's the fastest of its three, statistic by
statistic (as Table I). Speed-up = Cascade alone's value / the composition's. The joint step's added time per incoming
shot is (a) the joint composition's value minus the composition's, and (b) the mean over the repetitions of the paired
difference (repetition k with the joint step minus repetition k without; the same shots), with its range. Per routed
shot it is that difference divided by the routed fraction, and also, measured on the routed shots alone: single mode, the
paired difference of the mean via-B latency; batched mode, the paired difference of the B stage's wall time (BB72's
b_stage_s, d = 13's equivalent when present) divided by the number of routed shots."""
import argparse
import json
import os
import statistics as st

ROUNDS_PER_SHOT = {"bb72": 6, "d13": 13}


def load(card, d):
    A = json.load(open(os.path.join(d, "attempts.json")))
    out = []
    for s in A["slots"]:
        k = s.get("accepted_attempt")
        row = None
        if k is not None:
            f = os.path.join(d, s["attempts"][k]["dir"], "rows.jsonl")
            rows = [json.loads(x) for x in open(f) if x.strip()]
            row = rows[-1] if rows else None
        m = s["meta"]
        sid = s["id"]
        if A["round"] == "r1":
            code, dec = "bb72", ("alone" if m["config"] < 0 else ("joint" if sid.endswith(".joint") else "comp"))
            if m["config"] >= 0 and m.get("engine") != "base":
                continue
        else:
            if m["config"] not in ("C2", "C5"):
                continue
            code, dec = "d13", ("alone" if m["config"] == "C2" else ("joint" if sid.endswith(".joint") else "comp"))
        out.append(dict(card=card, code=code, p=m["p"], mode=m["mode"], rep=m["rep"], dec=dec, id=sid,
                        status=s["status"], attempts=len(s["attempts"]), row=row, round=A["round"], dry=A.get("dry"),
                        host=A.get("host"), tree=A.get("tree_deployed"), sets_file=A.get("sets_file")))
    return out


def stats(r, code):
    """The row's statistics: single {mean, p999, via_b, n_via_b, frac}, batched {per_shot, per_round, b_stage_s, frac}."""
    row = r["row"]
    if row is None:
        return None
    frac = row.get("frac_B")
    if r["mode"] == "single":
        lat = row.get("latency_us") or {}
        vb = row.get("latency_via_B_us") or {}
        return dict(mean=lat.get("mean"), p999=lat.get("p999"), via_b=vb.get("mean"), n_via_b=vb.get("n"), frac=frac,
                    fail=row.get("fail"), shots=row.get("shots", lat.get("n")))
    ps = row.get("per_shot_us")
    pr = row.get("per_round_us", ps / ROUNDS_PER_SHOT[code] if ps is not None else None)
    bs = row.get("b_stage_s")
    return dict(per_shot=ps, per_round=pr, b_stage_s=bs, frac=frac, fail=row.get("fail"), shots=row.get("shots"))


def agg(vals, how):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return max(v) if how == "slowest" else min(v)


def summarize(recs):
    cells = {}
    for r in recs:
        cells.setdefault((r["card"], r["code"], r["p"], r["mode"]), {}).setdefault(r["dec"], []).append(r)
    res = []
    order = lambda kv: (kv[0][0] != "RTX", kv[0][1] != "bb72", kv[0][2] if kv[0][1] == "bb72" else -kv[0][2],
                        kv[0][3] != "single")                    # Table I's order: BB72 0.2, 0.3 %; d = 13 0.6, 0.35 %
    for (card, code, p, mode), D in sorted(cells.items(), key=order):
        keys = ("mean", "p999") if mode == "single" else ("per_shot", "per_round")
        E = {"card": card, "code": code, "p": p, "mode": mode, "decoders": {}}
        S = {}
        for dec in ("alone", "comp", "joint"):
            rs = sorted(D.get(dec, []), key=lambda r: r["rep"])
            S[dec] = {r["rep"]: stats(r, code) for r in rs}
            how = "fastest" if dec == "alone" else "slowest"
            E["decoders"][dec] = dict(
                how=how, reps={k: v for k, v in S[dec].items()},
                status={r["rep"]: r["status"] for r in rs}, attempts={r["rep"]: r["attempts"] for r in rs},
                value={k: agg([v[k] if v else None for v in S[dec].values()], how) for k in keys},
                frac=st.mean([v["frac"] for v in S[dec].values() if v and v.get("frac") is not None])
                if any(v and v.get("frac") is not None for v in S[dec].values()) else None)
        a = E["decoders"]["alone"]["value"]
        for dec in ("comp", "joint"):
            v = E["decoders"][dec]["value"]
            E["decoders"][dec]["speedup"] = {k: (a[k] / v[k] if a.get(k) and v.get(k) else None) for k in keys}
        # the joint step's added time
        k0 = keys[0] if mode == "single" else "per_shot"
        vc, vj = E["decoders"]["comp"]["value"].get(k0), E["decoders"]["joint"]["value"].get(k0)
        reps = sorted(set(S["comp"]) & set(S["joint"]))
        paired = [S["joint"][k][k0] - S["comp"][k][k0] for k in reps if S["joint"][k] and S["comp"][k]]
        frac = E["decoders"]["joint"]["frac"]
        add = dict(stat=k0, slowest_minus_slowest=(vj - vc) if vc is not None and vj is not None else None,
                   paired=paired, paired_mean=st.mean(paired) if paired else None,
                   paired_range=[min(paired), max(paired)] if paired else None, routed_frac=frac)
        add["per_routed_from_incoming"] = (add["paired_mean"] / frac) if paired and frac else None
        if mode == "single":
            pv = [S["joint"][k]["via_b"] - S["comp"][k]["via_b"] for k in reps
                  if S["joint"][k] and S["comp"][k] and S["joint"][k]["via_b"] is not None]
            add["per_routed_via_b_paired"] = pv
            add["per_routed_via_b_mean"] = st.mean(pv) if pv else None
            add["per_incoming_via_b"] = add["per_routed_via_b_mean"] * frac if pv and frac else None
        else:
            pv = []                              # B stage's wall time per routed shot, paired
            for k in reps:
                j, c = S["joint"][k], S["comp"][k]
                if j and c and j.get("b_stage_s") is not None and c.get("b_stage_s") is not None and j.get("frac"):
                    pv.append((j["b_stage_s"] - c["b_stage_s"]) * 1e6 / (j["frac"] * j["shots"]))
            add["per_routed_b_stage_paired"] = pv
            add["per_routed_b_stage_mean"] = st.mean(pv) if pv else None
            add["per_incoming_b_stage"] = add["per_routed_b_stage_mean"] * frac if pv and frac else None
        if mode == "batched":
            add["per_round_slowest_minus_slowest"] = (
                E["decoders"]["joint"]["value"]["per_round"] - E["decoders"]["comp"]["value"]["per_round"]) \
                if E["decoders"]["joint"]["value"].get("per_round") is not None and \
                E["decoders"]["comp"]["value"].get("per_round") is not None else None
        E["joint_added"] = add
        E["fails"] = {dec: {k: (v or {}).get("fail") for k, v in S[dec].items()} for dec in S}
        res.append(E)
    return res


def fmt(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def markdown(res):
    L = []
    for E in res:
        d = E["decoders"]
        pc = f"{E['p'] * 100:g} %"
        L.append(f"### {E['card']}, {'BB72' if E['code'] == 'bb72' else 'd = 13'}, p = {pc}, {E['mode']}")
        if E["mode"] == "single":
            L.append("| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |")
            L.append("|---|---|---|---|---|---|")
            for dec, name in (("alone", "Cascade alone"), ("comp", "composition"), ("joint", "with the joint step")):
                v = d[dec]["value"]
                reps = "; ".join(f"{fmt(r['mean'])} / {fmt(r['p999'])}" for _, r in sorted(d[dec]["reps"].items()) if r)
                sp = "" if dec == "alone" else fmt(d[dec]["speedup"]["mean"], 2)
                fr = "" if dec == "alone" else f"{(d[dec]['frac'] or 0) * 100:.2f} %"
                L.append(f"| {name} | {fmt(v['mean'])} | {fmt(v['p999'])} | {sp} | {fr} | {reps} |")
        else:
            L.append("| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |")
            L.append("|---|---|---|---|---|---|")
            for dec, name in (("alone", "Cascade alone"), ("comp", "composition"), ("joint", "with the joint step")):
                v = d[dec]["value"]
                reps = "; ".join(fmt(r["per_round"], 4) for _, r in sorted(d[dec]["reps"].items()) if r)
                sp = "" if dec == "alone" else fmt(d[dec]["speedup"]["per_round"], 2)
                fr = "" if dec == "alone" else f"{(d[dec]['frac'] or 0) * 100:.2f} %"
                L.append(f"| {name} | {fmt(v['per_round'], 4)} | {fmt(v['per_shot'], 4)} | {sp} | {fr} | {reps} |")
        a = E["joint_added"]
        nd = 2 if E["mode"] == "single" else 4
        line = (f"Joint step, added per incoming shot ({a['stat']}): slowest - slowest {fmt(a['slowest_minus_slowest'], nd)} us; "
                f"paired mean {fmt(a['paired_mean'], nd)} us (range {fmt((a['paired_range'] or [None])[0], nd)} to "
                f"{fmt((a['paired_range'] or [None, None])[1], nd)}); per routed shot {fmt(a['per_routed_from_incoming'], 2)} us")
        if E["mode"] == "single":
            line += (f". On the routed shots alone (via-B latency, paired mean over repetitions): "
                     f"{fmt(a['per_routed_via_b_mean'], 2)} us per routed shot "
                     f"({', '.join(fmt(x, 2) for x in a['per_routed_via_b_paired'])}), "
                     f"{fmt(a['per_incoming_via_b'], 3)} us per incoming shot")
        else:
            line += (f". On the routed shots alone (B stage's wall time, paired mean): "
                     f"{fmt(a['per_routed_b_stage_mean'], 3)} us per routed shot "
                     f"({', '.join(fmt(x, 3) for x in a['per_routed_b_stage_paired'])}), "
                     f"{fmt(a['per_incoming_b_stage'], 4)} us per incoming shot")
        L.append("")
        L.append(line + ".")
        L.append(f"Failures per process (descriptive): {json.dumps(E['fails'])}")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", nargs="+", required=True, help="CARD=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    recs = []
    for x in a.rec:
        card, d = x.split("=", 1)
        recs += load(card, d)
    res = summarize(recs)
    bad = [dict(card=r["card"], id=r["id"], status=r["status"], attempts=r["attempts"]) for r in recs
           if r["status"] != "established" or r["attempts"] > 1]
    with open(a.out, "w") as fh:
        json.dump(dict(cells=res, not_first_attempt=bad), fh, indent=1, default=str)
    md = markdown(res)
    if a.md:
        open(a.md, "w").write(md + "\n")
    print(md)
    print("slots not established at the first attempt:", json.dumps(bad))


if __name__ == "__main__":
    main()
