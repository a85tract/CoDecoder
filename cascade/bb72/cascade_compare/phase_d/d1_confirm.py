"""Phase D1's one analysis (PHASE_D1_PLAN.md section 3) and its speed summary (4.1).

  python d1_confirm.py --lock LOCK --F SHA --read DIR --compiled DIR --speed ROWS.jsonl --out DIR [--first-lock F]

The refusals of 3.1 come first: if one fires, nothing is written, the reasons are printed and the exit status is 3.
Then, at each claim point and on each path of C14' (eager; compiled single and compiled batched), D's and C14''s
failures on the same shots (D: S55'-deep on a kept shot, C14' on that path on a routed shot; in batched mode B's
decision from the routed batches) and bb.route.paired; the four comparisons (H1e, H2e, H1c, H2c) under one Holm
procedure at 0.05 (pc_common.holm), a compiled comparison an intersection-union test of single and batched unless they
coincide (1.2); the verdicts as pc_report.headline gives them (3.3); what 3.4 reports; the descriptive rows; and the
speed summary of the timing rows (pc_report.pipes per engine, Cascade alone's rows with both). Each run writes
OUT/run_<k>/ (k one more than the runs already there): d1_result.json and d1_tables.md. Nothing earlier is overwritten."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase_c"))
sys.path.insert(0, HERE)
import d1_common as d1  # noqa: E402
import pc_common as pc  # noqa: E402

PATHS = ("eager", "single", "batched")          # C14''s paths: eager, compiled@1, compiled@4096


class Refusal(Exception):
    pass


def need(ok, why, reasons):
    if not ok:
        reasons.append(why)


def check_read(L, F, p, z, meta, idx, reasons):
    """3.1 on the eager read at p."""
    from bb.cascade_data import eval_seed
    n, c0 = d1.n_chunks(L, p), d1.c0_of(F)
    need(meta["family"] == L["family"] == d1.FAMILY, f"{p}: family {meta['family']}", reasons)
    need(meta["kind"] == L["kind_seed"], f"{p}: seed kind {meta['kind']}, the lock's is {L['kind_seed']}", reasons)
    need(list(meta["chunks"]) == [c0, c0 + n], f"{p}: chunks {meta['chunks']}, not [{c0}, {c0 + n}] for F", reasons)
    cm = meta["chunk_meta"]
    need([c["chunk"] for c in cm] == list(range(c0, c0 + n)), f"{p}: the chunks read are not c0 .. c0 + n - 1",
         reasons)
    need(all(c["seed"] == eval_seed(L["kind_seed"], p, c["chunk"], L["family"]) for c in cm),
         f"{p}: a chunk's seed is not eval_seed(kind, p, j, family)", reasons)
    need(len(z["obs"]) == pc.CHUNK * n, f"{p}: {len(z['obs'])} shots, not {pc.CHUNK * n}", reasons)
    for t in (d1.B, d1.S):
        if t not in idx:
            reasons.append(f"{p}: {t} is not in the read")
            continue
        m = meta["models"][idx[t]]
        need(m["sha256"] == L["nets"][t], f"{p}: {t} sha256 differs from the lock", reasons)
        need(m["prec"] == L["read_prec"][t], f"{p}: {t} read in {m['prec']}, locked {L['read_prec'][t]}", reasons)
        need((m["gpu"], m["torch"]) == (L["gpu"], L["torch"]),
             f"{p}: {t} read on {m['gpu']} (torch {m['torch']}), locked {L['gpu']} (torch {L['torch']})", reasons)
        need(m.get("tree") == d1.tree_tag(F), f"{p}: {t} read in tree {m.get('tree')!r}, not F", reasons)


def check_compiled(L, F, p, meta, read_dir, cj, cz, routed, n, reasons):
    """3.1 on the compiled read at p."""
    need(cj["read_json_sha256"] == pc.sha256_file(os.path.join(read_dir, f"{pc.p_tag(p)}.json")),
         f"{p}: the compiled read is not of this eager read", reasons)
    need(cj["digests_ok"] and cj["shots"] == n and list(cj["chunks"]) == list(meta["chunks"])
         and cj["kind"] == L["kind_seed"], f"{p}: the compiled read's rows are not the eager read's", reasons)
    fp = L["frozen_paths"][d1.B]
    need(cj["paths"] == d1.COMPILED_PATHS and fp["single"] == d1.COMPILED_PATHS["single"]
         and fp["batched"] == d1.COMPILED_PATHS["batched"], f"{p}: compiled paths {cj['paths']} are not the lock's",
         reasons)
    for k in ("check_first", "check_every"):
        need(cj[k]["differ"] == 0 and cj[k]["conf_bits_differ"] == 0 and cj[k]["rows"] > 0,
             f"{p}: the concurrent compiled@1 path failed its {k} check", reasons)
    need(cj["S_anchor"]["pass"], f"{p}: S55'-deep's read engine failed the S anchor", reasons)
    need((cj["gpu"], cj["torch"]) == (L["gpu"], L["torch"]), f"{p}: compiled read on {cj['gpu']} {cj['torch']}",
         reasons)
    need(cj["tree_sha"] == F and cj["tree"] == d1.tree_tag(F), f"{p}: compiled read in tree {cj['tree']!r}", reasons)
    need(cj["sha256"] == L["nets"][d1.B], f"{p}: the compiled read's C14' is not the locked network", reasons)
    need(np.array_equal(cz["routed"], routed), f"{p}: the compiled read's routed rows are not the analysis's", reasons)
    need(len(cz["conf1"]) == n and len(cz["conf4"]) == n and len(cz["confR"]) == len(routed),
         f"{p}: the compiled read's arrays have the wrong lengths", reasons)


def unpack_bits(x):
    return np.unpackbits(x, axis=1, count=pc.K)


def kept_counts(fd, fc, fs, toB):
    """Among the kept shots: breaks (S fails, C14' right), repairs (the reverse), both fail; C14''s failures on routed
    and kept shots (the cap)."""
    keep = ~toB
    return {"routed_frac": float(toB.mean()), "routed": int(toB.sum()),
            "kept_breaks": int((fs & ~fc & keep).sum()), "kept_repairs": int((~fs & fc & keep).sum()),
            "kept_both": int((fs & fc & keep).sum()), "C_fail_routed": int((fc & toB).sum()),
            "C_fail_kept": int((fc & keep).sum()), "D_equals_C_on_routed": bool(np.array_equal(fd[toB], fc[toB]))}


def describe(stats, fd, fc):
    n = len(fd)
    rate_c = fc.sum() / n
    return {"D": pc.point(int(fd.sum()), n), "C": pc.point(int(fc.sum()), n), "stats_95": stats,
            "diff_share_of_C": stats["diff_ler"] / rate_c if rate_c else None}


def analyse(a):
    from bb.route import paired
    import pc_report
    import pc_route
    reasons = []
    L = d1.lock(a.lock)
    lock_sha = pc.sha256_file(a.lock)
    F = a.F
    c0 = d1.c0_of(F)
    try:
        first = d1.first_lock(L, a.first_lock)
        need(d1.configs_copied(L, first), "the lock's configurations are not configurations 2 and 7 of the first lock",
             reasons)
    except SystemExit as e:
        reasons.append(str(e))
    need(L["family"] == d1.FAMILY, f"lock family {L['family']}", reasons)
    per_p, data = {}, {}
    for p in sorted({c["p"] for c in L["configs"]}):
        ci, c = d1.config_at(L, p)
        z, meta, idx = pc.load_read(a.read, p)
        check_read(L, F, p, z, meta, idx, reasons)
        if reasons:
            continue
        n = len(z["obs"])
        toB, toR = pc_route.masks(z, meta, idx, c, c["tau"])
        routed = np.nonzero(toB)[0]
        cj = json.load(open(os.path.join(a.compiled, f"{pc.p_tag(p)}.json")))
        cz = np.load(os.path.join(a.compiled, f"{pc.p_tag(p)}.npz"))
        check_compiled(L, F, p, meta, a.read, cj, cz, routed, n, reasons)
        data[p] = (ci, c, z, meta, idx, toB, toR, routed, cj, cz)
    rows = [json.loads(x) for x in open(a.speed)] if os.path.exists(a.speed) else []
    pipe = [r for r in rows if r.get("role") == "pipeline"]
    need(len(pipe) > 0, f"{a.speed}: no timing rows", reasons)
    bad = [r for r in pipe if r.get("locked_sha256") != lock_sha or r.get("tree") != d1.tree_tag(F)]
    need(not bad, f"{len(bad)} timing rows are not of D1's lock and F", reasons)
    if reasons:
        raise Refusal(reasons)

    # ---------------------------------------------------------------------------------------------- per point, path
    for p, (ci, c, z, meta, idx, toB, toR, routed, cj, cz) in data.items():
        n = len(z["obs"])
        obs = unpack_bits(z["obs"])
        fs = z[f"fail{idx[d1.S]}"].astype(bool)
        fc = {"eager": z[f"fail{idx[d1.B]}"].astype(bool)}
        fd = {"eager": pc_route.config_fail(z, idx, c, toB, toR)}
        b1, b4, bR = unpack_bits(cz["bits1"]), unpack_bits(cz["bits4"]), unpack_bits(cz["bitsR"])
        fc["single"] = (b1 != obs).any(1)
        fc["batched"] = (b4 != obs).any(1)
        fd["single"] = fs.copy()
        fd["single"][toB] = fc["single"][toB]
        fd["batched"] = fs.copy()
        fd["batched"][routed] = (bR != obs[routed]).any(1)
        pe = unpack_bits(z[f"pred{idx[d1.B]}"])
        res = {"p": p, "config": ci, "q1": c["q1"], "tau1": c["tau"]["tau1"], "shots": n, "chunks": meta["chunks"],
               "paths": {}}
        for path in PATHS:
            st = paired(fd[path], fc[path], 0.05)
            res["paths"][path] = {**describe(st, fd[path], fc[path]), **kept_counts(fd[path], fc[path], fs, toB),
                                  "S_alone_vs_C": paired(fs, fc[path], 0.05)}
        res["compiled_coincide"] = bool(np.array_equal(fc["single"], fc["batched"])
                                        and np.array_equal(fd["single"], fd["batched"]))
        diffs = {}                                     # how the compiled paths differ from the eager one
        keep = ~toB
        for path, b in (("single", b1), ("batched", b4)):
            dec = (b != pe).any(1)
            ch = fc[path] != fc["eager"]
            diffs[path] = {"C_decisions_differ": int(dec.sum()), "C_failures_change": int(ch.sum()),
                           "C_fail_eager_only": int((fc["eager"] & ~fc[path]).sum()),
                           "C_fail_compiled_only": int((~fc["eager"] & fc[path]).sum()),
                           "C_failures_change_on_kept": int((ch & keep).sum()),
                           "D_failures_change": int((fd[path] != fd["eager"]).sum())}
        dR = (bR != b4[routed]).any(1)
        diffs["routed_batches_vs_whole"] = {"routed": int(len(routed)), "B_decisions_differ": int(dR.sum())}
        res["compiled_vs_eager"] = diffs
        desc = []                                      # descriptive rows: the first lock's other grid points
        conf_s = z[f"conf{idx[d1.S]}"].astype(np.float64)
        for g in L["grid"]:
            if abs(g["p"] - p) > 1e-12 or abs(g["q1"] - c["q1"]) < 1e-12:
                continue
            tb = conf_s < g["tau1"]
            row = {"q1": g["q1"], "tau1": g["tau1"], "routed_frac": float(tb.mean())}
            for path in ("eager", "single"):
                f = fs.copy()
                f[tb] = fc[path][tb]
                s = paired(f, fc[path], 0.05)
                row[path] = {k: s[k] for k in ("fail_a", "fail_b", "only_a", "only_b", "diff_ler", "diff_lo",
                                               "diff_hi")}
            desc.append(row)
        res["descriptive"] = desc
        res["compiled_read"] = {k: cj[k] for k in ("impl", "check_first", "check_every", "S_anchor", "cache", "secs",
                                                    "driver")}
        per_p[p] = res

    # ------------------------------------------------------------------------------------------- Holm, verdicts
    comps = []
    for cp in L["comparisons"]:
        r = per_p[cp["p"]]
        if cp["path"] == "eager":
            pv = r["paths"]["eager"]["stats_95"]["mcnemar_p"]
        else:
            ps_, pb_ = (r["paths"][m]["stats_95"]["mcnemar_p"] for m in ("single", "batched"))
            pv = ps_ if r["compiled_coincide"] else max(ps_, pb_)
        comps.append({**cp, "mcnemar_p": pv})
    th = pc.holm([x["mcnemar_p"] for x in comps])
    for j, x in enumerate(comps):
        r = per_p[x["p"]]
        x["holm_alpha"] = th[j]
        parts = ("eager",) if x["path"] == "eager" else ("single", "batched")
        at = {}
        for m in parts:
            s = r["paths"][m]["stats_95"]
            at[m] = pc_report.paired_counts(s, th[j]) if th[j] is not None else None
        x["at_holm"] = at
        if th[j] is None:
            x["verdict"] = "level"
        else:
            vs = {at[m]["verdict"] for m in parts}
            x["verdict"] = vs.pop() if len(vs) == 1 else "level"
    out = {"lock": os.path.abspath(a.lock), "lock_sha256": lock_sha, "F": F, "c0": c0,
           "first_lock_sha256": L["source"]["sha256"], "read": os.path.abspath(a.read),
           "compiled": os.path.abspath(a.compiled), "family_size": len(comps), "comparisons": comps,
           "points": {str(p): r for p, r in per_p.items()}, "speed": speed(L, pipe, lock_sha),
           "claims": {}}
    for p in per_p:
        cmp_c = [x for x in comps if x["p"] == p and x["path"] == "compiled"][0]
        cmp_e = [x for x in comps if x["p"] == p and x["path"] == "eager"][0]
        sp = out["speed"]["points"][str(p)]
        faster = {e: sp[e]["faster_both_modes"] and sp[e]["eligible"] for e in ("base", "new")}
        out["claims"][str(p)] = {"compiled_verdict": cmp_c["verdict"], "eager_verdict": cmp_e["verdict"],
                                 "faster": faster,
                                 "more_accurate_and_faster": {e: cmp_c["verdict"] == "better" and faster[e]
                                                              for e in faster}}
    return out


def speed(L, pipe, lock_sha):
    """pc_report.pipes once per engine (today's: base; milestone 2's: new), Cascade alone's rows with both; speed-ups,
    per round, the clock record and the in-process S checks."""
    import pc_report
    res = {"processes": len(pipe), "points": {}}
    for p in sorted({c["p"] for c in L["configs"]}):
        ci, _ = d1.config_at(L, p)
        pt = {}
        for eng in ("base", "new"):
            sel = [r for r in pipe if r["config"] == -1 or r.get("engine") == eng]
            agg = pc_report.pipes(sel, L["family"], lock_sha)
            al_s, al_b = agg.get((-1, p, "single")), agg.get((-1, p, "batched"))
            d_s, d_b = agg.get((ci, p, "single")), agg.get((ci, p, "batched"))
            procs = [r for r in sel if r["p"] == p]
            dproc = [r for r in procs if r["config"] == ci]
            s_ok = all(r.get("S_check") and r["S_check"]["differ"] == 0 and r["S_check"]["margin_bits_differ"] == 0
                       for r in dproc)
            have = all(x is not None for x in (al_s, al_b, d_s, d_b))
            e = {"alone_single": al_s, "alone_batched": al_b, "D_single": d_s, "D_batched": d_b,
                 "S_check_all_zero": s_ok, "eligible": False, "faster_both_modes": False}
            if have:
                e["eligible"] = s_ok and min(x["processes"] for x in (al_s, al_b, d_s, d_b)) >= pc.REPS
                e["speedup_single_mean"] = al_s["mean_us"] / d_s["mean_us"]
                e["speedup_batched"] = al_b["per_shot_us"] / d_b["per_shot_us"]
                e["per_round_us"] = {"alone_single": al_s["mean_us"] / pc.R, "D_single": d_s["mean_us"] / pc.R,
                                     "alone_batched": al_b["per_shot_us"] / pc.R,
                                     "D_batched": d_b["per_shot_us"] / pc.R}
                e["faster_both_modes"] = d_s["mean_us"] < al_s["mean_us"] and d_b["per_shot_us"] < al_b["per_shot_us"]
                e["slower_tail"] = d_s["p999_us"] > al_s["p999_us"]
            e["clock"] = [{"config": r["config"], "mode": r["metric"], "rep": r["rep"],
                           "min_ratio": (r.get("clock") or {}).get("clock_min_ratio"),
                           "accepted": (r.get("clock") or {}).get("accepted"),
                           "other_pids": (r.get("clock") or {}).get("other_pids")} for r in procs]
            e["fails"] = sorted({(r["config"], r["metric"], r["fail"], r["shots"]) for r in procs})
            pt[eng] = e
        res["points"][str(p)] = pt
    return res


def fmt_e(x, d=2):
    return "-" if x is None else f"{x:.{d}e}"


def tables(out):
    """d1_tables.md: the four comparisons, the details per point and path, the compiled paths against eager, the
    descriptive rows and the speed table."""
    L = []
    L.append("| comparison | p | path | D / C14' failures (shots) | only D / only C14' | McNemar p | Holm level | "
             "difference [95 %] | interval at the Holm level | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in out["comparisons"]:
        r = out["points"][str(x["p"])]
        parts = ("eager",) if x["path"] == "eager" else ("single", "batched")
        for m in parts:
            s = r["paths"][m]["stats_95"]
            h = x["at_holm"].get(m) if x["at_holm"] else None
            hl = f"[{fmt_e(h['diff_lo'])}, {fmt_e(h['diff_hi'])}]" if h else "-"
            L.append(f"| {x['name']} | {100 * x['p']:.1f} % | {m if x['path'] == 'compiled' else 'eager'} | "
                     f"{s['fail_a']:,} / {s['fail_b']:,} ({s['shots']:,}) | {s['only_a']} / {s['only_b']} | "
                     f"{s['mcnemar_p']:.3g} | {x['holm_alpha'] if x['holm_alpha'] is not None else 'none'} | "
                     f"{fmt_e(s['diff_ler'])} [{fmt_e(s['diff_lo'])}, {fmt_e(s['diff_hi'])}] | {hl} | {x['verdict']} |")
        if x["path"] == "compiled":
            L.append(f"| {x['name']} | | compiled (p = {x['mcnemar_p']:.3g}) | | | | | | | {x['verdict']} |")
    L.append("")
    L.append("| p | path | routed | kept: breaks / repairs / both fail | C14' failures routed / kept | "
             "share of C14''s rate | S55'-deep alone / C14' |")
    L.append("|---|---|---|---|---|---|---|")
    for p, r in out["points"].items():
        for m in PATHS:
            q = r["paths"][m]
            sa = q["S_alone_vs_C"]
            L.append(f"| {100 * float(p):.1f} % | {m} | {q['routed']:,} ({100 * q['routed_frac']:.2f} %) | "
                     f"{q['kept_breaks']} / {q['kept_repairs']} / {q['kept_both']} | {q['C_fail_routed']:,} / "
                     f"{q['C_fail_kept']:,} | {100 * q['diff_share_of_C']:.2f} % | {sa['fail_a']:,} / {sa['fail_b']:,} |")
    L.append("")
    L.append("| p | compiled path | C14' decisions differing from eager | failures changed (eager only / compiled only) "
             "| of them on kept shots | D failures changed |")
    L.append("|---|---|---|---|---|---|")
    for p, r in out["points"].items():
        for m in ("single", "batched"):
            q = r["compiled_vs_eager"][m]
            L.append(f"| {100 * float(p):.1f} % | {m} | {q['C_decisions_differ']} | {q['C_failures_change']} "
                     f"({q['C_fail_eager_only']} / {q['C_fail_compiled_only']}) | {q['C_failures_change_on_kept']} | "
                     f"{q['D_failures_change']} |")
    L.append("")
    L.append("| p | q1 | routed | eager: D / C14' (only D / only C14'), difference [95 %] | compiled@1: the same |")
    L.append("|---|---|---|---|---|")
    for p, r in out["points"].items():
        for g in r["descriptive"]:
            cells = []
            for m in ("eager", "single"):
                s = g[m]
                cells.append(f"{s['fail_a']:,} / {s['fail_b']:,} ({s['only_a']} / {s['only_b']}), "
                             f"{fmt_e(s['diff_ler'])} [{fmt_e(s['diff_lo'])}, {fmt_e(s['diff_hi'])}]")
            L.append(f"| {100 * float(p):.1f} % | {100 * g['q1']:.0f} % | {100 * g['routed_frac']:.2f} % | "
                     f"{cells[0]} | {cells[1]} |")
    L.append("")
    L.append("| p | engine | single mean, us (per round): D / Cascade | speed-up | p99.9, us: D / Cascade | "
             "batched, us per shot (per round): D / Cascade | speed-up | processes | clock flags | S checks |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p, pt in out["speed"]["points"].items():
        for eng, e in pt.items():
            if e["D_single"] is None or e["alone_single"] is None or e["D_batched"] is None:
                L.append(f"| {100 * float(p):.1f} % | {eng} | missing | | | | | | | |")
                continue
            ds, as_, db, ab = e["D_single"], e["alone_single"], e["D_batched"], e["alone_batched"]
            flags = sum(1 for x in e["clock"] if not x["accepted"])
            L.append(f"| {100 * float(p):.1f} % | {eng} | {ds['mean_us']:.1f} ({ds['mean_us'] / 6:.1f}) / "
                     f"{as_['mean_us']:.1f} ({as_['mean_us'] / 6:.1f}) | {e['speedup_single_mean']:.2f} | "
                     f"{ds['p999_us']:.0f} / {as_['p999_us']:.0f} | {db['per_shot_us']:.2f} "
                     f"({db['per_shot_us'] / 6:.3f}) / {ab['per_shot_us']:.2f} ({ab['per_shot_us'] / 6:.2f}) | "
                     f"{e['speedup_batched']:.2f} | {min(x['processes'] for x in (ds, as_, db, ab))} | "
                     f"{flags} of {len(e['clock'])} | {'all zero' if e['S_check_all_zero'] else 'DIFFER'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", default=d1.LOCK)
    ap.add_argument("--F", required=True)
    ap.add_argument("--read", required=True)
    ap.add_argument("--compiled", required=True)
    ap.add_argument("--speed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-lock", default=None)
    a = ap.parse_args()
    try:
        out = analyse(a)
    except Refusal as e:
        for r in e.args[0]:
            print(f"REFUSED: {r}", flush=True)
        sys.exit(3)
    os.makedirs(a.out, exist_ok=True)
    k = 1 + len([x for x in os.listdir(a.out) if x.startswith("run_")])
    d = os.path.join(a.out, f"run_{k}")
    os.makedirs(d)
    json.dump(out, open(os.path.join(d, "d1_result.json"), "w"), indent=1, default=str)
    open(os.path.join(d, "d1_tables.md"), "w").write(tables(out))
    print(tables(out), flush=True)
    print(json.dumps({"run": d, "verdicts": {x["name"]: x["verdict"] for x in out["comparisons"]}}), flush=True)


if __name__ == "__main__":
    main()
