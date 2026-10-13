"""Phase C analysis (PHASE_C_PLAN.md sections 3-6): the tables of the final report and the mechanical verdicts.

  python pc_report.py --curves uniform=DIR[,DIR2] --curves depspam_fm=DIR[,DIR2] [--confirm F.json ...] \
      [--check check_<setting>.json ...] [--locked locked_<setting>.json ...] \
      [--speed speed_components.json --speed-rows rows.jsonl] [--speed-late speed_components.json] \
      [--paper-timing paper_timing.json] --out DIR_OUT

Curves: every network alone on the development block (kind 10) of each setting, as pc_eval.py read it (a ConvLines
network on its fused engine, the path it is timed on; the table names each read's precision): block failures, the 95 %
Wilson interval, P_L by the paper's Eq. (1) at the estimate and at both interval ends (Eq. (1) is applied to rates
only, never to a difference), and each network against Cascade of the setting (C14 / C14') on the same shots: both
discordant directions, the exact McNemar test and Newcombe's matched-pair interval. A second directory per setting (the
C6' add-on) must hold the same shots: its chunks' detector digests must equal the first directory's. Routing: the
confirmation results of pc_route confirm.

The headline family (gate findings 2-3): every comparison with Cascade that can make or veto a headline, in both
settings (our networks alone at every p of the curves, every locked configuration of both confirmations), is judged by
one Holm procedure at ALPHA = 0.05; a rejected comparison is 'better' or 'worse' when Newcombe's interval at its Holm
level excludes 0 on that side, everything else is 'level'. Speed: pc_speed's frozen components and rows; a pipeline's
value is the slowest of all its processes (at least REPS = 3), as for the networks, and it is compared with Cascade
alone timed in the same job(s) (C7), a network alone with Cascade in the networks' job (C3). The speed file must be the
one the locks froze, and every confirmation and check must come from its setting's lock (--locked). Eligibility (gate
finding 1): a decoder, and Cascade itself, enter a headline only with every frozen path screened (for a routed decoder
its S's too), REPS processes behind every value, the same GPU model, and every claim-point check passed (pc_speed
check). Verdicts: PHASE_C_PLAN.md section 6, computed, not judged."""
import argparse
import json
import os

import numpy as np

import pc_common as pc

PAPER = {  # cascade_compare/README.md, Fig. 2(a) digitised (per logical qubit per round); '~' values not used
    "Cascade": {0.001: 8.17e-7, 0.002: 1.04e-5, 0.003: 6.58e-5, 0.004: 2.19e-4},
    "Tesseract": {0.001: 5.74e-7, 0.0015: 2.90e-6, 0.002: 8.13e-6, 0.0025: 2.18e-5, 0.003: 4.73e-5, 0.0035: 9.44e-5,
                  0.004: 1.64e-4, 0.0045: 2.95e-4, 0.005: 5.59e-4, 0.0055: 9.63e-4},
    "Relay": {0.001: 1.41e-6, 0.0015: 6.46e-6, 0.002: 2.00e-5, 0.0025: 4.39e-5, 0.003: 9.50e-5, 0.0035: 1.86e-4,
              0.004: 3.54e-4, 0.0045: 5.59e-4, 0.005: 8.75e-4, 0.0055: 1.23e-3},
}
MODES_NET, MODES_PIPE = ("single", "amortized"), ("single", "batched")


def e(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}e}"


def r1(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def paired_counts(s, alpha):
    """bb.route.paired at another alpha, from a paired result's counts (McNemar's test and Newcombe's interval depend on
    the counts only)."""
    from bb.route import paired
    both = s["fail_a"] - s["only_a"]
    fa, fb = np.zeros(s["shots"], bool), np.zeros(s["shots"], bool)
    fa[:both + s["only_a"]] = True
    fb[:both] = True
    fb[both + s["only_a"]:both + s["only_a"] + s["only_b"]] = True
    return paired(fa, fb, alpha)


def curves(dirs, fam):
    """{p: {tag: point}} and the paired rows against B, from one or two reads of the same shots."""
    from bb.route import paired
    B = pc.BIG[fam]
    pts, comp = {}, []
    for p in pc.PS:
        reads = []
        for d in dirs:
            if os.path.exists(os.path.join(d, f"{pc.p_tag(p)}.npz")):
                reads.append(pc.load_read(d, p))
        if not reads:
            continue
        z0, m0, i0 = reads[0]
        for z, m, _ in reads[1:]:
            if pc.det_digests(m) != pc.det_digests(m0) or m["family"] != m0["family"]:
                raise SystemExit(f"{fam} at {p}: the add-on read holds other shots than the first read")
        fails, prec = {}, {}
        for z, m, idx in reads:
            for t, i in idx.items():
                fails[t] = z[f"fail{i}"].astype(bool)
                prec[t] = m["models"][i]["prec"] + (" compiled" if m["models"][i].get("compiled") else "")
        pts[p] = {t: dict(pc.point(int(f.sum()), len(f)), read=prec[t]) for t, f in fails.items()}
        if B in fails:
            for t, f in fails.items():
                if t != B:
                    comp.append({"family": fam, "p": p, "net": t, "ours": t in pc.OURS, "stats": paired(f, fails[B], 0.05)})
    return pts, comp


def headline(comps, conf):
    """One Holm procedure at ALPHA over the headline family of both settings; sets each member's 'verdict' and
    'holm_alpha' (None when not rejected). Returns the family's size."""
    fam_rows = [r for c in comps for r in c if r["ours"]] + [r for cf in conf for r in cf["rows"]]
    stats = [r["stats"] if "stats" in r else r["vs_B_95"] for r in fam_rows]
    th = pc.holm([s["mcnemar_p"] for s in stats])
    for j, r in enumerate(fam_rows):
        r["holm_alpha"] = th[j]
        r["verdict"] = "level" if th[j] is None else paired_counts(stats[j], th[j])["verdict"]
    return len(fam_rows)


def curve_table(pts, fam):
    """The paper's digitised values sit beside depspam_fm only (README: our uniform circuits are about 1.75x easier,
    so numbers from them must not sit beside Cascade's curve)."""
    tags = [t for t in pc.NETS if any(t in v for v in pts.values())]
    paper = ("Cascade", "Tesseract", "Relay") if fam == "depspam_fm" else ()
    head = "| p (%) | " + " | ".join(tags) + "".join(f" | paper {k}" for k in paper) + " |"
    read = {t: q["read"] for v in pts.values() for t, q in v.items()}
    out = [f"P_L per logical qubit per round (Eq. (1)), {fam}; block failures / shots below each value. Read in: "
           + ", ".join(f"{t} {read[t]}" for t in tags) + f" (fused: {pc.PRECISION['fused']})", "", head,
           "|" + "---|" * (len(tags) + 1 + len(paper))]
    for p in sorted(pts):
        cells = []
        for t in tags:
            q = pts[p].get(t)
            cells.append("-" if q is None else f"{e(q['pl'])} [{e(q['pl_lo'], 1)}, {e(q['pl_hi'], 1)}]<br>"
                                               f"{q['fail']:,} / {q['shots']:,}")
        out.append(f"| {100 * p:.2f} | " + " | ".join(cells) + "".join(f" | {e(PAPER[k].get(p))}" for k in paper)
                   + " |")
    return "\n".join(out)


def comp_table(comp, fam):
    B = pc.BIG[fam]
    out = [f"Each network against {B} on the same shots ({fam}): failures of the network only / of {B} only, "
           "difference of block failure rates with Newcombe's 95 % interval, exact McNemar p; the verdict of the one "
           "Holm procedure over the headline family (our networks; the Cascade networks are descriptive)", "",
           "| p (%) | network | only it | only " + B + " | difference [95 %] | McNemar p | Holm level | verdict |",
           "|---|---|---|---|---|---|---|---|"]
    for r in comp:
        s = r["stats"]
        out.append(f"| {100 * r['p']:.2f} | {r['net']} | {s['only_a']:,} | {s['only_b']:,} | "
                   f"{e(s['diff_ler'])} [{e(s['diff_lo'], 1)}, {e(s['diff_hi'], 1)}] | {e(s['mcnemar_p'], 1)} | "
                   f"{e(r.get('holm_alpha'), 1)} | {r.get('verdict', 'descriptive')} |")
    return "\n".join(out)


def net_ok(sp, t, chk, gpu):
    """Why network t's speed cannot enter a headline in a setting (empty: it can): its frozen paths screened, REPS
    processes behind each value, the check of both modes at both claim points passed."""
    v = sp["nets"].get(t)
    if v is None:
        return ["not timed"]
    why = (["no screened path"] if v["no_path_agrees"] else []) + \
          ([f"{v['processes']} processes"] if v["processes"] < pc.REPS else [])
    rows = [r for r in (chk or {}).get("rows", []) if r["what"] == "alone" and r["net"] == t]
    for p in pc.RELAY_P:
        for m in MODES_NET:
            x = [r for r in rows if r["p"] == p and r["mode"] == m]
            if not x:
                why.append(f"not checked: {m} at {p}")
            elif not x[0]["pass"]:
                why.append(f"check failed: {m} at {p}")
    if chk and chk.get("gpu") != gpu:
        why.append("checked on another GPU model")
    return why


def pipes(rows, fam, sha):
    """The pipelines of one lock: (config, p, mode) -> the slowest of all their processes (gate finding 4); config -1 is
    Cascade alone, timed in the same job(s) beside them."""
    out = {}
    for x in rows:
        if x.get("role") == "pipeline" and x["family"] == fam and x["locked_sha256"] == sha:
            out.setdefault((x["config"], x["p"], x["metric"]), []).append(x)
    agg = {}
    for k, xs in out.items():
        a = {"processes": len(xs), "gpus": sorted({x["gpu"] for x in xs}),
             "sessions": sorted({x["session"] for x in xs}), "S": xs[0]["S"], "path_S": xs[0]["path_S"],
             "path_B": xs[0].get("path_B"), "frac_B": [min(x["frac_B"] for x in xs), max(x["frac_B"] for x in xs)],
             "frac_R": [min(x["frac_R"] for x in xs), max(x["frac_R"] for x in xs)], "type": xs[0]["type"]}
        if k[2] == "single":
            a.update(mean_us=max(x["latency_us"]["mean"] for x in xs), p999_us=max(x["latency_us"]["p999"] for x in xs))
        else:
            a.update(per_shot_us=max(x["per_shot_us"] for x in xs))
        agg[k] = a
    return agg


def verdicts(fam, comp, conf, sp, rows, chk):
    """PHASE_C_PLAN.md section 6, mechanically."""
    B = pc.BIG[fam]
    if not sp or B not in sp.get("nets", {}):
        return {"note": "no speed results"}
    gpu = sp["gpu"]
    c_t1, c_a, c_p999 = sp["nets"][B]["t1_us"], sp["nets"][B]["a_best_us"], sp["nets"][B]["p999_us"]
    why_B = net_ok(sp, B, chk, gpu)
    out = {"B": B, "gpu": gpu, "cascade_t1_us": c_t1, "cascade_a_us": c_a, "cascade_p999_us": c_p999,
           "cascade_not_eligible": why_B, "decoders": []}
    for t in pc.OURS:
        acc = {r["p"]: r["verdict"] for r in comp if r["net"] == t}
        v = sp["nets"].get(t)
        if v is None and not acc:
            continue
        why = net_ok(sp, t, chk, gpu)
        fast = bool(v) and v["t1_us"] < c_t1 and v["a_best_us"] < c_a
        out["decoders"].append({"decoder": f"{t} alone", "accuracy": acc, "faster": fast, "not_eligible": why,
                                "faster_tail": bool(v) and v["p999_us"] is not None and c_p999 is not None and
                                v["p999_us"] < c_p999,
                                "beats": not why and not why_B and fast and
                                all(acc.get(p) == "better" for p in pc.RELAY_P) and "worse" not in acc.values()})
    groups = {}
    for cf in conf:
        if cf["family"] != fam:
            continue
        agg = pipes(rows, fam, cf["locked_sha256"])
        ck = [r for r in (chk or {}).get("rows", []) if r["what"] == "routed"]
        if chk and chk["locked_sha256"] != cf["locked_sha256"]:
            raise SystemExit(f"{fam}: the check and the confirmation come from different locks")
        for r in cf["rows"]:
            x = {"p": r["p"], "S": r["S"], "verdict": r["verdict"], "i": r["i"]}
            if r["p"] in pc.RELAY_P:
                vS = sp["nets"].get(r["S"])                 # plan 6, condition 4: S's own frozen paths screened
                why = [f"S {r['S']} not timed"] if vS is None else \
                      ([f"S {r['S']}: no screened path"] if vS["no_path_agrees"] else [])
                for m in MODES_PIPE:
                    a, ca = agg.get((r["i"], r["p"], m)), agg.get((-1, r["p"], m))
                    c = [y for y in ck if y["config"] == r["i"] and y["mode"] == m]
                    if a is None:
                        why.append(f"{m} not timed")
                    else:
                        why += ([f"{m}: {a['processes']} processes"] if a["processes"] < pc.REPS else []) + \
                               ([f"{m}: timed on {a['gpus']}"] if a["gpus"] != [gpu] else [])
                    if ca is None:                          # Cascade alone, timed beside the pipelines (C7)
                        why.append(f"{m}: Cascade alone not timed beside it")
                    elif a is not None and ca["sessions"] != a["sessions"]:
                        why.append(f"{m}: Cascade alone timed in jobs {ca['sessions']}, the pipeline in "
                                   f"{a['sessions']}")
                    else:
                        why += ([f"{m}: Cascade alone {ca['processes']} processes"]
                                if ca["processes"] < pc.REPS else []) + \
                               ([f"{m}: Cascade alone timed on {ca['gpus']}"] if ca["gpus"] != [gpu] else [])
                    why += [f"{m} not checked"] if not c else ([] if c[0]["pass"] else [f"{m} check failed"])
                s1, bt = agg.get((r["i"], r["p"], "single")), agg.get((r["i"], r["p"], "batched"))
                cs1, cbt = agg.get((-1, r["p"], "single")), agg.get((-1, r["p"], "batched"))
                x.update(single_mean_us=s1 and s1["mean_us"], single_p999_us=s1 and s1["p999_us"],
                         amortized_us=bt and bt["per_shot_us"], not_eligible=why,
                         cascade_single_mean_us=cs1 and cs1["mean_us"], cascade_single_p999_us=cs1 and cs1["p999_us"],
                         cascade_amortized_us=cbt and cbt["per_shot_us"],
                         faster=bool(s1 and bt and cs1 and cbt and s1["mean_us"] < cs1["mean_us"]
                                     and bt["per_shot_us"] < cbt["per_shot_us"]),
                         faster_tail=bool(s1 and cs1 and s1["p999_us"] < cs1["p999_us"]))
            groups.setdefault(r["decoder"], []).append(x)
    for key, rs in groups.items():                          # a routed decoder: its locked configurations over p
        at = [x for x in rs if x["p"] in pc.RELAY_P]
        why = [f"{x['p']}: {w}" for x in at for w in x["not_eligible"]]
        out["decoders"].append({"decoder": key, "by_p": rs, "not_eligible": why,
                                "faster": bool(at) and all(x["faster"] for x in at),
                                "faster_tail": bool(at) and all(x["faster_tail"] for x in at),
                                "beats": not why and not why_B and len(at) == len(pc.RELAY_P) and
                                all(x["faster"] and x["verdict"] == "better" for x in at)
                                and not any(x["verdict"] == "worse" for x in rs)})
    return out


def speed_tables(sp, late, paper):
    """Every path of every network (MED 7: per-round values = per shot / 6, the precision of each path, the paper's
    timings in their own column, never in a verdict)."""
    pt = {}
    for v in (paper or {}).get("values", []):
        pt.setdefault(v["net"], []).append(f"{v['us_per_round']} us/round ({v['label']})")
    out = [f"Speed on {sp['gpu']} (driver {sp.get('driver')}, torch {sp.get('torch')}): every path of every network, "
           "the slowest of its processes. Single shot = host round "
           "trip at batch 1; amortized = host to host per shot at the batch; per round = per shot / 6. 'frozen' marks "
           "the paths the lock froze (single, amortized, batched pipelines); screened = the screen of PHASE_C_PLAN.md "
           "2.4 (failures at 0.55 %, routing at the claim points, a ConvLines path bitwise), per batch size; bitwise = "
           "every decision and confidence of the screen's shots equal to the accuracy path's (a ConvLines network's: "
           "its fused read). "
           "The paper's column (H200, FP8 through torchao; " + ((paper or {}).get("source") or "not digitised yet") +
           ") is external context.", "",
           "| network | path | precision | single shot mean (us) | per round (us) | p99.9 (us) | GPU term (us) | "
           "amortized @1,024 (us) | @4,096 (us) | @4,096 per round (us) | processes | screened | bitwise | frozen | "
           "paper (H200, FP8) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    nets = list(sp["nets"].items()) + [(t, dict(v, late=True)) for t, v in (late or {}).get("nets", {}).items()
                                       if t not in sp["nets"]]
    for t, v in nets:
        fz = {m: f"{x}@{b}" for m, (x, b) in pc.frozen_paths(v).items()}
        for path, w in v["paths"].items():
            used = [m for m, pb in fz.items() if pb.startswith(path + "@")]
            a4 = w["a_us"].get(str(pc.VERDICT_B))
            bw = all(v.get("bitwise", {}).get(f"{path}@{B}", False) for B in w["a_us"])
            bad = [B for B, s in w["screened"].items() if not s]
            scr = "yes" if not bad else "no" if len(bad) == len(w["screened"]) else "not at " + ", ".join(bad)
            out.append(f"| {t}{' (late timing)' if v.get('late') else ''} | {path} | {w['prec']} "
                       f"| {r1(w['t1_us'])} | {r1(w['t1_us'] and w['t1_us'] / pc.R, 2)} | {r1(w['p999_us'])} | "
                       f"{r1(w['gpu_term_us'])} | {r1(w['a_us'].get('1024'), 2)} | {r1(a4, 2)} | "
                       f"{r1(a4 and a4 / pc.R, 3)} | {min(w['processes'].values())} | {scr} | {bw} | "
                       f"{', '.join(used) or '-'} | {'; '.join(pt.get(t, [])) or '-'} |")
    return "\n".join(out)


def pipe_table(rows, conf, chk):
    """The routed pipelines, the slowest of their processes (gate finding 4), with the per-round values and the paths'
    precision (finding 7), and the cost that the seeded reconstruction of Relay-BP's decisions would add (finding 8)."""
    build = {c["family"]: c.get("relay_build_ms", {}) for c in chk}
    out = ["Routed pipelines end to end, the slowest of their processes, and Cascade alone timed beside them in the "
           "same job (its single path; batched on its amortized path), which the verdicts compare them with. Relay-BP "
           "here is deployed: one decoder per process, kept across shots. The accuracy reads decide the same shots "
           "with a new decoder per routed shot seeded by its index (reproducible); building one takes the time in the "
           "last column per routed shot, which no timing includes; times the routed fraction it would add that much "
           "per incoming shot.", "",
           "| setting | config | p (%) | mode | processes | S path (precision) | B path (precision) | mean (us) | "
           "per round (us) | "
           "p99.9 (us) | per shot (us) | per round (us) | to B | to R | seeded build: ms per routed shot / us per shot "
           "|", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cf in conf:
        for (i, p, m), a in sorted(pipes(rows, cf["family"], cf["locked_sha256"]).items()):
            bm = build.get(cf["family"], {}).get(str(p))
            fr = a["frac_R"][1]
            seeded = f"{bm:.1f} / {1e3 * bm * fr:.1f}" if bm is not None and fr else "-"
            ps, pb = (x and f"{x} ({pc.precision(x.split('@')[0], k)})" for x, k in ((a["path_S"], pc.NETS[a["S"]][2]),
                                                                                     (a["path_B"], "cascade")))
            if m == "single":
                vals = f"{a['mean_us']:.1f} | {a['mean_us'] / pc.R:.2f} | {a['p999_us']:.1f} | - | -"
            else:
                vals = f"- | - | - | {a['per_shot_us']:.3f} | {a['per_shot_us'] / pc.R:.4f}"
            name = f"Cascade alone ({a['S']})" if i < 0 else f"{i} {a['type']}"
            out.append(f"| {cf['family']} | {name} | {100 * p:.2f} | {m} | {a['processes']} | {ps} | "
                       f"{pb or '-'} | {vals} | {a['frac_B'][1]:.4f} | {fr:.4f} | {seeded} |")
    return "\n".join(out)


def check_table(chk):
    out = ["Claim-point check (pc_speed check): the frozen timed paths against the evaluated computation on the same "
           "fresh shots; failures and routed shots, each as evaluated / timed (only timed / only evaluated); pass = "
           "pc_common.agrees, and for a ConvLines network (alone, or as S) its read bitwise its timed path; 'differ' "
           "counts the shots whose decision / confidence differs from the read", "",
           "| setting | p (%) | what | decoder | mode | paths | failures | to B | to R | differ | pass |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]

    def c(x):
        return "-" if x is None else f"{x['eval']} / {x['timed']} ({x['only_timed']} / {x['only_eval']})"
    for ck in chk:
        for r in ck["rows"]:
            dec = r.get("net") or f"{r['config']} {r['type']} ({r['decoder']})"
            paths = r.get("path") or ", ".join(f"{k} {v}" for k, v in r.get("paths", {}).items()) or "-"
            rt = r.get("routing", {})
            dif = (f"{r['differ']} / {r['conf_differ']}" if "conf_differ" in r else
                   f"S bitwise: {r['S_bitwise']}" if "S_bitwise" in r else "-")
            out.append(f"| {ck['family']} | {100 * r['p']:.2f} | {r['what']} | {dec} | {r['mode']} | {paths} | "
                       f"{c(r['fail'])} | {c(rt.get('to_B'))} | {c(rt.get('to_R'))} | {dif} | {r['pass']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", action="append", default=[], help="SETTING=DIR[,DIR2]")
    ap.add_argument("--confirm", action="append", default=[])
    ap.add_argument("--check", action="append", default=[])
    ap.add_argument("--locked", action="append", default=[], help="each setting's lock (confirmation, check)")
    ap.add_argument("--speed", default="")
    ap.add_argument("--speed-rows", default="")
    ap.add_argument("--speed-late", default="", help="a later timing of LATE networks (descriptive only)")
    ap.add_argument("--paper-timing", default="", help="the paper's digitised timings (external context)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = json.load(open(a.speed)) if a.speed else None
    late = json.load(open(a.speed_late)) if a.speed_late else None
    if late and sp and late["gpu"] != sp["gpu"]:
        raise SystemExit(f"--speed-late timed on {late['gpu']}, the networks on {sp['gpu']}")
    paper = json.load(open(a.paper_timing)) if a.paper_timing else None
    rows = [json.loads(x) for x in open(a.speed_rows) if x.strip()] if a.speed_rows else []
    conf = [json.load(open(f)) for f in a.confirm]
    chk = [json.load(open(f)) for f in a.check]
    sp_sha, locks = pc.sha256_file(a.speed) if a.speed else None, {}
    for f in a.locked:                                      # the speed file must be the one each lock froze
        lk = json.load(open(f))
        locks[lk["family"]] = pc.sha256_file(f)
        if sp_sha and lk["speed_sha256"] != sp_sha:
            raise SystemExit(f"{a.speed}: not the speed components that {f} froze ({lk['speed_sha256'][:16]})")
    for x in conf + chk:                                    # and every confirmation and check comes from that lock
        if x["locked_sha256"] != locks.get(x["family"]):
            raise SystemExit(f"{x['family']}: a confirmation or check made from another lock than --locked gives")
    cur = {}
    for spec in a.curves:
        fam, dirs = spec.split("=", 1)
        cur[fam] = curves(dirs.split(","), fam)
    m = headline([c for _, c in cur.values()], conf)
    md = [f"Headline family: {m} comparisons with Cascade (both settings), one Holm procedure at {pc.ALPHA}; planned "
          f"{pc.M_HEADLINE}.", ""]
    js = {"m_headline": m, "curves": {}, "verdicts": {}}
    for fam, (pts, comp) in cur.items():
        js["curves"][fam] = {"points": {str(p): v for p, v in pts.items()}, "vs_cascade": comp}
        md += [f"## Curves, {fam}", "", curve_table(pts, fam), "", comp_table(comp, fam), ""]
        js["verdicts"][fam] = verdicts(fam, comp, conf, sp, rows, next((c for c in chk if c["family"] == fam), None))
    for cf in conf:
        md += [f"## Routing confirmation, {cf['family']} (locked {cf['locked_sha256'][:12]})", "",
               "| config | p (%) | S | failures / shots | P_L | Cascade alone | to B | to R | kept: repairs / regressions vs B "
               "| vs Cascade [95 %] | McNemar p | planned power | Holm level | verdict | not worse by > margin "
               "(secondary) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in cf["rows"]:
            v = r["vs_B_95"]
            md.append(f"| {r['i']} {r['type']} | {100 * r['p']:.2f} | {r['S']} | {r['fail']:,} / {r['shots']:,} | "
                      f"{e(r['pl'])} | {r['B_alone']['fail']:,} | {r['to_B']:.4f} | {r['to_R']:.4f} | "
                      f"{r['kept_repairs_vs_B']} / {r['kept_regressions_vs_B']} | {e(v['diff_ler'])} "
                      f"[{e(v['diff_lo'], 1)}, {e(v['diff_hi'], 1)}] | {e(v['mcnemar_p'], 1)} | "
                      f"{r1(r['planned_power'], 3)} | {e(r['holm_alpha'], 1)} | {r['verdict']} | "
                      f"{r['noninferior']} (margin {e(r['ni_margin'])}) |")
        md.append("")
    if chk:
        md += ["## Claim-point check", "", check_table(chk), ""]
    if sp:
        md += ["## Speed", "", speed_tables(sp, late, paper), ""]
        if conf and rows:
            md += [pipe_table(rows, conf, chk), ""]
    md += ["## Verdicts (PHASE_C_PLAN.md section 6)", "", "```", json.dumps(js["verdicts"], indent=1), "```"]
    open(os.path.join(a.out, "phase_c_tables.md"), "w").write("\n".join(md) + "\n")
    json.dump(js | {"confirm": conf}, open(os.path.join(a.out, "phase_c_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({f: {"decoders_that_beat": [d["decoder"] for d in v.get("decoders", []) if d.get("beats")],
                          "cascade_not_eligible": v.get("cascade_not_eligible")} for f, v in js["verdicts"].items()}))


if __name__ == "__main__":
    main()
