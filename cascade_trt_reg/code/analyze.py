"""CASCADE-TRT analysis (exploratory): the outputs of run_steps.sh -> summary.json and tables.md.

    python analyze.py OUT_DIR PUB_DIR

Statistic (the task's): Cascade alone = the fastest of its 3 processes, a pair = the slowest of its 3; the slowest
Cascade is shown beside it. Cascade alone comes from step 3 (trt_alone.py: host round trip mean at batch 1, host-to-host
amortized time per shot at the batch); a pair's speed-up = Cascade alone (same engine) / the pair. The registered H100
values are read from speed_rt/cascade/results/table.json (card H100): the comparison rows (slowest Cascade, the
registered statistic) and the sensitivity rows (fastest Cascade)."""
import collections
import json
import os
import sys

R = {"bb": 6, "d13": 13}
SERIES = {"bb": "seq", "d13": "seq"}   # step 3 as reported: one Cascade graph at a time (alone_seq.jsonl). Records only:
# v1 = reps 1-3 (both graphs in one process; d = 13 TensorRT plans of FP32/TF32), v2 = d = 13 reps 11-13 (BF16 plans)
BATCH = {"bb": 4096, "d13": 16}


def rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{"):
            out.append(json.loads(ln))
    return out


def alone(O):
    """{(net, engine): {single: [host_rt mean per rep], single_p999, gpu_us, batched: [h2h per shot], dev, clocks}}."""
    A = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows(os.path.join(O, "alone.jsonl")) + rows(os.path.join(O, "alone_seq.jsonl")):
        a = A[(r["net"], r["engine"], "seq" if r.get("seq") else ("v2" if r["rep"] > 10 else "v1"))]
        a["single"].append(r["host_rt_b1"]["mean"])
        a["single_p999"].append(r["host_rt_b1"]["p999"])
        a["gpu_us"].append(sum(r["gpu_us_b1"]) / len(r["gpu_us_b1"]))
        a["batched"].append(r["amortized"]["h2h"])
        a["dev"].append(r["amortized"]["dev"])
        a["clock_min"].append(min(x for x in (r["clock_b1"].get("clock_min_ratio"), r["clock_bB"].get("clock_min_ratio"))
                                  if x is not None) if any(r[c].get("clock_min_ratio") for c in ("clock_b1", "clock_bB"))
                              else None)
        a["accepted"].append(bool(r["clock_b1"].get("accepted")) and bool(r["clock_bB"].get("accepted")))
        a["cuda_graph"].append(r["trt"]["b1"]["cuda_graph"] if r["trt"].get("b1") else None)
    return A


def ext(v):
    return {"fastest": min(v), "slowest": max(v), "n": len(v), "all": v} if v else None


def pair_rows(O):
    """{(harness, config, p, mode, engine): [value per process]} with value = latency mean (single) or per shot
    (batched), plus frac_B and p999."""
    P = collections.defaultdict(list)
    for h in ("d1", "d13", "pc"):
        for r in rows(os.path.join(O, f"pairs_{h}.jsonl")) + rows(os.path.join(O, f"alone_pipe_{h}.jsonl")):
            if "ctrt" not in r or r.get("role") not in (None, "pipeline") or r.get("metric") in ("error",):
                continue
            mode = r.get("mode")
            if mode == "single" and isinstance(r.get("latency_us"), dict):
                v = r["latency_us"]["mean"]
                p999 = r["latency_us"].get("p999")
            elif mode == "batched" and r.get("per_shot_us") is not None:
                v, p999 = r["per_shot_us"], None
            else:
                continue
            P[(h, str(r.get("config")), float(r["p"]), mode, r["ctrt"]["engine"])].append(
                {"v": v, "p999": p999, "frac_B": r.get("frac_B"), "clock": (r.get("clock") or {}).get("clock_min_ratio"),
                 "accepted": (r.get("clock") or {}).get("accepted"), "trt": r["ctrt"].get("trt"),
                 "rep": r.get("rep"), "shots": r.get("shots")})
    return P


def registered(PUB):
    t = json.load(open(os.path.join(PUB, "speed_rt", "cascade", "results", "table.json")))
    reg = {}
    for r in t:
        if r.get("card") != "H100":
            continue
        reg[(r["row"], r["item"])] = r
    return reg


def reg_cmp(reg, row, item):
    r = reg.get((row, item))
    return None if r is None else {"ratio": r["ratio"], "pair": r["ours"], "alone_slowest": r["theirs"]}


def reg_num(reg, row, item):
    r = reg.get((row, item))
    return None if r is None else r["new"]


SETTINGS = [  # (label, harness, config, p, net, registered row, registered item stem, sensitivity stem)
    ("BB72 D1 pair S55p_deep -> C14p (multi-shot engine), p = 0.2 %", "d1", "0", 0.002, "bb", "C2",
     "BB72 p=0.2 %: pair on the multi-shot engine vs Cascade alone, {m}",
     ("C1", "sensitivity, decides nothing: fastest accepted Cascade-alone process; BB72 p=0.2 %: Cascade alone, {m}")),
    ("BB72 D1 pair S55p_deep -> C14p (multi-shot engine), p = 0.3 %", "d1", "1", 0.003, "bb", "C2",
     "BB72 p=0.3 %: pair on the multi-shot engine vs Cascade alone, {m}",
     ("C1", "sensitivity, decides nothing: fastest accepted Cascade-alone process; BB72 p=0.3 %: Cascade alone, {m}")),
    ("d = 13 pair (C5), p = 0.6 %", "d13", "C5", 0.006, "d13", "C8", "d=13 p=0.6 %: pair vs Cascade alone, {m}",
     ("C8", "sensitivity, decides nothing: fastest accepted Cascade-alone process; d=13 p=0.6 %: Cascade alone, {m}")),
    ("d = 13 pair (C5), p = 0.35 %", "d13", "C5", 0.0035, "d13", "C8", "d=13 p=0.35 %: pair vs Cascade alone, {m}",
     ("C8", "sensitivity, decides nothing: fastest accepted Cascade-alone process; d=13 p=0.35 %: Cascade alone, {m}")),
    ("BB72 phase C pair S55p -> C14p, p = 0.2 %", "pc", "2", 0.002, "bb", "C6", "BB72 p=0.2 %: pair vs Cascade alone, {m}",
     ("C6", "sensitivity, decides nothing: fastest accepted Cascade-alone process; BB72 p=0.2 % (phase C): Cascade alone, {m}")),
    ("BB72 phase C pair S55p -> C14p, p = 0.3 %", "pc", "7", 0.003, "bb", "C6", "BB72 p=0.3 %: pair vs Cascade alone, {m}",
     ("C6", "sensitivity, decides nothing: fastest accepted Cascade-alone process; BB72 p=0.3 % (phase C): Cascade alone, {m}")),
]


def main(O, PUB):
    A = alone(O)
    P = pair_rows(O)
    reg = registered(PUB)
    S = {"alone": {}, "check": {}, "pairs": [], "alone_pipe": {}}
    for (net, eng, ser), a in sorted(A.items()):
        S["alone"][f"{net}/{eng}/{ser}"] = {k: ext(v) if k in ("single", "single_p999", "gpu_us", "batched", "dev") else v
                                     for k, v in a.items()}
    for net in ("bb", "d13"):
        f = os.path.join(O, f"check_{net}_split.json")
        if os.path.exists(f):
            c = json.load(open(f))
            S["check"][net] = {k: c.get(k) for k in ("total_bB", "total_b1", "p", "trt", "kind", "chunks", "batch")}
    for (h, cfg, p, mode, eng), v in sorted(P.items()):
        if cfg in ("-1", "C2"):
            S["alone_pipe"][f"{h}/{p}/{mode}/{eng}"] = ext([x["v"] for x in v])
    for label, h, cfg, p, net, row, item, (srow, sitem) in SETTINGS:
        for mode in ("single", "batched"):
            e = {"setting": label, "mode": mode, "unit": "us" if mode == "single" else "us per round"}
            div = 1 if mode == "single" else R[net]
            for eng in ("compiled", "trt"):
                pv = P.get((h, cfg, p, mode, eng), [])
                al = A.get((net, eng, SERIES[net]), {}).get(mode)
                ap = P.get(("d1", "-1", 0.002, mode, eng) if net == "bb" else ("d13", "C2", 0.006, mode, eng), [])
                pair_slow = max(x["v"] for x in pv) / div if pv else None
                al_fast = min(al) / div if al else None
                al_slow = max(al) / div if al else None
                ap_fast = min(x["v"] for x in ap) / div if ap else None
                e[eng] = {"pair_slowest": pair_slow, "pair_all": [x["v"] / div for x in pv],
                          "pair_p999_slowest": max((x["p999"] for x in pv if x["p999"]), default=None),
                          "frac_B": [x["frac_B"] for x in pv], "clock_min": [x["clock"] for x in pv],
                          "alone_fastest": al_fast, "alone_slowest": al_slow,
                          "speedup_step3": al_fast / pair_slow if al_fast and pair_slow else None,
                          "speedup_step3_slowest_alone": al_slow / pair_slow if al_slow and pair_slow else None,
                          "alone_pipe_fastest": ap_fast, "alone_pipe_slowest": max(x["v"] for x in ap) / div if ap else None,
                          "alone_pipe_n": len(ap),
                          "speedup": ap_fast / pair_slow if ap_fast and pair_slow else None}
            rc = reg_cmp(reg, row, item.format(m=mode))
            rs = reg_num(reg, srow, sitem.format(m=mode))
            e["registered"] = rc and {**rc, "alone_fastest": rs,
                                      "ratio_fastest_alone": rs / rc["pair"] if rs and rc["pair"] else None}
            S["pairs"].append(e)
    json.dump(S, open(os.path.join(O, "summary.json"), "w"), indent=1)
    with open(os.path.join(O, "tables.md"), "w") as fh:
        fh.write(tables(S))
    print(json.dumps(S["alone"], default=str)[:3000])
    for e in S["pairs"]:
        c, t, g = e["compiled"], e["trt"], e["registered"] or {}
        g4 = lambda x: "-" if x is None else f"{x:.4g}"                     # noqa: E731
        print(f"{e['setting']} {e['mode']}: pair compiled {g4(c['pair_slowest'])} trt {g4(t['pair_slowest'])} | alone "
              f"compiled {g4(c['alone_fastest'])} trt {g4(t['alone_fastest'])} | speed-up compiled {g4(c['speedup'])} "
              f"trt {g4(t['speedup'])} | registered {g4(g.get('ratio'))} (fastest alone {g4(g.get('ratio_fastest_alone'))})")


def f(x, d=1):
    return "-" if x is None else f"{x:,.{d}f}"


def tables(S):
    out = ["## Cascade alone (us; fastest / slowest of 3 processes)", "",
           "| Cascade | measure | compiled fastest | compiled slowest | TensorRT fastest | TensorRT slowest | "
           "compiled / TensorRT (fastest) |", "|---|---|---|---|---|---|---|"]
    names = {"bb": "BB72 C14p", "d13": "d = 13 CS13"}
    for net in ("bb", "d13"):
        c, t = S["alone"].get(f"{net}/compiled/{SERIES[net]}"), S["alone"].get(f"{net}/trt/{SERIES[net]}")
        if not c or not t:
            continue
        for k, lab, div in (("single", "batch 1: host round trip, mean", 1), ("single_p999", "batch 1: round trip p99.9", 1),
                            ("gpu_us", "batch 1: GPU time per call", 1),
                            ("batched", f"batch {BATCH[net]:,}: host to host, per shot", 1),
                            ("batched", f"batch {BATCH[net]:,}: host to host, per round", R[net])):
            cc, tt = c[k], t[k]
            out.append(f"| {names[net]} | {lab} | {f(cc['fastest'] / div, 2)} | {f(cc['slowest'] / div, 2)} | "
                       f"{f(tt['fastest'] / div, 2)} | {f(tt['slowest'] / div, 2)} | "
                       f"{cc['fastest'] / tt['fastest']:.2f}x |")
    out += ["", "## Decision check (fresh shots, seed kind 47)", "",
            "| Cascade | engines at batch | shots | decisions differ | failures compiled | failures TensorRT | "
            "only compiled fails | only TensorRT fails | exact McNemar p |", "|---|---|---|---|---|---|---|---|---|"]
    for net in ("bb", "d13"):
        c = S["check"].get(net)
        if not c:
            continue
        for k, lab in (("total_bB", f"{BATCH[net]:,}"), ("total_b1", "1")):
            x = c[k]
            out.append(f"| {names[net]} | {lab} | {x['shots']:,} | {x['decisions_differ']:,} | {x['fail_compiled']:,} | "
                       f"{x['fail_trt']:,} | {x['only_compiled_fails']} | {x['only_trt_fails']} | "
                       f"{x['mcnemar_exact_p']:.3g} |")
    out += ["", "## Pairs: speed-up over Cascade alone (Cascade alone fastest of 3, pair slowest of 3)", "",
            "| setting | mode | unit | pair, compiled B | pair, TensorRT B | Cascade alone (pipeline), compiled | "
            "Cascade alone (pipeline), TensorRT | speed-up, compiled | speed-up, TensorRT | registered H100 speed-up "
            "(slowest Cascade) | registered, fastest Cascade | speed-up from step 3, compiled | "
            "speed-up from step 3, TensorRT |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in S["pairs"]:
        c, t, g = e["compiled"], e["trt"], e["registered"] or {}
        if c["pair_slowest"] is None and t["pair_slowest"] is None:
            continue
        d = 1 if e["mode"] == "single" else 3
        out.append(f"| {e['setting']} | {e['mode']} | {e['unit']} | {f(c['pair_slowest'], d)} | {f(t['pair_slowest'], d)} | "
                   f"{f(c['alone_pipe_fastest'], d)} | {f(t['alone_pipe_fastest'], d)} | {f(c['speedup'], 2)} | "
                   f"{f(t['speedup'], 2)} | {f(g.get('ratio'), 2)} | {f(g.get('ratio_fastest_alone'), 2)} | "
                   f"{f(c['speedup_step3'], 2)} | {f(t['speedup_step3'], 2)} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
