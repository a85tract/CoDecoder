"""Stage 4 on gross (bb/STAGE4_GROSS_PLAN.md): cells G-4R and G-4F, the frozen network g2_D12k113c10 through the fusedt
engine (bb.fast) on one fresh split "test4" (15 chunks, 150,000 shots at p 0.004). The frozen parameters are in
data/bb/stage4_gross/cell.json, written by `freeze`. bb/stage3_gross.py and its records are unchanged.
  freeze    python -m bb.stage4_gross freeze --tag g2_D12k113c10_fusedt_s4: from fusedt's CAL read (tag T, made on the pod by
            `python -m bb.fast eq --kind fusedt --label s4`) the routing thresholds tau = the q-quantile of the 100,000 CAL
            margins (bb.route's rule; routed iff margin < tau): q 1 % (G-4R), 2 % (G-4F), 0.5 % (G-4's old q, reported);
            writes the packed CAL read (cal_fusedt.npz) and cell.json
  infer     python -m bb.stage4_gross infer --split test4 --chunks 0-14 --ckpt CK: the network through fusedt (checkpoint
            sha256 and engine signature equal to cell.json's); blind on test4
  beacon    python -m bb.stage4_gross beacon --commit SHA --after EPOCH: the first drand mainnet round at least 600 s after
            max(the freeze commit's committer time, EPOCH = when it was seen public on GitHub) -> data/bb/stage4_gross/beacon.json
  fullread  python -m bb.stage4_gross fullread --split test4 --chunks 0-9 --procs 128: full-DEM Relay-BP (bb.opp relay_full)
            one shot per task over all the chunks at once (no barrier per chunk), saved every ~60 s and resumed by a rerun;
            when every shot is decoded, bb.opp's per-chunk npz and a row; blind
  cputerm   python -m bb.stage4_gross cputerm --cell G-4R --split test4 --first 1000 --proc K --out R: the cell's fallback on
            its routed shots (margin < tau, split order, the first FIRST timed); CPU term = the routed fraction x the mean
            time per timed shot
  analyze   python -m bb.stage4_gross analyze --out data/bb/stage4_gross: the one analysis (plan section 5)"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import get_context

import numpy as np

from bb import opp, shots
from bb.stage3 import _count, _rows, _sha, _wilson

STAGE4G = os.path.join(shots.HERE, "data", "bb", "stage4_gross")
CELL = os.path.join(STAGE4G, "cell.json")
CALPACK = os.path.join(STAGE4G, "cal_fusedt.npz")
REPO = os.path.dirname(shots.HERE)
CODE, P, SPLIT, CHUNKS = "bb144", 0.004, "test4", "0-14"
NET_SHA = "eb380295a6eccfde022d8e61ad1c3f2b9154d54b3c04cc5eea84e3f91c224692"    # g2_D12k113c10/final_ema.pt
GPU = "H100"                              # the pod's card: an NVIDIA H100 (its name is recorded in the signature)
GATING = ("relay", "bposd", "gpu_bposd")
CELLS = {"G-4R": {"q": 0.01, "fallback": "relay", "vs": list(GATING), "bar": "gating", "first_routed": 1000},
         "G-4F": {"q": 0.02, "fallback": "relay_full", "vs": ["relay_full"] + list(GATING), "bar": "relay_full",
                  "first_routed": 100}}
BESIDE = {"G-4 (q 0.5 %)": {"q": 0.005, "fallback": "relay"}}    # G-4's old q on fusedt's margins: reported, not gated
OPP_FIRST, FULL_FIRST, CPU_K, GPU_K = 1000, 300, 3, 3
HANDOFF_MS = 1.1e-3                       # PLAN 2.7: the handoff allowance, 1.1 us
COMMAND = "python -m bb.stage4_gross analyze --out data/bb/stage4_gross"


def _cell():
    return json.load(open(CELL))


def _tau(margin, q):
    return float(np.quantile(margin, q))                  # bb.route.frontier's rule


def freeze(a):
    from bb import route
    ch = shots.parse_range("0-9")
    bits, margin, sig, meta = route._load_net(CODE, P, "cal", ch, a.tag)
    if sig.get("kind") != "fusedt" or sig.get("net_sha256") != NET_SHA or GPU not in sig.get("gpu", ""):
        raise SystemExit(f"not fusedt's read of the frozen network on an {GPU}: {sig}")
    obs = shots.load_many(CODE, P, "cal", ch)[1]
    nf = (bits != obs).any(1)
    rf = (opp.load_preds(CODE, P, "cal", ch, "relay")[0] != obs).any(1)
    full = np.load(os.path.join(shots.HERE, "data", "bb", "fullroute_gross", f"{CODE}_p{P}_cal_pick_relay_full.npz"))
    cal = {"shots": len(nf), "network_alone": int(nf.sum()), "sector_relay_alone": int(rf.sum()), "q": {}}
    for q in sorted({v["q"] for v in list(CELLS.values()) + list(BESIDE.values())}):
        tau = _tau(margin, q)
        r = margin < tau
        row = {"tau": tau, "routed": int(r.sum()), "hybrid_sector_relay": int(np.where(r, rf, nf).sum())}
        hit = np.isin(np.flatnonzero(r), full["idx"])
        if hit.all():                                     # every routed shot was decoded by the full-DEM CAL study's read
            ff = nf.copy()
            ff[full["idx"]] = np.where(r[full["idx"]], full["fail"], nf[full["idx"]])
            row["hybrid_full_dem"] = int(ff.sum())
        else:
            row["hybrid_full_dem"] = f"not computable: {int((~hit).sum())} routed shots outside the full-DEM CAL study's read"
        cal["q"][str(q)] = row
    ref = {}
    for tag in a.compare.split(",") if a.compare else []:
        try:
            b2, m2, s2, _ = route._load_net(CODE, P, "cal", ch, tag)
        except FileNotFoundError:
            continue
        ref[tag] = {"kind": s2.get("kind"), "decisions_differ": int((b2 != bits).any(1).sum()),
                    "margins_bitwise_equal": int((m2 == margin).sum()),
                    "margin_absdiff_max": float(np.abs(m2.astype(np.float64) - margin).max())}
    os.makedirs(STAGE4G, exist_ok=True)
    np.savez_compressed(CALPACK, margin=margin.astype(np.float32), pred=np.packbits(bits.astype(np.uint8), axis=1),
                        sig=json.dumps(sig, sort_keys=True), meta=json.dumps(meta), chunks="0-9", tag=a.tag)
    cells = {k: dict(v, tau=cal["q"][str(v["q"])]["tau"]) for k, v in CELLS.items()}
    beside = {k: dict(v, tau=cal["q"][str(v["q"])]["tau"]) for k, v in BESIDE.items()}
    c = {"plan": "ising_compare/bb/STAGE4_GROSS_PLAN.md", "code": CODE, "p": P, "split": SPLIT, "chunks": CHUNKS,
         "shots": 10_000 * len(shots.parse_range(CHUNKS)), "tag": a.tag, "net_sha256": NET_SHA, "engine": sig,
         "gpu_name": f"fusedt:{NET_SHA[:12]}", "cells": cells, "beside": beside, "alpha": 0.05 / 3,
         "cal": cal, "cal_read_vs": ref, "cal_pack": os.path.relpath(CALPACK, REPO),
         "cal_pack_sha256": _sha(open(CALPACK, "rb").read()), "opponents_first": OPP_FIRST,
         "full_dem_first": FULL_FIRST, "cpu_processes": CPU_K, "gpu_processes": GPU_K,
         "sessions": {"gpu": a.gpu_session, "cpu": a.cpu_session},
         "rows": {"gpu": "data/bb/stage4_gross/rows_gpu.csv", "cpu": "data/bb/stage4_gross/rows_cpu.csv"},
         "analysis_command": COMMAND}
    json.dump(c, open(CELL, "w"), indent=1)
    print(json.dumps({k: c[k] for k in ("cells", "beside", "cal", "cal_read_vs")}, indent=1), flush=True)


def infer(a):
    from bb import fast, route
    c = _cell()
    a.kind, a.cache = "fusedt", ""
    model, ck, sha, eng, _ = fast._engine(a)              # as bb.fast eq made the CAL read
    if sha != c["net_sha256"]:
        raise SystemExit(f"{a.ckpt} is not the frozen network")
    sig = eng.signature()
    if sig != c["engine"]:
        raise SystemExit(f"the engine signature differs from the frozen one: "
                         f"{ {k: (v, c['engine'].get(k)) for k, v in sig.items() if c['engine'].get(k) != v} }")
    meta = {"ckpt": os.path.abspath(a.ckpt), "run": ck["hp"]["run"], "run_id": ck["hp"]["run_id"],
            "samples": ck.get("samples"), "status": ck.get("status")}
    blind = a.split in ("test", "test4")                  # plan section 5: no test count before the analysis
    for j in shots.parse_range(a.chunks):
        det, obs = shots.load(c["code"], c["p"], a.split, j)
        bits, margin = eng.run(det)
        fail = (bits != obs).any(1)
        np.savez(route.net_path(c["code"], c["p"], a.split, j, c["tag"]), pred=bits, margin=margin, fail=fail,
                 sig=json.dumps(sig, sort_keys=True), meta=json.dumps(meta))
        print(f"net-{c['tag']} {a.split} chunk {j}: {'stored' if blind else f'{int(fail.sum())} / {len(fail)} failures'}",
              flush=True)


def beacon(a):
    sys.path.insert(0, shots.HERE)
    from icmp import t2
    tc = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", a.commit], cwd=REPO).strip())
    t = max(tc, int(a.after))
    b = t2.fetch_beacon(t2.beacon_round(t))
    t2.check_beacon(b, freeze_time=t)
    b.update(freeze_commit=subprocess.check_output(["git", "rev-parse", a.commit], cwd=REPO, text=True).strip(),
             freeze_commit_time=tc, public_seen_time=int(a.after), reference_time=t, delay_s=t2.DELAY, split=SPLIT,
             tags={"bb144_p0.004": shots.TEST4_TAG[(CODE, P)].decode()},
             seed_rule="2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62")
    os.makedirs(STAGE4G, exist_ok=True)
    json.dump(b, open(shots.BEACON_GROSS4, "w"), indent=1)
    print(json.dumps({k: b[k] for k in ("round", "round_time", "randomness", "freeze_commit", "reference_time")}),
          flush=True)


def _job(task):
    idx, det = task
    pred, secs, conv = opp._run(det)
    return idx, pred, secs, conv


def fullread(a):
    from bb.timing import _cpu
    chunks = shots.parse_range(a.chunks)
    parts = [shots.load(CODE, P, a.split, j) for j in chunks]           # sha256-checked
    det, obs = np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts])
    rec = os.path.join(opp.PREDS, f"{CODE}_p{P}_{a.split}_{a.dec}_part{a.chunks}.npz")
    keys = ("idx", "pred", "secs", "conv")
    got = [tuple(np.load(rec)[k] for k in keys)] if os.path.exists(rec) else []
    have = np.zeros(len(det), bool)
    if got:
        have[got[0][0]] = True
    todo = np.flatnonzero(~have)
    cpu, t0 = _cpu(), time.time()

    def save():
        idx, pred, secs, conv = (np.concatenate([g[i] for g in got]) for i in range(4))
        np.savez(rec[:-4] + ".tmp.npz", idx=idx.astype(np.int64), pred=pred.astype(np.uint8),
                 secs=secs.astype(np.float32), conv=conv.astype(bool), host=platform.node(), cpu=cpu, procs=a.procs)
        os.replace(rec[:-4] + ".tmp.npz", rec)
        print(f"{a.dec} {a.split} {a.chunks}: {len(idx)} / {len(det)} shots, mean {secs.mean():.2f} s, "
              f"{time.time() - t0:.0f} s", flush=True)               # blind: no failure count
        return idx, pred, secs, conv

    opp._init(a.dec, CODE, P)                             # fail fast, as bb.opp
    if len(todo):
        last = time.time()
        tasks = ((todo[s:s + a.slice], det[todo[s:s + a.slice]]) for s in range(0, len(todo), a.slice))
        with get_context("fork").Pool(a.procs, initializer=opp._init, initargs=(a.dec, CODE, P)) as pool:
            for r in pool.imap_unordered(_job, tasks):
                got.append(r)
                if time.time() - last > 60:
                    save()
                    last = time.time()
    idx, pred, secs, conv = save()
    o = np.argsort(idx, kind="stable")
    if not np.array_equal(idx[o], np.arange(len(det))):
        raise SystemExit(f"incomplete or duplicated: {len(idx)} records for {len(det)} shots")
    pred, secs, conv = pred[o], secs[o], conv[o]
    s = 0
    for j, (d, _) in zip(chunks, parts):
        e = s + len(d)
        np.savez(opp.pred_path(CODE, P, a.split, j, a.dec), pred=pred[s:e], fail=(pred[s:e] != obs[s:e]).any(1),
                 secs=secs[s:e], conv=conv[s:e])
        s = e
    row = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(), "cpu": cpu,
           "procs": a.procs, "code": CODE, "p": P, "split": a.split, "chunks": a.chunks, "first": 0, "decoder": a.dec,
           "shots": len(det), "failures": "", "ler": "", "mean_ms": 1e3 * float(secs.mean()),
           "p50_ms": 1e3 * float(np.median(secs)), "p99_ms": 1e3 * float(np.quantile(secs, 0.99)),
           "max_ms": 1e3 * float(secs.max()), "wall_s": time.time() - t0}
    import csv
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("row", row, flush=True)


def _pol(c, cell):
    """The policy record bb.timing.population reads: the network read's tag, its engine signature and the cell's tau."""
    return {"code": c["code"], "p": c["p"], "tag": c["tag"], "engine": c["engine"], "policy": {"tau": cell["tau"]}}


def cputerm(a):
    from bb import timing
    c = _cell()
    cell = c["cells"][a.cell]
    chunks = shots.parse_range(c["chunks"])
    cc = timing.corecheck()
    idx, pop = timing.population(_pol(c, cell), a.split, chunks, a.first)   # checks the read's signature
    frac = pop["routed"] / pop["shots"]
    det = shots.load_many(c["code"], c["p"], a.split, chunks)[0][idx]
    secs = timing._secs(cell["fallback"], c["code"], c["p"], det) if len(det) else np.zeros(1)
    timing.append_row(a.out, {"device": timing._cpu(), "role": "cpu_term", "code": c["code"], "p": c["p"],
                              "name": a.cell, "proc": a.proc, "n": len(det), "value_ms": frac * 1e3 * float(secs.mean()),
                              "binding": _sha(open(CELL, "rb").read())[:16],
                              "note": json.dumps({"split": a.split, "routed_fraction": frac, "fallback": cell["fallback"],
                                                  "mean_ms_routed": 1e3 * float(secs.mean()), "population": pop,
                                                  "corecheck_s": cc})})


def _speed(G, C, c, name, cell):
    s, sp = c["sessions"], c["split"]
    g = _count(G, c["gpu_processes"], "max", role="gpu_term", name=c["gpu_name"], session=s["gpu"])
    h = _count(G, c["gpu_processes"], "max", role="handoff", name=c["gpu_name"], session=s["gpu"])
    h = dict(h, value_ms=None if h["value_ms"] is None else max(HANDOFF_MS, h["value_ms"]))
    k = c["cpu_processes"]
    cpu_t = _count(C, k, "max", role="cpu_term", name=name, session=s["cpu"], note_split=sp)
    gating = {d: _count(C, k, "min", role="opponent", name=d, p=c["p"], session=s["cpu"], note_split=sp)
              for d in ("relay", "bposd")}
    gating["gpu_bposd"] = _count(G, k, "min", role="opponent", name="gpu_bposd", p=c["p"], session=s["gpu"],
                                 note_split=sp)
    out = {"split": sp, "gpu_term": g, "handoff": h, "cpu_term": cpu_t, "gating": gating}
    if cell["bar"] == "relay_full":
        out["relay_full"] = _count(C, k, "min", role="opponent", name="relay_full", p=c["p"], session=s["cpu"],
                                   note_split=sp)
        bars = {"relay_full": out["relay_full"]}
    else:
        bars = gating
    need = [g, h, cpu_t] + list(bars.values())
    if any(x["value_ms"] is None for x in need):
        return dict(out, error="missing timing rows", ok=False)
    out["composite_ms"] = g["value_ms"] + h["value_ms"] + cpu_t["value_ms"]
    out["bar_from"] = min(bars, key=lambda d: bars[d]["value_ms"])
    out["bar_ms"] = bars[out["bar_from"]]["value_ms"]
    out["ok"] = out["composite_ms"] <= out["bar_ms"]
    if all(x["value_ms"] is not None for x in gating.values()):         # G-4F: the fastest gating opponent beside
        out["fastest_gating_ms"] = min(x["value_ms"] for x in gating.values())
    return out


def analyze(a):
    from bb import route
    c = _cell()
    code, p, sp, chunks = c["code"], c["p"], c["split"], shots.parse_range(c["chunks"])
    obs = shots.load_many(code, p, sp, chunks)[1]
    n = len(obs)
    fails, dec = {}, {}
    for d in ("relay_full",) + GATING:
        fails[d] = (opp.load_preds(code, p, sp, chunks, d)[0] != obs).any(1)
        k = int(fails[d].sum())
        dec[d] = {"shots": n, "failures": k, "rate": k / n, "wilson95": _wilson(k, n)}
    bits, margin, sig, meta = route._load_net(code, p, sp, chunks, c["tag"])
    preds = {d: opp.load_preds(code, p, sp, chunks, d)[0] for d in ("relay", "relay_full")}
    alone = (bits != obs).any(1)
    err = None
    if sig != c["engine"] or sig["net_sha256"] != c["net_sha256"]:
        err = "the test read is not the frozen engine"

    def hybrid(cell):
        routed = margin < cell["tau"]
        f = (np.where(routed[:, None], preds[cell["fallback"]], bits) != obs).any(1)
        k = int(f.sum())
        return f, {"tau": cell["tau"], "q": cell["q"], "fallback": cell["fallback"], "routed": int(routed.sum()),
                   "routed_fraction": float(routed.mean()), "failures": k, "rate": k / n, "wilson95": _wilson(k, n)}

    G, C = (_rows(os.path.join(shots.HERE, c["rows"][x])) for x in ("gpu", "cpu"))
    cells = {}
    for name, cell in c["cells"].items():
        f, r = hybrid(cell)
        r["vs"] = {d: route.paired(f, fails[d], c["alpha"]) for d in cell["vs"]}
        r["accuracy"] = all(v["verdict"] == "better" for v in r["vs"].values())
        r["timing"] = _speed(G, C, c, name, cell)
        r["speed"] = r["timing"]["ok"]
        r["claim"] = bool(r["accuracy"] and r["speed"] and err is None)
        cells[name] = r
    beside = {"network alone": {"failures": int(alone.sum()), "rate": int(alone.sum()) / n,
                                "wilson95": _wilson(int(alone.sum()), n)}}
    for name, cell in c["beside"].items():
        beside[name] = hybrid(cell)[1]
    res = {"plan": c["plan"], "alpha": c["alpha"], "shots": n, "engine_equal": err is None, "net_sha256": sig["net_sha256"],
           "cells": cells, "beside": beside, "decoders": dec,
           "idle_checks": {"rows": sum(x["role"] == "idle" for x in G),
                           "not_ok": sum(x["role"] == "idle" and not x["note"].get("ok") for x in G)},
           "command": c["analysis_command"]}
    if err:
        res["error"] = err
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    open(os.path.join(a.out, "RESULTS_stage4_gross.md"), "w").write(_report(res))
    _figure(res, os.path.join(a.out, "gross_stage4.png"))
    print(_report(res), flush=True)


def _report(res):
    L = ["# Gross (BB144) stage 4, cells G-4R and G-4F: results of the one analysis (generated by "
         "`python -m bb.stage4_gross analyze`)", "",
         f"{res['shots']:,} test4 shots. Each comparison: exact two-sided McNemar p < {res['alpha']:.6f} and the paired "
         f"Newcombe (1 - alpha) interval of LER(hybrid) - LER(opponent) entirely below 0; a cell's claim needs all its "
         f"comparisons and its speed condition.", ""]
    for name, r in res["cells"].items():
        t = r["timing"]
        L += [f"## {name}", "",
              f"tau = {r['tau']} (q = {r['q']} on CAL), {r['routed']:,} test shots routed to {r['fallback']} "
              f"({100 * r['routed_fraction']:.3f} %); failures {r['failures']:,} (rate {r['rate']:.3e}, Wilson 95 % "
              f"[{r['wilson95'][0]:.3e}, {r['wilson95'][1]:.3e}]).", "",
              "| opponent | its failures | only hybrid / only it | McNemar p | Newcombe interval | verdict |",
              "|---|---|---|---|---|---|"]
        for d, v in r["vs"].items():
            L.append(f"| {d} | {v['fail_b']:,} | {v['only_a']:,} / {v['only_b']:,} | {v['mcnemar_p']:.2e} | "
                     f"[{v['diff_lo']:.2e}, {v['diff_hi']:.2e}] | {v['verdict']} |")
        L.append("")
        if "composite_ms" in t:
            L += ["| composite (ms) | GPU term | handoff | CPU term | bar (ms) | bar from | speed | accuracy | claim |",
                  "|---|---|---|---|---|---|---|---|---|",
                  f"| {t['composite_ms']:.4f} | {t['gpu_term']['value_ms']:.4f} | {t['handoff']['value_ms']:.4f} | "
                  f"{t['cpu_term']['value_ms']:.4f} | {t['bar_ms']:.4f} | {t['bar_from']} | {r['speed']} | "
                  f"{r['accuracy']} | {r['claim']} |", "",
                  "Gating opponents' times (ms per shot; the fastest of 3 processes): " +
                  ", ".join(f"{d} {b['value_ms']:.3f}" if b["value_ms"] is not None else f"{d} missing"
                            for d, b in t["gating"].items()), ""]
        else:
            L += [f"Speed: {t.get('error')}; accuracy {r['accuracy']}; claim {r['claim']}", ""]
    L += ["## Reported beside (not gated)", "", "| decoder | failures | rate | Wilson 95 % |", "|---|---|---|---|"]
    for name, v in list(res["beside"].items()) + list(res["decoders"].items()):
        extra = f" ({v['routed']:,} routed, tau {v['tau']:.6f})" if "routed" in v else ""
        L.append(f"| {name}{extra} | {v['failures']:,} | {v['rate']:.3e} | [{v['wilson95'][0]:.3e}, "
                 f"{v['wilson95'][1]:.3e}] |")
    if "error" in res:
        L += ["", f"ERROR: {res['error']}"]
    return "\n".join(L) + "\n"


def _figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, ours_c, surf = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"     # as bb.stage3's figure
    plt.rcParams.update({"text.color": ink, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": "#c3c2b7"})
    c, b, d = res["cells"], res["beside"], res["decoders"]
    items = [("G-4R", c["G-4R"], ours_c), ("G-4F", c["G-4F"], ours_c), ("network alone", b["network alone"], "#8fb8e8"),
             ("G-4 (q 0.5 %)", b["G-4 (q 0.5 %)"], "#8fb8e8"), ("sector Relay-BP", d["relay"], ink2),
             ("BP-OSD (CPU)", d["bposd"], ink2), ("cudaq-qec BP-OSD", d["gpu_bposd"], ink2),
             ("full-DEM Relay-BP", d["relay_full"], ink2)]
    fig, x = plt.subplots(figsize=(5.6, 3.6), facecolor=surf)
    for i, (lb, v, col) in enumerate(items):
        r, ci = v["rate"], v["wilson95"]
        x.errorbar(i, 100 * r, yerr=[[100 * (r - ci[0])], [100 * (ci[1] - r)]], fmt="o", color=col, capsize=3,
                   markersize=6, elinewidth=1)
    x.set_facecolor(surf)
    x.set_ylim(bottom=0)
    x.set_xticks(range(len(items)), [i[0] for i in items], rotation=45, ha="right", fontsize=8)
    x.set_title(f"G-4R, G-4F: [[144,12,12]], p = 0.004, {res['shots']:,} test4 shots", fontsize=9)
    x.set_ylabel("block failure rate (%)", fontsize=8)
    x.grid(axis="y", alpha=0.3)
    x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "infer", "beacon", "fullread", "cputerm", "analyze"])
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="g2_D12k113c10_fusedt_s4")
    ap.add_argument("--compare", default="g2_D12k113c10_fusedt_h100,g2_D12k113c10",
                    help="freeze: CAL reads compared with fusedt's new one when present (the fast-engine study's fusedt, graph32)")
    ap.add_argument("--gpu-session", default="h100-s4")
    ap.add_argument("--cpu-session", default="M4 session s4")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--after", type=int, default=0)
    ap.add_argument("--dec", default="relay_full")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--slice", type=int, default=1, help="fullread: shots per task")
    ap.add_argument("--cell", default="G-4R", choices=list(CELLS))
    ap.add_argument("--first", type=int, default=1000)
    ap.add_argument("--proc", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
