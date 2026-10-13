"""Stage 3 on BB72 (bb/STAGE3_PLAN.md): host check, beacon, H100-vs-RTX agreement, the F cells' CPU terms, the
pre-freeze numbers and the one analysis.
  hostcheck  python -m bb.stage3 hostcheck --out H.json: regenerates cal chunks 0 and 19 at both p on this host and
             compares them with the stored chunks (sha256 = the manifest; np.savez stamps the write time into the
             file, so the check compares the arrays); exit 1 on any difference (then no test sampling here)
  beacon     python -m bb.stage3 beacon --commit SHA: the first drand mainnet round at least 600 s after the freeze
             commit's committer time (icmp.t2's chain, endpoints and checks), written to data/bb/stage3/beacon.json
  agree      python -m bb.stage3 agree --out A.json: r11 / r12 on the cal splits, H100 reads (tags r11h, r12h)
             against the RTX reads (r11, r12): decisions, margins, the F cells' routed sets
  cputerm    python -m bb.stage3 cputerm --cell BB72-2F --split test --tag r12 --first 5000 --proc K --out T.csv:
             the cell's fallback (full-DEM Relay-BP; sector Relay-BP in Amendment S) on its routed shots (margin <
             tau, split order, the first FIRST timed); CPU term = the split's routed fraction x the mean time per
             timed shot (0 if nothing is routed)
  prefreeze  python -m bb.stage3 prefreeze --gpu-rows G.csv --cpu-rows C.csv: Q8's stage-2 verdicts and BB72-2F's
             screen (STAGE3_PLAN.md 4.2 session A), written into data/bb/stage3/cells.json
  analyze    python -m bb.stage3 analyze --out data/bb/stage3: the one analysis (STAGE3_PLAN.md 5)
With --amend data/bb/stage3/amendment_S.json (AMENDMENT_S.md), cputerm and analyze take the amendment's cells, and
analyze its CPU-term rows and session (the bars stay session B's) and writes results_amendment_S.json, its report, its
figure and bb72_stage3_all.png (every cell of the study); the registered cells and outputs are left as they are."""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys

import numpy as np

from bb import shots

STAGE3 = os.path.join(shots.HERE, "data", "bb", "stage3")
CELLS = os.path.join(STAGE3, "cells.json")
REPO = os.path.dirname(shots.HERE)
GATING = ("relay", "bposd", "gpu_bposd")
HANDOFF_MS = 1.1e-3                       # PLAN 2.7: the handoff allowance, 1.1 us
RTX = {"gpu_term_ms": 0.0733, "handoff_ms": 0.0125, "gpu_bposd_ms": {"0.003": 2.362, "0.002": 1.075},
       "session": "Alpine (CU Boulder Research Computing)"}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _cells(amend=""):
    cells = json.load(open(CELLS))
    if amend:                                               # Amendment S: its cells replace the registered ones
        cells["amendment"] = json.load(open(amend))
        cells["cells"] = cells["amendment"]["cells"]
    return cells


def _split(cells, name):
    s = cells["splits"][name]
    return s["code"], s["p"], s["chunks"]


# ---- host check and beacon --------------------------------------------------------------------------------------------
def hostcheck(a):
    import stim
    from bb.timing import _cpu
    flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
    out = {"host": platform.node(), "cpu": _cpu(), "avx2": " avx2" in flags, "machine": platform.machine(),
           "python": sys.version.split()[0], "numpy": np.__version__, "stim": stim.__version__, "chunks": []}
    ok = True
    for p in (0.003, 0.002):
        for j in (0, 19):
            det0, obs0 = shots.load("bb72", p, "cal", j)
            det, obs = shots.sample("bb72", p, "cal", j)
            same = bool(np.array_equal(det, det0) and np.array_equal(obs, obs0))
            out["chunks"].append({"file": shots.name("bb72", p, "cal", j), "seed": shots.seed_of("cal", "bb72", p, j),
                                  "equal": same, "det_sha256": _sha(np.packbits(det, axis=1).tobytes())})
            ok = ok and same
    out["stim_modules"] = sorted(m for m in sys.modules if m.startswith("stim._"))
    out["ok"] = ok
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    raise SystemExit(0 if ok else 1)


def beacon(a):
    sys.path.insert(0, shots.HERE)
    from icmp import t2
    t = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", a.commit], cwd=REPO).strip())
    b = t2.fetch_beacon(t2.beacon_round(t))
    t2.check_beacon(b, freeze_time=t)
    b.update(freeze_commit=subprocess.check_output(["git", "rev-parse", a.commit], cwd=REPO, text=True).strip(),
             freeze_commit_time=t, delay_s=t2.DELAY,
             tags={f"{c}_p{p}": shots.TEST_TAG[(c, p)].decode() for c, p in shots.TEST_TAG},
             seed_rule="2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62")
    json.dump(b, open(shots.BEACON, "w"), indent=1)
    print(json.dumps({k: b[k] for k in ("round", "round_time", "randomness", "freeze_commit", "freeze_commit_time")}),
          flush=True)


# ---- the platform change: H100 against RTX on cal ---------------------------------------------------------------------
def agree(a):
    from bb import route
    cells = _cells()
    taus = {(c["p"], c["tag"]): (n, c["tau"]) for n, c in cells["cells"].items() if c["tau"] is not None}
    out = {}
    for p in (0.003, 0.002):
        ch = shots.parse_range("0-19")
        for t in ("r11", "r12"):
            ph, mh, sh, _ = route._load_net("bb72", p, "cal", ch, t + "h")
            pr, mr, sr, _ = route._load_net("bb72", p, "cal", ch, t)
            eq = (ph == pr).all(1)
            d = np.abs(mh.astype(np.float64) - mr)
            r = {"shots": len(eq), "decisions_differ": int((~eq).sum()), "margin_absdiff_max": float(d.max()),
                 "margin_absdiff_p99": float(np.quantile(d, 0.99)), "margins_bitwise_equal": int((d == 0).sum()),
                 "gpu_h100": sh["gpu"], "gpu_rtx": sr["gpu"],
                 "signature_differs_in": sorted(k for k in set(sh) | set(sr) if sh.get(k) != sr.get(k))}
            if (p, t) in taus:
                n, tau = taus[(p, t)]
                rh, rr = mh < tau, mr < tau
                r["routed"] = {"cell": n, "tau": tau, "h100": int(rh.sum()), "rtx": int(rr.sum()),
                               "only_h100": int((rh & ~rr).sum()), "only_rtx": int((rr & ~rh).sum())}
            out[f"{t}_p{p}"] = r
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


# ---- the F cells' CPU terms -------------------------------------------------------------------------------------------
def cputerm(a):
    from bb import route, timing
    cells = _cells(a.amend)
    c = cells["cells"][a.cell]
    code, p, _ = _split(cells, c["split"])
    chunks = shots.parse_range(a.chunks)
    cc = timing.corecheck()
    _, margin, sig, _ = route._load_net(code, p, a.split, chunks, a.tag)
    if sig["net_sha256"] != c["net_sha256"]:
        raise SystemExit("this read is not the cell's network")
    routed = np.flatnonzero(margin < c["tau"])
    idx = routed[:a.first]
    det = shots.load_many(code, p, a.split, chunks)[0][idx]
    secs = timing._secs(c["fallback"], code, p, det) if len(idx) else np.zeros(0)
    frac = len(routed) / len(margin)
    mean_ms = 1e3 * float(secs.mean()) if len(idx) else 0.0
    timing.append_row(a.out, {"device": timing._cpu(), "role": "cpu_term", "code": code, "p": p, "name": a.cell,
                              "proc": a.proc, "n": len(idx), "value_ms": frac * mean_ms,
                              "binding": timing.sig_digest({"cell": a.cell, "tau": c["tau"], "net": c["net_sha256"]}),
                              "note": json.dumps({"split": a.split, "chunks": a.chunks, "tag": a.tag, "shots": len(margin),
                                                  "routed": len(routed), "routed_fraction": frac, "timed": len(idx),
                                                  "mean_ms_routed": mean_ms, "read_gpu": sig["gpu"],
                                                  "routed_sha": _sha(routed.astype(np.int64).tobytes())[:16],
                                                  "corecheck_s": cc})})


# ---- counting the timing rows (PLAN 2.7, STAGE3_PLAN.md 4) --------------------------------------------------------------
def _rows(path):
    import csv
    rows = []
    for r in csv.DictReader(open(path)):
        r["note"] = json.loads(r["note"]) if r.get("note") else {}
        r["value_ms"] = float(r["value_ms"]) if r.get("value_ms") not in (None, "") else None
        r["p"] = float(r["p"]) if r.get("p") else None
        rows.append(r)
    return rows


def _count(rows, k, how, **match):
    """The counted value of rows matching every key (note keys too), one per process: need at least k; 'max' or 'min'."""
    sel = []
    for r in rows:
        if all((r["note"].get(key[5:]) if key.startswith("note_") else r.get(key)) == v for key, v in match.items()):
            if r["role"] in ("gpu_term", "handoff") or r["name"] == "gpu_bposd":
                if not r["note"].get("accepted"):
                    continue
            sel.append(r)
    vals = [r["value_ms"] for r in sel]
    procs = {r["process"] for r in sel}
    if len(vals) < k or len(procs) != len(vals):
        return {"value_ms": None, "n": len(vals), "error": f"{len(vals)} rows from {len(procs)} processes, need {k}"}
    return {"value_ms": (max if how == "max" else min)(vals), "n": len(vals), "all_ms": vals}


def _gpu_side(rows, net, session):
    g = _count(rows, 5, "max", role="gpu_term", name=f"fused:{net[:12]}", session=session)
    h = _count(rows, 5, "max", role="handoff", name=f"fused:{net[:12]}", session=session)
    hv = None if h["value_ms"] is None else max(HANDOFF_MS, h["value_ms"])
    return g, dict(h, value_ms=hv)


def prefreeze(a):
    cells = _cells()
    G, C = _rows(a.gpu_rows), _rows(a.cpu_rows)
    pre = {"gpu_session": a.gpu_session, "cpu_session": a.cpu_session, "networks": {}, "opponents": {}}
    for t, net in cells["networks"].items():
        g, h = _gpu_side(G, net, a.gpu_session)
        pre["networks"][t] = {"gpu_term": g, "handoff": h}
    for p in ("0.003", "0.002"):
        o = {d: _count(C, 3, "min", role="opponent", name=d, p=float(p), session=a.cpu_session, note_split="cal")
             for d in ("relay", "bposd")}
        o["gpu_bposd"] = _count(G, 3, "min", role="opponent", name="gpu_bposd", p=float(p), session=a.gpu_session,
                                note_split="cal")
        pre["opponents"][p] = o
    ver = {}
    for n in ("BB72-3", "BB72-2"):                          # Q8: PLAN 2.6 (ii) with the RTX session, as registered
        c = cells["cells"][n]
        p = str(c["p"])
        cpu = [pre["opponents"][p][d]["value_ms"] for d in ("relay", "bposd")]
        comp_rtx = RTX["gpu_term_ms"] + RTX["handoff_ms"]
        bar_rtx = min(cpu + [RTX["gpu_bposd_ms"][p]])
        nw = pre["networks"][c["tag"]]
        comp_h = nw["gpu_term"]["value_ms"] + nw["handoff"]["value_ms"]
        bar_h = min(cpu + [pre["opponents"][p]["gpu_bposd"]["value_ms"]])
        ver[n] = {"rtx": {"composite_ms": comp_rtx, "bar_ms": bar_rtx, "pass": comp_rtx <= bar_rtx, "src": RTX},
                  "h100": {"composite_ms": comp_h, "bar_ms": bar_h, "pass": comp_h <= bar_h}}
    pre["stage2_speed"] = ver
    c = cells["cells"]["BB72-2F"]                           # 0.1 item 1: the screen on the H100-routed cal workload
    cpu_t = _count(C, 3, "max", role="cpu_term", name="BB72-2F", session=a.cpu_session, note_split="cal")
    nw = pre["networks"][c["tag"]]
    comp = nw["gpu_term"]["value_ms"] + nw["handoff"]["value_ms"] + cpu_t["value_ms"]
    o = pre["opponents"]["0.002"]
    bar = min(o[d]["value_ms"] for d in GATING)
    pre["BB72-2F_screen"] = {"cpu_term": cpu_t, "composite_ms": comp, "bar_ms": bar,
                             "bar_from": min(GATING, key=lambda d: o[d]["value_ms"]), "pass": comp <= bar}
    cells["prefreeze"] = pre
    cells["cells"]["BB72-2F"]["status"] = "tested" if comp <= bar else "dropped: failed its pre-freeze screen"
    json.dump(cells, open(CELLS, "w"), indent=1)
    print(json.dumps(pre, indent=1), flush=True)


# ---- the one analysis -------------------------------------------------------------------------------------------------
def _wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [ctr - h, ctr + h]


def analyze(a):
    from bb import opp, route
    cells = _cells(a.amend)
    am = cells.get("amendment")
    G, C = (_rows(os.path.join(shots.HERE, cells["rows"][k])) for k in ("gpu", "cpu"))
    post = cells["post"]
    Ct, ct_session = (_rows(os.path.join(shots.HERE, am["rows_cpu"])), am["cpu_session"]) if am else \
        (C, post["cpu_session"])
    res = {"plan": am["document"] if am else "ising_compare/bb/STAGE3_PLAN.md", "alpha": route.ALPHA_TEST,
           "study_union_bound": am["study_union_bound"] if am else 0.05 * 5 / 3, "prefreeze": cells["prefreeze"],
           "splits": {}, "cells": {}}
    if am:
        res.update(title=am["title"], amendment={k: am[k] for k in ("registered", "document_sha256", "cpu_session")})
    obs, nets, opps = {}, {}, {}
    for s in cells["splits"]:
        code, p, ch = _split(cells, s)
        chunks = shots.parse_range(ch)
        obs[s] = shots.load_many(code, p, "test", chunks)[1]
        n = len(obs[s])
        res["splits"][s] = {"code": code, "p": p, "chunks": ch, "shots": n, "decoders": {}}
        for d in GATING + ("relay_full",):
            pred, secs = opp.load_preds(code, p, "test", chunks, d)
            opps[(s, d)] = (pred != obs[s]).any(1)
            k = int(opps[(s, d)].sum())
            res["splits"][s]["decoders"][d] = {"failures": k, "rate": k / n, "wilson95": _wilson(k, n)}
    for name, c in cells["cells"].items():
        s = c["split"]
        code, p, ch = _split(cells, s)
        chunks = shots.parse_range(ch)
        o = obs[s]
        if (s, c["tag"]) not in nets:
            nets[(s, c["tag"])] = route._load_net(code, p, "test", chunks, c["tag"])
        bits, margin, sig, meta = nets[(s, c["tag"])]
        pol = json.load(open(os.path.join(route.POLICIES, c["policy"])))
        diff = sorted(k for k in set(sig) | set(pol["engine"]) if sig.get(k) != pol["engine"].get(k))
        r = {"split": s, "tag": c["tag"], "tau": c["tau"], "fallback": c["fallback"], "engine_differs_in": diff,
             "engine_gpu": sig["gpu"], "status": c.get("status", "tested")}
        if diff not in ([], ["gpu"]) or sig["gpu"] != cells["gpu"] or sig["net_sha256"] != c["net_sha256"]:
            r["error"] = "the read is not the frozen engine on the frozen GPU"
        routed = margin < c["tau"] if c["tau"] is not None else np.zeros(len(o), bool)
        if routed.any():
            fb = opp.load_preds(code, p, "test", chunks, c["fallback"])[0]
            ours = (np.where(routed[:, None], fb, bits) != o).any(1)
        else:
            ours = (bits != o).any(1)
        k = int(ours.sum())
        r.update(shots=len(o), routed=int(routed.sum()), routed_fraction=float(routed.mean()), failures=k,
                 rate=k / len(o), wilson95=_wilson(k, len(o)), network_alone=int((bits != o).any(1).sum()), vs={})
        for d in c["vs"]:
            r["vs"][d] = route.paired(ours, opps[(s, d)], route.ALPHA_TEST)
        r["accuracy"] = all(v["verdict"] == "better" for v in r["vs"].values())
        if c["fallback"] is None:                          # PLAN 3.2: full-DEM Relay-BP beside, first 200,000 shots
            m = min(200_000, len(o))
            r["reference_full_dem_first200k"] = route.paired(ours[:m], opps[(s, "relay_full")][:m], route.ALPHA_TEST)
            if len(o) > m:
                r["reference_full_dem_all"] = route.paired(ours, opps[(s, "relay_full")], route.ALPHA_TEST)
        # speed (4.3): GPU side from the pre-freeze H100 session; CPU rows from session B; bars on the same hosts
        g, h = _gpu_side(G, c["net_sha256"], post["gpu_session_pre"])
        cpu_t = {"value_ms": 0.0, "n": 0, "note": "nothing routed"} if c["fallback"] is None else \
            _count(Ct, 3, "max", role="cpu_term", name=name, session=ct_session, note_split="test")
        bars = {d: _count(C, 3, "min", role="opponent", name=d, p=p, session=post["cpu_session"], note_split="test")
                for d in ("relay", "bposd")}
        bars["gpu_bposd"] = _count(G, 3, "min", role="opponent", name="gpu_bposd", p=p,
                                   session=post["gpu_session_post"], note_split="test")
        sp = {"gpu_term": g, "handoff": h, "cpu_term": cpu_t, "gating": bars}
        if None in (g["value_ms"], h["value_ms"], cpu_t["value_ms"]) or any(b["value_ms"] is None for b in bars.values()):
            sp["error"] = "missing timing rows"
            r["speed"] = False
        else:
            sp["composite_ms"] = g["value_ms"] + h["value_ms"] + cpu_t["value_ms"]
            sp["fastest_gating_ms"] = min(b["value_ms"] for b in bars.values())
            sp["fastest_gating"] = min(bars, key=lambda d: bars[d]["value_ms"])
            if c["speed"] == "relay_full_own":
                own = _count(C, 3, "min", role="opponent", name="relay_full", p=p, session=post["cpu_session"],
                             note_split="test")
                sp["relay_full_own"] = own
                sp["bar_ms"] = own["value_ms"]
            else:
                sp["bar_ms"] = sp["fastest_gating_ms"]
            r["speed"] = sp["bar_ms"] is not None and sp["composite_ms"] <= sp["bar_ms"]
        r["timing"] = sp
        r["claim"] = bool(r["accuracy"] and r["speed"] and "error" not in r and r["status"] == "tested")
        res["cells"][name] = r
    res["idle_checks"] = {"rows": sum(r["role"] == "idle" for r in G),
                          "not_ok": sum(r["role"] == "idle" and not r["note"].get("ok") for r in G)}
    tail = am["suffix"] if am else ""
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"results{tail}.json"), "w"), indent=1)
    open(os.path.join(a.out, f"RESULTS_stage3{tail}.md"), "w").write(_report(res))
    _figure(res, os.path.join(a.out, f"bb72_stage3{tail}.png"))
    if am:                                                  # the registered cells and these, one row per test split
        every = {**json.load(open(os.path.join(STAGE3, "results.json")))["cells"], **res["cells"]}
        _figure(dict(res, cells={n: every[n] for n in am["figure_all"]}), os.path.join(a.out, "bb72_stage3_all.png"))
    print(_report(res), flush=True)


def _report(res):
    L = [res.get("title", "# BB72 stage 3: results of the one analysis (generated by `python -m bb.stage3 analyze`)"), "",
         f"Each comparison: exact two-sided McNemar p < {res['alpha']:.6f} and the paired Newcombe (1 - alpha) interval "
         f"of LER(ours) - LER(opponent) entirely below 0. Union bound over all claims of the study: "
         f"{res['study_union_bound']:.4f}.", "",
         "| cell | test shots | ours: failures (routed) | opponent | its failures | only ours / only it | McNemar p | "
         "Newcombe interval | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["cells"].items():
        for d, v in r["vs"].items():
            L.append(f"| {n} | {r['shots']:,} | {r['failures']:,} ({r['routed']:,}) | {d} | {v['fail_b']:,} | "
                     f"{v['only_a']:,} / {v['only_b']:,} | {v['mcnemar_p']:.2e} | [{v['diff_lo']:.2e}, "
                     f"{v['diff_hi']:.2e}] | {v['verdict']} |")
    L += ["", "| cell | composite (ms) | GPU term | handoff | CPU term | bar (ms) | bar from | speed | accuracy | "
          "claim |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["cells"].items():
        t = r["timing"]
        if "composite_ms" in t:
            src = "full-DEM Relay-BP's own time" if "relay_full_own" in t else t["fastest_gating"]
            L.append(f"| {n} | {t['composite_ms']:.4f} | {t['gpu_term']['value_ms']:.4f} | {t['handoff']['value_ms']:.4f} "
                     f"| {t['cpu_term']['value_ms']:.4f} | {t['bar_ms']:.4f} | {src} | {r['speed']} | {r['accuracy']} | "
                     f"{r['claim']} |")
        else:
            L.append(f"| {n} | {t.get('error')} | | | | | | {r['speed']} | {r['accuracy']} | {r['claim']} |")
    for n, r in res["cells"].items():
        for k in ("reference_full_dem_first200k", "reference_full_dem_all"):
            if k in r:
                v = r[k]
                L.append(f"- {n} against full-DEM Relay-BP ({k[19:]}, reported, not gated): {v['fail_a']:,} vs "
                         f"{v['fail_b']:,} of {v['shots']:,}, p {v['mcnemar_p']:.2e}, {v['verdict']}")
    if "amendment" in res:
        for n, r in res["cells"].items():
            L.append(f"- {n}: {r['routed']:,} shots routed to {r['fallback']}; on the same {r['shots']:,} shots the network "
                     f"alone fails {r['network_alone']:,} times, the hybrid {r['failures']:,} times (reported, not gated)")
    return "\n".join(L) + "\n"


def _figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(res["cells"])
    ncol = len(names) if len(names) <= 4 else 3
    ink, ink2, ours_c, surf = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"     # dataviz reference palette: series 1 + ink
    plt.rcParams.update({"text.color": ink, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": "#c3c2b7"})
    fig, ax = plt.subplots(-(-len(names) // ncol), ncol, figsize=(3.2 * ncol, 3.4 * -(-len(names) // ncol)),
                           sharey=False, facecolor=surf, squeeze=False)
    lab = {"relay": "sector Relay-BP", "bposd": "BP-OSD (CPU)", "gpu_bposd": "cudaq-qec BP-OSD", "relay_full":
           "full-DEM Relay-BP"}
    for x, n in zip(ax.flat, names):
        r = res["cells"][n]
        s = res["splits"][r["split"]]
        items = [("ours" if r["fallback"] is None else "hybrid", r["rate"], r["wilson95"], ours_c)]
        items += [(lab[d], s["decoders"][d]["rate"], s["decoders"][d]["wilson95"], ink2)
                  for d in ("relay", "bposd", "gpu_bposd", "relay_full")]
        for i, (lb, v, ci, col) in enumerate(items):
            x.errorbar(i, 100 * v, yerr=[[100 * (v - ci[0])], [100 * (ci[1] - v)]], fmt="o", color=col, capsize=3,
                       markersize=6, elinewidth=1)
        x.set_facecolor(surf)
        x.set_ylim(bottom=0)
        x.set_xticks(range(len(items)), [i[0] for i in items], rotation=45, ha="right", fontsize=8)
        x.set_title(f"{n} (p = {s['p']}, {s['shots']:,} shots)", fontsize=9)
        x.set_ylabel("block failure rate (%)", fontsize=8)
        x.grid(axis="y", alpha=0.3)
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hostcheck", "beacon", "agree", "cputerm", "prefreeze", "analyze"])
    ap.add_argument("--out", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--cell", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunks", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--first", type=int, default=5000)
    ap.add_argument("--proc", type=int, default=1)
    ap.add_argument("--gpu-rows", default="")
    ap.add_argument("--cpu-rows", default="")
    ap.add_argument("--gpu-session", default="")
    ap.add_argument("--cpu-session", default="")
    ap.add_argument("--amend", default="")
    a = ap.parse_args()
    {"hostcheck": hostcheck, "beacon": beacon, "agree": agree, "cputerm": cputerm, "prefreeze": prefreeze,
     "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
