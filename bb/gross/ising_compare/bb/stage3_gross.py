"""Stage 3 on gross, cell G-4 (bb/STAGE3_GROSS_PLAN.md): host check, the frozen engine's reads and their check, the
beacon, the CPU term, the pre-freeze speed gate (Q3) and the one analysis. BB72's bb/stage3.py is reused unchanged.
The frozen parameters are in data/bb/stage3_gross/cell.json; the policy is data/bb/stage3_gross/frontierf_*.json.
  hostcheck  python -m bb.stage3_gross hostcheck --out H.json: cal chunk 0 (sampled on Alpine) and monitor chunk 10
             (sampled on the gross2 stage-1 pod) at p 0.004 regenerated on this host and compared as arrays; exit 1 on any
             difference (then no test sampling here)
  infer      python -m bb.stage3_gross infer --split test --chunks 0-29 --ckpt CK [--tag T]: the frozen network through
             its frozen engine (checkpoint sha256 and engine signature equal to the policy's); blind on the test split
  engcheck   python -m bb.stage3_gross engcheck --ref DIR --tag T --out E.json: this host's cal read (tag T) against
             stage 2's H100 read (the policy's tag, files in DIR): decisions, margins, the routed set at tau; exit 1
             unless every decision and the routed set are equal
  beacon     python -m bb.stage3_gross beacon --commit SHA --after EPOCH: the first drand mainnet round at least 600 s
             after max(the freeze commit's committer time, EPOCH = when the commit was seen public on GitHub), written
             to data/bb/stage3_gross/beacon.json
  cputerm    python -m bb.stage3_gross cputerm --split test --chunks 0-29 --first 1000 --proc K --out R: sector
             Relay-BP on the routed shots (margin < tau, split order, the first FIRST timed); CPU term = the split's
             routed fraction x the mean time per timed shot
  prefreeze  python -m bb.stage3_gross prefreeze --gpu-rows G --cpu-rows C: session A's speed gate (plan 4.2), written
             into cell.json
  analyze    python -m bb.stage3_gross analyze --out data/bb/stage3_gross: the one analysis (plan section 5)"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

from bb import shots
from bb.stage3 import _count, _rows, _sha, _wilson

STAGE3G = os.path.join(shots.HERE, "data", "bb", "stage3_gross")
CELL = os.path.join(STAGE3G, "cell.json")
REPO = os.path.dirname(shots.HERE)
GATING = ("relay", "bposd", "gpu_bposd")
HANDOFF_MS = 1.1e-3                       # PLAN 2.7: the handoff allowance, 1.1 us


def _cell():
    return json.load(open(CELL))


def _policy(c):
    path = os.path.join(shots.HERE, c["policy"])
    raw = open(path, "rb").read()
    if _sha(raw) != c["policy_sha256"]:
        raise SystemExit(f"{c['policy']} is not the frozen policy")
    return json.loads(raw)


def hostcheck(a):
    import stim
    from bb.timing import _cpu
    flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
    out = {"host": platform.node(), "cpu": _cpu(), "avx2": " avx2" in flags, "machine": platform.machine(),
           "python": sys.version.split()[0], "numpy": np.__version__, "stim": stim.__version__, "chunks": []}
    ok = True
    for split, j in (("cal", 0), ("monitor", 10)):
        det0, obs0 = shots.load("bb144", 0.004, split, j)
        det, obs = shots.sample("bb144", 0.004, split, j)
        same = bool(np.array_equal(det, det0) and np.array_equal(obs, obs0))
        out["chunks"].append({"file": shots.name("bb144", 0.004, split, j), "seed": shots.seed_of(split, "bb144", 0.004, j),
                              "equal": same, "det_sha256": _sha(np.packbits(det, axis=1).tobytes())})
        ok = ok and same
    out["stim_modules"] = sorted(m for m in sys.modules if m.startswith("stim._"))
    out["ok"] = ok and out["avx2"]
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    raise SystemExit(0 if out["ok"] else 1)


def infer(a):
    import torch
    from bb import engine, route
    c = _cell()
    pol = _policy(c)
    code, p = c["code"], c["p"]
    model, ck = engine.load_net(a.ckpt, torch.device("cuda"), False)
    sha = engine.sha256_file(a.ckpt)
    if sha != c["net_sha256"]:
        raise SystemExit(f"{a.ckpt} is not the frozen network")
    eng = engine.Deployed(model, pol["engine"]["kind"], sha)
    sig = eng.signature()
    if sig != pol["engine"]:
        raise SystemExit(f"the engine signature differs from the frozen one: "
                         f"{ {k: (v, pol['engine'].get(k)) for k, v in sig.items() if pol['engine'].get(k) != v} }")
    meta = {"ckpt": os.path.abspath(a.ckpt), "run": ck["hp"]["run"], "run_id": ck["hp"]["run_id"],
            "samples": ck.get("samples"), "status": ck.get("status")}
    tag = a.tag or c["tag"]
    blind = a.split == "test"                              # plan section 5: no test count before the analysis
    for j in shots.parse_range(a.chunks):
        det, obs = shots.load(code, p, a.split, j)
        bits, margin = eng.run(det)
        fail = (bits != obs).any(1)
        np.savez(route.net_path(code, p, a.split, j, tag), pred=bits, margin=margin, fail=fail,
                 sig=json.dumps(sig, sort_keys=True), meta=json.dumps(meta))
        print(f"net-{tag} {a.split} chunk {j}: {'stored' if blind else f'{int(fail.sum())} / {len(fail)} failures'}",
              flush=True)


def engcheck(a):
    from bb import route
    c = _cell()
    ch = shots.parse_range("0-9")
    ph, mh, sh, _ = route._load_net(c["code"], c["p"], "cal", ch, a.tag)
    z = [np.load(os.path.join(a.ref, os.path.basename(route.net_path(c["code"], c["p"], "cal", j, c["tag"]))))
         for j in ch]
    pr, mr = np.concatenate([x["pred"] for x in z]), np.concatenate([x["margin"] for x in z])
    sr = json.loads(str(z[0]["sig"]))
    eq = (ph == pr).all(1)
    d = np.abs(mh.astype(np.float64) - mr)
    rh, rr = mh < c["tau"], mr < c["tau"]
    out = {"shots": len(eq), "decisions_differ": int((~eq).sum()), "margin_absdiff_max": float(d.max()),
           "margins_bitwise_equal": int((d == 0).sum()), "routed_here": int(rh.sum()), "routed_stage2": int(rr.sum()),
           "routed_differ": int((rh != rr).sum()), "signature_equal": sh == sr, "gpu": sh["gpu"]}
    out["ok"] = out["decisions_differ"] == 0 and out["routed_differ"] == 0 and out["signature_equal"]
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    raise SystemExit(0 if out["ok"] else 1)


def beacon(a):
    sys.path.insert(0, shots.HERE)
    from icmp import t2
    tc = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", a.commit], cwd=REPO).strip())
    t = max(tc, int(a.after))
    b = t2.fetch_beacon(t2.beacon_round(t))
    t2.check_beacon(b, freeze_time=t)
    b.update(freeze_commit=subprocess.check_output(["git", "rev-parse", a.commit], cwd=REPO, text=True).strip(),
             freeze_commit_time=tc, public_seen_time=int(a.after), reference_time=t, delay_s=t2.DELAY,
             tags={"bb144_p0.004": shots.TEST_TAG[("bb144", 0.004)].decode()},
             seed_rule="2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62")
    json.dump(b, open(shots.BEACON_GROSS, "w"), indent=1)
    print(json.dumps({k: b[k] for k in ("round", "round_time", "randomness", "freeze_commit", "reference_time")}),
          flush=True)


def cputerm(a):
    from bb import timing
    c = _cell()
    pol = _policy(c)
    chunks = shots.parse_range(a.chunks)
    cc = timing.corecheck()
    idx, pop = timing.population(pol, a.split, chunks, a.first)    # checks the read's signature against the policy
    frac = pop["routed"] / pop["shots"]
    det = shots.load_many(c["code"], c["p"], a.split, chunks)[0][idx]
    secs = timing._secs(c["fallback"], c["code"], c["p"], det) if len(det) else np.zeros(1)
    timing.append_row(a.out, {"device": timing._cpu(), "role": "cpu_term", "code": c["code"], "p": c["p"],
                              "name": c["cell"], "proc": a.proc, "n": len(det), "value_ms": frac * 1e3 * float(secs.mean()),
                              "binding": c["policy_sha256"][:16],
                              "note": json.dumps({"split": a.split, "routed_fraction": frac,
                                                  "mean_ms_routed": 1e3 * float(secs.mean()), "population": pop,
                                                  "corecheck_s": cc})})


def _gpu_side(rows, c, session):
    name = f"{c['engine']['kind']}:{c['net_sha256'][:12]}"
    g = _count(rows, 5, "max", role="gpu_term", name=name, session=session)
    h = _count(rows, 5, "max", role="handoff", name=name, session=session)
    return g, dict(h, value_ms=None if h["value_ms"] is None else max(HANDOFF_MS, h["value_ms"]))


def _speed(G, C, c, gpu_session, gpu_opp_session, cpu_session, split):
    g, h = _gpu_side(G, c, gpu_session)
    cpu_t = _count(C, 3, "max", role="cpu_term", name=c["cell"], session=cpu_session, note_split=split)
    bars = {d: _count(C, 3, "min", role="opponent", name=d, p=c["p"], session=cpu_session, note_split=split)
            for d in ("relay", "bposd")}
    bars["gpu_bposd"] = _count(G, 3, "min", role="opponent", name="gpu_bposd", p=c["p"], session=gpu_opp_session,
                               note_split=split)
    sp = {"split": split, "gpu_term": g, "handoff": h, "cpu_term": cpu_t, "gating": bars}
    if None in (g["value_ms"], h["value_ms"], cpu_t["value_ms"]) or any(b["value_ms"] is None for b in bars.values()):
        return dict(sp, error="missing timing rows", ok=False)
    sp["composite_ms"] = g["value_ms"] + h["value_ms"] + cpu_t["value_ms"]
    sp["bar_ms"] = min(b["value_ms"] for b in bars.values())
    sp["bar_from"] = min(bars, key=lambda d: bars[d]["value_ms"])
    sp["ok"] = sp["composite_ms"] <= sp["bar_ms"]
    return sp


def prefreeze(a):
    c = _cell()
    s = c["sessions"]
    pre = _speed(_rows(a.gpu_rows), _rows(a.cpu_rows), c, s["gpu_pre"], s["gpu_pre"], s["cpu_A"], "cal")
    c["prefreeze"] = pre
    c["status"] = "tested" if pre["ok"] else "stopped: failed the pre-freeze speed gate (plan 4.2, Q3)"
    json.dump(c, open(CELL, "w"), indent=1)
    print(json.dumps(pre, indent=1), flush=True)


def analyze(a):
    from bb import opp, route
    c = _cell()
    pol = _policy(c)
    code, p, sp, chunks = c["code"], c["p"], c["split"], shots.parse_range(c["chunks"])
    obs = shots.load_many(code, p, sp, chunks)[1]
    n = len(obs)
    dec = {}
    fails = {}
    for d in GATING:
        fails[d] = (opp.load_preds(code, p, sp, chunks, d)[0] != obs).any(1)
        k = int(fails[d].sum())
        dec[d] = {"shots": n, "failures": k, "rate": k / n, "wilson95": _wilson(k, n)}
    m = c["reference_first"]
    full = (opp.load_preds(code, p, sp, shots.parse_range(c["reference_chunks"]), "relay_full")[0] != obs[:m]).any(1)
    dec["relay_full"] = {"shots": m, "failures": int(full.sum()), "rate": int(full.sum()) / m,
                         "wilson95": _wilson(int(full.sum()), m)}
    bits, margin, sig, meta = route._load_net(code, p, sp, chunks, c["tag"])
    routed = margin < c["tau"]
    ours = (np.where(routed[:, None], opp.load_preds(code, p, sp, chunks, c["fallback"])[0], bits) != obs).any(1)
    alone = (bits != obs).any(1)
    k = int(ours.sum())
    r = {"cell": c["cell"], "shots": n, "tau": c["tau"], "q": c["q"], "fallback": c["fallback"],
         "routed": int(routed.sum()), "routed_fraction": float(routed.mean()), "failures": k, "rate": k / n,
         "wilson95": _wilson(k, n), "network_alone": {"failures": int(alone.sum()), "rate": int(alone.sum()) / n,
                                                      "wilson95": _wilson(int(alone.sum()), n)},
         "engine_equal": sig == pol["engine"], "net_sha256": sig["net_sha256"], "status": c.get("status")}
    if not r["engine_equal"] or sig["net_sha256"] != c["net_sha256"]:
        r["error"] = "the test read is not the frozen engine"
    r["vs"] = {d: route.paired(ours, fails[d], route.ALPHA_TEST) for d in GATING}
    r["accuracy"] = all(v["verdict"] == "better" for v in r["vs"].values())
    r["reference_full_dem_first10k"] = route.paired(ours[:m], full, route.ALPHA_TEST)
    G, C = (_rows(os.path.join(shots.HERE, c["rows"][x])) for x in ("gpu", "cpu"))
    s = c["sessions"]
    r["timing"] = _speed(G, C, c, s["gpu_pre"], s["gpu_post"], s["cpu_B"], sp)
    r["speed"] = r["timing"]["ok"]
    r["claim"] = bool(r["accuracy"] and r["speed"] and "error" not in r and r["status"] == "tested")
    res = {"plan": c["plan"], "alpha": route.ALPHA_TEST, "cell": r, "decoders": dec, "prefreeze": c.get("prefreeze"),
           "idle_checks": {"rows": sum(x["role"] == "idle" for x in G),
                           "not_ok": sum(x["role"] == "idle" and not x["note"].get("ok") for x in G)},
           "command": c["analysis_command"]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    open(os.path.join(a.out, "RESULTS_stage3_gross.md"), "w").write(_report(res))
    _figure(res, os.path.join(a.out, "gross_stage3.png"))
    print(_report(res), flush=True)


def _report(res):
    r, t, dec = res["cell"], res["cell"]["timing"], res["decoders"]
    L = ["# Gross (BB144) stage 3, cell G-4: results of the one analysis (generated by "
         "`python -m bb.stage3_gross analyze`)", "",
         f"Each comparison: exact two-sided McNemar p < {res['alpha']:.6f} and the paired Newcombe (1 - alpha) interval of "
         f"LER(hybrid) - LER(opponent) entirely below 0; the claim needs all three and the speed condition.", "",
         f"Hybrid: tau = {r['tau']} (q = {r['q']} on CAL), {r['routed']:,} of {r['shots']:,} test shots routed to sector "
         f"Relay-BP ({100 * r['routed_fraction']:.3f} %); failures {r['failures']:,} (rate {r['rate']:.3e}, Wilson 95 % "
         f"[{r['wilson95'][0]:.3e}, {r['wilson95'][1]:.3e}]). The network alone: {r['network_alone']['failures']:,} "
         f"(reported, not gated).", "",
         "| opponent | its failures | only hybrid / only it | McNemar p | Newcombe interval | verdict |",
         "|---|---|---|---|---|---|"]
    for d, v in r["vs"].items():
        L.append(f"| {d} | {v['fail_b']:,} | {v['only_a']:,} / {v['only_b']:,} | {v['mcnemar_p']:.2e} | "
                 f"[{v['diff_lo']:.2e}, {v['diff_hi']:.2e}] | {v['verdict']} |")
    v = r["reference_full_dem_first10k"]
    L += ["", f"Full-DEM Relay-BP on the first {v['shots']:,} test shots (reported, not gated): {v['fail_b']:,} failures "
              f"against the hybrid's {v['fail_a']:,} there (only hybrid {v['only_a']} / only it {v['only_b']}, McNemar p "
              f"{v['mcnemar_p']:.2e}, {v['verdict']}).", ""]
    if "composite_ms" in t:
        L += ["| composite (ms) | GPU term | handoff | CPU term | bar (ms) | bar from | speed | accuracy | claim |",
              "|---|---|---|---|---|---|---|---|---|",
              f"| {t['composite_ms']:.4f} | {t['gpu_term']['value_ms']:.4f} | {t['handoff']['value_ms']:.4f} | "
              f"{t['cpu_term']['value_ms']:.4f} | {t['bar_ms']:.4f} | {t['bar_from']} | {r['speed']} | {r['accuracy']} | "
              f"{r['claim']} |", "",
              "Gating opponents' times (ms per shot; the fastest of 3 processes): " +
              ", ".join(f"{d} {b['value_ms']:.3f}" for d, b in t["gating"].items())]
    else:
        L.append(f"Speed: {t.get('error')}; accuracy {r['accuracy']}; claim {r['claim']}")
    if "error" in r:
        L.append(f"ERROR: {r['error']}")
    return "\n".join(L) + "\n"


def _figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, ours_c, surf = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"     # as bb.stage3's figure
    plt.rcParams.update({"text.color": ink, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": "#c3c2b7"})
    r, dec = res["cell"], res["decoders"]
    items = [("hybrid", r["rate"], r["wilson95"], ours_c),
             ("network alone", r["network_alone"]["rate"], r["network_alone"]["wilson95"], "#8fb8e8"),
             ("sector Relay-BP", dec["relay"]["rate"], dec["relay"]["wilson95"], ink2),
             ("BP-OSD (CPU)", dec["bposd"]["rate"], dec["bposd"]["wilson95"], ink2),
             ("cudaq-qec BP-OSD", dec["gpu_bposd"]["rate"], dec["gpu_bposd"]["wilson95"], ink2),
             (f"full-DEM Relay-BP\n(first {dec['relay_full']['shots']:,})", dec["relay_full"]["rate"],
              dec["relay_full"]["wilson95"], ink2)]
    fig, x = plt.subplots(figsize=(5.2, 3.6), facecolor=surf)
    for i, (lb, v, ci, col) in enumerate(items):
        x.errorbar(i, 100 * v, yerr=[[100 * (v - ci[0])], [100 * (ci[1] - v)]], fmt="o", color=col, capsize=3,
                   markersize=6, elinewidth=1)
    x.set_facecolor(surf)
    x.set_ylim(bottom=0)
    x.set_xticks(range(len(items)), [i[0] for i in items], rotation=45, ha="right", fontsize=8)
    x.set_title(f"G-4: [[144,12,12]], p = 0.004, {r['shots']:,} test shots", fontsize=9)
    x.set_ylabel("block failure rate (%)", fontsize=8)
    x.grid(axis="y", alpha=0.3)
    x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hostcheck", "infer", "engcheck", "beacon", "cputerm", "prefreeze", "analyze"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunks", default="0-29")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--after", type=int, default=0)
    ap.add_argument("--first", type=int, default=1000, help="cputerm: routed shots timed (PLAN 2.7: 1,000 on gross)")
    ap.add_argument("--proc", type=int, default=1)
    ap.add_argument("--gpu-rows", default="")
    ap.add_argument("--cpu-rows", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
