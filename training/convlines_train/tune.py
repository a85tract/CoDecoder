"""Threshold tuning: route a front end's least confident shots to a strong decoder and choose the threshold.

A trained front end decides every shot and gives a confidence (|c| for a surface-code network, the margin between the
two best logical classes for a BB network). A shot whose confidence is below the threshold tau goes to a strong
decoder instead. On calibration shots this tool
  1. runs the front end on every shot (its decisions, confidences and time per shot);
  2. runs the strong decoder only on the least confident shots, up to the maximum routed fraction q_max, one call per
     shot, timing every call;
  3. for each routed fraction q of a grid, takes tau(q) = the q-quantile of the confidence (numpy's linear rule, the
     studies' rule; tau(0) = 0 routes nothing), routes the shots with confidence < tau(q) (always a prefix of the
     least confident shots), and counts the pair's failures (a failure: any observable decided wrongly) and its mean
     time per shot, t_front + (the routed shots' strong-decoder times + a handoff time per routed shot) / n;
  4. picks q by a stated rule:
       budget  the fewest failures among the q whose mean time is within --budget-us (ties: the smaller q);
       paper   the rule of the paper (appendix "Choosing the routed fraction"): discard every q whose mean time exceeds
               the speed limit --limit-us (default: none); among the rest take the smallest q with at most
               M - 2 sqrt(2 M) failures, M = --target, a planning count of failures on these calibration shots (the
               studies used one near the most accurate baseline decoder's count), or --target-strong F: M = round(F x
               the strong decoder's failures on every calibration shot), which decodes every shot (the BB72, gross and
               d = 17 studies: F = 0.9, the strong decoder being their fallback); if none qualifies, the q with the
               fewest failures among them (the studies' fallback at d = 15 and 17).
The result (the frontier, the rule, its inputs and the choice) is a JSON file, with the frontier as CSV beside it; on
new shots the pair routes a shot when its confidence < tau.

Times are what this host measured: the strong decoder's wall time per call (Python call included) and the front end's
batch time per shot on --device. They rank the routed fractions on this host; they are not the deployed engines'
times. --front-us replaces the measured front-end time, --timing batch times the strong decoder once over all routed
shots and charges each the mean (for decoders that only run in batches, such as GPU decoders).

The strong decoder is any callable det (k, m) uint8 -> predicted observables (k, n_obs) (or (k,)); on the command line
it is a factory "module:function" called with the stim.Circuit (convlines_train.strong: PyMatching, plain and
correlated, and BP-OSD). In Python, tune() takes the callable itself. The front end is convlines_train.train's
final_ema.pt, a retrained network of the paper's circuits (training/surface's final_ema.pt) or a released network's
name (convlines_train.frontend.load). Calibration shots: --shots N with --seed S (one Stim call), --chunks BASE:K
(K chunks of 10,000, chunk j with seed BASE + j: the studies' calibration sets) or --calib FILE.npz.

    python -m convlines_train.tune --frontend RUN/final_ema.pt --shots 200000 --seed 7 \\
        --strong convlines_train.strong:pymatching --max-fraction 0.1 --rule budget --budget-us 40 --out RUN/tune.json
    python -m convlines_train.tune --frontend d9_b3_r764 --chunks 60090000:20 \\
        --strong convlines_train.strong:pymatching_correlated --grid 0,0.01,0.02,0.04,0.06,0.08,0.1 \\
        --rule paper --target 2500 --out tune_d9.json             (study 2's calibration set and rule at d = 9)"""
import argparse
import csv
import importlib
import json
import math
import os
import time

import numpy as np


def grid_of(max_fraction, step):
    """0, step, 2 step, ... up to max_fraction (included)."""
    n = int(math.floor(max_fraction / step + 1e-9))
    return sorted({round(k * step, 10) for k in range(n + 1)} | {round(max_fraction, 10)})


def run_strong(strong, det, idx, timing="shot"):
    """(predictions (len(idx), n_obs) uint8, microseconds per shot (len(idx),)) of the strong decoder on det[idx]."""
    if len(idx) == 0:
        return np.zeros((0, 1), np.uint8), np.zeros(0)
    if timing == "batch":
        t0 = time.perf_counter_ns()
        pred = np.asarray(strong(det[idx]), np.uint8).reshape(len(idx), -1)
        return pred, np.full(len(idx), (time.perf_counter_ns() - t0) / 1e3 / len(idx))
    preds, us = [], np.zeros(len(idx))
    for k, i in enumerate(idx):
        t0 = time.perf_counter_ns()
        p = strong(det[i:i + 1])
        us[k] = (time.perf_counter_ns() - t0) / 1e3
        preds.append(np.asarray(p, np.uint8).reshape(1, -1))
    return np.concatenate(preds), us


def frontier(conf, front_pred, obs, strong_pred, strong_us, grid, front_us=0.0, handoff_us=0.0):
    """One row per q of the grid. strong_pred / strong_us belong to the least confident shots in increasing confidence
    (a stable sort), as many as tau(max(grid)) routes."""
    conf = np.asarray(conf, np.float64)
    n = len(conf)
    obs = np.asarray(obs, np.uint8).reshape(n, -1)
    order = np.argsort(conf, kind="stable")
    front_fail = np.any(np.asarray(front_pred, np.uint8).reshape(n, -1) != obs, axis=1)
    k_max = len(strong_pred)
    strong_fail = np.any(strong_pred != obs[order[:k_max]], axis=1)
    sf = np.concatenate([[0], np.cumsum(strong_fail)])
    ff = np.concatenate([[0], np.cumsum(front_fail[order])])          # front-end failures among the k least confident
    tus = np.concatenate([[0.0], np.cumsum(strong_us)])
    rows = []
    for q in grid:
        tau = float(np.quantile(conf, q)) if q > 0 else 0.0
        k = int((conf < tau).sum())
        assert k <= k_max, f"q = {q}: {k} routed shots, only {k_max} decoded by the strong decoder"
        fails = int(front_fail.sum() - ff[k] + sf[k])
        rows.append(dict(q=q, tau=tau, routed=k, routed_fraction=k / n, failures=fails, ler=fails / n,
                         repaired=int(ff[k] - (front_fail[order[:k]] & strong_fail[:k]).sum()),
                         introduced=int((~front_fail[order[:k]] & strong_fail[:k]).sum()),
                         mean_us=front_us + (tus[k] + k * handoff_us) / n))
    return rows


def choose(rows, rule, budget_us=None, target=None, limit_us=None):
    """(chosen row, a description of the rule with its numbers)."""
    if rule == "budget":
        assert budget_us is not None, "rule budget needs --budget-us"
        ok = [r for r in rows if r["mean_us"] <= budget_us]
        if not ok:
            raise SystemExit(f"no q meets the budget {budget_us} us (q = 0 takes {rows[0]['mean_us']:.3f} us per shot)")
        best = min(ok, key=lambda r: (r["failures"], r["q"]))
        return best, dict(rule="budget", text="the fewest failures among the q within the time budget (ties: the "
                          "smaller q)", budget_us=budget_us, candidates=len(ok))
    if rule == "paper":
        assert target is not None, "rule paper needs --target"
        lim = target - 2 * math.sqrt(2 * target)
        fast = [r for r in rows if limit_us is None or r["mean_us"] <= limit_us]
        if not fast:
            raise SystemExit(f"no q meets the speed limit {limit_us} us")
        ok = [r for r in fast if r["failures"] <= lim]
        best = ok[0] if ok else min(fast, key=lambda r: (r["failures"], r["q"]))
        return best, dict(rule="paper", text="discard every q over the speed limit; the smallest q with at most "
                          "M - 2 sqrt(2 M) failures; none: the q with the fewest failures", target=target,
                          failure_limit=lim, limit_us=limit_us, qualifies=bool(ok))
    raise ValueError(rule)


def tune(front_pred, conf, obs, det, strong, max_fraction=0.1, grid=None, front_us=0.0, handoff_us=0.0,
         timing="shot", rule="budget", budget_us=None, target=None, limit_us=None, target_strong=None):
    """The whole procedure on given front-end outputs: (result dict, per-shot strong-decoder arrays). target_strong = F
    decodes every shot with the strong decoder and takes M = round(F x its failures)."""
    conf = np.asarray(conf, np.float64)
    grid = sorted(grid) if grid else grid_of(max_fraction, 0.005)
    tau_max = float(np.quantile(conf, grid[-1])) if grid[-1] > 0 else 0.0
    order = np.argsort(conf, kind="stable")
    idx = order if target_strong is not None else order[:int((conf < tau_max).sum())]
    strong_pred, strong_us = run_strong(strong, det, idx, timing)
    strong_failures = None
    if target_strong is not None:
        strong_failures = int(np.any(strong_pred != np.asarray(obs, np.uint8).reshape(len(conf), -1)[order], 1).sum())
        target = round(target_strong * strong_failures)
    rows = frontier(conf, front_pred, obs, strong_pred, strong_us, grid, front_us, handoff_us)
    best, how = choose(rows, rule, budget_us, target, limit_us)
    if target_strong is not None and rule == "paper":
        how.update(target_strong=target_strong, target_from="round(target_strong x strong_failures)")
    n = len(conf)
    res = dict(shots=n, max_fraction=grid[-1], strong_decoded=len(idx), front_us=front_us, handoff_us=handoff_us,
               strong_timing=timing, strong_us_mean=float(strong_us.mean()) if len(idx) else None,
               front_failures=rows[0]["failures"], strong_failures=strong_failures, choice=how, chosen=best,
               tau=best["tau"], q=best["q"], frontier=rows)
    return res, dict(index=idx, pred=strong_pred, us=strong_us)


def load_factory(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def calibration_shots(a, fe):
    if a.chunks:                                         # BASE:K[:SIZE]: K chunks, chunk j drawn with seed BASE + j
        base, k, size = (int(float(x)) for x in (a.chunks + ":10000").split(":")[:3])
        dets, obss = [], []
        for j in range(k):
            d, o = fe.circuit.compile_detector_sampler(seed=base + j).sample(size, separate_observables=True)
            dets.append(d.astype(np.uint8))
            obss.append(o.astype(np.uint8))
        return np.concatenate(dets), np.concatenate(obss)
    if a.calib:
        z = np.load(a.calib)
        det = z["det"]
        if det.shape[1] != fe.num_detectors:
            det = np.unpackbits(det, axis=1, count=fe.num_detectors)
        return det.astype(np.uint8), z["obs"].astype(np.uint8).reshape(len(det), -1)
    det, obs = fe.circuit.compile_detector_sampler(seed=a.seed).sample(a.shots, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def main():
    from convlines_train import frontend
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frontend", required=True, help="a trained front end (convlines_train.train's final_ema.pt)")
    ap.add_argument("--circuit", default="", help="the circuit of the calibration shots (default: the front end's)")
    ap.add_argument("--calib", default="", help="calibration shots: an npz with det (packed or not) and obs")
    ap.add_argument("--chunks", default="", help="BASE:K[:SIZE] = K chunks of SIZE (10,000) shots, chunk j with Stim "
                    "seed BASE + j (the studies' calibration sets: 60,000,000 + 10,000 d at d = 7-13, K = 20)")
    ap.add_argument("--shots", type=int, default=200_000, help="calibration shots to sample when --calib is not given")
    ap.add_argument("--seed", type=int, default=None, help="their Stim seed (required with --shots)")
    ap.add_argument("--strong", default="convlines_train.strong:pymatching_correlated", help="factory module:function")
    ap.add_argument("--max-fraction", type=float, default=0.1)
    ap.add_argument("--step", type=float, default=0.005, help="grid step of q")
    ap.add_argument("--grid", default="", help="explicit comma list of q (overrides --max-fraction and --step)")
    ap.add_argument("--front-us", type=float, default=None, help="front-end time per shot (default: measured)")
    ap.add_argument("--handoff-us", type=float, default=0.0, help="time charged per routed shot")
    ap.add_argument("--timing", choices=["shot", "batch"], default="shot")
    ap.add_argument("--rule", choices=["budget", "paper"], default="budget")
    ap.add_argument("--budget-us", type=float, default=None)
    ap.add_argument("--target", type=float, default=None, help="paper rule: the planning count M")
    ap.add_argument("--target-strong", type=float, default=None, help="paper rule: M = round(F x the strong decoder's "
                    "failures on every calibration shot), F = this (0.9 in the BB72, gross and d = 17 studies)")
    ap.add_argument("--limit-us", type=float, default=None, help="paper rule: the speed limit")
    ap.add_argument("--device", default="")
    ap.add_argument("--save-shots", action="store_true", help="also write the per-shot arrays (npz)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (a.calib or a.chunks) and a.seed is None:
        ap.error("--seed is required when the calibration shots are sampled")
    if a.target is not None and a.target_strong is not None:
        ap.error("give --target or --target-strong, not both")
    import stim
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fe = frontend.load(a.frontend, device)
    if a.circuit:
        fe.circuit = stim.Circuit.from_file(a.circuit)
        assert fe.circuit.num_detectors == fe.num_detectors, "the circuit's detectors differ from the front end's"
    det, obs = calibration_shots(a, fe)
    fe.infer(det[:min(len(det), 4096)])                                                        # warm-up
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    pred, conf = fe.infer(det)
    front_us = (time.perf_counter_ns() - t0) / 1e3 / len(det) if a.front_us is None else a.front_us
    strong = load_factory(a.strong)(fe.circuit)
    grid = [float(v) for v in a.grid.split(",")] if a.grid else grid_of(a.max_fraction, a.step)
    res, per_shot = tune(pred, conf, obs, det, strong, grid=grid, front_us=front_us, handoff_us=a.handoff_us,
                         timing=a.timing, rule=a.rule, budget_us=a.budget_us, target=a.target, limit_us=a.limit_us,
                         target_strong=a.target_strong)
    res.update(frontend=a.frontend, frontend_sha256=frontend.sha256(fe.file), strong=a.strong,
               device=device, front_us_measured=a.front_us is None,
               calibration=dict(file=a.calib, sha256=frontend.sha256(a.calib)) if a.calib else
               dict(chunks=a.chunks, shots=len(det)) if a.chunks else dict(shots=a.shots, seed=a.seed),
               circuit_sha256=frontend.text_sha256(str(fe.circuit)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    with open(os.path.splitext(a.out)[0] + "_frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["frontier"][0]))
        w.writeheader()
        w.writerows(res["frontier"])
    if a.save_shots:
        np.savez(os.path.splitext(a.out)[0] + "_shots.npz", conf=conf, pred=pred, obs=obs, **per_shot)
    print(f"{len(det)} calibration shots; front end alone {res['front_failures']} failures, {front_us:.3f} us per shot; "
          f"strong decoder on {len(per_shot['index'])} shots" + ("" if res["strong_failures"] is None else
          f", alone {res['strong_failures']} failures"), flush=True)
    for r in res["frontier"]:
        print(f"q = {100 * r['q']:5.2f} %  tau = {r['tau']:.4f}  routed {r['routed']:7d}  failures {r['failures']:6d}  "
              f"mean {r['mean_us']:9.3f} us", flush=True)
    c = res["chosen"]
    print(f"rule {a.rule}: q = {100 * c['q']:.2f} %, tau = {c['tau']!r}, {c['failures']} failures, "
          f"{c['mean_us']:.3f} us per shot -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
