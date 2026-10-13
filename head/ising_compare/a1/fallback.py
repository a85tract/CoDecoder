"""A1's fallback answers (revision/REVISION_PLAN.md 2.1, "Pair curves"): each fallback decides every shot of a set once,
and every arm's pair takes the same answers. Output npz: pred (the answer: (n,) on the surface, (n, 12) on BB72), secs
(n,) the per-shot decode time measured here, meta (JSON: decoder, set digest, host).
  surface  correlated PyMatching on the verdict DEM (icmp.pm.matcher(9, True, verdict circuit)): answers from
           decode_batch per chunk (spawned workers), and one single-shot decode per shot timed with perf_counter, whose
           answer must equal the batch's (these times come from a shared node; the time axis takes A3's)
  bb72     `relay`: full-DEM Relay-BP under A6's policy as revision/REVISION_PLAN.md 2.6 states it (A6 has not run, so
           A1 runs the policy itself; A1 keeps this policy for its primary analysis whatever A6 later fixes:
           revision/A1_READ_PLAN.md section 6): every shot gets a new decoder (relay-bp 0.2.2, asserted in the job and
           in every worker; RelayDecoderF32 with qnd.teachers.CONFIGS['relay'], the registered BB72 configuration)
           built from the full-DEM matrices of the set's circuit (qnd.circuits.problem: both sectors, identical
           mechanisms merged; built once per worker process) and seeded with the shot's index in its set (a1.sets.index:
           10,000 c + r for row r of chunk c, the same in any subset of the chunks or shots), so an answer depends only
           on the shot, whatever the workers and tasks (relay(); tests/a1_fallback_layouts.py checks it on dry shots);
           secs = build + decode (a routed shot's latency includes its decoder build), and build (n,) is kept beside
           it. Needs relay-bp: bb's venv (a1/jobs/args/fallback_relay.txt). `standin`
           writes a stand-in for the dry runs only: the true observables with a random logical flipped on a fraction
           --flip of the shots and a constant time, labelled as such in meta
    python -m a1.fallback pm --kind dry --chunks 0-9 --out FILE --workers 8
    python -m a1.fallback relay --code bb72 --kind a1dry --chunks 4-9 --out FILE --workers 32
    python -m a1.fallback standin --code bb72 --kind a1dry --chunks 0-9 --out FILE"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import time

import numpy as np

from a1 import sets


def _pm_part(det):
    from icmp import circuits, pm
    m = pm.matcher(9, True, circuits.circuit(9, noise=circuits.NOISE_VERDICT))
    ans = pm.decode(m, det, True)
    secs = np.empty(len(det))
    one = np.empty(len(det), np.uint8)
    for i in range(len(det)):
        t0 = time.perf_counter()
        one[i] = np.asarray(m.decode(det[i], enable_correlations=True), np.uint8).reshape(-1)[0]
        secs[i] = time.perf_counter() - t0
    return ans, secs, int((one != ans).sum())


RELAY_BP = "0.2.2"                  # the relay-bp of qnd.teachers.CONFIGS['relay'] (the registered configuration)
_MATS = {}


def relay_bp_version():
    import importlib.metadata as md
    v = md.version("relay-bp")
    if v != RELAY_BP:                   # an Exception, not SystemExit: it must cross pool.map from a worker
        raise RuntimeError(f"relay-bp {v}, not {RELAY_BP}")
    return v


def _relay_mats():
    """(H, L, q, keep) of the BB72 X circuit at p = 0.3 %, full DEM (both sectors), built once per process."""
    if not _MATS:
        from qnd import circuits
        H, L, q, keep = circuits.problem("bb72", "X", sets.P["bb72"], "all")
        _MATS.update(H=H.tocsr().astype(np.uint8), L=L.tocsr().astype(np.uint8), q=q.astype(np.float64), keep=keep)
    return _MATS["H"], _MATS["L"], _MATS["q"], _MATS["keep"]


def _relay_part(args):
    """(the shots' indices in their set, det rows) -> (pred (n, 12) uint8, secs (n,), build (n,)) under A6's policy."""
    import relay_bp
    from qnd import teachers
    relay_bp_version()
    idx, det = args
    H, L, q, keep = _relay_mats()
    pred = np.zeros((len(det), L.shape[0]), np.uint8)
    secs, build = np.empty(len(det)), np.empty(len(det))
    for i in range(len(det)):
        t0 = time.perf_counter()
        dec = relay_bp.RelayDecoderF32(H, error_priors=q, seed=int(idx[i]), **teachers.CONFIGS["relay"][1])
        t1 = time.perf_counter()
        e = dec.decode(np.ascontiguousarray(det[i, keep]))
        secs[i], build[i] = time.perf_counter() - t0, t1 - t0
        pred[i] = (L @ e.astype(np.uint8)) % 2
    return pred, secs, build


def relay(det, idx, workers, part=2000):
    """A6's policy on det's shots, shot k seeded with idx[k] (its index in its set): tasks of `part` shots over
    `workers` spawned processes; the answers do not depend on either -> (pred, secs, build)."""
    assert len(idx) == len(det)
    tasks = [(idx[i:i + part], det[i:i + part]) for i in range(0, len(det), part)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_relay_part, tasks)
    return tuple(np.concatenate([r[k] for r in res]) for k in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pm", "relay", "standin"])
    ap.add_argument("--code", default="surface", choices=["surface", "bb72"])
    ap.add_argument("--kind", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--flip", type=float, default=0.002, help="standin: the share of shots answered wrong")
    a = ap.parse_args()
    code = "surface" if a.cmd == "pm" else a.code
    det, obs, digest = sets.load(code, a.kind, sets.parse_range(a.chunks))
    t0 = time.time()
    extra = {}
    if a.cmd == "pm":
        parts = [det[i:i + 2000] for i in range(0, len(det), 2000)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(_pm_part, parts)
        pred = np.concatenate([r[0] for r in res])
        secs = np.concatenate([r[1] for r in res])
        mism = sum(r[2] for r in res)
        assert mism == 0, f"single-shot and batch answers differ on {mism} shots"
        meta = {"decoder": "correlated PyMatching (icmp.pm), verdict DEM", "single_vs_batch_differ": mism}
    elif a.cmd == "relay":
        import importlib.metadata as md
        assert code == "bb72", "Relay-BP is the BB72 fallback"
        v = relay_bp_version()
        pred, secs, extra["build"] = relay(det, sets.index(sets.parse_range(a.chunks)), a.workers)
        meta = {"decoder": "full-DEM Relay-BP under A6's policy (revision/REVISION_PLAN.md 2.6), A1's frozen fallback "
                           f"(revision/A1_READ_PLAN.md section 6): relay-bp {v} RelayDecoderF32, "
                           "qnd.teachers.CONFIGS['relay'], a new decoder per shot seeded with the shot's index in its set "
                           "(10,000 chunk + row); secs = build + decode",
                "mean_build_secs": float(extra["build"].mean()), "relay_bp": v, "stim": md.version("stim"),
                "workers": a.workers, "task_shots": 2000}
    else:
        assert a.kind == sets.DRY[code], "the stand-in is for the dry kinds only"
        rng = np.random.default_rng(20261011)
        pred = obs.copy()
        wrong = rng.random(len(obs)) < a.flip
        pred[wrong, rng.integers(0, obs.shape[1], int(wrong.sum()))] ^= 1
        secs = np.full(len(obs), 0.066)
        meta = {"decoder": f"DRY STAND-IN (truth, {a.flip} of the shots one logical flipped), not Relay-BP"}
    fails = int((pred.reshape(len(pred), -1) != obs[:, :pred.reshape(len(pred), -1).shape[1]]).any(1).sum())
    meta.update(code=code, kind=a.kind, chunks=a.chunks, shots=int(len(det)), set_digest=digest, failures=fails,
                mean_secs=float(secs.mean()), host=platform.node(), wall_s=time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    assert a.out.endswith(".npz")
    tmp = a.out[:-4] + ".tmp.npz"                # renamed into place only when complete
    np.savez(tmp, pred=pred, secs=secs, **extra, meta=json.dumps(meta))
    os.replace(tmp, a.out)
    print("FALLBACK", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
