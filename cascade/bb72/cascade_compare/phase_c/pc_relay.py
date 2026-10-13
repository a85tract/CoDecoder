"""Phase C (PHASE_C_PLAN.md sections 4-5): full-DEM Relay-BP on the union of the routed sets of one read.

A read is a bb.cascade_eval --save DIR --save-det output (kind 10 = the development block, 11 = the confirmation block,
13 = dry runs). Every configuration that ends in Relay-BP marks the shots it hands over (pc_route.relay_sets); Relay-BP
decodes the union once, so each decision serves every configuration that routes that shot (finding 15).
  dev   the candidates of the rule (pc_route.dev_candidates), thresholds from this read (quantiles, numpy's default)
  conf  the locked configurations (--locked locked_<setting>.json), thresholds as locked, never re-estimated
Relay-BP: the frozen BB72 configuration (qnd.teachers.CONFIGS['relay']: relay-bp 0.2.2 RelayDecoderF32) on the full
detector error model of the read's circuit family at p (both sectors, identical mechanisms merged by
qnd.circuits.dem_matrices), as the earlier curve study (uniform) and the training stage (depspam_fm) ran it, one shot per call, the wall time of each call
kept. relay-bp draws the gammas of every decode from its decoder's own generator, so a decoder kept across shots
decides a shot according to what it decoded before, and two runs that split the union differently differ in a few
shots (dry runs 2 and 7). Here every shot gets a new decoder seeded with the shot's index in the read: a decision
depends only on the read and the shot (dry run 8). Output per p: DIR/relay_<p>.npz (idx = shot indices into the read,
pred, fail, secs) and DIR/relay_<p>.json (counts, time quantiles, the union's detector digest, the configurations).
  python pc_relay.py --read DIR --mode dev --p 0.002,0.003 --procs 64 --out DIR_R
  python pc_relay.py --read DIR --mode conf --locked locked_uniform.json --p 0.002,0.003 --procs 64 --out DIR_R"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import time
from multiprocessing import get_context

import numpy as np

import pc_common as pc
import pc_route

_mat, _dec = {}, {}


def decoder(family, p, seed=None):
    """(RelayDecoderF32, L). seed None: the process's one decoder per (family, p), kept (relay-bp's default seed 0;
    the timed pipelines, as deployed); an integer: a new decoder whose generator starts at that seed."""
    import relay_bp
    from qnd import teachers
    if (family, p) not in _mat:
        from bb import cascade_data as cdata
        from qnd import circuits
        dem = cdata.circuit_at(pc.CODE, p, family).detector_error_model(decompose_errors=False)
        H, L, q = circuits.dem_matrices(dem)
        _mat[(family, p)] = (H.tocsr().astype(np.uint8), q.astype(np.float64), L.tocsr().astype(np.uint8))
    if seed is None and (family, p) in _dec:
        return _dec[(family, p)]
    H, q, L = _mat[(family, p)]
    d = (relay_bp.RelayDecoderF32(H, error_priors=q, seed=0 if seed is None else int(seed),
                                  **teachers.CONFIGS["relay"][1]), L)
    if seed is None:
        _dec[(family, p)] = d
    return d


def relay_task(args):
    """(family, p, seeds (n,) or None, det rows (n, 432) uint8) -> (pred (n, 12) uint8, secs (n,)). seeds None: the
    kept decoder (pc_speed's pipelines); otherwise a new decoder per shot, seeded; secs times the decode alone."""
    family, p, seeds, det = args
    pred = np.zeros((len(det), pc.K), np.uint8)
    secs = np.zeros(len(det))
    for i in range(len(det)):
        dec, L = decoder(family, p, None if seeds is None else seeds[i])
        t = time.perf_counter()
        e = dec.decode(np.ascontiguousarray(det[i]))
        secs[i] = time.perf_counter() - t
        pred[i] = (L @ e.astype(np.uint8)) % 2
    return pred, secs


def versions():
    out = {}
    for pkg in ("stim", "relay-bp", "numpy", "scipy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--mode", required=True, choices=["dev", "conf"])
    ap.add_argument("--locked", default="")
    ap.add_argument("--p", default=",".join(str(p) for p in pc.RELAY_P))
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--task", type=int, default=10, help="shots per Relay-BP task")
    ap.add_argument("--max-shots", type=int, default=0, help="dry runs only: decode at most this many of the union")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    locked = json.load(open(a.locked)) if a.locked else None
    with get_context("spawn").Pool(a.procs) as pool:
        for p in (float(x) for x in a.p.split(",")):
            t0 = time.time()
            z, meta, idx = pc.load_read(a.read, p)
            fam = meta["family"]
            if locked and locked["family"] != fam:
                raise SystemExit(f"{a.locked} is for {locked['family']}, the read is {fam}")
            if "det" not in z:
                raise SystemExit(f"{a.read} at {p}: no detector rows (cascade_eval --save-det)")
            if a.mode == "dev":
                cfgs = pc_route.dev_candidates(fam, p, idx, relay_only=True)
                taus = pc_route.dev_taus(z, meta, idx, cfgs)
            else:
                cfgs = [c for c in locked["configs"] if c["p"] == p and c["type"] in ("S-B-R", "S-R")]
                taus = [c["tau"] for c in cfgs]
            sets = [pc_route.masks(z, meta, idx, c, t)[1] for c, t in zip(cfgs, taus)]
            union = np.nonzero(np.any(sets, axis=0))[0] if sets else np.zeros(0, np.int64)
            n_union = len(union)
            if a.max_shots and n_union > a.max_shots:
                union = union[:a.max_shots]
            det = np.unpackbits(z["det"][union], axis=1, count=432, bitorder="little")
            obs = np.unpackbits(z["obs"][union], axis=1, count=pc.K)
            tasks = [(fam, p, union[i:i + a.task], det[i:i + a.task]) for i in range(0, len(det), a.task)]
            out = pool.map(relay_task, tasks, chunksize=1) if tasks else []
            pred = np.concatenate([o[0] for o in out]) if out else np.zeros((0, pc.K), np.uint8)
            secs = np.concatenate([o[1] for o in out]) if out else np.zeros(0)
            fail = (pred != obs).any(1)
            np.savez(os.path.join(a.out, f"relay_{pc.p_tag(p)}.npz"), idx=union, pred=pred, fail=fail, secs=secs)
            q = (lambda v: float(np.quantile(secs, v)) if len(secs) else None)
            rec = {"p": p, "family": fam, "mode": a.mode, "read": os.path.abspath(a.read), "read_kind": meta["kind"],
                   "read_chunks": meta["chunks"], "shots": int(len(z["obs"])), "configs": len(cfgs),
                   "union": n_union, "decoded": int(len(union)), "truncated": bool(len(union) < n_union),
                   "relay_fail": int(fail.sum()), "secs_mean": float(secs.mean()) if len(secs) else None,
                   "secs_p50": q(0.5), "secs_p99": q(0.99), "secs_p999": q(0.999),
                   "core_s": float(secs.sum()), "wall_s": round(time.time() - t0, 1), "procs": a.procs,
                   "task": a.task, "seed": "the shot's index in the read, a new decoder per shot",
                   "union_det_sha256": hashlib.sha256(np.packbits(det, axis=1).tobytes()).hexdigest()[:16],
                   "circuit_sha256": meta["circuit_sha256"], "host": platform.node(), "versions": versions(),
                   "locked_sha256": pc.sha256_file(a.locked) if a.locked else None}
            json.dump(rec, open(os.path.join(a.out, f"relay_{pc.p_tag(p)}.json"), "w"), indent=1)
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
