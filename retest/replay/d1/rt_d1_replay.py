"""Replay checks 2 and 3 of D1's shim (rt_d1.py), run through it:
    python retest/shim/rt_d1.py retest/replay/d1/rt_d1_replay.py OUT.json [--procs 4] [--original-read DIR]
Mode ON when RT_SEEDS is set (the replay uses the dry seeds), mode OFF when it is not. Writes OUT.json, exit status 1 on a failure.

ON  1. main process: eval_seed(14, p, c0 + j, depspam_fm) equals the beacon's seed of beacon index j, for every chunk of both
       blocks (1,100 and 1,700), recomputed here from the beacon's randomness with the seed formula (not through chunk_seeds);
       the seeds are in [2^62, 2^63) and differ from D1's original seeds of the same chunks;
    2. any other chunk of kind 14 / depspam_fm (outside the blocks, other p) raises ValueError;
    3. every other (kind, p, chunk, family) of a grid (kinds 0-99, 8 p, 14 chunks, both families: 22,288 cells, the 112 cells of kind 14 / depspam_fm
       being 1 and 2) equals the original function, loaded from the same file as a separate module;
    4. spawn workers, started as pc_eval's are (bb.cascade_eval's chunks_in_order over mp.get_context("spawn").Pool, the
       worker function bb.cascade_data._eval_chunk): the seed each worker computes for the first, second and last chunk
       of both blocks is the intended one (pool.starmap of eval_seed), and the shots the workers return equal a direct
       draw in this process from the intended seed (and differ from the draw from D1's original seed).
OFF 1. eval_seed is the original function (no patch attribute) and equals the original on the grid of 3 and on both blocks'
       first, second and last chunks (kind 14 / depspam_fm included);
    2. the workers compute the original seeds, and their first chunk of the block at p = 0.2 % has the digest of D1's
       original read when --original-read is given (informational: Stim's stream depends on the host's SIMD width)."""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT_DIR = os.path.dirname(os.path.dirname(HERE))
C0, CHUNK = 6_433_102, 10_000
SETS = {0.002: ("d1-0.002", 1100), 0.003: ("d1-0.003", 1700)}
KINDS, PS = range(100), (0.001, 0.002, 0.003, 0.0035, 0.004, 0.0055, 0.006, 0.01)
CHUNKS = (0, 1, 7, 99, 100, 1000, 3000, C0 - 1, C0, C0 + 1099, C0 + 1100, C0 + 1699, C0 + 1700, 9_999_999)
FAMS = ("uniform", "depspam_fm")


def digest(d):
    return hashlib.sha256(np.packbits(d, axis=1).tobytes()).hexdigest()[:16]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def outcome(f, *a):
    try:
        return ("ok", f(*a))
    except Exception as e:                       # noqa: BLE001
        return ("err", type(e).__name__)


def grid_equal(patched, orig, skip):
    """patched == orig on the grid except where skip(kind, p, chunk, family) -> (cells compared, mismatches)."""
    n = bad = 0
    for k in KINDS:
        for p in PS:
            for c in CHUNKS:
                for fam in FAMS:
                    if skip(k, p, c, fam):
                        continue
                    n += 1
                    bad += outcome(patched, k, p, c, fam) != outcome(orig, k, p, c, fam)
    return {"cells": n, "mismatches": bad}


def in_block(k, p, c, fam):
    return k == 14 and fam == "depspam_fm" and p in SETS and C0 <= c < C0 + SETS[p][1]


def kind14_depspam(k, p, c, fam):
    return k == 14 and fam == "depspam_fm"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--original-read", default="")
    a = ap.parse_args()
    from bb import cascade_data as cd
    from bb import cascade_eval as ce
    orig = load_module("cd_orig", cd.__file__).eval_seed
    on = bool(os.environ.get("RT_SEEDS"))
    res = {"mode": "ON" if on else "OFF", "node": os.uname().nodename, "pid": os.getpid(),
           "cpu_flags": sorted(f for f in open("/proc/cpuinfo").read().split("flags")[1].split("\n")[0].split()
                               if f.startswith(("avx", "sse4", "ssse3"))),
           "patched_function": bool(getattr(cd.eval_seed, "_rt_d1", None)), "checks": {}}
    ck = res["checks"]
    wsample = [(p, j) for p, (_, n) in SETS.items() for j in (0, 1, n - 1)]
    if on:
        sd = load_module("rt_seeds", os.path.join(RT_DIR, "seeds.py"))
        r = sd.randomness(os.environ["RT_BEACON"])
        f = json.load(open(os.environ["RT_SEEDS"]))
        res.update(seeds_file=os.environ["RT_SEEDS"], beacon_round=f["beacon_round"])
        bad = n = 0
        distinct_from_original = in_range = True
        for p, (name, nch) in SETS.items():
            s = f["sets"][name]
            for j in range(nch):
                want = sd.seed(r, s["tag"].encode(), s["indices"][j])
                got = cd.eval_seed(14, p, C0 + j, "depspam_fm")
                n += 1
                bad += got != want
                distinct_from_original &= got != orig(14, p, C0 + j, "depspam_fm")
                in_range &= 2 ** 62 <= got < 2 ** 63
        ck["1_main_process_seeds_are_the_beacons"] = {"cells": n, "mismatches": bad, "pass": bad == 0 and n == 2800}
        ck["1_seeds_in_range_and_differ_from_the_original"] = {"pass": bool(distinct_from_original and in_range)}
        errs = [outcome(cd.eval_seed, 14, p, c, "depspam_fm") for p, c in
                [(0.002, C0 - 1), (0.002, C0 + 1100), (0.003, C0 + 1700), (0.002, 0), (0.003, 9_999_999),
                 (0.001, C0), (0.0025, C0), (0.004, C0), (0.0035, C0)]]
        ck["2_other_kind14_chunks_are_errors"] = {"outcomes": errs, "pass": all(e == ("err", "ValueError") for e in errs)}
        g = grid_equal(cd.eval_seed, orig, kind14_depspam)
        k14 = [(p, c, fam) for p in PS for c in CHUNKS for fam in FAMS if fam != "depspam_fm"]
        same_k14 = all(outcome(cd.eval_seed, 14, p, c, fam) == outcome(orig, 14, p, c, fam) for p, c, fam in k14)
        ck["3_other_kinds_and_families_unchanged"] = {**g, "kind14_other_family_equal": same_k14,
                                                      "pass": g["mismatches"] == 0 and same_k14}
        want = {(p, j): sd.seed(r, f["sets"][SETS[p][0]]["tag"].encode(), f["sets"][SETS[p][0]]["indices"][j])
                for p, j in wsample}
        with mp.get_context("spawn").Pool(a.procs) as pool:
            pids = sorted(set(pool.starmap(os.getpid, [()] * (4 * a.procs))))
            got = pool.starmap(cd.eval_seed, [(14, p, C0 + j, "depspam_fm") for p, j in wsample])
            jobs = [("bb72", 14, p, C0 + j, CHUNK, "depspam_fm") for p, j in wsample]
            shots = list(ce.chunks_in_order(pool, jobs, 2 * a.procs))
        direct, orig_draw = [], []
        for (p, j), (det_w, obs_w) in zip(wsample, shots):
            circ = cd.circuit_at("bb72", p, "depspam_fm")
            d, o = circ.compile_detector_sampler(seed=want[(p, j)]).sample(CHUNK, separate_observables=True)
            d0, _ = circ.compile_detector_sampler(seed=orig(14, p, C0 + j, "depspam_fm")).sample(
                CHUNK, separate_observables=True)
            direct.append(bool(np.array_equal(det_w, d.astype(np.uint8)) and np.array_equal(obs_w, o.astype(np.uint8))))
            orig_draw.append(digest(d0.astype(np.uint8)) != digest(det_w))
        ck["4_workers_use_the_intended_seeds"] = {
            "worker_pids": pids, "chunks": [[p, C0 + j] for p, j in wsample],
            "worker_seeds_equal_beacon": [g_ == want[k] for g_, k in zip(got, wsample)],
            "worker_shots_equal_direct_draw": direct, "worker_shots_differ_from_original_seed_draw": orig_draw,
            "pass": all(g_ == want[k] for g_, k in zip(got, wsample)) and all(direct) and all(orig_draw)
                    and os.getpid() not in pids}
    else:
        g = grid_equal(cd.eval_seed, orig, lambda *x: False)
        blocks = [(p, C0 + j) for p, j in wsample]
        same = all(cd.eval_seed(14, p, c, "depspam_fm") == orig(14, p, c, "depspam_fm") for p, c in blocks)
        ck["1_patch_off_is_the_original"] = {**g, "blocks_equal": same, "no_patch_attribute": not res["patched_function"],
                                             "pass": g["mismatches"] == 0 and same and not res["patched_function"]}
        with mp.get_context("spawn").Pool(a.procs) as pool:
            got = pool.starmap(cd.eval_seed, [(14, p, c, "depspam_fm") for p, c in blocks])
            jobs = [("bb72", 14, p, c, CHUNK, "depspam_fm") for p, c in blocks]
            shots = list(ce.chunks_in_order(pool, jobs, 2 * a.procs))
        eq = [g_ == orig(14, p, c, "depspam_fm") for g_, (p, c) in zip(got, blocks)]
        ck["2_workers_use_the_original_seeds"] = {"worker_seeds_equal_original": eq, "pass": all(eq)}
        if a.original_read:
            m = json.load(open(os.path.join(a.original_read, "p0.00200.json")))["chunk_meta"]
            dig = {c["chunk"]: c["det_sha256"] for c in m}
            same_digest = {str(c): digest(det) == dig.get(c) for (p, c), (det, _) in zip(blocks, shots) if p == 0.002}
            ck["2_workers_reproduce_the_original_chunks_informational"] = {"digest_equal": same_digest,
                                                                          "pass": True}
    res["pass"] = all(v["pass"] for v in ck.values())
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"mode": res["mode"], "pass": res["pass"], **{k: v["pass"] for k, v in ck.items()}}), flush=True)
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
