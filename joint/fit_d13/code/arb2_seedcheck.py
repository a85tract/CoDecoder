"""ARB at d = 13: a read's seeds and rows, checked from the data (ARB's arb_seedcheck.py for BB72, with the d = 13
comparison's and the re-test's chunks). Run through the shim, so that eval_seed is the patched one:

  python arb2_shim.py arb2_seedcheck.py --read DIR --p P --release RELEASE [--sample 8] [--old-all --procs N]

Reads DIR/p<p>/part_*.npz and DIR/trace_<p>/ and writes DIR/seedcheck_<p>.json; exit status 1 when a check fails.
  1. the parts cover chunks 0 .. n - 1 of kind 11 at p in order (n at most the set's size), and every chunk's recorded
     seed is ARB's seed j of the set (arb2_seeds.chunk_seeds, recomputed from the beacon) and eval_seed's;
  2. the shim's trace: every record carries ARB's seed of its chunk; every chunk has exactly one call by the main process
     (the read records the seed) and one by a sampling worker (the draw), at least two workers when there are 16
     chunks or more; a resumed read (several attempts) may repeat chunks, which is reported, not failed;
  3. for the first chunk, the last and evenly spaced ones between (--sample): the saved rows and observables equal a
     fresh draw from the recorded seed, and the recorded digest (sv_common.digest) is the fresh draw's; the rows differ
     from those of the same chunk number of the comparison's confirmation block (kind 11, the pinned arithmetic seed),
     of the re-test's (retest/run/seeds.json, set d13-0.006 or d13-0.0035) and of the comparison's development block
     (kind 10), by digest; and for chunks 0-2, whether the comparison's seed reproduces the digest that the re-test's
     replay drew on the cluster with the patch off (retest/replay/d13/replay_seeds_off.json): this host's Stim then draws
     the comparison's rows from the comparison's seeds;
  4. every chunk digest of the read is distinct; with --old-all, none equals any chunk digest of the comparison's
     confirmation block, the re-test's block or the comparison's development block at this p (all recomputed here by
     Stim in --procs spawn workers; the reads' own records of these digests are on the cluster)."""
import argparse
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import time

import numpy as np

from surface import surface_data as sdata
from surface import sv_common as sc

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {0.006: "arb-d13-0.006", 0.0035: "arb-d13-0.0035"}
RT_SETS = {0.006: "d13-0.006", 0.0035: "d13-0.0035"}
CONF_CHUNKS = {0.006: 100, 0.0035: 3000}         # the comparison's and the re-test's confirmation blocks
DEV_CHUNKS = sc.DEV_CHUNKS                        # the comparison's development blocks (kind 10)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--old-all", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    if os.environ.get("ARB2_TRACE"):
        raise SystemExit("seedcheck: ARB2_TRACE is set; this process would write into the trace it reads")
    if not getattr(sdata.eval_seed, "_arb2", None):
        raise SystemExit("seedcheck: eval_seed is not the shim's (not run through arb2_shim.py)")
    sd = load("arb2_seeds", os.path.join(HERE, "arb2_seeds.py"))
    rt = load("rt_seeds", os.path.join(a.release, "retest", "seeds.py"))
    p, kind = a.p, sc.KIND_CONF
    exp = sd.chunk_seeds(SETS[p], os.environ["ARB2_SEEDS"], os.environ["ARB2_RELEASE"])
    rts = rt.chunk_seeds(RT_SETS[p], os.path.join(a.release, "retest", "run", "seeds.json"),
                         os.path.join(a.release, "retest", "run", "beacon.json"))
    orig = sdata.eval_seed._orig
    d = os.path.join(a.read, sc.p_tag(p))
    parts = sorted(glob.glob(os.path.join(d, "part_*.npz")))
    seeds, digs, chunks, where = [], [], [], {}
    for f in parts:
        with np.load(f) as z:
            m = json.loads(str(z["meta"]))
        s, e = m["chunks"]
        if m["p"] != p or m["kind"] != kind or len(m["seeds"]) != e - s or len(m["digests"]) != e - s:
            raise SystemExit(f"{f}: not a part of this read")
        for i, j in enumerate(range(s, e)):
            where[j] = (f, i)
        chunks += list(range(s, e))
        seeds += m["seeds"]
        digs += m["digests"]
    n = len(chunks)
    res = {"p": p, "set": SETS[p], "parts": len(parts), "n_chunks": n, "checks": {}}
    ck = res["checks"]
    ck["chunks_0_to_n_in_order"] = chunks == list(range(n)) and 0 < n <= len(exp)
    ck["recorded_seeds_are_arbs"] = seeds == exp[:n]
    ck["recorded_seeds_are_eval_seed"] = all(s == sdata.eval_seed(sc.D, kind, p, j) for j, s in zip(chunks, seeds))
    ck["seeds_in_2^62_2^63"] = all(2 ** 62 <= s < 2 ** 63 for s in seeds)
    ck["seeds_differ_from_retest_and_comparison"] = not (set(seeds) & set(rts)) and all(
        s != orig(sc.D, kind, p, j) for j, s in zip(chunks, seeds))
    recs = []
    for fn in sorted(glob.glob(os.path.join(a.read, f"trace_{p}", "trace_*.tsv"))):
        for line in open(fn):
            pid, ppid, p_, chunk_, seed_ = line.rstrip("\n").split("\t")
            recs.append((int(pid), int(ppid), float(p_), int(chunk_), int(seed_)))
    pids = {r[0] for r in recs}
    parents = {r[1] for r in recs if r[1] in pids}     # a main process is the parent of the workers that traced
    mains = sorted(parents) or sorted(pid for pid in pids if not any(r[1] == pid for r in recs))[:1]
    by = {"main": {}, "worker": {}}
    for r in recs:
        by["main" if r[0] in mains else "worker"].setdefault(r[3], []).append(r[4])
    attempts = len(glob.glob(os.path.join(d, "read_*.json")))
    once = {j: [exp[j]] for j in range(n)}
    tr = {"records": len(recs), "main_pids": len(mains), "worker_pids": len(pids - set(mains)), "attempts": attempts,
          "every_record_has_arbs_seed": all(0 <= r[3] < len(exp) and r[4] == exp[r[3]] and r[2] == p for r in recs),
          "main_called_every_chunk": set(by["main"]) == set(range(n)),
          "a_worker_drew_every_chunk": set(by["worker"]) == set(range(n)),
          "main_once_per_chunk": by["main"] == once, "worker_once_per_chunk": by["worker"] == once,
          "at_least_2_workers_if_16_chunks": len(pids - set(mains)) >= 2 or n < 16}
    ck["trace"] = tr
    ck["digests_distinct_within_read"] = len(set(digs)) == len(digs)
    off = json.load(open(os.path.join(a.release, "retest", "replay", "d13", "replay_seeds_off.json")))
    alpine = {r["chunk"]: r["digest"] for r in off["checks"]["spawned_sampling_pool"]["rows"] if r["p"] == p}
    idx = sorted({0, n - 1} | {round(i * (n - 1) / max(a.sample - 1, 1)) for i in range(a.sample)})
    ck["sample"] = []
    for i in idx:
        j = chunks[i]
        f, r = where[j]
        with np.load(f) as z:
            rows = slice(r * sc.CHUNK, (r + 1) * sc.CHUNK)
            det_s, obs_s, ch_s = z["det"][rows], z["obs"][rows], z["chunk"][rows]
        det, obs = sdata.sample(sc.D, p, sc.CHUNK, seeds[i])
        dg = sc.digest(det, obs)
        dc = sc.digest(*sdata.sample(sc.D, p, sc.CHUNK, orig(sc.D, kind, p, j))) if j < 10 ** 7 else None
        dr = sc.digest(*sdata.sample(sc.D, p, sc.CHUNK, rts[j])) if j < len(rts) else None
        dd = sc.digest(*sdata.sample(sc.D, p, sc.CHUNK, orig(sc.D, sc.KIND_DEV, p, j))) if j < DEV_CHUNKS[p] else None
        row = {"chunk": j, "seed": seeds[i], "digest": digs[i],
               "saved_rows_equal_fresh_draw": bool(np.array_equal(det_s, np.packbits(det, axis=1))),
               "saved_obs_equal_fresh_draw": bool(np.array_equal(obs_s, obs[:, 0])),
               "chunk_rows_labelled": bool((ch_s == j).all()),
               "recorded_digest_equals_fresh_draw": dg == digs[i],
               "differs_from_comparison_conf_chunk": dc != digs[i],
               "differs_from_retest_chunk": dr is None or dr != digs[i],
               "differs_from_comparison_dev_chunk": dd is None or dd != digs[i],
               "info_comparison_digest": dc, "info_retest_digest": dr, "info_dev_digest": dd}
        if j in alpine:
            row["info_comparison_seed_reproduces_cluster_digest"] = dc == alpine[j]
        ck["sample"].append(row)
    if a.old_all:
        from arb2_digest import chunk_digest
        jobs = ([("conf", j, orig(sc.D, kind, p, j)) for j in range(CONF_CHUNKS[p])]
                + [("retest", j, rts[j]) for j in range(CONF_CHUNKS[p])]
                + [("dev", j, orig(sc.D, sc.KIND_DEV, p, j)) for j in range(DEV_CHUNKS[p])])
        t1 = time.time()
        with mp.get_context("spawn").Pool(a.procs) as pool:
            out = pool.map(chunk_digest, [(p, s) for _, _, s in jobs], chunksize=4)
        old = {}
        for (blk, j, _), dgo in zip(jobs, out):
            old.setdefault(blk, {})[j] = dgo
        mine = set(digs)
        ck["old_all"] = {blk: {"chunks": len(v), "equal_to_a_read_digest": len(set(v.values()) & mine)}
                         for blk, v in old.items()}
        ck["old_all"]["secs"] = round(time.time() - t1, 1)
        ck["old_all"]["none_equal"] = all(v["equal_to_a_read_digest"] == 0 for k, v in ck["old_all"].items()
                                          if isinstance(v, dict))
        ck["old_all"]["cluster_digests_reproduced"] = {str(j): old["conf"].get(j) == dg for j, dg in alpine.items()}
        res["old_digests"] = old
    flat = [v for k, v in ck.items() if isinstance(v, bool)]
    ok = (all(flat) and all(tr[k] for k in ("every_record_has_arbs_seed", "main_called_every_chunk",
                                            "a_worker_drew_every_chunk", "at_least_2_workers_if_16_chunks"))
          and (attempts > 1 or (tr["main_once_per_chunk"] and tr["worker_once_per_chunk"]))
          and all(all(v for kk, v in s.items() if isinstance(v, bool) and not kk.startswith("info_"))
                  for s in ck["sample"])
          and (not a.old_all or ck["old_all"]["none_equal"]))
    res["pass"] = bool(ok)
    res["secs"] = round(time.time() - t0, 1)
    with open(os.path.join(a.read, f"seedcheck_{p}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("p", "set", "n_chunks", "pass", "secs")}
                     | {"trace": tr, "sampled": [s["chunk"] for s in ck["sample"]],
                        "cluster_digest_reproduced": [s.get("info_comparison_seed_reproduces_cluster_digest")
                                                      for s in ck["sample"]],
                        "old_all": {k: v for k, v in ck.get("old_all", {}).items()}}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
