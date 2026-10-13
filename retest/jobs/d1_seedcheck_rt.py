"""D1 re-test: the eager read's seeds against the beacon, and a sample of its chunks re-drawn from the recorded seeds.
Run through the shim (so eval_seed is the patched one):  python retest/shim/rt_d1.py d1_seedcheck_rt.py --read DIR --p P
Reads DIR/p<P>.json and .npz (bb.cascade_eval --save --save-det) and writes DIR/seedcheck_<P>.json; exit status 1 when a
check fails. The spawn workers that drew the shots ran the shim too (retest/shim/rt_d1.py); this check is the proof from
the data. Checks:
  1. the read is D1's kind 14, family depspam_fm, chunks c0 .. c0 + n - 1 (c0 = 6,433,102), and every chunk's recorded
     seed equals the beacon's seed of beacon index j (seeds.chunk_seeds, each recomputed from RT_BEACON) and eval_seed in
     this process;
  2. the shim's trace (DIR/trace/trace_<pid>.tsv, written by every process that called eval_seed for kind 14): for every chunk of
     the block exactly one call by the main process (the one whose pid is the parent of the others) and exactly one by a worker,
     each with the beacon's seed; with 16 chunks or more at least two different workers drew;
  3. for the first chunk, the last and evenly spaced chunks in between (--sample, default 8): the saved detector rows equal a
     fresh draw from the recorded seed in this process, and the chunk's det_sha256 in chunk_meta equals the fresh draw's;
     the draw from the original D1 seed of that chunk is different (the check can tell the two apart)."""
import argparse
import glob
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT_DIR = os.path.dirname(HERE)
C0, CHUNK = 6_433_102, 10_000
SETS = {0.002: "d1-0.002", 0.003: "d1-0.003"}


def digest(d):
    return hashlib.sha256(np.packbits(d, axis=1).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--sample", type=int, default=8)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rt_seeds", os.path.join(RT_DIR, "seeds.py"))
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    from bb import cascade_data as cdata
    if os.environ.get("RT_D1_TRACE"):
        raise SystemExit("seedcheck: RT_D1_TRACE is set; this process would write into the trace it reads")
    if not getattr(cdata.eval_seed, "_rt_d1", None):
        raise SystemExit("seedcheck: eval_seed is not the shim's (RT_SEEDS unset, or not run through rt_d1.py)")
    p, tag = a.p, f"p{a.p:.5f}"
    exp = sd.chunk_seeds(SETS[p])
    meta = json.load(open(os.path.join(a.read, f"{tag}.json")))
    c0, c1 = meta["chunks"]
    n = c1 - c0
    cm = meta["chunk_meta"]
    res = {"p": p, "set": SETS[p], "chunks": [c0, c1], "n_chunks": n, "checks": {}}
    ck = res["checks"]
    ck["kind_family_block"] = meta["kind"] == 14 and meta["family"] == "depspam_fm" and c0 == C0 and 0 < n <= len(exp)
    ck["chunks_in_order"] = [c["chunk"] for c in cm] == list(range(c0, c1))
    ck["recorded_seeds_are_the_beacons"] = [c["seed"] for c in cm] == exp[:n]
    ck["recorded_seeds_are_eval_seed"] = all(c["seed"] == cdata.eval_seed(14, p, c["chunk"], "depspam_fm") for c in cm)
    ck["seeds_in_2^62_2^63"] = all(2 ** 62 <= c["seed"] < 2 ** 63 for c in cm)
    recs = []
    for fn in sorted(glob.glob(os.path.join(a.read, "trace", "trace_*.tsv"))):
        for line in open(fn):
            pid, ppid, p_, chunk_, seed_ = line.split("\t")
            recs.append((int(pid), int(ppid), float(p_), int(chunk_), int(seed_)))
    pids = {r[0] for r in recs}
    mains = [m for m in pids if all(r[1] == m for r in recs if r[0] != m)]
    main = mains[0] if len(mains) == 1 else None
    work = sorted({r[0] for r in recs} - {main})
    by = {"main": {}, "worker": {}}
    for r in recs:
        by["main" if r[0] == main else "worker"].setdefault(r[3], []).append(r[4])
    want = {c0 + j: [exp[j]] for j in range(n)}
    ck["trace"] = {"records": len(recs), "main_pid": main, "worker_pids": len(work),
                   "main_drew_every_chunk_once_with_the_beacons_seed": by["main"] == want,
                   "a_worker_drew_every_chunk_once_with_the_beacons_seed": by["worker"] == want,
                   "all_p_equal": all(r[2] == p for r in recs), "at_least_2_workers_if_16_chunks": len(work) >= 2 or n < 16}
    z = np.load(os.path.join(a.read, f"{tag}.npz"))
    det, chunk = z["det"], z["chunk"]
    ck["rows"] = len(det) == n * CHUNK and len(chunk) == len(det)
    circ = cdata.circuit_at("bb72", p, "depspam_fm")
    idx = sorted({0, n - 1} | {round(i * (n - 1) / max(a.sample - 1, 1)) for i in range(a.sample)})
    ck["sample"] = []
    for i in idx:
        c = cm[i]
        fresh, _ = circ.compile_detector_sampler(seed=c["seed"]).sample(CHUNK, separate_observables=True)
        fresh = fresh.astype(np.uint8)
        other, _ = circ.compile_detector_sampler(seed=cdata.eval_seed._orig(14, p, c["chunk"], "depspam_fm")).sample(
            CHUNK, separate_observables=True)
        rows = slice(i * CHUNK, (i + 1) * CHUNK)
        ck["sample"].append({"chunk": c["chunk"], "seed": c["seed"],
                             "saved_rows_equal_fresh_draw": bool(np.array_equal(det[rows], np.packbits(
                                 fresh, axis=1, bitorder="little"))),
                             "chunk_rows_labelled": bool((chunk[rows] == c["chunk"]).all()),
                             "det_sha256_equal_fresh_draw": digest(fresh) == c["det_sha256"],
                             "original_seed_draw_differs": digest(other.astype(np.uint8)) != c["det_sha256"]})
    tr = ck["trace"]
    ok = (all(v for k, v in ck.items() if k not in ("sample", "trace")) and main is not None
          and all(v for k, v in tr.items() if isinstance(v, bool))
          and all(all(s[k] for k in s if k not in ("chunk", "seed")) for s in ck["sample"]))
    res["pass"] = bool(ok)
    json.dump(res, open(os.path.join(a.read, f"seedcheck_{p}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("p", "set", "n_chunks", "pass")} | {"sampled": [s["chunk"] for s in ck["sample"]]}),
          flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
