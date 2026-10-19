"""ARB: a read's seeds and rows, checked from the data (retest/jobs/d1_seedcheck_rt.py with ARB's sets, and digest
comparisons with D1's and the re-test's recorded chunks). Run through the shim, so that eval_seed is the patched one:

  python arb_shim.py arb_seedcheck.py --read DIR --p P --release RELEASE [--sample 8]

Reads DIR/p<P>.json and .npz and DIR/trace_<P>/ and writes DIR/seedcheck_<P>.json; exit status 1 when a check fails.
  1. the read is kind 14, family depspam_fm, chunks c0 .. c0 + n - 1 (c0 = 6,433,102, n at most D1's block), and every
     chunk's recorded seed is ARB's seed j of the set (arb_seeds.chunk_seeds, recomputed from the beacon) and eval_seed's;
  2. the shim's trace: for every chunk exactly one call by the main process and one by a sampling worker, each with
     ARB's seed (at least two workers when there are 16 chunks or more);
  3. for the first chunk, the last and evenly spaced ones between (--sample): the saved rows equal a fresh draw from
     the recorded seed and the recorded digest is the fresh draw's; a draw from D1's arithmetic seed of that chunk
     differs; and whether that draw reproduces D1's recorded digest of the chunk (this host's Stim draws D1's rows);
  4. every chunk digest of the read is distinct, none equals any digest of D1's reads or of the re-test's reads at either
     p (release: cascade_compare/phase_d/out/read/p*.json, retest/run/d1/*/read/p*.json)."""
import argparse
import glob
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C0, CHUNK = 6_433_102, 10_000
SETS = {0.002: "arb-0.002", 0.003: "arb-0.003"}
TAGS = {0.002: "p0.00200", 0.003: "p0.00300"}


def digest(d):
    return hashlib.sha256(np.packbits(d, axis=1).tobytes()).hexdigest()[:16]


def recorded(release):
    """{(source, p): {chunk: det_sha256}} of D1's and the re-test's reads."""
    out = {}
    for p, t in TAGS.items():
        for src, path in (("d1", f"cascade/bb72/cascade_compare/phase_d/out/read/{t}.json"),
                          ("retest", f"retest/run/d1/{p}/read/{t}.json")):
            m = json.load(open(os.path.join(release, path)))
            out[(src, p)] = {c["chunk"]: c["det_sha256"] for c in m["chunk_meta"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--sample", type=int, default=8)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("arb_seeds", os.path.join(HERE, "arb_seeds.py"))
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    from bb import cascade_data as cdata
    if os.environ.get("ARB_TRACE"):
        raise SystemExit("seedcheck: ARB_TRACE is set; this process would write into the trace it reads")
    if not getattr(cdata.eval_seed, "_arb", None):
        raise SystemExit("seedcheck: eval_seed is not the shim's (not run through arb_shim.py)")
    p, tag = a.p, TAGS[a.p]
    exp = sd.chunk_seeds(SETS[p], os.environ["ARB_SEEDS"], os.environ["ARB_RELEASE"])
    meta = json.load(open(os.path.join(a.read, f"{tag}.json")))
    c0, c1 = meta["chunks"]
    n = c1 - c0
    cm = meta["chunk_meta"]
    res = {"p": p, "set": SETS[p], "chunks": [c0, c1], "n_chunks": n, "checks": {}}
    ck = res["checks"]
    ck["kind_family_block"] = meta["kind"] == 14 and meta["family"] == "depspam_fm" and c0 == C0 and 0 < n <= len(exp)
    ck["chunks_in_order"] = [c["chunk"] for c in cm] == list(range(c0, c1))
    ck["recorded_seeds_are_arbs"] = [c["seed"] for c in cm] == exp[:n]
    ck["recorded_seeds_are_eval_seed"] = all(c["seed"] == cdata.eval_seed(14, p, c["chunk"], "depspam_fm") for c in cm)
    ck["seeds_in_2^62_2^63"] = all(2 ** 62 <= c["seed"] < 2 ** 63 for c in cm)
    recs = []
    for fn in sorted(glob.glob(os.path.join(a.read, f"trace_{p}", "trace_*.tsv"))):
        for line in open(fn):
            pid, ppid, p_, chunk_, seed_ = line.split("\t")
            recs.append((int(pid), int(ppid), float(p_), int(chunk_), int(seed_)))
    pids = {r[0] for r in recs}
    mains = [m for m in pids if all(r[1] == m for r in recs if r[0] != m)]
    main_pid = mains[0] if len(mains) == 1 else None
    work = sorted(pids - {main_pid})
    by = {"main": {}, "worker": {}}
    for r in recs:
        by["main" if r[0] == main_pid else "worker"].setdefault(r[3], []).append(r[4])
    want = {c0 + j: [exp[j]] for j in range(n)}
    ck["trace"] = {"records": len(recs), "main_pid": main_pid, "worker_pids": len(work),
                   "main_drew_every_chunk_once_with_arbs_seed": by["main"] == want,
                   "a_worker_drew_every_chunk_once_with_arbs_seed": by["worker"] == want,
                   "all_p_equal": all(r[2] == p for r in recs), "at_least_2_workers_if_16_chunks": len(work) >= 2 or n < 16}
    z = np.load(os.path.join(a.read, f"{tag}.npz"))
    det, chunk = z["det"], z["chunk"]
    ck["rows"] = len(det) == n * CHUNK and len(chunk) == len(det)
    old = recorded(a.release)
    mine = [c["det_sha256"] for c in cm]
    every_old = {v for d in old.values() for v in d.values()}
    ck["digests"] = {"distinct_within_read": len(set(mine)) == len(mine),
                     "none_equals_d1_or_retest": not (set(mine) & every_old),
                     "recorded_digests_compared": len(every_old),
                     "same_chunk_differs_from_d1": all(c["det_sha256"] != old[("d1", p)][c["chunk"]] for c in cm),
                     "same_chunk_differs_from_retest": all(c["det_sha256"] != old[("retest", p)][c["chunk"]] for c in cm)}
    circ = cdata.circuit_at("bb72", p, "depspam_fm")
    idx = sorted({0, n - 1} | {round(i * (n - 1) / max(a.sample - 1, 1)) for i in range(a.sample)})
    ck["sample"] = []
    for i in idx:
        c = cm[i]
        fresh, _ = circ.compile_detector_sampler(seed=c["seed"]).sample(CHUNK, separate_observables=True)
        fresh = fresh.astype(np.uint8)
        d1seed = cdata.eval_seed._orig(14, p, c["chunk"], "depspam_fm")
        other, _ = circ.compile_detector_sampler(seed=d1seed).sample(CHUNK, separate_observables=True)
        dother = digest(other.astype(np.uint8))
        rows = slice(i * CHUNK, (i + 1) * CHUNK)
        ck["sample"].append({"chunk": c["chunk"], "seed": c["seed"],
                             "saved_rows_equal_fresh_draw": bool(np.array_equal(det[rows], np.packbits(
                                 fresh, axis=1, bitorder="little"))),
                             "chunk_rows_labelled": bool((chunk[rows] == c["chunk"]).all()),
                             "det_sha256_equal_fresh_draw": digest(fresh) == c["det_sha256"],
                             "d1_seed_draw_differs": dother != c["det_sha256"],
                             "info_d1_seed_draw_reproduces_d1_digest": dother == old[("d1", p)][c["chunk"]]})
    tr, dg = ck["trace"], ck["digests"]
    ok = (all(v for k, v in ck.items() if k not in ("sample", "trace", "digests")) and main_pid is not None
          and all(v for v in tr.values() if isinstance(v, bool))
          and all(v for v in dg.values() if isinstance(v, bool))
          and all(all(s[k] for k in s if k not in ("chunk", "seed") and not k.startswith("info_")) for s in ck["sample"]))
    res["pass"] = bool(ok)
    json.dump(res, open(os.path.join(a.read, f"seedcheck_{p}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("p", "set", "n_chunks", "pass")}
                     | {"sampled": [s["chunk"] for s in ck["sample"]],
                        "d1_seed_reproduces_d1": [s["info_d1_seed_draw_reproduces_d1_digest"] for s in ck["sample"]],
                        "digests": dg}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
