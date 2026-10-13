"""Correlated PM on every SURF-v2 calibration shot of a distance, one shot at a time on one pinned core: its answer and
its time, the q rule's inputs (surf_v2/qrule.py: F_cal and cost(q)). Run once per distance, in one job holding a whole
EPYC 7713 node (the CPU term's node type), while the networks train.

The loop is the CPU term's (timing.cpu_term.loop: 50 untimed warm-up decodes, then perf_counter around each
matcher.decode(syndrome, enable_correlations=True)) over all 200,000 shots in set order. Its ratio to the Grace basis is
the CPU term's: G(d) (timing/grace_basis.csv, correlated PM) over the same node's correlated PM per block on validation
chunk 0, the median of the three `timing.cpu_term ref` repetitions the job ran first (--ref: their directory). The
answers on chunk 0 are checked against PyMatching's batch decode.

    python -m surf_v2.calcost --d 11 --shots CAL --ref REFDIR --core 8 --out CC
-> CC/calcost_d11.npz (dec uint8, time_us float64 on this node, obs uint8) and its .json record."""
import argparse
import glob
import json
import os
import time

import numpy as np

from convlines import circuits, provenance, testset
from timing import cpu_term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--shots", required=True, help="the sv2cal chunk files")
    ap.add_argument("--set", default="sv2cal", help="sv2dry for dry runs")
    ap.add_argument("--chunks", default="0-19")
    ap.add_argument("--ref", required=True, help="the directory with this job's ref_d<d>_rep*.json")
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refs = sorted(glob.glob(os.path.join(a.ref, f"ref_d{a.d}_rep*.json")))
    ref = [json.load(open(f)) for f in refs]
    assert len(ref) == 3 and all(r["kind"] == "correlated" for r in ref), f"three correlated refs of d = {a.d}: {refs}"
    det, obs = testset.load_chunks(a.set, a.d, testset.parse_range(a.chunks), a.shots)
    m = circuits.matcher(a.d, correlated=True)
    pinned = cpu_term.pin(a.core)
    h = provenance.host()
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ts, dec = cpu_term.loop(m, np.ascontiguousarray(det), True)
    t1 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    batch = circuits.pm_decode(m, det[:testset.CHUNK], correlated=True)
    assert np.array_equal(batch, dec[:testset.CHUNK]), "the single-shot answers differ from the batch decode"
    g, src = cpu_term.grace_numerator(a.d)
    node = float(np.median([r["per_block_us"] for r in ref]))
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"calcost_d{a.d}")
    testset.write_npz(stem + ".npz", dict(dec=dec.astype(np.uint8), time_us=ts.astype(np.float64), obs=obs.astype(np.uint8)))
    rec = dict(what="correlated PM's answer and time on every calibration shot (surf_v2/calcost.py)", d=a.d, set=a.set,
               chunks=a.chunks, shots=int(len(obs)), manifest_sha256=testset.sha256_bytes(open(os.path.join(
                   testset.ROOT, testset.SETS[a.set]), "rb").read()), core=a.core, pinned=pinned, host=h,
               versions=provenance.versions(), code_digest=provenance.code_digest(), release_commit=provenance.commit(),
               failures=int((dec != obs).sum()), mean_us=float(ts.mean()), ref_reps_us=[r["per_block_us"] for r in ref],
               ref_files=[os.path.basename(f) for f in refs], node_median_us=node, grace_numerator_us=g,
               grace_numerator_source=src, ratio=g / node, started=t0, ended=t1,
               npz_sha256=testset.sha256_bytes(open(stem + ".npz", "rb").read()))
    with open(stem + ".json", "x") as fh:
        json.dump(rec, fh, indent=1)
    print(f"d{a.d}: correlated PM {rec['failures']} failures in {len(obs)} calibration shots; {rec['mean_us']:.2f} us per "
          f"shot on this core (ref {node:.2f} us per block, ratio {rec['ratio']:.4f})", flush=True)


if __name__ == "__main__":
    main()
