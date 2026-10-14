"""D9-SHALLOW's per-shot reads of the 1,000,000 test shots (PLAN.md section 4), untimed, after commit B is public.

  base  the observable, plain PM's and correlated PM's answers: convlines.circuits (PyMatching 2.4.0 on the DEM of
        d9_legacy.stim, decompose_errors=True, approximate_disjoint_errors=True; correlated = enable_correlations=True),
        on the pod's CPUs                                                          -> READS/d9stest_base.npz, .json
  net   a network's logit c from the deployed Triton engine at batch one (timing.deployed.Logits: FROZEN config, fp16
        matrix inputs, fp32 accumulation, tables folded on the CPU in fp32): SURF-v2's test-read path and the
        re-test's mask path                                                        -> READS/d9stest_<net>.npz, .json
Every chunk is checked against the manifest (sha256) before it is read; every record names the manifest's digests, the
code digest, the network's sha256 and the host.
    PYTHONPATH=<tree>/surface_v2:<tree> python d9_shallow/reads_d9s.py base|net --shots SHOTS --manifest MAN.json
        [--net NAME] --out READS [--procs N]"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "surface_v2"), ROOT]

import numpy as np  # noqa: E402

from convlines import circuits, provenance, testset  # noqa: E402
from timing import deployed  # noqa: E402

D = 9
_M = {}


def chunks(shots, man):
    """[(path, sha256)] of the set's chunks in chunk order, exactly j = 0..n-1."""
    fs = sorted(man["files"].items(), key=lambda kv: kv[1]["j"])
    assert [v["j"] for _, v in fs] == list(range(man["chunks"])), "the manifest's chunks are not j = 0..n-1"
    return [(os.path.join(shots, man["split"], f), v["sha256"]) for f, v in fs]


def load_set(shots, man):
    sh = [testset.load(p, D, sha) for p, sha in chunks(shots, man)]
    det, obs = np.concatenate([s[0] for s in sh]), np.concatenate([s[1] for s in sh])
    assert deployed.shots_sha256(det, obs) == man["shots_sha256"], "the set's digest differs from the manifest"
    return det, obs


def _init():
    _M["plain"], _M["corr"] = circuits.matcher(D), circuits.matcher(D, True)


def _pm(job):
    det, obs = testset.load(job[0], D, job[1])
    return obs, circuits.pm_decode(_M["plain"], det), circuits.pm_decode(_M["corr"], det, correlated=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("base", "net"))
    ap.add_argument("--shots", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--net", default="")
    ap.add_argument("--build", default="build/d7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    assert man["set"] == "test" and man["d"] == D and man["shots"] == 1_000_000
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    common = dict(set=man["set"], manifest_list_sha256=man["list_sha256"], shots_sha256=man["shots_sha256"],
                  code_digest=provenance.code_digest(), host=provenance.host(), start=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if a.mode == "base":
        with mp.get_context("fork").Pool(a.procs, initializer=_init) as pool:
            parts = pool.map(_pm, chunks(a.shots, man))
        obs, pm, cpm = (np.concatenate([p[i] for p in parts]) for i in range(3))
        rec = dict(common, kind="base", shots=len(obs), observable_ones=int(obs.sum()), pm_failures=int((pm != obs).sum()),
                   corr_failures=int((cpm != obs).sum()), dem=dict(decompose_errors=True, approximate_disjoint_errors=True),
                   correlated="enable_correlations=True", circuit_sha256=circuits.CIRCUIT_SHA256[D])
        assert rec["observable_ones"] == man["observable_ones"], "the observables differ from the manifest's count"
        np.savez(os.path.join(a.out, "d9stest_base.npz"), obs=obs, pm=pm, cpm=cpm)
        tag = "base"
    else:
        from convlines import decoder
        det, obs = load_set(a.shots, man)
        c = np.asarray(deployed.Logits(D, a.build, a.net)(det), np.float32)
        import torch
        rec = dict(common, kind="net", network=a.net, network_sha256=decoder.weights_sha256(a.net),
                   logits="deployed", engine="timing.deployed.Logits: the FROZEN Triton engine, batch one",
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__, shots=len(obs),
                   observable_ones=int(obs.sum()), failures_alone=int(((c > 0).astype(np.uint8) != obs).sum()))
        np.savez(os.path.join(a.out, f"d9stest_{a.net}.npz"), c=c, obs=obs)
        tag = a.net
    rec.update(secs=time.time() - t0, node=platform.node())
    with open(os.path.join(a.out, f"d9stest_{tag}.json"), "x") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in rec if k not in ("host",)}), flush=True)


if __name__ == "__main__":
    main()
