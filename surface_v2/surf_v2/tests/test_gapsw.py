"""Gap switching's CPU tests: the q rule on the gap reproduces surf_v2/qrule.py's fit_rule, the complementary gap agrees
with the exhaustive search on a small circuit, moving the observable's cut into the bulk leaves every gap unchanged, and
a graph whose detector-detector cycle flips the observable is refused, and the timing directory is one completed job's
from an allowed deployment (surf_v2.gapsw_analysis job).

    python -m surf_v2.tests.test_gapsw            (from the release root; prints every test and exits 1 on a failure)"""
import traceback

import numpy as np

from surf_v2 import gapsw, qrule
from surf_v2.gapsw_validate import Exhaustive


def small_circuit(d=3):
    import stim
    return stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d, after_clifford_depolarization=0.006,
                                  before_round_data_depolarization=0.006, before_measure_flip_probability=0.004,
                                  after_reset_flip_probability=0.004)


def shots(c, n, seed):
    det, obs = c.compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    return det.astype(np.uint8), obs[:, 0].astype(np.uint8)


def test_fit_gap_mirrors_fit_rule():
    rng = np.random.default_rng(5)
    n = 20_000
    obs = rng.integers(0, 2, n).astype(np.uint8)
    c = (rng.normal(3, 2, n).astype(np.float32) * np.where(obs == 1, 1, -1)).astype(np.float32)
    corr = np.where(rng.random(n) < 0.02, 1 - obs, obs).astype(np.uint8)
    grid = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    for M, bar in ((900, 40.0), (900, 13.0), (200, 40.0), (900, 5.0)):
        cost = lambda q, r: 30.0 + 100 * q                               # noqa: E731
        ref = qrule.fit_rule(c, corr, obs, grid, M, bar, dict(kind="frontier", net_us=10.0, router_us=0.0), cost=cost)
        got = gapsw.fit_gap(np.abs(c), (c > 0).astype(np.uint8), corr, obs, grid, M, bar, 10.0, 10.0, cost)
        keys = ("q", "tau", "routed", "failures", "cost_us", "projected_us", "admissible", "routed_fraction")
        assert [{k: x[k] for k in keys} for x in ref["table"]] == [{k: x[k] for k in keys} for x in got["table"]]
        pick = lambda x: x and {k: x[k] for k in keys}                    # noqa: E731
        assert pick(ref["chosen"]) == pick(got["registered"]["chosen"]) and ref["qualifies"] == got["registered"]["qualifies"]
        assert ref["limit"] == got["limit"] and ref["network_failures"] == got["plain_failures"]


def test_gap_against_exhaustive_search():
    c = small_circuit(3)
    det, obs = shots(c, 400, 11)
    fe = gapsw.FrontEnd(circuit=c)
    dec, w, gap = fe.batch(det)
    ex = Exhaustive(fe.m)
    for i in range(len(det)):
        b = ex.classes(det[i], 14)
        assert b is not None
        assert abs(w[i] - min(b)) < 1e-3 and abs(gap[i] - (b[1 - dec[i]] - b[dec[i]])) < 1e-3, (i, w[i], gap[i], b)
        assert abs(b[0] - b[1]) < 1e-3 or int(np.argmin(b)) == dec[i]


def test_moved_cut_leaves_the_gap():
    import pymatching
    c = small_circuit(5)
    det, obs = shots(c, 2000, 12)
    fe = gapsw.FrontEnd(circuit=c)
    dec, w, gap = fe.batch(det)
    rng = np.random.default_rng(3)
    Q = rng.random(fe.n) < 0.5                                           # any detector set moves the cut
    m = pymatching.Matching()
    for u, v, a in fe.m.edges():
        f = (int(0 in a["fault_ids"]) + int(Q[u]) + (int(Q[v]) if v is not None else 0)) % 2
        kw = dict(fault_ids={0} if f else set(), weight=a["weight"], error_probability=a["error_probability"])
        (m.add_boundary_edge(u, **kw) if v is None else m.add_edge(u, v, **kw))
    fe2 = gapsw.FrontEnd(matcher=m)
    assert len(fe2.S) > 0, "the moved cut needs a colouring"
    dec2, w2, gap2 = fe2.batch(det)
    shift = (det[:, Q].sum(1) % 2).astype(np.uint8)
    assert np.array_equal(w2, w) and np.array_equal(gap2, gap)
    assert np.array_equal((dec2 ^ shift)[gap > 0], dec[gap > 0])
    ex = Exhaustive(m)
    for i in range(0, len(det), 10):
        b = ex.classes(det[i], 14)
        if b is not None:
            assert abs(gap2[i] - (b[1 - dec2[i]] - b[dec2[i]])) < 1e-3


def test_odd_detector_cycle_refused():
    import pymatching
    m = pymatching.Matching()
    m.add_edge(0, 1, fault_ids=set(), weight=1.0)
    m.add_edge(1, 2, fault_ids={0}, weight=1.0)
    m.add_edge(0, 2, fault_ids=set(), weight=1.0)
    m.add_boundary_edge(0, fault_ids={0}, weight=1.0)
    try:
        gapsw.FrontEnd(matcher=m)
    except ValueError:
        return
    raise AssertionError("a cycle that flips the observable was accepted")


def test_timing_job_record():
    """gapsw_analysis.job on planted timing directories: write_job's record passes; another deployed commit, no job.json,
    a file added or changed after the job, a ref of another job, a front-end term of another deployment and (outside dry
    runs) a record of another read path are refused."""
    import json
    import os
    import tempfile
    from convlines import provenance
    from surf_v2 import gapsw_analysis as ga
    own, dep, dig = gapsw.gapsw_sha256(), provenance.commit(), provenance.code_digest()
    fz = dict(read_code_digest=dig)
    keep = os.environ.get("SLURM_JOB_ID")
    os.environ["SLURM_JOB_ID"] = "7"

    def make(tmp, ref_job="7", fe_commit=dep, digest=dig, record=True):
        os.makedirs(os.path.join(tmp, "masks"))
        for f, r in (("ref_d7_rep1.json", dict(slurm_job=ref_job)), (f"term_{gapsw.KEY}_d7_rep1.json", dict(slurm_job="7")),
                     (f"fe_{gapsw.KEY}_d7_rep1.json", dict(release_commit=fe_commit, gapsw_sha256=own))):
            with open(os.path.join(tmp, f), "w") as fh:
                json.dump(dict(r, code_digest=digest), fh)
        np.savez(os.path.join(tmp, "masks", f"val_{gapsw.KEY}_d7.npz"), routed=np.zeros(3, np.uint8))
        if record:
            ga.write_job(tmp)
        return tmp

    def refused(tmp, why, allowed=None):
        try:
            ga.job(tmp, allowed or {dep}, fz, own, False, {})
        except AssertionError:
            return
        raise AssertionError(f"{why} passed")
    try:
        with tempfile.TemporaryDirectory() as root:
            good = make(os.path.join(root, "good"))
            assert dep and ga.job(good, {dep}, fz, own, False, {}) == dict(slurm_job="7", deployed=dep)
            refused(good, "another deployed commit", {"0" * 40})
            extra = os.path.join(good, "ref_d7_rep2.json")
            with open(extra, "w") as fh:
                json.dump(dict(slurm_job="7", code_digest=dig), fh)
            refused(good, "a file added after the job")
            os.remove(extra)
            f = os.path.join(good, f"term_{gapsw.KEY}_d7_rep1.json")
            raw = open(f).read()
            with open(f, "w") as fh:
                fh.write(raw + " ")
            refused(good, "a file changed after the job")
            with open(f, "w") as fh:
                fh.write(raw)
            assert ga.job(good, {dep}, fz, own, False, {})["slurm_job"] == "7"
            refused(make(os.path.join(root, "norecord"), record=False), "a directory without job.json")
            refused(make(os.path.join(root, "job8"), ref_job="8"), "a ref of another job")
            refused(make(os.path.join(root, "fe"), fe_commit="0" * 40), "a front-end term of another deployment")
            other = make(os.path.join(root, "digest"), digest="0" * 64)
            refused(other, "a record of another read path")
            assert ga.job(other, {dep}, fz, own, True, {})["deployed"] == dep, "a dry run takes any read path"
    finally:
        if keep is None:
            os.environ.pop("SLURM_JOB_ID", None)
        else:
            os.environ["SLURM_JOB_ID"] = keep


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    bad = 0
    for k, f in tests:
        try:
            f()
            print(f"PASS {k}", flush=True)
        except Exception:
            bad += 1
            print(f"FAIL {k}\n{traceback.format_exc()}", flush=True)
    print(f"{len(tests) - bad} of {len(tests)} passed", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
