"""SURF-v2's CPU tests: the beacon, the push time and the seeds, the sets, the session's networks, the q rule's edge cases
and its per-distance fallback on planted calibration reads, the analysis's sessions and CPU terms on planted records,
Holm, the paired statistics, and the frozen configuration against the release's records.

    python -m surf_v2.tests.test_surf_v2            (from the release root; prints every test and exits 1 on a failure)"""
import copy
import csv
import json
import os
import sys
import traceback

import numpy as np

from convlines import beacon, stats, testset
from surf_v2 import analysis, qrule

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = json.load(open(os.path.join(ROOT, "surf_v2", "config.json")))
BB = os.path.join(ROOT, "bb72", "ising_compare", "data", "bb", "stage3")


def test_beacon_record_and_round():
    bb = json.load(open(os.path.join(BB, "beacon.json")))          # BB72's stage-3 record: the same chain and rule
    try:
        beacon.check(bb)
    except AssertionError:
        pass
    else:
        raise AssertionError("a record with another study's tags passed")
    b = {k: v for k, v in bb.items() if k not in ("tags", "seed_rule")}
    beacon.check(b, push_time=b["freeze_commit_time"])
    assert beacon.beacon_round(b["freeze_commit_time"]) == b["round"] == 6532307
    assert beacon.round_time(b["round"]) == b["round_time"]
    assert beacon.round_time(beacon.beacon_round(1791399615)) >= 1791399615 + 600 > beacon.round_time(6532306)
    bad = copy.deepcopy(b)
    bad["signature"] = "00" + bad["signature"][2:]
    for x in bad["fetched"]:
        x["signature"] = bad["signature"]
    try:
        beacon.check(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("a changed signature passed")
    try:
        beacon.check(b, push_time=b["freeze_commit_time"] + 31)
    except AssertionError:
        pass
    else:
        raise AssertionError("a round not fixed by the push time passed")


def planted_activity(t, commit):
    """GitHub's activity list (newest first) for a freeze commit pushed at t, deleted and pushed again an hour later."""
    import datetime
    iso = lambda u: datetime.datetime.fromtimestamp(u, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    z = "0" * 40
    return [dict(id=103, before=z, after=commit, ref="refs/heads/surf-v2", timestamp=iso(t + 3600),
                 activity_type="branch_creation"),
            dict(id=102, before=commit, after=z, ref="refs/heads/surf-v2", timestamp=iso(t + 1800),
                 activity_type="branch_deletion"),
            dict(id=101, before=z, after=commit, ref="refs/heads/surf-v2", timestamp=iso(t), activity_type="branch_creation"),
            dict(id=100, before="a" * 40, after="b" * 40, ref="refs/heads/main", timestamp=iso(t - 60), activity_type="push")]


def test_push_time_is_the_earliest_activity():
    """T is the earliest activity record whose after is the freeze commit: a deletion and a later push of the same
    commit do not move it, and a record claiming the later push is refused."""
    bb = json.load(open(os.path.join(BB, "beacon.json")))
    t = bb["freeze_commit_time"]
    act = planted_activity(t, "f" * 40)
    pt, first = beacon.push_activity(act, "f" * 40)
    assert pt == t and first["id"] == 101
    assert beacon.activity_time("2026-10-13T17:02:11Z") == 1791910931
    b = {k: v for k, v in bb.items() if k not in ("tags", "seed_rule")}
    b.update(push_time=t, freeze_commit="f" * 40, activity=act, push_activity=first)
    beacon.check(b)
    late = dict(b, push_time=t + 3600, push_activity=act[0])
    for bad in (late, dict(b, activity=act[:2])):
        try:
            beacon.check(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("a push time other than the earliest activity record passed")


def test_seed_formula_reproduces_bb72():
    b = json.load(open(os.path.join(BB, "beacon.json")))
    rows = list(csv.DictReader(open(os.path.join(BB, "test_rows.csv"))))
    tags = {0.003: b"ising_compare bb test BB72-3\n", 0.002: b"ising_compare bb test BB72-2\n"}
    for r in rows:
        assert beacon.seed_tag(b["randomness"], tags[float(r["p"])], int(r["chunk"])) == int(r["seed"]), r["file"]
    assert len(rows) == 50
    s = {beacon.seed(b["randomness"], d, j) for d in beacon.DISTANCES for j in range(200 if d == 7 else 100)}
    assert len(s) == 700 and min(s) >= 2 ** 62, "SURF-v2 test seeds: distinct, above every registered namespace"
    assert beacon.TAGS[9] == b"convlines surf-v2 test d9\n" and len(set(beacon.TAGS.values())) == 6


def test_test_manifest_seeds_from_the_beacon():
    """testset.seeds (the analysis's check_beacon) on a planted sv2test manifest: BB72's round as a SURF-v2 record."""
    import hashlib
    import tempfile
    bb = json.load(open(os.path.join(BB, "beacon.json")))
    b = {k: v for k, v in bb.items() if k not in ("tags", "seed_rule")}
    act = planted_activity(b["freeze_commit_time"], "f" * 40)
    b.update(push_time=b["freeze_commit_time"], freeze_commit="f" * 40, push_source="test", activity=act,
             push_activity=act[2], tags={str(d): t.decode() for d, t in beacon.TAGS.items()}, seed_rule=beacon.SEED_RULE)
    with tempfile.TemporaryDirectory() as tmp:
        bp = os.path.join(tmp, "beacon.json")
        with open(bp, "w") as fh:
            json.dump(b, fh, indent=1)
        files = {testset.name("sv2test", d, j): dict(d=d, j=j, seed=beacon.seed(b["randomness"], d, j))
                 for d in testset.SV2_DISTANCES for j in range(testset.SV2["sv2test"]["chunks"][d])}
        man = dict(files=files, beacon=dict(record_sha256=hashlib.sha256(open(bp, "rb").read()).hexdigest()))
        keep = testset.manifest
        try:
            testset.manifest = lambda name_: man
            assert len(files) == 700 and testset.seeds("sv2test", bp)
            f0, f1 = testset.name("sv2test", 7, 0), testset.name("sv2test", 7, 1)
            files[f0] = dict(files[f0], seed=files[f0]["seed"] + 1)
            assert not testset.seeds("sv2test", bp), "a seed not derived from the beacon passed"
            files[f0] = dict(files[f0], seed=files[f0]["seed"] - 1)
            e0, e1 = files[f0], files[f1]
            files[f0], files[f1] = e1, e0
            assert not testset.seeds("sv2test", bp), "entries under each other's names passed"
            files[f0], files[f1] = e0, e1
            del files[f1]
            assert not testset.seeds("sv2test", bp), "an incomplete manifest passed"
            files[f1] = e1
            assert testset.seeds("sv2test", bp)
            man["beacon"]["record_sha256"] = "0" * 64
            try:
                testset.seeds("sv2test", bp)
            except AssertionError:
                pass
            else:
                raise AssertionError("a beacon record the manifest does not name passed")
        finally:
            testset.manifest = keep


def test_sets():
    for d in testset.SV2_DISTANCES:
        assert testset.sv2_seed("sv2cal", d, 0) == 81_000_000 + 10_000 * d
        assert testset.sv2_seed("sv2cal", d, 19) == 81_000_000 + 10_000 * d + 19
        assert testset.sv2_seed("sv2dry", d, 3) == 89_000_000 + 10_000 * d + 3
        for s, n in (("sv2cal", 20), ("sv2dry", 4), ("sv2test", 200 if d == 7 else 100)):
            assert testset.SV2[s]["chunks"][d] == n
        try:
            testset.sv2_seed("sv2cal", d, 20)
        except AssertionError:
            pass
        else:
            raise AssertionError("chunk 20 of sv2cal")
    assert testset.name("sv2test", 17, 5) == "d17_p0.006_X_sv2test_005.npz"
    assert {k: testset.parse_range(v)[-1] + 1 for k, v in CFG["test"]["chunks"].items()} == \
        {str(d): testset.SV2["sv2test"]["chunks"][d] for d in testset.SV2_DISTANCES}


def test_session_leaves_out_the_exceptions():
    """surf_v2.session's networks: SKIP before the freeze, the freeze record's exceptions after it, never a network that
    is not new, and SKIP disagreeing with the freeze refused."""
    import tempfile
    from surf_v2 import session
    keep = os.environ.pop("SKIP", None)
    try:
        none = os.path.join(ROOT, "surf_v2", "no_such_freeze.json")
        full = session.networks(none)
        assert len(full) == 12 and full[:2] == [(7, "d7_b1w64_sv2"), (7, "d7_b1w64_r740")]
        rest = [x for x in full if x[1] != "d13_b5_sv2"]
        os.environ["SKIP"] = "d13_b5_sv2"
        assert session.networks(none) == rest
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "freeze.json")
            with open(f, "w") as fh:
                json.dump(dict(exceptions={"13": dict(row_network="d13_b5_r775")}), fh)
            assert session.networks(f) == rest
            del os.environ["SKIP"]
            assert session.networks(f) == rest, "the freeze's exceptions without SKIP"
            for skip, fz in (("d15_b5_sv2", f), ("d13_b5_r775", none)):
                os.environ["SKIP"] = skip
                try:
                    session.networks(fz)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(f"SKIP={skip} passed")
    finally:
        os.environ.pop("SKIP", None)
        if keep is not None:
            os.environ["SKIP"] = keep


def planted(n=1000, ties=0):
    """|c| = 1, 2, ..., n in shot order (the first `ties` shots all at 1), signs positive; obs 1 (the network right)."""
    az = np.arange(1, n + 1, dtype=np.float32)
    az[:ties] = 1.0
    return az.copy(), np.ones(n, np.uint8), np.ones(n, np.uint8)       # c, corr's answers, obs


FR = {"kind": "frontier", "net_us": 10.0, "router_us": 1.1}
GRID = [0.0, 0.01, 0.02, 0.04]


def test_q0_chosen():
    c, corr, obs = planted()
    c[[100, 200, 300, 400, 500]] *= -1                                   # five network failures, far from tau
    p = qrule.fit_rule(c, corr, obs, GRID, 20, 100.0, FR, cost=lambda q, r: 50.0)
    ch = p["chosen"]
    assert ch["q"] == 0 and ch["tau"] == 0.0 and ch["routed"] == 0 and ch["failures"] == 5
    assert ch["projected_us"] == 10.0, "projected(0) = net_us, no router cost"
    assert p["qualifies"] and p["outcome"] == "q = 0: the network alone"
    assert p["table"][1]["projected_us"] == 10.0 + 1.1 + 0.01 * 50.0


def test_none_qualifies_argmin_tie_smaller_q():
    c, corr, obs = planted()
    c[list(range(0, 10)) + list(range(20, 25)) + list(range(100, 135))] *= -1    # network: 10 + 5 + 35 failures
    corr[list(range(25, 35))] = 0                                                 # correlated PM: 10 among |c| 26-35
    p = qrule.fit_rule(c, corr, obs, GRID, 40, 100.0, FR, cost=lambda q, r: 1.0)
    f = {x["q"]: x["failures"] for x in p["table"]}
    assert f == {0.0: 50, 0.01: 40, 0.02: 40, 0.04: 45}, f
    assert [x["routed"] for x in p["table"]] == [0, 10, 20, 40]
    assert not p["qualifies"] and p["chosen"]["q"] == 0.01, "the fewest failures, ties to the smaller q"


def test_fallback_per_distance():
    """The registered fallback at each d (config qrule.fallback): a planted calibration read of 200,000 shots that meets
    the limit at no admissible q. d = 7 (its policy was fitted before G7): the largest admissible q, 8 %; d = 9 (argmin,
    README 16.1): the fewest calibration failures, the tie to the smaller q."""
    assert CFG["qrule"]["fallback"] == {"7": "largest", **{k: "argmin" for k in ("9", "11", "13", "15", "17")}}
    n = 200000
    c = np.arange(1, n + 1, dtype=np.float32)                   # |c| = 1 ... n, in shot order
    corr, obs = np.ones(n, np.uint8), np.ones(n, np.uint8)
    c[100000:104000] *= -1                                       # 4,000 network failures that no q <= 16 % routes
    corr[3000:3100] = 0                                          # correlated PM wrong on 100 shots routed from q = 2 %
    s7 = qrule.settings(CFG, 7)
    p = qrule.fit_rule(c, corr, obs, **s7)
    f = {x["q"]: x["failures"] for x in p["table"]}
    assert f[0.0] == f[0.01] == 4000 and all(f[q] == 4100 for q in s7["grid"] if q >= 0.02), f
    assert [x["q"] for x in p["table"] if x["admissible"]] == [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
    assert not p["qualifies"] and p["fallback"] == "largest" and p["chosen"]["q"] == 0.08 and p["chosen"]["routed"] == 16000
    assert qrule.fit_rule(c, corr, obs, **dict(s7, fallback="argmin"))["chosen"]["q"] == 0.0, "argmin differs here"
    p9 = qrule.fit_rule(c, corr, obs, cost=lambda q, r: 1.0, **qrule.settings(CFG, 9))
    assert not p9["qualifies"] and p9["fallback"] == "argmin" and p9["chosen"]["q"] == 0.0
    assert qrule.policy_fallback(dict(rule="C4.2: the smallest admissible q with F_cal(q) <= M - 2 sqrt(2 M), else the "
                                           "largest admissible q; no admissible q: no policy")) == "largest"
    assert qrule.policy_fallback(dict(rule="C4.2 (16.1)", fallback="argmin")) == "argmin"


def test_ties_realized_fraction():
    c, corr, obs = planted(ties=30)
    p = qrule.fit_rule(c, corr, obs, GRID, 2500, 100.0, FR, cost=lambda q, r: 7.0)
    x = {r["q"]: r for r in p["table"]}
    assert x[0.01]["tau"] == 1.0 and x[0.01]["routed"] == 0 and x[0.01]["routed_fraction"] == 0.0
    assert x[0.01]["cost_us"] == 0.0 and x[0.01]["projected_us"] == 10.0 + 1.1, "r_q = 0: no routed cost"
    assert x[0.02]["routed"] == 0 and x[0.04]["routed"] == 40 and x[0.04]["tau"] > 40.0
    assert abs(x[0.04]["projected_us"] - (11.1 + 0.04 * 7.0)) < 1e-12, "the projection uses r_q"
    c2, corr2, obs2 = planted()
    c2[:5], c2[5:35] = 0.5, 1.0                                          # 30 ties at the 1 % quantile
    y = {r["q"]: r for r in qrule.fit_rule(c2, corr2, obs2, GRID, 2500, 100.0, FR, cost=lambda q, r: 7.0)["table"]}
    assert y[0.01]["tau"] == 1.0 and y[0.01]["routed"] == 5 < 10, "r_q = 0.5 % < q = 1 %"
    assert abs(y[0.01]["projected_us"] - (11.1 + 0.005 * 7.0)) < 1e-12


def test_qualifying_q_over_the_bar_is_skipped():
    c, corr, obs = planted()
    c[:40] *= -1                                                         # 40 failures, all among the least confident
    path = {"kind": "path", "path_us": {"0.0": 10.0, "0.01": 50.0, "0.02": 15.0, "0.04": None}}
    p = qrule.fit_rule(c, corr, obs, GRID, 60, 20.0, path)
    x = {r["q"]: r for r in p["table"]}
    assert x[0.01]["failures"] <= p["limit"] and not x[0.01]["admissible"], "q = 1 % qualifies but is over the bar"
    assert not x[0.04]["admissible"], "an unmeasured path is not admissible"
    assert p["chosen"]["q"] == 0.02 and p["qualifies"]


def test_no_admissible_q():
    c, corr, obs = planted()
    path = {"kind": "path", "path_us": {"0.0": 30.0, "0.01": 31.0, "0.02": 32.0, "0.04": None}}
    p = qrule.fit_rule(c, corr, obs, GRID, 2500, 20.0, path)
    assert p["chosen"] is None and not p["eligible"] and p["outcome"].startswith("no admissible q")
    q = qrule.fit_rule(c, corr, obs, GRID, 2500, 9.0, FR, cost=lambda q, r: 1.0)
    assert q["chosen"] is None, "net_us over the bar: nothing admissible"


def test_holm_and_pair():
    adj, rej = analysis.holm([0.01, 0.04, 0.03, 0.005], 0.05)
    assert [round(x, 6) for x in adj] == [0.03, 0.06, 0.06, 0.02] and rej == [True, False, False, True]
    adj, rej = analysis.holm([0.2, 0.2], 0.05)
    assert adj == [0.4, 0.4] and rej == [False, False]
    c = np.array([-3, -0.5, 0.2, 2, 0.7], np.float32)
    p, r = analysis.pair(c, np.array([1, 1, 0, 0, 1], np.uint8), 0.7)
    assert r.tolist() == [False, True, True, False, False] and p.tolist() == [0, 1, 0, 1, 1]
    assert analysis.tau_q(np.arange(1, 1001, dtype=np.float32), 0.01) == float(np.quantile(np.arange(1, 1001,
                                                                                                    dtype=np.float32), 0.01))


def plant_session(tmp, pod, when, uuid, timed):
    """A planted session directory tmp/pod/session of one network, n9: both GPU checks passed at time when on GPU uuid,
    and (timed) five GPU-term processes and a handoff, all accepted."""
    sd = os.path.join(tmp, pod, "session")
    os.makedirs(sd)
    for w in ("start", "end"):
        with open(os.path.join(sd, f"gpucheck_{w}.json"), "w") as fh:
            json.dump(dict(passed=True, time=when, clocks=dict(gpu_uuid=uuid)), fh)
    if timed:
        rows = []
        for k in range(6):
            f, kind = (f"n9_p{k}.json", "gpu") if k < 5 else (f"n9_handoff_p{k}.json", "handoff")
            with open(os.path.join(sd, f), "w") as fh:
                json.dump(dict(accepted=True, network_sha256="s", code_digest="g", per_shot_mean=30.0 + k, tau=1.0,
                               handoff_us=1.1, handoff_estimate_us=0.7, routed=0, limit=None, shots=200000,
                               gpu="NVIDIA H100 80GB HBM3"), fh)
            rows.append(dict(kind=kind, proc=k, file=f, exit=0, accepted=True))
        with open(os.path.join(sd, "attempts.json"), "w") as fh:
            json.dump({"n9": rows}, fh)
    return sd


def test_sessions_lost_and_ordered():
    """analysis.session and analysis.started on planted session directories: one without attempts.json is lost; the
    order is by the start checks' times (time zones included), not by the directories' names."""
    import tempfile
    fz = dict(networks={"n9": dict(d=9, sha256="s", tau=1.0)}, read_code_digest="g")
    with tempfile.TemporaryDirectory() as tmp:
        lost = plant_session(tmp, "pod2", "2026-10-13T10:00:00+0000", "GPU-a", False)
        later = plant_session(tmp, "pod", "2026-10-13T05:00:00-0700", "GPU-b", True)          # 12:00 UTC
        x = sorted((analysis.session(s, CFG, fz, ["n9"], True) for s in (later, lost)), key=analysis.started)
        assert [s["source"] for s in x] == ["pod2", "pod"] and [s["gpu_uuid"] for s in x] == ["GPU-a", "GPU-b"]
        assert x[0]["lost"] and not x[0]["qualified"] and not x[0]["networks"]
        assert x[1]["qualified"] and not x[1]["lost"] and x[1]["networks"]["n9"]["a_us"] == 34.0
        assert analysis.session(later, CFG, fz, ["n9"], False)["qualified"], "a registered session, not dry"
        f = os.path.join(later, "n9_p2.json")
        good = json.load(open(f))
        for bad in (dict(limit=2000), dict(shots=2000), dict(gpu="NVIDIA H200")):
            with open(f, "w") as fh:
                json.dump(dict(good, **bad), fh)
            try:
                analysis.session(later, CFG, fz, ["n9"], False)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"a timing record with {bad} passed")


def test_session_anomalies_count_none():
    """analysis.sessions on planted session directories (the second review's finding 4): two sessions on one GPU, or
    more than two session directories, leave no session counted and give the reason, which the analysis reports as
    family B's "not established (<reason>)" after writing families A and C; with two sessions on two GPUs the first
    that qualified counts, and a path that does not exist is no session."""
    import tempfile
    fz = dict(networks={"n9": dict(d=9, sha256="s", tau=1.0)}, read_code_digest="g")
    with tempfile.TemporaryDirectory() as tmp:
        a = plant_session(tmp, "pod", "2026-10-13T05:00:00-0700", "GPU-a", True)              # 12:00 UTC
        b = plant_session(tmp, "pod2", "2026-10-13T13:00:00+0000", "GPU-b", True)
        c = plant_session(tmp, "pod3", "2026-10-13T14:00:00+0000", "GPU-a", True)
        din = {}
        sess, count, why = analysis.sessions([b, a, os.path.join(tmp, "pod4", "session")], CFG, fz, ["n9"], True, din)
        assert [s["source"] for s in sess] == ["pod", "pod2"] and count is sess[0] and why is None
        assert sess[1]["problems"][-1].startswith("not counted") and len(din) == 2 * 9, sorted(din)
        sess, count, why = analysis.sessions([c, a], CFG, fz, ["n9"], True, {})
        assert count is None and why == "two timing sessions on one GPU, GPU-a", why
        assert [(s["label"], s["source"]) for s in sess] == [("session 1", "pod"), ("session 2", "pod3")]
        sess, count, why = analysis.sessions([a, b, c], CFG, fz, ["n9"], True, {})
        assert count is None and why == ("3 timing session directories, more than two; "
                                         "two timing sessions on one GPU, GPU-a"), why
        assert [s["label"] for s in sess] == ["session 1", "session 2", "session 3"]
        assert all(s["qualified"] for s in sess)


def test_cpu_term_one_job():
    """analysis.cpu on planted records: one job of the frozen read path, the deployed engine's mask, and at d = 9-17
    the session's handoff routing as many validation shots; a ref from another job or another mask is refused, while
    another routed count, a missing repetition or no CPU term at all leave the network without a term and with a
    problem (family B not established), and refuse nothing."""
    import tempfile
    val = json.load(open(os.path.join(ROOT, "routing", "manifest.json")))["val"]["9"]["shots_sha256"]
    fz = dict(networks={"n9": dict(d=9, tau=1.0)}, read_code_digest="g")
    with tempfile.TemporaryDirectory() as tmp:
        def put(f, **kw):
            with open(os.path.join(tmp, f), "w") as fh:
                json.dump(dict(slurm_job="7", code_digest="g", **kw), fh)
        for r in (1, 2, 3):
            put(f"term_n9_rep{r}.json", network="n9", tau=1.0, term_us=0.1 * r, routed=5, shots_sha256=val,
                mask_kind="deployed")
            put(f"ref_d9_rep{r}.json", d=9, per_block_us=80.0 + r)
        x, problems = analysis.cpu(tmp, CFG, fz, ["n9"], False, {"n9": 5})
        assert not problems and x["n9"]["b_node_us"] == 0.30000000000000004 and x["n9"]["ref_median_us"] == 82.0
        for bad in ("job", "mask"):
            put("ref_d9_rep3.json", d=9, per_block_us=83.0)
            put("term_n9_rep3.json", network="n9", tau=1.0, term_us=0.3, routed=5, shots_sha256=val, mask_kind="deployed")
            if bad == "job":
                with open(os.path.join(tmp, "ref_d9_rep3.json"), "w") as fh:
                    json.dump(dict(slurm_job="8", code_digest="g", d=9, per_block_us=83.0), fh)
            elif bad == "mask":
                put("term_n9_rep3.json", network="n9", tau=1.0, term_us=0.3, routed=5, shots_sha256=val, mask_kind="fp32")
            try:
                analysis.cpu(tmp, CFG, fz, ["n9"], False, {"n9": 5})
            except AssertionError:
                pass
            else:
                raise AssertionError(f"another {bad} passed")
        put("ref_d9_rep3.json", d=9, per_block_us=83.0)
        put("term_n9_rep3.json", network="n9", tau=1.0, term_us=0.3, routed=5, shots_sha256=val, mask_kind="deployed")
        x, problems = analysis.cpu(tmp, CFG, fz, ["n9"], False, {"n9": 6})
        assert x == {} and len(problems) == 1 and "the counted session's handoff 6" in problems[0], problems
        os.remove(os.path.join(tmp, "term_n9_rep3.json"))
        x, problems = analysis.cpu(tmp, CFG, fz, ["n9"], False, {"n9": 5})
        assert x == {} and problems == ["n9: 2 term and 3 ref repetitions"], problems
    with tempfile.TemporaryDirectory() as tmp:
        x, problems = analysis.cpu(os.path.join(tmp, "no_cpu_term"), CFG, fz, ["n9"], False, {})
        assert x == {} and problems == ["n9: 0 term and 0 ref repetitions"], problems


def test_paired_statistics():
    fa = np.zeros(100000, bool)
    fb = np.zeros(100000, bool)
    fa[:900] = True
    fb[:1000] = True
    fb[2000:2100] = True
    s = stats.paired(fa, fb, 0.025)
    assert (s["only_a"], s["only_b"], s["verdict"]) == (0, 200, "better") and s["diff_hi"] < 0
    s = stats.paired(fb, fa, 0.05)
    assert s["verdict"] == "worse" and s["mcnemar_p"] < 1e-50


def test_config_against_the_release():
    sys.path.insert(0, ROOT)
    from timing import composite
    sp = json.load(open(os.path.join(ROOT, "results", "controls_speed.json")))
    keys = {v: k for k, v in CFG["baselines"].items()}
    for d, *_, rec, opp, plan, num, allow, meas in composite.rows():
        t = CFG["registered_times_us"][str(d)]
        for n, us in opp.items():
            assert t[keys[n]] == us, (d, n)
        if d != 17:
            for m in ("M1", "M4"):
                assert t[f"ising_{m.lower()}_corr"] == sp["d"][str(d)][m]["grace_us"]
        assert len(t) == 8
        if d != 17:
            assert CFG["registered_composites_us"][str(d)]["composite"] == rec
    r17 = json.load(open(os.path.join(ROOT, "results", "r17_summary.json")))["ours_speed"]
    assert CFG["registered_composites_us"]["17"]["composite"] == r17["total"]
    q = CFG["qrule"]
    assert q["M"] == {"7": 3401, "9": 2500, "11": 1750, "13": 1200, "15": 802, "17": 554}
    assert q["bar_us"] == {"7": 15.16, "9": 33.99, "11": 65.32, "13": 86.66, "15": 205.53, "17": 322.23}
    ext = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
    assert all(q["grid"][k] == ext for k in ("7", "9", "11", "13", "15")) and q["grid"]["17"] == ext + [
        0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
    assert [q["projection"][k]["net_us"] for k in ("9", "11", "13", "15", "17")] == [
        31.44, 52.31, 52.31, 55.77154946088791, 73.54281725168228]
    assert all(q["projection"][k]["router_us"] == 1.1 for k in ("9", "11", "13", "15", "17"))
    pu = q["projection"]["7"]["path_us"]
    assert [pu[k] for k in ("0.0", "0.01", "0.02", "0.04", "0.06", "0.08", "0.1")] == [
        10.5, 11.62, 11.62, 12.72, 13.71, 14.79, 15.72] and pu["0.12"] is None
    assert [k for k in ext if pu[str(k)] is not None and pu[str(k)] <= q["bar_us"]["7"]] == ext[:6], "q = 0-8 % at d = 7"
    assert [CFG["registered_choices"][k]["q"] for k in ("7", "9", "11", "13", "15", "17")] == [0.04, 0.01, 0.01, 0.02,
                                                                                               0.06, 0.1]
    from convlines import decoder
    for k, n in CFG["networks"].items():
        for name in n.values():
            assert decoder.config(name)["d"] == int(k), name
    assert decoder.config("d17_b5_r787_q08")["policy"]["tau"] == 4.539173126220703
    assert decoder.config("d17_b5_r787_q08")["tables"] == decoder.config("d17_b5_r787")["tables"]
    from convlines.evaluate import KEYS
    assert sorted(KEYS.values()) == sorted(CFG["baselines"]) == sorted(analysis.NV)


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
