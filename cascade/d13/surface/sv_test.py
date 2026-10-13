"""Unit checks of the evaluation code (CPU): the statistics, the labels of item 2, the routing rule, the decision
assembly, the queue's batches and the output loader, and the stage gate's findings of 2026-10-12 that a CPU can check:
publication once under concurrent writers (7), the screen summary's refusals (4), the code identity and the analysis
refusal (3), the committed lock and gate (2), the read journal's validation for a resume (6) and the gate's rules (5),
and from the recheck of 960b82f two successive resumes of a journal cut short (6) and no resume of a read whose
kernel caches changed (2). One JSON line per check, then ALL PASS (exit 0) or FAILED (exit 1, finding 1).
    python -m surface.sv_test
    python -m surface.sv_test compare DIR_A DIR_B   (a resumed read's p<p> directory against an uninterrupted one)
    python -m surface.sv_test report DRY_DIR OUT    (finding 8 of the recheck: the report JSON names the lock's sha)"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.stats import binom, binomtest, norm

from surface import sv_common as sc
from surface import sv_read as sv
from surface import sv_route as sr
from surface import sv_speed as ss

RESULTS = []


def check(name, ok, **info):
    RESULTS.append(bool(ok))
    print(json.dumps({"check": name, "pass": bool(ok), **info}), flush=True)


def refused(fn, *args, **kw):
    """True if fn refuses (SystemExit) -> the message is kept for the record."""
    try:
        fn(*args, **kw)
    except SystemExit as e:
        refused.last = str(e)[:200]
        return True
    return False


refused.last = ""


def test_stats():
    rng = np.random.default_rng(1)
    ok = True
    for _ in range(200):
        a, b, c = (int(x) for x in rng.integers(0, 3000, 3))
        d = int(rng.integers(10 ** 5, 10 ** 7))
        z = float(rng.uniform(0.5, 4))
        hi = sc.newcombe_hi(np.array([a], float), np.array([b], float), np.array([c], float), np.array([d], float), z)
        ok &= abs(hi[0] - sc.newcombe10(a, b, c, d, z)[2]) < 1e-12
    check("newcombe_vectorised_equals_scalar", ok)
    ok, n_rej = True, 0
    for _ in range(300):
        a, b, c = int(rng.integers(500, 3000)), int(rng.integers(50, 400)), int(rng.integers(50, 400))
        d = int(rng.integers(10 ** 6, 4 * 10 ** 7))
        delta = 0.1 * (a + c) / (a + b + c + d)
        pv = sc.ni_p(a, b, c, d, delta)
        for al in (0.05, 0.05 / 40, 1e-4, 1e-6):
            byint = sc.newcombe10(a, b, c, d, float(norm.ppf(1 - al)))[2] < delta
            ok &= (pv < al) == byint
            n_rej += byint
    check("noninferiority_p_matches_interval", ok, rejections=int(n_rej))
    ok = sc.mcnemar_p(0, 0) == 1.0 and all(
        abs(sc.mcnemar_p(b, c) - binomtest(b, b + c, 0.5).pvalue) < 1e-15 and
        abs(sc.mcnemar_p(b, c) - min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))) < 1e-12
        for b, c in ((3, 9), (120, 80), (400, 401), (1, 30)))
    check("mcnemar_exact_two_sided", ok)
    lv = sc.holm([0.01, 0.04, 0.03, 0.005], 0.05)
    check("holm_step_down", lv == {3: 0.0125, 0: 0.05 / 3, 2: None, 1: None}, levels={str(k): v for k, v in lv.items()})
    lv = sc.holm([0.001, 0.0125, 0.02], 0.05)
    check("holm_all_rejected", lv == {0: 0.05 / 3, 1: 0.025, 2: 0.05})
    pl = sc.p_block_to_pl(0.0051)
    check("eq1_k1_R13", abs(pl - (1 - (1 - 2 * 0.0051) ** (1 / 13)) / 2) < 1e-15, pl=pl)
    cnt = {"both": 1200, "only_x": 300, "only_y": 280, "shots": 1_000_000}
    p1, p2 = sc.ni_power(cnt, 2_000_000, 6e-5, 0.05 / 40), sc.ni_power(cnt, 2_000_000, 6e-5, 0.05 / 40)
    p3 = sc.ni_power(cnt, 8_000_000, 6e-5, 0.05 / 40)
    check("ni_power_deterministic_and_rising", p1 == p2 and p3 > p1, p_2e6=p1, p_8e6=p3)
    m1 = sc.mcnemar_power({"both": 100, "only_x": 50, "only_y": 100, "shots": 10 ** 6}, 2 * 10 ** 6, 0.05 / 40)
    check("mcnemar_power", 0.9 < m1 <= 1.0, power=m1)


def test_labels():
    L = sr.label
    ok = (L(True, 30, 10, True) == "worse" and L(True, 10, 30, False) == "better" and L(False, 10, 12, True) ==
          "non-inferior" and L(False, 10, 12, False, True) == "inconclusive" and L(False, 10, 12, False) == "level"
          and L(True, 30, 10, None) == "worse" and L(True, 30, 10, False, True) == "worse")
    check("labels_item2", ok)


def test_routing_and_assembly():
    tau = sc.TAU_C4
    c = np.array([tau, np.nextafter(np.float32(tau), np.float32(0)), -1.0, 3.0, np.nan], np.float32)
    r = sc.routed(c, tau)
    exact = float(np.float32(tau)) < tau
    check("routing_float32_in_float64", list(r) == [exact, True, True, False, False], routed=r.tolist(),
          f32_tau_below=exact)
    n = 8
    obs = np.zeros(n, np.uint8)
    S = {"c": np.array([0.1, -0.2, 2.0, -3.0, 0.5, 5.0, -0.05, 1.0], np.float32)}
    S["dec"] = (S["c"] > 0).astype(np.uint8)
    Bz = np.array([1.0, -0.01, 3.0, -4.0, 0.02, 2.0, -0.5, 0.3], np.float32)
    B = {"z": Bz, "dec": (Bz > 0).astype(np.uint8)}
    M = {"pred": np.array([1, 1, 1, 1, 0, 0, 0, 0], np.uint8)}
    q = {"z": np.where(np.abs(S["c"]) < 0.6, Bz, np.nan).astype(np.float32)}
    q["dec"] = np.where(np.isnan(q["z"]), 255, q["z"] > 0).astype(np.uint8)
    o = {"S1": S, "SB": S, "B1": B, "Bb": B, "BQ1": q, "BQ5": q, "MS": M, "MB": M}
    ch = {"C1": {"tau1": 0.6, "tau2": 0.1}, "C5": {"tau1": 0.6}}
    d1, s1 = sc.assemble("C1", "single", o, ch)
    exp_s1 = np.array([1, 2, 0, 0, 2, 0, 1, 0])
    exp_d1 = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    d1b, s1b = sc.assemble("C1", "batched", o, ch)
    d5, s5 = sc.assemble("C5", "single", o, ch)
    d4, s4 = sc.assemble("C4", "single", o, ch)
    d2, _ = sc.assemble("C2", "batched", o, ch)
    d3, _ = sc.assemble("C3", "single", o, ch)
    ok = (list(s1) == list(exp_s1) and list(d1) == list(exp_d1) and list(d1b) == list(d1) and list(s1b) == list(s1)
          and list(s5) == [1, 1, 0, 0, 1, 0, 1, 0] and list(d5) == [1, 0, 1, 0, 1, 1, 0, 1]
          and list(s4) == [2, 2, 0, 0, 2, 0, 2, 2] and list(d2) == list(B["dec"]) and list(d3) == list(M["pred"]))
    check("assemble_configs_modes", ok, C1=d1.tolist(), stage=s1.tolist(), C5=d5.tolist(), C4=s4.tolist())
    q2 = dict(q)
    q2["dec"] = q["dec"].copy()
    q2["dec"][0] = 255
    try:
        sc.assemble("C1", "batched", {**o, "BQ1": q2}, ch)
        bad = False
    except SystemExit:
        bad = True
    check("assemble_refuses_missing_queue_output", bad)


class FakeGraph:
    """Graph.run_rows on the CPU: logit = row sum, records each batch's real rows."""

    def __init__(self, n):
        self.n, self.calls = n, []

    def run_rows(self, rows):
        self.calls.append(len(rows))
        s = rows.sum(1).float().numpy()
        return s, (s > 0).astype(np.uint8)


def test_queue():
    import torch
    g = FakeGraph(4)
    q = sc.Queue(g)
    got = []
    rows = torch.arange(30).reshape(10, 3)
    got += q.add(rows[:3], np.arange(3))
    wm1 = q.watermark(3)
    got += q.add(rows[3:9], np.arange(3, 9))
    wm2 = q.watermark(9)
    got += q.add(rows[9:9], np.arange(9, 9))
    got += q.add(rows[9:], np.arange(9, 10))
    got += q.flush()
    idx = np.concatenate([x[0] for x in got])
    ok = list(idx) == list(range(10)) and g.calls == [4, 4, 2] and wm1 == 0 and wm2 == 8 and q.watermark(10) == 10 \
        and [x[4] for x in got] == [0, 1, 2]
    check("queue_batches_across_chunks", ok, calls=g.calls, wm=[wm1, wm2], batch_numbers=[x[4] for x in got])


def test_loader():
    d = tempfile.mkdtemp()
    p = 0.006
    os.makedirs(os.path.join(d, sc.p_tag(p)))
    rng = np.random.default_rng(2)
    for s, e in ((0, 2), (2, 3)):
        n = (e - s) * sc.CHUNK
        dig = [f"d{j}" for j in range(s, e)]
        m = {"p": p, "digests": dig, "kind": 13}
        sc.save_npz(os.path.join(d, sc.p_tag(p), f"obs_{s:05d}-{e:05d}.npz"), m, obs=np.zeros(n, np.uint8))
        z = rng.normal(size=n).astype(np.float32)
        sc.save_npz(os.path.join(d, sc.p_tag(p), f"SB_{s:05d}-{e:05d}.npz"), {**m, "path": "engine@4096"}, c=z,
                    dec=(z > 0).astype(np.uint8))
        sh = np.arange(s * sc.CHUNK, e * sc.CHUNK, 7)
        sc.save_npz(os.path.join(d, sc.p_tag(p), f"BQ1_{s:05d}-{e:05d}.npz"), {**m, "path": "compiled@4096"},
                    shot=sh, z=np.ones(len(sh), np.float32), dec=np.ones(len(sh), np.uint8))
    o = sc.load_read(d, p, ["SB", "BQ1"], range(1, 3), {"kind": 13})
    queued = np.concatenate([np.arange(0, 2 * sc.CHUNK, 7), np.arange(2 * sc.CHUNK, 3 * sc.CHUNK, 7)])
    ok = (len(o["obs"]) == 2 * sc.CHUNK and o["digests"] == ["d1", "d2"] and
          int((o["BQ1"]["dec"] == 1).sum()) == int((queued >= sc.CHUNK).sum()) and
          np.isnan(o["BQ1"]["z"][1]) and o["BQ1"]["z"][3] == 1.0)
    try:
        sc.load_read(d, p, ["SB"], range(0, 3), {"kind": 10})
        ref = False
    except SystemExit:
        ref = True
    check("loader_cut_expand_refuse", ok and ref)


def _publish(args):
    return sc.publish(*args)


def test_publish():
    """Finding 7: 32 writers of one observable file in 8 processes; exactly one publishes, the file is whole, no
    temporary file is left, and an existing output is refused or, shared, left for the caller to compare."""
    d = tempfile.mkdtemp()
    f = os.path.join(d, "obs_00000-00001.npz")
    data = os.urandom(1 << 20)
    with mp.get_context("spawn").Pool(8) as pool:
        made = pool.map(_publish, [(f, data)] * 32, chunksize=1)
    ok = sum(made) == 1 and open(f, "rb").read() == data and os.listdir(d) == [os.path.basename(f)]
    ok &= refused(sc.save_npz, f, {}, x=np.zeros(1)) and sc.save_npz(f, {}, shared=True, x=np.zeros(1)) is None
    ok &= refused(sc.write_json, f, {}) and open(f, "rb").read() == data
    check("publish_once_concurrent", ok, published=int(sum(made)), writers=len(made))
    shutil.rmtree(d)


def fake_tree(d, dirty=False):
    os.makedirs(os.path.join(d, "surface", "__pycache__"))
    os.makedirs(os.path.join(d, "release", "weights"))
    open(os.path.join(d, "surface", "sv_x.py"), "w").write("x = 1\n")
    open(os.path.join(d, "surface", "job.sh"), "w").write("run true\n")
    open(os.path.join(d, "release", "weights", "w.bin"), "wb").write(b"\0" * 8)
    open(os.path.join(d, "DEPLOYED"), "w").write(f"abc{'+dirty' if dirty else ''} release def\n")


def test_code_identity():
    """Finding 3: the code digest follows every code and release file and nothing else; dirty trees are marked; an
    analysis under other code is refused."""
    d = tempfile.mkdtemp()
    fake_tree(d)
    c0 = sc.code(d, os.path.join(d, "release"))
    open(os.path.join(d, "surface", "__pycache__", "sv_x.cpython-312.pyc"), "wb").write(b"1")
    open(os.path.join(d, "surface", "NOTES.md"), "w").write("not code")
    c1 = sc.code(d, os.path.join(d, "release"))
    open(os.path.join(d, "surface", "sv_x.py"), "a").write("# changed\n")
    c2 = sc.code(d, os.path.join(d, "release"))
    open(os.path.join(d, "release", "weights", "w.bin"), "ab").write(b"\1")
    c3 = sc.code(d, os.path.join(d, "release"))
    e = tempfile.mkdtemp()
    fake_tree(e, dirty=True)
    cd = sc.code(e, os.path.join(e, "release"))
    here = {**sc.env(), "code": sc.code()}
    ok = (c0 == c1 and c2["digest"] != c1["digest"] and c3["digest"] != c2["digest"] and c0["clean"] and
          c0["commit"] == "abc" and c0["release"] == "def" and c0["files"] == 3 and not cd["clean"])
    ok &= refused(sc.check_analysis, {**here, "code": c0}, "test") and not refused(sc.check_analysis, here, "test")
    check("code_identity_and_refusal", ok, digests=[c0["digest"], c2["digest"], c3["digest"]])
    shutil.rmtree(d)
    shutil.rmtree(e)


def test_committed_pair():
    """Finding 2: no confirmation read or analysis without the passed gate of this lock and the receipt of their
    commit naming both files."""
    d = tempfile.mkdtemp()
    lk, gt, rc = (os.path.join(d, x) for x in ("locked_d13.json", "gate_d13.json", "lock_commit.json"))
    open(lk, "w").write('{"x": 1}')
    lsha = sc.sha256_file(lk)
    json.dump({"lock_sha256": lsha, "all_pass": True}, open(gt, "w"))
    no_receipt = refused(sc.committed_pair, lk, gt)
    json.dump({"commit": "c0ffee", "files": {"locked_d13.json": {"sha256": lsha},
                                             "gate_d13.json": {"sha256": sc.sha256_file(gt)}}}, open(rc, "w"))
    ok = no_receipt and sc.committed_pair(lk, gt)[1]["commit"] == "c0ffee"
    json.dump({"lock_sha256": lsha, "all_pass": False}, open(gt, "w"))
    ok &= refused(sc.committed_pair, lk, gt)                  # a failed gate (and the receipt no longer matches)
    json.dump({"lock_sha256": lsha, "all_pass": True, "note": 1}, open(gt, "w"))
    ok &= refused(sc.committed_pair, lk, gt)                  # a gate file other than the committed one
    check("committed_pair_required", ok)
    shutil.rmtree(d)


def screen_rows(rec, procs, sha=lambda net, path, b, i: "same", end=lambda i: True):
    """Rows as the screen and C3 write them, from processes procs = [(job, pid)]."""
    rows = []
    for i, (job, pid) in enumerate(procs):
        for net, paths in (("S", (("engine", 1), ("engine", 4096))), ("B", (("compiled", 1), ("compiled", 16),
                                                                           ("graph", 1), ("graph", 16)))):
            base = {"role": "net", "net": net, "rep": i + 1, "sha256": "n", "record": rec, "job": job, "host": "h",
                    "pid": pid}
            for path, b in paths:
                rows.append({**base, "metric": "agree", "path": path, "batch": b,
                             "out_sha256": {"0.006": sha(net, path, b, i)}})
                t = {"amortized": {"h2h": {16: 5.0}.get(b, 10.0) + (path == "graph"), "dev": 1.0}}
                if b == 1:
                    t.update(gpu_us=[1.0], host_rt={"mean": 20.0 + (path == "graph"), "p999": 30.0})
                rows.append({**base, "metric": "time", "path": path, "batch": b, **t})
            if end(i):
                rows.append({**base, "metric": "end"})
        for p in sc.PS:
            for mode in ("single", "batched"):
                rows.append({"role": "pipeline", "config": "C3", "p": p, "mode": mode, "job": job, "host": "h",
                             "pid": pid + 1, "latency_us": {"mean": 1.0, "p999": 2.0}, "per_shot_us": 1.0})
    return rows


def test_summarize():
    """Finding 4: three distinct processes that completed the screen, one sha256 per path; a faster path that is not
    reproducible is passed over, and with no reproducible candidate nothing is frozen and nothing written."""
    d = tempfile.mkdtemp()
    rec = {**sc.env(), "code": sc.code(), "gpu": sc.GPU, "torch": "t"}
    three = [("j1", 1), ("j1", 2), ("j1", 3)]

    def run(rows, name):
        f = os.path.join(d, name + ".jsonl")
        open(f, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
        out = os.path.join(d, name + ".json")
        if refused(ss.summarize, argparse.Namespace(rows=f, out=out)):
            return None if not os.path.exists(out) else "written"
        return json.load(open(out))
    good = run(screen_rows(rec, three), "good")
    ok = good is not None and good["frozen"]["B"] == {"single": "compiled@1", "amortized": "compiled@16"}
    flaky = run(screen_rows(rec, three, sha=lambda net, path, b, i: f"{i}" if (path, b) == ("compiled", 16) else "s"),
                "flaky")                                   # compiled@16 is the fastest but differs across processes
    ok &= flaky is not None and flaky["frozen"]["B"]["amortized"] == "graph@16"
    ok &= run(screen_rows(rec, three, sha=lambda net, path, b, i: f"{i}" if net == "B" else "s"), "none") is None
    ok &= run(screen_rows(rec, three, end=lambda i: i < 2), "crashed") is None      # one process never completed
    ok &= run(screen_rows(rec, [("j1", 1), ("j1", 2), ("j1", 1)]), "dup") is None   # a process counted once
    ok &= run(screen_rows({**rec, "code": {**rec["code"], "digest": "other"}}, three), "code") is None
    check("summarize_reproducible_only", ok, last_refusal=refused.last)
    shutil.rmtree(d)


def _journal(d, head, comps, parts):
    """A read's journal in d: its lines (the header, an attempt, a line per part (s, e) at kernel-cache digest k0) and
    every part's files."""
    lines = [json.dumps({"header": head}), json.dumps({"attempt": {"pid": 1}})]
    for s, e in parts:
        pf = {}
        for n in [f"obs_{s:05d}-{e:05d}.npz"] + [f"{c}_{s:05d}-{e:05d}.npz" for c in comps]:
            open(os.path.join(d, n), "wb").write(n.encode())
            pf[n] = sc.sha256_file(os.path.join(d, n))
        lines.append(json.dumps({"part": [s, e], "files": pf, "kcache": "k0"}))
    return lines


def test_resume_journal():
    """Finding 6: the journal's checks before a resume (settings, contiguous complete parts with their sha256), the
    quarantine of a part written only in part, and two successive resumes of a journal whose last line was cut short,
    the journal untouched between them (finding 6 of the recheck of 960b82f): each resume moves the cut line to
    quarantine and cuts it off, so that the lines appended after it stay readable."""
    d = tempfile.mkdtemp()
    comps, c0, c1 = ["SB", "MB"], 0, 250
    head = {"stage": "conf", "chunks": [c0, c1], "identity": {"code": "x"}, "kcache": "k0"}
    prog = os.path.join(d, "progress.jsonl")
    lines = _journal(d, head, comps, ((0, 100), (100, 200)))
    open(os.path.join(d, "SB_00200-00250.npz"), "wb").write(b"half")               # the part being written
    good = "\n".join(lines[:3]) + "\n"
    open(prog, "w").write(good + lines[3][:-9])                                     # part 100-200's line cut short
    k, files, moved = sv.resume_state(prog, head, d, comps, c0, c1)
    ok = k == 100 and len(files) == 3 and sorted(moved) == ["MB_00100-00200.npz", "SB_00100-00200.npz",
                                                             "SB_00200-00250.npz",
                                                             f"progress.jsonl.tail_at_{len(good)}"]
    ok &= open(prog).read() == good
    ok &= os.path.exists(os.path.join(d, "obs_00100-00200.npz"))                   # shared: verified when rewritten
    ok &= not os.path.exists(os.path.join(d, "SB_00200-00250.npz"))
    ok &= os.path.exists(os.path.join(d, "MB_00000-00100.npz"))
    sv.jappend(prog, {"attempt": {"pid": 2}})                                       # the first resume goes on
    n2 = os.path.getsize(prog)
    open(prog, "a").write(lines[3])        # and stops journaling part 100-200: the line whole, its newline not written
    k2, files2, moved2 = sv.resume_state(prog, head, d, comps, c0, c1)               # the second resume
    recs = [next(iter(json.loads(x))) for x in open(prog)]
    ok &= k2 == 100 and files2 == files and moved2 == [f"progress.jsonl.tail_at_{n2}"]
    ok &= recs == ["header", "attempt", "part", "attempt"]
    ok &= refused(sv.resume_state, prog, {**head, "identity": {"code": "y"}}, d, comps, c0, c1)   # other code
    open(os.path.join(d, "MB_00000-00100.npz"), "ab").write(b"x")
    ok &= refused(sv.resume_state, prog, head, d, comps, c0, c1)                                  # a changed file
    check("resume_journal_checks", ok, k=[k, k2], quarantined=[moved, moved2])
    shutil.rmtree(d)


def test_resume_kcache_changed():
    """Finding 2 of the recheck of 960b82f: a read with every part journaled resumes at its end, unless its kernel
    caches changed: the incomplete line of its end, or (a read stopped before its end) a part journaled at another
    digest than the header's, refuses the resume, whatever the caches are now, and leaves the journal as it was."""
    d = tempfile.mkdtemp()
    comps, c0, c1 = ["SB", "MB"], 0, 250
    head = {"stage": "speed", "chunks": [c0, c1], "identity": {"code": "x"}, "kcache": "k0"}
    prog = os.path.join(d, "progress.jsonl")
    lines = _journal(d, head, comps, ((0, 100), (100, 200), (200, 250)))
    open(prog, "w").write("\n".join(lines) + "\n")
    k, _, moved = sv.resume_state(prog, head, d, comps, c0, c1)
    ok = k == c1 and moved == []
    sv.jappend(prog, {"incomplete": {"kcache_end": {"digest": "k1", "files": 2}, "reference": "k0"}})
    before = open(prog).read()
    ok &= refused(sv.resume_state, prog, head, d, comps, c0, c1) and "never reused" in refused.last
    msgs = [refused.last]
    ok &= open(prog).read() == before
    open(prog, "w").write("\n".join(lines[:3] + [lines[3].replace('"k0"', '"k1"')]) + "\n")
    ok &= refused(sv.resume_state, prog, head, d, comps, c0, c1) and "never reused" in refused.last
    msgs.append(refused.last)
    check("resume_refuses_kcache_changed", ok, k=k, refusals=msgs)
    shutil.rmtree(d)


def test_gate_rules():
    """Finding 5: the gate file passes only for the gate's check of this lock with every output of its plan, on the
    plan's chunks and kind, the pipelines in one job and the reads in others."""
    d = tempfile.mkdtemp()
    lk = os.path.join(d, "locked_d13.json")
    L = {"identity": {**sc.env(), "code": sc.code()}, "pipes": sc.DRY_PIPES, "kinds": {"speed": 13}}
    open(lk, "w").write(json.dumps(L, default=list))
    L = json.load(open(lk))
    lsha, plan = sc.sha256_file(lk), L["pipes"]["gate"]
    rows = [{"file": f"{c}_{m}_{p}.npz", "config": c, "mode": m, "p": p, "rep": 1, "kind": 13,
             "chunks": [200, 201 if m == "single" else 205], "shots": plan[m], "lock_sha256": lsha, "job": "9",
             "read_jobs": ["10"]} for c in sc.PIPE_CONFIGS for m in sr.MODES for p in sc.PS]

    def gate(rows, name, **kw):
        ck = os.path.join(d, name + "_check.json")
        json.dump({"expect": "gate", "lock_sha256": lsha, "pass": {"single": True, "batched": True}, "rows": rows,
                   **kw}, open(ck, "w"))
        return not refused(sr.gate, argparse.Namespace(lock=lk, check=ck, out=os.path.join(d, name + "_gate.json")))
    ok = gate(rows, "good") and not gate(rows[1:], "missing") and not gate([{**r, "read_jobs": ["9"]} for r in rows],
                                                                          "samejob")
    ok &= not gate([{**r, "kind": 12} for r in rows], "kind") and not gate([{**r, "chunks": [0, 5]} for r in rows],
                                                                          "chunks")
    ok &= not gate([{**r, "job": str(i)} for i, r in enumerate(rows)], "jobs") and not gate(rows, "c7", expect="c7")
    check("gate_rules", ok, last_refusal=refused.last)
    shutil.rmtree(d)


def compare_reads(da, db):
    """The resume check on dry shots (finding 6): a read interrupted and resumed (db) against the same read
    uninterrupted (da), both a p<p> directory: the same part files, every array equal bit for bit, the same chunk
    digests and the same batch numbers (batch_sha256)."""
    fa, fb = (sorted(f for f in os.listdir(x) if f.endswith(".npz")) for x in (da, db))
    differ = [] if fa == fb else [["file lists", fa, fb]]
    for f in sorted(set(fa) & set(fb)):
        (ma, xa), (mb, xb) = sc.load_npz(os.path.join(da, f)), sc.load_npz(os.path.join(db, f))
        if (set(xa) != set(xb) or any(xa[k].dtype != xb[k].dtype or xa[k].tobytes() != xb[k].tobytes() for k in xa)
                or ma["digests"] != mb["digests"] or ma.get("batch_sha256") != mb.get("batch_sha256")):
            differ.append(f)
    return {"files": len(fa), "batched_files": sum("batch_sha256" in sc.load_npz(os.path.join(da, f))[0] for f in fa),
            "differ": differ}


def report_lock(e, out):
    """The recheck's finding 8 on a dry run's outputs (e: its directory, as e8 left it): the report JSON names the
    lock's sha256, and its verdicts and headline are the ones e8's report wrote. check_analysis is skipped, since the
    dry lock pins the code that ran the dry run, not this one; the rest of report runs as in e8."""
    real, sc.check_analysis = sc.check_analysis, lambda ref, what: None
    try:
        sr.report(argparse.Namespace(lock=f"{e}/locked_d13.json", confirm=f"{e}/confirm_d13.json",
                                     speed=f"{e}/speed.json", pipes=f"{e}/pipes.jsonl", bitwise=f"{e}/bitwise_c7.json",
                                     out=out))
    finally:
        sc.check_analysis = real
    j, old = json.load(open(out + ".json")), json.load(open(f"{e}/SURFACE_RESULT_draft.md.json"))
    lock_sha = sc.sha256_file(f"{e}/locked_d13.json")
    return {"lock_sha256": j.get("lock_sha256"), "is_the_lock's": j.get("lock_sha256") == lock_sha,
            "keys": sorted(j), "rest_as_e8": {k: v for k, v in j.items() if k != "lock_sha256"} == old}


def finish():
    bad = [i for i, r in enumerate(RESULTS) if not r]
    print("ALL PASS" if not bad else f"FAILED: {bad}", flush=True)
    if bad:
        raise SystemExit(1)                        # finding 1: a failed check fails the job


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "compare":      # python -m surface.sv_test compare DIR_A DIR_B
        r = compare_reads(sys.argv[2], sys.argv[3])
        print(json.dumps(r), flush=True)
        raise SystemExit(1 if r["differ"] or not r["files"] else 0)
    if len(sys.argv) == 4 and sys.argv[1] == "report":       # python -m surface.sv_test report DRY_DIR OUT
        r = report_lock(sys.argv[2], sys.argv[3])
        print(json.dumps(r), flush=True)
        raise SystemExit(0 if r["is_the_lock's"] and r["rest_as_e8"] else 1)
    test_stats()
    test_labels()
    test_routing_and_assembly()
    test_queue()
    test_loader()
    test_publish()
    test_code_identity()
    test_committed_pair()
    test_summarize()
    test_resume_journal()
    test_resume_kcache_changed()
    test_gate_rules()
    finish()


if __name__ == "__main__":
    main()
