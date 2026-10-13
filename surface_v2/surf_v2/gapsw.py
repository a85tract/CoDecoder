"""Decoder Switching (Toshio et al., arXiv:2510.25222) as a secondary baseline of the SURF-v2 evaluation: plain PM decides
every shot, its reliability is the complementary gap, and the shots of lowest gap take correlated PM's answer, the
fallback of the ConvLines pairs. Plain and correlated PM are the evaluation's (convlines.circuits: PyMatching 2.4.0 on the
verdict circuit's DEM, decompose_errors=True, approximate_disjoint_errors=True). The only difference from a ConvLines
pair is the front end. This file is the front end's read path: every record it writes carries its sha256.

The complementary gap of a shot is w(other) - w(best), on plain PM's own matching graph and edge weights: w(best) is the
weight of plain PM's correction (Matching.decode(..., return_weight=True)) and w(other) the least weight of a correction
whose logical class is the other one. w(other) comes from a second decode, as the paper does it ("we perform the MWPM
decoding again after constructing a decoding graph with modified boundary syndromes conditioned on the complementary
logical outcome"): the split graph is plain PM's graph with the boundary divided by the logical class. Every edge keeps
its weight and observable; the boundary edges whose flag is set go to one extra node b (index n, after the n detectors)
instead of the boundary. The flag of a boundary edge (u, boundary) is L(e) + s(u) mod 2, where L(e) is the edge's
observable bit and s a 2-colouring of the detectors with s(u) + s(v) = L(e) on every edge between two detectors (it exists
when no cycle of such edges flips the observable; the constructor checks it). For any correction J of syndrome x,
L(J) = deg_b(J) + <s, x> mod 2, so a decode of the split graph with node b set to y returns the least-weight correction
whose observable is y + <s, x>; y = 1 + dec + <s, x> gives w(other). (On the verdict circuits no edge between two
detectors flips the observable, so s = 0 and b collects the boundary edges that flip it.) The edge weights are the same
numbers in both graphs, so PyMatching discretizes them identically: w(best) equals the split graph's least weight in
plain PM's class bit for bit, and the gap is never negative (checked on every calibration shot).

Routing is the registered rule's (surf_v2/qrule.py, config qrule) with the gap in place of |c|: tau_q =
float(np.quantile(gap, q)) for q > 0, tau_0 = 0; a shot is routed when gap < tau_q; F_cal(q) counts the failures when the
routed shots take correlated PM's calibration answer (surf_v2/calcost.py) and the others plain PM's. The projection is
the front end's own: projected(0) = plain PM alone, projected(q) = the front end (plain PM and the gap) + r_q x cost(q),
all on the Grace basis, the front end's times measured on one pinned core and converted as the CPU term converts
(G(d) / the same node's correlated PM per block on validation chunk 0, median of three). The rule is applied as
registered (the speed bar) and without the bar; GAPSW_REGISTRATION.md says which one the comparison uses and why.

    python -m surf_v2.gapsw cal --d 9 --shots CAL --ref REF --core 8 --out OUT         (timed, one pinned core)
    python -m surf_v2.gapsw fit --cal OUT/gapcal_d9.npz --calcost CC/calcost_d9.npz --out POL/policy_d9.json
    python -m surf_v2.gapsw shares --set sv2test --shots TEST --save READS --logs LOGS --jobs 60
    python -m surf_v2.gapsw read --set sv2test --d 9 --chunks 0-9 --shots TEST --save READS   (one share)
    python -m surf_v2.gapsw mask --d 9 --val VAL --out MASKS         (the routed validation shots at the frozen tau)
    python -m surf_v2.gapsw fe --d 9 --val VAL --rep 1 --core 8 --out CPU    (the front end's term, one repetition)"""
import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from convlines import circuits, provenance, testset

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")
POLICIES = os.path.join(HERE, "gapsw")
KEY = "gapsw"


def gapsw_sha256():
    return provenance.sha_file(os.path.abspath(__file__))


def stamp():
    return dict(code_digest=provenance.code_digest(), gapsw_sha256=gapsw_sha256(), release_commit=provenance.commit(),
                host=provenance.host(), versions=provenance.versions())


class FrontEnd:
    """Plain PM (m) and the split graph (m2) of one circuit; see the module's docstring."""

    def __init__(self, d=None, circuit=None, matcher=None):
        import pymatching
        self.m = matcher if matcher is not None else circuits.matcher(d, False, circuit)
        n = self.n = self.m.num_detectors
        edges = self.m.edges()
        adj = [[] for _ in range(n)]
        for u, v, a in edges:
            assert set(a["fault_ids"]) <= {0}, f"edge ({u}, {v}) flips observables {a['fault_ids']}"
            if v is not None:
                f = int(0 in a["fault_ids"])
                adj[u].append((v, f))
                adj[v].append((u, f))
        s = np.full(n, -1, np.int64)
        for r in range(n):                                # 2-colouring, one component at a time
            if s[r] >= 0:
                continue
            s[r], comp, todo = 0, [r], [r]
            while todo:
                u = todo.pop()
                for v, f in adj[u]:
                    if s[v] < 0:
                        s[v] = s[u] ^ f
                        comp.append(v)
                        todo.append(v)
                    elif s[v] != s[u] ^ f:
                        raise ValueError("a cycle of edges between detectors flips the observable: no split")
            if 2 * int(s[comp].sum()) > len(comp):        # the smaller side, fewer bits to count per shot
                s[comp] ^= 1
        self.s = s.astype(np.uint8)
        self.S = np.flatnonzero(self.s)
        m2 = pymatching.Matching()
        self.to_b = 0
        for u, v, a in edges:
            kw = dict(fault_ids=set(a["fault_ids"]), weight=a["weight"], error_probability=a["error_probability"])
            if v is not None:
                m2.add_edge(u, v, **kw)
            elif int(0 in a["fault_ids"]) ^ int(self.s[u]):
                m2.add_edge(u, n, **kw)
                self.to_b += 1
            else:
                m2.add_boundary_edge(u, **kw)
        assert self.to_b and m2.num_detectors == n + 1, "the split graph has no class boundary"
        self.m2 = m2
        self.buf = np.zeros(n + 1, np.uint8)

    def parity(self, det):
        """<s, x> mod 2 of every shot, (N,) uint8."""
        return (det[:, self.S].sum(1, dtype=np.int64) & 1).astype(np.uint8)

    def least(self, det, value):
        """(N,) the least weight of a correction whose observable is value (N,) uint8, from the split graph."""
        ext = np.empty((len(det), self.n + 1), np.uint8)
        ext[:, :self.n] = det
        ext[:, self.n] = value ^ self.parity(det)
        p, w = self.m2.decode_batch(ext, return_weights=True)
        assert np.array_equal(np.asarray(p, np.uint8).reshape(len(det), -1)[:, 0], value), "the class bookkeeping failed"
        return np.asarray(w, np.float64)

    def batch(self, det):
        """det (N, n) uint8 -> plain PM's answer (N,) uint8, its weight and the gap (N,) float64."""
        det = np.ascontiguousarray(det, np.uint8)
        p, w = self.m.decode_batch(det, return_weights=True)
        dec = np.asarray(p, np.uint8).reshape(len(det), -1)[:, 0]
        w = np.asarray(w, np.float64)
        return dec, w, self.least(det, dec ^ 1) - w


def extended(det):
    """The shots with one spare byte each, (N, n + 1) uint8: the front end's input buffers (the spare byte is node b's)."""
    ext = np.zeros((len(det), det.shape[1] + 1), np.uint8)
    ext[:, :-1] = det
    return ext


def loop(fe, ext, warmup=50):
    """The front end one shot at a time, as the CPU term's loop (timing.cpu_term.loop: 50 untimed warm-up shots, then
    perf_counter around each shot): plain PM's decode with its weight, node b's bit, and the split graph's decode in the
    other class. ext: extended(det), whose spare byte is written here. -> answer (N,) uint8, weight, gap, the time of plain
    PM's decode and of the whole front end (N,) us."""
    m, m2, S, n = fe.m, fe.m2, fe.S, fe.n
    for i in range(min(warmup, len(ext))):
        x = ext[i]
        p, w = m.decode(x[:n], return_weight=True)
        x[n] = p[0] ^ 1 ^ (np.count_nonzero(x[S]) & 1 if S.size else 0)
        m2.decode(x, return_weight=True)
    N = len(ext)
    dec, wt, gap = np.empty(N, np.uint8), np.empty(N), np.empty(N)
    tp, tf = np.empty(N), np.empty(N)
    pc = time.perf_counter
    for i in range(N):
        x = ext[i]
        t0 = pc()
        p, w = m.decode(x[:n], return_weight=True)
        t1 = pc()
        x[n] = p[0] ^ 1 ^ (np.count_nonzero(x[S]) & 1 if S.size else 0)
        w2 = m2.decode(x, return_weight=True)[1]
        t2 = pc()
        dec[i], wt[i], gap[i], tp[i], tf[i] = p[0], w, w2 - w, t1 - t0, t2 - t0
    return dec, wt, gap, tp * 1e6, tf * 1e6


def save(stem, arrays, rec):
    """arrays in testset.write_npz's format and their record (stem.json), each written once."""
    testset.write_npz(stem + ".npz", arrays)
    rec = dict(rec, arrays={k: dict(dtype=str(v.dtype), sha256=hashlib.sha256(
        np.ascontiguousarray(v).tobytes()).hexdigest()) for k, v in sorted(arrays.items())},
        npz=os.path.basename(stem + ".npz"), npz_sha256=provenance.sha_file(stem + ".npz"))
    with open(stem + ".json", "x") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec


def load(path):
    """An npz this module wrote, checked against its record -> (arrays, record)."""
    rec = json.load(open(path[:-4] + ".json"))
    assert rec["npz_sha256"] == provenance.sha_file(path), f"{path}: not the file its record names"
    z = dict(np.load(path))
    for k, v in rec["arrays"].items():
        assert hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() == v["sha256"], \
            f"{path}: {k} differs from its record"
    return z, rec


def manifest_sha(set_):
    return testset.sha256_bytes(open(os.path.join(testset.ROOT, testset.SETS[set_]), "rb").read())


# ---- calibration: the front end on every calibration shot, one pinned core ----------------------------------------------
def cal(a):
    from timing import cpu_term
    refs = sorted(glob.glob(os.path.join(a.ref, f"ref_d{a.d}_rep*.json")))
    ref = [json.load(open(f)) for f in refs]
    assert len(ref) == 3 and all(r["kind"] == "correlated" for r in ref), f"three correlated refs of d = {a.d}: {refs}"
    det, obs = testset.load_chunks(a.set, a.d, testset.parse_range(a.chunks), a.shots)
    det = np.ascontiguousarray(det)
    t_build = time.perf_counter()
    fe = FrontEnd(a.d)
    t_build = time.perf_counter() - t_build
    pinned = cpu_term.pin(a.core)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dec, w, gap, tp_in, tf = loop(fe, extended(det))
    tp, dec_alone = cpu_term.loop(fe.m, det, False)                  # plain PM alone, the official loop
    t1 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    assert np.array_equal(dec_alone, dec), "plain PM alone and inside the front end answer differently"
    bdec, bw, bgap = fe.batch(det)                                   # the checks, after the timed loops
    assert np.array_equal(bdec, dec) and np.array_equal(bw, w) and np.array_equal(bgap, gap), \
        "the single-shot front end differs from the batch decode"
    w0, w1 = fe.least(det, np.zeros(len(det), np.uint8)), fe.least(det, np.ones(len(det), np.uint8))
    assert np.array_equal(np.where(dec == 1, w1, w0), w), "plain PM's weight is not the least weight in its class"
    assert np.array_equal(np.where(dec == 1, w0, w1) - w, gap) and (gap >= 0).all(), "a gap is not w(other) - w(best)"
    g, src = cpu_term.grace_numerator(a.d)
    node = float(np.median([r["per_block_us"] for r in ref]))
    os.makedirs(a.out, exist_ok=True)
    rec = save(os.path.join(a.out, f"gapcal_d{a.d}"),
               dict(dec=dec, gap=gap, weight=w, obs=obs.astype(np.uint8), time_front_us=tf, time_plain_us=tp,
                    time_plain_in_front_us=tp_in),
               dict(what="SURF-v2 gap switching: the front end on every calibration shot (surf_v2/gapsw.py cal)", d=a.d,
                    set=a.set, chunks=a.chunks, shots=int(len(obs)), manifest_sha256=manifest_sha(a.set), core=a.core,
                    pinned=pinned, started=t0, ended=t1, build_s=t_build, plain_failures=int((dec != obs).sum()),
                    gap_zero=int((gap == 0).sum()), gap_distinct=int(len(np.unique(gap))), split_to_b=fe.to_b,
                    split_colour_ones=int(len(fe.S)), mean_front_us=float(tf.mean()), mean_plain_us=float(tp.mean()),
                    mean_plain_in_front_us=float(tp_in.mean()), ref_reps_us=[r["per_block_us"] for r in ref],
                    ref_files=[os.path.basename(f) for f in refs], node_median_us=node, grace_numerator_us=g,
                    grace_numerator_source=src, ratio=g / node,
                    checks="single shot = batch (answer, weight, gap); plain alone = in the front end; w(best) = the "
                           "split graph's least weight in plain PM's class; gap = the other class's least weight - "
                           "w(best) >= 0; the split decode's observable is the class it was set to", **stamp()))
    print(f"d{a.d}: plain PM {rec['plain_failures']} failures in {len(obs)}; front end {rec['mean_front_us']:.2f} us, "
          f"plain alone {rec['mean_plain_us']:.2f} us per shot on this core (ratio {rec['ratio']:.4f}); gap 0 on "
          f"{rec['gap_zero']} shots; checks passed", flush=True)


# ---- the q rule on the gap --------------------------------------------------------------------------------------------
def fit_gap(gap, dec, corr, obs, grid, M, bar_us, front_us, plain_us, cost):
    """surf_v2.qrule.fit_rule with the gap in place of |c| and plain PM's answer in place of c > 0. front_us, plain_us:
    the Grace-basis times of the front end and of plain PM alone; cost(q, routed) -> cost(q) in Grace us. -> the table,
    the registered choice (admissible: projected <= bar) and the choice without the bar (every q admissible)."""
    gap = np.asarray(gap)
    dec, corr, obs = (np.asarray(x, np.uint8) for x in (dec, corr, obs))
    rows = []
    for q in sorted(grid):
        tau = float(np.quantile(gap, q)) if q > 0 else 0.0
        r = gap < tau
        x = dict(q=q, tau=tau, routed=int(r.sum()), failures=int((np.where(r, corr, dec) != obs).sum()))
        x["cost_us"] = (float(cost(q, r)) if r.any() else 0.0) if q > 0 else None
        x["projected_us"] = plain_us if q == 0 else front_us + float(r.mean()) * x["cost_us"]
        x["admissible"] = x["projected_us"] <= bar_us
        x["routed_fraction"] = float(r.mean())
        rows.append(x)
    lim = M - 2 * math.sqrt(2 * M)

    def choose(adm):
        ok = [x for x in adm if x["failures"] <= lim]
        ch = (ok[0] if ok else min(adm, key=lambda x: (x["failures"], x["q"]))) if adm else None
        out = ("q = 0: plain PM alone" if ch and ch["q"] == 0 else "routed pair" if ch else
               "no admissible q: plain PM alone, labelled")
        return dict(chosen=ch, qualifies=bool(ok), eligible=bool(adm), outcome=out)
    return dict(grid=sorted(grid), margin=M, limit=lim, bar_us=bar_us, front_us=front_us, plain_us=plain_us, table=rows,
                registered=choose([x for x in rows if x["admissible"]]), no_bar=choose(rows), shots=int(len(gap)),
                plain_failures=int((dec != obs).sum()), pm_corr_failures=int((corr != obs).sum()))


def fit(a):
    cf = json.load(open(CONFIG))
    from surf_v2 import qrule
    z, rec = load(a.cal)
    cc, crec = qrule.load_calcost(a.calcost)
    d, k = rec["d"], str(rec["d"])
    want = (cf["dry"]["set"], cf["dry"]["cal_chunks"]) if a.dry else (cf["calibration"]["set"], cf["calibration"]["chunks"])
    assert (rec["set"], rec["chunks"]) == want and (crec["set"], crec["chunks"]) == want, f"not the calibration set {want}"
    assert crec["d"] == d and crec["manifest_sha256"] == rec["manifest_sha256"], "calcost belongs to another set"
    assert np.array_equal(cc["obs"], z["obs"]), "calcost's shots are not the front end's"
    assert rec["gapsw_sha256"] == gapsw_sha256(), "the calibration read was made by another gapsw.py"
    t_grace = cc["time_us"] * crec["ratio"]
    front, plain = rec["mean_front_us"] * rec["ratio"], rec["mean_plain_us"] * rec["ratio"]
    pol = fit_gap(z["gap"], z["dec"], cc["dec"], z["obs"], cf["qrule"]["grid"][k], cf["qrule"]["M"][k],
                  cf["qrule"]["bar_us"][k], front, plain, cost=lambda q, r: t_grace[r].mean())
    op = pol[a.operative]
    pol.update(what="SURF-v2 gap switching: the routed fraction on the calibration shots (surf_v2/gapsw.py fit)", d=d,
               operative=a.operative, chosen=op["chosen"], outcome=op["outcome"],
               q=op["chosen"]["q"] if op["chosen"] else None, tau=op["chosen"]["tau"] if op["chosen"] else None,
               rule=cf["qrule"]["source"], dry=bool(a.dry), set=rec["set"], chunks=rec["chunks"],
               cal=os.path.basename(a.cal), cal_sha256=provenance.sha_file(a.cal),
               cal_record_sha256=provenance.sha_file(a.cal[:-4] + ".json"), calcost=os.path.basename(a.calcost),
               calcost_sha256=provenance.sha_file(a.calcost), calcost_ratio=crec["ratio"], front_ratio=rec["ratio"],
               front_node_us=rec["mean_front_us"], plain_node_us=rec["mean_plain_us"],
               registered_plain_pm_us=cf["registered_times_us"][k]["plain_pm"],
               registered_corr_pm_us=cf["registered_times_us"][k]["corr_pm"], config_sha256=provenance.sha_file(CONFIG),
               gapsw_sha256=gapsw_sha256())
    for x in pol["table"]:
        print(f"q={100 * x['q']:4.1f} % tau={x['tau']:.6f} routed={x['routed']:6d} F_cal={x['failures']} projected "
              f"{x['projected_us']:.2f} us admissible {x['admissible']}", flush=True)
    print(f"d{d}: limit {pol['limit']:.1f}; registered: {pol['registered']['outcome']}; without the bar: q = "
          f"{pol['no_bar']['chosen']['q']}, F_cal {pol['no_bar']['chosen']['failures']}; correlated PM "
          f"{pol['pm_corr_failures']}, plain PM {pol['plain_failures']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "x") as fh:
        json.dump(pol, fh, indent=1, sort_keys=True)


def policy(d, pdir=POLICIES):
    """The frozen policy of d (surf_v2/gapsw/policy_d<d>.json) with its sha256."""
    f = os.path.join(pdir, f"policy_d{d}.json")
    return json.load(open(f)), provenance.sha_file(f)


# ---- the test read: answers and gaps on a set's chunks, in shares ----------------------------------------------------
def read(a):
    ch = testset.parse_range(a.chunks)
    det, obs = testset.load_chunks(a.set, a.d, ch, a.shots)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dec, w, gap = FrontEnd(a.d).batch(det)
    t1 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(a.save, exist_ok=True)
    rec = save(os.path.join(a.save, f"{a.set}_d{a.d}_{KEY}_{a.chunks}"),
               dict(obs=obs.astype(np.uint8), dec=dec, gap=gap, weight=w),
               dict(what="SURF-v2 gap-switching read (surf_v2/gapsw.py read)", set=a.set, d=a.d, chunks=a.chunks,
                    shots=int(len(obs)), manifest_sha256=manifest_sha(a.set), decoder=KEY, started=t0, ended=t1,
                    **stamp()))
    print(f"d{a.d} chunks {a.chunks}: plain PM {int((dec != obs).sum())} failures in {len(obs)}, gap 0 on "
          f"{int((gap == 0).sum())}; {rec['npz']}", flush=True)


def shares(a):
    """surf_v2.shares for the front end: consecutive chunks per share, several processes at a time; a share whose
    record exists is skipped."""
    cf = json.load(open(CONFIG))
    ds = cf["distances"] if a.d == "all" else [int(x) for x in a.d.split(",")]
    os.makedirs(a.save, exist_ok=True)
    os.makedirs(a.logs, exist_ok=True)
    todo = []
    for d in ds:
        ch = testset.parse_range(a.chunks or cf["test"]["chunks"][str(d)])
        for i in range(0, len(ch), a.size):
            part = ch[i:i + a.size]
            assert part == list(range(part[0], part[-1] + 1))
            spec = f"{part[0]}-{part[-1]}"
            if not os.path.exists(os.path.join(a.save, f"{a.set}_d{d}_{KEY}_{spec}.json")):
                todo.append((d, spec))
    print(f"{len(todo)} shares to read with {a.jobs} processes", flush=True)
    run, bad = {}, []
    while todo or run:
        while todo and len(run) < a.jobs:
            d, spec = todo.pop(0)
            log = open(os.path.join(a.logs, f"{a.set}_d{d}_{KEY}_{spec}.log"), "w")
            p = subprocess.Popen([sys.executable, "-m", "surf_v2.gapsw", "read", "--set", a.set, "--d", str(d), "--chunks",
                                  spec, "--shots", a.shots, "--save", a.save], stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS="1"))
            run[p] = (d, spec, log, time.time())
        time.sleep(2)
        for p in [p for p in run if p.poll() is not None]:
            d, spec, log, t0 = run.pop(p)
            log.close()
            print(f"d{d} chunks {spec}: exit {p.returncode} in {time.time() - t0:.0f} s", flush=True)
            if p.returncode:
                bad.append((d, spec))
    print(f"failed shares: {bad}" if bad else "every share read", flush=True)
    raise SystemExit(1 if bad else 0)


# ---- timing after the freeze: the routed validation shots and the front end's term -----------------------------------
def val_shots(d, val):
    sh = [testset.load(os.path.join(val, testset.name("val", d, j)), d) for j in range(20)]
    return np.ascontiguousarray(np.concatenate([s[0] for s in sh])), np.concatenate([s[1] for s in sh])


def mask(a):
    """timing.deployed's mask format for timing.cpu_term term: the validation shots routed at the frozen tau."""
    from timing import deployed
    pol, psha = policy(a.d, a.policies)
    assert pol["d"] == a.d and pol["tau"] is not None and bool(pol["dry"]) == a.dry, f"policy of d = {a.d}: {pol['outcome']}"
    det, obs = val_shots(a.d, a.val)
    dec, w, gap = FrontEnd(a.d).batch(det)
    routed = gap < pol["tau"]
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f"val_{KEY}_d{a.d}.npz"), c=gap, routed=routed.astype(np.uint8), d=a.d,
             network=f"{KEY}_d{a.d}", tau=np.float64(pol["tau"]), logits="gap", shots_sha256=deployed.shots_sha256(det, obs),
             policy_sha256=psha, gapsw_sha256=gapsw_sha256())
    print(f"d{a.d}: {int(routed.sum())} of {len(det)} validation shots routed at tau {pol['tau']!r} (q {pol['q']})",
          flush=True)


def fe_term(a):
    """One fresh-process repetition of the front end's term: the loop over the 200,000 validation shots."""
    from timing import cpu_term, deployed
    det, obs = val_shots(a.d, a.val)
    fe = FrontEnd(a.d)
    res = dict(what="fe", d=a.d, rep=a.rep, core=a.core, load1=os.getloadavg()[0], start=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               shots=len(det), shots_sha256=deployed.shots_sha256(det, obs), **stamp())
    res["pinned"] = cpu_term.pin(a.core)
    _, _, _, tp, tf = loop(fe, extended(det))
    res.update(term_us=float(tf.mean()), plain_in_front_us=float(tp.mean()), end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"fe_{KEY}_d{a.d}_rep{a.rep}.json"), "x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"d{a.d} front end rep {a.rep}: {res['term_us']:.2f} us per shot (plain PM's decode {res['plain_in_front_us']:.2f})",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cal")
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--shots", required=True)
    p.add_argument("--set", default="sv2cal")
    p.add_argument("--chunks", default="0-19")
    p.add_argument("--ref", required=True, help="the directory with this job's ref_d<d>_rep*.json (timing.cpu_term ref)")
    p.add_argument("--core", type=int, default=8)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--cal", required=True)
    p.add_argument("--calcost", required=True)
    p.add_argument("--operative", choices=("no_bar", "registered"), default="no_bar",
                   help="the choice the comparison uses (GAPSW_REGISTRATION.md): the rule without its speed bar, or as registered")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--out", required=True)
    for name in ("read", "shares"):
        p = sub.add_parser(name)
        p.add_argument("--set", required=True, choices=sorted(testset.SETS))
        p.add_argument("--d", required=name == "read", type=int if name == "read" else str, default="all")
        p.add_argument("--chunks", required=name == "read", default="",
                       help="read: the share's chunks; shares: every d's chunks (dry runs), default the config's test chunks")
        p.add_argument("--shots", required=True)
        p.add_argument("--save", required=True)
        if name == "shares":
            p.add_argument("--logs", required=True)
            p.add_argument("--size", type=int, default=10)
            p.add_argument("--jobs", type=int, default=8)
    p = sub.add_parser("mask")
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--policies", default=POLICIES)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("fe")
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--core", type=int, default=8)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cal": cal, "fit": fit, "read": read, "shares": shares, "mask": mask, "fe": fe_term}[a.cmd](a)


if __name__ == "__main__":
    main()
