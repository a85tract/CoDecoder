"""ARB5 (exploratory, nothing registered): the scores of the joint decision on the routed shots of the SURF-v2 networks.

For every shot whose |c| is below tau at q_max (the largest q the analysis reads), this computes, one shot at a time on one
pinned core: plain PM's answer and weight, plain PM's complementary gap (surf_v2.gapsw.FrontEnd's split graph, the gap
code of the SURF-v2 gap switching), correlated PM's answer and its weight on the reweighted edges (PyMatching's
return_weight with enable_correlations), and correlated PM's least weight in the other logical class (a correlated decode
of the split DEM: the DEM with node b added to every boundary component that flips the observable, decoded with b set to
the other class; its first pass and reweighting are those of the other class's best matching, so this gap is a proxy of
the gap on the unconstrained second pass's reweighted graph, which PyMatching does not expose). Each step is timed with
perf_counter after 50 untimed warm-up shots. It also writes the per-chunk failure counts of every shot that the analysis
needs, and, at d = 7-17 on sv2cal, the registered front-end times (surf_v2/gapsw cal, exclusive node, core 8) averaged
over the network's routed calibration shots.

Runs in the release's container with PYTHONPATH = a deployed SURF-v2 tree (convlines, surf_v2):
    python arb5_alpine.py sv2 --d 7 --core RANK --out OUT     (RANK: which core of the job's cpuset)
    python arb5_alpine.py d9s --routed d9s_routed.npz --manifest d9s_test.json --shots DIR --core RANK --out OUT"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import time

import numpy as np
import pymatching
import stim

from convlines import circuits, testset
from surf_v2 import gapsw

E = "<cluster-scratch>/surf_v2/eval"
NET = {7: "d7_b1w64_sv2", 11: "d11_b5_sv2", 13: "d13_b5_sv2", 15: "d15_b5_sv2", 17: "d17_b5_sv2"}
CHUNKS = {7: 200, 11: 100, 13: 100, 15: 100, 17: 100}
QS = (0.01, 0.02, 0.04, 0.05, 0.06)
QMAX = 0.05
# the policies' tau (surf_v2/policies/policy_d<d>_*.json table), to check that this code's quantile is the rule's
POLICY_TAU = {7: {0.01: 0.45196643471717834, 0.02: 0.90324866771698, 0.04: 1.705165147781372},
              11: {0.01: 1.250405192375183, 0.02: 2.222313165664673, 0.04: 3.536041498184204},
              13: {0.01: 1.5388953685760498, 0.02: 2.597458839416504, 0.04: 4.052864074707031},
              15: {0.01: 1.9195706844329834, 0.02: 3.141986846923828, 0.04: 4.708978176116943},
              17: {0.01: 2.2122440338134766, 0.02: 3.504336357116699, 0.04: 5.074368476867676}}
REGISTERED = {7: (0.02, 0.90324866771698), 11: (0.0, 0.0), 13: (0.0, 0.0), 15: (0.01, 1.9195706844329834),
              17: (0.01, 2.2122440338134766)}


def split_dem(d):
    """The DEM with node b (index n) added to every boundary component that flips observable 0; components keep their
    order and the decomposition separators, so correlated matching sees the same hyperedges."""
    dem = circuits.dem(circuits.circuit(d)).flattened()
    n = dem.num_detectors
    out = stim.DetectorErrorModel()
    added = 0
    for ins in dem:
        if ins.type != "error":
            out.append(ins)
            continue
        comps, cur = [], []
        for t in ins.targets_copy():
            if t.is_separator():
                comps.append(cur)
                cur = []
            else:
                cur.append(t)
        comps.append(cur)
        tg = []
        for k, comp in enumerate(comps):
            dets = [t for t in comp if t.is_relative_detector_id()]
            if any(t.is_logical_observable_id() for t in comp):
                assert len(dets) == 1, f"a component with {len(dets)} detectors flips the observable: no split"
                comp = comp + [stim.target_relative_detector_id(n)]
                added += 1
            if k:
                tg.append(stim.target_separator())
            tg.extend(comp)
        out.append(stim.DemInstruction("error", ins.args_copy(), tg))
    assert out.num_detectors == n + 1 and added
    return out, n, added


def pin(rank):
    """Pin this process to the rank-th core of the job's cpuset -> (the core, the cpuset)."""
    allowed = sorted(os.sched_getaffinity(0))
    core = allowed[rank]
    os.sched_setaffinity(0, {core})
    return core, allowed


def loop(d, det, warmup=50, diagnostic=True):
    """The scores and times of every shot of det (N, n) uint8, one shot at a time (diagnostic: also the untimed
    correlated decode of the split DEM in correlated PM's own class)."""
    fe = gapsw.FrontEnd(d)
    assert fe.S.size == 0, "the verdict circuit needs a colouring: not handled here"
    m, m2, n = fe.m, fe.m2, fe.n
    mc = circuits.matcher(d, True)
    sdem, n2, added = split_dem(d)
    assert n2 == n
    m2c = pymatching.Matching.from_detector_error_model(sdem, enable_correlations=True)
    N = len(det)
    ext = np.zeros((N, n + 1), np.uint8)
    ext[:, :n] = det
    o = {k: np.empty(N, np.uint8) for k in ("pm", "cpm", "cls_other", "cls_same")}
    o.update({k: np.empty(N) for k in ("w_pm", "w_split", "w_cpm", "w_c_other", "w_c_same",
                                       "t_pm", "t_split", "t_cpm", "t_c_other")})
    pc = time.perf_counter
    for i in list(range(min(warmup, N))) + list(range(N)):
        x = ext[i]
        t0 = pc()
        p, w = m.decode(x[:n], return_weight=True)
        t1 = pc()
        x[n] = p[0] ^ 1
        w2 = m2.decode(x, return_weight=True)[1]
        t2 = pc()
        q, wc = mc.decode(x[:n], return_weight=True, enable_correlations=True)
        t3 = pc()
        x[n] = q[0] ^ 1
        r, wo = m2c.decode(x, return_weight=True, enable_correlations=True)
        t4 = pc()
        if diagnostic:
            x[n] = q[0]
            s, ws = m2c.decode(x, return_weight=True, enable_correlations=True)    # untimed
        else:
            s, ws = q, wc
        o["pm"][i], o["cpm"][i], o["cls_other"][i], o["cls_same"][i] = p[0], q[0], r[0], s[0]
        o["w_pm"][i], o["w_split"][i], o["w_cpm"][i], o["w_c_other"][i], o["w_c_same"][i] = w, w2, wc, wo, ws
        o["t_pm"][i], o["t_split"][i], o["t_cpm"][i], o["t_c_other"][i] = \
            (t1 - t0) * 1e6, (t2 - t1) * 1e6, (t3 - t2) * 1e6, (t4 - t3) * 1e6
    o["gap_pm"] = o["w_split"] - o["w_pm"]
    o["gap_cpm"] = o["w_c_other"] - o["w_cpm"]
    info = dict(n=n, split_dem_components_to_b=added, split_graph_to_b=fe.to_b)
    return o, info


def env():
    return dict(host=platform.node(), cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
                python=platform.python_version(), numpy=np.__version__, stim=stim.__version__,
                pymatching=pymatching.__version__, code_tree=os.environ.get("PYTHONPATH", ""),
                gapsw_sha256=gapsw.gapsw_sha256(), job=os.environ.get("SLURM_JOB_ID"))


def shares(pattern):
    """Concatenate the share files of one decoder in chunk order -> dict of arrays."""
    fs = glob.glob(pattern)
    fs.sort(key=lambda f: int(re.search(r"_(\d+)-(\d+)\.npz$", f).group(1)))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}, [os.path.basename(f) for f in fs]


def counts(chunk, bad, nch):
    return np.bincount(chunk[bad], minlength=nch).astype(np.int64)


def sv2(a):
    d, net = a.d, NET[a.d]
    core, allowed = pin(a.core)
    nch = CHUNKS[d]
    z = np.load(glob.glob(f"{E}/pod/reads/test/sv2test_d{d}_{net}_0-{nch - 1}.npz")[0])
    c, obs, pair_dec, routed_read = z["c"], z["obs"].astype(np.uint8), z["dec"], z["routed"]
    assert c.dtype == np.float32 and len(c) == nch * testset.CHUNK
    cp, cp_files = shares(f"{E}/reads/test/sv2test_d{d}_corr_pm_*.npz")
    pp, pp_files = shares(f"{E}/reads/test/sv2test_d{d}_plain_pm_*.npz")
    gg, gg_files = shares(f"{E}/gapsw/reads/test/sv2test_d{d}_gapsw_*.npz")
    for x in (cp, pp, gg):
        assert np.array_equal(x["obs"].astype(np.uint8), obs), "the reads' observables differ"
    cpm, pm = cp["dec"].astype(np.uint8), pp["dec"].astype(np.uint8)
    assert np.array_equal(gg["dec"].astype(np.uint8), pm), "gap switching's front end is not plain PM's read"
    # tau of each q from the network's calibration read, as surf_v2.qrule.fit_rule computes it
    zc = np.load(f"{E}/pod/reads/cal/sv2cal_d{d}_{net}_0-19.npz")
    ccal = zc["c"]
    assert ccal.dtype == np.float32
    tau = {q: float(np.quantile(np.abs(ccal), q)) for q in QS}
    for q, t in POLICY_TAU[d].items():
        assert tau[q] == t, f"tau({q}) = {tau[q]!r}, the policy's {t!r}"
    qreg, treg = REGISTERED[d]
    netd = (c > 0).astype(np.uint8)
    az = np.abs(c)
    assert np.array_equal(routed_read.astype(bool), az < treg), "the read's routing is not |c| < the registered tau"
    pair = np.where(az < treg, cpm, netd)
    sel = np.flatnonzero(az < tau[QMAX])
    chunk = np.arange(len(c)) // testset.CHUNK
    cnt = dict(cpm_fail=counts(chunk, cpm != obs, nch), net_fail=counts(chunk, netd != obs, nch),
               pm_fail=counts(chunk, pm != obs, nch), pair_fail=counts(chunk, pair != obs, nch))
    # the routed shots' detection events, each chunk checked against the manifest
    man = testset.manifest("sv2test")
    dets = []
    for j in np.unique(chunk[sel]):
        f = testset.name(man["split"], d, int(j))
        det, ob = testset.load(f"{E}/shots/sv2test/{f}", d, man["files"][f]["sha256"])
        rows = sel[chunk[sel] == j] - j * testset.CHUNK
        assert np.array_equal(ob[rows], obs[sel[chunk[sel] == j]]), f"{f}: observables differ from the reads"
        dets.append(det[rows])
    det = np.concatenate(dets)
    t0 = time.time()
    o, info = loop(d, det)
    wall = time.time() - t0
    chk = dict(pm_equal_read=int((o["pm"] == pm[sel]).sum()), cpm_equal_read=int((o["cpm"] == cpm[sel]).sum()),
               gap_pm_equal_read=int((o["gap_pm"] == gg["gap"][sel]).sum()),
               other_class_ok=int((o["cls_other"] == 1 - o["cpm"]).sum()),
               same_class_ok=int((o["cls_same"] == o["cpm"]).sum()),
               w_c_same_equal_w_cpm=int((o["w_c_same"] == o["w_cpm"]).sum()),
               gap_cpm_negative=int((o["gap_cpm"] < 0).sum()), shots=int(len(sel)),
               pair_dec_equal_read=int((pair == pair_dec).sum()), all_shots=int(len(c)))
    # the registered front-end times (exclusive node, core 8) on the network's routed calibration shots
    g = np.load(f"{E}/gapsw/cal/gapcal_d{d}.npz")
    assert np.array_equal(g["obs"].astype(np.uint8), zc["obs"].astype(np.uint8)), "gapcal and the cal read differ"
    gj = json.load(open(f"{E}/gapsw/cal/gapcal_d{d}.json"))
    reg = {}
    for q in QS:
        r = np.abs(ccal) < tau[q]
        reg[str(q)] = dict(routed=int(r.sum()), front_us=float(g["time_front_us"][r].mean()),
                           split_us=float((g["time_front_us"] - g["time_plain_in_front_us"])[r].mean()),
                           plain_us=float(g["time_plain_us"][r].mean()))
    reg["all"] = dict(routed=int(len(ccal)), front_us=float(g["time_front_us"].mean()),
                      split_us=float((g["time_front_us"] - g["time_plain_in_front_us"]).mean()),
                      plain_us=float(g["time_plain_us"].mean()), ratio_to_grace=gj["ratio"], core=gj["core"],
                      node=gj["host"]["node"] if isinstance(gj["host"], dict) else str(gj["host"]))
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/arb5_d{d}.npz", idx=sel.astype(np.int64), c=c[sel], obs=obs[sel], cpm_read=cpm[sel],
             pm_read=pm[sel], gap_read=gg["gap"][sel], **o, **{f"count_{k}": v for k, v in cnt.items()})
    rec = dict(what="ARB5 scores on the routed shots (arb5_alpine.py sv2)", d=d, network=net, set="sv2test",
               chunks=nch, qmax=QMAX, tau={str(q): t for q, t in tau.items()}, registered=dict(q=qreg, tau=treg),
               checks=chk, info=info, wall_s=wall, core=core, cpuset=allowed, registered_times=reg,
               files=dict(net=f"sv2test_d{d}_{net}_0-{nch - 1}.npz", corr=cp_files, plain=pp_files, gapsw=gg_files),
               loop_means_us={k: float(o[k].mean()) for k in ("t_pm", "t_split", "t_cpm", "t_c_other")}, **env())
    json.dump(rec, open(f"{a.out}/arb5_d{d}.json", "w"), indent=1)
    print(json.dumps(dict(d=d, checks=chk, wall_s=round(wall, 1), loop=rec["loop_means_us"])), flush=True)


def d9s(a):
    core, allowed = pin(a.core)
    d = 9
    r = np.load(a.routed)
    idx = r["idx"].astype(np.int64)
    man = json.load(open(a.manifest))
    files = man["files"]
    cir = circuits.circuit(d)
    os.makedirs(a.shots, exist_ok=True)
    chunk = idx // testset.CHUNK
    dets, regen = [], []
    for j in np.unique(chunk):
        f = f"d9_p0.006_X_d9stest_{int(j):03d}.npz"
        e = files[f]
        assert e["j"] == j
        path = os.path.join(a.shots, f)
        if not os.path.exists(path):
            testset.write_npz(path, testset.sample(cir, e["seed"]))
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        regen.append(got == e["sha256"])
        assert got == e["sha256"], f"{f}: regenerated sha256 {got} != manifest {e['sha256']}"
        det, ob = testset.load(path, d, e["sha256"])
        rows = idx[chunk == j] - j * testset.CHUNK
        assert np.array_equal(ob[rows], r["obs"][chunk == j]), f"{f}: observables differ from the reads"
        dets.append(det[rows])
    det = np.concatenate(dets)
    t0 = time.time()
    o, info = loop(d, det)
    wall = time.time() - t0
    chk = dict(pm_equal_read=int((o["pm"] == r["pm"]).sum()), cpm_equal_read=int((o["cpm"] == r["cpm"]).sum()),
               other_class_ok=int((o["cls_other"] == 1 - o["cpm"]).sum()),
               same_class_ok=int((o["cls_same"] == o["cpm"]).sum()),
               w_c_same_equal_w_cpm=int((o["w_c_same"] == o["w_cpm"]).sum()),
               gap_cpm_negative=int((o["gap_cpm"] < 0).sum()), shots=int(len(idx)),
               chunks_regenerated_equal_manifest=int(sum(regen)))
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/arb5_d9.npz", idx=idx, **o)
    rec = dict(what="ARB5 scores on the routed shots (arb5_alpine.py d9s)", d=d, network="d9_b1w64_sh", set="d9stest",
               checks=chk, info=info, wall_s=wall, core=core, cpuset=allowed,
               loop_means_us={k: float(o[k].mean()) for k in ("t_pm", "t_split", "t_cpm", "t_c_other")}, **env())
    json.dump(rec, open(f"{a.out}/arb5_d9.json", "w"), indent=1)
    print(json.dumps(dict(d=d, checks=chk, wall_s=round(wall, 1), loop=rec["loop_means_us"])), flush=True)


def timing(a):
    """The loop alone, without the diagnostic decode, on the first --max routed shots at q (from arb5_alpine's own
    record of the distance), one process on one pinned core: the CPU time of the gaps on the paper's CPU."""
    core, allowed = pin(a.core)
    out = {}
    for d in a.d:
        z = np.load(f"{a.out}/arb5_d{d}.npz")
        rec = json.load(open(f"{a.out}/arb5_d{d}.json"))
        q = a.q[str(d)] if isinstance(a.q, dict) else a.q
        tau = rec["tau"][str(q)] if d != 9 else 2.337686538696289
        idx = z["idx"][np.abs(z["c"]) < tau] if d != 9 else z["idx"]
        idx = idx[:a.max]
        chunk = idx // testset.CHUNK
        dets = []
        for j in np.unique(chunk):
            if d == 9:
                f = f"d9_p0.006_X_d9stest_{int(j):03d}.npz"
                det, _ = testset.load(os.path.join(a.shots9, f), d)
            else:
                man = testset.manifest("sv2test")
                f = testset.name(man["split"], d, int(j))
                det, _ = testset.load(f"{E}/shots/sv2test/{f}", d, man["files"][f]["sha256"])
            dets.append(det[idx[chunk == j] - j * testset.CHUNK])
        det = np.concatenate(dets)
        o, info = loop(d, det, diagnostic=False)
        sel = np.searchsorted(z["idx"], idx)
        assert np.array_equal(z["idx"][sel], idx)
        same = dict(gap_pm=int((o["gap_pm"] == z["gap_pm"][sel]).sum()), gap_cpm=int((o["gap_cpm"] == z["gap_cpm"][sel]).sum()))
        out[str(d)] = dict(q=q, tau=tau, shots=int(len(idx)), same_as_main_job=same,
                           mean_us={k: float(o[k].mean()) for k in ("t_pm", "t_split", "t_cpm", "t_c_other")},
                           median_us={k: float(np.median(o[k])) for k in ("t_pm", "t_split", "t_cpm", "t_c_other")})
        np.savez(f"{a.save}/time_d{d}.npz", idx=idx, **{k: o[k] for k in ("t_pm", "t_split", "t_cpm", "t_c_other")})
        print(json.dumps({d: out[str(d)]}), flush=True)
    json.dump(dict(what="ARB5 gap timing (arb5_alpine.py time)", core=core, cpuset=allowed, results=out, **env()),
              open(f"{a.save}/time.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("sv2")
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--core", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("d9s")
    p.add_argument("--routed", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shots", required=True)
    p.add_argument("--core", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("time")
    p.add_argument("--d", type=int, nargs="+", required=True)
    p.add_argument("--q", type=json.loads, required=True, help='JSON: {"7": 0.02, ...}')
    p.add_argument("--max", type=int, default=10000)
    p.add_argument("--shots9", required=True)
    p.add_argument("--core", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--save", required=True)
    a = ap.parse_args()
    {"sv2": sv2, "d9s": d9s, "time": timing}[a.cmd](a)


if __name__ == "__main__":
    main()
