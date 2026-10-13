"""X2 surface: per-shot quantities of the registered test reads at d = 13 (T2) and d = 17 (T6), one chunk of 10,000
shots per task, written to $X2_OUT/surface/d{d}/c{j:03d}.npz (a finished chunk is skipped, so a rerun resumes).

  python surface_chunk.py --d 13 --chunks 0-99 --procs 32
  python surface_chunk.py --d 17 --chunks 0 --procs 1 --smoke      # timings, checks and the PyMatching docs

Per chunk: rebuild the shots from the manifest seed and check the file bytes' sha256; join the records (each checked
against its provenance); re-decode with plain and correlated PyMatching and check the recorded predictions; the class
margins on the split-boundary graph (x2lib); syndrome counts for every shot; and, for every shot where the network or
correlated matching fails plus every 32nd shot, the edges of the least-weight correction in the true and the wrong
class (plain matching on the split graph), from which the structure of the difference between the two classes is
measured. The correlated class weights are kept as a secondary quantity: each forced correlated decode reweights its
edges after its own first pass, so the lower of the two need not be correlated PM's own class (smoke test: 117 of
10,000 shots at d = 13), while the plain weights are exact (their minimum is plain PM's solution weight)."""
import argparse
import os
import sys
import time

import numpy as np

import x2lib as X

ENV = {}


def init(d):
    import pymatching
    files, csha = X.manifest(d)
    c = X.circuit(d, csha)
    geo = X.Geometry(d)
    assert geo.m == c.num_detectors, f"grid has {geo.m} detectors, circuit {c.num_detectors}"
    dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    sdem, info = X.split_dem(dem, geo)
    edges, kind = X.graph_edges(dem, geo)
    ENV.update(d=d, files=files, c=c, geo=geo, info=info, edges=edges, kind=kind,
               m_plain=pymatching.Matching.from_detector_error_model(dem),
               m_corr=pymatching.Matching.from_detector_error_model(dem, enable_correlations=True),
               s_plain=pymatching.Matching.from_detector_error_model(sdem),
               s_corr=pymatching.Matching.from_detector_error_model(sdem, enable_correlations=True))


def pair_counts(det, edges, kind, block=2000):
    """(n, 3) fired X-detector pairs joined by a DEM edge: time-like, space-like, space-time."""
    out = np.zeros((len(det), 3), np.int32)
    for k in range(3):
        e = edges[kind == k]
        for a in range(0, len(det), block):
            b = det[a:a + block]
            out[a:a + block, k] = (b[:, e[:, 0]] & b[:, e[:, 1]]).sum(1)
    return out


def structure(geo, det_row, st, sw):
    """Features of the true-class correction st and of the difference D = st ^ sw (edge sets): 21 numbers."""
    m = geo.m

    def kinds(s):
        n = np.zeros(4, np.int32)
        for u, v in s:
            if (u < m and not geo.is_x[u]) or (0 <= v < m and not geo.is_x[v]):
                continue                                       # Z-type part: no bearing on the X observable
            n[X.edge_kind(geo, u, v)] += 1
        return n

    def max_run(s):
        """Most time-like edges at one site."""
        cnt = {}
        for u, v in s:
            if 0 <= v < m and u < m and geo.is_x[u] and X.edge_kind(geo, u, v) == 0:
                key = (geo.i[u], geo.j[u])
                cnt[key] = cnt.get(key, 0) + 1
        return max(cnt.values()) if cnt else 0

    D = st ^ sw
    nodes = sorted({x for e in D for x in e if 0 <= x < m and geo.is_x[x]})
    nodes = np.array(nodes, np.int64)
    dec = nodes[det_row[nodes] == 1] if len(nodes) else nodes        # decisive defects: fired, touched by D
    t_ext = int(geo.t[nodes].max() - geo.t[nodes].min()) if len(nodes) else -1
    if len(dec):
        bd = geo.bdist[dec]
        dec_f = [len(dec), bd.mean(), bd.min(), bd.max(), int(geo.t[dec].max() - geo.t[dec].min()),
                 int(geo.k[dec].max() - geo.k[dec].min()), int(geo.j[dec].max() - geo.j[dec].min())]
    else:
        dec_f = [0, np.nan, -1, -1, -1, -1, -1]
    return np.array([*kinds(st), max_run(st), *kinds(D), max_run(D), len(nodes), t_ext, *dec_f], np.float64)


STRUCT_NAMES = ("st_time", "st_space", "st_spacetime", "st_boundary", "st_maxrun",
                "D_time", "D_space", "D_spacetime", "D_boundary", "D_maxrun", "D_nodes", "D_text",
                "dec_n", "dec_bdist_mean", "dec_bdist_min", "dec_bdist_max", "dec_text", "dec_rowspan", "dec_colspan")


def run_chunk(j, out_dir, smoke=False):
    d, geo = ENV["d"], ENV["geo"]
    path = os.path.join(out_dir, f"c{j:03d}.npz")
    if os.path.exists(path):
        return j, "skip", 0.0
    T = {}
    t0 = time.time()
    name, seed, sha = ENV["files"][j]
    det, obs = X.rebuild(ENV["c"], seed, sha)
    T["rebuild"] = time.time() - t0
    rec = X.records(d, j, sha)
    for part in X.PARTS:
        assert np.array_equal(rec[part]["obs"].astype(np.uint8), obs), f"chunk {j}: {part} obs differ"
    c = rec["ours_deployed"]["c"].astype(np.float32)
    tau = X.TESTS[d]["tau"]
    routed = np.abs(c.astype(np.float64)) < tau
    net = (c > 0).astype(np.uint8)
    cpm = rec["pm_corr"]["pred"].astype(np.uint8).reshape(-1)
    pm = rec["pm"]["pred"].astype(np.uint8).reshape(-1)
    pair = np.where(routed, cpm, net).astype(np.uint8)
    # the record's own routing (d = 13: the registered tau; d = 17: study 4's tau, which the executable ran at)
    tau_rec = 1.8486117124557495 if d == 13 else 5.121974468231201
    rr = np.abs(c.astype(np.float64)) < tau_rec
    chk = dict(rec_routed_eq=int(np.array_equal(rr, rec["ours_deployed"]["routed"].astype(bool))),
               rec_pred_eq=int(np.array_equal(np.where(rr, cpm, net), rec["ours_deployed"]["pred"].astype(np.uint8))))
    t0 = time.time()
    n = len(obs)
    cpm_re, wc = ENV["m_corr"].decode_batch(det, return_weights=True, enable_correlations=True)
    T["corr"] = time.time() - t0
    pm_re, wp = ENV["m_plain"].decode_batch(det, return_weights=True)
    T["plain"] = time.time() - t0 - T["corr"]
    chk.update(cpm_mismatch=int((cpm_re[:, 0] != cpm).sum()), pm_mismatch=int((pm_re[:, 0] != pm).sum()))
    t0 = time.time()
    a0 = np.hstack([det, np.zeros((n, 1), np.uint8)])
    a1 = np.hstack([det, np.ones((n, 1), np.uint8)])
    cp0, cw0 = ENV["s_corr"].decode_batch(a0, return_weights=True, enable_correlations=True)
    cp1, cw1 = ENV["s_corr"].decode_batch(a1, return_weights=True, enable_correlations=True)
    pp0, pw0 = ENV["s_plain"].decode_batch(a0, return_weights=True)
    pp1, pw1 = ENV["s_plain"].decode_batch(a1, return_weights=True)
    T["forced"] = time.time() - t0
    cp0, cp1, pp0, pp1 = cp0[:, 0], cp1[:, 0], pp0[:, 0], pp1[:, 0]
    chk.update(corr_class_split_bad=int((cp0 == cp1).sum()), plain_class_split_bad=int((pp0 == pp1).sum()),
               plain_min_dev=float(np.abs(np.minimum(pw0, pw1) - wp).max()),
               corr_argmin_vs_cpm=int((np.where(cw0 <= cw1, cp0, cp1) != cpm).sum()),
               plain_argmin_vs_pm=int((np.where(pw0 <= pw1, pp0, pp1) != pm).sum()))
    cw_true = np.where(cp0 == obs, cw0, cw1)
    cw_wrong = np.where(cp0 == obs, cw1, cw0)
    pw_true = np.where(pp0 == obs, pw0, pw1)
    pw_wrong = np.where(pp0 == obs, pw1, pw0)
    t0 = time.time()
    is_x = geo.is_x
    n_x = det[:, is_x].sum(1).astype(np.int32)
    n_det = det.sum(1).astype(np.int32)
    pairs = pair_counts(det, ENV["edges"], ENV["kind"])
    nb = (d - 1) // 2
    bd_hist = np.stack([det[:, is_x & (geo.bdist == b)].sum(1) for b in range(1, nb + 1)], 1).astype(np.int32)
    T["syndrome"] = time.time() - t0
    t0 = time.time()
    fail_net, fail_cpm = net != obs, cpm != obs
    sel = np.nonzero(fail_net | fail_cpm | (np.arange(n) % X.REF_EVERY == 0))[0]
    if smoke:
        sel = sel[:200]
    feats = np.zeros((len(sel), len(STRUCT_NAMES)), np.float64)
    for r, i in enumerate(sel):
        a_true = 0 if pp0[i] == obs[i] else 1
        syn = (a0[i], a1[i])
        st = X.edge_set(ENV["s_plain"].decode_to_edges_array(syn[a_true]))
        sw = X.edge_set(ENV["s_plain"].decode_to_edges_array(syn[1 - a_true]))
        feats[r] = structure(geo, det[i], st, sw)
    T["structure"] = time.time() - t0
    T["structure_per_shot_ms"] = 1e3 * T["structure"] / max(len(sel), 1)
    arrays = dict(j=np.int32(j), seed=np.int64(seed), obs=obs, c=c, routed=routed, net=net, cpm=cpm, pm=pm, pair=pair,
                  n_det=n_det, n_x=n_x, pairs=pairs, bd_hist=bd_hist,
                  cw_true=cw_true, cw_wrong=cw_wrong, pw_true=pw_true, pw_wrong=pw_wrong, wc=wc, wp=wp,
                  sel=sel.astype(np.int32), feats=feats, feat_names=np.array(STRUCT_NAMES),
                  check_names=np.array(sorted(chk)), check_vals=np.array([float(chk[k]) for k in sorted(chk)]))
    if not smoke:
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    return j, chk, T


def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--chunks", default="0-99")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(X.OUT, "surface", f"d{a.d}")
    os.makedirs(out_dir, exist_ok=True)
    chunks = parse(a.chunks)
    t0 = time.time()
    if a.smoke:
        import pymatching
        init(a.d)
        print(f"init {time.time() - t0:.1f} s; split: {ENV['info']}; X edges by kind (time, space, spacetime):",
              np.bincount(ENV["kind"], minlength=3).tolist())
        print("decode_batch doc:", " | ".join(ln.strip() for ln in (pymatching.Matching.decode_batch.__doc__ or "")
                                             .split("\n") if "weight" in ln)[:900])
        for j in chunks:
            print(run_chunk(j, out_dir, smoke=True))
        return
    todo = [j for j in chunks if not os.path.exists(os.path.join(out_dir, f"c{j:03d}.npz"))]
    print(f"d = {a.d}: {len(chunks)} chunks, {len(todo)} to do, {a.procs} processes", flush=True)
    bad = []
    if a.procs == 1:
        init(a.d)
        it = map(lambda j: run_chunk(j, out_dir), todo)
    else:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(a.procs, initializer=init, initargs=(a.d,))
        it = pool.imap_unordered(_job, [(j, out_dir) for j in todo])
    for k, (j, chk, T) in enumerate(it):
        flags = {x: v for x, v in chk.items() if (x.endswith("_eq") and v != 1) or (x == "plain_min_dev" and v > 1e-3)
                 or (x in ("cpm_mismatch", "pm_mismatch", "corr_class_split_bad", "plain_class_split_bad") and v != 0)}
        if flags:
            bad.append((j, flags))
        print(f"[{k + 1}/{len(todo)}] chunk {j} {time.time() - t0:.0f} s; "
              f"{ {x: round(v, 1) for x, v in T.items()} } {'FLAGS ' + str(flags) if flags else 'ok'}", flush=True)
    print(f"done d = {a.d} in {time.time() - t0:.0f} s; chunks with flags: {len(bad)}")
    for x in bad[:20]:
        print("  ", x)
    sys.exit(0)


def _job(args):
    return run_chunk(*args)


if __name__ == "__main__":
    main()
