"""ARBCP: Cascade's compiled paths on the re-test's routed shots of Table I (BB72, D1), with the logits, and the frozen
joint decision A3 on each path (exploratory follow-up of ARB; the re-test's compiled files hold decisions only).

  python cp_run.py --p 0.002 --data DIR --out DIR --c14 C14p.pt --s55 S55p_deep.pt --lock d1_locked_depspam_fm.json

DIR/read/p<p>.npz and DIR/compiled/p<p>.npz are the re-test's eager read and compiled read (d1_compiled.py read).
Routed: S55'-deep's read margin (float64 of the read's float32) below D1's tau1, in shot order (= the compiled read's
routed). On the routed rows only:
  single   C14' compiled@1 through d1_compiled.Multi (32 streams x 16 batch-1 calls per CUDA graph), the re-test's
           single path, on every routed row
  batched  C14' compiled@4096 (pc_speed.Graph) over the routed rows in shot order, consecutive batches of 4,096 from the
           first routed row, the last padded with zeros: the re-test's bitsR (B in D's batched mode, as the pipeline
           queues them), which d1_confirm.py uses for the composition's batched failures
The compiled computation is pc_speed.make_fwd's (bits = z > 0, conf = min |z|) with the logits z as one more output.
Kernels: D1's seed caches (Triton, Inductor), copied into the job. Checks (counted, nothing refused): every routed row's
decision and confidence bitwise against the re-test's bits1 / conf1 and bitsR / confR; before that, on the first
--diag rows, the unmodified computation (pc_speed.build's Graph at 1 and 4,096) against the same saved values and
against the computation with z.
Scores: S55'-deep's 4,096 class scores from the fused engine (ARB's arb_fused.Scored, with arb_scores.py's checks) on
the routed rows where S55'-deep's decision differs from C14''s on any path (eager, single, batched; mine or saved).
A3 (frozen, ARB): on a path's disagreement rows answer argmax_y [ s(y) + w (y . z) ], float64 on the GPU as
arb_analysis.a3_answers, w = 10^(32/40) at 0.2 %, 10^(23/40) at 0.3 %; eager with the read's float16 logits z0.
Writes OUT/p<p>_logits.npz (the routed rows' logits on both paths), OUT/p<p>_routed.npz (per routed row: classes of
the observables, S55'-deep, C14' on each path, A3 on each path), OUT/p<p>_scores.npy + _scores_rows.npy and OUT/p<p>.json
(checks, kept-shot counts, timings)."""
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")       # as d1_compiled.py (Multi's streams)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pc_common as pc  # noqa: E402
import pc_speed as ps  # noqa: E402
import d1_common as d1  # noqa: E402
import pc_eval  # noqa: E402,F401  (ConvLines on the fused engine; the GPU guard)
from bb import cascade_eval as ce  # noqa: E402
import arb_fused  # noqa: E402

TAU1 = {0.002: 7.3331888389587405, 0.003: 7.60439167022705}
WK = {0.002: 32, 0.003: 23}                                       # w = 10 ** (k / 40), ARB's grid
W_GIVEN = {0.002: 6.309573444801933, 0.003: 3.758374042884441}
K, NY, D = 12, 4096, 432
PW = (1 << np.arange(K)).astype(np.int64)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def u32(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def cls_of_packed(packed):
    return np.unpackbits(packed, axis=1, count=K).astype(np.int64) @ PW


def cls_of_bits(bits):
    return bits.astype(np.int64) @ PW


def cache_counters():
    from torch._dynamo.utils import counters
    return {k: int(v) for k, v in counters["inductor"].items() if "cache" in k}


def make_fwd3(pr):
    """pc_speed.make_fwd(pr, "minabs") for Cascade, returning the logits too."""
    def fwd(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = pr.model(x).float()
            a = z.abs()
            conf = a.min(1).values
            return (z > 0).to(torch.uint8), conf, z
    return fwd


class Multi3:
    """d1_compiled.Multi with the logits as one more output: K CUDA graphs on K streams, graph k holding R consecutive
    batch-1 calls on its static input; row a goes to graph (a // R) % K, call a % R."""

    def __init__(self, fn, K_, R, dev):
        self.K, self.R = K_, R
        self.st = [torch.cuda.Stream() for _ in range(K_)]
        self.inp = [torch.zeros(R, D, dtype=torch.uint8, device=dev) for _ in range(K_)]
        self.x = [torch.zeros(1, D, dtype=torch.uint8, device=dev) for _ in range(K_)]
        self.ob = [torch.zeros(R, K, dtype=torch.uint8, device=dev) for _ in range(K_)]
        self.oc = [torch.zeros(R, dtype=torch.float32, device=dev) for _ in range(K_)]
        self.oz = [torch.zeros(R, K, dtype=torch.float32, device=dev) for _ in range(K_)]
        self.g = []
        torch.cuda.synchronize()
        for k in range(K_):
            s = self.st[k]
            with torch.no_grad(), torch.cuda.stream(s):
                for _ in range(3):
                    fn(self.x[k])
            s.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, stream=s):
                for r in range(R):
                    self.x[k].copy_(self.inp[k][r:r + 1])
                    bits, conf, z = fn(self.x[k])
                    self.ob[k][r:r + 1].copy_(bits)
                    self.oc[k][r:r + 1].copy_(conf)
                    self.oz[k][r:r + 1].copy_(z)
            self.g.append(g)
        torch.cuda.synchronize()

    def run(self, dg):
        n, K_, R = len(dg), self.K, self.R
        m = -(-n // (K_ * R)) * K_ * R
        bits = torch.empty(m, K, dtype=torch.uint8, device=dg.device)
        conf = torch.empty(m, dtype=torch.float32, device=dg.device)
        z = torch.empty(m, K, dtype=torch.float32, device=dg.device)
        main = torch.cuda.current_stream()
        for s in self.st:
            s.wait_stream(main)
        for lo in range(0, n, K_ * R):
            for k in range(K_):
                a = lo + k * R
                h = max(0, min(R, n - a))
                with torch.cuda.stream(self.st[k]):
                    if h < R:
                        self.inp[k].zero_()
                    if h:
                        self.inp[k][:h].copy_(dg[a:a + h])
                    self.g[k].replay()
                    bits[a:a + R].copy_(self.ob[k])
                    conf[a:a + R].copy_(self.oc[k])
                    z[a:a + R].copy_(self.oz[k])
        for s in self.st:
            main.wait_stream(s)
        return bits[:n], conf[:n], z[:n]


class Graph3:
    """pc_speed.Graph with the logits as one more output (B rows per replay, the last batch padded with zeros)."""

    def __init__(self, fn, B, dev):
        self.B = B
        self.x = torch.zeros(B, D, dtype=torch.uint8, device=dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(st):
            for _ in range(3):
                fn(self.x)
        torch.cuda.current_stream().wait_stream(st)
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g):
            self.bits, self.conf, self.z = fn(self.x)

    def run(self, det):
        ob, oc, oz = [], [], []
        for i in range(0, len(det), self.B):
            d = det[i:i + self.B]
            self.x.zero_()
            self.x[:len(d)].copy_(d)
            self.g.replay()
            ob.append(self.bits[:len(d)].clone())
            oc.append(self.conf[:len(d)].clone())
            oz.append(self.z[:len(d)].clone())
        return torch.cat(ob), torch.cat(oc), torch.cat(oz)


def sliced(fn, det_packed, idx, slice_rows, dev, nout):
    """fn over the rows det_packed[idx], slice by slice (as d1_compiled.sliced) -> numpy outputs."""
    n = len(idx)
    outs = None
    for lo in range(0, n, slice_rows):
        hi = min(n, lo + slice_rows)
        r = fn(torch.as_tensor(d1.unpack_det(det_packed[idx[lo:hi]]), device=dev))
        r = [t.cpu().numpy() for t in r]
        if outs is None:
            outs = [np.empty((n,) + t.shape[1:], t.dtype) for t in r]
        for o, t in zip(outs, r):
            o[lo:hi] = t
    assert len(outs) == nout
    return outs


def compare(b, c, b_ref_packed, c_ref):
    bp = np.packbits(b, axis=1)
    return {"rows": int(len(b)), "decisions_differ": int((bp != b_ref_packed).any(1).sum()),
            "conf_bits_differ": int((u32(c) != u32(c_ref)).sum())}


def a3_answers(sc, z, w, block=16384, dev=None):
    """argmax_y [ s(y) + w (y . z) ] -> int64 classes (arb_analysis.a3_answers at one w, float64 on the GPU); also the
    gap between the best and the second-best objective (near-tie diagnostics)."""
    n = len(z)
    out = np.empty(n, np.int64)
    gap = np.empty(n, np.float64)
    Y = ((np.arange(NY)[:, None] >> np.arange(K)) & 1).astype(np.float64)
    Yt = torch.as_tensor(Y.T.copy(), device=dev)
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        S = torch.as_tensor(np.asarray(sc[lo:hi]), device=dev).double()
        Yz = torch.as_tensor(np.ascontiguousarray(z[lo:hi], dtype=np.float64), device=dev) @ Yt
        J = S + w * Yz
        a = torch.argmax(J, 1)
        top2 = torch.topk(J, 2, dim=1).values
        out[lo:hi] = a.cpu().numpy()
        gap[lo:hi] = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    return out, gap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--c14", required=True)
    ap.add_argument("--s55", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--streams", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--slice", type=int, default=1_048_576)
    ap.add_argument("--diag", type=int, default=8192)
    a = ap.parse_args()
    t_start = time.time()
    p, tag = a.p, pc.p_tag(a.p)
    os.makedirs(a.out, exist_ok=True)
    rec = {"p": p, "tag": tag}
    secs = {}
    w = 10 ** (WK[p] / 40)
    if w != W_GIVEN[p]:
        raise SystemExit(f"w {w!r} is not the frozen {W_GIVEN[p]!r}")
    L = d1.lock(a.lock)
    _, c = d1.config_at(L, p)
    tau1 = float(c["tau"]["tau1"])
    if tau1 != TAU1[p] or c["S"] != "S55p_deep" or c["B"] != "C14p":
        raise SystemExit(f"lock {a.lock}: configuration at {p} is not D1's")
    rec.update(w=w, tau1=tau1, lock_sha256=sha256_file(a.lock))
    rd, cd = os.path.join(a.data, "read"), os.path.join(a.data, "compiled")
    meta = json.load(open(os.path.join(rd, f"{tag}.json")))
    cmeta = json.load(open(os.path.join(cd, f"{tag}.json")))
    if [m["kind"] for m in meta["models"]] != ["cascade", "convlines"]:
        raise SystemExit("the read's models are not (C14', S55'-deep)")
    z = np.load(os.path.join(rd, f"{tag}.npz"))
    cz = np.load(os.path.join(cd, f"{tag}.npz"))
    t0 = time.time()
    conf_s = z["conf1"]
    routed = np.nonzero(conf_s.astype(np.float64) < tau1)[0]
    if not np.array_equal(routed, cz["routed"]):
        raise SystemExit("the routed rows are not the compiled read's")
    nr = len(routed)
    det_p = z["det"]
    n = len(det_p)
    obs, pred0, pred1 = z["obs"], z["pred0"], z["pred1"]
    fail0, fail1 = z["fail0"].astype(bool), z["fail1"].astype(bool)
    chunk = z["chunk"]
    z0 = z["z0"]
    bits1_s, conf1_s = cz["bits1"], cz["conf1"]
    bits4_s = cz["bits4"]
    bitsR_s, confR_s = cz["bitsR"], cz["confR"]
    secs["load"] = round(time.time() - t0, 1)
    rec.update(shots=int(n), routed=int(nr), compiled_json_routed=cmeta["routed"])
    print(f"{tag}: {n:,} shots, routed {nr:,}", flush=True)

    # ---- kept shots and Cascade alone on every shot (the re-test's decisions)
    yO = cls_of_packed(obs)
    keep = np.ones(n, bool)
    keep[routed] = False
    fC = {"eager": fail0, "single": cls_of_packed(bits1_s) != yO, "batched": cls_of_packed(bits4_s) != yO}
    if not np.array_equal(fail0, cls_of_packed(pred0) != yO) or not np.array_equal(fail1, cls_of_packed(pred1) != yO):
        raise SystemExit("the read's failures are not its decisions against the observables")
    rec["kept"] = {"shots": int(keep.sum()), "S_fail": int((fail1 & keep).sum())}
    rec["alone"] = {}
    for path, f in fC.items():
        rec["alone"][path] = {"fail": int(f.sum()), "kept_S_only": int((fail1 & ~f & keep).sum()),
                              "kept_C_only": int((~fail1 & f & keep).sum()), "kept_both": int((fail1 & f & keep).sum())}
    del fC

    # ---- C14' compiled
    dev, env = ps.setup()
    rec["env"] = env
    pr = ce.Predictor(a.c14, dev, "ema", "bf16")
    rec["c14"] = {"path": os.path.abspath(a.c14), "file_sha256": pr.sig["sha256"],
                  "read_sha256": meta["models"][0]["sha256"], "kind": pr.kind}
    if pr.kind != "cascade":
        raise SystemExit(f"{a.c14}: not a Cascade network")
    t0 = time.time()
    ref1 = ps.build(pr, "compiled", 1)                                  # the unmodified computation (diagnostics)
    g4 = ps.build(pr, "compiled", pc.VERDICT_B)
    rec["cache_after_unmodified"] = cache_counters()
    multi = Multi3(torch.compile(make_fwd3(pr), dynamic=False), a.streams, a.rounds, dev)   # one compiled object per
    g43 = Graph3(torch.compile(make_fwd3(pr), dynamic=False), pc.VERDICT_B, dev)           # batch, as pc_speed.build
    rec["cache_after_with_z"] = cache_counters()
    secs["build"] = round(time.time() - t0, 1)
    print("built", json.dumps({"cache_unmodified": rec["cache_after_unmodified"],
                               "cache_with_z": rec["cache_after_with_z"]}), flush=True)

    # diagnostics on the first --diag routed rows: unmodified vs saved, with z vs unmodified
    t0 = time.time()
    nd = min(a.diag, nr)
    dd = torch.as_tensor(d1.unpack_det(det_p[routed[:nd]]), device=dev)
    b_u1, c_u1 = (t.cpu().numpy() for t in ref1.run(dd))
    b_u4, c_u4 = (t.cpu().numpy() for t in g4.run(dd))
    b_z1, c_z1, _ = (t.cpu().numpy() for t in multi.run(dd))
    b_z4, c_z4, _ = (t.cpu().numpy() for t in g43.run(dd))
    rec["diag"] = {"rows": int(nd),
                   "single_unmodified_vs_saved": compare(b_u1, c_u1, bits1_s[routed[:nd]], conf1_s[routed[:nd]]),
                   "batched_unmodified_vs_saved": compare(b_u4, c_u4, bitsR_s[:nd], confR_s[:nd]),
                   "single_with_z_vs_unmodified": compare(b_z1, c_z1, np.packbits(b_u1, axis=1), c_u1),
                   "batched_with_z_vs_unmodified": compare(b_z4, c_z4, np.packbits(b_u4, axis=1), c_u4)}
    del dd, ref1, g4
    torch.cuda.empty_cache()
    secs["diag"] = round(time.time() - t0, 1)
    print("diag", json.dumps(rec["diag"]), flush=True)

    t0 = time.time()
    bB, cB, zB = sliced(g43.run, det_p, routed, a.slice, dev, 3)       # batched: the routed queue in shot order
    secs["batched"] = round(time.time() - t0, 1)
    t0 = time.time()
    b1, c1, z1 = sliced(multi.run, det_p, routed, a.slice, dev, 3)     # single
    secs["single"] = round(time.time() - t0, 1)
    del multi, g43
    torch.cuda.empty_cache()
    rec["check"] = {"single_vs_bits1": compare(b1, c1, bits1_s[routed], conf1_s[routed]),
                    "batched_vs_bitsR": compare(bB, cB, bitsR_s, confR_s),
                    "batched_vs_bits4_on_routed": {"decisions_differ": int((np.packbits(bB, axis=1)
                                                                            != bits4_s[routed]).any(1).sum())},
                    "single_z_consistent": bool(np.array_equal((z1 > 0).astype(np.uint8), b1)
                                                and np.array_equal(u32(np.abs(z1).min(1)), u32(c1))),
                    "batched_z_consistent": bool(np.array_equal((zB > 0).astype(np.uint8), bB)
                                                 and np.array_equal(u32(np.abs(zB).min(1)), u32(cB)))}
    print("check", json.dumps(rec["check"]), flush=True)
    np.savez(os.path.join(a.out, f"{tag}_logits.npz"), routed=routed, z_single=z1, z_batched=zB,
             bits_single=np.packbits(b1, axis=1), conf_single=c1, bits_batched=np.packbits(bB, axis=1),
             conf_batched=cB)

    # ---- classes on the routed rows
    yO_r = yO[routed]
    yN = cls_of_packed(pred1[routed])
    y = {"eager": cls_of_packed(pred0[routed]), "single": cls_of_bits(b1), "batched": cls_of_bits(bB),
         "single_saved": cls_of_packed(bits1_s[routed]), "batched_saved": cls_of_packed(bitsR_s),
         "batched_whole_saved": cls_of_packed(bits4_s[routed])}
    zz = {"eager": z0[routed].astype(np.float64), "single": z1.astype(np.float64), "batched": zB.astype(np.float64)}
    if not np.array_equal((zz["eager"] > 0).astype(np.int64) @ PW, y["eager"]):
        rec["eager_z0_sign_differs"] = int(((zz["eager"] > 0).astype(np.int64) @ PW != y["eager"]).sum())
    differ_any = np.zeros(nr, bool)
    for v in y.values():
        differ_any |= v != yN
    U = np.nonzero(differ_any)[0]                                      # positions among the routed rows
    rows = routed[U]
    rec["score_rows"] = int(len(U))
    print(f"scores on {len(U):,} routed rows", flush=True)

    # ---- S55'-deep's class scores (arb_scores.py's procedure and checks)
    t0 = time.time()
    ps_ = ce.Predictor(a.s55, dev, "ema", "bf16")
    rec["s55"] = {"path": os.path.abspath(a.s55), "file_sha256": ps_.sig["sha256"],
                  "read_sha256": meta["models"][1]["sha256"], "kind": ps_.kind}
    if ps_.kind != "convlines":
        raise SystemExit(f"{a.s55}: not a ConvLines network")
    dg = torch.as_tensor(d1.unpack_det(det_p[rows]), device=dev)
    orig = arb_fused.run_with_programs(ps_.fused, dg)
    eng = arb_fused.Scored(ps_.model, pc.FUSED_READ_B)
    exp = arb_fused.run_with_programs(eng, dg, scores=True)
    del dg
    bits_read = np.unpackbits(pred1[rows], axis=1, count=K)
    ck = {"batched_bits_equal_read": bool(np.array_equal(orig["bits"].numpy(), bits_read)),
          "batched_margin_equal_read": bool(np.array_equal(u32(orig["margin"].numpy()), u32(conf_s[rows])))}
    for k in ("bits", "margin", "m1", "i1", "m2"):
        x_, y_ = exp[k].numpy(), orig[k].numpy()
        if x_.dtype == np.float32:
            x_, y_ = u32(x_), u32(y_)
        ck[f"scored_{k}_equal_batched"] = bool(np.array_equal(x_, y_))
    ck["read_cls_equal_bits"] = bool(np.array_equal(z["cls1"][rows].astype(np.int64), yN[U]))
    sc = exp["sc"].numpy()
    m1, i1, m2 = exp["m1"].numpy(), exp["i1"].numpy(), exp["m2"].numpy()
    bad = {"program_m1": 0, "program_i1": 0, "program_m2": 0, "argmax_cls": 0, "margin": 0}
    by = eng.NY // eng.NP
    for lo in range(0, len(U), 16384):
        s = sc[lo:lo + 16384]
        g = s.reshape(len(s), eng.NP, by)
        pm1, pi1 = g.max(2), g.argmax(2)
        g2 = g.copy()
        np.put_along_axis(g2, pi1[:, :, None], -np.inf, axis=2)
        pm2 = g2.max(2)
        sl = slice(lo, lo + len(s))
        bad["program_m1"] += int((u32(pm1) != u32(m1[sl])).any(1).sum())
        bad["program_i1"] += int(((pi1 + np.arange(eng.NP) * by) != i1[sl]).any(1).sum())
        bad["program_m2"] += int((u32(pm2) != u32(m2[sl])).any(1).sum())
        am = s.argmax(1)
        bad["argmax_cls"] += int((am != yN[U][sl]).sum())
        top1 = s[np.arange(len(s)), am]
        s2 = s.copy()
        s2[np.arange(len(s)), am] = -np.inf
        bad["margin"] += int((u32(top1 - s2.max(1)) != u32(conf_s[rows[sl]])).sum())
    ck["scores_mismatches"] = bad
    ck["pass"] = all(v for k, v in ck.items() if isinstance(v, bool)) and not any(bad.values())
    rec["scores_check"] = ck
    secs["scores"] = round(time.time() - t0, 1)
    print("scores", json.dumps(ck), flush=True)
    np.save(os.path.join(a.out, f"{tag}_scores.npy"), sc)
    np.save(os.path.join(a.out, f"{tag}_scores_rows.npy"), rows)
    if not ck["pass"]:
        json.dump(rec, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
        raise SystemExit("the score export failed its checks: nothing else is computed")

    # ---- A3 on each path's own disagreement rows
    t0 = time.time()
    pos = np.full(nr, -1, np.int64)
    pos[U] = np.arange(len(U))
    ans, gaps = {}, {}
    for path in ("eager", "single", "batched", "single_saved", "batched_saved"):
        zp = zz[path.split("_")[0]]
        dpos = np.nonzero(y[path] != yN)[0]
        A = y[path].copy()
        g = np.full(nr, np.nan)
        if len(dpos):
            out, gp = a3_answers(sc[pos[dpos]], zp[dpos], w, dev=dev)
            A[dpos] = out
            g[dpos] = gp
        ans[path], gaps[path] = A, g
    secs["a3"] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(a.out, f"{tag}_routed.npz"), routed=routed, chunk=chunk[routed], yO=yO_r, yN=yN,
                        **{f"y_{k}": v for k, v in y.items()}, **{f"a3_{k}": v for k, v in ans.items()},
                        **{f"gap_{k}": v for k, v in gaps.items()})
    # quick counts (the full tables come from cp_count.py)
    fN_kept = int((fail1 & keep).sum())
    q = {}
    for path in ("eager", "single", "batched"):
        comp = fN_kept + int((y[path] != yO_r).sum())
        joint = fN_kept + int((ans[path] != yO_r).sum())
        q[path] = {"composition": comp, "joint": joint}
    rec["quick"] = q
    secs["total"] = round(time.time() - t_start, 1)
    rec["secs"] = secs
    rec["torch"], rec["gpu"] = torch.__version__, torch.cuda.get_device_name(dev)
    json.dump(rec, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    print("quick", json.dumps(q), "secs", json.dumps(secs), flush=True)


if __name__ == "__main__":
    main()
