"""Phase D1's compiled reads (PHASE_D1_PLAN.md 2.6): C14' on its frozen timed paths over the shots of D1's eager read.

  read     python d1_compiled.py read --read DIR --p P --out DIR [--lock L] [--streams K --rounds R]
  profile  python d1_compiled.py profile --p P --rows N --grid KxR,KxR,... --out F.jsonl   (dry runs, kind 13)
  verify   python d1_compiled.py verify --p P --rows N --streams K --rounds R --out F.jsonl (dry runs, kind 13)
  anchor   python d1_compiled.py anchor --streams K --rounds R --out F.json                 (dry runs)

read: the detector rows that pc_eval.py saved (--save-det) in DIR/p<p>.npz, each chunk's digest checked against the
read's chunk_meta, then C14' through phase C's compiled path objects (pc_speed.build; the graphs come from the job's
copy of phase C's Triton and Inductor caches):
  batched  compiled@4096 over every row in order, consecutive batches of 4,096 from the block's first row, the last
           padded with zeros (Cascade alone in batched mode); then over the rows the lock's configuration at p routes
           (S55'-deep's read margin below tau1, in float64), in shot order, in batches of 4,096, the last padded (B in
           D's batched mode, as the pipeline queues them)
  single   compiled@1 over every row through Multi: K CUDA graphs on K streams, each holding R consecutive batch-1
           calls of the compiled computation on its own static input, one row per call. Checked bitwise, every
           decision and confidence, against pc_speed.build(C14', "compiled", 1).run (phase C's Graph, one row at a
           time) on the block's first check_first rows before the read and on every check_every-th row after it; one
           differing bit and nothing is written.
Before them, the S anchor: S55'-deep's read engine (pc_eval.py's fused path) on the first 32,768 rows of C4b's frozen
read at p must give that read's every decision and margin bit. Writes OUT/p<p>.npz (bits1, conf1, bits4, conf4,
routed, bitsR, confR; the bits packed) and OUT/p<p>.json, then OUT/done_<p>. Nothing here computes a failure: the
analysis (d1_confirm.py) is the only step that compares a decision with the observables.

profile, verify and anchor are the dry run's (PHASE_D1_PLAN.md 5.3): Multi's rate at each K x R against phase C's
Graph one row at a time; Multi bitwise against phase C's Graph on N rows of the dry kind 13; and the anchor, C14' on
compiled@1 (Multi) and compiled@4096 on the rows of the first lock's claim-point check (C4b's frozen read), whose
counts for C14' alone and for configurations 2 and 7 must equal that check's (phase_c/out/lock_1e8d1c6d)."""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")   # hardware queues for Multi's streams (no effect on numerics)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase_c"))
sys.path.insert(0, HERE)
import d1_common as d1  # noqa: E402
import pc_common as pc  # noqa: E402

D = 432
S_ANCHOR_ROWS = 32_768


def emit(path, row):
    row = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(),
           "session": os.environ.get("SLURM_JOB_ID", ""), **row}
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def cascade(dev, L):
    """C14' as phase C's speed steps load it (bb.cascade_eval.Predictor, EMA weights, bf16), refused unless it is the
    locked network."""
    from bb.cascade_eval import Predictor
    pr = Predictor(pc.NETS[d1.B][0], dev, "ema", "bf16")
    if pr.sig["sha256"] != L["nets"][d1.B]:
        raise SystemExit(f"{d1.B}: sha256 differs from the locked network")
    return pr


def compiled_fn(pr):
    """Phase C's compiled computation (pc_speed.build's "compiled" path before its graph capture)."""
    import torch
    import pc_speed as ps
    return torch.compile(ps.make_fwd(pr), dynamic=False)


def cache_counters():
    from torch._dynamo.utils import counters
    return {k: int(v) for k, v in counters["inductor"].items() if "cache" in k}


class Multi:
    """K CUDA graphs on K streams. Graph k holds R consecutive calls of fn, the compiled batch-1 computation, each on
    graph k's static input x (1, 432): call r copies row r of the graph's input block into x and writes its decision
    and confidence to row r of the graph's output blocks. Each graph is warmed up and captured on its own stream, so
    that every stream has its own cuBLAS workspace."""

    def __init__(self, fn, K, R, dev):
        import torch
        self.K, self.R = K, R
        self.st = [torch.cuda.Stream() for _ in range(K)]
        self.inp = [torch.zeros(R, D, dtype=torch.uint8, device=dev) for _ in range(K)]
        self.x = [torch.zeros(1, D, dtype=torch.uint8, device=dev) for _ in range(K)]
        self.ob = [torch.zeros(R, pc.K, dtype=torch.uint8, device=dev) for _ in range(K)]
        self.oc = [torch.zeros(R, dtype=torch.float32, device=dev) for _ in range(K)]
        self.g = []
        torch.cuda.synchronize()
        for k in range(K):
            s = self.st[k]
            with torch.no_grad(), torch.cuda.stream(s):
                for _ in range(3):
                    fn(self.x[k])
            s.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, stream=s):
                for r in range(R):
                    self.x[k].copy_(self.inp[k][r:r + 1])
                    bits, conf = fn(self.x[k])
                    self.ob[k][r:r + 1].copy_(bits)
                    self.oc[k][r:r + 1].copy_(conf)
            self.g.append(g)
        torch.cuda.synchronize()

    def run(self, dg):
        """Rows dg (n, 432) uint8 on the device -> (bits (n, 12) uint8, conf (n,) float32) on the device; row a goes to
        graph (a // R) % K, call a % R."""
        import torch
        n, K, R = len(dg), self.K, self.R
        m = -(-n // (K * R)) * K * R
        bits = torch.empty(m, pc.K, dtype=torch.uint8, device=dg.device)
        conf = torch.empty(m, dtype=torch.float32, device=dg.device)
        main = torch.cuda.current_stream()
        for s in self.st:
            s.wait_stream(main)
        for lo in range(0, n, K * R):
            for k in range(K):
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
        for s in self.st:
            main.wait_stream(s)
        return bits[:n], conf[:n]


def to_np(bits, conf):
    return bits.cpu().numpy(), conf.float().cpu().numpy()


def bitwise(b, c, b_ref, c_ref):
    """Rows whose decision differs and confidences whose bit pattern differs."""
    return {"rows": int(len(b)), "differ": int((b != b_ref).any(1).sum()),
            "conf_bits_differ": int((d1.u32(c) != d1.u32(c_ref)).sum())}


def ref_run(ref1, rows_np, dev):
    """Phase C's compiled@1 Graph, one row at a time, on host rows."""
    import torch
    return to_np(*ref1.run(torch.as_tensor(rows_np, device=dev)))


def sliced(fn, det_packed, idx, slice_rows, dev):
    """fn over the rows det_packed[idx] (idx None: every row), slice by slice on the device -> numpy bits, conf."""
    import torch
    n = len(det_packed) if idx is None else len(idx)
    bits, conf = np.empty((n, pc.K), np.uint8), np.empty(n, np.float32)
    for lo in range(0, n, slice_rows):
        hi = min(n, lo + slice_rows)
        rows = det_packed[lo:hi] if idx is None else det_packed[idx[lo:hi]]
        b, c = fn(torch.as_tensor(d1.unpack_det(rows), device=dev))
        bits[lo:hi], conf[lo:hi] = to_np(b, c)
    return bits, conf


def s_anchor(dev, L, p):
    """S55'-deep's read engine (pc_eval.py: the fused engine at FUSED_READ_B shots per call) against C4b's frozen read
    at p: every decision and margin bit of the first S_ANCHOR_ROWS rows."""
    import torch
    import pc_eval  # noqa: F401  (patches bb.cascade_eval.Predictor: ConvLines networks on the fused engine)
    from bb.cascade_eval import Predictor
    pr = Predictor(pc.NETS[d1.S][0], dev, "ema", "bf16")
    if pr.sig["sha256"] != L["nets"][d1.S] or pr.prec != L["read_prec"][d1.S]:
        raise SystemExit(f"{d1.S}: not the locked network or precision")
    z, meta, idx = pc.load_read(d1.C4B_READ, p)
    det = d1.unpack_det(z["det"][:S_ANCHOR_ROWS])
    out = pr(torch.as_tensor(det, device=dev))
    b, c = out["bits"].cpu().numpy(), out["conf"].float().cpu().numpy()
    rb = np.unpackbits(z[f"pred{idx[d1.S]}"][:S_ANCHOR_ROWS], axis=1, count=pc.K)
    r = bitwise(b, c, rb, z[f"conf{idx[d1.S]}"][:S_ANCHOR_ROWS])
    r["pass"] = r["differ"] == 0 and r["conf_bits_differ"] == 0
    return r


def read(a):
    import torch
    import pc_speed as ps
    t_start = time.time()
    L = d1.lock(a.lock)
    p = a.p
    ci, c = d1.config_at(L, p)
    cr = L["compiled_read"]
    K, R = a.streams or cr["streams"], a.rounds or cr["rounds"]
    SL, C1, CE = cr["slice_rows"], cr["check_first"], cr["check_every"]
    if not (K and R) or SL % (K * R) or SL % pc.VERDICT_B:
        raise SystemExit(f"streams {K}, rounds {R}, slice {SL}: the slice must be a multiple of K x R and of 4,096")
    os.makedirs(a.out, exist_ok=True)
    done = os.path.join(a.out, f"done_{p}")
    if os.path.exists(done):
        raise SystemExit(f"{done} exists: the compiled read at {p} is done")
    z, meta, idx = pc.load_read(a.read, p)
    if meta["family"] != L["family"] or meta["kind"] != L["kind_seed"]:
        raise SystemExit(f"{a.read} at {p}: family {meta['family']} kind {meta['kind']}, not the lock's")
    for t in (d1.B, d1.S):
        if t not in idx or meta["models"][idx[t]]["sha256"] != L["nets"][t]:
            raise SystemExit(f"{a.read} at {p}: {t} missing or not the locked network")
    det_p = z["det"]
    n = len(det_p)
    ok, nch = d1.chunk_digests(det_p, z["chunk"], meta)
    if not ok:
        raise SystemExit(f"{a.read} at {p}: the saved detector rows do not match the read's chunk digests")
    t_dig = time.time()
    dev, env = ps.setup()
    sa = s_anchor(dev, L, p)
    if not sa["pass"]:
        raise SystemExit(f"the S anchor failed at {p}: {sa}")
    pr = cascade(dev, L)
    t0 = time.time()
    multi = Multi(compiled_fn(pr), K, R, dev)
    ref1 = ps.build(pr, "compiled", 1)
    g4 = ps.build(pr, "compiled", pc.VERDICT_B)
    t_build = time.time() - t0
    run1 = lambda dg: multi.run(dg)                                       # noqa: E731
    run4 = lambda dg: g4.run(dg)                                          # noqa: E731

    t0 = time.time()                                                     # the check before the read
    first = d1.unpack_det(det_p[:min(n, C1)])
    b_m, c_m = to_np(*multi.run(torch.as_tensor(first, device=dev)))
    b_r, c_r = ref_run(ref1, first, dev)
    chk_first = bitwise(b_m, c_m, b_r, c_r)
    chk_first["secs"] = round(time.time() - t0, 1)
    if chk_first["differ"] or chk_first["conf_bits_differ"]:
        raise SystemExit(f"Multi is not bitwise phase C's compiled@1 Graph on the first rows: {chk_first}")

    t0 = time.time()
    bits4, conf4 = sliced(run4, det_p, None, SL, dev)                     # Cascade alone, batched
    t4 = time.time() - t0
    conf_s = z[f"conf{idx[d1.S]}"].astype(np.float64)
    routed = np.nonzero(conf_s < c["tau"]["tau1"])[0]
    t0 = time.time()
    bitsR, confR = sliced(run4, det_p, routed, SL, dev)                   # B in D's batched mode
    tR = time.time() - t0
    t0 = time.time()
    bits1, conf1 = sliced(run1, det_p, None, SL, dev)                     # single: every row
    t1 = time.time() - t0

    t0 = time.time()                                                     # the check after the read
    every = np.arange(0, n, CE)
    b_r, c_r = ref_run(ref1, d1.unpack_det(det_p[every]), dev)
    chk_every = bitwise(bits1[every], conf1[every], b_r, c_r)
    chk_every["secs"] = round(time.time() - t0, 1)
    if chk_every["differ"] or chk_every["conf_bits_differ"]:
        raise SystemExit(f"Multi is not bitwise phase C's compiled@1 Graph on every {CE}-th row: {chk_every}")

    sha, tag = d1.tree()
    rec = {"p": p, "family": meta["family"], "kind": meta["kind"], "chunks": meta["chunks"], "shots": int(n),
           "n_chunks": nch, "digests_ok": True, "config": ci, "tau1": c["tau"]["tau1"], "routed": int(len(routed)),
           "net": d1.B, "sha256": pr.sig["sha256"], **env, "tree_sha": sha, "tree": tag,
           "read": os.path.abspath(a.read), "read_json_sha256": pc.sha256_file(os.path.join(a.read, f"{pc.p_tag(p)}.json")),
           "lock": os.path.abspath(a.lock), "lock_sha256": pc.sha256_file(a.lock),
           "paths": dict(d1.COMPILED_PATHS), "frozen_paths": L["frozen_paths"][d1.B],
           "impl": {"single": f"Multi: {K} streams x {R} batch-1 calls per graph", "streams": K, "rounds": R,
                    "slice_rows": SL, "batched": "pc_speed.build(C14p, 'compiled', 4096).run"},
           "check_first": chk_first, "check_every": dict(chk_every, every=CE), "S_anchor": sa,
           "cache": cache_counters(), "caches": {k: os.environ.get(k, "") for k in ("TRITON_CACHE_DIR",
                                                                                     "TORCHINDUCTOR_CACHE_DIR")},
           "secs": {"digests": round(t_dig - t_start, 1), "build": round(t_build, 1), "batched": round(t4, 1),
                    "routed": round(tR, 1), "single": round(t1, 1), "total": round(time.time() - t_start, 1)},
           "us_per_row": {"single": 1e6 * t1 / n, "batched": 1e6 * t4 / n}}
    np.savez(os.path.join(a.out, f"{pc.p_tag(p)}.npz"), bits1=np.packbits(bits1, axis=1), conf1=conf1,
             bits4=np.packbits(bits4, axis=1), conf4=conf4, routed=routed, bitsR=np.packbits(bitsR, axis=1),
             confR=confR)
    json.dump(rec, open(os.path.join(a.out, f"{pc.p_tag(p)}.json"), "w"), indent=1)
    open(done, "w").close()
    emit(None, {k: v for k, v in rec.items() if k not in ("frozen_paths",)})


def dry_rows(p, n, procs=8):
    """n shots of the dry kind 13 at p (chunks from 0): never reported."""
    from bb import cascade_data as cdata
    det, _ = cdata.eval_shots(pc.CODE, p, range(-(-n // pc.CHUNK)), kind=pc.KIND_DRY, procs=procs, family=d1.FAMILY)
    return np.ascontiguousarray(det[:n], dtype=np.uint8)


def profile(a):
    """Multi's time per row at each K x R of --grid, and phase C's Graph one row at a time, on dry rows."""
    import torch
    import pc_speed as ps
    L = d1.lock(a.lock)
    det = dry_rows(a.p, a.rows)
    dev, env = ps.setup()
    pr = cascade(dev, L)
    fn = compiled_fn(pr)
    ref1 = ps.build(pr, "compiled", 1)
    nref = min(2_000, a.rows)
    t0 = time.time()
    b_r, c_r = ref_run(ref1, det[:nref], dev)
    emit(a.out, {"what": "profile", "impl": "phase C Graph, one row at a time", "rows": nref,
                 "us_per_row": 1e6 * (time.time() - t0) / nref, "p": a.p, **env})
    dg = torch.as_tensor(det, device=dev)
    for kr in a.grid.split(","):
        K, R = (int(x) for x in kr.split("x"))
        nr = min(len(det), max(nref, -(-2_000 * K // (K * R)) * K * R))      # about 2,000 rows per stream
        t0 = time.time()
        m = Multi(fn, K, R, dev)
        t_build = time.time() - t0
        m.run(dg[:K * R])
        torch.cuda.synchronize()
        t0 = time.time()
        b, c = m.run(dg[:nr])
        torch.cuda.synchronize()
        dt = time.time() - t0
        b, c = to_np(b, c)
        emit(a.out, {"what": "profile", "impl": "Multi", "streams": K, "rounds": R, "rows": nr,
                     "us_per_row": 1e6 * dt / nr, "build_s": round(t_build, 1),
                     "vs_ref": bitwise(b[:nref], c[:nref], b_r, c_r), "p": a.p,
                     "max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")})
        del m
        torch.cuda.empty_cache()
    emit(a.out, {"what": "profile", "cache": cache_counters()})


def verify(a):
    """Multi at K x R bitwise against phase C's compiled@1 Graph on --rows dry rows, and against itself (twice)."""
    import torch
    import pc_speed as ps
    L = d1.lock(a.lock)
    det = dry_rows(a.p, a.rows)
    dev, env = ps.setup()
    pr = cascade(dev, L)
    m = Multi(compiled_fn(pr), a.streams, a.rounds, dev)
    ref1 = ps.build(pr, "compiled", 1)
    dg = torch.as_tensor(det, device=dev)
    t0 = time.time()
    b1, c1 = to_np(*m.run(dg))
    t_m = time.time() - t0
    b2, c2 = to_np(*m.run(dg))
    t0 = time.time()
    b_r, c_r = ref_run(ref1, det, dev)
    t_r = time.time() - t0
    r = {"what": "verify", "p": a.p, "kind": pc.KIND_DRY, "streams": a.streams, "rounds": a.rounds, **env,
         "vs_ref": bitwise(b1, c1, b_r, c_r), "repeat": bitwise(b2, c2, b1, c1),
         "us_per_row": {"multi": 1e6 * t_m / len(det), "ref": 1e6 * t_r / len(det)}, "cache": cache_counters()}
    r["pass"] = not any(r[k]["differ"] or r[k]["conf_bits_differ"] for k in ("vs_ref", "repeat"))
    emit(a.out, r)


def anchor(a):
    """C14' on compiled@1 (Multi) and compiled@4096 (phase C's Graph) on the rows of the first lock's claim-point
    check (C4b's frozen read), against that check's counts: C14' alone (single: compiled@1, amortized: compiled@4096)
    and configurations 2 and 7 in both modes."""
    import torch
    import pc_speed as ps
    L = d1.lock(a.lock)
    first = d1.first_lock(L)
    chk = json.load(open(d1.CHECK_FIRST))
    dev, env = ps.setup()
    pr = cascade(dev, L)
    m = Multi(compiled_fn(pr), a.streams, a.rounds, dev)
    g4 = ps.build(pr, "compiled", pc.VERDICT_B)
    rows, ok = [], True
    for p in pc.RELAY_P:
        z, meta, idx = pc.load_read(d1.C4B_READ, p)
        det = d1.unpack_det(z["det"])
        obs = np.unpackbits(z["obs"], axis=1, count=pc.K)
        dg = torch.as_tensor(det, device=dev)
        b1, c1 = to_np(*m.run(dg))
        b4, c4 = to_np(*g4.run(dg))
        pe = np.unpackbits(z[f"pred{idx[d1.B]}"], axis=1, count=pc.K)
        ce, fe = z[f"conf{idx[d1.B]}"], z[f"fail{idx[d1.B]}"].astype(bool)
        for mode, path, b, cf in (("single", "compiled@1", b1, c1), ("amortized", "compiled@4096", b4, c4)):
            mine = {"differ": int((b != pe).any(1).sum()), "conf_differ": int((cf != ce).sum()),
                    "fail": ps._cmp((b != obs).any(1), fe)}
            theirs = [r for r in chk["rows"] if r["what"] == "alone" and r["net"] == d1.B and r["p"] == p
                      and r["mode"] == mode][0]
            same = mine["differ"] == theirs["differ"] and mine["conf_differ"] == theirs["conf_differ"] and \
                mine["fail"] == theirs["fail"]
            ok &= same
            rows.append({"p": p, "what": "alone", "mode": mode, "path": path, "mine": mine,
                         "check": {k: theirs[k] for k in ("differ", "conf_differ", "fail")}, "match": same})
        fs = z[f"fail{idx[d1.S]}"].astype(bool)
        cs = z[f"conf{idx[d1.S]}"].astype(np.float64)
        for i in L["source"]["configs"]:
            c = first["configs"][i]
            if c["p"] != p:
                continue
            toB = cs < c["tau"]["tau1"]
            ii = np.nonzero(toB)[0]
            fde = fs.copy()
            fde[toB] = fe[toB]
            bR, _ = to_np(*g4.run(dg[torch.as_tensor(ii, device=dev)]))
            for mode, frouted in (("single", (b1[ii] != obs[ii]).any(1)), ("batched", (bR != obs[ii]).any(1))):
                fd = fs.copy()
                fd[ii] = frouted
                mine = {"fail": ps._cmp(fd, fde), "to_B": int(toB.sum())}
                theirs = [r for r in chk["rows"] if r["what"] == "routed" and r.get("config") == i
                          and r["mode"] == mode][0]
                same = mine["fail"] == theirs["fail"] and mine["to_B"] == theirs["routing"]["to_B"]["timed"]
                ok &= same
                rows.append({"p": p, "what": "routed", "config": i, "mode": mode, "mine": mine,
                             "check": {"fail": theirs["fail"], "to_B": theirs["routing"]["to_B"]["timed"]},
                             "match": same})
    out = {"what": "anchor", "check": d1.CHECK_FIRST, "read": d1.C4B_READ, "streams": a.streams,
           "rounds": a.rounds, **env, "rows": rows, "all_match": bool(ok), "cache": cache_counters()}
    json.dump(out, open(a.out, "w"), indent=1)
    emit(None, {"what": "anchor", "all_match": bool(ok), "rows": len(rows)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("read")
    s.add_argument("--read", required=True)
    s.add_argument("--p", type=float, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--lock", default=d1.LOCK)
    s.add_argument("--streams", type=int, default=0)
    s.add_argument("--rounds", type=int, default=0)
    s = sub.add_parser("profile")
    s.add_argument("--p", type=float, default=0.003)
    s.add_argument("--rows", type=int, default=65_536)
    s.add_argument("--grid", default="1x1,4x16,8x16,16x16,32x16,64x16,16x64,32x32")
    s.add_argument("--lock", default=d1.LOCK)
    s.add_argument("--out", default="")
    s = sub.add_parser("verify")
    s.add_argument("--p", type=float, default=0.003)
    s.add_argument("--rows", type=int, default=100_000)
    s.add_argument("--streams", type=int, required=True)
    s.add_argument("--rounds", type=int, required=True)
    s.add_argument("--lock", default=d1.LOCK)
    s.add_argument("--out", default="")
    s = sub.add_parser("anchor")
    s.add_argument("--streams", type=int, required=True)
    s.add_argument("--rounds", type=int, required=True)
    s.add_argument("--lock", default=d1.LOCK)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"read": read, "profile": profile, "verify": verify, "anchor": anchor}[a.cmd](a)


if __name__ == "__main__":
    main()
