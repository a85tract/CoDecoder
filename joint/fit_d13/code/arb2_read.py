"""ARB at d = 13: the read (PLAN.md step 2). Fresh shots of the d = 13 comparison's circuit at p through both decoders,
streamed chunk by chunk (10,000 shots each), with the per-shot outputs written in parts of 100 chunks. Run through
arb2_shim.py (kind 11 drawn from ARB's seeds), one p per run:

  python arb2_shim.py arb2_read.py --p 0.006 --chunks 0-400 --out OUT/read [--b-path graph|eager] [--resume]

Decoders, from the release's cascade/d13/surface/sv_common.py, unchanged:
  S = ConvLines d13_b5_r775 (sc.load_S: sha256 checked) on the release's deployed Triton engine (sc.s_fn: timing.engine
      FROZEN, fp16 matmul inputs, fp32 accumulation), in batches of 4,096 consecutive shots of the read carried across
      chunk boundaries, the last one zero-padded: the comparison's SB (sc.Queue over sc.Graph at sc.S_BATCH, as
      sv_read builds it). Logit c (fp32), decision c > 0, confidence |c|.
  B = Cascade-surface CS13 final_ema.pt (sc.load_B: the source checkpoint's sha256 checked) in eval mode under bf16
      autocast, uncompiled: sc.b_fn(model, "graph") is the module itself (the comparison's name for its uncompiled path;
      "compiled" would be torch.compile). It runs in batches of --b-batch = 16 consecutive shots (the comparison's b*),
      aligned with the chunks (10,000 = 625 x 16). --b-path eager calls the module directly; --b-path graph replays one
      CUDA graph of the same module at batch 16 (sc.Graph), which launches the same kernels without Python overhead,
      and every part re-runs its first --check-eager shots by direct eager calls and compares the logits bit for bit.
      Logit z (fp32 of the bf16 output), decision z > 0, confidence |z|.
The GPU guard and settings are sc.setup_gpu's (a full RTX PRO 6000 Blackwell Server Edition, TF32 off, cuDNN autotuning
off and deterministic). Nothing here routes or reads a threshold: the composition is assembled by arb2_analysis.py.

Output in OUT/p<p> (sc.p_tag): part_<c0>-<c1>.npz per 100 chunks, with
  obs (n,) uint8, det (n, 273) uint8 = np.packbits(detector bits, axis=1) (big bit order, as sc.digest packs them),
  chunk (n,) uint16, c (n,) float32, sdec (n,) uint8 (the engine's decision), z (n,) float32, bdec (n,) uint8,
  meta: JSON (p, kind, chunks, every chunk's seed as eval_seed gives it in this process and its digest sc.digest).
The journal progress.jsonl (header, then one line per part with its file's sha256) lets a stopped read continue with
--resume from its first unwritten part (the parts already written are checked against their sha256; S's batches then
start at the resume point, which changes no output: the engine decides each shot on its own). At the end
read_<time>.json: settings, identity, nets, files and sha256, timings, the eager checks and the failure counts of S and
B alone (the composition and everything after it is the analysis's)."""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from surface import surface_data as sdata
from surface import sv_common as sc

KIND = sc.KIND_CONF                      # 11: mapped to ARB's seeds by arb2_shim.py


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def jappend(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--chunks", required=True, help="a-b: chunks a .. b-1 of 10,000 shots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--b-path", choices=["graph", "eager"], default="graph")
    ap.add_argument("--b-batch", type=int, default=16)
    ap.add_argument("--check-eager", type=int, default=1024, help="per part: B's first N shots also by direct eager "
                    "calls, compared bit for bit (--b-path graph only)")
    ap.add_argument("--procs", type=int, default=4, help="sampling processes (spawn)")
    ap.add_argument("--ahead", type=int, default=8, help="chunks sampled ahead of the one decoded")
    ap.add_argument("--part", type=int, default=sc.PART)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    import torch
    if not getattr(sdata.eval_seed, "_arb2", None):
        raise SystemExit("arb2_read: eval_seed is not arb2_shim's (run through arb2_shim.py)")
    chunks = sc.chunk_range(a.chunks)
    c0, c1 = chunks.start, chunks.stop
    p = a.p
    out = os.path.join(a.out, sc.p_tag(p))
    os.makedirs(out, exist_ok=True)
    bb = a.b_batch
    if sc.CHUNK % bb:
        raise SystemExit(f"B's batch {bb} does not divide the chunk {sc.CHUNK}")
    t_start = time.time()
    dev, gpu = sc.setup_gpu()
    rec = sc.record(gpu)
    lay = sc.sl.Layout(sc.D)
    Dd = lay.D
    t_build = time.time()
    smod, tabs, scfg, srec = sc.load_S()
    bmod, brec = sc.load_B(dev, "final")
    qS = sc.Queue(sc.Graph(sc.s_fn(tabs, scfg), sc.S_BATCH, Dd, dev))
    fB = sc.b_fn(bmod, "graph")                    # the module itself under bf16 autocast (uncompiled)
    gB = sc.Graph(fB, bb, Dd, dev) if a.b_path == "graph" else None
    t_build = time.time() - t_build
    head = {"p": p, "kind": KIND, "chunks": [c0, c1], "part": a.part, "s_batch": sc.S_BATCH, "b_batch": bb,
            "b_path": a.b_path, "check_eager": a.check_eager if a.b_path == "graph" else 0,
            "nets": {"S": srec["sha256"], "B": brec["sha256"]}, "seeds": sdata.eval_seed._arb2,
            "identity": {k: rec.get(k) for k in ("gpu", "torch", "real_executable", "prefix", "packages_sha256",
                                                    "code")}}
    head = json.loads(json.dumps(head))
    prog = os.path.join(out, "progress.jsonl")
    files, k = {}, c0
    if os.path.exists(prog):
        if not a.resume:
            raise SystemExit(f"{prog} exists: continue with --resume; a read never overwrites")
        recs = [json.loads(ln) for ln in open(prog).read().split("\n") if ln.strip()]
        if recs[0].get("header") != head:
            raise SystemExit(f"{prog}: the stopped read had other settings; no resume")
        for r in recs[1:]:
            if "part" not in r:
                continue
            s, e = r["part"]
            if s != k or sha256_file(os.path.join(out, r["file"])) != r["sha256"]:
                raise SystemExit(f"{prog}: parts not contiguous or a file changed; no resume")
            files[r["file"]] = r["sha256"]
            k = e
    else:
        with open(prog, "x") as fh:
            fh.write(json.dumps({"header": head}) + "\n")
    jappend(prog, {"attempt": {"pid": os.getpid(), "host": rec["host"], "time": rec["time"], "from_chunk": k}})
    P0 = (k - c0) * sc.CHUNK                       # read position of the first shot decoded by this process
    st = {"sample_wait_s": 0.0, "S_s": 0.0, "B_s": 0.0, "eager_check_s": 0.0, "write_s": 0.0, "h2d_s": 0.0}
    chk = {"shots": 0, "z_differ": 0, "dec_differ": 0}
    fails = {"S": 0, "B": 0, "shots": 0}
    parts = {}

    def part_of(j):
        return c0 + ((j - c0) // a.part) * a.part

    def slot(j):
        s = part_of(j)
        if s not in parts:
            e = min(s + a.part, c1)
            n = (e - s) * sc.CHUNK
            parts[s] = {"e": e, "obs": np.zeros(n, np.uint8), "det": np.zeros((n, Dd // 8), np.uint8),
                        "chunk": np.zeros(n, np.uint16), "c": np.full(n, np.nan, np.float32),
                        "sdec": np.full(n, 255, np.uint8), "z": np.full(n, np.nan, np.float32),
                        "bdec": np.full(n, 255, np.uint8), "seeds": {}, "digests": {}, "sdone": 0}
        return parts[s]

    def put_S(idx, lg, dc):
        """S's results at read positions idx (they may span parts)."""
        js = c0 + idx // sc.CHUNK
        ss = c0 + ((js - c0) // a.part) * a.part
        for s in np.unique(ss):
            sel = ss == s
            pa = parts[int(s)]
            loc = idx[sel] - (int(s) - c0) * sc.CHUNK
            pa["c"][loc], pa["sdec"][loc] = lg[sel], dc[sel]
            pa["sdone"] += int(sel.sum())

    def write(s):
        pa = parts.pop(s)
        e = pa["e"]
        if pa["sdone"] != len(pa["obs"]) or (pa["sdec"] == 255).any() or (pa["bdec"] == 255).any():
            raise RuntimeError(f"part {s}: incomplete")
        tw = time.time()
        meta = {"p": p, "kind": KIND, "chunks": [s, e], "seeds": [pa["seeds"][j] for j in range(s, e)],
                "digests": [pa["digests"][j] for j in range(s, e)], "b_path": a.b_path, "b_batch": bb,
                "s_batch": sc.S_BATCH, "nets": head["nets"]}
        name = f"part_{s:05d}-{e:05d}.npz"
        tmp = os.path.join(out, name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.savez(fh, obs=pa["obs"], det=pa["det"], chunk=pa["chunk"], c=pa["c"], sdec=pa["sdec"], z=pa["z"],
                     bdec=pa["bdec"], meta=np.array(json.dumps(meta)))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, os.path.join(out, name))
        sha = sha256_file(os.path.join(out, name))
        files[name] = sha
        fails["S"] += int((pa["sdec"] != pa["obs"]).sum())
        fails["B"] += int((pa["bdec"] != pa["obs"]).sum())
        fails["shots"] += len(pa["obs"])
        jappend(prog, {"part": [s, e], "file": name, "sha256": sha, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        st["write_s"] += time.time() - tw
        print(f"arb2_read: p {p} part {s}-{e} written ({time.time() - t_start:.0f} s)", flush=True)

    def run_B(xg, check):
        """B on one chunk's rows (GPU) in aligned batches of bb -> (z, dec) numpy; the first `check` shots again by
        direct eager calls, compared bit for bit."""
        n = xg.shape[0]
        zt = torch.empty(n, dtype=torch.float32, device=dev)
        dt = torch.empty(n, dtype=torch.uint8, device=dev)
        t = time.time()
        with torch.no_grad():
            for i in range(0, n, bb):
                if gB is not None:
                    gB.x.copy_(xg[i:i + bb])
                    gB.replay()
                    zt[i:i + bb].copy_(gB.logit)
                    dt[i:i + bb].copy_(gB.dec)
                else:
                    lg, dc = fB(xg[i:i + bb])
                    zt[i:i + bb].copy_(lg)
                    dt[i:i + bb].copy_(dc)
        z, dc = zt.cpu().numpy(), dt.cpu().numpy()
        st["B_s"] += time.time() - t
        if check and gB is not None:
            t = time.time()
            m = min(check, n)
            zg = torch.empty(m, dtype=torch.float32, device=dev)
            dg = torch.empty(m, dtype=torch.uint8, device=dev)
            with torch.no_grad():
                for i in range(0, m, bb):                          # direct eager calls, one sync at the end
                    lg, d2 = fB(xg[i:i + bb])
                    zg[i:i + bb].copy_(lg)
                    dg[i:i + bb].copy_(d2)
            ze, de = zg.cpu().numpy(), dg.cpu().numpy()
            chk["shots"] += m
            chk["z_differ"] += int((ze.view(np.uint32) != z[:m].view(np.uint32)).sum())
            chk["dec_differ"] += int((de != dc[:m]).sum())
            st["eager_check_s"] += time.time() - t
        return z, dc

    todo = range(k, c1)
    src = sdata.stream_shots(sc.D, p, todo, n=sc.CHUNK, kind=KIND, procs=a.procs, ahead=a.ahead)
    tw0 = time.time()
    for j in todo:
        jj, det, obs = next(src)
        st["sample_wait_s"] += time.time() - tw0
        assert jj == j and det.shape == (sc.CHUNK, Dd) and obs.shape == (sc.CHUNK, 1)
        pa = slot(j)
        off = (j - part_of(j)) * sc.CHUNK
        sl_ = slice(off, off + sc.CHUNK)
        pa["seeds"][j] = int(sdata.eval_seed(sc.D, KIND, p, j))         # this process's seed of the chunk (traced)
        pa["digests"][j] = sc.digest(det, obs)
        pa["obs"][sl_], pa["chunk"][sl_] = obs[:, 0], j
        pa["det"][sl_] = np.packbits(det, axis=1)
        t = time.time()
        xg = torch.from_numpy(det).pin_memory().to(dev, non_blocking=True)
        st["h2d_s"] += time.time() - t
        z, bd = run_B(xg, a.check_eager if j == part_of(j) else 0)    # the first chunk of every part
        pa["z"][sl_], pa["bdec"][sl_] = z, bd
        t = time.time()
        for ii, lg, dc, _, _ in qS.add(xg, (j - c0) * sc.CHUNK + np.arange(sc.CHUNK)):
            put_S(ii, lg, dc)
        st["S_s"] += time.time() - t
        wm = qS.watermark((j - c0 + 1) * sc.CHUNK)   # the first read position S has not decided
        for s in sorted(parts):
            if (parts[s]["e"] - c0) * sc.CHUNK <= wm:
                write(s)
        tw0 = time.time()
    src.close()                                    # ends the sampling pool
    t = time.time()
    for ii, lg, dc, _, _ in qS.flush():
        put_S(ii, lg, dc)
    st["S_s"] += time.time() - t
    for s in sorted(parts):
        write(s)
    secs = time.time() - t_start
    summ = {"p": p, "kind": KIND, "chunks": [c0, c1], "from_chunk": k, "first_position": P0,
            "shots_this_process": (c1 - k) * sc.CHUNK, "head": head, "record": rec, "nets": {"S": srec, "B": brec},
            "files": files, "secs": round(secs, 1), "build_s": round(t_build, 1),
            "timings": {kk: round(v, 1) for kk, v in st.items()}, "eager_check": chk, "fails_alone": fails,
            "S_batches": qS.batches, "complete": len(files) == -(-(c1 - c0) // a.part)}
    name = os.path.join(out, f"read_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json")
    with open(name, "w") as fh:
        json.dump(summ, fh, indent=1, default=str)
    print("arb2_read:", json.dumps({kk: summ[kk] for kk in ("p", "chunks", "from_chunk", "secs", "timings",
                                                             "eager_check", "fails_alone", "complete")}), flush=True)
    if not summ["complete"] or chk["z_differ"] or chk["dec_differ"]:
        raise SystemExit("arb2_read: incomplete, or the graph's logits differ from the eager calls'")


if __name__ == "__main__":
    main()
