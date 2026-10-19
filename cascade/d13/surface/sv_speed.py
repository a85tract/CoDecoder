"""Speed of the evaluation (cascade_compare/SURFACE_PLAN.md section 5: phase C's final protocol) and the bitwise checks
of section 4.4 (and item 8 of the independent check: the same check before the lock).

  screen     python -m surface.sv_speed screen --net S|B --rep I --out rows.jsonl --save DIR
  summarize  python -m surface.sv_speed summarize --rows rows.jsonl --out speed.json
  pipeline   python -m surface.sv_speed pipeline --config C1|C2|C3|C4|C5 --p P --mode single|batched --rep I \
                 --lock locked.json --out rows.jsonl --save DIR [--chunks a-b] [--n N]
  bitwise    python -m surface.sv_speed bitwise --lock locked.json --pipes DIR --read DIR --out check.json

screen (step C3, before the development reads): one network in one fresh process; the job runs S and B REPS = 3 times,
interleaved. Paths: S on the release's engine in a CUDA graph ("engine"); B on "graph" (the module in a CUDA graph) and
"compiled" (torch.compile in its default mode, then a CUDA graph); every batch size of BATCHES. Per path and batch size:
the outputs on the screen's shots (kind 12, SCREEN_CHUNKS, at both p; 100,000 shots per p, 20,000 below batch 64) and
their sha256 (the reproducibility check across processes); as a diagnostic, the shots whose decision or confidence differ
from the network's path at 4,096 (S: engine@4096, B: compiled@4096) and phase C's agrees on the failures at 0.6 % and on
the routed sets at each quantile of Q1 and R2 at both p; at batch 1 the GPU term (200 warm-up replays, 5 x 100 timed
with CUDA events) and the host round trip (--n-rt calls after 200 warm-up: pinned H2D copy of one shot, replay, D2H copy
of logit and decision, synchronise: mean, p50, p99, p99.9, max); at every batch size the amortized time per shot (at
least --secs after 10 warm-up replays; input resident on the GPU, "dev", and copied host to host, "h2h").
M alone is timed by pipeline --config C3 in the same job.

summarize: per network and path and batch size the slowest process's values, over the processes that completed the
screen (their "end" row), each process counted once (job, host, pid); a (path, batch) is reproducible when at least
REPS = 3 such processes ran it and their outputs have one sha256. Frozen: S single engine@1, S batched engine@4096; B
single = the reproducible path at batch 1 with the lowest single-shot round trip, B amortized = the reproducible (path,
batch <= 4,096) with the lowest host-to-host time per shot (its batch is b*). No reproducible candidate: refused, with
nothing frozen and no summary written (finding 4 of the stage gate); so are M's rows from fewer than three processes.
The summary names the card, driver, venv and code identity and the kernel-cache digest; the lock carries them.

pipeline (step C7, and the gate before the lock): a configuration end to end at p on kind-12 shots with the lock's
thresholds, one fresh process per repetition; --stage gate or c7 takes the chunks, sizes and repetitions from the lock's
plan (C3, before the lock: --chunks, --n). The graphs are built, and with a lock the kernel caches compared with its
digest, before any shot is drawn. single: --n shots (20,000) after 200 untimed, one at a time: S at batch 1,
then B at batch 1 if routed, then M's decode in this process if routed; C2: B alone on its single path; C3: M alone.
Latency per shot from the pinned row to the decision on the host. batched: --n shots (500,000): S over batches of 4,096
(the last one zero-padded), routed rows queue on the GPU and B runs them at b* (item 9; the last batch padded), M runs
in W_M worker processes with pre-built matchers (decode_batch on bit-packed rows); C2: B alone at b*; C3: M alone over
the workers. Wall time from the first copy to the last decision, per shot and per round (/ 13). Excluded: Stim
sampling, matcher construction, graph capture. Each process saves its logits, stages and decisions on its timed shots.

pipeline --joint-w W (C5 only): the joint decision (sv_joint.py) on the routed shots whose two decisions differ, inside
the timed part: B runs as sv_joint.JointB13 (B's computation on its frozen path, then the joint step, one CUDA graph per
replay); in batched mode S's logits and decisions queue with the routed rows. The row carries sv_joint.record(W) and
path_B ends in "+joint"; pred holds the decisions with the joint step and b_log B's logits.

bitwise: every saved pipeline output of a stage (--expect gate: e3, before the lock commit; c7: e7) against the read
computations on the same shots (sv_read on the pipelines' chunks in a separate job: S1, B1, MS on the single mode's
shots in READ/single, SB, Bb, BQ1, BQ5, MB on the batched mode's in READ/batched): S's logits against S1 (single) / SB
(batched), B's against B1 / BQ1 / BQ5 (C2: B1 / Bb) on the shots that reached B, M's decisions against MS / MB, and
every decision and stage against sv_common.assemble; the pipeline's kernel-cache digest at its end and the reads' must
be the lock's (nothing compiled anew since the lock). Refused (finding 5 of the stage gate) unless every output of the
lock's plan for the stage is there (C1, C2, C4, C5 at both p in both modes, each repetition) and no other, each made
with this lock, its environment and code, on the plan's chunks, shots and seed kind, by a process of its own, all of
one job; and unless the reads are complete reads of the same kind and lock in other jobs. A mode passes when every
process of every configuration at both p agrees bit for bit. The output lists per pipeline file its sha256 and the
timing it carries, which the report takes (sv_route report)."""
import argparse
import glob
import json
import multiprocessing as mp
import os
import platform
import time

import numpy as np

from surface import sv_common as sc

ROUTE_Q = tuple(sorted(set(sc.Q1) | set(sc.R2)))
REF = {"S": ("engine", sc.S_BATCH), "B": ("compiled", 4096)}
M_TASK = 256
MODES = ("single", "batched")


def proc(r):
    """The process a row comes from."""
    return r.get("job"), r.get("host"), r.get("pid")


def emit(path, row):
    row = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(),
           "job": os.environ.get("SLURM_JOB_ID", ""), "pid": os.getpid(), **row}
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("record", "route")})[:1500], flush=True)


def quant(ts):
    ts = np.asarray(ts, np.float64)
    return {"mean": float(ts.mean()), "p50": float(np.quantile(ts, 0.5)), "p99": float(np.quantile(ts, 0.99)),
            "p999": float(np.quantile(ts, 0.999)), "max": float(ts.max()), "n": int(len(ts))}


def gpu_us(gr, warm=200, reps=5, calls=100):
    import torch
    for _ in range(warm):
        gr.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(calls):
            gr.replay()
        e1.record()
        e1.synchronize()
        out.append(1e3 * e0.elapsed_time(e1) / calls)
    return out


def host_rt(gr, rows, n=10_000, warm=200):
    """Per-call host wall time (us) of the single-shot round trip, cycling through pinned rows."""
    import torch
    import time as _t
    H = torch.as_tensor(rows[:4096]).pin_memory()
    hl, hd = torch.empty(1, dtype=torch.float32).pin_memory(), torch.empty(1, dtype=torch.uint8).pin_memory()
    st = torch.cuda.current_stream()
    ts = np.empty(n)
    for i in range(n + warm):
        t = _t.perf_counter()
        gr.x.copy_(H[i % len(H):i % len(H) + 1], non_blocking=True)
        gr.replay()
        hl.copy_(gr.logit, non_blocking=True)
        hd.copy_(gr.dec, non_blocking=True)
        st.synchronize()
        if i >= warm:
            ts[i - warm] = 1e6 * (_t.perf_counter() - t)
    return ts


def amortized(gr, rows, secs=2.0, warm=10):
    """{dev, h2h}: per-shot wall time (us) over consecutive batches, the input resident / copied host to host."""
    import torch
    import time as _t
    B = gr.n
    nb = max(1, min(8, len(rows) // B))
    H = torch.as_tensor(rows[:nb * B]).reshape(nb, B, -1).pin_memory() if len(rows) >= B else \
        torch.as_tensor(rows[:1]).repeat(B, 1).reshape(1, B, -1).pin_memory()
    nb = len(H)
    hl = torch.empty(B, dtype=torch.float32).pin_memory()
    hd = torch.empty(B, dtype=torch.uint8).pin_memory()
    st = torch.cuda.current_stream()
    gr.x.copy_(H[0])
    out = {}
    for mode in ("dev", "h2h"):
        for _ in range(warm):
            gr.replay()
        st.synchronize()
        m, t0 = 0, _t.perf_counter()
        while True:
            if mode == "h2h":
                gr.x.copy_(H[m % nb], non_blocking=True)
            gr.replay()
            if mode == "h2h":
                hl.copy_(gr.logit, non_blocking=True)
                hd.copy_(gr.dec, non_blocking=True)
                st.synchronize()
            m += 1
            if m % 8 == 0:
                st.synchronize()
                if _t.perf_counter() - t0 >= secs:
                    break
        st.synchronize()
        out[mode] = 1e6 * (_t.perf_counter() - t0) / (m * B)
    return out


def run_all(gr, xg):
    """The path over rows xg (on the GPU) in consecutive batches of its size, the last one zero-padded."""
    lg, dc = [], []
    for i in range(0, len(xg), gr.n):
        a, b = gr.run_rows(xg[i:i + gr.n])
        lg.append(a)
        dc.append(b)
    return np.concatenate(lg), np.concatenate(dc)


def out_sha(lg, dc):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(lg, np.float32).tobytes() + np.ascontiguousarray(dc).tobytes()
                          ).hexdigest()[:16]


def screen(a):
    import torch
    dev, gpu = sc.setup_gpu()
    kind = sc.kind_of("speed", a.dry)
    ch = sc.chunk_range(a.chunks)
    rec = sc.record(gpu)
    lay = sc.sl.Layout(sc.D)
    sh = {p: sc.sdata.eval_shots(sc.D, p, ch, n=sc.CHUNK, kind=kind, procs=4) for p in sc.PS}
    n_big, n_small = (int(x) for x in a.n_agree.split(","))
    xg = {p: torch.as_tensor(sh[p][0][:n_big], device=dev) for p in sc.PS}
    obs = {p: sh[p][1][:n_big, 0] for p in sc.PS}
    if a.net == "S":
        _, tabs, scfg, net = sc.load_S()
        makers = {"engine": lambda: sc.s_fn(tabs, scfg)}
    else:
        bmod, net = sc.load_B(dev)
        cf = sc.b_fn(bmod, "compiled")
        makers = {"graph": lambda: sc.b_fn(bmod, "graph"), "compiled": lambda: cf}
    base = {"role": "net", "net": a.net, "rep": a.rep, "kind_seed": kind, "chunks": [ch.start, ch.stop],
            "sha256": net["sha256"], "record": rec}
    os.makedirs(a.save, exist_ok=True)
    batches = [int(x) for x in a.batches.split(",")]
    rp, rb = REF[a.net]
    rb = min(rb, max(batches))                   # below 4,096 only on a MIG slice's code check (B's 4,096 needs 96 GB)
    t = time.time()
    g = sc.Graph(makers[rp](), rb, lay.D, dev)
    ref = {p: run_all(g, xg[p]) for p in sc.PS}
    del g
    torch.cuda.empty_cache()
    emit(a.out, {**base, "metric": "reference", "path": rp, "batch": rb, "secs": round(time.time() - t, 1)})
    for path, mk in makers.items():
        for B in batches:
            t = time.time()
            try:
                g = sc.Graph(mk(), B, lay.D, dev)
            except Exception as e:                       # recorded; the other paths go on
                emit(a.out, {**base, "metric": "error", "path": path, "batch": B, "error": repr(e)[:400]})
                continue
            row = {**base, "path": path, "batch": B, "build_s": round(time.time() - t, 1)}
            n = n_small if B < 64 else n_big
            t = time.time()
            res, sha, route, diff = {}, {}, {}, {}
            for p in sc.PS:
                lg, dc = run_all(g, xg[p][:n])
                res[p] = (lg, dc)
                sha[str(p)] = out_sha(lg, dc)
                rl, rd = ref[p][0][:n], ref[p][1][:n]
                diff[str(p)] = {"dec": int((dc != rd).sum()), "conf": int((np.abs(lg) != np.abs(rl)).sum())}
                route[str(p)] = {}
                ra, ta = np.abs(rl).astype(np.float64), np.abs(lg).astype(np.float64)
                for qq in ROUTE_Q:
                    tau = float(np.quantile(ra, qq))
                    me, mt = ra < tau, ta < tau
                    u, v = int((mt & ~me).sum()), int((~mt & me).sum())
                    route[str(p)][str(qq)] = [int(me.sum()), u, v, bool(agrees(u, v, int(me.sum())))]
            ft, fe = res[sc.P_HIGH][1] != obs[sc.P_HIGH][:n], ref[sc.P_HIGH][1][:n] != obs[sc.P_HIGH][:n]
            u, v = int((ft & ~fe).sum()), int((~ft & fe).sum())
            sc.save_npz(os.path.join(a.save, f"{a.net}_{path}_{B}_rep{a.rep}.npz"),
                        {**base, "path": path, "batch": B, "n": n},
                        **{f"logit_{sc.p_tag(p)}": res[p][0] for p in sc.PS},
                        **{f"dec_{sc.p_tag(p)}": res[p][1] for p in sc.PS})
            emit(a.out, {**row, "metric": "agree", "shots": n, "out_sha256": sha, "differ": diff,
                         "fail_timed": int(ft.sum()), "fail_ref": int(fe.sum()), "only_timed": u, "only_ref": v,
                         "agrees_fail": bool(agrees(u, v, int(fe.sum()))), "route": route,
                         "secs": round(time.time() - t, 1)})
            tm = {}
            rows = sh[sc.P_HIGH][0]
            if B == 1:
                tm["gpu_us"] = gpu_us(g)
                tm["host_rt"] = quant(host_rt(g, rows, n=a.n_rt))
            tm["amortized"] = amortized(g, rows, secs=a.secs)
            emit(a.out, {**row, "metric": "time", **tm})
            del g
            torch.cuda.empty_cache()
    emit(a.out, {**base, "metric": "end", "kcache_end": sc.kcache()})


def agrees(only_t, only_e, n_e):
    """Phase C's accuracy tolerance (pc_common.agrees): |only_t - only_e| <= 0.01 n_e + 3 sqrt(only_t + only_e)."""
    return abs(only_t - only_e) <= 0.01 * n_e + 3.0 * (only_t + only_e) ** 0.5


def summarize(a):
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    ident = {}
    for r in rows:
        rec = r.get("record")
        if rec:
            ident.setdefault(json.dumps([rec.get(k) for k in sc.IDENTITY], sort_keys=True), rec)
    if len(ident) != 1:
        raise SystemExit(f"rows from {len(ident)} environments or code versions")
    rec = next(iter(ident.values()))
    sc.check_analysis(rec, "summarize")            # finding 3: under the screen's code and venv only
    out = {"identity": {k: rec.get(k) for k in sc.IDENTITY + ("driver", "cuda", "cudnn", "triton", "tree")},
           "drivers": sorted({r["record"].get("driver", "") for r in rows if r.get("record")}),
           "kcache": sc.kcache(), "rows": os.path.abspath(a.rows), "rows_sha256": sc.sha256_file(a.rows),
           "nets": {}, "frozen": {}, "components": {}, "M": {}, "flags": []}
    for net in ("S", "B"):
        rs = [r for r in rows if r.get("role") == "net" and r["net"] == net]
        if not rs:
            raise SystemExit(f"no screen rows of {net}")
        out["nets"][net] = {"sha256": rs[0]["sha256"], "paths": {}}
        ended = {proc(r) for r in rs if r["metric"] == "end"}       # the processes that completed the screen
        shas, vals = {}, {}
        for r in rs:
            if proc(r) not in ended:
                continue
            k = (r.get("path"), r.get("batch"))
            if r["metric"] == "agree":
                shas.setdefault(k, {})[proc(r)] = json.dumps(r["out_sha256"], sort_keys=True)
            elif r["metric"] == "time":
                vals.setdefault(k, {})[proc(r)] = r
        for (path, B), tv in sorted(vals.items()):
            hs = shas.get((path, B), {})
            ps = sorted(set(tv) & set(hs), key=str)                 # distinct processes with outputs and timing
            if not ps:
                continue
            tr = [tv[x] for x in ps]
            v = {"processes": len(ps), "reproducible": len(ps) >= sc.REPS and len({hs[x] for x in ps}) == 1,
                 "a_us": max(x["amortized"]["h2h"] for x in tr), "a_dev_us": max(x["amortized"]["dev"] for x in tr)}
            if B == 1:
                v.update(t1_us=max(x["host_rt"]["mean"] for x in tr), p999_us=max(x["host_rt"]["p999"] for x in tr),
                         gpu_us=max(float(np.mean(x["gpu_us"])) for x in tr))
            out["nets"][net]["paths"][f"{path}@{B}"] = v
        P = out["nets"][net]["paths"]

        def pick(cands, key, what):
            ok = [c for c in cands if c in P and P[c]["reproducible"]]
            if not ok:                             # finding 4: never freeze a path that is not reproducible
                raise SystemExit(f"{net} {what}: no reproducible path among {sorted(cands)} (at least {sc.REPS} "
                                 "processes that completed the screen, one output sha256): nothing frozen")
            return min(ok, key=lambda c: P[c][key])
        if net == "S":
            fz = {"single": pick(["engine@1"], "t1_us", "single"),
                  "batched": pick([f"engine@{sc.S_BATCH}"], "a_us", "batched")}
            comp = {"t1_us": P[fz["single"]]["t1_us"], "p999_us": P[fz["single"]]["p999_us"],
                    "a_us": P[fz["batched"]]["a_us"]}
        else:
            fz = {"single": pick([k for k in P if k.endswith("@1")], "t1_us", "single"),
                  "amortized": pick([k for k in P if int(k.split("@")[1]) <= 4096], "a_us", "amortized")}
            comp = {"t1_us": P[fz["single"]]["t1_us"], "p999_us": P[fz["single"]]["p999_us"],
                    "a_us": P[fz["amortized"]]["a_us"], "b_star": int(fz["amortized"].split("@")[1])}
        out["frozen"][net], out["components"][net] = fz, comp
    seen = {}
    for r in rows:
        if r.get("role") == "pipeline" and r["config"] == "C3":
            if proc(r) in seen.setdefault((r["p"], r["mode"]), set()):
                continue
            seen[r["p"], r["mode"]].add(proc(r))
            m = out["M"].setdefault(str(r["p"]), {})
            key = "single_us" if r["mode"] == "single" else "batched_us"
            val = r["latency_us"]["mean"] if r["mode"] == "single" else r["per_shot_us"]
            m[key] = max(m.get(key, 0.0), val)
            m.setdefault("processes_" + r["mode"], 0)
            m["processes_" + r["mode"]] += 1
            if r["mode"] == "single":
                m["p999_us"] = max(m.get("p999_us", 0.0), r["latency_us"]["p999"])
    few = {f"{p} {m}": len(seen.get((p, m), ())) for p in sc.PS for m in MODES if len(seen.get((p, m), ())) < sc.REPS}
    if few:
        raise SystemExit(f"M alone (C3) from fewer than {sc.REPS} processes: {few}")
    sc.write_json(a.out, out)
    print(json.dumps({"frozen": out["frozen"], "components": out["components"], "M": out["M"],
                      "flags": out["flags"], "kcache": out["kcache"]}), flush=True)


# ---------------------------------------------------------------------------------------------------------- pipeline

_PM = {}


def _m_init(p):
    _PM["m"] = sc.m_matcher(p)
    _PM["m"].decode(np.zeros(sc.sl.Layout(sc.D).D, np.uint8), enable_correlations=True)


def _m_ready(_):
    time.sleep(0.05)
    return os.getpid()


def _m_decode(args):
    idx, packed, Dd = args
    det = np.unpackbits(packed, axis=1, count=Dd)
    return idx, sc.m_batch(_PM["m"], det)


def pipeline(a):
    import torch
    dev, gpu = sc.setup_gpu()
    rec = sc.record(gpu)
    lock = json.load(open(a.lock)) if a.lock else None
    if lock is None and a.config != "C3":
        raise SystemExit("pipeline needs the lock (thresholds, frozen paths) except for C3")
    if lock is not None:
        sc.same_identity(rec, lock["identity"], "this pipeline")
        if lock.get("dry", False) != a.dry:
            raise SystemExit("a dry lock serves dry runs only")
    kind = sc.kind_of("speed", a.dry)
    if lock is not None:                          # finding 5: the lock's plan fixes chunks, sizes and repetitions
        plan = lock["pipes"].get(a.stage) if a.stage else None
        if plan is None or a.config not in sc.PIPE_CONFIGS or a.rep not in plan["reps"]:
            raise SystemExit(f"{a.config} rep {a.rep}: not in the lock's plan of --stage {a.stage!r} (gate or c7)")
        ch, n = range(*plan["chunks"]), plan[a.mode]
    else:
        ch = sc.chunk_range(a.chunks) if a.chunks else range(*sc.PIPE_CHUNKS)
        n = a.n or (20_000 if a.mode == "single" else 500_000)
    nch = (n + sc.CHUNK - 1) // sc.CHUNK
    if nch > len(ch):
        raise SystemExit(f"{n} shots need {nch} chunks, the range has {len(ch)}")
    Dd = sc.sl.Layout(sc.D).D
    cfg = a.config
    chs = lock["p"][str(a.p)]["chosen"] if lock is not None else {}
    fz = lock["frozen"] if lock is not None else {}
    tau_s = {"C1": chs.get("C1", {}).get("tau1"), "C5": chs.get("C5", {}).get("tau1"), "C4": sc.TAU_C4}.get(cfg)
    tau2 = chs.get("C1", {}).get("tau2")
    uses_S, uses_B, uses_M = cfg in ("C1", "C4", "C5"), cfg in ("C1", "C2", "C5"), cfg in ("C1", "C3", "C4")
    jw = a.joint_w
    if jw is not None:
        from surface import sv_joint
        if cfg != "C5":
            raise SystemExit(f"--joint-w: the joint decision is C5's (S -> B), not {cfg}'s")
    nets = {}
    t_build = time.time()
    gS = gB = None
    if uses_S:
        _, tabs, scfg, nets["S"] = sc.load_S()
        gS = sc.Graph(sc.s_fn(tabs, scfg), 1 if a.mode == "single" else sc.S_BATCH, Dd, dev)
    if uses_B:
        bmod, nets["B"] = sc.load_B(dev)
        path, bb = (fz["B"]["single"] if a.mode == "single" else fz["B"]["amortized"]).split("@")
        if jw is not None:                        # B and the joint step; single mode reads S's outputs in place
            one = a.mode == "single"
            gB = sv_joint.JointB13(sc.b_fn(bmod, path), int(bb), Dd, dev, jw, c_src=gS.logit if one else None,
                                   ds_src=gS.dec if one else None)
        else:
            gB = sc.Graph(sc.b_fn(bmod, path), int(bb), Dd, dev)
    if lock is not None:
        for t, r in nets.items():
            if r["sha256"] != lock["nets"][t]["sha256"]:
                raise SystemExit(f"{t}: sha256 differs from the lock's")
    t_build = time.time() - t_build
    rec["kcache"] = sc.kcache()                   # after this process's compiles and captures
    if lock is not None and rec["kcache"]["digest"] != lock["kcache"]["digest"]:
        raise SystemExit(f"kernel caches {rec['kcache']} after this pipeline's compiles, not the lock's: something was "
                         "compiled or tuned anew; no shot drawn")
    det, obs = sc.sdata.eval_shots(sc.D, a.p, range(ch.start, ch.start + nch), n=sc.CHUNK, kind=kind, procs=4)
    digs = [sc.digest(det[i * sc.CHUNK:(i + 1) * sc.CHUNK], obs[i * sc.CHUNK:(i + 1) * sc.CHUNK]) for i in range(nch)]
    det, obs = det[:n], obs[:n, 0]
    base = {"role": "pipeline", "stage": a.stage, "config": cfg, "p": a.p, "mode": a.mode, "rep": a.rep,
            "kind_seed": kind, "chunks": [ch.start, ch.start + nch], "shots": n, "digests": digs, "tau_S": tau_s,
            "tau2": tau2 if cfg == "C1" else None, "nets": {t: r["sha256"] for t, r in nets.items()},
            "path_S": gS and f"engine@{gS.n}", "path_B": gB and f"{path}@{gB.n}" + ("+joint" if jw is not None else ""),
            "joint": sv_joint.record(jw) if jw is not None else None,
            "lock_sha256": sc.sha256_file(a.lock) if a.lock else "", "record": rec, "build_s": round(t_build, 1),
            "job": rec["job"], "host": rec["host"], "pid": os.getpid()}
    s_log = np.full(n, np.nan, np.float32)
    b_log = np.full(n, np.nan, np.float32)
    m_pred = np.full(n, 255, np.uint8)
    pred = np.zeros(n, np.uint8)
    stage = np.zeros(n, np.int8)
    st = torch.cuda.current_stream()
    if a.mode == "single":
        mt = None
        if uses_M:
            mt = sc.m_matcher(a.p)
            for i in range(3):
                mt.decode(det[i], enable_correlations=True)
        H = torch.as_tensor(det).pin_memory()
        hl, hd = torch.empty(1, dtype=torch.float32).pin_memory(), torch.empty(1, dtype=torch.uint8).pin_memory()
        g0 = gS if uses_S else gB
        lat = np.empty(n)
        for i in range(-200, n):                         # 200 untimed shots first (the stream's last 200)
            j = i % n
            t0 = time.perf_counter()
            k, sl_, bl_, mp_ = 0, np.nan, np.nan, 255
            if cfg == "C3":
                pr = mp_ = int(mt.decode(det[j], enable_correlations=True)[0])
                k = 2
            else:
                g0.x.copy_(H[j:j + 1], non_blocking=True)
                g0.replay()
                hl.copy_(g0.logit, non_blocking=True)
                hd.copy_(g0.dec, non_blocking=True)
                st.synchronize()
                pr = int(hd[0])
                if cfg == "C2":
                    k, bl_ = 1, float(hl[0])
                else:
                    sl_ = float(hl[0])
                    if abs(sl_) < tau_s:
                        if cfg == "C4":
                            k = 2
                        else:
                            k = 1
                            gB.x.copy_(gS.x)
                            gB.replay()
                            hl.copy_(gB.logit, non_blocking=True)
                            hd.copy_(gB.dec, non_blocking=True)
                            st.synchronize()
                            bl_, pr = float(hl[0]), int(hd[0])
                            if cfg == "C1" and abs(bl_) < tau2:
                                k = 2
                    if k == 2:
                        pr = mp_ = int(mt.decode(det[j], enable_correlations=True)[0])
            t1 = time.perf_counter()
            if i >= 0:
                lat[i] = 1e6 * (t1 - t0)
                stage[i], pred[i], s_log[i], b_log[i], m_pred[i] = k, pr, sl_, bl_, mp_
        fail = pred != obs
        row = {**base, "metric": "single", "latency_us": quant(lat), "fail": int(fail.sum()),
               "frac_B": float((stage >= 1).mean()) if cfg in ("C1", "C5") else float(cfg == "C2"),
               "frac_M": float((stage == 2).mean())}
        for kk, name in ((0, "first"), (1, "via_B"), (2, "via_M")):
            if (stage == kk).any():
                row[f"latency_{name}_us"] = quant(lat[stage == kk])
        extra = {"lat_us": lat.astype(np.float32)}
    else:
        pool = None
        if uses_M:
            pool = mp.get_context("spawn").Pool(sc.W_M, initializer=_m_init, initargs=(a.p,))
            pids = set()
            while len(pids) < sc.W_M:                     # every worker built its matcher before the clock starts
                pids |= set(pool.map(_m_ready, range(4 * sc.W_M), chunksize=1))
        pend = []

        def to_m(idx):
            for k0 in range(0, len(idx), M_TASK):
                jj = idx[k0:k0 + M_TASK]
                pend.append(pool.apply_async(_m_decode, ((jj, np.packbits(det[jj], axis=1), Dd),)))
        t0 = time.perf_counter()
        t_b = 0.0
        if cfg == "C3":
            to_m(np.arange(n))
            t_gpu = 0.0
        else:
            g0 = gS if uses_S else gB
            bs = g0.n
            nb = (n + bs - 1) // bs
            H = torch.zeros(nb * bs, Dd, dtype=torch.uint8)
            H[:n] = torch.as_tensor(det)
            H = H.reshape(nb, bs, Dd).pin_memory()
            hl0 = torch.empty(bs, dtype=torch.float32).pin_memory()
            hd0 = torch.empty(bs, dtype=torch.uint8).pin_memory()
            if gB is not None:
                hlb = torch.empty(gB.n, dtype=torch.float32).pin_memory()
                hdb = torch.empty(gB.n, dtype=torch.uint8).pin_memory()
            for _ in range(3):
                g0.replay()
                if gB is not None:
                    gB.replay()
            st.synchronize()
            qr, qi, qn = [], [], 0
            qc, qd = [], []
            t0 = time.perf_counter()

            def run_b(rows, idx, js=None):
                nonlocal t_b
                tb = time.perf_counter()
                m_ = len(idx)
                gB.x.zero_()
                gB.x[:m_].copy_(rows)
                if js is not None:                 # the joint step: S's logits and decisions of the queued rows
                    gB.c[:m_].copy_(js[0])
                    gB.ds[:m_].copy_(js[1])
                gB.replay()
                hlb.copy_(gB.logit, non_blocking=True)
                hdb.copy_(gB.dec, non_blocking=True)
                st.synchronize()
                zl = hlb[:m_].numpy().copy()
                b_log[idx], pred[idx] = zl, hdb[:m_].numpy()
                if cfg == "C1":
                    sel = sc.routed(zl, tau2)
                    if sel.any():
                        stage[idx[sel]] = 2
                        to_m(idx[sel])
                t_b += time.perf_counter() - tb
            for j in range(nb):
                g0.x.copy_(H[j], non_blocking=True)
                g0.replay()
                hl0.copy_(g0.logit, non_blocking=True)
                hd0.copy_(g0.dec, non_blocking=True)
                st.synchronize()
                lo, hi = j * bs, min(n, (j + 1) * bs)
                lg = hl0[:hi - lo].numpy().copy()
                pred[lo:hi] = hd0[:hi - lo].numpy()
                if cfg == "C2":
                    b_log[lo:hi] = lg
                    stage[lo:hi] = 1
                    continue
                s_log[lo:hi] = lg
                r = np.nonzero(sc.routed(lg, tau_s))[0]
                if not len(r):
                    continue
                idx = lo + r
                if cfg == "C4":
                    stage[idx] = 2
                    to_m(idx)
                    continue
                stage[idx] = 1
                rr = torch.as_tensor(r, device=dev)
                qr.append(g0.x[rr])
                if jw is not None:
                    qc.append(g0.logit[rr])
                    qd.append(g0.dec[rr])
                qi.append(idx)
                qn += len(idx)
                while qn >= gB.n:
                    rows, ii = torch.cat(qr), np.concatenate(qi)
                    if jw is not None:
                        cc, dd = torch.cat(qc), torch.cat(qd)
                        run_b(rows[:gB.n], ii[:gB.n], (cc[:gB.n], dd[:gB.n]))
                        qc, qd = ([cc[gB.n:]], [dd[gB.n:]]) if qn > gB.n else ([], [])
                    else:
                        run_b(rows[:gB.n], ii[:gB.n])
                    qr, qi, qn = ([rows[gB.n:]], [ii[gB.n:]], qn - gB.n) if qn > gB.n else ([], [], 0)
            if qn:
                run_b(torch.cat(qr), np.concatenate(qi), (torch.cat(qc), torch.cat(qd)) if jw is not None else None)
            t_gpu = time.perf_counter() - t0
        msecs = 0.0
        for f in pend:
            idx, mp_ = f.get()
            m_pred[idx] = mp_
            pred[idx] = mp_
        wall = time.perf_counter() - t0
        if pool is not None:
            pool.close()
            pool.join()
        if cfg == "C3":
            stage[:] = 2
        fail = pred != obs
        row = {**base, "metric": "batched", "batch_S": gS.n if gS is not None else None,
               "batch_B": gB.n if gB is not None else None, "m_workers": sc.W_M if uses_M else 0, "wall_s": wall,
               "gpu_phase_s": t_gpu, "b_stage_s": t_b, "per_shot_us": 1e6 * wall / n,
               "per_round_us": 1e6 * wall / (n * 13), "throughput_shots_s": n / wall, "fail": int(fail.sum()),
               "frac_B": float((stage >= 1).mean()) if cfg in ("C1", "C5") else float(cfg == "C2"),
               "frac_M": float((stage == 2).mean())}
        extra = {}
    row["kcache_end"] = sc.kcache()
    os.makedirs(a.save, exist_ok=True)
    name = f"{cfg}_{a.mode}_{sc.p_tag(a.p)}_rep{a.rep}.npz"
    sc.save_npz(os.path.join(a.save, name), row,      # the timing travels with the outputs it was measured on
                s_log=s_log, b_log=b_log, m_pred=m_pred, stage=stage, pred=pred, obs=obs, **extra)
    row["saved"] = name
    emit(a.out, row)


# ----------------------------------------------------------------------------------------------------------- bitwise

def bitwise(a):
    """Every pipeline output of the stage a.expect against the read computations on the same shots (module
    docstring)."""
    lock = json.load(open(a.lock))
    lock_sha = sc.sha256_file(a.lock)
    sc.check_analysis(lock["identity"], "bitwise")
    plan, kind, kd = lock["pipes"][a.expect], lock["kinds"]["speed"], lock["kcache"]["digest"]
    want = {f"{c}_{m}_{sc.p_tag(p)}_rep{r}.npz" for c in sc.PIPE_CONFIGS for m in MODES for p in sc.PS
            for r in plan["reps"]}
    have = {os.path.basename(f) for f in glob.glob(os.path.join(a.pipes, "*.npz"))}
    if have != want:                               # finding 5: the whole stage, nothing missing, nothing else
        raise SystemExit(f"{a.pipes}: missing {sorted(want - have)}, not in the {a.expect} plan {sorted(have - want)}")
    rwant = {**{f"record.{k}": lock["identity"][k] for k in sc.IDENTITY}, "kind": kind, "lock_sha256": lock_sha}
    res, cache, procs = [], {}, set()
    for name in sorted(want):
        f = os.path.join(a.pipes, name)
        m, z, fsha = sc.load_npz(f, sha=True)
        cfg, mode, p = m["config"], m["mode"], m["p"]
        n = plan[mode]
        span = [plan["chunks"][0], plan["chunks"][0] + -(-n // sc.CHUNK)]
        if (m["lock_sha256"] != lock_sha or m.get("stage") != a.expect or m["kind_seed"] != kind or
                m["chunks"] != span or m["shots"] != n or f"{cfg}_{mode}_{sc.p_tag(p)}_rep{m['rep']}.npz" != name):
            raise SystemExit(f"{f}: not this lock's {a.expect} plan (lock, stage, seed kind, chunks, shots or name)")
        sc.same_identity(m["record"], lock["identity"], f)
        procs.add(proc(m))
        key = (p, mode, tuple(m["chunks"]))
        if key not in cache:
            comps = ("S1", "B1", "MS") if mode == "single" else ("SB", "Bb", "BQ1", "BQ5", "MB")
            cache[key] = sc.load_read(os.path.join(a.read, mode), p, comps, range(*m["chunks"]), rwant, complete=kd)
        o = cache[key]
        if m["job"] in o["jobs"]:                  # item 8 and finding 5: the reads ran in another job
            raise SystemExit(f"{f}: the reads of its shots ran in its own job {m['job']}")
        if o["digests"][:len(m["digests"])] != m["digests"]:
            raise SystemExit(f"{f}: the pipeline ran other shots than the read")
        oo = {c: {k: v[:n] for k, v in o[c].items()} for c in o if c in sc.COMPS}
        single = mode == "single"
        ch = lock["p"][str(p)]["chosen"]
        jw = (m.get("joint") or {}).get("w")
        if (jw is None) != (a.joint_w is None or cfg != "C5") or (jw is not None and jw != a.joint_w):
            raise SystemExit(f"{f}: joint {m.get('joint')}, --joint-w {a.joint_w} (C5 only)")
        if jw is not None:                         # C5 with the joint step (sv_joint.assemble_joint)
            from surface import sv_joint
            dec, stage, _ = sv_joint.assemble_joint(sc, mode, oo, ch, jw)
        else:
            dec, stage = sc.assemble(cfg, mode, oo, ch)
        chk = {"decisions": int((z["pred"] != dec).sum()), "stages": int((z["stage"] != stage).sum())}
        if cfg in ("C1", "C4", "C5"):
            s = oo["S1" if single else "SB"]["c"]
            chk["S_logits"] = int((z["s_log"].view(np.uint32) != s.view(np.uint32)).sum())
        if cfg in ("C1", "C2", "C5"):
            bname = "B1" if single else {"C1": "BQ1", "C5": "BQ5", "C2": "Bb"}[cfg]
            via = z["stage"] >= 1 if cfg != "C2" else np.ones(n, bool)
            bz = oo[bname]["z"]
            chk["B_logits"] = int((z["b_log"][via].view(np.uint32) != bz[via].view(np.uint32)).sum())
        if cfg in ("C1", "C3", "C4"):
            via = z["stage"] == 2
            mp_ = oo["MS" if single else "MB"]["pred"]
            chk["M_decisions"] = int((z["m_pred"][via] != mp_[via]).sum())
        # pinned caches (item 8): nothing compiled anew since the lock, by the pipeline or by a read
        chk["kcache_differ"] = int(m["kcache_end"]["digest"] != kd) + int(m["record"]["kcache"]["digest"] != kd) + \
            sum(int(o["meta"][c]["record"]["kcache"]["digest"] != kd) for c in o["meta"])
        ok = all(v == 0 for v in chk.values())
        timing = {"value": m["latency_us"]["mean"], "p999": m["latency_us"]["p999"]} if single else \
            {"value": m["per_shot_us"]}
        res.append({"file": name, "sha256": fsha, "config": cfg, "mode": mode, "p": p, "rep": m["rep"], "shots": n,
                    "chunks": m["chunks"], "kind": m["kind_seed"], "lock_sha256": m["lock_sha256"], "pass": ok, **chk,
                    "job": m["job"], "host": m["host"], "pid": m["pid"], "read_jobs": sorted(o["jobs"]),
                    "timing": timing, "kcache": m["record"]["kcache"]})
        print(json.dumps({k: v for k, v in res[-1].items() if k not in ("kcache", "sha256")}), flush=True)
    if len(procs) != len(want) or len({x[0] for x in procs}) != 1:
        raise SystemExit(f"{a.expect}: {len(procs)} processes in {len({x[0] for x in procs})} jobs made the "
                         f"{len(want)} outputs: each comes from a process of its own, all of one job")
    out = {"expect": a.expect, "lock_sha256": lock_sha, "pipes": os.path.abspath(a.pipes),
           "read": os.path.abspath(a.read), "identity": {**sc.env(), "code": sc.code()},
           "pass": {m: all(r["pass"] for r in res if r["mode"] == m) for m in MODES}, "rows": res}
    sc.write_json(a.out, out)
    print(json.dumps({"pass": out["pass"], "files": len(res)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--net", required=True, choices=["S", "B"])
    s.add_argument("--rep", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--save", required=True)
    s.add_argument("--chunks", default=f"{sc.SCREEN_CHUNKS[0]}-{sc.SCREEN_CHUNKS[1]}")
    s.add_argument("--batches", default=",".join(str(b) for b in sc.BATCHES))
    s.add_argument("--n-agree", default="100000,20000", help="shots per p at batch >= 64, below 64")
    s.add_argument("--n-rt", type=int, default=10_000)
    s.add_argument("--secs", type=float, default=2.0)
    s.add_argument("--dry", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--rows", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("pipeline")
    s.add_argument("--config", required=True, choices=list(sc.CONFIGS))
    s.add_argument("--p", type=float, required=True)
    s.add_argument("--mode", required=True, choices=["single", "batched"])
    s.add_argument("--rep", type=int, required=True)
    s.add_argument("--lock", default="")
    s.add_argument("--stage", default="", choices=["", "gate", "c7"], help="with the lock: its plan (chunks, shots, "
                   "repetitions)")
    s.add_argument("--out", required=True)
    s.add_argument("--save", required=True)
    s.add_argument("--chunks", default="", help="C3 before the lock")
    s.add_argument("--n", type=int, default=0, help="C3 before the lock")
    s.add_argument("--dry", action="store_true")
    s.add_argument("--joint-w", type=float, default=None, help="C5: the joint decision's weight w (sv_joint.py)")
    s = sub.add_parser("bitwise")
    s.add_argument("--lock", required=True)
    s.add_argument("--expect", required=True, choices=["gate", "c7"])
    s.add_argument("--pipes", required=True)
    s.add_argument("--read", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--joint-w", type=float, default=None, help="the C5 files' joint weight (sv_joint.assemble_joint)")
    a = ap.parse_args()
    {"screen": screen, "summarize": summarize, "pipeline": pipeline, "bitwise": bitwise}[a.cmd](a)


if __name__ == "__main__":
    main()
