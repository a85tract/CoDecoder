"""Phase C's speed protocol on an NVIDIA H100 80GB HBM3, a second column for speed only (H100_TIMING.md).

The RTX PRO 6000 results stay the primary ones (PHASE_C_PLAN.md 5.1). This entry point admits one other card, the H100,
where pc_common.gpu_env admits only phase C's, and labels every row "H100 (secondary, speed only)" (card_role): nothing
it writes enters a lock, a check, a confirmation read, a report or a verdict. It changes nothing in pc_common, pc_speed,
the locks or the primary outputs. It runs pc_speed's own code on the paths frozen in the lock (C3's
speed_components.json, refused unless it is the file the lock froze), and refuses every other card, phase C's included.

  nets       python pc_h100.py nets --net TAG --rep I --locked locked_<setting>.json --speed speed_components.json \
                 --out rows.jsonl
  pipeline   python pc_h100.py pipeline ...                       (pc_speed.py pipeline's arguments, unchanged)
  outputs    python pc_h100.py outputs --locked locked_<setting>.json --speed speed_components.json --out DIR
  screen     python pc_h100.py screen --outputs DIR --read DIR_R --locked locked_<setting>.json \
                 --speed speed_components.json --out screen_<setting>.json                     (CPU only)
  summarize  python pc_h100.py summarize --rows rows.jsonl --speed speed_components.json --env env.json \
                 [--screen screen_<setting>.json ...] --out h100_speed.json

nets: one network alone in one fresh process (the job runs every network REPS = 3 times, rep-major, as C3 does), on its
frozen paths only (pc_common.frozen_paths): the batch sizes the plan times on each frozen path (fused and graph: 1, 16,
64, 256, 1,024 and 4,096; compiled: 1 and 4,096), each built as pc_speed.nets builds it and measured by pc_speed's
gpu_us, host_rt and amortized (batch 1: the GPU term and 10,000 host round trips; every batch: amortized, input resident
and host to host). The rows are pc_speed.nets' (kind 12, p = 0.55 %, the network's setting, chunks 0-3: the measurements
read the first 32,768). Before a path is timed it decodes min(rows, 20,000 below batch 64) rows once, as the screen of
pc_speed.nets does before its timing. There is no screen in this step; "outputs" and "screen" are the screen.

pipeline: pc_speed.pipeline itself, on the H100, with Relay-BP on this machine's CPU (single mode in the process,
batched mode in NCPU_RELAY worker processes): every locked configuration at the claim points and Cascade alone
(--config -1), both modes, as C7 runs them.

outputs (the H100 half of the screen, GPU): C4b's check shots (kind 12, chunks 10-19 at each claim point; generated here
by bb.cascade_data.eval_shots, each chunk's digest recorded as bb.cascade_eval records it) through every frozen path
pc_speed.check runs, as it runs them: every network alone (Cascade of the setting, its depth-rule variant and our five)
on its single, amortized and batched paths (check runs the first two; the batched path is added so that every frozen
path is screened), and every locked configuration at the claim point in both modes (S on its path, the locked thresholds
applied to its confidences in float64, B on the shots S routes: batch 1 row by row, or the routed rows in queue order at
4,096). Writes DIR/h100_<p>.npz (decisions packed, confidences, routing masks packed) and DIR/h100_<p>.json (digests,
card, paths); also times the seeded Relay-BP decoder's build on this CPU, as check does.

screen (CPU only, once C4b's read of the same shots exists): the H100 outputs against that read, the RTX PRO 6000's
accuracy paths, judged as pc_speed.check judges a timed path: a ConvLines path must be bitwise (every decision and every
confidence), a Cascade path passes by pc_common.agrees on its failures; a configuration passes on its failures, its
shots routed to B and to R (each by agrees) and its S bitwise, Relay-BP deciding every shot that either version routes
to it with the decoder seeded by the shot's index (pc_relay). Refused unless the read's chunk digests are the outputs'.
Writes screen_<setting>.json in check_<setting>.json's form.

summarize: per network and frozen path the slowest of its processes at every batch size, per configuration and mode the
slowest process (PHASE_C_PLAN.md 5.2-5.3), the RTX PRO 6000's frozen values beside them, and per value the screen's
verdict: passed, failed, or not screened (no screen file yet). A failed or unscreened value is shown and marked."""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np

import pc_common as pc
import pc_speed as ps

H100 = "NVIDIA H100 80GB HBM3"
ROLE = "H100 (secondary, speed only)"
PLAN_B = {"fused": pc.BATCHES, "graph": pc.BATCHES, "compiled": ps.COMPILED_B}   # the batch sizes C3 times per path
NETS_CHUNKS = 4                                    # rows of the timings: pc_speed.nets' first 40,000 (32,768 are read)
SHOT_PROCS = 25                                    # processes of the shot generation (the shots depend on the seeds only)
ALONE_MODES = ("single", "amortized", "batched")


def gpu_env(dev):
    """The H100, refused unless it is one (a MIG slice's name adds its profile), with its driver and the venv's torch;
    every row records card_role, so that no H100 value can pass for a phase-C value."""
    import torch
    name = torch.cuda.get_device_name(dev)
    if name != H100:
        raise SystemExit(f"GPU {name!r}: pc_h100.py times on an {H100} only; phase C's card runs pc_speed.py")
    return {"gpu": name, "driver": os.environ.get("PC_GPU_DRIVER", ""), "torch": torch.__version__, "card_role": ROLE}


pc.gpu_env = gpu_env                               # pc_speed.setup() looks it up here; the H100 only, labelled


def more_procs():
    """bb.cascade_data.eval_shots with SHOT_PROCS processes: the same shots (each chunk from its own seed), sooner;
    it returns before anything is timed."""
    from bb import cascade_data as cdata
    if getattr(cdata.eval_shots, "h100", False):
        return
    f = cdata.eval_shots

    def eval_shots(code, p, chunks, n=10000, kind=0, procs=1, family="uniform"):
        chunks = list(chunks)
        return f(code, p, chunks, n=n, kind=kind, procs=max(procs, min(len(chunks), SHOT_PROCS)), family=family)
    eval_shots.h100 = True
    cdata.eval_shots = eval_shots


def locked_speed(a):
    L = json.load(open(a.locked))
    return L, ps.frozen_components(a.speed, L)


def nets(a):
    import torch
    from bb import cascade_data as cdata
    from bb.cascade_eval import Predictor
    from bb.timing import gpu_watch
    more_procs()
    dev, env = ps.setup()
    L, sp = locked_speed(a)
    path0, fam, kind = pc.NETS[a.net]
    fz = pc.frozen_paths(sp["nets"][a.net])
    pr = Predictor(path0, dev, "ema", "bf16")
    if a.net in L["nets"] and pr.sig["sha256"] != L["nets"][a.net]:
        raise SystemExit(f"{a.net}: sha256 differs from the locked network")
    D = pr.model.D if kind == "convlines" else 432
    base = {"role": "net", "net": a.net, "kind": kind, "rep": a.rep, **env, "sha256": pr.sig["sha256"],
            "tree": os.environ.get("PC_TREE", ""), "speed_sha256": L["speed_sha256"]}
    todo = sorted({fz["single"]} | {(fz[m][0], B) for m in ("amortized", "batched") for B in PLAN_B[fz[m][0]]})
    det, _ = cdata.eval_shots(pc.CODE, ps.P_AGREE, range(NETS_CHUNKS), kind=a.kind, procs=4, family=fam)
    dg = torch.as_tensor(det, device=dev)
    fwd = ps.make_fwd(pr)
    cfn = None
    for name, B in todo:
        t_build = time.time()
        try:
            if name == "fused":
                gr = ps.build(pr, name, B)
            else:
                if name == "compiled" and cfn is None:
                    cfn = torch.compile(fwd, dynamic=False)
                gr = ps.Graph(cfn if name == "compiled" else fwd, B, D, dev)
        except Exception as e:                              # recorded, the other paths go on
            ps.emit(a.out, {**base, "path": name, "batch": B, "metric": "error", "error": repr(e)[:300]})
            continue
        row = {**base, "path": name, "batch": B, "prec": pc.precision(name, kind),
               "build_s": round(time.time() - t_build, 1), "frozen": [m for m, x in fz.items() if x == (name, B)]}
        gr.run(dg[:ps.N_AGREE_B1 if B < 64 else len(dg)])  # decoded once before the timing, as in pc_speed.nets
        torch.cuda.synchronize()
        stop = gpu_watch()
        res = {}
        if B == 1:
            res["gpu_us"] = ps.gpu_us(gr)
            res["host_rt"] = ps.quant(ps.host_rt(gr, dg, n=a.n_rt))
        res["amortized"] = ps.amortized(gr, dg, secs=a.secs)
        ps.emit(a.out, {**row, "metric": "time", **res, "clock": stop()})
        del gr
        torch.cuda.empty_cache()


def pipeline(a):
    more_procs()
    ps.pipeline(a)


def chunk_digests(det, c0):
    """bb.cascade_eval's digest of each chunk's detector rows (np.packbits, big-endian bits, first 16 hex digits)."""
    return [{"chunk": c0 + k, "det_sha256": hashlib.sha256(np.packbits(det[i:i + pc.CHUNK], axis=1).tobytes())
             .hexdigest()[:16]} for k, i in enumerate(range(0, len(det), pc.CHUNK))]


def key(t, path, B, what):
    return f"{t}__{path}@{B}__{what}"


def outputs(a):
    import torch
    from bb import cascade_data as cdata
    from bb.cascade_eval import Predictor
    more_procs()
    dev, env = ps.setup()
    L, sp = locked_speed(a)
    fam = L["family"]
    alone = [t for t in dict.fromkeys((L["B"], pc.DEPTH[fam]) + pc.OURS) if t in sp["nets"]]
    frozen = {t: {m: f"{x}@{b}" for m, (x, b) in pc.frozen_paths(v).items()} for t, v in sp["nets"].items()}
    preds, graphs = {}, {}
    os.makedirs(a.out, exist_ok=True)

    def run(t, mode, rows, score="minabs"):
        """pc_speed.check's outputs(): network t on its frozen path for this mode over rows on the device."""
        if t not in preds:
            preds[t] = Predictor(pc.NETS[t][0], dev, "ema", "bf16")
            if preds[t].sig["sha256"] != L["nets"].get(t, preds[t].sig["sha256"]):
                raise SystemExit(f"{t}: sha256 differs from the locked network")
        path, B = pc.frozen_paths(sp["nets"][t])[mode]
        if (t, path, B, score) not in graphs:
            graphs[(t, path, B, score)] = ps.build(preds[t], path, B, score)
        if not len(rows):
            return np.zeros((0, pc.K), np.uint8), np.zeros(0, np.float32)
        bits, conf = graphs[(t, path, B, score)].run(rows)
        return bits.cpu().numpy(), conf.float().cpu().numpy()

    c0, c1 = pc.CHECK_CHUNKS, 2 * pc.CHECK_CHUNKS
    for p in pc.RELAY_P:
        t0 = time.time()
        det, obs = cdata.eval_shots(pc.CODE, p, range(c0, c1), kind=a.kind, procs=4, family=fam)
        n = len(det)
        dg = torch.as_tensor(det, device=dev)
        full, keep = {}, {}

        def whole(t, mode):
            path, B = pc.frozen_paths(sp["nets"][t])[mode]
            if (t, path, B) not in full:
                full[(t, path, B)] = run(t, mode, dg)
                keep[key(t, path, B, "bits")] = np.packbits(full[(t, path, B)][0], axis=1)
                keep[key(t, path, B, "conf")] = full[(t, path, B)][1]
            return full[(t, path, B)]

        for t in alone:
            for mode in ALONE_MODES:
                whole(t, mode)
        cfgs = [(i, c) for i, c in enumerate(L["configs"]) if c["p"] == p]
        for i, c in cfgs:                                   # pc_speed.check's timed versions of each configuration
            for mode in ("single", "batched"):
                bS, cS = whole(c["S"], mode)
                cS = cS.astype(np.float64)
                if c["type"] == "S-R":
                    toB, toR, dec = np.zeros(n, bool), cS < c["tau"]["tau"], bS
                else:
                    toB = cS < c["tau"]["tau1"]
                    ii = np.nonzero(toB)[0]
                    if mode == "single" and c.get("score", "minabs") == "minabs":
                        bB, cB = (x[ii] for x in whole(c["B"], mode))
                    else:
                        bB, cB = run(c["B"], mode, dg[torch.as_tensor(ii, device=dev)], c.get("score", "minabs"))
                    dec = bS.copy()
                    dec[ii] = bB
                    toR = np.zeros(n, bool)
                    if c["type"] == "S-B-R":
                        toR[ii[cB.astype(np.float64) < c["tau"]["tau2"]]] = True
                keep[f"cfg{i}__{mode}__toB"] = np.packbits(toB)
                keep[f"cfg{i}__{mode}__toR"] = np.packbits(toR)
                keep[f"cfg{i}__{mode}__dec"] = np.packbits(dec, axis=1)
        np.savez_compressed(os.path.join(a.out, f"h100_{pc.p_tag(p)}.npz"), **keep)
        meta = {"p": p, "family": fam, "kind": a.kind, "chunks": [c0, c1], "shots": n, **env,
                "tree": os.environ.get("PC_TREE", ""), "locked": os.path.abspath(a.locked),
                "locked_sha256": pc.sha256_file(a.locked), "speed_sha256": L["speed_sha256"],
                "nets": {t: preds[t].sig["sha256"] for t in preds}, "frozen": frozen, "alone": alone,
                "configs": [i for i, _ in cfgs], "chunk_meta": chunk_digests(det, c0),
                "fail": {k.split("__")[0] + " " + k.split("__")[1]: int((np.unpackbits(v, axis=1, count=pc.K) != obs)
                                                                         .any(1).sum())
                         for k, v in keep.items() if k.endswith("__bits")},
                "secs": round(time.time() - t0, 1)}
        ps.relay_import()
        import pc_relay
        pc_relay.decoder(fam, p)                            # the matrices once, then the seeded builds, as check does
        tb = time.perf_counter()
        for k in range(20):
            pc_relay.decoder(fam, p, k)
        meta["relay_build_ms"] = 1e3 * (time.perf_counter() - tb) / 20
        json.dump(meta, open(os.path.join(a.out, f"h100_{pc.p_tag(p)}.json"), "w"), indent=1)
        print(json.dumps({k: meta[k] for k in ("p", "family", "shots", "secs", "relay_build_ms")} |
                         {"fail": meta["fail"]}), flush=True)


def screen(a):
    """pc_speed.check's judgement of the H100's outputs against the RTX PRO 6000's read (module docstring)."""
    from multiprocessing import get_context
    import pc_route
    L, sp = locked_speed(a)
    fam = L["family"]
    ps.relay_import()
    import pc_relay
    alone = [t for t in dict.fromkeys((L["B"], pc.DEPTH[fam]) + pc.OURS) if t in sp["nets"]]
    out = {"card_role": ROLE, "family": fam, "locked": os.path.abspath(a.locked), "locked_sha256": pc.sha256_file(a.locked),
           "speed_sha256": L["speed_sha256"], "read": os.path.abspath(a.read), "outputs": os.path.abspath(a.outputs),
           "tolerance": {"rel": pc.AGREE_REL, "z": pc.AGREE_Z},
           "frozen": {t: {m: f"{x}@{b}" for m, (x, b) in pc.frozen_paths(v).items()} for t, v in sp["nets"].items()},
           "relay_union": {}, "secs": {}, "rows": [], "host": platform.node()}
    pool = get_context("spawn").Pool(pc.NCPU_RELAY, initializer=ps.relay_import)
    for p in pc.RELAY_P:
        t0 = time.time()
        z, meta, idx = pc.load_read(a.read, p)
        h = np.load(os.path.join(a.outputs, f"h100_{pc.p_tag(p)}.npz"))
        hm = json.load(open(os.path.join(a.outputs, f"h100_{pc.p_tag(p)}.json")))
        if meta["family"] != fam or hm["family"] != fam:
            raise SystemExit(f"{a.read} at {p}: family {meta['family']} / {hm['family']}, locked {fam}")
        if pc.det_digests(meta) != {c["chunk"]: c["det_sha256"] for c in hm["chunk_meta"]}:
            raise SystemExit(f"{a.read} at {p}: other shots than the H100 outputs {a.outputs}")
        if hm["speed_sha256"] != L["speed_sha256"] or hm["locked_sha256"] != out["locked_sha256"]:
            raise SystemExit(f"{a.outputs} at {p}: made with another lock or other speed components")
        for t, i in idx.items():
            if t in L["nets"] and meta["models"][i]["sha256"] != L["nets"][t]:
                raise SystemExit(f"{t}: sha256 differs from the locked network")
            if t in hm["nets"] and meta["models"][i]["sha256"] != hm["nets"][t]:
                raise SystemExit(f"{t}: the read's network is not the H100's")
        for k in ("gpu", "driver", "torch", "card_role"):
            out[k] = hm[k]
        n = len(z["obs"])
        det = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
        obs = np.unpackbits(z["obs"], axis=1, count=pc.K)

        def whole(t, mode):
            path, B = pc.frozen_paths(sp["nets"][t])[mode]
            return (np.unpackbits(h[key(t, path, B, "bits")], axis=1, count=pc.K), h[key(t, path, B, "conf")])

        def bitwise(t, mode):
            if pc.NETS[t][2] != "convlines":
                return True
            bits, conf = whole(t, mode)
            return bool(np.array_equal(bits, np.unpackbits(z[f"pred{idx[t]}"], axis=1, count=pc.K))
                        and np.array_equal(conf, z[f"conf{idx[t]}"]))

        head = {"p": p, "kind_seed": meta["kind"], "shots": n}
        for t in alone:
            if t not in idx:
                continue
            fe = z[f"fail{idx[t]}"].astype(bool)
            pe = np.unpackbits(z[f"pred{idx[t]}"], axis=1, count=pc.K)
            for mode in ALONE_MODES:
                bits, conf = whole(t, mode)
                c = ps._cmp((bits != obs).any(1), fe)
                bw = bitwise(t, mode)
                out["rows"].append({**head, "what": "alone", "net": t, "mode": mode, "path": out["frozen"][t][mode],
                                    "differ": int((bits != pe).any(1).sum()),
                                    "conf_differ": int((conf != z[f"conf{idx[t]}"]).sum()), "bitwise_required":
                                    pc.NETS[t][2] == "convlines", "bitwise": bw if pc.NETS[t][2] == "convlines" else
                                    None, "fail": c, "pass": c["pass"] and bw})
        cfgs = [(i, c) for i, c in enumerate(L["configs"]) if c["p"] == p]
        ver = {}
        for i, c in cfgs:
            ver[(i, "eval")] = pc_route.masks(z, meta, idx, c, c["tau"]) + (None,)
            for mode in ("single", "batched"):
                ver[(i, mode)] = (np.unpackbits(h[f"cfg{i}__{mode}__toB"], count=n).astype(bool),
                                  np.unpackbits(h[f"cfg{i}__{mode}__toR"], count=n).astype(bool),
                                  np.unpackbits(h[f"cfg{i}__{mode}__dec"], axis=1, count=pc.K))
        union = np.nonzero(np.any([v[1] for v in ver.values()], axis=0))[0] if ver else np.zeros(0, np.int64)
        has, rfail, rsec = np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan)
        if len(union):
            res = pool.map(pc_relay.relay_task, [(fam, p, union[k:k + 10], det[union[k:k + 10]])
                                                 for k in range(0, len(union), 10)], chunksize=1)
            has[union] = True
            rfail[union] = (np.concatenate([r[0] for r in res]) != obs[union]).any(1)
            rsec[union] = np.concatenate([r[1] for r in res])
        for i, c in cfgs:
            toB_e, toR_e, _ = ver[(i, "eval")]
            fe = pc_route.config_fail(z, idx, c, toB_e, toR_e, (has, rfail, rsec))
            for mode in ("single", "batched"):
                toB, toR, dec = ver[(i, mode)]
                ft = (dec != obs).any(1)
                ft[toR] = rfail[toR]
                route = {}
                if c["type"] != "S-R":
                    route["to_B"] = ps._cmp(toB, toB_e)
                if c["type"] != "S-B":
                    route["to_R"] = ps._cmp(toR, toR_e)
                cf = ps._cmp(ft, fe)
                sbw = bitwise(c["S"], mode)
                out["rows"].append({**head, "what": "routed", "config": i, "type": c["type"],
                                    "decoder": pc.decoder_key(c), "S": c["S"], "B": c.get("B"), "mode": mode,
                                    "paths": {r: out["frozen"][c[r]][mode] for r in ("S", "B") if c.get(r)},
                                    "fail": cf, "routing": route, "S_bitwise": sbw,
                                    "pass": cf["pass"] and all(v["pass"] for v in route.values()) and sbw})
        out["relay_union"][str(p)] = int(len(union))
        out["secs"][str(p)] = round(time.time() - t0, 1)
    pool.close()
    pool.join()
    out["decoders"] = {}
    for r in out["rows"]:
        k = f"{r['net']} alone {r['mode']}" if r["what"] == "alone" else f"config {r['config']} {r['mode']}"
        out["decoders"][k] = out["decoders"].get(k, True) and r["pass"]
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"family": fam, "failing": [k for k, v in out["decoders"].items() if not v],
                      "secs": out["secs"]}))


def verdict(rows):
    """passed / failed / not screened over a list of screen rows."""
    if not rows:
        return "not screened"
    return "passed" if all(r["pass"] for r in rows) else "failed"


def summarize(a):
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    sp = json.load(open(a.speed))
    envs = sorted({(r.get("gpu"), r.get("torch"), r.get("card_role")) for r in rows if r.get("role")})
    if len(envs) != 1 or envs[0][2] != ROLE:
        raise SystemExit(f"rows from several cards, venvs or roles: {envs}")
    env = json.load(open(a.env)) if a.env else {}
    scr = {}
    for f in a.screen or []:
        s = json.load(open(f))
        if s.get("card_role") != ROLE or s["speed_sha256"] != pc.sha256_file(a.speed):
            raise SystemExit(f"{f}: not an H100 screen of these speed components")
        scr[s["family"]] = s
    srows = [r for s in scr.values() for r in s["rows"]]
    out = {"card_role": ROLE, "gpu": envs[0][0], "torch": envs[0][1],
           "driver": ", ".join(sorted({r.get("driver", "") for r in rows if r.get("role")})), "env": env,
           "rows_sha256": pc.sha256_file(a.rows), "speed_sha256": pc.sha256_file(a.speed),
           "rtx": {k: sp[k] for k in ("gpu", "driver", "torch")}, "screened_settings": sorted(scr), "nets": {},
           "pipelines": {}}
    for t in pc.NETS:
        tm = [r for r in rows if r.get("role") == "net" and r["net"] == t and r["metric"] == "time"]
        if not tm or t not in sp["nets"]:
            continue
        v = sp["nets"][t]
        fz = pc.frozen_paths(v)
        by = {}
        for r in tm:
            by.setdefault(f"{r['path']}@{r['batch']}", []).append(r)
        timed = {}
        for k, rs in sorted(by.items(), key=lambda x: (x[0].split("@")[0], int(x[0].split("@")[1]))):
            path, B = k.split("@")
            b1 = int(B) == 1
            rv = sp["nets"][t]["paths"].get(path, {})
            timed[k] = {"processes": len(rs), "a_us": max(r["amortized"]["h2h"] for r in rs),
                        "a_dev_us": max(r["amortized"]["dev"] for r in rs),
                        "t1_us": max(r["host_rt"]["mean"] for r in rs) if b1 else None,
                        "p999_us": max(r["host_rt"]["p999"] for r in rs) if b1 else None,
                        "gpu_term_us": max(float(np.mean(r["gpu_us"])) for r in rs) if b1 else None,
                        "clock_ok": all(bool((r.get("clock") or {}).get("accepted")) for r in rs),
                        "rtx_a_us": rv.get("a_us", {}).get(B), "rtx_screened": rv.get("screened", {}).get(B)}
        st = {m: verdict([r for r in srows if r["what"] == "alone" and r["net"] == t and r["mode"] == m])
              for m in ALONE_MODES}
        k1, ka, k4 = (f"{x}@{b}" for x, b in (fz["single"], fz["amortized"], fz["batched"]))
        g = lambda k, f: timed.get(k, {}).get(f)                                   # noqa: E731
        out["nets"][t] = {"kind": pc.NETS[t][2], "setting": pc.NETS[t][1],
                          "frozen": {"single": k1, "amortized": ka, "batched": k4},
                          "t1_us": g(k1, "t1_us"), "p999_us": g(k1, "p999_us"), "gpu_term_us": g(k1, "gpu_term_us"),
                          "a_frozen_us": g(ka, "a_us"), "a4096_us": g(k4, "a_us"),
                          "processes": min(g(k, "processes") or 0 for k in (k1, ka, k4)),
                          "screen": st, "timed": timed,
                          "rtx": {"t1_us": v["t1_us"], "p999_us": v["p999_us"], "a_best_us": v["a_best_us"],
                                  "a_best": v["a_best"], "a4096_us": v["a4096_us"], "processes": v["processes"]}}
    locks = {}
    for f in a.locked or []:
        lk = json.load(open(f))
        locks[lk["family"]] = lk
    for r in rows:
        if r.get("role") != "pipeline":
            continue
        k = f"{r['family']} config {r['config']} p {r['p']} {r['mode']}"
        out["pipelines"].setdefault(k, []).append(r)
    pipes = {}
    for k, rs in out["pipelines"].items():
        r0 = rs[0]
        fam, i, p, mode = r0["family"], r0["config"], r0["p"], r0["mode"]
        fr = scr[fam]["rows"] if fam in scr else []           # the screen of the pipeline's own setting
        if i < 0:                                           # Cascade alone: single on its single path, batched on
            m = "single" if mode == "single" else "amortized"   # its amortized path (pc_speed.pipeline)
            st = verdict([r for r in fr if r["what"] == "alone" and r["net"] == r0["S"] and r["mode"] == m
                          and r["p"] == p])
            dec = "Cascade alone"
        else:
            st = verdict([r for r in fr if r["what"] == "routed" and r["config"] == i and r["mode"] == mode
                          and r["p"] == p])
            dec = pc.decoder_key(locks[fam]["configs"][i]) if fam in locks else r0["type"]
        v = {"family": fam, "config": i, "p": p, "mode": mode, "type": r0["type"], "decoder": dec, "S": r0["S"],
             "B": r0["B"], "q1": r0.get("q1"), "r": r0.get("r"), "q": r0.get("q"), "path_S": r0.get("path_S"),
             "path_B": r0.get("path_B"), "processes": len(rs), "screen": st,
             "frac_B": float(np.mean([r["frac_B"] for r in rs])), "frac_R": float(np.mean([r["frac_R"] for r in rs])),
             "clock_ok": all(bool((r.get("clock") or {}).get("accepted")) for r in rs)}
        if mode == "single":
            v["latency_us"] = {q: max(r["latency_us"][q] for r in rs) for q in ("mean", "p50", "p99", "p999", "max")}
            for name in ("via_B", "via_R"):
                xs = [r[f"latency_{name}_us"]["mean"] for r in rs if f"latency_{name}_us" in r]
                v[f"latency_{name}_mean_us"] = max(xs) if xs else None
        else:
            v["per_round_us"] = max(r["per_round_us"] for r in rs)
            v["per_shot_us"] = max(r["per_shot_us"] for r in rs)
            v["throughput_shots_s"] = min(r["throughput_shots_s"] for r in rs)
            xs = [r["relay_s_mean"] for r in rs if r.get("relay_s_mean") is not None]
            v["relay_s_mean"] = max(xs) if xs else None
        pipes[k] = v
    out["pipelines"] = pipes
    json.dump(out, open(a.out, "w"), indent=1)
    for t, v in out["nets"].items():
        print(json.dumps({"net": t, "t1_us": v["t1_us"] and round(v["t1_us"], 1), "rtx_t1_us": round(v["rtx"]["t1_us"], 1),
                          "a_frozen_us": v["a_frozen_us"] and round(v["a_frozen_us"], 3), "frozen": v["frozen"]["amortized"],
                          "rtx_a_us": round(v["rtx"]["a_best_us"], 3), "processes": v["processes"],
                          "screen": v["screen"]}))
    print(json.dumps({"pipelines": len(pipes), "complete": sum(v["processes"] >= pc.REPS for v in pipes.values())}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("nets")
    s.add_argument("--net", required=True, choices=list(pc.NETS))
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--n-rt", type=int, default=10_000, help="single-shot round trips timed")
    s.add_argument("--secs", type=float, default=2.0, help="seconds per amortized measurement")
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED)
    s.add_argument("--locked", required=True)
    s.add_argument("--speed", required=True, help="the speed_components.json frozen in the lock")
    s.add_argument("--out", required=True)
    s = sub.add_parser("pipeline")
    s.add_argument("--locked", required=True)
    s.add_argument("--config", type=int, required=True)
    s.add_argument("--p", type=float, default=None)
    s.add_argument("--mode", required=True, choices=["single", "batched"])
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--n", type=int, default=20_000)
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED)
    s.add_argument("--speed", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("outputs")
    s.add_argument("--locked", required=True)
    s.add_argument("--speed", required=True)
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED)
    s.add_argument("--out", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--outputs", required=True)
    s.add_argument("--read", required=True, help="C4b's read of the setting (check.sh: $R/out/check/<setting>/read)")
    s.add_argument("--locked", required=True)
    s.add_argument("--speed", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--rows", required=True)
    s.add_argument("--speed", required=True)
    s.add_argument("--env", default="")
    s.add_argument("--locked", nargs="*")
    s.add_argument("--screen", nargs="*")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"nets": nets, "pipeline": pipeline, "outputs": outputs, "screen": screen, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
