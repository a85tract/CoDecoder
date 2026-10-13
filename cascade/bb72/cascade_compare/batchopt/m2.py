"""Milestone 2 (batchopt/M2.md): develop, screen, inspect and time mfused.py's engine on the card of this process, and lever 2
(the routed pipeline's last, mostly padded Cascade batch on a small CUDA graph). Phase C's files are read, never written.

  dev       development sweep on --nets: the engine with every kernel at G in --gs shots per program and each stage count
            in --stages (and the vote's warps and int8 table, --extra), against today's engine: the decisions and the
            margins' bits against the frozen read, every intermediate buffer launch by launch at --buf-n shots, each
            kernel's eager time without PDL at --batches (torch.profiler), the whole call's graph replay, registers,
            spills and shared memory per kernel -> OUT/dev.jsonl
  refs      C3's shots (kind 12, chunks 0-9, at 0.2, 0.3 and 0.55 %, both settings; M1's files, checked against their
            sha256) and today's engine on them for every ConvLines network, chunk by chunk at 4,096 as the reads ran it
            -> OUT/refs/; anchors: today's engine against C4b's frozen reads, and against M1's references
  screen    see screen()                     ptx       see ptx()                  nets      see nets()
  pipeline  see pipeline()                   tail      see tail()                 report    see report()

Every step that times refuses any card but phase C's RTX PRO 6000 (pc_common.gpu_env); with M2_CARD=H100 it admits
the H100 80GB HBM3 instead and labels each row "H100 (secondary, speed only)".

Configurations (--cfg, --cfgs): base (today's engine, pc_fused.Batched), new (mfused.Engine as committed), or new with
overrides: new:g<G> (every kernel at G), new:s<stages>, new:<kernel>=<G> (stem, block, last, pool, vote), new:vw<warps>,
new:i8 (the pinned int8 psi table), new:i8x (unpinned: the deliberate control), new:noskip, new:novec, joined by ':'."""
import argparse
import difflib
import gzip
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase_c"))
sys.path.insert(0, HERE)
import m1  # noqa: E402
import pc_common as pc  # noqa: E402
import prof_batched as pb  # noqa: E402

if os.environ.get("M2_CARD") == "H100":            # the pod session (M2.md section 9): the H100 only, labelled
    def _gpu_env_h100(dev, _name="NVIDIA H100 80GB HBM3"):
        import torch
        name = torch.cuda.get_device_name(dev)
        if name != _name:
            raise SystemExit(f"GPU {name!r}: M2_CARD=H100 admits an {_name} only")
        return {"gpu": name, "driver": os.environ.get("PC_GPU_DRIVER", ""), "torch": torch.__version__,
                "card_role": "H100 (secondary, speed only)"}
    pc.gpu_env = _gpu_env_h100                      # pc_speed.setup() looks it up here

M1_REFS = "<cluster-scratch>/cascade_batchopt/out/m1_refs"
CONVLINES = pc.OURS                                 # S55, S55p, S55p_deep, r11, r12
BUFS = ("x", "u0", "u1", "U", "s", "z", "m1", "i1", "m2", "bits", "margin")
emit, bits_of, toolchain = m1.emit, m1.bits_of, m1.toolchain


# ------------------------------------------------------------------------------------------------ configurations
def cfg_of(name):
    """A configuration's name -> (base, mfused cfg or None)."""
    if name == "base":
        return True, None
    parts = name.split(":")
    if parts[0] != "new":
        raise SystemExit(f"configuration {name}: base or new[:...]")
    import mfused
    c = {"G": {}, "stages": {}, "warps": {}}
    for x in parts[1:]:
        if re.fullmatch(r"g\d+", x):
            c["G"] = {k: int(x[1:]) for k in mfused.KERNELS}
        elif re.fullmatch(r"s\d", x):
            c["stages"] = {k: int(x[1:]) for k in mfused.KERNELS}
        elif re.fullmatch(r"(stem|block|last|pool|vote)=\d+", x):
            k, v = x.split("=")
            c["G"][k] = int(v)
        elif re.fullmatch(r"(stem|block|last|pool|vote)s\d", x):
            c["stages"][x[:-2]] = int(x[-1])
        elif x in ("seq", "par"):
            c["seq"] = x == "seq"
        elif re.fullmatch(r"qg\d", x):
            c["qg"] = int(x[2:])
        elif re.fullmatch(r"vw\d+", x):
            c["warps"]["vote"] = int(x[2:])
        elif x in ("i8", "i8x", "f16"):
            c["psi8"] = {"i8": 1, "i8x": 2, "f16": 0}[x]
        elif x == "noskip":
            c["pskip"] = False
        elif x == "novec":
            c["vec"] = False
        else:
            raise SystemExit(f"configuration {name}: unknown part {x}")
    return False, c


def make(m, nmax, name):
    import mfused
    b, c = cfg_of(name)
    return mfused.engine(m, nmax, c, base=b)


def read_ref(t, fam, p, dev):
    """C4b's frozen read of setting fam at p: (det, bits, margin) of network t on the device."""
    import torch
    z, meta, idx = pc.load_read(m1.READ.format(fam), p)
    det = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
    return (torch.as_tensor(det, device=dev), torch.as_tensor(np.unpackbits(z[f"pred{idx[t]}"], axis=1, count=pc.K),
                                                               device=dev),
            torch.as_tensor(z[f"conf{idx[t]}"], device=dev))


def buffers(m, name, xb, n):
    """Today's engine and name's, launch by launch on the same n shots: after each launch, every intermediate buffer's
    first n rows, bit for bit -> [(launch, [buffers that differ])]."""
    import torch
    import pc_fused
    e0, e1 = pc_fused.Batched(m, n), make(m, n, name)
    for e in (e0, e1):                                  # Batched leaves these uninitialized: the same start
        for t in (e.x, e.m1, e.i1, e.m2):
            t.zero_()

    def view(e, b):
        t = e.u[int(b[1])] if b in ("u0", "u1") else getattr(e, b)
        return bits_of(t[:n])
    l0, l1 = e0.launches(xb, n), e1.launches(xb, n)
    if [x for x, _ in l0] != [x for x, _ in l1]:
        raise SystemExit("launch lists differ")
    out = []
    for (ln, f0), (_, f1) in zip(l0, l1):
        f0()
        f1()
        torch.cuda.synchronize()
        out.append((ln, [b for b in BUFS if not torch.equal(view(e0, b), view(e1, b))]))
    del e0, e1
    return out


def kernel_times(e, xb, n, reps=None):
    """Each kernel's time without PDL, launched eagerly in the engine's order (M1's per-kernel measure): the GPU is
    held by a sleep kernel while the launches and CUDA events between them are queued, so that the kernels then run
    back to back; the median over the calls after the first (us per launch and per shot)."""
    import torch
    pdl = e.pdl
    e.pdl = False
    e(xb, n)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e(xb, n)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    it = reps or max(4, min(40, int(0.04 / max(dt, 1e-6))))
    ln = e.launches(xb, n)
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(len(ln) + 1)] for _ in range(it)]
    torch.cuda._sleep(int(5e7))                         # about 20 ms at 2.4 GHz: the queue fills meanwhile
    for i in range(it):
        ev[i][0].record()
        for k, (_, fn) in enumerate(ln):
            fn()
            ev[i][k + 1].record()
    torch.cuda.synchronize()
    out = {}
    for k, (name, _) in enumerate(ln):
        us = float(np.median([1e3 * ev[i][k].elapsed_time(ev[i][k + 1]) for i in range(1, it)]))
        out[name] = {"us": us, "us_shot": us / n}
    out["_calls"] = it - 1
    e.pdl = pdl
    return out


# ------------------------------------------------------------------------------------------------------------ dev
def dev(a):
    import torch
    import pc_speed
    dv, cd = pb.setup()
    path = os.path.join(a.out, "dev.jsonl")
    base = {"step": "dev", "gpu": cd["gpu"], "driver": cd["driver"], "toolchain": toolchain()}
    batches = [int(b) for b in a.batches.split(",")]
    nmax = max(batches)
    names = ["base"] + ([] if a.gs == "none" else [f"new:g{g}:s{s}" + (":seq" if f == "seq" else "")
                                                   for g in a.gs.split(",") for s in a.stages.split(",")
                                                   for f in a.forms.split(",")]) + [x for x in a.extra.split(",") if x]
    for t in a.nets.split(","):
        pr = pb.load(t, dv)
        m = pr.model
        det, rb, rm = read_ref(t, pc.NETS[t][1], 0.003, dv)
        for name in names:
            t0 = time.time()
            row = {**base, "net": t, "cfg": name}
            try:
                e = make(m, nmax, name)
                xb = det[:nmax].contiguous()
                bits, mg = e(xb, nmax)
                torch.cuda.synchronize()
            except Exception as ex:                     # recorded; the sweep goes on
                emit(path, {**row, "error": repr(ex)[:1500]})
                continue
            row["read_differ"] = int((bits != rb[:nmax]).any(1).sum())
            row["read_margin_bits_differ"] = int((bits_of(mg) != bits_of(rm[:nmax])).sum())
            row["kernels"] = [pb.ptx_summary(ln, fn()) for ln, fn in e.launches(xb, nmax)]
            torch.cuda.synchronize()
            row["times"] = {}
            for n in batches:
                row["times"][str(n)] = kernel_times(e, det[:n].contiguous(), n)
            for n in (nmax, 1):                         # the whole call in a CUDA graph (PDL as frozen)
                g = pc_speed.Graph(lambda x, e=e, n=n: e(x, n), n, m.D, dv)
                g.x.copy_(det[:n])
                row.setdefault("graph_us", {})[str(n)] = pb.replay_us(g, target_s=0.3)
                del g
            if name != "base":
                row["buffers"] = {}
                for n in [int(x) for x in a.buf_n.split(",")]:
                    row["buffers"][str(n)] = buffers(m, name, det[:n].contiguous(), n)
                row["ok"] = row["read_differ"] == 0 and row["read_margin_bits_differ"] == 0 and \
                    not any(b for v in row["buffers"].values() for _, b in v)
            row["secs"] = round(time.time() - t0, 1)
            emit(path, row)
            del e
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------- refs
def refs(a):
    import torch
    import pc_fused
    dv, cd = pb.setup()
    os.makedirs(os.path.join(a.out, "refs"), exist_ok=True)
    m1r = json.load(open(os.path.join(M1_REFS, "refs.json")))
    out = {"env": cd, "toolchain": toolchain(), "c3": {}, "anchors": [], "nets": {}}
    c4b = m1.c4b_sets(CONVLINES)
    c3, res = {}, {}
    for fam in pc.SETTINGS:                             # C3's shots: M1's files, checked against M1's record
        for p in m1.C3_P:
            nm = f"c3_{fam}_{p}"
            f = os.path.join(M1_REFS, "refs", f"{nm}.npz")
            if pc.sha256_file(f) != m1r["c3"][nm]["sha256"]:
                raise SystemExit(f"{f}: not the file M1 recorded")
            c3[nm] = np.load(f)
    for t in CONVLINES:
        pr = pb.load(t, dv)
        e = pc_fused.Batched(pr.model, pc.FUSED_READ_B)
        out["nets"][t] = {"sha256": pr.sig["sha256"], "D": pr.model.D, "K": e.K, "S": e.S, "C": e.C, "CB": e.CB,
                          "H": e.H, "NT": e.NT, "NIT": e.NIT, "TPI": e.tpi, "SP": e.SP, "KS": e.KS, "Q": e.Q,
                          "QP": e.QP, "NQT": e.NQT, "NP": e.NP, "NNZ": e.NNZ, "FF": e.pool["FF"],
                          "blocks": len(e.blk), "pdl": e.pdl, "stages": e.stages}
        print(json.dumps({"net": t, **out["nets"][t]}), flush=True)
        for name, det, ref, _ in c4b:                   # anchor: today's engine against the frozen reads
            b, m = m1.reads_engine(e, torch.as_tensor(det, device=dv))
            b, m = b.cpu().numpy(), m.cpu().numpy()
            out["anchors"].append({"what": "today's engine against C4b's frozen read", "net": t, "set": name,
                                   "rows": len(det), "differ": int((b != ref[t][0]).any(1).sum()),
                                   "margin_bits_differ": int((m.view(np.int32) != ref[t][1].view(np.int32)).sum())})
            print(json.dumps(out["anchors"][-1]), flush=True)
        for nm, z in c3.items():
            det = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
            b, m = m1.reads_engine(e, torch.as_tensor(det, device=dv))
            res.setdefault(nm, {})[t] = (np.packbits(b.cpu().numpy(), axis=1), m.cpu().numpy())
            if t == m1.OWN[str(z["family"])]:           # anchor: M1's references of the setting's S55
                out["anchors"].append({"what": "today's engine against M1's C3 reference", "net": t, "set": nm,
                                       "differ": int((res[nm][t][0] != z[f"bits_{t}"]).any(1).sum()),
                                       "margin_bits_differ": int((res[nm][t][1].view(np.int32)
                                                                  != z[f"margin_{t}"].view(np.int32)).sum())})
                print(json.dumps(out["anchors"][-1]), flush=True)
        del e
        torch.cuda.empty_cache()
    for nm, z in c3.items():
        f = os.path.join(a.out, "refs", f"{nm}.npz")
        o = res[nm]
        np.savez(f, det=z["det"], obs=z["obs"], p=z["p"], kind=z["kind"], chunks=z["chunks"], family=z["family"],
                 **{f"bits_{t}": o[t][0] for t in o}, **{f"margin_{t}": o[t][1] for t in o})
        out["c3"][nm] = {"rows": int(len(z["det"])), "sha256": pc.sha256_file(f), "m1_sha256": m1r["c3"][nm]["sha256"],
                         "nets": sorted(o)}
    out["ok"] = all(x["differ"] == 0 and x["margin_bits_differ"] == 0 for x in out["anchors"])
    pb.dump(a.out, "refs.json", out)
    print(json.dumps({"refs_ok": out["ok"], "anchors": len(out["anchors"])}), flush=True)


def c3_sets(refs_dir, nets):
    """C3's shots and today's outputs on them (the refs step) -> m1.c4b_sets' form."""
    out = []
    r = json.load(open(os.path.join(refs_dir, "refs.json")))
    for fam in pc.SETTINGS:
        for p in m1.C3_P:
            nm = f"c3_{fam}_{p}"
            f = os.path.join(refs_dir, "refs", f"{nm}.npz")
            if pc.sha256_file(f) != r["c3"][nm]["sha256"]:
                raise SystemExit(f"{f}: not the file the refs step wrote")
            z = np.load(f)
            ref = {t: (np.unpackbits(z[f"bits_{t}"], axis=1, count=pc.K), z[f"margin_{t}"]) for t in nets
                   if f"bits_{t}" in z.files}
            det = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
            out.append((nm, det, ref, {"kind": int(z["kind"]), "chunks": list(range(int(z["chunks"])))}))
    return out


# --------------------------------------------------------------------------------------------------------- screen
def screen(a):
    """Every --cfgs engine of every --nets network in a CUDA graph (pc_speed.Graph, as the timings run it) at each
    --batches, --passes times over every shot set (C4b's four frozen reads and C3's six sets, 100,000 shots each;
    --rows caps the rows per set at a batch, e.g. 1:500), against the frozen reads and C3's references: every decision
    and every FP32 margin's bit pattern; then at --buf-n shots, launch by launch, every intermediate buffer against
    today's engine -> OUT/screen.jsonl. A configuration passes when nothing differs; --control names configurations
    expected to differ (the deliberate control), whose differences are recorded."""
    import torch
    import pc_speed
    dv, cd = pb.setup()
    path = os.path.join(a.out, "screen.jsonl")
    nets = a.nets.split(",")
    sets = m1.c4b_sets(nets) + c3_sets(a.refs, nets)
    sets = [(nm, torch.as_tensor(det, device=dv), {t: (torch.as_tensor(b, device=dv), torch.as_tensor(mg, device=dv))
                                                   for t, (b, mg) in ref.items()}, dg) for nm, det, ref, dg in sets]
    base = {"step": "screen", "gpu": cd["gpu"], "driver": cd["driver"], "toolchain": toolchain()}
    cap = {int(b): int(r) for b, r in (x.split(":") for x in a.rows.split(","))} if a.rows else {}
    control = set(a.control.split(",")) if a.control else set()
    errors = 0
    for t in nets:
        pr = pb.load(t, dv)
        m = pr.model
        for name in a.cfgs.split(",") + sorted(control):
            for B in [int(b) for b in (a.control_batches if name in control else a.batches).split(",")]:
                t0 = time.time()
                try:
                    e = make(m, B, name)
                    g = pc_speed.Graph(lambda x, e=e, B=B: e(x, B), B, m.D, dv)
                except Exception as ex:                 # recorded; the screen goes on, then exits 1
                    emit(path, {**base, "net": t, "cfg": name, "batch": B, "error": repr(ex)[:400]})
                    errors += 1
                    continue
                res = []
                for ps in range(a.passes):
                    for nm, det, ref, _ in sets:
                        if t not in ref:
                            continue
                        r = min(cap.get(B, len(det)), len(det))
                        bits, mg = g.run(det[:r])
                        rb, rm = ref[t][0][:r], ref[t][1][:r]
                        res.append({"set": nm, "pass": ps, "rows": r, "replays": -(-r // B),
                                    "differ": int((bits != rb).any(1).sum()),
                                    "margin_bits_differ": int((bits_of(mg) != bits_of(rm)).sum())})
                ok = all(x["differ"] == 0 and x["margin_bits_differ"] == 0 for x in res)
                emit(path, {**base, "net": t, "cfg": name, "control": name in control, "batch": B,
                            "tier": e.tier(B) if hasattr(e, "tier") else None, "ok": ok,
                            "rows": sum(x["rows"] for x in res), "sets": res, "secs": round(time.time() - t0, 1)})
                del g, e
                torch.cuda.empty_cache()
            xs = sets[0][1]
            for n in [int(x) for x in (a.control_batches if name in control else a.buf_n).split(",")] if a.buf_n else []:
                bad = buffers(m, name, xs[:n].contiguous(), n)
                emit(path, {**base, "net": t, "cfg": name, "control": name in control, "buffers_n": n,
                            "ok": not any(b for _, b in bad), "buffers": bad})
                torch.cuda.empty_cache()
    if errors:                                          # a configuration that did not build fails the step
        raise SystemExit(f"screen: {errors} configuration(s) raised; their rows in {path} say why")


# ------------------------------------------------------------------------------------------------------------ ptx
def ptx(a):
    """Each --cfgs engine's kernels at each --batches (so each G of the GROUPS table): PTX and TTGIR -> OUT/ptx/, launch
    options, registers, spills, opcode counts, and every new kernel against today's same launch (the normalized PTX's
    diff, the floating-point opcode counts per program: a G-shot kernel should have G times today's) -> OUT/ptx.json."""
    import torch
    dv, cd = pb.setup()
    pr = pb.load(a.net, dv)
    d = os.path.join(a.out, "ptx")
    os.makedirs(d, exist_ok=True)
    z, _, _ = pc.load_read(m1.READ.format(pc.NETS[a.net][1]), 0.003)
    out = {"env": cd, "toolchain": toolchain(), "net": a.net, "kernels": {}}
    names = ["base"] + a.cfgs.split(",")
    for B in [int(b) for b in a.batches.split(",")]:
        det = torch.as_tensor(np.unpackbits(z["det"][:B], axis=1, count=432, bitorder="little"), device=dv)
        for name in names:
            try:
                e = make(pr.model, B, name)
                e(det, B)
            except Exception as ex:                     # recorded; the other configurations go on
                out["kernels"][f"{name}@{B}.error"] = {"error": repr(ex)[:2000]}
                continue
            tier = e.tier(B) if hasattr(e, "tier") else None
            G = dict(tier["G"], pool=tier["G"]["pool"] * tier["qg"]) if tier else {}   # (shot, query tile) pairs
            for ln, fn in e.launches(det, B):
                k = fn()
                asm = k.asm
                key = f"{name}@{B}.{ln}"
                for ext in ("ptx", "ttgir"):
                    with gzip.open(os.path.join(d, f"{key}.{ext}.gz"), "wt") as fh:
                        fh.write(asm[ext])
                md = k.metadata
                kind = "stem" if ln == "stem" else "last" if ln == f"block{len(e.blk) - 1}" else \
                    "block" if ln.startswith("block") else ln
                out["kernels"][key] = {"name": md.name, "num_warps": md.num_warps, "num_stages": md.num_stages,
                                       "shared": md.shared, "regs": getattr(k, "n_regs", None),
                                       "spills": getattr(k, "n_spills", None), "hash": md.hash,
                                       "G": G.get(kind, 1), "opcodes": m1.opcodes(asm["ptx"])}
            torch.cuda.synchronize()
            del e
            torch.cuda.empty_cache()
    diffs = {}
    for new, kn in out["kernels"].items():
        if new.startswith("base@") or new.endswith(".error"):
            continue
        cfgb, ln = new.split(".", 1)
        old = f"base@{cfgb.split('@')[1]}.{ln}"
        if old not in out["kernels"] or kn["hash"] == out["kernels"][old]["hash"]:
            continue
        po, pn = (gzip.open(os.path.join(d, f"{k}.ptx.gz"), "rt").read() for k in (old, new))
        df = list(difflib.unified_diff(m1.normalized(po), m1.normalized(pn), old, new, lineterm="", n=1))
        with gzip.open(os.path.join(d, f"diff.{new}.txt.gz"), "wt") as fh:
            fh.write("\n".join(df) + "\n")
        oo, on = out["kernels"][old]["opcodes"], kn["opcodes"]
        fp = sorted(x for x in set(oo) | set(on) if m1.FP_OPS.match(x))
        diffs[new] = {"against": old, "G": kn["G"],
                      "diff_lines": sum(1 for x in df if x[:1] in "+-" and x[:3] not in ("+++", "---")),
                      "fp_ops": {x: [oo.get(x, 0), on.get(x, 0)] for x in fp},
                      "fp_ratio_not_G": {x: [oo.get(x, 0), on.get(x, 0)] for x in fp
                                         if on.get(x, 0) != kn["G"] * oo.get(x, 0)},
                      "fp_ops_only_old": [x for x in fp if x not in on], "fp_ops_only_new": [x for x in fp if x not in oo]}
    out["diffs"] = diffs
    pb.dump(a.out, "ptx.json", out)
    for k, v in diffs.items():
        print(json.dumps({"kernel": k, "G": v["G"], "diff_lines": v["diff_lines"], "only_old": v["fp_ops_only_old"],
                          "only_new": v["fp_ops_only_new"], "not_G": v["fp_ratio_not_G"],
                          "regs": out["kernels"][k]["regs"], "spills": out["kernels"][k]["spills"],
                          "shared": out["kernels"][k]["shared"]}))


# ----------------------------------------------------------------------------------------------- the timed paths
class EngineGraph:
    """A ConvLines engine (today's or the new one) captured at batch B behind pc_speed.Graph's interface; the batch-1
    path of today's engine is the registered one (pc_speed.FusedGraph), as phase C times it."""

    def __new__(cls, pr, engine, B):
        import torch
        import pc_speed
        if engine == "base" and B == 1:
            return pc_speed.FusedGraph(pr.model, pr.sig["sha256"])
        e = make(pr.model, B, engine)
        g = pc_speed.Graph(lambda x: e(x, B), B, pr.model.D, torch.device("cuda"))
        g.engine = e                                    # the graph replays e's buffers
        return g


def frozen_speed():
    """Phase C's frozen speed components and each setting's lock (Amendment 1), read only; the lock names the file."""
    sp = os.path.join(pc.W4, "out", "speed", "speed_components.json")
    locks = {fam: os.path.join(pc.W4, "out", "amend1", f"locked_{fam}.json") for fam in pc.SETTINGS}
    for fam, f in locks.items():
        if pc.sha256_file(sp) != json.load(open(f))["speed_sha256"]:
            raise SystemExit(f"{sp}: not the speed components frozen in {f}")
    return json.load(open(sp)), locks


def path_of(t, engine, mode, B=None):
    """Network t's timed path in a mode: Cascade on its frozen path (phase C's speed components), a ConvLines network
    on the engine at batch 1 (single) or B (default 4,096) -> (path, batch)."""
    if pc.NETS[t][2] == "cascade":
        return pc.frozen_paths(frozen_speed()[0]["nets"][t])[mode]
    return "fused", 1 if mode == "single" else (B or pc.VERDICT_B)


def build_path(pr, t, engine, path, B, score="minabs"):
    import pc_speed
    if path == "fused":
        return EngineGraph(pr, engine, B)
    return pc_speed.build(pr, path, B, score)


# ----------------------------------------------------------------------------------------------------------- nets
def nets(a):
    """One network alone in one fresh process, phase C's nets protocol (pc_speed.nets, PHASE_C_PLAN.md 5.2) on the
    paths timed here: a ConvLines network on --engine (base: today's, the registered engine at batch 1; new: mfused)
    at every --batches, a Cascade network on its frozen single and amortized paths (compiled@1, compiled@4096). Per
    path and batch: the screen against the accuracy path (pc_speed.nets' agree row: the failures at 0.55 % and the
    routing at the claim points on fresh kind-12 shots of the network's setting; a ConvLines path must be bitwise),
    then gpu_us and host_rt at batch 1 and amortized (resident and host to host) at every batch, with
    bb.timing.gpu_watch's clock record -> --out rows (pc_speed's row format plus engine, toolchain)."""
    import torch
    import pc_speed as ps
    from bb import cascade_data as cdata
    from bb.cascade_eval import Predictor
    from bb.timing import gpu_watch
    dv, env = ps.setup()
    path_, fam, kind = pc.NETS[a.net]
    pr = Predictor(path_, dv, "ema", "bf16")
    D = pr.model.D if kind == "convlines" else 432
    engine = a.engine if kind == "convlines" else "cascade"
    base = {"role": "net", "net": a.net, "kind": kind, "engine": engine, "rep": a.rep, "group": a.group, **env,
            "sha256": pr.sig["sha256"], "tree": os.environ.get("PC_TREE", ""), "toolchain": toolchain()}
    if kind == "convlines":                             # the accuracy path: the reads' fused engine (pc_eval.py)
        import pc_fused
        acc = pc_fused.Batched(pr.model, pc.FUSED_READ_B).run
    else:                                               # the accuracy path: bb.cascade_eval's batches
        def acc(d):
            o = pr(d)
            return o["bits"], o["conf"]

    def evaluated(d):
        out = [acc(d[i:i + pc.CHUNK]) for i in range(0, len(d), pc.CHUNK)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]).float()

    det, obs = cdata.eval_shots(pc.CODE, ps.P_AGREE, range(ps.N_AGREE_B // pc.CHUNK), kind=a.kind, procs=4, family=fam)
    dg, og = torch.as_tensor(det, device=dv), torch.as_tensor(obs, device=dv)
    ref, ref_c = evaluated(dg)
    fail_ref = (ref != og).any(1)
    rq = {}
    for p in pc.RELAY_P:
        d = torch.as_tensor(cdata.eval_shots(pc.CODE, p, range(ps.N_AGREE_B // pc.CHUNK), kind=a.kind, procs=4,
                                             family=fam)[0], device=dv)
        rq[p] = (d,) + evaluated(d)
    if a.paths:                                         # e.g. compiled@512 (lever 2's small graph)
        todo = [(x.split("@")[0], int(x.split("@")[1])) for x in a.paths.split(",")]
    elif kind == "convlines":
        todo = [("fused", int(b)) for b in a.batches.split(",")]
    else:
        todo = [path_of(a.net, None, "single"), path_of(a.net, None, "amortized")]
    for name, B in todo:
        t_build = time.time()
        try:
            gr = build_path(pr, a.net, engine, name, B)
        except Exception as e:                          # recorded, the other paths go on
            ps.emit(a.out, {**base, "path": name, "batch": B, "metric": "error", "error": repr(e)[:300]})
            continue
        row = {**base, "path": name, "batch": B, "prec": pc.precision(name, kind),
               "build_s": round(time.time() - t_build, 1),
               "tier": gr.engine.tier(B) if hasattr(getattr(gr, "engine", None), "tier") else None}
        n_ag = ps.N_AGREE_B1 if B < 64 else ps.N_AGREE_B
        bits, conf = gr.run(dg[:n_ag])
        ft, fe = (bits != og[:n_ag]).any(1), fail_ref[:n_ag]
        only_t, only_e = int((ft & ~fe).sum()), int((~ft & fe).sum())
        differ = int((bits != ref[:n_ag]).any(1).sum())
        c_differ = int((bits_of(conf.float()) != bits_of(ref_c[:n_ag])).sum())
        route, differ_c = {}, 0
        for p, (d, be, ce) in rq.items():
            bt, ct = gr.run(d[:n_ag])
            differ_c += int((bt != be[:n_ag]).any(1).sum())
            c_differ += int((bits_of(ct.float()) != bits_of(ce[:n_ag])).sum())
            ct, ce_ = ct.double().cpu().numpy(), ce[:n_ag].double().cpu().numpy()
            route[str(p)] = {}
            for q in pc.ROUTE_Q:
                tau = float(np.quantile(ce_, q))
                me, mt = ce_ < tau, ct < tau
                route[str(p)][str(q)] = [int(me.sum()), int((mt & ~me).sum()), int((~mt & me).sum())]
        ag = {**row, "metric": "agree", "p": ps.P_AGREE, "kind_seed": a.kind, "shots": n_ag, "differ": differ,
              "differ_claim": differ_c, "conf_differ": c_differ, "bitwise": differ == differ_c == c_differ == 0,
              "fail_timed": int(ft.sum()), "fail_eval": int(fe.sum()), "only_timed": only_t, "only_eval": only_e,
              "route": route}
        ps.emit(a.out, {**ag, "ok": ps.screened(ag)})
        stop = gpu_watch()
        res = {}
        if B == 1:
            res["gpu_us"] = ps.gpu_us(gr)
            res["host_rt"] = ps.quant(ps.host_rt(gr, dg, n=a.n_rt))
        res["amortized"] = ps.amortized(gr, dg, secs=a.secs)
        ps.emit(a.out, {**row, "metric": "time", **res, "clock": stop()})
        del gr
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------ lever 2: the tail rule
def replay_ms(gr, reps=3):
    """Mean GPU time of one replay (ms) from CUDA events around reps back-to-back replays, after one warm-up."""
    import torch
    gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        gr.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / reps


class Tail:
    """Lever 2 (gate finding 5) for one stage's graph: the small graph (the same path at batch b) is built beside the
    full one, and both are timed before the clock starts. A last batch of r < B real rows runs as ceil(r / b) replays
    of the small graph when ceil(r / b) x t_small < t_full, else as one replay of the full graph, padded, as phase C
    runs it. run(rows, put) feeds the chosen graph chunk by chunk in queue order and hands each chunk's outputs to
    put(lo, hi, bits, conf) (host tensors of the chunk's real rows)."""

    def __init__(self, full, small):
        self.full, self.small = full, small
        self.hb = full.bits.new_empty(full.bits.shape, device="cpu").pin_memory()
        self.hc = full.conf.new_empty(full.conf.shape, device="cpu").pin_memory()
        self.t_full = replay_ms(full)
        self.t_small = replay_ms(small) if small is not None else None
        self.used = []

    def plan(self, r, force=None):
        """(graph, chunk) for a last batch of r real rows, by the rule (force: "small" or "full", for the checks)."""
        small = self.small is not None and r < self.full.B and -(-r // self.small.B) * self.t_small < self.t_full
        if force is not None:
            small = force == "small" and self.small is not None and r < self.full.B
        return (self.small, self.small.B) if small else (self.full, self.full.B)

    force = None

    def run(self, rows, r, put, st):
        g, b = self.plan(r, self.force)
        self.used.append({"rows": int(r), "graph_batch": g.B, "replays": -(-r // b)})
        for lo in range(0, r, b):
            hi = min(r, lo + b)
            g.x.zero_()
            g.x[:hi - lo].copy_(rows[lo:hi], non_blocking=True)
            g.replay()
            self.hb[:g.B].copy_(g.bits, non_blocking=True)
            self.hc[:g.B].copy_(g.conf, non_blocking=True)
            st.synchronize()
            put(lo, hi, self.hb[:hi - lo], self.hc[:hi - lo])

    def record(self):
        return {"t_full_ms": self.t_full, "t_small_ms": self.t_small, "batch_full": self.full.B,
                "batch_small": self.small.B if self.small is not None else None, "used": self.used}


class Stream:
    """pc_speed.pipeline's batched loop, unchanged but for the tail: S over the batches of the pinned host rows H
    (n real shots), the rows S routes (conf < t_S in float64) queued on the GPU in shot order, B whenever B_Q of them
    wait and once at the end on the rest (by Tail's rule when lever 2 is on; Cascade alone: its own last batch), B's
    routing to Relay-BP (S -> B -> R: conf < tau2 in float64) handed to relay(indices) -> pred, route (1 = B, 2 = R)."""

    def __init__(self, gS, gB, tail, t_S, tau2, relay, n, alone, keep_conf=False):
        import torch
        import pc_speed as ps
        self.gS, self.gB, self.tail, self.t_S, self.tau2, self.relay, self.alone = gS, gB, tail, t_S, tau2, relay, alone
        self.bS, self.BQ = gS.B, ps.B_Q
        self.hc_s = torch.empty(self.bS).pin_memory()
        self.hb_s = torch.empty(self.bS, pc.K, dtype=torch.uint8).pin_memory()
        self.hc_b = torch.empty(self.BQ).pin_memory()
        self.hb_b = torch.empty(self.BQ, pc.K, dtype=torch.uint8).pin_memory()
        self.pred = np.zeros((n, pc.K), np.uint8)
        self.route = np.zeros(n, np.int8)
        self.n, self.t_b, self.st = n, 0.0, torch.cuda.current_stream()
        self.conf_b = np.full(n, np.nan, np.float32) if keep_conf else None     # B's confidences (checks only)

    def b_out(self, idx, bits, conf):
        self.pred[idx] = bits.numpy()
        if self.conf_b is not None:
            self.conf_b[idx] = conf.numpy()
        if self.relay is not None:
            sel = conf.numpy().astype(np.float64) < self.tau2
            if sel.any():
                ii = idx[sel]
                self.route[ii] = 2
                self.relay(ii)

    def run_b(self, rows, idx, last=False):
        tb = time.perf_counter()
        if last and self.tail is not None:
            self.tail.run(rows, len(idx), lambda lo, hi, bits, conf: self.b_out(idx[lo:hi], bits, conf), self.st)
        else:
            gB = self.gB
            gB.x.zero_()
            gB.x[:len(idx)].copy_(rows)
            gB.replay()
            self.hb_b.copy_(gB.bits, non_blocking=True)
            self.hc_b.copy_(gB.conf, non_blocking=True)
            self.st.synchronize()
            self.b_out(idx, self.hb_b[:len(idx)], self.hc_b[:len(idx)])
        self.t_b += time.perf_counter() - tb

    def run(self, H):
        import torch
        gS, bS, BQ, n = self.gS, self.bS, self.BQ, self.n
        q_rows, q_idx = [], []
        for j in range(len(H)):
            lo, hi = j * bS, min(n, (j + 1) * bS)
            if self.alone and self.tail is not None and j == len(H) - 1 and hi - lo < bS:
                def put(a_, b_, bits, conf, lo=lo):
                    self.pred[lo + a_:lo + b_] = bits.numpy()
                self.tail.run(H[j], hi - lo, put, self.st)
                continue
            gS.x.copy_(H[j], non_blocking=True)
            gS.replay()
            self.hb_s.copy_(gS.bits, non_blocking=True)
            self.hc_s.copy_(gS.conf, non_blocking=True)
            self.st.synchronize()
            self.pred[lo:hi] = self.hb_s[:hi - lo].numpy()
            r = np.nonzero(self.hc_s[:hi - lo].numpy().astype(np.float64) < self.t_S)[0]
            if not len(r):
                continue
            idx = lo + r
            self.route[idx] = 1
            q_rows.append(gS.x[torch.as_tensor(r, device=gS.x.device)])
            q_idx.append(idx)
            while sum(len(x) for x in q_idx) >= BQ:
                rows, idx = torch.cat(q_rows), np.concatenate(q_idx)
                self.run_b(rows[:BQ], idx[:BQ])
                q_rows, q_idx = [rows[BQ:]], [idx[BQ:]]
        if q_idx and sum(len(x) for x in q_idx):
            self.run_b(torch.cat(q_rows), np.concatenate(q_idx), last=True)


# ------------------------------------------------------------------------------------------------------- pipeline
def pipeline(a):
    """pc_speed.pipeline (PHASE_C_PLAN.md 5.3) for the claim points' S -> B and S -> B -> R configurations of
    Amendment 1's lock (read only, with the speed components it froze), and Cascade alone (--config -1), in one fresh
    process: the same harness, shots (kind 12 from chunk 0), routing (the locked thresholds in float64), queue and
    Relay-BP workers, with S on --engine (base: today's paths, frozen; new: mfused at batch 1 in single mode, at 4,096 in
    batched mode) and, with --tail-b b > 0, lever 2's tail rule (Tail) on B's last batch (Cascade alone: on its last
    batch) in batched mode -> --out rows (pc_speed's row format plus engine, tail, toolchain)."""
    import torch
    import pc_speed as ps
    from multiprocessing import get_context
    from bb import cascade_data as cdata
    from bb.cascade_eval import Predictor
    from bb.timing import gpu_watch
    dev, env = ps.setup()
    sp, locks = frozen_speed()
    L = json.load(open(locks[a.setting]))
    alone = a.config < 0
    if alone and a.p not in pc.RELAY_P:
        raise SystemExit(f"--config -1 (Cascade alone) needs --p in {pc.RELAY_P}")
    c = {"type": "alone", "p": a.p, "S": L["B"]} if alone else L["configs"][a.config]
    fam, p, typ = L["family"], c["p"], c["type"]
    if typ not in ("alone", "S-B", "S-B-R") or p not in pc.RELAY_P:
        raise SystemExit(f"config {a.config}: {typ} at {p}, not a claim point's S -> B or S -> B -> R")
    tau = c.get("tau", {})
    det, obs = cdata.eval_shots(pc.CODE, p, range((a.n + pc.CHUNK - 1) // pc.CHUNK), kind=a.kind, procs=4, family=fam)
    det, obs = det[:a.n], obs[:a.n]
    S = Predictor(pc.NETS[c["S"]][0], dev, "ema", "bf16")
    Bn = Predictor(pc.NETS[c["B"]][0], dev, "ema", "bf16") if c.get("B") else None
    for t, pr in ((c["S"], S), (c.get("B"), Bn)):
        if pr is not None and pr.sig["sha256"] != L["nets"][t]:
            raise SystemExit(f"{t}: sha256 differs from the locked network")
    use_relay = typ == "S-B-R"
    t_S = -np.inf if alone else tau["tau1"]
    engine = None if alone else a.engine
    base = {"role": "pipeline", "config": a.config, "type": typ, "p": p, "family": fam, "S": c["S"], "B": c.get("B"),
            "q1": c.get("q1"), "r": c.get("r"), "q": c.get("q"), "score": c.get("score"), "mode": a.mode,
            "rep": a.rep, "kind_seed": a.kind, **env, "locked_sha256": pc.sha256_file(locks[a.setting]),
            "speed_sha256": L["speed_sha256"], "shots": len(det), "tree": os.environ.get("PC_TREE", ""),
            "engine": engine, "tail_b": a.tail_b if a.mode == "batched" else 0, "group": a.group,
            "toolchain": toolchain()}
    D = det.shape[1]

    def s_check(gS):
        """A ConvLines S's graph against C4b's frozen read of this setting at p (32,768 shots): every decision and
        margin bit pattern, in this process (M1's in-process check; the screen is the full one)."""
        if pc.NETS[c["S"]][2] != "convlines":
            return None
        d_, rb, rm = read_ref(c["S"], fam, p, dev)
        n_ = min(len(d_), 32_768 if gS.B > 1 else 2_000)
        b_, m_ = gS.run(d_[:n_])
        return {"rows": n_, "differ": int((b_ != rb[:n_]).any(1).sum()),
                "margin_bits_differ": int((bits_of(m_.float()) != bits_of(rm[:n_])).sum())}

    def net(pr, tag, mode, score="minabs", B=None):
        """The network on its path for this mode: Cascade frozen, a ConvLines network on the engine (alone: its
        amortized path, which is fused@4096 or fused@1024; here only Cascade runs alone)."""
        path, b = path_of(tag, engine, mode)
        b = B or b
        return build_path(pr, tag, engine, path, b, score), f"{path}@{b}" + (f"+{engine}" if path == "fused" else "")

    if a.mode == "single":
        gS, base["path_S"] = net(S, c["S"], "single")
        base["S_check"] = s_check(gS)
        gB, base["path_B"] = net(Bn, c["B"], "single", c.get("score", "minabs")) if Bn is not None else (None, None)
        if use_relay:
            ps.relay_import()
            import pc_relay
            dec, Lm = pc_relay.decoder(fam, p)
            for i in range(3):
                dec.decode(np.ascontiguousarray(det[i]))
        H = torch.as_tensor(det).pin_memory()
        hb, hc = torch.empty(1, pc.K, dtype=torch.uint8).pin_memory(), torch.empty(1).pin_memory()
        st = torch.cuda.current_stream()
        lat, cat, fail = np.empty(len(det)), np.zeros(len(det), np.int8), np.zeros(len(det), bool)
        stop = gpu_watch()
        for i in range(-200, len(det)):                    # 200 untimed shots first
            j = i % len(det)
            t0 = time.perf_counter()
            gS.x.copy_(H[j:j + 1], non_blocking=True)
            gS.replay()
            hb.copy_(gS.bits, non_blocking=True)
            hc.copy_(gS.conf, non_blocking=True)
            st.synchronize()
            k = 0
            if float(hc[0]) < t_S:
                k = 1
                gB.x.copy_(gS.x)
                gB.replay()
                hb.copy_(gB.bits, non_blocking=True)
                hc.copy_(gB.conf, non_blocking=True)
                st.synchronize()
                if typ == "S-B-R" and float(hc[0]) < tau["tau2"]:
                    k = 2
            if k == 2:
                e = dec.decode(np.ascontiguousarray(det[j]))
                pred = (Lm @ e.astype(np.uint8)) % 2
            else:
                pred = hb[0].numpy()
            if i >= 0:
                lat[i] = 1e6 * (time.perf_counter() - t0)
                cat[i] = k
                fail[i] = bool((pred != obs[i]).any())
        out = {**base, "metric": "single", "latency_us": ps.quant(lat), "clock": stop(),
               "frac_B": float((cat >= 1).mean()), "frac_R": float((cat == 2).mean()), "fail": int(fail.sum())}
        for k, name in ((0, "S_only"), (1, "via_B"), (2, "via_R")):
            if (cat == k).any():
                out[f"latency_{name}_us"] = ps.quant(lat[cat == k])
        ps.emit(a.out, out)
        return

    gS, base["path_S"] = net(S, c["S"], "amortized" if alone else "batched")
    base["S_check"] = s_check(gS)
    bS = gS.B
    gB, base["path_B"] = net(Bn, c["B"], "batched", c.get("score", "minabs")) if Bn is not None else (None, None)
    small = None
    if a.tail_b:                                        # lever 2: the same path at the small batch, beside the full
        tag, pr_ = (c["S"], S) if alone else (c["B"], Bn)
        small, base["path_tail"] = net(pr_, tag, "amortized" if alone else "batched", c.get("score", "minabs"),
                                       B=a.tail_b)
    pool = get_context("spawn").Pool(pc.NCPU_RELAY, initializer=ps._relay_init, initargs=(fam, p)) if use_relay \
        else None
    pids = set()
    while pool is not None and len(pids) < pc.NCPU_RELAY:  # every worker has built its decoder before the clock starts
        pids |= set(pool.map(ps._relay_ready, range(2 * pc.NCPU_RELAY), chunksize=1))
    nb = (len(det) + bS - 1) // bS
    H = torch.zeros(nb * bS, D, dtype=torch.uint8)
    H[:len(det)] = torch.as_tensor(det)
    H = H.reshape(nb, bS, D).pin_memory()
    st = torch.cuda.current_stream()
    for _ in range(3):                                      # warm-up of the graphs
        gS.replay()
        if gB is not None:
            gB.replay()
        if small is not None:
            small.replay()
    st.synchronize()
    tail = Tail(gS if alone else gB, small) if small is not None else None     # both timed before the clock starts
    pending = []

    def relay(ii):
        for k in range(0, len(ii), ps.RELAY_TASK):
            jj = ii[k:k + ps.RELAY_TASK]
            pending.append(pool.apply_async(ps._relay_one, ((fam, p, jj, det[jj]),)))
    sm = Stream(gS, gB, tail, t_S, tau.get("tau2"), relay if use_relay else None, len(det), alone)
    stop = gpu_watch()
    t0 = time.perf_counter()
    sm.run(H)
    t_gpu = time.perf_counter() - t0
    pred, route, t_b = sm.pred, sm.route, sm.t_b
    secs = []
    for f in pending:
        idx, pr_, s_ = f.get()
        pred[idx] = pr_
        secs.append(s_)
    wall = time.perf_counter() - t0
    clock = stop()
    if pool is not None:
        pool.close()
        pool.join()
    secs = np.concatenate(secs) if secs else np.zeros(0)
    fail = (pred != obs).any(1)
    ps.emit(a.out, {**base, "metric": "batched", "batch_S": bS, "batch_B": ps.B_Q if gB is not None else None,
                    "relay_workers": pc.NCPU_RELAY if pool else 0, "wall_s": wall, "gpu_phase_s": t_gpu,
                    "b_stage_s": t_b, "per_shot_us": 1e6 * wall / len(det),
                    "per_round_us": 1e6 * wall / (len(det) * pc.R), "throughput_shots_s": len(det) / wall,
                    "frac_B": float((route >= 1).mean()), "frac_R": float((route == 2).mean()),
                    "relay_s_mean": float(secs.mean()) if len(secs) else None, "fail": int(fail.sum()),
                    "clock": clock, "tail": tail.record() if tail is not None else None})


# ------------------------------------------------------------------------------------------------- lever 2's check
class StubS:
    """An S that routes exactly the given shots (conf 0 there, 1 elsewhere; decisions 0), batch after batch, behind
    pc_speed.Graph's interface: the queue's and the tail's bookkeeping under test, not a network."""

    def __init__(self, routed, B, D, dev):
        import torch
        self.B, self.j, self.routed = B, 0, routed
        self.x = torch.zeros(B, D, dtype=torch.uint8, device=dev)
        self.bits = torch.zeros(B, pc.K, dtype=torch.uint8, device=dev)
        self.conf = torch.ones(B, device=dev)

    def replay(self):
        self.conf.fill_(1.0)
        r = self.routed[(self.routed >= self.j * self.B) & (self.routed < (self.j + 1) * self.B)] - self.j * self.B
        if len(r):
            self.conf[r] = 0.0
        self.j += 1


def direct_b(gB, small, rows, BQ):
    """B's outputs on queued rows computed directly: B_Q at a time on the full graph, the rest b at a time on the
    small graph (the queue order and the chunks the tail rule forced to the small graph uses) -> host bits, conf."""
    import torch
    n = len(rows)
    nf = n - n % BQ
    outs = []
    for g, lo, hi, b in [(gB, 0, nf, BQ), (small, nf, n, small.B if small is not None else BQ)]:
        g = g if g is not None else gB
        for k in range(lo, hi, b):
            d = rows[k:min(hi, k + b)]
            g.x.zero_()
            g.x[:len(d)].copy_(d)
            g.replay()
            outs.append((g.bits[:len(d)].cpu(), g.conf[:len(d)].cpu()))
    if not outs:
        return torch.zeros(0, pc.K, dtype=torch.uint8), torch.zeros(0)
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def tail(a):
    """Lever 2's checks for --setting's Cascade (the lock's B) with the small graph at --tail-b, on the card of this
    process (the tolerance screen of the small graph itself is the tailnets step: pc_speed.nets' agree rows, phase C's
    tolerance, which is not bitwise):
      bounds  for every r in --bounds, r rows routed by a stub S over three batches of C4b's shots, through Stream (the
              timed loop) with the last batch forced onto the small graph and then onto the full graph, against the
              same queue computed directly (direct_b): every decision and every confidence bit, the routing to R
              (S -> B -> R's tau2), and the shots that reach B
      claim   every locked S -> B and S -> B -> R configuration at the claim points through Stream with the new S
              engine and the tail forced onto the small graph, on C4b's frozen reads (100,000 shots per p): S's margins
              against the read (bitwise), the routed queue against direct_b (bitwise), and pc_speed.check's batched
              comparison with the read's evaluated computation (pc_route.masks; Relay-BP's seeded decision per shot,
              pc_relay): pc_common.agrees on the failures and on the shots routed to B and to R
    -> OUT/tail_<setting>.json"""
    import torch
    import pc_route
    import pc_speed as ps
    from multiprocessing import get_context
    from bb.cascade_eval import Predictor
    dv, env = ps.setup()
    sp, locks = frozen_speed()
    L = json.load(open(locks[a.setting]))
    fam, tB = L["family"], L["B"]
    ps.relay_import()
    import pc_relay
    out = {"setting": fam, "B": tB, "tail_b": a.tail_b, **env, "tree": os.environ.get("PC_TREE", ""),
           "toolchain": toolchain(), "locked_sha256": pc.sha256_file(locks[fam]), "bounds": [], "claim": []}
    pB = Predictor(pc.NETS[tB][0], dv, "ema", "bf16")
    if pB.sig["sha256"] != L["nets"][tB]:
        raise SystemExit(f"{tB}: sha256 differs from the locked network")
    path, B = pc.frozen_paths(sp["nets"][tB])["batched"]
    gB = ps.build(pB, path, B)
    small = ps.build(pB, path, a.tail_b)
    tl_ = Tail(gB, small)
    out["paths"] = {"full": f"{path}@{B}", "small": f"{path}@{a.tail_b}", "t_full_ms": tl_.t_full,
                    "t_small_ms": tl_.t_small}
    z, meta, idx = pc.load_read(m1.READ.format(fam), 0.003)
    det = torch.as_tensor(np.unpackbits(z["det"], axis=1, count=432, bitorder="little"), device=dv)
    BQ, n3 = ps.B_Q, 3 * ps.B_Q
    H = det[:n3].cpu().reshape(3, ps.B_Q, -1).pin_memory()
    rng = np.random.default_rng(20261011)
    tau2 = 0.5                                          # a stand-in threshold for the R routing of the bounds check
    for r in [int(x) for x in a.bounds.split(",")]:
        routed = np.sort(rng.choice(n3, size=r, replace=False))
        ref_b, ref_c = direct_b(gB, small, det[routed], BQ) if r else (None, None)
        for force in ("small", "full"):
            sm = Stream(StubS(routed, BQ, det.shape[1], dv), gB, tl_, 0.5, tau2, lambda ii: None, n3, False, True)
            tl_.force, tl_.used = force, []
            sm.run(H)
            row = {"r": r, "force": force, "used": list(tl_.used), "to_B": int((sm.route >= 1).sum())}
            if force == "full":
                ref_bf, ref_cf = direct_b(gB, None, det[routed], BQ) if r else (None, None)
            else:
                ref_bf, ref_cf = ref_b, ref_c
            if r:
                got = torch.as_tensor(sm.pred[routed])
                row["differ"] = int((got != ref_bf).any(1).sum())
                row["conf_bits_differ"] = int((sm.conf_b[routed].view(np.int32) != ref_cf.numpy().view(np.int32)).sum())
                row["to_R_differ"] = int(((sm.route[routed] == 2) != (ref_cf.numpy().astype(np.float64) < tau2)).sum())
            row["ok"] = row["to_B"] == r and row.get("differ", 0) == 0 and row.get("to_R_differ", 0) == 0 \
                and row.get("conf_bits_differ", 0) == 0 \
                and sum(u["rows"] for u in row["used"]) == r % BQ
            out["bounds"].append(row)
            print(json.dumps(row), flush=True)
    tl_.force = "small"
    pool = get_context("spawn").Pool(pc.NCPU_RELAY, initializer=ps.relay_import)
    pS = {}
    for p in pc.RELAY_P:
        z, meta, idx = pc.load_read(m1.READ.format(fam), p)
        n = len(z["obs"])
        det_np = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
        det = torch.as_tensor(det_np, device=dv)
        obs = np.unpackbits(z["obs"], axis=1, count=pc.K)
        nb = -(-n // BQ)
        H = torch.zeros(nb * BQ, det.shape[1], dtype=torch.uint8)
        H[:n] = torch.as_tensor(det_np)
        H = H.reshape(nb, BQ, -1).pin_memory()
        rows = []
        for i, c in [(i, c) for i, c in enumerate(L["configs"]) if c["p"] == p and c["type"] in ("S-B", "S-B-R")]:
            if c["S"] not in pS:
                pr = Predictor(pc.NETS[c["S"]][0], dv, "ema", "bf16")
                pS[c["S"]] = (pr, EngineGraph(pr, "new", BQ))
            gS = pS[c["S"]][1]
            bS, cS = gS.run(det)                        # S's margins against the read: bitwise
            s_bitwise = bool(np.array_equal(bS.cpu().numpy(), np.unpackbits(z[f"pred{idx[c['S']]}"], axis=1,
                                                                                count=pc.K))
                             and np.array_equal(bits_of(cS).cpu().numpy(), z[f"conf{idx[c['S']]}"].view(np.int32)))
            tau = c["tau"]
            sm = Stream(gS, gB, tl_, tau["tau1"], tau.get("tau2"), (lambda ii: None) if c["type"] == "S-B-R" else None,
                        n, False, True)
            tl_.used = []
            sm.run(H)
            ii = np.nonzero(cS.cpu().numpy().astype(np.float64) < tau["tau1"])[0]
            ref_b, ref_c = direct_b(gB, small, det[torch.as_tensor(ii, device=dv)], BQ)
            rep_b, rep_c = direct_b(gB, small, det[torch.as_tensor(ii, device=dv)], BQ)    # the graphs' own repeat
            rep = int((rep_b != ref_b).any(1).sum()) + int((bits_of(rep_c) != bits_of(ref_c)).sum())
            q_differ = int((torch.as_tensor(sm.pred[ii]) != ref_b).any(1).sum()) \
                + int((sm.conf_b[ii].view(np.int32) != ref_c.numpy().view(np.int32)).sum())
            toR = sm.route == 2
            q_r_differ = 0
            if c["type"] == "S-B-R":
                q_r_differ = int((toR[ii] != (ref_c.numpy().astype(np.float64) < tau["tau2"])).sum())
            toB_e, toR_e = pc_route.masks(z, meta, idx, c, tau)
            rows.append((i, c, sm, toR, toB_e, toR_e, s_bitwise, q_differ, q_r_differ, list(tl_.used), rep))
        union = np.nonzero(np.any([np.stack([x[3], x[5]]) .any(0) for x in rows], axis=0))[0] if rows else []
        has, rfail = np.zeros(n, bool), np.zeros(n, bool)
        if len(union):                                  # one seeded Relay-BP decision per shot, for every version
            res = pool.map(pc_relay.relay_task, [(fam, p, union[k:k + 10], det_np[union[k:k + 10]])
                                                 for k in range(0, len(union), 10)], chunksize=1)
            has[union] = True
            rfail[union] = (np.concatenate([x[0] for x in res]) != obs[union]).any(1)
        for i, c, sm, toR, toB_e, toR_e, s_bitwise, q_differ, q_r_differ, used, rep in rows:
            fe = pc_route.config_fail(z, idx, c, toB_e, toR_e, (has, rfail, None))
            ft = (sm.pred != obs).any(1)
            ft[toR] = rfail[toR]
            route = {"to_B": ps._cmp(sm.route >= 1, toB_e)}
            if c["type"] == "S-B-R":
                route["to_R"] = ps._cmp(toR, toR_e)
            cf = ps._cmp(ft, fe)
            row = {"p": p, "config": i, "type": c["type"], "S": c["S"], "B": c["B"], "shots": n, "used": used,
                   "S_bitwise": s_bitwise, "queue_differ": q_differ, "queue_to_R_differ": q_r_differ,
                   "repeat_differ": rep, "routed_B": int((sm.route >= 1).sum()),
                   "fail": cf, "routing": route,
                   "pass_tolerance": cf["pass"] and all(v["pass"] for v in route.values()),
                   "pass_queue": s_bitwise and q_differ == 0 and q_r_differ == 0}
            out["claim"].append(row)
            print(json.dumps({k: row[k] for k in ("p", "config", "type", "S_bitwise", "queue_differ", "repeat_differ",
                                                  "queue_to_R_differ", "pass_tolerance", "pass_queue")}
                             | {"fail_e_t": [cf["eval"], cf["timed"]], "only_t_e": [cf["only_timed"], cf["only_eval"]],
                                "used": used}), flush=True)
    pool.close()
    pool.join()
    out["ok_bounds"] = all(x["ok"] for x in out["bounds"])
    out["ok_queue"] = all(x["pass_queue"] for x in out["claim"])
    out["ok_tolerance"] = all(x["pass_tolerance"] for x in out["claim"])
    pb.dump(a.out, f"tail_{fam}.json", out)
    print(json.dumps({k: out[k] for k in ("setting", "ok_bounds", "ok_queue", "ok_tolerance")}), flush=True)


# ---------------------------------------------------------------------------------------------------------- drive
def plan(a):
    """The timed processes of one job, rep-major (each repetition runs every configuration once, in a fixed order) ->
    [(group, argv)]. nets: every network alone, both engines adjacent (a Cascade network once). pipe: per (setting,
    claim point) group, Cascade alone (single; batched, phase C's; batched with the tail rule) and each locked S -> B
    and S -> B -> R configuration in both modes on today's engine and the new one (batched: also with the tail rule)."""
    out = []
    for rep in range(1, a.reps + 1):
        if a.what == "nets":
            for t in a.nets.split(","):
                for e in (("cascade",) if pc.NETS[t][2] == "cascade" else ("base", "new")):
                    out.append((t, ["nets", "--net", t, "--engine", e, "--rep", str(rep), "--out", a.rows]
                                + a.extra.split()))
            continue
        for fam in a.settings.split(","):
            L = json.load(open(frozen_speed()[1][fam]))
            for p in [float(x) for x in a.ps.split(",")]:
                grp = f"{fam}_{p}"
                runs = [(-1, "single", "base", 0), (-1, "batched", "base", 0), (-1, "batched", "base", a.tail_b)]
                for i, c in enumerate(L["configs"]):
                    if c["p"] == p and c["type"] in ("S-B", "S-B-R"):
                        runs += [(i, "single", "base", 0), (i, "single", "new", 0), (i, "batched", "base", 0),
                                 (i, "batched", "new", 0), (i, "batched", "new", a.tail_b)]
                for i, m, e, tb in runs:
                    if tb == 0 and m == "batched" and a.tail_b and i >= 0 and e == "new" and a.no_untailed:
                        continue
                    out.append((grp, ["pipeline", "--setting", fam, "--config", str(i), "--p", str(p), "--mode", m,
                                      "--engine", e, "--tail-b", str(tb), "--rep", str(rep),
                                      "--n", str(a.ns if m == "single" else a.nb), "--out", a.rows] + a.extra.split()))
    return out


def drive(a):
    """Runs plan(a)'s processes one after the other, each fresh (no GPU here); then, once, every group in which a
    process missed the clock rule (bb.timing.gpu_watch: a sample below 97 % of the maximum SM clock, or another process
    on the card) is run again whole, with --group rerun; both runs are kept (gate finding 6). The first process that
    exits non-zero ends the step with exit 1."""
    import subprocess
    py = [sys.executable, os.path.abspath(__file__)]   # this container's venv: each process a fresh interpreter
    todo = plan(a)
    log = os.path.join(os.path.dirname(os.path.abspath(a.rows)), f"drive_{a.what}.jsonl")

    def run(grp, argv, tag):
        t0 = time.time()
        rc = subprocess.run(py + argv + ["--group", tag], check=False).returncode
        with open(log, "a") as fh:
            fh.write(json.dumps({"group": grp, "tag": tag, "argv": argv, "rc": rc, "secs": round(time.time() - t0, 1),
                                 "date": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        if rc:                                          # a failed process fails the step (review finding 1)
            raise SystemExit(f"drive {a.what}: {grp} {tag} {' '.join(argv)} exited {rc}")
    for grp, argv in todo:
        run(grp, argv, "main")
    bad = set()
    for r in (json.loads(x) for x in open(a.rows)) if os.path.exists(a.rows) else ():
        if r.get("group") == "main" and r.get("metric") in ("time", "single", "batched") \
                and not (r.get("clock") or {}).get("accepted", False):
            bad.add(f"{r['family']}_{r['p']}" if r.get("role") == "pipeline" else r["net"])
    with open(log, "a") as fh:
        fh.write(json.dumps({"clock_flagged_groups": sorted(bad)}) + "\n")
    print(json.dumps({"clock_flagged_groups": sorted(bad)}), flush=True)
    for grp, argv in todo:
        if grp in bad:
            run(grp, argv, "rerun")


# --------------------------------------------------------------------------------------------------------- report
C3_JOB, C7_JOB = "CJ30", "CJ39"           # phase C's timing jobs (RESULTS.md 4.1)
TARGETS = {"RTX PRO 6000": {0.002: 0.50, 0.003: 1.00},   # S -> B batched, us per round, at the claim points
           "H100": {0.002: 0.60, 0.003: 0.90}}          # (M2.md 5.4, 9.5)


def jl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def slowest(rows, val):
    """The slowest-process statistic (phase C's): (max of val over rows, flagged: any row missed the clock rule,
    processes)."""
    vs = [(val(r), not (r.get("clock") or {}).get("accepted", False)) for r in rows]
    vs = [(v, f) for v, f in vs if v is not None]
    if not vs:
        return None
    return {"value": max(v for v, _ in vs), "flagged": any(f for _, f in vs), "n": len(vs),
            "flagged_n": sum(f for _, f in vs), "min": min(v for v, _ in vs)}


def fmt(s, d=3, scale=1.0):
    if s is None:
        return "-"
    return f"{s['value'] * scale:,.{d}f}{' †' if s['flagged'] else ''}" + ("" if s["n"] >= 3 else f" (n={s['n']})")


def ratio(a, b, d=2):
    """a / b with the flags of both (gate finding 6)."""
    if a is None or b is None:
        return "-"
    return f"{a['value'] / b['value']:.{d}f}{' †' if a['flagged'] or b['flagged'] else ''}"


def report(a):
    """The milestone's tables from the job directories (--dirs, comma-separated) and phase C's rows (read only):
    the screen, the PTX check, the tail checks, the networks alone and the pipelines against phase C's values and
    against today's engine in the same job, S -> B against the targets, lever 2's use in the timed processes, the clock
    record and the timed paths' in-process checks, with every clock flag carried into each aggregate and ratio
    -> OUT/summary.json, OUT/summary.md (no GPU)."""
    dirs = [d for d in a.dirs.split(",") if d]
    out, md = {"dirs": dirs}, []
    scr = [r for d in dirs for r in jl(os.path.join(d, "screen.jsonl"))]
    if scr:
        md.append("## Screen\n")
        md.append("| net | cfg | control | batch | rows | decisions differing | margin bit patterns differing | ok |")
        md.append("|---|---|---|---:|---:|---:|---:|---|")
        tot = {}
        for r in scr:
            if "batch" in r and "sets" in r:
                dif = sum(x["differ"] for x in r["sets"])
                mb = sum(x["margin_bits_differ"] for x in r["sets"])
                md.append(f"| {r['net']} | {r['cfg']} | {r.get('control')} | {r['batch']} | {r['rows']:,} | {dif:,} | "
                          f"{mb:,} | {r['ok']} |")
                k = (r["cfg"], r.get("control"))
                t = tot.setdefault(k, {"rows": 0, "differ": 0, "margin_bits_differ": 0, "rows_ok": 0, "n": 0})
                t["rows"] += r["rows"]
                t["differ"] += dif
                t["margin_bits_differ"] += mb
                t["n"] += 1
                t["rows_ok"] += bool(r["ok"])
            elif "buffers_n" in r:
                k = (r["cfg"], r.get("control"))
                t = tot.setdefault(k, {"rows": 0, "differ": 0, "margin_bits_differ": 0, "rows_ok": 0, "n": 0})
                t.setdefault("buffers", []).append({"net": r["net"], "n": r["buffers_n"], "ok": r["ok"],
                                                    "bad": [x for x in r["buffers"] if x[1]]})
        out["screen"] = {f"{k[0]}{' (control)' if k[1] else ''}": v for k, v in tot.items()}
        md.append("")
        for k, v in out["screen"].items():
            bad = [x for x in v.get("buffers", []) if not x["ok"]]
            md.append(f"- {k}: {v['rows']:,} shot evaluations in {v['n']} (net, batch) rows, {v['differ']:,} decisions "
                      f"and {v['margin_bits_differ']:,} margin bit patterns differing; buffers launch by launch: "
                      f"{len(v.get('buffers', [])) - len(bad)} of {len(v.get('buffers', []))} bitwise"
                      + (f" (differing: {bad[:4]})" if bad else ""))
        md.append("")
    px = [json.load(open(os.path.join(d, "ptx.json"))) for d in dirs if os.path.exists(os.path.join(d, "ptx.json"))]
    if px:
        md.append("## PTX against today's kernels\n")
        md.append("| kernel | G | regs | spills | shared | arithmetic opcodes not G times today's (today, new) | "
                  "data movement (ldmatrix, stmatrix, shfl) not G times | only today's | only new |")
        md.append("|---|---:|---:|---:|---:|---|---|---|---|")
        mv = re.compile(r"^(ldmatrix|stmatrix|shfl)")
        for p in px:
            for k, v in p["diffs"].items():
                kn = p["kernels"][k]
                ar = {x: y for x, y in v["fp_ratio_not_G"].items() if not mv.match(x)}
                dm = {x: y for x, y in v["fp_ratio_not_G"].items() if mv.match(x)}
                md.append(f"| {k} | {v['G']} | {kn['regs']} | {kn['spills']} | {kn['shared']} | {ar or '-'} | "
                          f"{dm or '-'} | {v['fp_ops_only_old'] or '-'} | {v['fp_ops_only_new'] or '-'} |")
        md.append("")
    tails = [json.load(open(os.path.join(d, f))) for d in dirs for f in sorted(os.listdir(d)) if f.startswith("tail_")
             and f.endswith(".json")] if dirs else []
    if tails:
        md.append("## Lever 2's checks\n")
        for t in tails:
            md.append(f"- {t['setting']} ({t['B']}, small graph {t['paths']['small']}, full {t['paths']['full']}; "
                      f"replays {t['paths']['t_small_ms']:.1f} and {t['paths']['t_full_ms']:.1f} ms): bounds "
                      f"{sum(x['ok'] for x in t['bounds'])} of {len(t['bounds'])} pass; queue against the direct "
                      f"computation {'bitwise' if t['ok_queue'] else 'DIFFERS'}; tolerance against the read "
                      f"{'passes' if t['ok_tolerance'] else 'FAILS'}")
            for c in t["claim"]:
                md.append(f"  - p {c['p']}, config {c['config']} ({c['type']}): routed to B {c['routed_B']:,}, tail "
                          f"{c['used']}, S bitwise {c['S_bitwise']}, queue differ {c['queue_differ']}, repeat differ "
                          f"{c['repeat_differ']}; failures eval/timed {c['fail']['eval']}/{c['fail']['timed']} "
                          f"(only timed {c['fail']['only_timed']}, only eval {c['fail']['only_eval']}); routing "
                          + "; ".join(f"{k} {v['eval']}/{v['timed']} ({v['only_timed']}, {v['only_eval']})"
                                      for k, v in c["routing"].items()) + f"; pass {c['pass_tolerance']}")
        for r in (r for d in dirs for r in jl(os.path.join(d, "tailnets.jsonl")) if r.get("metric") == "time"):
            md.append(f"- {r['net']} {r['path']}@{r['batch']} alone (phase C's nets protocol, rep {r['rep']}): "
                      f"{r['amortized']['h2h']:.3f} us per shot host to host, {r['amortized']['dev']:.3f} resident"
                      + ("" if (r.get("clock") or {}).get("accepted", False) else " †"))
        md.append("")
        out["tail"] = tails
    pcr = jl(os.path.join(pc.W4, "out", "speed", "rows.jsonl"))
    nets_ = [r for d in dirs for r in jl(os.path.join(d, "nets.jsonl"))]
    if nets_:
        md.append("## Networks alone (phase C's nets protocol; slowest of the processes; † a process missed the clock "
                  "rule)\n")
        md.append("| net | engine | group | batch | single mean (us) | p99.9 (us) | amortized h2h (us/shot) | per round "
                  "| resident (us/shot) | screen |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        agg = {}
        for r in nets_:
            if r.get("metric") != "time":
                continue
            for g in (r.get("group", "main"), "all"):          # all: every process of the job (phase C's rule)
                agg.setdefault((r["net"], r["engine"], g, r["path"], r["batch"]), []).append(r)
        ok = {}
        for r in nets_:
            if r.get("metric") == "agree":
                ok.setdefault((r["net"], r["engine"], r["path"], r["batch"]), []).append(bool(r["ok"]))
        res = {}
        order = CONVLINES + ("C14", "C14p")
        for (t, e, g, path, B), rows in sorted(agg.items(), key=lambda kv: (
                order.index(kv[0][0]) if kv[0][0] in order else 99, kv[0][1], {"main": 0, "rerun": 1}.get(kv[0][2], 2),
                kv[0][4])):
            h = slowest(rows, lambda r: r["amortized"]["h2h"])
            d_ = slowest(rows, lambda r: r["amortized"]["dev"])
            s1 = slowest(rows, lambda r: r.get("host_rt", {}).get("mean"))
            p9 = slowest(rows, lambda r: r.get("host_rt", {}).get("p999"))
            sc = ok.get((t, e, path, B), [])
            res[f"{t}|{e}|{g}|{path}@{B}"] = {"h2h": h, "dev": d_, "single": s1, "p999": p9,
                                              "screen_ok": all(sc) and len(sc) > 0, "screen_n": len(sc)}
            md.append(f"| {t} | {e} | {g} | {path}@{B} | {fmt(s1, 1)} | {fmt(p9, 1)} | {fmt(h)} | "
                      f"{fmt(h, 3, 1 / pc.R)} | {fmt(d_)} | {sum(sc)}/{len(sc)} |")
        out["nets"] = res
        c3 = {}
        rtx = all("RTX PRO 6000" in r.get("gpu", "") for r in nets_)  # C3 and C7 ran on the RTX PRO 6000 only
        for r in pcr if rtx else ():
            if r.get("session") == C3_JOB and r.get("role") == "net" and r.get("metric") == "time":
                c3.setdefault((r["net"], r["path"], r["batch"]), []).append(r)
        if not rtx:
            md.append("\nThese rows come from another card than phase C's C3 (the RTX PRO 6000): no C3 value is "
                      "compared with them.")
        md.append("\nPhase C's C3 (job CJ30) on the same paths, slowest of 3: " + "; ".join(
            f"{t} fused@{B} {fmt(slowest(c3.get((t, 'fused', B), []), lambda r: r['amortized']['h2h']))}"
            for t in CONVLINES for B in (1024, 4096)) + "; single " + "; ".join(
            f"{t} {fmt(slowest(c3.get((t, 'fused', 1), []), lambda r: r.get('host_rt', {}).get('mean')), 1)}"
            for t in CONVLINES) + "; C14 compiled@4096 " +
            fmt(slowest(c3.get(("C14", "compiled", 4096), []), lambda r: r["amortized"]["h2h"])) + ".\n")
        md.append("The new engine against today's, amortized host to host (us per shot; slowest of every process of "
                  "the job and of C3's three; speed-up = the other's time / the new engine's; † carried from either "
                  "side):\n")
        md.append("| net | batch | phase C's C3 | today's, this job | new | new, per round | speed-up over today's | "
                  "speed-up over C3 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in CONVLINES:
            for B in pc.BATCHES:
                o, n = (res.get(f"{t}|{e}|all|fused@{B}", {}).get("h2h") for e in ("base", "new"))
                if n is None:
                    continue
                c = slowest(c3.get((t, "fused", B), []), lambda r: r["amortized"]["h2h"])
                md.append(f"| {t} | {B} | {fmt(c)} | {fmt(o)} | {fmt(n)} | {fmt(n, 3, 1 / pc.R)} | {ratio(o, n)} | "
                          f"{ratio(c, n)} |")
        md.append("\nOne shot at a time (batch 1: the host round trip's mean and p99.9 in us; the same rules):\n")
        md.append("| net | phase C's C3 mean | today's mean | new mean | speed-up over today's | speed-up over C3 | "
                  "today's p99.9 | new p99.9 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in CONVLINES:
            o, n = (res.get(f"{t}|{e}|all|fused@1", {}) for e in ("base", "new"))
            if not n:
                continue
            c = slowest(c3.get((t, "fused", 1), []), lambda r: r.get("host_rt", {}).get("mean"))
            md.append(f"| {t} | {fmt(c, 1)} | {fmt(o.get('single'), 1)} | {fmt(n['single'], 1)} | "
                      f"{ratio(o.get('single'), n['single'])} | {ratio(c, n['single'])} | {fmt(o.get('p999'), 1)} | "
                      f"{fmt(n['p999'], 1)} |")
        md.append("\nCascade alone in this job against C3: " + "; ".join(
            f"{t} compiled@4096 {fmt(res.get(f'{t}|cascade|all|compiled@4096', {}).get('h2h'))} (C3 "
            f"{fmt(slowest(c3.get((t, 'compiled', 4096), []), lambda r: r['amortized']['h2h']))}), compiled@1 single "
            f"mean {fmt(res.get(f'{t}|cascade|all|compiled@1', {}).get('single'), 1)} (C3 "
            f"{fmt(slowest(c3.get((t, 'compiled', 1), []), lambda r: r.get('host_rt', {}).get('mean')), 1)})"
            for t in ("C14", "C14p") if f"{t}|cascade|all|compiled@4096" in res) + ".\n")
    pipe = [r for d in dirs for r in jl(os.path.join(d, "pipe.jsonl"))]
    if pipe:
        md.append("## Pipelines (slowest of the processes; † a process missed the clock rule; speed-up = Cascade "
                  "alone's / the decoder's, same group of the same job)\n")
        md.append("| setting, p | decoder | S engine | tail | group | single mean (us) | per round | p99.9 | batched "
                  "(us/shot) | per round | to B, to R | speed-up single | speed-up batched | the same, against "
                  "Cascade alone with the tail rule |")
        md.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")
        agg = {}
        for r in pipe:
            k = (r["family"], r["p"], r["config"], r["type"], r.get("engine"), r.get("tail_b", 0))
            for g in (r.get("group", "main"), "all"):          # all: every process of the job (phase C's rule)
                agg.setdefault(k + (g, r["metric"]), []).append(r)
        c7 = {}
        rtx = all("RTX PRO 6000" in r.get("gpu", "") for r in pipe)
        for r in pcr if rtx else ():
            if r.get("session") == C7_JOB and r.get("role") == "pipeline":
                c7.setdefault((r["family"], r["p"], r["config"], r["metric"]), []).append(r)
        res = {}
        gord = {"main": 0, "rerun": 1, "all": 2}
        keys = sorted({k[:-1] for k in agg}, key=lambda k: (pc.SETTINGS.index(k[0]), k[1], gord.get(k[6], 3), k[2],
                                                              str(k[4]), k[5]))
        for k in keys:
            fam, p, cfg, typ, e, tb, g = k
            sg = slowest(agg.get(k + ("single",), []), lambda r: r["latency_us"]["mean"])
            s9 = slowest(agg.get(k + ("single",), []), lambda r: r["latency_us"]["p999"])
            bt = slowest(agg.get(k + ("batched",), []), lambda r: r["per_shot_us"])
            ca_s = slowest(agg.get((fam, p, -1, "alone", None, 0, g, "single"), []), lambda r: r["latency_us"]["mean"])
            ca_b = slowest(agg.get((fam, p, -1, "alone", None, 0, g, "batched"), []), lambda r: r["per_shot_us"])
            ca_t = slowest(agg.get((fam, p, -1, "alone", None, max(x[5] for x in agg), g, "batched"), []),
                           lambda r: r["per_shot_us"])
            fr = agg.get(k + ("batched",), [])
            fb = f"{np.mean([r['frac_B'] for r in fr]):.2%}, {np.mean([r['frac_R'] for r in fr]):.2%}" if fr else "-"
            name = "Cascade alone" if cfg < 0 else {"S-B": "S -> B", "S-B-R": "S -> B -> R"}[typ]
            res["|".join(map(str, k))] = {"single": sg, "p999": s9, "batched": bt, "cascade_single": ca_s,
                                          "cascade_batched": ca_b, "cascade_batched_tail": ca_t}
            md.append(f"| {fam} {p} | {name} (config {cfg}) | {e or '-'} | {tb or '-'} | {g} | {fmt(sg, 1)} | "
                      f"{fmt(sg, 2, 1 / pc.R)} | {fmt(s9, 1)} | {fmt(bt)} | {fmt(bt, 3, 1 / pc.R)} | {fb} | "
                      f"{ratio(ca_s, sg) if cfg >= 0 else '-'} | {ratio(ca_b, bt) if cfg >= 0 else '-'} | "
                      f"{ratio(ca_t, bt) if cfg >= 0 else '-'} |")
        out["pipe"] = res
        mine = {(k[0], k[1], k[2]) for k in agg}
        if not rtx:
            md.append("\nThese rows come from another card than phase C's C7 (the RTX PRO 6000): no C7 value is "
                      "compared with them.")
        md.append("\nPhase C's C7 (job CJ39), slowest of 3, per shot (per round = / 6): " + "; ".join(
            f"{fam} {p} config {i}: single {fmt(slowest(c7.get((fam, p, i, 'single'), []), lambda r: r['latency_us']['mean']), 1)}, "
            f"batched {fmt(slowest(c7.get((fam, p, i, 'batched'), []), lambda r: r['per_shot_us']))}"
            for fam in pc.SETTINGS for p in pc.RELAY_P for i in sorted({k[2] for k in c7 if k[0] == fam and k[1] == p})
            if (fam, p, i) in mine) + "\n")
        out["c7"] = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": slowest(v, (lambda r: r["latency_us"]["mean"]) if k[3] == "single"
                                                            else (lambda r: r["per_shot_us"])) for k, v in c7.items()}
        tbm = max(x[5] for x in agg)

        def pg(fam, p, cfg, typ, e, tb, what):                 # every process of the job
            return res.get(f"{fam}|{p}|{cfg}|{typ}|{e}|{tb}|all", {}).get(what)

        def c7s(fam, p, cfg, m):
            return out["c7"].get(f"{fam}|{p}|{cfg}|{m}")
        md.append("Against today's engine in this job and phase C's C7 (us per shot; slowest of every process of the job "
                  "and of C7's three; speed-up = the other's time / this one's; † carried from either side):\n")
        md.append("| setting, p | decoder (config) | mode | C7 | today's | new | new, lever 2 | per round: today's, new, "
                  "new with lever 2 | today's / new | today's / (new, lever 2) | C7 / new | C7 / (new, lever 2) |")
        md.append("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        for fam in pc.SETTINGS:
            for p in pc.RELAY_P:
                for cfg, typ in sorted({(k[2], k[3]) for k in agg if k[0] == fam and k[1] == p and k[2] >= 0}):
                    for m in ("single", "batched"):
                        c, o, n = c7s(fam, p, cfg, m), pg(fam, p, cfg, typ, "base", 0, m), pg(fam, p, cfg, typ, "new", 0, m)
                        lv = pg(fam, p, cfg, typ, "new", tbm, m) if m == "batched" else None
                        d, dr = (1, 2) if m == "single" else (3, 3)
                        md.append(f"| {fam} {p} | {'S -> B' if typ == 'S-B' else 'S -> B -> R'} ({cfg}) | {m} | "
                                  f"{fmt(c, d)} | {fmt(o, d)} | {fmt(n, d)} | {fmt(lv, d)} | "
                                  + ", ".join(fmt(x, dr, 1 / pc.R) for x in (o, n, lv)) + f" | {ratio(o, n)} | "
                                  f"{ratio(o, lv)} | {ratio(c, n)} | {ratio(c, lv)} |")
        md.append("\nCascade alone beside them (us per shot; the same rules; per round = / 6):\n")
        md.append("| setting, p | mode | C7 | this job, phase C's way | this job, with the tail rule | phase C's way, this "
                  "job / C7 |")
        md.append("|---|---|---:|---:|---:|---:|")
        for fam in pc.SETTINGS:
            for p in pc.RELAY_P:
                for m in ("single", "batched"):
                    c, o = c7s(fam, p, -1, m), pg(fam, p, -1, "alone", None, 0, m)
                    if o is None:
                        continue
                    lv = pg(fam, p, -1, "alone", None, tbm, m) if m == "batched" else None
                    d = 1 if m == "single" else 3
                    md.append(f"| {fam} {p} | {m} | {fmt(c, d)} | {fmt(o, d)} | {fmt(lv, d)} | {ratio(o, c, 3)} |")
        card = "H100" if any("H100" in r.get("gpu", "") for r in pipe) else "RTX PRO 6000"

        def within(s, T):
            return "-" if s is None else ("yes" if s["value"] / pc.R <= T else "no") + (" †" if s["flagged"] else "")

        def span(s):                                            # the fastest and the slowest process, per round
            return "-" if s is None else f"{s['min'] / pc.R:.3f}-{s['value'] / pc.R:.3f}"
        md.append(f"\nS -> B per round against the targets ({card}, M2.md; us per round, slowest of every "
                  "process of the job, † carried; range: the fastest and the slowest process):\n")
        md.append("| setting | p | config | target | today's | new | new, range | new, lever 2 | new, lever 2, range | "
                  "new within the target | new, lever 2 within the target |")
        md.append("|---|---:|---:|---:|---:|---:|---|---:|---|---|---|")
        for fam in pc.SETTINGS:
            for p in pc.RELAY_P:
                for cfg in sorted({k[2] for k in agg if k[0] == fam and k[1] == p and k[3] == "S-B"}):
                    o, n, lv = (pg(fam, p, cfg, "S-B", e, tb, "batched") for e, tb in (("base", 0), ("new", 0),
                                                                                       ("new", tbm)))
                    T = TARGETS[card][p]
                    md.append(f"| {fam} | {p} | {cfg} | {T:.2f} | {fmt(o, 3, 1 / pc.R)} | {fmt(n, 3, 1 / pc.R)} | "
                              f"{span(n)} | {fmt(lv, 3, 1 / pc.R)} | {span(lv)} | {within(n, T)} | {within(lv, T)} |")
        md.append("\nLever 2 in the timed processes (the last batch's rows, the graph's batch and its replays; each "
                  "process timed one replay of each graph before its clock started):\n")
        used = {}
        for r in pipe:
            if r.get("metric") == "batched" and r.get("tail"):
                u = used.setdefault((r["family"], r["p"], r["config"], r.get("engine"), r["tail_b"]),
                                    {"used": set(), "full": [], "small": []})
                u["used"] |= {(x["rows"], x["graph_batch"], x["replays"]) for x in r["tail"]["used"]}
                u["full"].append(r["tail"]["t_full_ms"])
                u["small"].append(r["tail"]["t_small_ms"])
        for (fam, p, cfg, e, tb), u in sorted(used.items(), key=lambda kv: (pc.SETTINGS.index(kv[0][0]), kv[0][1],
                                                                             kv[0][2])):
            md.append(f"- {fam} {p} {'Cascade alone' if cfg < 0 else f'config {cfg}'}: "
                      + "; ".join(f"{a_:,} rows on the graph at {b_:,}, {c_} replay{'s' if c_ > 1 else ''}"
                                  for a_, b_, c_ in sorted(u["used"]))
                      + f" ({len(u['full'])} processes; one replay {min(u['small']):.1f}-{max(u['small']):.1f} ms small, "
                        f"{min(u['full']):.1f}-{max(u['full']):.1f} ms full)")
        md.append("")
    timed = [(os.path.basename(d.rstrip("/")), r) for d in dirs for f in ("nets.jsonl", "tailnets.jsonl", "pipe.jsonl")
             for r in jl(os.path.join(d, f)) if r.get("metric") in ("time", "single", "batched")]
    if timed:
        md.append("## Clock record (bb.timing.gpu_watch: a timed row is flagged † when a sample fell below 97 % of the "
                  "maximum SM clock or another process was on the card; clock = the lowest sample / the maximum)\n")
        for d in dirs:
            for f in ("drive_nets.jsonl", "drive_pipe.jsonl"):
                for r in jl(os.path.join(d, f)):
                    if "clock_flagged_groups" in r:
                        md.append(f"- {os.path.basename(d.rstrip('/'))}: groups with a flagged row in the main loop, "
                                  f"each re-run once whole: {', '.join(r['clock_flagged_groups']) or 'none'}")
        md.append("\n| job | group | timed rows | flagged | lowest clock | lowest clock of the unflagged rows | rows with "
                  "another process |")
        md.append("|---|---|---:|---:|---:|---:|---:|")
        byjob, bywhat = {}, {}
        for job, r in timed:
            ck = r.get("clock") or {}
            fl = not ck.get("accepted", False)
            if r.get("role") == "pipeline":
                what = f"{r['family']} {r['p']} " + ("Cascade alone" if r["config"] < 0 else
                                                     f"config {r['config']} ({r['type']})") + f", {r['mode']}" \
                    + (f", {r['engine']}" if r.get("engine") else "") + (f", tail {r['tail_b']}" if r.get("tail_b") else "")
            else:
                what = f"{r['net']} {r['engine']} {r['path']}@{r['batch']}"
            for k, dct in (((job, r.get("group", "main")), byjob), ((job, what, r.get("group", "main")), bywhat)):
                s = dct.setdefault(k, {"n": 0, "flagged": 0, "clock": [], "clock_ok": [], "other": 0})
                s["n"] += 1
                s["flagged"] += fl
                if ck.get("clock_min_ratio") is not None:
                    s["clock"].append(ck["clock_min_ratio"])
                    if not fl:
                        s["clock_ok"].append(ck["clock_min_ratio"])
                s["other"] += bool(ck.get("other_pids"))
        for (job, g), s in sorted(byjob.items()):
            md.append(f"| {job} | {g} | {s['n']} | {s['flagged']} | {min(s['clock'], default=float('nan')):.4f} | "
                      f"{min(s['clock_ok'], default=float('nan')):.4f} | {s['other']} |")
        md.append("\nEvery timed configuration with a flagged row (rows flagged / rows, the lowest clock):\n")
        md.append("| job | what | group | flagged | lowest clock | rows with another process |")
        md.append("|---|---|---|---:|---:|---:|")
        for (job, what, g), s in bywhat.items():
            if s["flagged"]:
                md.append(f"| {job} | {what} | {g} | {s['flagged']} / {s['n']} | "
                          f"{min(s['clock'], default=float('nan')):.4f} | {s['other']} |")
        md.append("")
        out["clock"] = {"|".join(k): {x: v for x, v in s.items() if x not in ("clock", "clock_ok")}
                        | {"clock_min": min(s["clock"], default=None)} for k, s in bywhat.items()}
    sc = [r for d in dirs for r in jl(os.path.join(d, "pipe.jsonl")) if r.get("S_check")]
    ag = [r for d in dirs for f in ("nets.jsonl", "tailnets.jsonl") for r in jl(os.path.join(d, f))
          if r.get("metric") == "agree"]
    if sc or ag:
        md.append("## In-process checks of the timed paths\n")
        if sc:
            ok_ = [r for r in sc if r["S_check"]["differ"] == 0 and r["S_check"]["margin_bits_differ"] == 0]
            md.append(f"- S against C4b's frozen read inside each timed pipeline process: {len(ok_)} of {len(sc)} "
                      f"processes bitwise (decisions and margin bit patterns; "
                      f"{sorted({r['S_check']['rows'] for r in sc})} shots)")
        per = {}
        for r in ag:
            s = per.setdefault((r["net"], r.get("engine"), f"{r['path']}@{r['batch']}"),
                               {"n": 0, "ok": 0, "bitwise": 0, "differ": [], "fail": []})
            s["n"] += 1
            s["ok"] += bool(r["ok"])
            s["bitwise"] += bool(r["bitwise"])
            s["differ"].append(r["differ"])
            s["fail"].append((r["fail_timed"], r["fail_eval"]))
        cl = [s for (t, _, _), s in per.items() if pc.NETS[t][2] == "convlines"]
        if cl:
            md.append(f"- phase C's agree row before each ConvLines timing (bitwise required): "
                      f"{sum(s['bitwise'] for s in cl)} of {sum(s['n'] for s in cl)} bitwise, "
                      f"{sum(s['ok'] for s in cl)} passed")
        for (t, e, pth), s in sorted(per.items()):
            if pc.NETS[t][2] == "cascade":
                md.append(f"- {t} {pth} against its accuracy path (phase C's tolerance): {s['ok']} of {s['n']} pass, "
                          f"{s['bitwise']} bitwise; decisions differing {min(s['differ']):,}-{max(s['differ']):,}; "
                          f"failures timed/evaluated " + ", ".join(f"{a_:,}/{b_:,}" for a_, b_ in sorted(set(s["fail"]))))
        md.append("")
    pb.dump(a.out, "summary.json", out)
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:80]))


def gate(a):
    """Exit 0 only if OUT's checks passed: its screen (required with --need-screen) with every row of every
    configuration but the control bitwise and every control configuration differing somewhere (the screen sees what it
    should), and its tail checks (tail_<setting>.json, one for each of --need-tail's settings) all passed; OUT must hold
    at least one of the two. The timing jobs depend on it (sbatch --dependency=afterok)."""
    rows = jl(os.path.join(a.out, "screen.jsonl"))
    bad = [r for r in rows if not r.get("control") and not r.get("ok")]
    ctl = {}
    for r in rows:
        if r.get("control"):
            ctl[r["cfg"]] = ctl.get(r["cfg"], False) or not r.get("ok")
    tails = [json.load(open(os.path.join(a.out, f))) for f in sorted(os.listdir(a.out))
             if f.startswith("tail_") and f.endswith(".json")]
    tbad = [t["setting"] for t in tails if not (t["ok_bounds"] and t["ok_queue"] and t["ok_tolerance"])]
    missing = (["screen"] if a.need_screen and not rows else []) \
        + [x for x in a.need_tail.split(",") if x and x not in {t["setting"] for t in tails}]
    ok = bool(rows or tails) and not bad and all(ctl.values()) and not tbad and not missing
    print(json.dumps({"gate": ok, "rows": len(rows), "failing": [(r.get("net"), r.get("cfg"), r.get("batch"),
                                                                  r.get("buffers_n")) for r in bad][:20],
                      "control_differs": ctl, "tails": [t["setting"] for t in tails], "tail_failing": tbad,
                      "missing": missing}), flush=True)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dev")
    s.add_argument("--nets", default="S55")
    s.add_argument("--gs", default="1,2,4,8")
    s.add_argument("--stages", default="1,2")
    s.add_argument("--forms", default="par,seq")
    s.add_argument("--extra", default="")
    s.add_argument("--batches", default="4096,1024,256,64,16,1")
    s.add_argument("--buf-n", default="4093,7")
    s.add_argument("--out", required=True)
    s = sub.add_parser("refs")
    s.add_argument("--out", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--nets", default=",".join(CONVLINES))
    s.add_argument("--cfgs", default="new")
    s.add_argument("--control", default="new:i8x")
    s.add_argument("--batches", default="4096,4100,4093,1024,1031,256,263,255,64,67,16,19,7,1")
    s.add_argument("--control-batches", default="4096,255,1")
    s.add_argument("--rows", default="64:32000,67:32000,16:8000,19:8000,7:7000,1:500")
    s.add_argument("--passes", type=int, default=2)
    s.add_argument("--buf-n", default="4096,4100,4093,1031,263,255,67,19,7,1")
    s.add_argument("--refs", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("nets")
    s.add_argument("--net", required=True)
    s.add_argument("--engine", default="new")
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED)
    s.add_argument("--batches", default=",".join(str(b) for b in pc.BATCHES))
    s.add_argument("--paths", default="")
    s.add_argument("--group", default="main")
    s.add_argument("--n-rt", type=int, default=10_000)
    s.add_argument("--secs", type=float, default=2.0)
    s.add_argument("--out", required=True)
    s = sub.add_parser("pipeline")
    s.add_argument("--setting", required=True, choices=pc.SETTINGS)
    s.add_argument("--config", type=int, required=True)
    s.add_argument("--p", type=float, default=None)
    s.add_argument("--mode", required=True, choices=("single", "batched"))
    s.add_argument("--engine", default="new")
    s.add_argument("--tail-b", type=int, default=0)
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--group", default="main")
    s.add_argument("--n", type=int, default=500_000)
    s.add_argument("--kind", type=int, default=pc.KIND_SPEED)
    s.add_argument("--out", required=True)
    s = sub.add_parser("drive")
    s.add_argument("--what", required=True, choices=("nets", "pipe"))
    s.add_argument("--rows", required=True)
    s.add_argument("--reps", type=int, default=pc.REPS)
    s.add_argument("--nets", default=",".join(CONVLINES + ("C14", "C14p")))
    s.add_argument("--settings", default=",".join(pc.SETTINGS))
    s.add_argument("--tail-b", type=int, default=0)
    s.add_argument("--no-untailed", action="store_true")
    s.add_argument("--ns", type=int, default=20_000)
    s.add_argument("--nb", type=int, default=500_000)
    s.add_argument("--ps", default=",".join(str(p) for p in pc.RELAY_P))
    s.add_argument("--extra", default="", help="appended to every process's arguments")
    s = sub.add_parser("gate")
    s.add_argument("--out", required=True)
    s.add_argument("--need-screen", action="store_true")
    s.add_argument("--need-tail", default="", help="settings whose tail checks OUT must hold, comma-separated")
    s = sub.add_parser("report")
    s.add_argument("--dirs", default="")
    s.add_argument("--out", required=True)
    s = sub.add_parser("tail")
    s.add_argument("--setting", required=True, choices=pc.SETTINGS)
    s.add_argument("--tail-b", type=int, required=True)
    s.add_argument("--bounds", default="")
    s.add_argument("--out", required=True)
    s = sub.add_parser("ptx")
    s.add_argument("--net", default="S55")
    s.add_argument("--cfgs", default="new")
    s.add_argument("--batches", default="4096,1")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.rows if a.cmd == "drive" else a.out))
                if a.cmd in ("drive", "nets", "pipeline") else a.out, exist_ok=True)
    if a.cmd == "tail" and not a.bounds:              # around 0, b, 2b, B_Q and B_Q + b
        b, q = a.tail_b, 4096
        a.bounds = ",".join(str(x) for x in sorted({1, b - 1, b, b + 1, 2 * b - 1, 2 * b, 2 * b + 1, q - b, q - 1, q,
                                                    q + 1, q + b - 1, q + b, q + b + 1, 2 * q + 3 * b + 7, 3 * q - 1}))
    {"dev": dev, "refs": refs, "screen": screen, "ptx": ptx, "nets": nets, "pipeline": pipeline, "tail": tail,
     "drive": drive, "report": report, "gate": gate}[a.cmd](a)


if __name__ == "__main__":
    main()
