"""Milestone 1: screen, time, profile and inspect mshot.py's multi-shot kernels on the card of this
process. Phase C's files are read, never written.

  refs     the screen's inputs beside C4b's frozen reads: C3's shots (kind 12, chunks 0-9, at 0.2, 0.3 and 0.55 %, each
           setting, as pc_speed.nets draws them) and today's engine on them (pc_fused.Batched at 4,096, chunk by chunk,
           as the reads ran it) -> OUT/refs/; anchors: today's engine against C4b's frozen reads, and C3's uniform
           0.3 % shots and outputs against the profile's (job CJ45, the same card model)
  screen   every --cfgs engine in a CUDA graph at each --batches, --passes times over every shot set, against the frozen
           RTX reads (C4b: chunks 10-19 at 0.2 and 0.3 %, both settings' reads) and C3's references: every decision and
           every FP32 margin's bit pattern; then launch by launch at --buf-n shots, every intermediate buffer against
           today's engine -> OUT/screen.jsonl
  time     one fresh process per config: the whole S55 per shot (pc_speed.amortized: input resident and host to host;
           CUDA-event replays) at --batch, each kernel's time (torch.profiler over graph replays with PDL, and over
           eager launches without PDL), registers, spills and shared memory per kernel, the SM clock, and a bitwise
           check against the frozen reads in the process -> OUT/time.jsonl
  ncuwork  one eager call per --cfgs at --batch, warm-ups outside, with an NVTX range "m1" around each launch the
           configuration replaces (today's engine: block 1, the pool, the vote), which Nsight Compute profiles
  ptx      each --cfgs kernel's PTX and TTGIR -> OUT/ptx/, with launch options, registers and spills, opcode counts,
           and today's block and vote against the new kernels (diffs of the normalized PTX)
  ncuparse Nsight Compute's raw page (the replaced kernels and today's, labelled by ncuwork's order) and its source page
           (warp-state samples by SASS opcode) -> OUT/ncu_m1.json
  report   OUT's JSON -> OUT/summary.json and OUT/summary.md (no GPU)

Configurations are named by components joined with "+": base (today's engine); b<blocks><form><vec><G> (the residual
blocks listed, e.g. b1 or b01; s = one product per shot, k = stacked rows; v = vectorized gather; G shots per program;
optional n<stages>);
v<G>[w<warps>][i8|i8x] (the vote at G shots per program; i8: int8 psi pinned to the fp16 layout, i8x: unpinned);
pskip (the pool without its padded tiles). See mshot.py."""
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
import pc_common as pc  # noqa: E402
import prof_batched as pb  # noqa: E402

READ = pc.W4 + "/out/check/{}/read"                 # C4b's frozen reads (kind 12, chunks 10-19), RTX PRO 6000
W6 = "<cluster-scratch>/cascade_batchopt/out/CJ45"   # the profile's: uniform 0.3 %, kind 12, chunks 0-9
C3_P = (0.002, 0.003, 0.0055)                       # C3's points: the claim points and the failure screen
OWN = {"uniform": "S55", "depspam_fm": "S55p"}     # the setting's S55
BUFS = ("x", "u0", "u1", "U", "s", "z", "m1", "i1", "m2", "bits", "margin")


# ------------------------------------------------------------------------------------------------ configurations
def cfg_of(name):
    """A configuration's name -> mshot cfg (module docstring); base -> {} (today's engine)."""
    c = {}
    for part in name.split("+"):
        m = re.fullmatch(r"b(\d+)([sk])(v?)(\d+)(?:n(\d))?", part)
        v = re.fullmatch(r"v(\d+)(?:w(\d+))?(i8x?)?", part)
        if part == "base":
            continue
        if m:
            c.update(blocks=tuple(int(i) for i in m.group(1)), form={"s": "safe", "k": "stack"}[m.group(2)],
                     vec=bool(m.group(3)), bg=int(m.group(4)), bs=int(m.group(5) or 0))
        elif v:
            c.update(vg=int(v.group(1)), vw=int(v.group(2) or 4), psi8={None: 0, "i8": 1, "i8x": 2}[v.group(3)])
        elif part == "pskip":
            c["pool"] = "skip"
        else:
            raise SystemExit(f"{name}: unknown component {part}")
    return c


def toolchain():
    """The frozen toolchain this process runs: torch, Triton, CUDA, ptxas, the venv, the Triton cache."""
    import torch
    import triton
    out = {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
           "python": sys.executable, "triton_path": os.path.dirname(triton.__file__),
           "triton_cache": os.environ.get("TRITON_CACHE_DIR", ""), "tree": os.environ.get("PC_TREE", "")}
    try:    # the ptxas Triton compiles this card's arch with (3.5+: get_ptxas; before: _path_to_binary), its --version
        import subprocess
        from triton.backends.nvidia import compiler as tnc
        if hasattr(tnc, "get_ptxas"):
            cc = torch.cuda.get_device_capability()
            path = tnc.get_ptxas(cc[0] * 10 + cc[1]).path
        else:
            path = tnc._path_to_binary("ptxas")[0]
        ver = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
        out["ptxas"] = [path, ver.splitlines()[-1]]
    except Exception as e:                              # recorded, not fatal
        out["ptxas"] = repr(e)[:200]
    return out


def emit(path, row):
    row = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "job": os.environ.get("SLURM_JOB_ID", ""), "pid": os.getpid(),
           **row}
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print(json.dumps({k: v for k, v in row.items() if not isinstance(v, (list, dict))}), flush=True)


def bits_of(t):
    """A tensor's raw bit pattern (as integers of its element size), so that -0.0 / +0.0 and NaN payloads count."""
    import torch
    return t.view({4: torch.int32, 2: torch.int16, 1: torch.uint8}[t.element_size()])


# ------------------------------------------------------------------------------------------------------ shot sets
def c4b_sets(nets):
    """C4b's frozen reads -> [(name, det (N, D) uint8, {net: (bits (N, K) uint8, margin (N,) fp32)}, digest)]."""
    out = []
    for fam in pc.SETTINGS:
        for p in pc.RELAY_P:
            z, meta, idx = pc.load_read(READ.format(fam), p)
            ref = {}
            for t in nets:
                if t in idx:
                    m = z[f"conf{idx[t]}"]
                    if m.dtype != np.float32:
                        raise SystemExit(f"{fam} {p}: conf{idx[t]} is {m.dtype}, not float32")
                    ref[t] = (np.unpackbits(z[f"pred{idx[t]}"], axis=1, count=pc.K), m)
            det = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
            out.append((f"c4b_{fam}_{p}", det, ref, {"kind": meta["kind"], "chunks": sorted(pc.det_digests(meta))}))
    return out


def c3_sets(out_dir, nets):
    """C3's shots and today's outputs on them (the refs step) -> the same form as c4b_sets."""
    out = []
    for fam in pc.SETTINGS:
        for p in C3_P:
            z = np.load(os.path.join(out_dir, "refs", f"c3_{fam}_{p}.npz"))
            ref = {t: (np.unpackbits(z[f"bits_{t}"], axis=1, count=pc.K), z[f"margin_{t}"]) for t in nets
                   if f"bits_{t}" in z.files}
            det = np.unpackbits(z["det"], axis=1, count=432, bitorder="little")
            out.append((f"c3_{fam}_{p}", det, ref, {"kind": int(z["kind"]), "chunks": list(range(int(z["chunks"])))}))
    return out


def reads_engine(e, dg):
    """Today's engine as the reads ran it: chunk by chunk (pc.CHUNK rows), each through Batched.run."""
    import torch
    out = [e.run(dg[i:i + pc.CHUNK]) for i in range(0, len(dg), pc.CHUNK)]
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


def refs(a):
    import torch
    import pc_fused
    from bb import cascade_data as cdata
    dev, cd = pb.setup()
    os.makedirs(os.path.join(a.out, "refs"), exist_ok=True)
    out = {"env": cd, "toolchain": toolchain(), "c3": {}, "anchors": []}
    nets = sorted(set(OWN.values()))
    c4b = c4b_sets(nets)
    for t in nets:
        pr = pb.load(t, dev)
        e = pc_fused.Batched(pr.model, pc.FUSED_READ_B)
        out.setdefault("nets", {})[t] = {"sha256": pr.sig["sha256"], "D": pr.model.D, "K": e.K, "S": e.S, "C": e.C,
                                         "CB": e.CB, "H": e.H, "NT": e.NT, "NIT": e.NIT, "TPI": e.tpi, "SP": e.SP,
                                         "Q": e.Q, "QP": e.QP, "NQT": e.NQT, "NP": e.NP, "blocks": len(e.blk),
                                         "pdl": e.pdl, "stages": e.stages}
        for name, det, ref, _ in c4b:                   # anchor: today's engine against the frozen reads
            if t not in ref:
                continue
            b, m = reads_engine(e, torch.as_tensor(det, device=dev))
            b, m = b.cpu().numpy(), m.cpu().numpy()
            out["anchors"].append({"what": "today's engine against C4b's frozen read", "net": t, "set": name,
                                   "rows": len(det), "differ": int((b != ref[t][0]).any(1).sum()),
                                   "margin_bits_differ": int((m.view(np.int32) != ref[t][1].view(np.int32)).sum())})
            print(json.dumps(out["anchors"][-1]), flush=True)
        del e
        torch.cuda.empty_cache()
    for fam in pc.SETTINGS:
        t = OWN[fam]
        pr = pb.load(t, dev)
        e = pc_fused.Batched(pr.model, pc.FUSED_READ_B)
        for p in C3_P:
            t0 = time.time()
            det, obs = cdata.eval_shots(pc.CODE, p, range(a.chunks), kind=pc.KIND_SPEED, procs=a.procs, family=fam)
            b, m = reads_engine(e, torch.as_tensor(det, device=dev))
            b, m = b.cpu().numpy(), m.cpu().numpy()
            f = os.path.join(a.out, "refs", f"c3_{fam}_{p}.npz")
            np.savez(f, det=np.packbits(det, axis=1, bitorder="little"), obs=np.packbits(obs, axis=1),
                     **{f"bits_{t}": np.packbits(b, axis=1), f"margin_{t}": m}, p=p, kind=pc.KIND_SPEED,
                     chunks=a.chunks, family=fam)
            out["c3"][f"c3_{fam}_{p}"] = {"rows": len(det), "net": t, "sha256": pc.sha256_file(f),
                                          "fail": int((b != obs).any(1).sum()), "secs": round(time.time() - t0, 1)}
            print(json.dumps(out["c3"][f"c3_{fam}_{p}"]), flush=True)
            if fam == "uniform" and p == 0.003 and os.path.exists(os.path.join(W6, "ref_S55.npz")):
                w = np.load(os.path.join(W6, "ref_S55.npz"))
                ws = np.load(os.path.join(W6, "shots.npz"))
                out["anchors"].append({"what": "C3 uniform 0.3 % against the profile's shots and reference (job CJ45)",
                                       "net": t, "same_shots": bool(np.array_equal(ws["det"], det)),
                                       "differ": int((w["bits"] != b).any(1).sum()),
                                       "margin_bits_differ": int((w["margin"].view(np.int32)
                                                                  != m.view(np.int32)).sum())})
                print(json.dumps(out["anchors"][-1]), flush=True)
        del e
        torch.cuda.empty_cache()
    pb.dump(a.out, "refs.json", out)


# --------------------------------------------------------------------------------------------------------- screen
def buffers(m, cfg, xb, n):
    """Today's engine and cfg's, launch by launch on the same n shots: after each launch, every intermediate buffer's
    first n rows, bit for bit -> [(launch, [buffers that differ])]."""
    import torch
    import mshot
    import pc_fused
    e0, e1 = pc_fused.Batched(m, n), mshot.engine(m, n, cfg)
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
    for (name, f0), (_, f1) in zip(l0, l1):
        f0()
        f1()
        torch.cuda.synchronize()
        out.append((name, [b for b in BUFS if not torch.equal(view(e0, b), view(e1, b))]))
    del e0, e1
    return out


def screen(a):
    import torch
    import mshot
    import pc_speed
    dev, cd = pb.setup()
    path = os.path.join(a.out, "screen.jsonl")
    nets = a.nets.split(",")
    sets = c4b_sets(nets) + c3_sets(a.refs, nets)
    sets = [(nm, torch.as_tensor(det, device=dev), {t: (torch.as_tensor(b, device=dev),
                                                        torch.as_tensor(mg, device=dev)) for t, (b, mg) in ref.items()},
             dg) for nm, det, ref, dg in sets]
    base = {"step": "screen", "gpu": cd["gpu"], "driver": cd["driver"], "toolchain": toolchain()}
    small = {int(b): int(r) for b, r in (x.split(":") for x in a.small.split(","))} if a.small else {}
    for t in nets:
        pr = pb.load(t, dev)
        m = pr.model
        for name in a.cfgs.split(","):
            cfg = cfg_of(name)
            for B in [int(b) for b in a.batches.split(",")] + sorted(small):
                t0 = time.time()
                try:
                    e = mshot.engine(m, B, cfg)
                    g = pc_speed.Graph(lambda x, e=e, B=B: e(x, B), B, m.D, dev)
                except Exception as ex:                 # recorded; the screen goes on
                    emit(path, {**base, "net": t, "cfg": name, "batch": B, "error": repr(ex)[:400]})
                    continue
                res = []
                for ps in range(a.passes if B not in small else 1):
                    for nm, det, ref, _ in sets:
                        if t not in ref:
                            continue
                        r = small.get(B, len(det))
                        bits, mg = g.run(det[:r])
                        rb, rm = ref[t][0][:r], ref[t][1][:r]
                        res.append({"set": nm, "pass": ps, "rows": r,
                                    "differ": int((bits != rb).any(1).sum()),
                                    "margin_bits_differ": int((bits_of(mg) != bits_of(rm)).sum())})
                ok = all(x["differ"] == 0 and x["margin_bits_differ"] == 0 for x in res)
                emit(path, {**base, "net": t, "cfg": name, "mshot": cfg, "batch": B, "ok": ok,
                            "rows": sum(x["rows"] for x in res), "sets": res, "secs": round(time.time() - t0, 1)})
                del g, e
                torch.cuda.empty_cache()
            if a.buf_n:                                 # launch by launch against today's engine
                xs = sets[0][1]
                for n in [int(x) for x in a.buf_n.split(",")]:
                    bad = buffers(m, cfg, xs[:n].contiguous(), n)
                    emit(path, {**base, "net": t, "cfg": name, "mshot": cfg, "buffers_n": n,
                                "ok": not any(b for _, b in bad), "buffers": bad})
                    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------- time
def time_(a):
    import torch
    import mshot
    import pc_speed
    dev, cd = pb.setup()
    pr = pb.load(a.net, dev)
    m = pr.model
    fam = pc.NETS[a.net][1]
    z, meta, idx = pc.load_read(READ.format(fam), 0.003)
    dg = torch.as_tensor(np.unpackbits(z["det"], axis=1, count=432, bitorder="little"), device=dev)
    rb = torch.as_tensor(np.unpackbits(z[f"pred{idx[a.net]}"], axis=1, count=pc.K), device=dev)
    rm = torch.as_tensor(z[f"conf{idx[a.net]}"], device=dev)
    B, cfg = a.batch, cfg_of(a.cfg)
    t0 = time.time()
    e = mshot.engine(m, B, cfg)
    gr = pc_speed.Graph(lambda x: e(x, B), B, m.D, dev)
    row = {"step": "time", "net": a.net, "cfg": a.cfg, "mshot": cfg, "rep": a.rep, "batch": B, "env": cd,
           "toolchain": toolchain(), "build_s": round(time.time() - t0, 1), "pdl": bool(e.pdl)}
    n = min(len(dg), a.check)
    bits, mg = gr.run(dg[:n])                           # the bitwise screen in every timing process
    row.update(bitwise_rows=n, bitwise=bool(torch.equal(bits, rb[:n]) and torch.equal(bits_of(mg), bits_of(rm[:n]))))
    w = pb.Watch()
    row["amortized_us"] = pc_speed.amortized(gr, dg, secs=a.secs)    # C3's metric: resident / host to host
    row["replay_us"] = pb.replay_us(gr)
    row["clock"] = w.stop()
    row["gpu_us_shot"] = row["replay_us"]["mean"] / B
    names = [nm for nm, _ in e.launches(gr.x, B)]
    it = pb.iters_for(row["replay_us"]["mean"])
    tr = os.path.join(a.out, f"trace_{os.getpid()}")
    row["kernels_graph"] = pb.ksum_fused(pb.kprof(gr.replay, it, tr), it, B, names)
    e.pdl = False                                       # eager launches without PDL: each kernel's own time
    e(gr.x, B)
    torch.cuda.synchronize()
    row["kernels_eager_nopdl"] = pb.ksum_fused(pb.kprof(lambda: e(gr.x, B), it, tr), it, B, names)
    e.pdl = row["pdl"]
    os.remove(tr + ".json.gz")
    row["kernels"] = [pb.ptx_summary(nm, fn()) for nm, fn in e.launches(gr.x, B)]
    torch.cuda.synchronize()
    emit(os.path.join(a.out, "time.jsonl"), row)


# -------------------------------------------------------------------------------------------------- ncu and ptx
def ncuwork(a):
    import torch
    import mshot
    dev, cd = pb.setup()
    pr = pb.load(a.net, dev)
    z, _, _ = pc.load_read(READ.format(pc.NETS[a.net][1]), 0.003)
    det = torch.as_tensor(np.unpackbits(z["det"][:a.batch], axis=1, count=432, bitorder="little"), device=dev)
    order = []
    for name in a.cfgs.split(","):
        c = cfg_of(name)
        want = [f"block{i}" for i in c.get("blocks", ())] + (["pool"] if c.get("pool") else []) \
            + (["vote"] if c.get("vg") or c.get("psi8") else []) if c else ["block1", "pool", "vote"]
        e = mshot.engine(pr.model, a.batch, c)
        for _ in range(2):
            e(det, a.batch)
        torch.cuda.synchronize()
        for ln, fn in e.launches(det, a.batch):         # the whole call, so that each kernel reads real inputs
            if ln in want:
                torch.cuda.nvtx.range_push("m1")
            fn()
            torch.cuda.synchronize()
            if ln in want:
                torch.cuda.nvtx.range_pop()
                order.append({"cfg": name, "launch": ln})
        del e
        torch.cuda.empty_cache()
    pb.dump(a.out, f"ncu_order_{a.tag}.json", {"cfgs": order, "batch": a.batch, "net": a.net, "gpu": cd["gpu"]})


PTX_SKIP = re.compile(r"^(//|\.|\$|\{|\}|\)|$)")
PTX_OP = re.compile(r"^(?:@!?%p\w+\s+)?([a-z][\w.]*)")
FP_OPS = re.compile(r"^(mma|wgmma|fma|add\.(rn\.)?f32|sub\.(rn\.)?f32|mul\.(rn\.)?f32|neg\.f32|div|rcp|rsqrt|sqrt|ex2|"
                    r"lg2|cvt\.[a-z.]*f(16|32)|max\.f32|min\.f32|abs\.f32|shfl|ldmatrix|stmatrix)")


def opcodes(ptx):
    """PTX -> {opcode with its modifiers: count} over the instructions (labels, directives and braces skipped)."""
    out = {}
    for ln in ptx.splitlines():
        s = ln.strip()
        if PTX_SKIP.match(s) or s.endswith(":"):
            continue
        mo = PTX_OP.match(s)
        if mo:
            out[mo.group(1)] = out.get(mo.group(1), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def normalized(ptx):
    """PTX lines with register numbers and labels replaced, for a diff that shows instructions, not renaming."""
    out = []
    for ln in ptx.splitlines():
        s = ln.strip()
        if not s or s.startswith(("//", ".loc", ".file")):
            continue
        s = re.sub(r"%(r|rd|f|fd|h|hh|p|rs|b)\d+", r"%\1", s)
        out.append(re.sub(r"\$L__\w+", "$L", s))
    return out


def ptx(a):
    import torch
    import mshot
    dev, cd = pb.setup()
    pr = pb.load(a.net, dev)
    d = os.path.join(a.out, "ptx")
    os.makedirs(d, exist_ok=True)
    z, _, _ = pc.load_read(READ.format(pc.NETS[a.net][1]), 0.003)
    det = torch.as_tensor(np.unpackbits(z["det"][:a.batch], axis=1, count=432, bitorder="little"), device=dev)
    out = {"env": cd, "toolchain": toolchain(), "net": a.net, "batch": a.batch, "kernels": {}}
    for name in a.cfgs.split(","):
        try:
            e = mshot.engine(pr.model, a.batch, cfg_of(name))
            e(det, a.batch)
        except Exception as ex:                         # recorded; the other configurations go on
            out["kernels"][f"{name}.error"] = {"error": repr(ex)[:2000]}
            print(json.dumps({"cfg": name, "error": repr(ex)[:2000]}), flush=True)
            continue
        for ln, fn in e.launches(det, a.batch):
            k = fn()
            asm = k.asm
            key = f"{name}.{ln}"
            for ext in ("ptx", "ttgir"):
                with gzip.open(os.path.join(d, f"{key}.{ext}.gz"), "wt") as fh:
                    fh.write(asm[ext])
            md = k.metadata
            out["kernels"][key] = {"name": md.name, "num_warps": md.num_warps, "num_stages": md.num_stages,
                                   "shared": md.shared, "regs": getattr(k, "n_regs", None),
                                   "spills": getattr(k, "n_spills", None), "hash": md.hash,
                                   "opcodes": opcodes(asm["ptx"])}
        torch.cuda.synchronize()
        del e
        torch.cuda.empty_cache()
    diffs = {}
    for new in out["kernels"]:                          # every replaced kernel against today's same launch
        cfg, ln = new.split(".", 1)
        old = f"base.{ln}"
        if cfg == "base" or ln == "error" or old not in out["kernels"] \
                or out["kernels"][new]["hash"] == out["kernels"][old]["hash"]:
            continue
        po, pn = (gzip.open(os.path.join(d, f"{k}.ptx.gz"), "rt").read() for k in (old, new))
        df = list(difflib.unified_diff(normalized(po), normalized(pn), old, new, lineterm="", n=1))
        with gzip.open(os.path.join(d, f"diff.{new}.txt.gz"), "wt") as fh:
            fh.write("\n".join(df) + "\n")
        oo, on = out["kernels"][old]["opcodes"], out["kernels"][new]["opcodes"]
        fp = sorted(x for x in set(oo) | set(on) if FP_OPS.match(x))
        diffs[new] = {"against": old, "diff_lines": sum(1 for x in df if x[:1] in "+-" and x[:3] not in ("+++", "---")),
                      "fp_ops": {x: [oo.get(x, 0), on.get(x, 0)] for x in fp},
                      "fp_ops_only_old": [x for x in fp if x not in on], "fp_ops_only_new": [x for x in fp if x not in oo]}
    out["diffs"] = diffs
    pb.dump(a.out, "ptx.json", out)
    for k, v in diffs.items():
        print(json.dumps({"kernel": k, "diff_lines": v["diff_lines"], "only_old": v["fp_ops_only_old"],
                          "only_new": v["fp_ops_only_new"], "regs": out["kernels"][k]["regs"],
                          "spills": out["kernels"][k]["spills"], "shared": out["kernels"][k]["shared"]}))


NCU_M1 = {"duration_ns": r"gpu__time_duration\.sum", "l1_pct": r"l1tex__throughput\.avg\.pct_of_peak_sustained_active",
          "lsu_wf": r"l1tex__data_pipe_lsu_wavefronts\.sum",
          "lsu_wf_pct": r"l1tex__data_pipe_lsu_wavefronts\.sum\.pct_of_peak_sustained_elapsed",
          "shared_wf": r"l1tex__data_pipe_lsu_wavefronts_mem_shared\.sum",
          "shared_wf_pct": r"l1tex__data_pipe_lsu_wavefronts_mem_shared\.sum\.pct_of_peak_sustained_elapsed",
          "shared_wf_ld": r"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld\.sum",
          "shared_wf_st": r"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st\.sum",
          "bank_conflicts": r"l1tex__data_bank_conflicts_pipe_lsu_mem_shared\.sum",
          "bank_conflicts_ld": r"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld\.sum",
          "bank_conflicts_st": r"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st\.sum",
          "bank_conflicts_ldgsts": r"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts\.sum",
          "ldsm_inst": r"smsp__inst_executed_op_ldsm\.sum", "shared_inst": r"smsp__sass_inst_executed_op_shared\.sum",
          "ldgsts_inst": r"smsp__inst_executed_op_ldgsts\.sum",
          "global_ld_inst": r"smsp__sass_inst_executed_op_global_ld\.sum",
          "global_ld_req": r"l1tex__t_requests_pipe_lsu_mem_global_op_ld\.sum",
          "global_ld_sectors": r"l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum",
          "l1_hit_pct": r"l1tex__t_sector_hit_rate\.pct", "l2_pct": r"lts__throughput\.avg\.pct_of_peak_sustained_elapsed",
          "l2_bytes": r"lts__t_bytes\.sum", "dram_read": r"dram__bytes(_op)?_read\.sum", "dram_write": r"dram__bytes(_op)?_write\.sum",
          "occ_achieved": r"sm__warps_active\.avg\.pct_of_peak_sustained_active",
          "occ_theoretical": r"sm__maximum_warps_per_active_cycle_pct", "regs": r"launch__registers_per_thread",
          "smem_block": r"launch__shared_mem_per_block", "occ_limit_regs": r"launch__occupancy_limit_registers",
          "occ_limit_smem": r"launch__occupancy_limit_shared_mem", "grid": r"launch__grid_size",
          "block": r"launch__block_size", "inst": r"smsp__inst_executed\.sum",
          "issue_pct": r"sm__instruction_throughput\.avg\.pct_of_peak_sustained_active",
          "ffma": r"sm__sass_thread_inst_executed_op_ffma_pred_on\.sum",
          "fadd": r"smsp__sass_thread_inst_executed_op_fadd_pred_on\.sum",
          "fmul": r"smsp__sass_thread_inst_executed_op_fmul_pred_on\.sum",
          "hmma_ops": r"sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32_sparsity_off\.sum"}
STALL = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")
UNIT = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12, "ns": 1, "nsecond": 1, "us": 1e3,
        "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9}     # ncu scales units: to bytes, ns


def ncu_labels(out_dir, tag="m1"):
    """The profiled kernels' (configuration, launch) in profiling order (ncuwork's ranges)."""
    o = json.load(open(os.path.join(out_dir, f"ncu_order_{tag}.json")))
    return [(c["cfg"], c["launch"]) for c in o["cfgs"]], o["batch"]


def src_stalls(path):
    """ncu's source page (SASS, CSV): per kernel, the warp-state samples summed by opcode (its first token), top 12."""
    import csv
    out, cur, hdr = [], None, None
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.reader(fh):
            if "Source" in r and any("Sampling" in x for x in r):
                hdr = {h: i for i, h in enumerate(r)}
                cur = {}
                out.append(cur)
                continue
            if cur is None or len(r) < len(hdr):
                continue
            ins = r[hdr["Source"]].strip()
            if not ins:
                continue
            op = re.sub(r"^@!?U?P\w+\s+", "", ins).split()[0] if ins.split() else "?"
            for h, i in hdr.items():
                if h.startswith("Warp Stall Sampling (All"):
                    try:
                        cur[op] = cur.get(op, 0.0) + float(r[i].replace(",", "") or 0)
                    except ValueError:
                        pass
    return [dict(sorted(k.items(), key=lambda kv: -kv[1])[:12]) for k in out]


def ncuparse(a):
    out = ncu_dir(a.out)
    pb.dump(a.out, "ncu_m1.json", out)
    for k in out:
        print(json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in k.items()
                          if x in ("cfg", "launch", "duration_ns", "lsu_wf_pct", "shared_wf_pct", "l1_pct", "regs",
                                   "occ_achieved", "bank_conflicts")}))


def ncu_dir(dr):
    """One job's Nsight Compute export (raw and source pages, ncuwork's order) -> [one dict per profiled kernel]."""
    lab, B = ncu_labels(dr)
    rows = pb.ncu_rows(os.path.join(dr, "ncu_m1_raw.csv"))
    if len(rows) != len(lab):
        print(json.dumps({"warning": f"{len(rows)} profiled kernels, {len(lab)} expected"}))
    out = []
    for (cfg, ln), d in zip(lab, rows):
        k = {"cfg": cfg, "launch": ln, "name": str(d.get("Kernel Name", ("?", ""))[0])[:80], "batch": B}
        for key, pat in NCU_M1.items():
            hit = [v for h, v in d.items() if re.fullmatch(pat, h)]
            if hit and isinstance(hit[0][0], float):
                k[key] = hit[0][0] * UNIT.get(hit[0][1].strip().lower().split("/")[0], 1)
        k["stalls"] = dict(sorted(((STALL.fullmatch(h).group(1), v[0]) for h, v in d.items()
                                   if STALL.fullmatch(h) and isinstance(v[0], float)), key=lambda x: -x[1])[:8])
        out.append(k)
    f = os.path.join(dr, "ncu_m1_src.csv.gz")
    if os.path.exists(f):
        try:
            st = src_stalls(f)
            for k, s in zip(out, st if len(st) == len(out) else []):
                k["stall_samples_by_opcode"] = s
        except Exception as ex:                         # the source page is extra; the counters stand alone
            print(json.dumps({"src_stalls": repr(ex)[:300]}))
    return out


# --------------------------------------------------------------------------------------------------------- report
def occupancy(regs, shared, warps, sm=(65536, 102400, 1024, 48, 24)):
    """Theoretical blocks per SM and warp occupancy (%) on the RTX PRO 6000 (sm_120: 64K registers, 100 KB of shared
    memory with 1 KB reserved per block, 48 warps, 24 blocks): registers go per warp in units of 256."""
    nreg, nsh, res, mw, mb = sm
    by_regs = nreg // (warps * (-(-max(regs, 1) * 32 // 256) * 256)) if regs else mb
    by_smem = nsh // (shared + res)
    b = min(by_regs, by_smem, mw // warps, mb)
    return b, 100.0 * b * warps / mw


def jl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def report(a):
    """The screen's verdicts, the slowest timing process per configuration and kernel, registers and Nsight Compute's
    counters -> summary.json, summary.md."""
    out, md = {}, []
    dirs = a.dirs.split(",") if a.dirs else [a.out]
    out["dirs"] = dirs
    sc = [r for d in dirs for r in jl(os.path.join(d, "screen.jsonl"))]
    if sc:
        by = {}
        for r in sc:
            k = (r["net"], r["cfg"])
            d = by.setdefault(k, {"graph_ok": True, "rows": 0, "batches": set(), "buffers_ok": None, "errors": []})
            if "error" in r:
                d["errors"].append(r["error"])
                d["graph_ok"] = False
            elif "buffers_n" in r:
                d["buffers_ok"] = r["ok"] if d["buffers_ok"] is None else d["buffers_ok"] and r["ok"]
            else:
                d["graph_ok"] &= r["ok"]
                d["rows"] += r["rows"]
                d["batches"].add(r["batch"])
        out["screen"] = {f"{n} {c}": {**d, "batches": sorted(d["batches"])} for (n, c), d in by.items()}
        md += ["## Screen", "", "| net | configuration | batches | shot evaluations | decisions and margins | "
               "buffers launch by launch |", "|---|---|---|---:|---|---|"]
        for (n, c), d in by.items():
            md.append(f"| {n} | {c} | {', '.join(str(b) for b in sorted(d['batches']))} | {d['rows']:,} | "
                      f"{'bitwise' if d['graph_ok'] else 'DIFFER' + (' (error)' if d['errors'] else '')} | "
                      f"{'-' if d['buffers_ok'] is None else 'bitwise' if d['buffers_ok'] else 'DIFFER'} |")
        md.append("")
    tm = [r for d in dirs for r in jl(os.path.join(d, "time.jsonl"))]
    if tm:
        by = {}
        for r in tm:                                    # per job: each job times today's engine too
            by.setdefault((r["net"], r["cfg"], r["batch"], r.get("job", "")), []).append(r)
        rows = {}
        for (n, c, B, j), rs in by.items():
            ker = {}
            for r in rs:
                for k in r["kernels_eager_nopdl"]["rows"]:
                    ker.setdefault(k["name"], []).append(k["us_shot"])
            regs = {k["name"]: [k.get("regs"), k.get("spills"), k.get("shared_bytes")] for k in rs[0]["kernels"]}
            rows[f"{n} {c} {B} {j}"] = {
                "reps": len(rs), "dev_us": max(r["amortized_us"]["dev"] for r in rs),
                "h2h_us": max(r["amortized_us"]["h2h"] for r in rs), "dev_us_all": [r["amortized_us"]["dev"] for r in rs],
                "replay_us_shot": max(r["gpu_us_shot"] for r in rs), "bitwise": all(r["bitwise"] for r in rs),
                "clock_min_ratio": min((r["clock"].get("sm_min_ratio") or 0) for r in rs),
                "kernels_us_shot": {k: max(v) for k, v in ker.items()}, "kernels_regs_spills_smem": regs}
        out["time"] = rows
        b0 = {(k.split()[0], k.split()[3] if len(k.split()) > 3 else ""): v for k, v in rows.items()
              if k.split()[1] == "base"}
        md += ["## Time (slowest process; µs per shot)", "", "| job | net | configuration | processes | S55 resident | "
               "base / this | host to host | bitwise | min SM clock ratio | kernels (eager, no PDL) |",
               "|---|---|---|---:|---:|---:|---:|---|---:|---|"]
        for k, v in rows.items():
            n, c, B, j = (k.split() + [""])[:4]
            ref = b0.get((n, j), {}).get("dev_us")
            ks = ", ".join(f"{x} {y:.3f}" for x, y in v["kernels_us_shot"].items())
            md.append(f"| {j} | {n} | {c} | {v['reps']} | {v['dev_us']:.3f} | "
                      f"{ref / v['dev_us'] if ref else float('nan'):.3f} | {v['h2h_us']:.3f} | {v['bitwise']} | "
                      f"{v['clock_min_ratio']:.3f} | {ks} |")
        md.append("")
    nc = [k for d in dirs if os.path.exists(os.path.join(d, "ncu_m1_raw.csv")) for k in ncu_dir(d)]
    if nc:
        out["ncu"] = nc
        md += ["## Nsight Compute (base clock; per shot unless marked)", "",
               "| configuration | launch | µs | L1 % | LSU wavefronts: all / shared / global | bank conflicts | ldmatrix | "
               "global loads | L2 MB | DRAM MB | L1 hit % | L2 % | occupancy % | registers | shared KB | top stalls | "
               "top SASS samples | FFMA / FADD / FMUL / HMMA ops |", "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"
               "---:|---:|---:|---:|---|---|---|"]
        for k in nc:
            B, g = k["batch"], (lambda x: k.get(x, float("nan")))
            ss = k.get("stall_samples_by_opcode", {})
            tot = sum(ss.values()) or 1.0
            md.append(f"| {k['cfg']} | {k['launch']} | {g('duration_ns') / 1e3:.0f} | {g('l1_pct'):.0f} | "
                      f"{g('lsu_wf') / B:,.0f} / {g('shared_wf') / B:,.0f} / {(g('lsu_wf') - g('shared_wf')) / B:,.0f} | "
                      f"{g('bank_conflicts') / B:,.0f} | {g('ldsm_inst') / B:,.0f} | {g('global_ld_inst') / B:,.0f} | "
                      f"{g('l2_bytes') / B / 1e6:.2f} | {(g('dram_read') + g('dram_write')) / B / 1e6:.3f} | "
                      f"{g('l1_hit_pct'):.0f} | {g('l2_pct'):.0f} | {g('occ_achieved'):.0f} / {g('occ_theoretical'):.0f} | "
                      f"{g('regs'):.0f} | {g('smem_block') / 1e3:.1f} | "
                      + ", ".join(f"{x} {y:.1f}" for x, y in list(k["stalls"].items())[:3]) + " | "
                      + ", ".join(f"{x} {100 * y / tot:.0f}%" for x, y in list(ss.items())[:3]) + " | "
                      f"{g('ffma') / B:,.0f} / {g('fadd') / B:,.0f} / {g('fmul') / B:,.0f} / {g('hmma_ops') / B:,.0f} |")
        md.append("")
    px = [os.path.join(d, "ptx.json") for d in dirs if os.path.exists(os.path.join(d, "ptx.json"))]
    if px:
        p = json.load(open(px[0]))
        out["ptx"] = {"kernels": {k: {x: v.get(x) for x in ("name", "num_warps", "num_stages", "shared", "regs", "spills",
                                                             "error")} for k, v in p["kernels"].items()},
                      "diffs": p["diffs"]}
        md += ["## Kernels (registers, spills, shared memory; PTX against today's)", "",
               "| kernel | warps | stages | registers | spills | shared bytes | blocks per SM | occupancy % | diff lines | "
               "FP opcodes only in old / new |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for k, v in p["kernels"].items():
            d = p["diffs"].get(k, {})
            if "error" in v:
                md.append(f"| {k} | error: {v['error'][:150]} |")
                continue
            b, oc = occupancy(v["regs"], v["shared"], v["num_warps"])
            md.append(f"| {k} | {v['num_warps']} | {v['num_stages']} | {v['regs']} | {v['spills']} | {v['shared']} | "
                      f"{b} | {oc:.0f} | {d.get('diff_lines', '')} | {d.get('fp_ops_only_old', '')} / "
                      f"{d.get('fp_ops_only_new', '')} |")
        md.append("")
    pb.dump(a.out, "summary.json", out)
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:80]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")                 # required set below: the login nodes' Python is 3.6
    sub.required = True
    s = sub.add_parser("refs")
    s.add_argument("--chunks", type=int, default=10)
    s.add_argument("--procs", type=int, default=4)
    s = sub.add_parser("screen")
    s.add_argument("--nets", default="S55,S55p")
    s.add_argument("--cfgs", required=True)
    s.add_argument("--batches", default="4096,4093")
    s.add_argument("--small", default="255:10200,7:7000,1:500", help="batch:rows, one pass each")
    s.add_argument("--passes", type=int, default=2)
    s.add_argument("--buf-n", default="4096,4093,7,1")
    s.add_argument("--refs", required=True, help="the refs step's output directory")
    s = sub.add_parser("time")
    s.add_argument("--net", default="S55")
    s.add_argument("--cfg", required=True)
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--batch", type=int, default=pc.VERDICT_B)
    s.add_argument("--secs", type=float, default=2.0)
    s.add_argument("--check", type=int, default=32768)
    for n in ("ncuwork", "ptx"):
        s = sub.add_parser(n)
        s.add_argument("--net", default="S55")
        s.add_argument("--cfgs", required=True)
        s.add_argument("--batch", type=int, default=pc.VERDICT_B)
        s.add_argument("--tag", default="m1")
    s = sub.add_parser("report")
    s.add_argument("--dirs", default="", help="comma-separated job directories to merge (default: --out)")
    sub.add_parser("ncuparse")
    for p_ in sub.choices.values():
        p_.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"refs": refs, "screen": screen, "time": time_, "ncuwork": ncuwork, "ptx": ptx, "ncuparse": ncuparse,
     "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
