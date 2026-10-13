#!/usr/bin/env python3
"""Load every network of the Cascade comparison from this release and decode recorded shots again (one GPU).

    python cascade/repro/check_networks.py --archive DIR --d1 DIR --out DIR

--archive is phase C's run directory's out/ (check/, dev_late/ ...: the per-shot reads listed in
phase_c/out/npz_sha256.txt), --d1 is D1's out/ (read/, compiled/: phase_d/out/npz_sha256.txt). Needs phase C's GPU
model (an RTX PRO 6000; pc_common.gpu_env refuses any other card) and QND_RELAY set to relay-bp's repository at d185194
(the circuits). Every per-shot file is checked against its listed sha256 before it is read.

  1. weights/MANIFEST.sha256 holds every checkpoint's sha256; each checkpoint's source_sha256 is the sha256 that the
     locks and the reads record for that network; its parameter count.
  2. Phase C's reads, chunk by chunk as the reads ran (pc_eval.py: Cascade eagerly in BF16 or, for C6 and C6', under
     torch.compile where the read did; ConvLines on the fused engine): the shots rebuilt from their seeds (circuit
     sha256 and each chunk's digest against the read's record), then every released network's decisions, confidences
     and raw outputs (Cascade's logits z, ConvLines' class) against the read's, bitwise.
  3. D1's read (kind 14, its first chunk) for C14' and S55'-deep in the same way, and D1's compiled reads: C14' on phase
     C's compiled paths (pc_speed.build, torch.compile and a CUDA graph) at batch 4,096 on the read's first 8,192 rows
     and at batch 1 on its first 2,000 rows.
  4. The multi-shot engine (mfused, as m2.py builds it) on the ConvLines networks at several batch sizes against the
     reads, bitwise (M2's screen found no differing decision or margin bit).
  5. With --curves: the curve sets' networks (r11 and r12, the ConvLines networks of weights/bb) on the eager fp32
     path that wrote their predictions, on the first w2test chunk at p = 0.1 % and 0.3 %: decisions, failures and
     r12's routing at q = 0.0025 against the recorded predictions (the curve record claims no bitwise margins).
A ConvLines check in 2 to 4 passes only if it is bitwise, as the records require; a Cascade check passes if every
decision agrees, and it reports whether the confidences are bitwise too (the records claim that only for the
compiled@1 path against its own path object). Writes OUT/check_networks.json; exit 1 if a check fails."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAS = os.path.dirname(HERE)
CC = os.path.join(CAS, "cascade_compare")
PC, PD, BO = (os.path.join(CC, d) for d in ("phase_c", "phase_d", "batchopt"))
sys.path[:0] = [os.path.join(CAS, "ising_compare"), os.path.join(CAS, "qldpc_neural"), PC, PD, BO]

import pc_common as pc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(CAS))                   # cascade/bb72 -> the repository root
W = os.path.join(REPO, "weights", "cascade_bb72")
FILES = {"C14": "C14.pt", "C14p": "C14p.pt", "C6": "C6.pt", "C6p": "C6p.pt", "S55": "S55.pt", "S55p": "S55p.pt",
         "S55p_deep": "S55p_deep.pt"}
PARAMS = {"C14": 19_687_937, "C14p": 19_687_937, "S55": 483_357, "S55p": 483_357, "S55p_deep": 841_053}  # the paper
READS = (("check/uniform/read", 0.003, 10), ("check/depspam_fm/read", 0.003, 10),     # (read, p, chunk)
         ("check/depspam_fm/eager", 0.003, 10), ("dev_late/depspam_fm", 0.003, 0))
BATCHES = (4096, 4100, 1031, 255, 16, 1)          # the multi-shot engine's screen covered 1 to 4,100
CURVES = (("w2test", 0, 0), ("w2test", 4, 0))     # (split, index into curves.PS, chunk): p = 0.1 % and 0.3 %
RES = []


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def result(what, ok, **detail):
    RES.append({"check": what, "ok": bool(ok), **detail})
    print(f"{'ok  ' if ok else 'FAIL'} {what}: " + ", ".join(f"{k} {v}" for k, v in detail.items()), flush=True)
    return ok


def listed(path):
    """{relative path: sha256} of an npz_sha256.txt."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                h, f = line.split(None, 1)
                out[f.strip()] = h
    return out


def verified(root, rel, sums, prefix=""):
    path = os.path.join(root, rel)
    h = sha256_file(path)
    result(f"sha256 of {prefix}{rel}", h == sums.get(prefix + rel), sha256=h[:16])
    return path


def weights():
    man = {os.path.basename(k): v for k, v in listed(os.path.join(REPO, "weights", "MANIFEST.sha256")).items()
           if k.startswith("cascade_bb72/")}
    src, locks = {}, {}
    for fam in pc.SETTINGS:
        with open(os.path.join(PC, "out", f"locked_{fam}.json")) as fh:
            locks.update(json.load(fh)["nets"])
    import torch
    for t, f in FILES.items():
        path = os.path.join(W, f)
        h = sha256_file(path)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        src[t] = ck["source_sha256"]
        result(f"{f}: sha256 as in weights/MANIFEST.sha256", h == man.get(f), sha256=h[:16])
        if t in locks:
            result(f"{f}: source_sha256 is the lock's sha256 for {t}", src[t] == locks[t], source=src[t][:16])
    return src


def unpack(x, n=pc.K):
    return np.unpackbits(x, axis=1, count=n)


def compare(tag, label, out, z, i, sel, kind, bitwise_required):
    """One network's outputs on the read's rows sel against the read's arrays of model i."""
    bits = out["bits"].cpu().numpy()
    conf = out["conf"].float().cpu().numpy()
    dec = int((bits != unpack(z[f"pred{i}"][sel])).any(1).sum())
    cb = int((conf.view(np.uint32) != z[f"conf{i}"][sel].view(np.uint32)).sum())
    if kind == "cascade":
        raw = out["z"].cpu().numpy().astype(np.float16)
        rd = int((raw.view(np.uint16) != z[f"z{i}"][sel].view(np.uint16)).any(1).sum())
    else:
        rd = int((out["cls"].cpu().numpy().astype(np.uint16) != z[f"cls{i}"][sel]).sum())
    fail = (bits != unpack(z["obs"][sel])).any(1)
    fd = int((fail != z[f"fail{i}"][sel]).sum())
    ok = dec == 0 and fd == 0 and (not bitwise_required or (cb == 0 and rd == 0))
    return result(f"{tag} {label}", ok, rows=len(sel), decisions_differ=dec, conf_bits_differ=cb, raw_differ=rd,
                  failures=int(fail.sum()), failures_recorded=int(z[f"fail{i}"][sel].sum()))


def shots(meta, chunk, p):
    """The chunk's shots rebuilt from its seed, checked against the read's circuit sha256 and chunk digest."""
    from bb import cascade_data as cdata
    fam, kind = meta["family"], meta["kind"]
    circ = hashlib.sha256(str(cdata.circuit_at(meta["code"], p, fam)).encode()).hexdigest()
    result(f"circuit {fam} p = {p}: sha256 as recorded", circ == meta["circuit_sha256"], sha256=circ[:16])
    det, obs = cdata.eval_shots(meta["code"], p, [chunk], kind=kind, family=fam)
    cm = next(c for c in meta["chunk_meta"] if c["chunk"] == chunk)
    dg = hashlib.sha256(np.packbits(det, axis=1).tobytes()).hexdigest()[:16]
    result(f"shots {fam} kind {kind} p = {p} chunk {chunk}: seed and digest as recorded",
           dg == cm["det_sha256"] and cdata.eval_seed(kind, p, chunk, fam) == cm["seed"], rows=len(det), digest=dg)
    return det, obs


class Nets:
    """Predictors as pc_eval.py makes them (compiled or not, as the read was), one per (network, compiled)."""

    def __init__(self, dev):
        import pc_eval
        self.pc_eval, self.dev, self.cache = pc_eval, dev, {}

    def get(self, tag, compiled):
        key = (tag, bool(compiled))
        if key not in self.cache:
            from bb import cascade_eval as ce
            path = os.path.realpath(os.path.join(W, FILES[tag]))
            self.pc_eval.COMPILE.clear()
            if compiled:
                self.pc_eval.COMPILE.add(path)
            self.cache[key] = ce.Predictor(path, self.dev, "ema", "auto")
        return self.cache[key]


def phase_c_reads(arch, src, nets, sums):
    import torch
    by_src = {v: k for k, v in src.items()}
    seen = set()
    for rd, p, chunk in READS:
        npz = verified(arch, f"{rd}/{pc.p_tag(p)}.npz", sums)
        with open(os.path.join(arch, rd, f"{pc.p_tag(p)}.json")) as fh:
            meta = json.load(fh)
        z = np.load(npz)
        sel = np.flatnonzero(z["chunk"] == chunk)
        det, obs = shots(meta, chunk, p)
        result(f"{rd}: observables of chunk {chunk} as recorded", np.array_equal(unpack(z["obs"][sel]), obs))
        dg = torch.as_tensor(det, device=nets.dev)
        for i, sig in enumerate(meta["models"]):
            tag = by_src.get(sig["sha256"])
            if tag is None:
                continue                                    # r11, r12: released on main (weights/bb)
            pr = nets.get(tag, sig.get("compiled"))
            t0 = time.time()
            out = pr(dg, None)
            label = f"on {rd} p = {p} chunk {chunk} ({sig['prec']}{', compiled' if sig.get('compiled') else ''})"
            compare(tag, label, out, z, i, sel, pr.kind, pr.kind == "convlines")
            seen.add(tag)
            print(f"     {tag}: {time.time() - t0:.1f} s", flush=True)
    result("every released network decoded on a recorded read", seen == set(FILES), missing=sorted(set(FILES) - seen))


def d1_reads(d1dir, src, nets):
    import torch
    import pc_speed
    sums = listed(os.path.join(PD, "out", "npz_sha256.txt"))
    by_src = {v: k for k, v in src.items()}
    p = 0.002
    npz = verified(d1dir, f"read/{pc.p_tag(p)}.npz", sums, prefix="out/")
    with open(os.path.join(d1dir, "read", f"{pc.p_tag(p)}.json")) as fh:
        meta = json.load(fh)
    z = np.load(npz)
    chunk = meta["chunk_meta"][0]["chunk"]
    sel = np.flatnonzero(z["chunk"] == chunk)
    det, obs = shots(meta, chunk, p)
    saved = np.unpackbits(z["det"][sel], axis=1, count=det.shape[1], bitorder="little")
    result(f"D1 read: saved detector rows of chunk {chunk} are the rebuilt shots", np.array_equal(saved, det))
    dg = torch.as_tensor(det, device=nets.dev)
    for i, sig in enumerate(meta["models"]):
        tag = by_src.get(sig["sha256"])
        pr = nets.get(tag, sig.get("compiled"))
        compare(tag, f"on D1's read p = {p} chunk {chunk} ({sig['prec']})", pr(dg, None), z, i, sel, pr.kind,
                pr.kind == "convlines")
    cz = np.load(verified(d1dir, f"compiled/{pc.p_tag(p)}.npz", sums, prefix="out/"))
    pr = nets.get("C14p", False)
    for B, n, b_key, c_key in ((4096, 8192, "bits4", "conf4"), (1, 2000, "bits1", "conf1")):
        g = pc_speed.build(pr, "compiled", B)
        bits, conf = g.run(dg[:n])
        b, c = bits.cpu().numpy(), conf.float().cpu().numpy()
        dec = int((b != unpack(cz[b_key][:n])).any(1).sum())
        cb = int((c.view(np.uint32) != cz[c_key][:n].view(np.uint32)).sum())
        result(f"C14p compiled@{B} on D1's first {n:,} rows against D1's compiled read", dec == 0, decisions_differ=dec,
               conf_bits_differ=cb)
    return det


def multishot(nets, arch, src):
    import torch
    import m2
    import mfused
    result("the multi-shot engine's launch tiers", len(mfused.TIERS) == 6, tiers=len(mfused.TIERS))
    by_src = {v: k for k, v in src.items()}
    for rd, p, chunk in READS[:2]:
        with open(os.path.join(arch, rd, f"{pc.p_tag(p)}.json")) as fh:
            meta = json.load(fh)
        z = np.load(os.path.join(arch, rd, f"{pc.p_tag(p)}.npz"))
        sel = np.flatnonzero(z["chunk"] == chunk)
        det, _ = shots(meta, chunk, p)
        dg = torch.as_tensor(det, device=nets.dev)
        for i, sig in enumerate(meta["models"]):
            tag = by_src.get(sig["sha256"])
            if tag is None or not tag.startswith("S55"):
                continue
            m = nets.get(tag, False).model
            for B in BATCHES:
                e = m2.make(m, B, "new")
                bits, margin = e.run(dg)
                out = {"bits": bits, "conf": margin,
                       "cls": (bits.long() << torch.arange(bits.shape[1], device=bits.device)).sum(1)}
                compare(tag, f"multi-shot engine at batch {B} on {rd} chunk {chunk}", out, z, i, sel, "convlines", True)


def curve_sets(cdir, dev):
    """The curve sets' networks, r11 and r12 of weights/bb, on recorded w2test chunks (eager fp32, as they ran)."""
    from bb import engine
    from bb.cascade import curves as cv
    data = os.path.join(CAS, "ising_compare", "data", "bb", "cascade")
    cv.SHOTS, cv.PREDS, cv.ROWS = (os.path.join(cdir, "shots"), os.path.join(cdir, "preds"),
                                   os.path.join(data, "shots_w2.csv"))
    sums = listed(os.path.join(data, "preds_sha256.txt"))
    with open(os.path.join(data, "taus.json")) as fh:
        taus = json.load(fh)
    for tag in cv.TAGS:
        f = os.path.join(REPO, "weights", "bb", f"{tag}_final_ema.pt")
        sha = engine.sha256_file(f)
        if not result(f"curve sets: weights/bb/{tag}_final_ema.pt is {tag}", sha == cv.NET_SHA[tag], sha256=sha[:16]):
            continue
        model, _ = engine.load_net(f, dev)
        eng = engine.Deployed(model, "fp32", sha)
        for split, pi, j in CURVES:
            det, obs = cv.load(split, pi, j)            # refuses a chunk whose sha256 or seed is not shots_w2.csv's
            rec = os.path.relpath(cv.pred_path(split, pi, j, f"net-{tag}"), cdir)
            z = np.load(verified(cdir, rec, sums))
            bits, margin = eng.run(det)
            fail = (bits != obs).any(1)
            d = {"rows": len(fail), "decisions_differ": int((bits != z["pred"]).any(1).sum()),
                 "margins_differ": int((margin != z["margin"]).sum()),
                 "margin_max_diff": float(np.abs(margin.astype(np.float64) - z["margin"]).max()),
                 "failures": int(fail.sum()), "failures_recorded": int(z["fail"].sum())}
            ok = d["decisions_differ"] == 0 and (fail == z["fail"]).all()
            if tag == "r12":                            # the routing rule at q = 0.0025 (taus.json)
                t = taus[str(cv.PS[pi])]["0.0025"]
                d["routed"] = int((margin < t).sum())
                d["routed_differ"] = int(((margin < t) != (z["margin"] < t)).sum())
                ok = ok and d["routed_differ"] == 0
            result(f"curve sets: {tag} {split} chunk {j} at p {cv.PS[pi]}", ok, **d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, help="phase C's out/ directory with the per-shot reads")
    ap.add_argument("--d1", required=True, help="D1's out/ directory (read/, compiled/)")
    ap.add_argument("--curves", help="the curve sets' directory (shots/, preds/ as in "
                    "ising_compare/data/bb/cascade/preds_sha256.txt)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    import pc_speed
    t0 = time.time()
    dev, env = pc_speed.setup()                         # phase C's flags; refuses any card but phase C's
    result("GPU", True, gpu=env["gpu"], driver=env["driver"], torch=torch.__version__)
    src = weights()
    nets = Nets(dev)
    for t in FILES:
        n = sum(x.numel() for x in nets.get(t, False).model.parameters())
        result(f"{t}: parameters", PARAMS.get(t, n) == n, parameters=f"{n:,}")
    n14, n6 = (sum(x.numel() for x in nets.get(t, False).model.parameters()) for t in ("C14p", "S55p_deep"))
    result("Cascade's parameters over the six-block network's (the paper: 23 times)", round(n14 / n6) == 23,
           ratio=f"{n14 / n6:.2f}")
    sums = listed(os.path.join(PC, "out", "npz_sha256.txt"))
    phase_c_reads(a.archive, src, nets, sums)
    d1_reads(a.d1, src, nets)
    multishot(nets, a.archive, src)
    if a.curves:
        curve_sets(a.curves, dev)
    bad = [r for r in RES if not r["ok"]]
    with open(os.path.join(a.out, "check_networks.json"), "w") as fh:
        json.dump({"results": RES, "failed": len(bad), "seconds": round(time.time() - t0)}, fh, indent=1)
    print(f"{len(RES)} checks, {len(bad)} failed, {time.time() - t0:.0f} s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
