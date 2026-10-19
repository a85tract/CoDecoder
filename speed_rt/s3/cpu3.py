"""SPEED-RT stage 3 (speed_rt/STAGE3_batched.md): one fallback process of a CPU term C(N) or a baseline B(N).

Process k of a group of N decodes its equal contiguous chunk of the item's rows from memory with the decoder's batch entry
point, single-threaded: PyMatching's decode_batch (plain, or enable_correlations=True), relay-bp's
ObservableDecoderRunner.decode_observables_batch(parallel=False) on Relay-BP built as stage 2 built it
(qnd.teachers.Decoder("relay", ...), sector "basis" or the full DEM "all"). The rows: an item's routed shots in shot
order (the decision pass's flags; a CPU term), or every shot (a baseline), or the first FIRST of either (an item whose count
the registration cuts). Before the timed call the process builds its decoder, loads the shots and decodes WARM rows of the
reference chunk untimed; then the group's barrier: each process writes ready_<gid>_<k> (CLOCK_MONOTONIC, ns) into the
group's directory and waits until N exist; the common start T is the largest of them plus 20 ms, every process waits for
T, makes its one batch call and reads the clock at its end. The group's wall time is the largest end minus T.

    python -m speed_rt.s3.cpu3 --items ITEM[,ITEM...] --k K --n N --rep R --group DIR --out DIR
ITEMs are keys of ITEMS below, decoded one after another, each with its own barrier (group id <item>_N<n>_r<rep>) and
its record DIR/<item>/cpu3.json; their shots and decision-pass flags are read from $S3_DATA (the tmpfs)."""
import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(os.path.dirname(HERE))
DATA = os.environ.get("S3_DATA", "/dev/shm/srt3_box/data")
WARM = 20
TREES = {"bb72": "bb/bb72", "gross": "bb/gross_stage4"}
BB = {"bb72-p2": ("bb72", 0.002), "bb72-p3": ("bb72", 0.003), "gross-p4": ("bb144", 0.004)}


def items():
    """item -> (point, decoder, rows, first): decoder pm | cpm | sector | full; rows 'all' or a routing flag name."""
    it = {}
    for d in (7, 9, 11, 13, 15, 17):
        pt = f"surf-d{d}"
        it[f"{pt}.ours"] = (pt, "cpm", "q", None)
        it[f"{pt}.pm"] = (pt, "pm", "all", None)
        it[f"{pt}.cpm"] = (pt, "cpm", "all", None)
    for pt in ("bb72-p2", "bb72-p3"):
        it[f"{pt}.bal"] = (pt, "sector", "q", None)
        it[f"{pt}.acc"] = (pt, "full", "q", None)
        it[f"{pt}.sector"] = (pt, "sector", "all", None)
        it[f"{pt}.full"] = (pt, "full", "all", FIRST[f"{pt}.full"])
    it["gross-p4.bal"] = ("gross-p4", "sector", "bal", None)
    it["gross-p4.acc"] = ("gross-p4", "full", "acc", None)
    it["gross-p4.sector"] = ("gross-p4", "sector", "all", FIRST["gross-p4.sector"])
    it["gross-p4.full"] = ("gross-p4", "full", "all", FIRST["gross-p4.full"])
    return it


# The registered cuts (STAGE3_batched.md section 4): these baselines time the first FIRST shots of their timing set.
FIRST = {"bb72-p2.full": 4000, "bb72-p3.full": 4000, "gross-p4.sector": 1000, "gross-p4.full": 40}
USE = {"gross-p4": 2000}    # the gross code's timing set: the first 2,000 shots of s3-gross-p4 chunk 0
ITEMS = items()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_point(pt):
    """(det (n, m) uint8 of the point's timing set in chunk order, files with sha256) from $S3_DATA/shots/<point>/."""
    d = os.path.join(DATA, "shots", pt)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    dets, obs, rec = [], [], []
    for f in files:
        p = os.path.join(d, f)
        rec.append((f, sha256_file(p)))
        z = np.load(p)
        if pt.startswith("surf"):
            from convlines import testset
            x, o = testset.load(p, int(pt.split("-d")[1]))
        else:
            x = np.unpackbits(z["det"], axis=1, count=int(z["D"]))[:int(z["n"])]
            o = np.unpackbits(z["obs"], axis=1, count=int(z["k"]))[:int(z["n"])]
        dets.append(np.ascontiguousarray(x, np.uint8))
        obs.append(np.asarray(o, np.uint8).reshape(len(x), -1))
    det, ob = np.concatenate(dets), np.concatenate(obs)
    if pt in USE:
        det, ob = det[:USE[pt]], ob[:USE[pt]]
    return det, ob, rec


def decoder(pt, kind):
    """-> (f(rows uint8 (r, m)) -> predicted observables, a description)."""
    if pt.startswith("surf"):
        from convlines import circuits
        d = int(pt.split("-d")[1])
        m = circuits.matcher(d, correlated=kind == "cpm")
        if kind == "cpm":
            return (lambda x: m.decode_batch(x, enable_correlations=True)), f"pymatching decode_batch correlated, d {d}"
        return (lambda x: m.decode_batch(x)), f"pymatching decode_batch plain, d {d}"
    import relay_bp
    from qnd import teachers
    code, p = BB[pt]
    t = teachers.Decoder("relay", code, "X", p, "basis" if kind == "sector" else "all")
    run = relay_bp.ObservableDecoderRunner(t.dec, t.L, include_decode_result=False)
    keep = t.keep
    return (lambda x: run.decode_observables_batch(np.ascontiguousarray(x[:, keep]), parallel=False, progress_bar=False),
            f"relay-bp RelayDecoderF32 {teachers.describe('relay', 'basis' if kind == 'sector' else 'all')}, batch call")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="items, comma-separated, decoded one after another")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = a.items.split(",")
    assert all(i in ITEMS for i in items), items
    sys.path.insert(0, os.path.join(REL, "surface_v2"))
    cache = {}
    for item in items:
        pt, kind, rows, first = ITEMS[item]
        if not pt.startswith("surf"):
            q = os.path.join(REL, TREES[pt.split("-")[0]], "qldpc_neural")
            sys.path[:] = [x for x in sys.path if not x.endswith("qldpc_neural")]
            sys.path.insert(0, q)
            for m in [m for m in sys.modules if m == "qnd" or m.startswith("qnd.")]:
                del sys.modules[m]
        t_start = time.time()
        if pt not in cache:
            cache.clear()
            cache[pt] = load_point(pt)
        det, obs, files = cache[pt]
        n_all = len(det) if first is None else min(first, len(det))
        if rows == "all":
            sel = np.arange(n_all)
        else:
            z = np.load(os.path.join(DATA, "decide", f"decide_{pt}.npz"))
            sel = np.flatnonzero(z[f"routed_{rows}"][:n_all].astype(bool))
        lo, hi = a.k * len(sel) // a.n, (a.k + 1) * len(sel) // a.n
        mine = np.ascontiguousarray(det[sel[lo:hi]])
        f, desc = decoder(pt, kind)
        warm = np.load(os.path.join(DATA, "warm", f"warm_{pt}.npy"))[:WARM]
        f(np.ascontiguousarray(warm))                                    # untimed: the decoder's first calls
        gid = f"{item}_N{a.n}_r{a.rep}"
        io0 = open("/proc/self/io").read()
        os.makedirs(a.group, exist_ok=True)
        tmp = os.path.join(a.group, f"ready_{gid}_{a.k}.tmp")
        with open(tmp, "w") as fh:
            fh.write(str(time.monotonic_ns()))
        os.rename(tmp, tmp[:-4])
        deadline = time.monotonic() + 60                 # a peer waits at most 60 s
        r = [os.path.join(a.group, f"ready_{gid}_{j}") for j in range(a.n)]
        while not all(os.path.exists(x) for x in r):
            if time.monotonic() > deadline:
                return 3
            time.sleep(0.002)
        T = max(int(open(x).read()) for x in r) + 20_000_000
        while time.monotonic_ns() < T:
            pass
        t0 = time.monotonic_ns()
        pred = f(mine) if len(mine) else np.zeros((0, obs.shape[1]), np.uint8)    # a process may have no row
        t1 = time.monotonic_ns()
        io1 = open("/proc/self/io").read()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        pred = np.asarray(pred, np.uint8).reshape(len(mine), obs.shape[1])
        failures = int((pred != obs[sel[lo:hi]]).any(1).sum())             # descriptive only
        od = os.path.join(a.out, item)
        os.makedirs(od, exist_ok=True)
        np.save(os.path.join(od, "pred.npy"), np.packbits(pred, axis=1))
        rec = dict(item=item, point=pt, decoder=kind, desc=desc, rows=rows, first=first, k=a.k, n=a.n, rep=a.rep, gid=gid,
                   timing_shots=int(n_all), rows_total=int(len(sel)), rows_mine=int(hi - lo), row_lo=int(lo),
                   row_hi=int(hi), T_ns=T, t0_ns=t0, t1_ns=t1, late_ns=t0 - T, call_s=(t1 - t0) / 1e9, files=files,
                   pred_sha256=hashlib.sha256(pred.tobytes()).hexdigest(), failures=failures, io_before=io0,
                   io_after=io1, ru_inblock=ru.ru_inblock, ru_majflt=ru.ru_majflt, maxrss_kb=ru.ru_maxrss,
                   setup_s=round(time.time() - t_start - (t1 - t0) / 1e9, 3), cpu=sorted(os.sched_getaffinity(0)),
                   threads=os.environ.get("RAYON_NUM_THREADS"))
        with open(os.path.join(od, "cpu3.json"), "x") as fh:
            json.dump(rec, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
