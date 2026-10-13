"""Streamed reads of the evaluation's computations (cascade_compare/SURFACE_PLAN.md 4.4; sv_common has the
computations): the chunks of 10,000 shots of a block are sampled from their seeds one at a time
(surface_data.stream_shots), each requested computation decodes them, and the per-shot outputs are written part by part
(sv_common.PART = 100 chunks per file) as soon as every computation has decided a part, so that a block of any size
holds a few chunks in memory. The batched computations carry their batches across chunk boundaries: SB in batches of
4,096 consecutive shots, Bb in batches of b*, and BQ1 / BQ5 on the queue of the shots SB routes under C1's / C5's tau1,
in order, in batches of b* (item 9), the last batch of each zero-padded, which are the pipelines' batches. S1 and B1 run
at batch 1 through sv_common.Bulk1. M's computations run in --workers processes that sample the same chunks from the
seeds themselves (no detector rows are sent between processes); MS times each decode. MP (plain PyMatching,
decode_batch) is context only (SURFACE_PLAN 4.1, 7).

  python -m surface.sv_read --stage dev --p 0.006 --chunks 0-100 --comp SB,Bb,MS --speed speed.json --out DIR
  python -m surface.sv_read --stage conf --p 0.0035 --chunks 0-4000 --comp SB,Bb,BQ1,BQ5,MB --lock locked.json \
      --gate gate.json --out DIR [--resume]
  python -m surface.sv_read --stage speed --p 0.006 --chunks 200-250 --comp ... --lock locked.json --out DIR  (checks)
  python -m surface.sv_read --dry --stage dev --p 0.006 --chunks 0-20 --comp S1,SB,B1,Bb,MS,MB --paths graph@1,compiled@4096

Stages give the seed kind (dev 10, conf 11, speed 12, monitor 1); --dry reads kind 13 only and writes nothing that a
real read takes. The paths of B (and b*) come from the lock (conf and speed stages: required) or from the speed
screen's summary (dev: required); a dry run may name them (--paths SINGLE,AMORTIZED).

Before any shot is drawn (finding 2 of the stage gate of 2026-10-12) a read refuses: another card, venv or code than
the lock's (or, for dev, the screen summary's); a conf read without the committed lock and gate (sv_common.
committed_pair); and kernel caches that differ, after this process's compiles and captures, from the lock's (dev: the
summary's). It takes the caches' digest again as it journals each part and at its end; if they changed, its journal
says so before its summary is written, the summary says incomplete, which no analysis accepts, and it exits nonzero.
Such a read never resumes (finding 2 of the recheck of 960b82f): restoring the caches does not make its parts
acceptable.

Output: DIR/p<p>/<comp>_<c0>-<c1>.npz (arrays and the record; SB, Bb, BQ1 and BQ5 also the sha256 of the batch number
of every shot), obs_<c0>-<c1>.npz (the observable and every chunk's digest; shared by every read of those chunks and
published once), the journal progress_<stage>_<comps>_<c0>-<c1>.jsonl (the read's settings and identity, then each
part as it is written, with its files' sha256 and the caches' digest, and an incomplete line if the caches changed),
and at the end read_<time>_<pid>.json (record, timings, files and their sha256, complete or not).

Resume (finding 6). A read that stopped (job time limit, node failure) continues with the same command and --resume:
the journal is checked (resume_state; a last line cut short by the interruption is moved to quarantine and cut off
before anything is appended), and the queues of the batched computations are rebuilt as they stood at the first part
not written (prime_queues), so that every batch, and every batch number, is the one an uninterrupted read forms; the
shots before that part that a rebuilt batch decides again must come out as written, bit for bit. Without --resume,
an existing journal or output is refused."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from surface import sv_common as sc

GPU_COMPS = ("S1", "SB", "B1", "Bb", "BQ1", "BQ5")
NOBATCH = np.uint32(0xFFFFFFFF)
_MW = {}


def _m_task(args):
    """One chunk through M in a worker: sample it from its seed, decode it (MS: one shot per call, timed; MB:
    decode_batch; MP: plain PyMatching's decode_batch, context only) -> (chunk, digest, ms_pred, ms_us, mb_pred,
    mp_pred, obs)."""
    p, kind, j, ms, mb, mpl = args
    if p not in _MW:
        _MW[p] = sc.m_matcher(p)
    if mpl and ("plain", p) not in _MW:
        _MW["plain", p] = sc.m_matcher(p, correlated=False)
    det, obs = sc.sdata.sample(sc.D, p, sc.CHUNK, sc.sdata.eval_seed(sc.D, kind, p, j))
    out = [j, sc.digest(det, obs), None, None, None, None, obs[:, 0]]
    if ms:
        out[2], out[3] = sc.m_single(_MW[p], det)
    if mb:
        out[4] = sc.m_batch(_MW[p], det)
    if mpl:
        out[5] = sc.m_batch(_MW["plain", p], det, correlated=False)
    return out


def frozen_paths(a, lock, speed):
    """B's single and amortized paths ('path@batch') and S's, from the lock, the screen's summary or --paths (dry)."""
    if lock is not None:
        return lock["frozen"]
    if speed is not None:
        return speed["frozen"]
    if a.paths and a.dry:
        b1, bb = a.paths.split(",")
        return {"S": {"single": "engine@1", "batched": f"engine@{sc.S_BATCH}"}, "B": {"single": b1, "amortized": bb}}
    raise SystemExit("B's paths: --lock (conf, speed), --speed (dev) or, in a dry run, --paths")


def split(spec):
    path, b = spec.split("@")
    return path, int(b)


def bsha(b):
    return hashlib.sha256(np.ascontiguousarray(b, np.uint32).tobytes()).hexdigest()[:16]


def jappend(path, obj):
    """One line of a read's journal, synced: a resume trusts what the journal says was written, and nothing else."""
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def part_file(out, c, s, c1):
    return os.path.join(out, f"{c}_{s:05d}-{min(s + sc.PART, c1):05d}.npz")


def resume_state(prog, head, out, comps, c0, c1):
    """An interrupted read's journal (finding 6 of the stage gate) -> (k, files, quarantined). The journal's header
    must be this read's (every setting that fixes the computations and their batches, the environment and code, the
    kernel caches, the lock), its parts contiguous from c0, and each part's files those of the observable and of every
    computation, present with the sha256 the journal names; k is the first chunk they do not cover. A read whose
    caches changed (a part journaled at another digest than the header's, or the incomplete line of its end) never
    resumes, so that none of its parts is reused (finding 2 of the recheck of 960b82f). This read's files of later
    parts (left when the read stopped while writing a part) move to quarantine/, and so does a last line cut short by
    the interruption (jappend writes a line and its newline at once, so a line without one was never finished), which
    is then cut off the journal: the lines appended next stay readable (finding 6 of the recheck). Outcome-blind:
    nothing here reads an observable or a failure, no complete part can be dropped, and a journal or file that fails a
    check stops the resume for a person to look at."""
    raw = open(prog, "rb").read()
    n = raw.rfind(b"\n") + 1                   # the finished lines end here; what follows was cut short
    recs = []
    for i, ln in enumerate(raw[:n].decode(errors="replace").split("\n")[:-1]):
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            raise SystemExit(f"{prog}: line {i + 1} is unreadable; no resume")
    if not recs or recs[0].get("header") != head:
        raise SystemExit(f"{prog}: the interrupted read had other settings, environment, code, kernel caches or lock; "
                         "no resume")
    bad = [r for r in recs[1:] if "incomplete" in r or "part" in r and r.get("kcache") != head["kcache"]]
    if bad:
        raise SystemExit(f"{prog}: the kernel caches changed during the read, so its parts are never reused; no "
                         f"resume ({json.dumps(bad[0])[:300]})")
    k, files = c0, {}
    for r in recs[1:]:
        if "part" not in r:
            continue
        s, e = r["part"]
        names = {f"obs_{s:05d}-{e:05d}.npz"} | {f"{c}_{s:05d}-{e:05d}.npz" for c in comps}
        if s != k or e != min(s + sc.PART, c1) or set(r["files"]) != names:
            raise SystemExit(f"{prog}: the parts are not contiguous from {c0} at {s}, or a part lacks a file")
        for name, sha in r["files"].items():
            f = os.path.join(out, name)
            if not os.path.isfile(f) or sc.sha256_file(f) != sha:
                raise SystemExit(f"{f}: missing or changed since the interrupted read wrote it; no resume")
        files.update(r["files"])
        k = e
    q = os.path.join(out, "quarantine", f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    moved = []
    for c in comps:
        for s in range(k, c1, sc.PART):
            f = part_file(out, c, s, c1)
            if os.path.exists(f):
                os.makedirs(q, exist_ok=True)
                shutil.move(f, q)
                moved.append(os.path.basename(f))
    if n < len(raw):                           # the line cut short: kept in quarantine, then cut off the journal
        os.makedirs(q, exist_ok=True)
        tail = f"{os.path.basename(prog)}.tail_at_{n}"
        with open(os.path.join(q, tail), "wb") as fh:
            fh.write(raw[n:])
        os.truncate(prog, n)
        moved.append(tail)
    return k, files, moved


def prime_queues(q, out, comps, c0, c1, k, kind, p, dev):
    """Resume (finding 6): the queues of the batched computations as they stood when the interrupted read began chunk
    k, so that every batch is the one an uninterrupted read forms. SB's batches are the 4,096-shot blocks from the
    read's first shot and Bb's the b*-shot blocks: each queue gets again the shots of its block that lie before chunk k.
    BQ1 / BQ5 receive the shots SB routes when an SB batch completes; the SB batch rebuilt here starts at shot a0, so
    each gets the shots routed before a0 that no full batch of b* had taken (their positions from its written parts).
    The rows come from the chunks sampled again from their seeds, whose digests must be the ones written; the batch
    counters continue the interrupted read's numbering. -> {comp: (positions, logits, decisions)} as written, for the
    shots before chunk k that a rebuilt batch decides again."""
    import torch
    P = (k - c0) * sc.CHUNK
    a0 = P - P % sc.S_BATCH
    need = {}
    if "SB" in q:
        need["SB"] = (np.arange(a0, P), a0 // sc.S_BATCH)
    for c in ("Bb", "BQ1", "BQ5"):
        if c not in q:
            continue
        bs = q[c].B
        if c == "Bb":
            b0 = P - P % bs
            need[c] = (np.arange(b0, P), b0 // bs)
        else:
            sh = []
            for s in range(c0, k, sc.PART):
                with np.load(part_file(out, c, s, c1)) as z:
                    sh.append(z["shot"] - c0 * sc.CHUNK)
            sh = np.concatenate(sh)
            r = sh[sh < a0]
            st = len(r) - len(r) % bs
            need[c] = (r[st:], st // bs)
    pos = np.unique(np.concatenate([v[0] for v in need.values()] + [np.zeros(0, np.int64)])).astype(np.int64)
    digs = {}
    for s in range(c0, k, sc.PART):
        digs.update(enumerate(sc.load_npz(os.path.join(out, f"obs_{s:05d}-{min(s + sc.PART, c1):05d}.npz"))[0]
                              ["digests"], start=s))
    rows = np.empty((len(pos), sc.sl.Layout(sc.D).D), np.uint8)
    pch = c0 + pos // sc.CHUNK
    for j in np.unique(pch):
        det, obs = sc.sdata.sample(sc.D, p, sc.CHUNK, sc.sdata.eval_seed(sc.D, kind, p, int(j)))
        if sc.digest(det, obs) != digs[int(j)]:
            raise SystemExit(f"resume: chunk {j} sampled again differs from the interrupted read's (digest)")
        sel = pch == j
        rows[sel] = det[pos[sel] - (int(j) - c0) * sc.CHUNK]
    for c, (ps, nb) in need.items():
        q[c].batches = int(nb)
        if len(ps) and q[c].add(torch.from_numpy(rows[np.searchsorted(pos, ps)]).to(dev), ps):
            raise RuntimeError(f"resume: {c} completed a batch while its queue was rebuilt")
    check = {}
    for c, (ps, _) in need.items():
        if c not in comps:
            continue
        lo = int(ps[0]) if len(ps) else a0
        ia, za, da = [], [], []
        for s in range(c0 + (lo // sc.CHUNK // sc.PART) * sc.PART, k, sc.PART):
            _, arr = sc.load_npz(part_file(out, c, s, c1))
            if c in ("BQ1", "BQ5"):
                ia.append(arr["shot"] - c0 * sc.CHUNK)
            else:
                ia.append((s - c0) * sc.CHUNK + np.arange(len(arr["dec"])))
            za.append(arr["c" if c == "SB" else "z"])
            da.append(arr["dec"])
        check[c] = (np.concatenate(ia), np.concatenate(za), np.concatenate(da)) if ia else \
            (np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.uint8))
    return check, {c: [int(len(v[0])), int(v[1])] for c, v in need.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["dev", "conf", "speed", "monitor"])
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--chunks", required=True, help="a-b: chunks a .. b-1 of 10,000 shots")
    ap.add_argument("--comp", required=True, help="comma list of " + ",".join(sc.COMPS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock", default="")
    ap.add_argument("--gate", default="", help="conf: the lock's gate file (sv_route gate)")
    ap.add_argument("--receipt", default="", help="conf: scripts/lockpair.sh's receipt (default: lock_commit.json "
                    "beside the lock)")
    ap.add_argument("--resume", action="store_true", help="continue the interrupted read of these chunks and "
                    "computations from its journal")
    ap.add_argument("--speed", default="", help="the screen's summary (sv_speed summarize): the frozen paths")
    ap.add_argument("--paths", default="", help="dry runs: B's single and amortized paths, e.g. graph@1,compiled@4096")
    ap.add_argument("--tau1", default="", help="dry runs without a lock: C1's and C5's tau1 for BQ1 / BQ5, e.g. 1.2,0.8")
    ap.add_argument("--ckpt", default="final", choices=["final", "budget"], help="budget: dry description only")
    ap.add_argument("--dry", action="store_true", help="kind 13 shots only")
    ap.add_argument("--workers", type=int, default=16, help="M's processes")
    ap.add_argument("--sample-procs", type=int, default=2)
    ap.add_argument("--streams", type=int, default=8, help="Bulk1: graphs, each on its own CUDA stream (B1 on the "
                    "dry run: 5,620 shots/s at 4, 7,040 at 8, 6,580 at 16, the same bits)")
    ap.add_argument("--calls", type=int, default=32, help="Bulk1: batch-1 calls per graph")
    ap.add_argument("--ref-s", type=int, default=0, help="also compare S's engine with its fp32 module on N shots")
    a = ap.parse_args()
    comps = a.comp.split(",")
    if set(comps) - set(sc.COMPS):
        raise SystemExit(f"unknown computations {set(comps) - set(sc.COMPS)}")
    if a.ckpt != "final" and not a.dry:
        raise SystemExit("every read uses final_ema.pt (SURFACE_PLAN 3.3)")
    kind = sc.kind_of(a.stage, a.dry)
    chunks = sc.chunk_range(a.chunks)
    c0, c1 = chunks.start, chunks.stop
    lock = json.load(open(a.lock)) if a.lock else None
    if a.stage in ("conf", "speed") and lock is None and not a.dry:
        raise SystemExit(f"--stage {a.stage} needs the lock")
    if lock is not None and lock.get("dry", False) != a.dry:
        raise SystemExit("a dry lock serves dry reads only, and a real lock real reads only")
    receipt = {}
    if a.stage == "conf":          # finding 2: the committed lock and gate before any confirmation shot is drawn
        if lock is None or not a.gate:
            raise SystemExit("--stage conf needs the lock and its gate file (--gate)")
        receipt = sc.committed_pair(a.lock, a.gate, a.receipt)[1]
    speed = json.load(open(a.speed)) if a.speed else None
    fz = frozen_paths(a, lock, speed)
    out = os.path.join(a.out, sc.p_tag(a.p))
    os.makedirs(out, exist_ok=True)
    prog = os.path.join(out, f"progress_{a.stage}_{'+'.join(sorted(comps))}_{c0:05d}-{c1:05d}.jsonl")
    resume = os.path.exists(prog)
    if resume and not a.resume:
        raise SystemExit(f"{prog} exists: an interrupted read continues with --resume; a read never overwrites")
    if not resume:
        for c in comps:
            for k in range(c0, c1, sc.PART):
                if os.path.exists(part_file(out, c, k, c1)):
                    raise SystemExit(f"{part_file(out, c, k, c1)} exists without a journal: a read never overwrites")
    taus = {}
    if {"BQ1", "BQ5"} & set(comps):
        if lock is not None:
            ch = lock["p"][str(a.p)]["chosen"]
            taus = {"BQ1": ch["C1"]["tau1"], "BQ5": ch["C5"]["tau1"]}
        elif a.dry and a.tau1:
            t1, t5 = (float(x) for x in a.tau1.split(","))
            taus = {"BQ1": t1, "BQ5": t5}
        else:
            raise SystemExit("BQ1 / BQ5 need C1's and C5's tau1 (the lock)")
    t_start = time.time()
    dev, gpu = sc.setup_gpu()                     # M's reads too run on the evaluation card's node (SURFACE_PLAN 5)
    rec = sc.record(gpu)
    lock_sha = sc.sha256_file(a.lock) if a.lock else ""
    ref = lock if lock is not None else speed     # the lock; before it, for the development reads, the screen's summary
    if ref is not None:
        sc.same_identity(rec, ref["identity"], "this read")
    nets = {}
    lay = sc.sl.Layout(sc.D)
    Dd = lay.D
    S_used = bool({"S1", "SB", "BQ1", "BQ5"} & set(comps)) or a.ref_s
    B_used = bool({"B1", "Bb", "BQ1", "BQ5"} & set(comps))
    t_build = time.time()
    if S_used:
        smod, tabs, scfg, nets["S"] = sc.load_S()
    if B_used:
        bmod, nets["B"] = sc.load_B(dev, a.ckpt)
    if lock is not None:
        for t, r in nets.items():
            if r["sha256"] != lock["nets"][t]["sha256"]:
                raise SystemExit(f"{t}: sha256 differs from the lock's")
    q, bulk, paths = {}, {}, {}
    if "S1" in comps:
        bulk["S1"] = sc.Bulk1(lambda: sc.s_fn(tabs, scfg), Dd, dev, a.calls, a.streams)
        paths["S1"] = fz["S"]["single"]
    if {"SB", "BQ1", "BQ5"} & set(comps):
        q["SB"] = sc.Queue(sc.Graph(sc.s_fn(tabs, scfg), sc.S_BATCH, Dd, dev))
        paths["SB"] = fz["S"]["batched"]
    if "B1" in comps:
        p1, b1 = split(fz["B"]["single"])
        assert b1 == 1, fz
        f1 = sc.b_fn(bmod, p1)
        bulk["B1"] = sc.Bulk1(lambda: f1, Dd, dev, a.calls, a.streams)
        paths["B1"] = fz["B"]["single"]
    if {"Bb", "BQ1", "BQ5"} & set(comps):
        pb, bstar = split(fz["B"]["amortized"])
        gb = sc.Graph(sc.b_fn(bmod, pb), bstar, Dd, dev)          # one graph serves Bb and both queues
        for c in ("Bb", "BQ1", "BQ5"):
            if c in comps:
                q[c] = sc.Queue(gb)
                paths[c] = fz["B"]["amortized"]
    t_build = time.time() - t_build
    rec["kcache"] = sc.kcache()       # after this process's compiles and captures
    ref_kc = ref["kcache"]["digest"] if ref is not None else rec["kcache"]["digest"]
    if rec["kcache"]["digest"] != ref_kc:          # finding 2: before any shot is drawn
        raise SystemExit(f"kernel caches {rec['kcache']} after this read's compiles and captures, not the reference's "
                         f"{ref_kc}: something was compiled or tuned anew; no shot drawn")
    ms_on, mb_on, mp_on = "MS" in comps, "MB" in comps, "MP" in comps
    if ms_on:
        paths["MS"] = "decode@1"
    if mb_on:
        paths["MB"] = "decode_batch"
    if mp_on:
        paths["MP"] = "plain decode_batch"
    head = json.loads(json.dumps({
        "stage": a.stage, "dry": a.dry, "p": a.p, "kind": kind, "chunks": [c0, c1], "comps": sorted(comps),
        "part": sc.PART, "s_batch": sc.S_BATCH, "paths": paths, "taus": taus, "ckpt": a.ckpt,
        "nets": {t: r["sha256"] for t, r in nets.items()}, "lock_sha256": lock_sha,
        "lock_commit": receipt.get("commit", ""), "bulk1": {"streams": a.streams, "calls": a.calls},
        "identity": {k: rec.get(k) for k in sc.IDENTITY}, "kcache": rec["kcache"]["digest"]}))
    files, moved, k = {}, [], c0
    if resume:
        k, files, moved = resume_state(prog, head, out, comps, c0, c1)
    else:
        with open(prog, "x") as fh:                # exclusive: one read of these chunks and computations
            fh.write(json.dumps({"header": head}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    jappend(prog, {"attempt": {"job": rec["job"], "host": rec["host"], "pid": os.getpid(), "time": rec["time"],
                               "from_chunk": k, "quarantined": moved}})
    P = (k - c0) * sc.CHUNK                        # the first shot (position in the read) not yet written
    N = (c1 - c0) * sc.CHUNK
    check, rebuilt, verified = {}, {}, {}
    if c0 < k < c1 and q:
        check, rebuilt = prime_queues(q, out, comps, c0, c1, k, kind, a.p, dev)
    pool = mp.get_context("spawn").Pool(a.workers) if (ms_on or mb_on or mp_on) and k < c1 else None
    mres, mpend = {j: True for j in range(c0, k)}, []
    parts = {}                                     # part start chunk -> {comp: arrays}
    obs_how = {"created": [], "verified": []}

    def part_of(pos):
        return c0 + ((pos // sc.CHUNK) // sc.PART) * sc.PART

    def slot(pos):
        k_ = part_of(pos)
        if k_ not in parts:
            n = (min(k_ + sc.PART, c1) - k_) * sc.CHUNK
            parts[k_] = {"obs": np.zeros(n, np.uint8), "dig": {}, "BQ1": [], "BQ5": []}
            for c in ("S1", "SB", "B1", "Bb"):
                if c in comps:
                    parts[k_][c] = (np.full(n, np.nan, np.float32), np.full(n, 255, np.uint8)) + \
                        ((np.full(n, NOBATCH, np.uint32),) if c in ("SB", "Bb") else ())
            if ms_on:
                parts[k_]["MS"] = (np.full(n, 255, np.uint8), np.full(n, np.nan, np.float32))
            for c in ("MB", "MP"):
                if c in comps:
                    parts[k_][c] = np.full(n, 255, np.uint8)
        return parts[k_], (pos - (k_ - c0) * sc.CHUNK)

    def verify(c, idx, lg, dc):
        """Shots before the resume point that a rebuilt batch decided again: as the interrupted read wrote them."""
        if c not in check:
            raise RuntimeError(f"resume: {c} decided shots before the resume point that it never wrote")
        ia, za, da = check[c]
        j = np.searchsorted(ia, idx)
        if (j >= len(ia)).any() or (ia[np.minimum(j, len(ia) - 1)] != idx).any():
            raise RuntimeError(f"resume: {c} decided a shot before the resume point that its written parts lack")
        bad = int((za[j].view(np.uint32) != np.asarray(lg, np.float32).view(np.uint32)).sum() + (da[j] != dc).sum())
        if bad:
            raise RuntimeError(f"resume: {c} gave {bad} values on shots before the resume point that differ from the "
                               "interrupted read's; no resume")
        verified[c] = verified.get(c, 0) + len(idx)

    def put(c, idx, lg, dc, b=None):
        """Results of computation c (batch number b) at read positions idx (they may span parts)."""
        idx = np.asarray(idx)
        old = idx < P
        if old.any():
            verify(c, idx[old], lg[old], dc[old])
            idx, lg, dc = idx[~old], lg[~old], dc[~old]
        if not len(idx):
            return
        ks = (idx // sc.CHUNK) // sc.PART
        for kk in np.unique(ks):
            sel = ks == kk
            pa, off = slot(int(idx[sel][0]))
            loc = idx[sel] - (int(idx[sel][0]) - off)
            if c in ("BQ1", "BQ5"):
                pa[c].append(((c0 * sc.CHUNK) + idx[sel], lg[sel], dc[sel], np.full(int(sel.sum()), b, np.uint32)))
            else:
                pa[c][0][loc] = lg[sel]
                pa[c][1][loc] = dc[sel]
                if c in ("SB", "Bb"):
                    pa[c][2][loc] = b
    stats = {c: {"secs": 0.0, "shots": 0, "fail": 0} for c in comps}
    write_t = [0.0]
    changed = []                                   # parts this process journaled with the caches changed

    def write(k_):
        pa = parts.pop(k_)
        e = min(k_ + sc.PART, c1)
        digs = [pa["dig"][j] for j in range(k_, e)]
        meta0 = {"stage": a.stage, "dry": a.dry, "p": a.p, "kind": kind, "chunks": [k_, e], "digests": digs,
                 "record": rec, "lock_sha256": lock_sha, "lock_commit": receipt.get("commit", ""), "ckpt": a.ckpt,
                 "bulk1": {"streams": a.streams, "calls": a.calls}}
        tw = time.time()
        pf = {}
        fo = f"obs_{k_:05d}-{e:05d}.npz"
        sha = sc.save_npz(os.path.join(out, fo), meta0, shared=True, obs=pa["obs"])
        if sha is None:                            # another read of these chunks published it first (e5 beside e6)
            mo, ao, sha = sc.load_npz(os.path.join(out, fo), sha=True)
            if mo["digests"] != digs or not np.array_equal(ao["obs"], pa["obs"]):
                raise RuntimeError(f"{fo}: another read of these chunks saw other shots")
            obs_how["verified"].append(fo)
        else:
            obs_how["created"].append(fo)
        pf[fo] = sha
        for c in comps:
            m = {**meta0, "comp": c, "path": paths[c],
                 "net": nets.get("S" if c[0] == "S" else "B") if c[0] in "SB" else {"net": "M", "p": a.p}}
            if c in ("BQ1", "BQ5"):
                m["tau1"] = taus[c]
                rows = pa[c]
                shot = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int64)
                z = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, np.float32)
                dc = np.concatenate([r[2] for r in rows]) if rows else np.zeros(0, np.uint8)
                m["batch_sha256"] = bsha(np.concatenate([r[3] for r in rows]) if rows else np.zeros(0, np.uint32))
                arr = {"shot": shot, "z": z, "dec": dc}
                obs_r = pa["obs"][shot - k_ * sc.CHUNK]
                stats[c]["fail"] += int((dc != obs_r).sum())
                stats[c]["shots"] += len(dc)
            elif c in ("MS", "MB", "MP"):
                pred = pa[c][0] if c == "MS" else pa[c]
                if (pred == 255).any():
                    raise RuntimeError(f"{c}: part {k_} incomplete")
                arr = {"pred": pred, **({"us": pa[c][1]} if c == "MS" else {})}
                stats[c]["fail"] += int((pred != pa["obs"]).sum())
                stats[c]["shots"] += len(pred)
            else:
                lg, dc = pa[c][0], pa[c][1]
                if (dc == 255).any() or c in ("SB", "Bb") and (pa[c][2] == NOBATCH).any():
                    raise RuntimeError(f"{c}: part {k_} incomplete")
                if c in ("SB", "Bb"):
                    m["batch_sha256"] = bsha(pa[c][2])
                arr = {("c" if c[0] == "S" else "z"): lg, "dec": dc}
                stats[c]["fail"] += int((dc != pa["obs"]).sum())
                stats[c]["shots"] += len(dc)
            name = f"{c}_{k_:05d}-{e:05d}.npz"
            pf[name] = sc.save_npz(os.path.join(out, name), m, **arr)
        files.update(pf)
        kc = sc.kcache()["digest"]                 # after every computation of the part (finding 2 of the recheck)
        if kc != ref_kc:
            changed.append([k_, e])
        jappend(prog, {"part": [k_, e], "files": pf, "kcache": kc})
        write_t[0] += time.time() - tw

    def m_collect(block=False):
        while mpend and (block or mpend[0].ready()):
            j, dg, msp, msu, mbp, mpp, ob = mpend.pop(0).get()
            mres[j] = True
            pa, off = slot((j - c0) * sc.CHUNK)
            if pa["dig"].setdefault(j, dg) != dg:
                raise RuntimeError(f"chunk {j}: M's shots differ from the GPU's (digest)")
            sl_ = slice(off, off + sc.CHUNK)
            pa["obs"][sl_] = ob
            if msp is not None:
                pa["MS"][0][sl_], pa["MS"][1][sl_] = msp, msu
            if mbp is not None:
                pa["MB"][sl_] = mbp
            if mpp is not None:
                pa["MP"][sl_] = mpp

    sref = {}
    fed, t_wait = P, 0.0
    gpu_on = bool(set(comps) & set(GPU_COMPS))
    tw0 = time.time()
    todo = range(k, c1)
    src = sc.shots(a.p, todo, kind, a.sample_procs) if (gpu_on or a.ref_s) and len(todo) else None
    for j in todo:
        pos0 = (j - c0) * sc.CHUNK
        if pool is not None:
            mpend.append(pool.apply_async(_m_task, ((a.p, kind, j, ms_on, mb_on, mp_on),)))
        if src is None:
            fed = pos0 + sc.CHUNK
            m_collect()
            flush_parts(parts, q, comps, fed, mres, c0, write, final=False)
            continue
        jj, det, obs = next(src)
        t_wait += time.time() - tw0
        assert jj == j
        pa, off = slot(pos0)
        dg = sc.digest(det, obs)
        if pa["dig"].setdefault(j, dg) != dg:
            raise RuntimeError(f"chunk {j}: the GPU's shots differ from M's (digest)")
        pa["obs"][off:off + sc.CHUNK] = obs[:, 0]
        idx = pos0 + np.arange(sc.CHUNK)
        if a.ref_s and not sref:
            sref = s_reference(smod, tabs, scfg, det[:a.ref_s], dev)
        for c in ("S1", "B1"):
            if c in bulk:
                t = time.time()
                lg, dc = bulk[c].run(det)
                stats[c]["secs"] += time.time() - t
                put(c, idx, lg, dc)
        if q:
            import torch
            t = time.time()
            xg = torch.from_numpy(det).pin_memory().to(dev, non_blocking=True)
            stats.setdefault("h2d", {"secs": 0.0})["secs"] += time.time() - t
            if "SB" in q:
                feed_S(q, xg, idx, taus, put, stats, comps)
            if "Bb" in q:
                t = time.time()
                for ii, lg, dc, _, b in q["Bb"].add(xg, idx):
                    put("Bb", ii, lg, dc, b)
                stats["Bb"]["secs"] += time.time() - t
        fed = pos0 + sc.CHUNK
        m_collect()
        flush_parts(parts, q, comps, fed, mres, c0, write, final=False)
        tw0 = time.time()
    if k < c1:                                     # the last batches (a read that resumes at its end has none)
        if "SB" in q:
            t = time.time()
            out_ = q["SB"].flush()
            stats.setdefault("SB", {"secs": 0.0})["secs"] += time.time() - t
            for ii, lg, dc, rows, b in out_:
                route_S(q, ii, lg, dc, rows, b, taus, put, stats, comps)
        for c in ("BQ1", "BQ5", "Bb"):
            if c in q:
                t = time.time()
                for ii, lg, dc, _, b in q[c].flush():
                    put(c, ii, lg, dc, b)
                stats[c]["secs"] += time.time() - t
    tm = time.time()
    m_collect(block=True)
    t_mtail = time.time() - tm
    flush_parts(parts, q, comps, N, mres, c0, write, final=True)
    if pool is not None:
        pool.close()
        pool.join()
    secs = time.time() - t_start
    kc_end = sc.kcache()
    complete = kc_end["digest"] == ref_kc and not changed and not parts
    if not complete:                               # journaled first, so that no resume reuses the parts
        jappend(prog, {"incomplete": {"kcache_end": kc_end, "reference": ref_kc, "parts_changed": changed,
                                      "job": rec["job"], "pid": os.getpid(),
                                      "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}})
    summ = {"stage": a.stage, "dry": a.dry, "p": a.p, "kind": kind, "chunks": [c0, c1], "shots": N, "comps": comps,
            "paths": paths, "taus": taus, "nets": nets, "record": rec, "kcache_end": kc_end,
            "kcache_parts_changed": changed, "complete": complete,
            "lock_sha256": lock_sha, "lock_commit": receipt.get("commit", ""), "journal": os.path.basename(prog),
            "resume": {"from_chunk": k, "quarantined": moved, "rebuilt": rebuilt, "verified": verified} if resume
            else None, "obs": obs_how,
            "secs": round(secs, 1), "build_s": round(t_build, 1), "sample_wait_s": round(t_wait, 1),
            "m_tail_s": round(t_mtail, 1), "write_s": round(write_t[0], 1),
            "stats": {c: {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in s.items()}
                      for c, s in stats.items()},
            "batches": {c: qq.batches for c, qq in q.items()}, "s_reference": sref, "files": files}
    for c in comps:
        s = summ["stats"][c]
        if s.get("shots"):
            s["shots_per_s"] = round(s["shots"] / s["secs"], 1) if s["secs"] else None
    name = os.path.join(out, f"read_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json")
    sc.write_json(name, summ)
    print(json.dumps({kk: summ[kk] for kk in ("stage", "p", "kind", "chunks", "complete", "resume", "secs",
                                              "sample_wait_s", "m_tail_s", "write_s", "stats", "s_reference")}),
          flush=True)
    if not complete:
        raise SystemExit(f"the kernel caches changed during the read ({kc_end} at the end, parts {changed} journaled "
                         f"at another digest, against {ref_kc}): no analysis takes its parts, and no resume reuses "
                         "them")


def feed_S(q, xg, idx, taus, put, stats, comps):
    t = time.time()
    out = q["SB"].add(xg, idx)
    stats.setdefault("SB", {"secs": 0.0})["secs"] += time.time() - t
    for ii, lg, dc, rows, b in out:
        route_S(q, ii, lg, dc, rows, b, taus, put, stats, comps)


def route_S(q, ii, lg, dc, rows, b, taus, put, stats, comps):
    """An SB batch's results (batch number b): stored, and the shots it routes under C1's / C5's tau1 queued for BQ1 /
    BQ5 (their time counted as theirs)."""
    import torch
    if "SB" in comps:
        put("SB", ii, lg, dc, b)
    for c in ("BQ1", "BQ5"):
        if c in q:
            t = time.time()
            sel = sc.routed(lg, taus[c])
            if sel.any():
                for jj, l2, d2, _, b2 in q[c].add(rows[torch.as_tensor(np.nonzero(sel)[0], device=rows.device)],
                                                  ii[sel]):
                    put(c, jj, l2, d2, b2)
            stats[c]["secs"] += time.time() - t


def flush_parts(parts, q, comps, fed, mres, c0, write, final):
    """Write every part that every computation has decided: SB / Bb / the queues up to their first waiting shot, M up
    to its first chunk not back."""
    wm = fed
    if "SB" in q:
        wm = min(wm, q["SB"].watermark(fed))
    for c in ("BQ1", "BQ5"):
        if c in q:
            wm = min(wm, q[c].watermark(q["SB"].watermark(fed)))
    if "Bb" in q:
        wm = min(wm, q["Bb"].watermark(fed))
    if {"MS", "MB", "MP"} & set(comps):
        j = c0
        while j in mres:
            j += 1
        wm = min(wm, (j - c0) * sc.CHUNK)
    for k in sorted(parts):
        last = (k - c0) * sc.CHUNK + len(parts[k]["obs"])
        if last <= wm or final:
            write(k)


def s_reference(smod, tabs, scfg, det, dev):
    """S's engine (batch 1 as deployed, timing.deployed.Logits' loop) against the fp32 module (TF32 off) on the same
    shots: the largest |c| difference and the decisions and routings (at C4's tau) that differ."""
    import torch
    fn = sc.s_fn(tabs, scfg)
    x = torch.as_tensor(det, device=dev)
    ce = np.empty(len(det), np.float32)
    with torch.no_grad():
        for i in range(len(det)):
            ce[i] = float(fn(x[i:i + 1])[0][0])
        m = smod.to(dev)
        cr = m(x)[0].float().cpu().numpy()
        smod.cpu()
    return {"shots": len(det), "max_abs_diff": float(np.abs(ce - cr).max()),
            "decisions_differ": int(((ce > 0) != (cr > 0)).sum()),
            "routing_differ_c4": int((sc.routed(ce, sc.TAU_C4) != sc.routed(cr, sc.TAU_C4)).sum())}


if __name__ == "__main__":
    main()
