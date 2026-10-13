"""The curve sets, exploratory (not a registered study): BB72's two networks across the physical error rates of Cascade's Fig. 2(a)
(Gu, Bonilla Ataides, Lukin, Yelin, arXiv:2604.08358), converted to Cascade's metric by its Eq. (1). Fresh shots from
their own seed namespace; no registered split (cal, dev, test, monitor) is read. Record: bb/cascade/CASCADE_CURVES.md.
  check   python -m bb.cascade.curves check: the p = 0.55 % circuit, absent from relay d185194's testdata, is the 0.5 %
          file with every noise argument "(0.005)" replaced by "(0.0055)"; the same substitution must map the relay
          files at 0.1 / 0.2 / 0.3 / 0.4 % onto its files at 0.15 / 0.25 / 0.35 / 0.45 % byte for byte (exit 1 if not)
  sample  python -m bb.cascade.curves sample --split w2test --rows R.csv [--procs P]: every chunk of the split's PLAN on
          this host, sampled and stored as bb.shots does (one stim seed per chunk of 10,000 shots, packed npz), one
          manifest row each (sha256, seed, host, cpu, stim); load() checks every read against W2_ROWS
  infer   python -m bb.cascade.curves infer --split w2test: r11 and r12 (BB_CKPT/<tag>/final_ema.pt, sha256 = the
          registered networks) through bb.engine's fp32 path (eager PyTorch, batches of 2,048) on every chunk:
          W2_DIR/preds/<chunk>_net-<tag>.npz (pred, margin, fail, engine signature); a stored chunk is kept
  taus    python -m bb.cascade.curves taus: tau_q(p) = the q-quantile of r12's margins on the w2cal block at p (numpy's
          default interpolation, bb.route.frontier's rule), q in QS -> W2_DIR/out/taus.json
  pilot   python -m bb.cascade.curves pilot --n N: full-DEM Relay-BP's time per shot on N routed (margin < tau at
          max(QS)) and N random shots at every p -> W2_DIR/out/pilot.json (timing only, never counted)
  relay   python -m bb.cascade.curves relay --run a --shard k/K [--pis 2-9]: full-DEM Relay-BP (bb.opp's relay_full: relay-bp
          0.2.2, the authors' gross-code parameters, both detector sectors, priors of the circuit at p) on every shot of
          the decode plan: at every p the shots r12 routes at max(QS) (the q = 0.25 % set is inside it), plus every shot
          of the first ALONE[p] chunks (full-DEM Relay-BP alone). Shots stored by an earlier run are skipped; the rest,
          in tasks of SLICE shots, go round-robin to the K shards; per chunk and shard
          W2_DIR/preds/<chunk>_relay_full_<run>_s<k>of<K>.npz (idx, pred, fail, secs); a stored file is kept (resume).
          Each shot is decoded once, so the accurate mode and Relay-BP alone share its decision (as bb.fullroute)
  report  python -m bb.cascade.curves report --out DIR: counts, P_block with its 95 % Wilson interval, P_L by Eq. (1),
          ratios to the paper's digitised curves -> DIR/curves_counts.json and DIR/curves.png
Seeds (chunk j of 10,000 shots at p = PS[pi]): w2test 2^50 + 2 x 10^9 + 10^6 pi + j, w2cal 2^50 + 3 x 10^9 + 10^6 pi + j."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import time
from functools import lru_cache
from glob import glob
from multiprocessing import get_context

import numpy as np

from qnd import circuits

CODE, K, R = "bb72", 12, 6
PS = (0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055)   # pi = the index here
SPLITS = {"w2test": 2, "w2cal": 3}
CHUNK = 10_000
# chunks per p; w2test was first (200, 60, 30, 20, 10, 5, ...), then extended at 0.1 / 0.15 % (w2cal: 2 and 22
# network failures per 100,000 shots there, so the first sizes would have given ~40 and ~130 instead of >= 200)
PLAN = {"w2test": (1000, 100, 30, 20, 10, 5, 5, 5, 5, 5), "w2cal": (10,) * 10}
ALONE = {2: 15, 4: 4, 6: 2, 8: 1}           # full-DEM Relay-BP alone: the first ALONE[pi] w2test chunks at PS[pi]
QS = (0.0025, 0.02)
TAGS = ("r11", "r12")
NET_SHA = {"r11": "58e60897c3f9302c2014e82257d0d08f80a6e8386d2dd5216a093230990b252f",
           "r12": "f2821c327b4e233a4dd1a75c465ceb261448d981c2b55166862fc34a99ec3e96"}
DERIVED = {0.0055: 0.005}                   # p -> the relay file whose noise arguments are rewritten
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "M",
         "MX", "MY", "MR", "MRX", "MRY", "MPP"}
SLICE = 20
W2 = os.environ.get("W2_DIR", "<cluster-scratch>/cascade_w2")
SHOTS, PREDS, OUT = (os.path.join(W2, d) for d in ("shots", "preds", "out"))
ROWS = os.environ.get("W2_ROWS", os.path.join(OUT, "rows_shots.csv"))
CKPT = os.environ.get("BB_CKPT", "<cluster-scratch>/bb/ckpt")
FIELDS = ["file", "sha256", "shots", "seed", "code", "p", "split", "chunk", "host", "cpu", "stim", "circuit_sha256"]

# Fig. 2(a) of the paper, digitised by the study lead from the SVG: P_L per logical qubit per round; p in percent
PAPER = {"Cascade": {0.10: 8.17e-7, 0.20: 1.04e-5, 0.30: 6.58e-5, 0.40: 2.19e-4},
         "Tesseract": {0.10: 5.74e-7, 0.15: 2.90e-6, 0.20: 8.13e-6, 0.25: 2.18e-5, 0.30: 4.73e-5, 0.35: 9.44e-5,
                       0.40: 1.64e-4, 0.45: 2.95e-4, 0.50: 5.59e-4, 0.55: 9.63e-4},
         "Relay": {0.10: 1.41e-6, 0.15: 6.46e-6, 0.20: 2.00e-5, 0.25: 4.39e-5, 0.30: 9.50e-5, 0.35: 1.86e-4,
                   0.40: 3.54e-4, 0.45: 5.59e-4, 0.50: 8.75e-4, 0.55: 1.23e-3},
         "BP+OSD": {0.10: 3.22e-5, 0.15: 1.22e-4, 0.20: 3.14e-4, 0.25: 7.67e-4, 0.30: 1.30e-3}}
PAPER_APPROX = {"Cascade": {0.50: 8e-4, 0.55: 1.2e-3}}     # "overlapped" in the figure: read approximately
# the registered stage-3 results on BB72 (data/bb/stage3/results.json; RESULTS_stage3.md), block failures / shots
REGISTERED = {0.20: {"r12 alone (BB72-2)": (254, 300_000), "r12 + full-DEM Relay-BP, q 0.25 % (BB72-2F)": (171, 300_000),
                     "full-DEM Relay-BP": (274, 300_000), "sector Relay-BP": (812, 300_000),
                     "ldpc BP-OSD (min-sum 80, OSD-CS 60)": (903, 300_000)},
              0.30: {"r11 alone (BB72-3)": (1341, 200_000), "r12 + full-DEM Relay-BP, q 2 % (BB72-3F)": (877, 200_000),
                     "full-DEM Relay-BP": (1092, 200_000), "sector Relay-BP": (2704, 200_000),
                     "ldpc BP-OSD (min-sum 80, OSD-CS 60)": (3299, 200_000)}}


def _pstr(p):
    return f"{p:.6f}".rstrip("0")


def seed_of(split, pi, j):
    if not (0 <= j < 10 ** 6 and 0 <= pi < len(PS)):
        raise ValueError((pi, j))
    return 2 ** 50 + 10 ** 9 * SPLITS[split] + 10 ** 6 * pi + j


def derived_text(src, dst):
    """(text of the relay circuit at src with every '(src)' argument written as '(dst)', substitutions made)."""
    path = circuits.relay_path(CODE, "X", src)
    blob = open(path, "rb").read()
    if hashlib.sha256(blob).hexdigest() != circuits._manifest()[os.path.basename(path)]:
        raise ValueError(f"sha256 mismatch for {path}")
    a, b = f"({_pstr(src)})", f"({_pstr(dst)})"
    t = blob.decode()
    return t.replace(a, b), t.count(a)


@lru_cache(None)
def circuit(p):
    import stim
    p = round(p, 6)
    if p in DERIVED:
        return stim.Circuit(derived_text(DERIVED[p], p)[0])
    return circuits.load(CODE, "X", p)


@lru_cache(None)
def circuit_sha(p):
    p = round(p, 6)
    if p in DERIVED:
        return hashlib.sha256(derived_text(DERIVED[p], p)[0].encode()).hexdigest()
    return circuits._manifest()[circuits.relay_name(CODE, "X", p)]


def _patch_circuits():
    """qnd.circuits.load (hence circuits.problem, the decoders' DEM) returns the derived circuit at a DERIVED p."""
    orig = circuits.load
    if getattr(orig, "w2", False):
        return

    def load(code, basis, p, check=True):
        if (code, basis) == (CODE, "X") and round(p, 6) in DERIVED:
            return circuit(p)
        return orig(code, basis, p, check)
    load.w2 = True
    circuits.load = load


def check(a):
    out, ok = {}, True
    for src, dst in ((0.001, 0.0015), (0.002, 0.0025), (0.003, 0.0035), (0.004, 0.0045), (0.005, 0.0055)):
        t, n = derived_text(src, dst)
        r = {"substitutions": n, "sha256": hashlib.sha256(t.encode()).hexdigest()}
        if dst in DERIVED:                      # only the noise arguments changed: round trip, counts, argument set
            import stim
            c, c0 = circuit(dst), circuits.load(CODE, "X", src)
            src_text = open(circuits.relay_path(CODE, "X", src)).read()
            noise = [i for i in c.flattened() if i.name in NOISE]
            noise0 = [i for i in c0.flattened() if i.name in NOISE]
            r.update(noise_instructions=len(noise), with_argument=sum(1 for i in noise if i.gate_args_copy()),
                     noise_args=sorted({x for i in noise for x in i.gate_args_copy()}),
                     source_noise_args=sorted({x for i in noise0 for x in i.gate_args_copy()}),
                     detectors=c.num_detectors, observables=c.num_observables,
                     round_trip=t.replace(f"({_pstr(dst)})", f"({_pstr(src)})") == src_text and
                     f"({_pstr(dst)})" not in src_text, stim=stim.__version__)
            good = (r["noise_args"] == [dst] and r["source_noise_args"] == [src] and len(noise) == len(noise0)
                    and r["with_argument"] == n
                    and r["round_trip"] and (c.num_detectors, c.num_observables) == (c0.num_detectors,
                                                                                     c0.num_observables))
        else:
            r["relay_file_sha256"] = circuits._manifest()[circuits.relay_name(CODE, "X", dst)]
            good = r["sha256"] == r["relay_file_sha256"]
        r["ok"] = bool(good)
        ok &= good
        out[f"{_pstr(src)}->{_pstr(dst)}"] = r
        print(f"{_pstr(src)} -> {_pstr(dst)}: {r}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "circuit_check.json"), "w"), indent=1)
    if not ok:
        raise SystemExit("circuit check FAILED")
    print("circuit check ok", flush=True)


def parse(s):
    """'0-1,4' -> [0, 1, 4]"""
    out = []
    for part in str(s).split(","):
        x, _, y = part.partition("-")
        out += list(range(int(x), int(y) + 1)) if y else [int(x)]
    return out


def name(split, pi, j):
    return f"{CODE}_p{PS[pi]}_{split}{j:04d}.npz"


def pred_path(split, pi, j, dec):
    return os.path.join(PREDS, name(split, pi, j)[:-4] + f"_{dec}.npz")


def chunks(split, pis=range(len(PS))):
    return [(pi, j) for pi in pis for j in range(PLAN[split][pi])]


def _sample(task):
    import stim
    split, pi, j = task
    f = os.path.join(SHOTS, name(split, pi, j))
    if os.path.exists(f):
        raise FileExistsError(f)
    det, obs = circuit(PS[pi]).compile_detector_sampler(seed=seed_of(split, pi, j)).sample(
        CHUNK, separate_observables=True)
    det, obs = det.astype(np.uint8), obs.astype(np.uint8)
    np.savez(f, det=np.packbits(det, axis=1), obs=np.packbits(obs, axis=1), n=len(det), D=det.shape[1], k=obs.shape[1])
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    return {"file": os.path.basename(f), "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(), "shots": len(det),
            "seed": seed_of(split, pi, j), "code": CODE, "p": PS[pi], "split": split, "chunk": j,
            "host": platform.node(), "cpu": cpu, "stim": stim.__version__, "circuit_sha256": circuit_sha(PS[pi]),
            "fired": float(det.mean() * det.shape[1]), "anyobs": float(obs.any(1).mean())}


def sample(a):
    os.makedirs(SHOTS, exist_ok=True)
    new = not os.path.exists(a.rows)
    have = set() if new else {r["file"] for r in csv.DictReader(open(a.rows))}
    todo = [(a.split, pi, j) for pi, j in chunks(a.split) if name(a.split, pi, j) not in have]   # extend, never redo
    with open(a.rows, "a", newline="") as fh, get_context("fork").Pool(a.procs) as pool:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for row in pool.imap(_sample, todo):
            w.writerow(row)
            fh.flush()
            print(row["file"], row["sha256"][:12], f"fired/shot {row['fired']:.1f} any-obs {row['anyobs']:.4f}",
                  flush=True)


@lru_cache(None)
def _manifest():
    return {r["file"]: r for r in csv.DictReader(open(ROWS))}


def load(split, pi, j):
    """Stored chunk j at PS[pi], sha256-checked against W2_ROWS: (det uint8 (n, D), obs uint8 (n, k))."""
    f = os.path.join(SHOTS, name(split, pi, j))
    r = _manifest()[os.path.basename(f)]
    blob = open(f, "rb").read()
    if hashlib.sha256(blob).hexdigest() != r["sha256"] or int(r["seed"]) != seed_of(split, pi, j):
        raise ValueError(f"{f} does not match its manifest row")
    z = np.load(io.BytesIO(blob))
    n = int(z["n"])
    return (np.unpackbits(z["det"], axis=1, count=int(z["D"]))[:n],
            np.unpackbits(z["obs"], axis=1, count=int(z["k"]))[:n])


def infer(a):
    import torch
    from bb import engine
    dev = torch.device("cuda")
    os.makedirs(PREDS, exist_ok=True)
    for tag in TAGS:
        f = os.path.join(CKPT, tag, "final_ema.pt")
        sha = engine.sha256_file(f)
        if sha != NET_SHA[tag]:
            raise SystemExit(f"{f} is not the registered {tag}")
        model, _ = engine.load_net(f, dev)
        eng = engine.Deployed(model, "fp32", sha)
        sig = json.dumps(eng.signature(), sort_keys=True)
        print(tag, sig, flush=True)
        t0, cnt = time.time(), {}
        for pi, j in chunks(a.split):
            out = pred_path(a.split, pi, j, f"net-{tag}")
            if not os.path.exists(out):
                det, obs = load(a.split, pi, j)
                bits, margin = eng.run(det)
                fail = (bits != obs).any(1)
                np.savez(out, pred=bits, margin=margin, fail=fail, sig=sig)
            c = cnt.setdefault(pi, [0, 0])
            c[0] += int(np.load(out)["fail"].sum())
            c[1] += CHUNK
        for pi, (fails, n) in cnt.items():
            print(f"net-{tag} {a.split} p {PS[pi]}: {fails} / {n} failures", flush=True)
        print(f"net-{tag} {a.split}: {time.time() - t0:.0f} s", flush=True)


def _net(split, pi, tag, js=None):
    """(fail (N,), margin (N,), pred (N, K)) of a network read over the split's chunks js at PS[pi]."""
    z = [np.load(pred_path(split, pi, j, f"net-{tag}")) for j in (range(PLAN[split][pi]) if js is None else js)]
    if len({str(x["sig"]) for x in z}) != 1:
        raise SystemExit(f"net-{tag} {split} p {PS[pi]}: chunks with different engine signatures")
    return tuple(np.concatenate([x[k] for x in z]) for k in ("fail", "margin", "pred"))


def taus(a):
    out = {}
    for pi, p in enumerate(PS):
        m = _net("w2cal", pi, "r12")[1]
        out[str(p)] = {"cal_shots": len(m), **{str(q): float(np.quantile(m, q)) for q in QS}}
        print(p, out[str(p)], flush=True)
    json.dump(out, open(os.path.join(OUT, "taus.json"), "w"), indent=1)


def _taus():
    return json.load(open(os.path.join(OUT, "taus.json")))


def decode_plan(pis=range(len(PS))):
    """{(pi, j): sorted shot indices of w2test chunk j at PS[pi] that full-DEM Relay-BP decodes}."""
    t, plan = _taus(), {}
    for pi, j in chunks("w2test", sorted(pis)):
        if j < ALONE.get(pi, 0):
            plan[(pi, j)] = np.arange(CHUNK)
        else:
            m = np.load(pred_path("w2test", pi, j, "net-r12"))["margin"]
            plan[(pi, j)] = np.flatnonzero(m < t[str(PS[pi])][str(max(QS))])
    return plan


_dec = {}


def _decoder(pi):
    if pi not in _dec:
        from qnd import teachers
        _patch_circuits()
        _dec[pi] = teachers.Decoder("relay", CODE, "X", PS[pi], "all")     # = bb.opp's relay_full
    return _dec[pi]


def _job(task):
    pi, j, idx = task
    pred, secs, _ = _decoder(pi).decode(load("w2test", pi, j)[0][idx])
    return pi, j, idx, pred, secs


def _relay_files(pi, j):
    return sorted(f for f in glob(pred_path("w2test", pi, j, "relay_full_*")) if not f.endswith(".tmp.npz"))


def relay(a):
    k, K = map(int, a.shard.split("of") if "of" in a.shard else a.shard.split("/"))
    tasks, need = [], {}
    pis = set(parse(a.pis))
    for (pi, j), idx in decode_plan(pis).items():
        have = [np.load(f)["idx"] for f in _relay_files(pi, j) if f"_relay_full_{a.run}_" not in f]
        miss = np.setdiff1d(idx, np.concatenate(have)) if have else idx
        for s in range(0, len(miss), SLICE):
            tasks.append((pi, j, miss[s:s + SLICE]))
    tasks.sort(key=lambda t: -t[0])                       # the slowest (highest p) first: a shorter tail
    tasks = [t for i, t in enumerate(tasks) if i % K == k]
    mine = lambda pi, j: pred_path("w2test", pi, j, f"relay_full_{a.run}_s{k}of{K}")   # noqa: E731
    tasks = [t for t in tasks if not os.path.exists(mine(t[0], t[1]))]
    for pi, j, idx in tasks:
        need[(pi, j)] = need.get((pi, j), 0) + len(idx)
    print(f"shard {k}/{K}: {len(tasks)} tasks, {sum(need.values())} shots, {len(need)} chunks", flush=True)
    for pi in range(len(PS)):
        n = sum(v for (q, _), v in need.items() if q == pi)
        if n:
            print(f"  p {PS[pi]}: {n} shots", flush=True)
    if a.dry:
        return
    got, t0, done = {u: [] for u in need}, time.time(), 0
    with get_context("fork").Pool(a.procs) as pool:
        for pi, j, idx, pred, secs in pool.imap_unordered(_job, tasks):
            got[(pi, j)].append((idx, pred, secs))
            done += len(idx)
            if sum(len(x[0]) for x in got[(pi, j)]) < need[(pi, j)]:
                continue
            parts = got.pop((pi, j))
            idx, pred, secs = (np.concatenate([x[i] for x in parts]) for i in range(3))
            o = np.argsort(idx)
            idx, pred, secs = idx[o], pred[o], secs[o]
            fail = (pred != load("w2test", pi, j)[1][idx]).any(1)
            tmp = mine(pi, j)[:-4] + ".tmp.npz"
            np.savez(tmp, idx=idx.astype(np.int32), pred=pred.astype(np.uint8), fail=fail, secs=secs.astype(np.float32),
                     host=platform.node(), procs=a.procs)
            os.replace(tmp, mine(pi, j))
            print(f"relay_full p {PS[pi]} chunk {j}: {int(fail.sum())} / {len(fail)} failures, mean "
                  f"{1e3 * secs.mean():.1f} ms per shot; {done} shots in {time.time() - t0:.0f} s", flush=True)
    print(f"shard {k}/{K} done: {done} shots in {time.time() - t0:.0f} s", flush=True)


def _pilot_job(task):
    pi, kind, j, idx = task
    _, secs, conv = _decoder(pi).decode(load("w2test", pi, j)[0][idx])
    return pi, kind, secs, conv


def pilot(a):
    t, tasks = _taus(), []
    rng = np.random.default_rng(20261008)
    for pi, p in enumerate(PS):
        left = a.n
        for j in range(PLAN["w2test"][pi]):
            idx = np.flatnonzero(np.load(pred_path("w2test", pi, j, "net-r12"))["margin"] < t[str(p)][str(max(QS))])
            idx = idx[:left]
            tasks += [(pi, "routed", j, idx[s:s + 5]) for s in range(0, len(idx), 5)]
            left -= len(idx)
            if left <= 0:
                break
        rnd = np.sort(rng.choice(CHUNK, a.n, replace=False))
        tasks += [(pi, "random", 0, rnd[s:s + 5]) for s in range(0, a.n, 5)]
    out, t0 = {}, time.time()
    with get_context("fork").Pool(a.procs) as pool:
        for pi, kind, secs, conv in pool.imap_unordered(_pilot_job, tasks):
            r = out.setdefault(f"{PS[pi]} {kind}", {"secs": [], "conv": 0})
            r["secs"] += [float(s) for s in secs]
            r["conv"] += int(conv.sum())
    for key in sorted(out):
        s = np.array(out[key]["secs"])
        out[key].update(n=len(s), mean_ms=1e3 * s.mean(), p50_ms=1e3 * np.median(s), max_ms=1e3 * s.max())
        print(key, {k: v for k, v in out[key].items() if k != "secs"}, flush=True)
    out["_meta"] = {"procs": a.procs, "host": platform.node(), "wall_s": time.time() - t0}
    json.dump(out, open(os.path.join(OUT, "pilot.json"), "w"), indent=1)


# ---- report -------------------------------------------------------------------------------------------------------
def p_l(pb):
    """Eq. (1): P_L = (1 - (2 (1 - P_block)^(1/k) - 1)^(1/R)) / 2, computed without cancellation."""
    if pb <= 0:
        return 0.0
    x = 2 * math.expm1(math.log1p(-pb) / K)                 # 2 (1 - P_block)^(1/k) - 2
    return -math.expm1(math.log1p(x) / R) / 2


def wilson(k, n, z=1.959963984540054):
    c, h = (k + z * z / 2) / (n + z * z), z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, c - h), min(1.0, c + h)


def point(k, n):
    lo, hi = wilson(k, n)
    return {"shots": n, "failures": int(k), "p_block": k / n, "p_block_95": [lo, hi], "P_L": p_l(k / n),
            "P_L_95": [p_l(lo), p_l(hi)]}


def _ratios(pt, pp):
    for v in pt.values():
        for ref in ("Cascade", "Tesseract", "Relay", "BP+OSD"):
            if pp in PAPER[ref]:
                v[f"ratio_to_{ref}"] = v["P_L"] / PAPER[ref][pp]
    return pt


def report(a):
    if a.replot:                                            # tables and plot again from the stored counts
        res = json.load(open(os.path.join(a.out, "curves_counts.json")))
        open(os.path.join(a.out, "curves_tables.md"), "w").write(tables(res))
        return plot(res, os.path.join(a.out, "curves.png"))
    t = _taus()
    res = {"PS_percent": [round(100 * p, 2) for p in PS], "eq1": {"k": K, "R": R}, "taus": t, "points": {},
           "relay_seconds": {}}
    for pi, p in enumerate(PS):
        pp = round(100 * p, 2)
        f11, _, _ = _net("w2test", pi, "r11")
        f12, m12, _ = _net("w2test", pi, "r12")
        N = len(f12)
        rel = np.zeros(N, bool)                             # Relay-BP's failure per shot where decoded
        seen = np.zeros(N, int)
        secs = np.full(N, np.nan)
        for j in range(PLAN["w2test"][pi]):
            for f in _relay_files(pi, j):
                z = np.load(f)
                i = j * CHUNK + z["idx"]
                rel[i], secs[i] = z["fail"], z["secs"]
                seen[i] += 1
        if seen.max(initial=0) > 1:
            raise SystemExit(f"p {p}: a shot decoded twice")
        pt = {"r11": point(f11.sum(), N), "r12": point(f12.sum(), N)}
        for q in QS:
            r = m12 < t[str(p)][str(q)]
            if not seen[r].all():
                raise SystemExit(f"p {p} q {q}: {int((~seen[r].astype(bool)).sum())} routed shots not decoded")
            hyb = np.where(r, rel, f12)
            pt[f"r12+relay q={q}"] = {**point(hyb.sum(), N), "tau": t[str(p)][str(q)], "routed": int(r.sum()),
                                      "routed_fraction": float(r.mean()), "r12_fail_in_routed": int(f12[r].sum()),
                                      "relay_fail_in_routed": int(rel[r].sum())}
        if pi in ALONE:
            n = ALONE[pi] * CHUNK
            if seen[:n].min() != 1:
                raise SystemExit(f"p {p}: Relay-BP alone incomplete")
            pt["relay_full"] = {**point(rel[:n].sum(), n), "r12_on_same_shots": int(f12[:n].sum()),
                                "both_fail": int((rel[:n] & f12[:n]).sum())}
            res["relay_seconds"][str(pp) + " random"] = {"n": n, "mean_ms": float(1e3 * np.nanmean(secs[:n]))}
        r2 = m12 < t[str(p)][str(max(QS))]
        res["relay_seconds"][str(pp) + " routed"] = {"n": int(r2.sum()), "mean_ms": float(1e3 * np.nanmean(secs[r2]))}
        res["relay_seconds"][str(pp) + " all decoded"] = {"n": int(seen.sum()), "sum_s": float(np.nansum(secs))}
        _ratios(pt, pp)
        res["points"][str(pp)] = pt
        print(pp, {k: (v["failures"], v["shots"], f"{v['P_L']:.3g}") for k, v in pt.items()}, flush=True)
    res["registered"] = {str(pp): _ratios({k: point(*c) for k, c in cs.items()}, pp) for pp, cs in REGISTERED.items()}
    res["paper"] = {k: {str(x): y for x, y in v.items()} for k, v in PAPER.items()}
    res["paper_approx"] = {k: {str(x): y for x, y in v.items()} for k, v in PAPER_APPROX.items()}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "curves_counts.json"), "w"), indent=1)
    open(os.path.join(a.out, "curves_tables.md"), "w").write(tables(res))
    plot(res, os.path.join(a.out, "curves.png"))


LABEL = {"r11": "r11 alone", "r12": "r12 alone", "r12+relay q=0.0025": "r12 + Relay-BP, q 0.25 %",
         "r12+relay q=0.02": "r12 + Relay-BP, q 2 %", "relay_full": "full-DEM Relay-BP alone"}


def _e(x):
    return f"{x:.2e}" if x else "0"


def _ratio(v, ref):
    return f"{v[f'ratio_to_{ref}']:.2f}" if f"ratio_to_{ref}" in v else "-"


def tables(res):
    """The two markdown tables of CASCADE_CURVES.md, straight from the counts (no hand transcription)."""
    out = ["| p (%) | Cascade | Tesseract | Relay (paper) | r11 alone | r12 alone | r12 + Relay-BP q 0.25 % | "
           "r12 + Relay-BP q 2 % | our full-DEM Relay-BP | what it means |", "|" + "---|" * 10]
    for pp, pt in res["points"].items():
        x = float(pp)
        paper = [_e(PAPER[r][x]) if x in PAPER[r] else (f"~{_e(PAPER_APPROX[r][x])}" if x in PAPER_APPROX.get(r, {})
                                                         else "-") for r in ("Cascade", "Tesseract", "Relay")]
        ours = [_e(pt[k]["P_L"]) if k in pt else "-" for k in LABEL]
        refs = [r for r in ("Cascade", "Tesseract") if x in PAPER[r]]
        words = "; ".join(f"{name} " + ", ".join(f"{pt[k][f'ratio_to_{r}']:.2f}x {r}" for r in refs)
                          for k, name in (("r12", "r12 alone"), ("r12+relay q=0.02", "accurate mode at 2 %")))
        out.append(f"| {x:g} | " + " | ".join(paper + ours) + f" | {words} |")
    out += ["", "| p (%) | decoder | shots | failures | P_block | P_block 95 % | P_L (Eq. 1) | P_L 95 % | "
                "routed (tau) | / Cascade | / Tesseract | / Relay (paper) |", "|" + "---|" * 12]
    for pp, pt in res["points"].items():
        for k, lab in LABEL.items():
            if k not in pt:
                continue
            v = pt[k]
            rt = f"{v['routed']:,} ({v['tau']:.4f})" if "routed" in v else "-"
            out.append(f"| {float(pp):g} | {lab} | {v['shots']:,} | {v['failures']:,} | {_e(v['p_block'])} | "
                       f"{_e(v['p_block_95'][0])} - {_e(v['p_block_95'][1])} | {_e(v['P_L'])} | "
                       f"{_e(v['P_L_95'][0])} - {_e(v['P_L_95'][1])} | {rt} | {_ratio(v, 'Cascade')} | "
                       f"{_ratio(v, 'Tesseract')} | {_ratio(v, 'Relay')} |")
    out += ["", "| p (%) | registered result | shots | failures | P_L (Eq. 1) | P_L 95 % | / Cascade | / Tesseract | "
                "/ Relay (paper) | / BP+OSD (paper) |", "|" + "---|" * 10]
    for pp, pt in res["registered"].items():
        for k, v in pt.items():
            out.append(f"| {float(pp):g} | {k} | {v['shots']:,} | {v['failures']:,} | {_e(v['P_L'])} | "
                       f"{_e(v['P_L_95'][0])} - {_e(v['P_L_95'][1])} | {_ratio(v, 'Cascade')} | {_ratio(v, 'Tesseract')} | "
                       f"{_ratio(v, 'Relay')} | {_ratio(v, 'BP+OSD')} |")
    return "\n".join(out) + "\n"


def plot(res, path):
    """Top: P_L against p, the paper's curves dashed, ours solid with 95 % intervals. Bottom: the same divided by the
    paper's Tesseract value at each p (BP+OSD left out there: 27-56x). One hue per decoder family in the dataviz
    skill's categorical order (Relay-BP: the paper's and ours share a hue, as do the accurate mode's two q)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    hue = {"Cascade": "#2a78d6", "r12": "#eb6834", "acc": "#1baf7a", "Tesseract": "#eda100", "Relay": "#e87ba4",
           "BP+OSD": "#008300", "r11": "#4a3aa7"}
    plt.rcParams.update({"text.color": ink, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.edgecolor": ink2})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.2, 9.6), dpi=150, facecolor=surf, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    ring = dict(mec=surf, mew=1.2, solid_capstyle="round", solid_joinstyle="round", dash_capstyle="round")
    tess = PAPER["Tesseract"]
    paper = (("Cascade", "o"), ("Tesseract", "s"), ("Relay", "^"), ("BP+OSD", "v"))
    ours = {"r11": (hue["r11"], "D", "network r11 alone (trained at 0.3 %)"),
            "r12": (hue["r12"], "D", "network r12 alone (trained at 0.3 %)"),
            "r12+relay q=0.0025": (hue["acc"], "^", "r12 + full-DEM Relay-BP on 0.25 % of shots"),
            "r12+relay q=0.02": (hue["acc"], "s", "r12 + full-DEM Relay-BP on 2 % of shots"),
            "relay_full": (hue["Relay"], "*", "full-DEM Relay-BP alone")}
    for axis, div in ((ax, None), (bx, tess)):
        f = (lambda x, y: y) if div is None else (lambda x, y: y / div[x])
        for ref, mk in paper:
            if div is not None and ref == "BP+OSD":
                continue
            xs = sorted(PAPER[ref])
            axis.plot(xs, [f(x, PAPER[ref][x]) for x in xs], color=hue[ref], marker=mk, ms=6, lw=1.5, ls=(0, (4, 2)),
                      **ring)
        xa = sorted(PAPER_APPROX["Cascade"])
        axis.plot(xa, [f(x, PAPER_APPROX["Cascade"][x]) for x in xa], color=hue["Cascade"], marker="o", ms=6,
                  mfc="none", mew=1.2, ls="none")
        for key, (col, mk, _) in ours.items():
            xs = [float(p) for p, pt in res["points"].items() if key in pt]
            ys = [f(x, res["points"][str(x)][key]["P_L"]) for x in xs]
            lo = [y - f(x, res["points"][str(x)][key]["P_L_95"][0]) for x, y in zip(xs, ys)]
            hi = [f(x, res["points"][str(x)][key]["P_L_95"][1]) - y for x, y in zip(xs, ys)]
            axis.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, ms=11 if mk == "*" else 7, lw=2,
                          elinewidth=1, capsize=2.5, **ring)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="major", color=grid, lw=0.6, ls="-")
        axis.set_facecolor(surf)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    ticks = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    bx.set_xticks(ticks)
    bx.set_xticklabels([f"{x:g}" for x in ticks], fontsize=8)
    bx.minorticks_off()
    ax.minorticks_off()
    bx.set_ylim(0.3, 4)
    bx.set_yticks([0.5, 1, 2, 3])
    bx.set_yticklabels(["0.5", "1", "2", "3"])
    bx.set_xlabel("physical error rate p (%)")
    ax.set_ylabel("P_L: logical error rate per logical qubit per round (Eq. 1)")
    bx.set_ylabel("P_L / the paper's Tesseract P_L at the same p")
    ax.set_title("BB72 [[72,12,6]], 6 rounds: our decoders (relay d185194 circuits, uniform circuit noise)\n"
                 "against Cascade's Fig. 2(a) (digitised; the paper does not specify its noise model or circuits)",
                 fontsize=9, color=ink)
    h1 = [Line2D([], [], color=hue[r], marker=m, ms=6, lw=1.5, ls=(0, (4, 2)), mec=surf, label=f"paper: {r}")
          for r, m in paper]
    h1.append(Line2D([], [], color=hue["Cascade"], marker="o", ms=6, mfc="none", ls="none",
                     label="paper: Cascade, read approx."))
    h2 = [Line2D([], [], color=c, marker=m, ms=10 if m == "*" else 6, lw=2, label=f"ours: {lab}")
          for c, m, lab in ours.values()]
    leg = ax.legend(handles=h1, fontsize=7.5, loc="upper left", frameon=False, title="dashed: paper",
                    title_fontsize=7.5, alignment="left")
    ax.add_artist(leg)
    ax.legend(handles=h2, fontsize=7.5, loc="lower right", frameon=False,
              title="solid: ours, bars = 95 % Wilson interval", title_fontsize=7.5, alignment="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surf)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("sample")
    s.add_argument("--split", required=True, choices=list(SPLITS))
    s.add_argument("--rows", required=True)
    s.add_argument("--procs", type=int, default=os.cpu_count())
    s = sub.add_parser("infer")
    s.add_argument("--split", required=True, choices=list(SPLITS))
    sub.add_parser("taus")
    s = sub.add_parser("pilot")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--procs", type=int, default=os.cpu_count())
    s = sub.add_parser("relay")
    s.add_argument("--run", required=True)
    s.add_argument("--shard", default="0/1")
    s.add_argument("--pis", default=f"0-{len(PS) - 1}", help="indices into PS, e.g. 2-9")
    s.add_argument("--procs", type=int, default=os.cpu_count())
    s.add_argument("--dry", action="store_true")
    s = sub.add_parser("report")
    s.add_argument("--out", required=True)
    s.add_argument("--replot", action="store_true", help="only the tables and the plot, from DIR/curves_counts.json")
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
