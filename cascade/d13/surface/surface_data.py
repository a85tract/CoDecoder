"""Shots for Cascade-surface (surface_train): circuits at any p (surface_layout.circuit_at), W5's seed namespace, a
multiprocess Stim stream for training, fresh evaluation shots and correlated PyMatching's decisions on them.

Seeds (W5's own namespace, 2^56 + ..., disjoint from every documented one: ising_compare and the release below 2^33,
interp at 2^40 + .., BB72 / BB144 at 2^41 + .. and 2^42 + .. (< 2^46), W1 at 2^50 + .., W3 and phase C in
[2^55 + 1e15, 2^55 + 6e15), registered test sets >= 2^62):
  train_seed(d, run, rank, chunk) = 2^56 + 1e15 d + 1e12 run + 1e10 rank + chunk          (run < 1000, rank < 100)
  eval_seed(d, kind, p, chunk)    = 2^56 + 3e16 + 1e15 d + 1e13 kind + 1e7 round(1e6 p) + chunk
                                                                           (kind < 100, p < 1 %, chunk < 1e7)
so for each d the training streams lie in [2^56 + 1e15 d, + 1e15) and the evaluation shots in
[2^56 + 3e16 + 1e15 d, + 1e15); for d <= 29 everything lies in [2^56 + 3e15, 2^56 + 6e16), below 2^57.
  kind 1 = the training monitor, 2 = smoke and unit checks, 3 = sampler checks and the host fingerprint,
  5 = the pilot's monitor, 6 = the choice of the lower test p, 10 = the development block (thresholds),
  11 = the confirmation block (read once), 12 = screens, checks and the speed pipelines' streams, 13 = dry runs.
  Run ids: 1 = Cascade-surface d = 13, 2 = d = 17; 900-999 = pilots and benchmarks (the plan: 900 the d = 13 pilot,
  901 the benchmarks, 902 the DDP check)."""
import collections
import multiprocessing as mp
import os
import time

import numpy as np

from surface import surface_layout as sl

SEED_BASE = 2 ** 56
BASE_P = sl.P0
MAX_EVAL = 1_000_000                                     # eval_shots' limit: larger blocks are streamed


def train_seed(d, run, rank, chunk):
    assert 3 <= d <= 29 and 0 <= run < 1000 and 0 <= rank < 100 and 0 <= chunk < 10 ** 10
    return SEED_BASE + d * 10 ** 15 + run * 10 ** 12 + rank * 10 ** 10 + chunk


def eval_seed(d, kind, p, chunk):
    assert 3 <= d <= 29 and 0 <= kind < 100 and 0 <= chunk < 10 ** 7 and 0 < p < 0.01
    return SEED_BASE + 3 * 10 ** 16 + d * 10 ** 15 + kind * 10 ** 13 + int(round(p * 1e6)) * 10 ** 7 + chunk


def sample(d, p, n, seed):
    """n shots of the d circuit at p from one seed -> (det (n, D) uint8, obs (n, 1) uint8)."""
    det, obs = sl.circuit_at(d, p).compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def _eval_chunk(args):
    d, kind, p, j, n = args
    return sample(d, p, n, eval_seed(d, kind, p, j))


def eval_shots(d, p, chunks, n=10000, kind=0, procs=1):
    """Fresh shots: chunks (an iterable of chunk numbers) of n shots each at p -> (det, obs) concatenated in order.
    Small blocks only (at most MAX_EVAL shots: the training monitor, checks): the development and confirmation reads
    stream their blocks (stream_shots), since 4e7 shots at d = 13 would take about 175 GB while concatenating."""
    jobs = [(d, kind, p, j, n) for j in chunks]
    if len(jobs) * n > MAX_EVAL:
        raise ValueError(f"eval_shots: {len(jobs) * n} shots exceed {MAX_EVAL}; stream them with stream_shots")
    if procs > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            out = pool.map(_eval_chunk, jobs)
    else:
        out = [_eval_chunk(x) for x in jobs]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def stream_shots(d, p, chunks, n=10000, kind=0, procs=1, ahead=2):
    """The shots of eval_shots chunk by chunk, in order: yields (chunk, det (n, D) uint8, obs (n, 1) uint8). With
    procs > 1 a spawned pool samples at most `ahead` chunks beyond the one yielded, so a block of any size holds a few
    chunks in memory (10,000 shots at d = 13: 22 MB of detector bits)."""
    chunks = list(chunks)
    if procs <= 1:
        for j in chunks:
            yield (j, *_eval_chunk((d, kind, p, j, n)))
        return
    with mp.get_context("spawn").Pool(procs) as pool:
        pend = collections.deque()
        for j in chunks:
            pend.append((j, pool.apply_async(_eval_chunk, ((d, kind, p, j, n),))))
            if len(pend) > ahead:
                j0, r = pend.popleft()
                yield (j0, *r.get())
        while pend:
            j0, r = pend.popleft()
            yield (j0, *r.get())


_PM = {}


def _pm_chunk(args):
    d, p, det, correlated = args
    sl.release_root()
    from convlines import circuits
    key = (d, p, correlated)
    if key not in _PM:
        _PM[key] = circuits.matcher(d, correlated, sl.circuit_at(d, p))
    return circuits.pm_decode(_PM[key], det, correlated)


def pm_predict(d, p, det, correlated=True, procs=1, chunk=2000):
    """PyMatching's predicted observable (n,) uint8 on det, through the release's matcher (DEM decomposed, approximate
    disjoint errors; correlated = enable_correlations in the matcher and in decode_batch)."""
    jobs = [(d, p, det[i:i + chunk], correlated) for i in range(0, len(det), chunk)]
    if procs > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            out = pool.map(_pm_chunk, jobs)
    else:
        out = [_pm_chunk(x) for x in jobs]
    return np.concatenate(out) if out else np.zeros(0, np.uint8)


def _stream_worker(d, run, rank, sched, chunk, c0, step, q):
    """Chunks c0, c0 + step, ...: `chunk` shots each at p = sched(c) (bit-packed, Stim's little-endian order)."""
    from surface.cascade_optim import p_schedule
    c = c0
    while True:
        t = time.time()
        p = p_schedule(c, *sched) if sched[0] else sched[2]
        det, obs = sl.circuit_at(d, p).compile_detector_sampler(seed=train_seed(d, run, rank, c)).sample(
            chunk, separate_observables=True, bit_packed=True)
        q.put((c, p, det, obs, time.time() - t))
        c += step


class StimStream:
    """Training shots from `workers` spawned processes; chunk c (one optimizer step's shots on this rank) is sampled
    at p_schedule(c, *sched) from train_seed(d, run, rank, c), so the stream does not depend on the number of workers
    and a resumed run continues it exactly. sched = (steps, p1, p2, f1, f2); steps 0 = p2 throughout."""

    def __init__(self, d, run, rank, sched, chunk, workers, start=0, depth=4):
        ctx = mp.get_context("spawn")
        self.qs = [ctx.Queue(maxsize=depth) for _ in range(workers)]
        self.ps = [ctx.Process(target=_stream_worker, args=(d, run, rank, sched, chunk, start + w, workers, q),
                               daemon=True) for w, q in enumerate(self.qs)]
        for pr in self.ps:
            pr.start()
        self.start = self.c = start
        self.t_sample = []

    def next(self):
        c, p, d, o, t = self.qs[(self.c - self.start) % len(self.qs)].get()
        assert c == self.c, (c, self.c)
        self.c += 1
        self.t_sample.append(t)
        return p, d, o

    def capacity(self, chunk):
        """Steady shots/s of all workers together (each worker's first chunk, which compiles its circuit, left out)."""
        t = self.t_sample[len(self.qs):]
        return len(self.qs) * chunk / (sum(t) / len(t)) if t else None

    def close(self):
        for pr in self.ps:
            pr.terminate()


def env_record(d):
    """Versions and a fingerprint of Stim's output on this host (1,024 shots at p = 0.006 from a fixed seed)."""
    import hashlib
    import platform
    import stim
    det, obs = sl.circuit_at(d, BASE_P).compile_detector_sampler(seed=eval_seed(d, 3, BASE_P, 0)).sample(
        1024, separate_observables=True, bit_packed=True)
    return {"stim": stim.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "host": platform.node(), "cpu": _cpu_model(),
            "stim_fingerprint": hashlib.sha256(det.tobytes() + obs.tobytes()).hexdigest()[:16]}


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _bench_worker(args):
    d, p, chunk, secs, w = args
    t0 = time.time()
    c = sl.circuit_at(d, p)
    t_load = time.time() - t0
    n, t_comp, t_samp, j = 0, 0.0, 0.0, 0
    while t_comp + t_samp < secs:
        t = time.time()
        s = c.compile_detector_sampler(seed=eval_seed(d, 3, p, 1000 * w + j))
        t_comp += time.time() - t
        t = time.time()
        s.sample(chunk, separate_observables=True, bit_packed=True)
        t_samp += time.time() - t
        n += chunk
        j += 1
    return n, t_comp, t_samp, t_load


def bench(d=13, p=0.007, chunk=3328, workers=(1,), secs=20.0):
    """Stim's throughput on this host: per worker count, shots/s with a fresh compile per chunk (as StimStream does)
    and the compile share. -> list of dicts."""
    out = []
    for W in workers:
        with mp.get_context("spawn").Pool(W) as pool:
            t0 = time.time()
            res = pool.map(_bench_worker, [(d, p, chunk, secs, w) for w in range(W)])
            wall = time.time() - t0
        n = sum(r[0] for r in res)
        comp = sum(r[1] for r in res)
        samp = sum(r[2] for r in res)
        out.append({"d": d, "workers": W, "chunk": chunk, "p": p, "shots": n,
                    "shots_per_s": round(n / ((comp + samp) / W), 1), "per_worker": round(n / (comp + samp), 1),
                    "compile_share": round(comp / (comp + samp), 3), "load_s": round(max(r[3] for r in res), 2),
                    "wall": round(wall, 1)})
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "check", "env", "layout"])
    ap.add_argument("--d", type=int, default=13)
    ap.add_argument("--p", type=float, default=0.007)
    ap.add_argument("--chunks", default="3328")
    ap.add_argument("--workers", default="1,4")
    ap.add_argument("--secs", type=float, default=20.0)
    a = ap.parse_args()
    if a.cmd == "bench":
        for ch in (int(x) for x in a.chunks.split(",")):
            for r in bench(a.d, a.p, ch, [int(x) for x in a.workers.split(",")], a.secs):
                print("SAMPLER " + json.dumps(r), flush=True)
    elif a.cmd == "check":
        print(json.dumps({"d": a.d, "circuit_at_reproduces_pinned_sha256": sl.check_circuit_at(a.d)}))
    elif a.cmd == "layout":
        print(json.dumps(sl.Layout(a.d).describe()))
    else:
        print(json.dumps(env_record(a.d)))
