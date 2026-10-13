"""Shots for Cascade (bb.cascade_train, bb.cascade_eval): circuits at any p, the study's seed namespace, a multiprocess Stim
stream for training and fresh evaluation shots.

Circuits. Relay-BP's BB72 X-memory files (qnd.circuits, revision d185194, sha256-pinned) exist on a p grid without 0.55 %,
Cascade's training point. Their noise is uniform: every noise channel and every noisy measurement carries the single
argument p. circuit_at(code, p) rewrites the arguments of the 0.003 file to p after checking that each equals 0.003;
check_circuit_at() confirms that this reproduces every pinned file of the code exactly (stim.Circuit equality).

Seeds (the study's own namespace, 2^55 + ..., disjoint from every documented namespace of ising_compare and qldpc_neural:
those lie below 2^33, at 2^40 + .., 2^41 + .., 2^42 + .. (< 2^43) and >= 2^62 for test sets):
  train_seed(run, rank, chunk) = 2^55 + 1e15 + 1e12 run + 1e10 rank + chunk      (run < 1000, rank < 100)
  eval_seed(kind, p, chunk)    = 2^55 + 2e15 + 1e13 kind + 1e7 round(1e6 p) + chunk   (kind < 100, chunk < 1e7)
  kind 0 = evaluation sets, 1 = the training monitor, 2 = smoke and unit checks, 3 = sampler checks,
  4 = BN recalibration shots (bb.cascade_eval --bn-recal), 5 = the stage-2 fidelity pilot's evaluation,
  6 = the stage-2 adapter and S55 checks.
  Stage-2 runs: train run ids 101 C14, 102 C6, 110 S55, 120 the pilot (both arms), 121 the C14 ramp checks,
  122-124 the resume checks, 190 the S55 smoke; the fixed monitoring block of C14, C6 and S55 is kind 1, chunks
  100-119 at 0.3 % and 100-109 at 0.55 %.

Circuit families. Every function takes family = "uniform" (the default: the
circuits and seeds above, unchanged) or "depspam_fm", the noise-model fit's variant (bb/cascade/noise.py from ising-cascade-cal
fb722bd4: the same gates, detectors and observables, DEPOLARIZE1(p) after every reset and before every measurement on
top of each flip p, and a noisy final data measurement), built by noise.build on circuit_at(code, p), so at any p.
Each family has its own seeds, FAMILY_SEED[family] above the uniform ones:
  depspam_fm  train_seed = 2^55 + 4e15 + ...  in [2^55 + 4e15, 2^55 + 5e15)
              eval_seed  = 2^55 + 5e15 + ...  in [2^55 + 5e15, 2^55 + 6e15)
disjoint from the uniform family's [2^55 + 1e15, 2^55 + 3e15) and from every other namespace. Kinds and run ids keep
their meaning; the fitted family's runs: 201 C14' (run C14_dfm), 210 S55' (S55_dfm); eval kinds 0 the frozen networks' rows, 1 the
monitoring block of C14' and S55' (chunks 100-119 at 0.3 %, 100-109 at 0.55 %), 3 the family's sampler and BP+OSD
checks, 6 its label check, 7 the routing thresholds' calibration blocks."""
import multiprocessing as mp
import os
import time
from functools import lru_cache

import numpy as np

SEED_BASE = 2 ** 55
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E",
         "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"}
MEAS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP", "MXX", "MYY", "MZZ"}
BASE_P = 0.003
FAMILY_SEED = {"uniform": 0, "depspam_fm": 3 * 10 ** 15}      # circuit family -> seed offset (module docstring)
FAMILIES = tuple(FAMILY_SEED)


def train_seed(run, rank, chunk, family="uniform"):
    assert 0 <= run < 1000 and 0 <= rank < 100 and 0 <= chunk < 10 ** 10
    return SEED_BASE + FAMILY_SEED[family] + 10 ** 15 + run * 10 ** 12 + rank * 10 ** 10 + chunk


def eval_seed(kind, p, chunk, family="uniform"):
    assert 0 <= kind < 100 and 0 <= chunk < 10 ** 7 and 0 < p < 1
    return SEED_BASE + FAMILY_SEED[family] + 2 * 10 ** 15 + kind * 10 ** 13 + int(round(p * 1e6)) * 10 ** 7 + chunk


def _rewrite(circ, p0, p):
    import stim
    out = stim.Circuit()
    for inst in circ:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, _rewrite(inst.body_copy(), p0, p), tag=inst.tag))
            continue
        args = inst.gate_args_copy()
        if inst.name in NOISE or (inst.name in MEAS and args):
            if len(args) != 1 or args[0] != p0:
                raise ValueError(f"not uniform noise at p = {p0}: {inst}")
            args = [p]
        out.append(stim.CircuitInstruction(inst.name, inst.targets_copy(), args, tag=inst.tag))
    return out


@lru_cache(None)
def circuit_at(code, p, family="uniform"):
    """The code's X-memory circuit at noise p: the pinned 0.003 file with every noise argument set to p (uniform), or
    the noise-model fit's noise.build(family, p) on that circuit (depspam_fm)."""
    if family != "uniform":
        if family not in FAMILY_SEED or code != "bb72":
            raise ValueError(f"no circuit family {family} for {code}")
        from bb.cascade import noise                     # the noise-model fit's noise models (bb/cascade/noise.py)
        return noise.build(family, float(p), src=circuit_at(code, p))
    from qnd import circuits
    return _rewrite(circuits.load(code, "X", BASE_P), BASE_P, float(p))


def check_circuit_at(code="bb72"):
    """circuit_at(p) == the pinned file for every p of the code in the circuit manifest. -> the p checked."""
    from qnd import circuits
    n, k, d, r, _ = circuits.CODES[code]
    ok = []
    for name in circuits._manifest():
        if name.startswith(f"circuit=bicycle_bivariate_{n}_{k}_{d}_memory_X,"):
            p = float(name.split("error_rate=")[1].split(",")[0])
            if circuit_at(code, p) != circuits.load(code, "X", p):
                raise ValueError(f"circuit_at({code}, {p}) differs from the pinned file")
            ok.append(p)
    return sorted(ok)


def sample(code, p, n, seed, family="uniform"):
    """n shots of the circuit at p from one seed -> (det (n, D) uint8, obs (n, k) uint8)."""
    det, obs = circuit_at(code, p, family).compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def _eval_chunk(args):
    code, kind, p, j, n = args[:5]
    family = args[5] if len(args) > 5 else "uniform"
    return sample(code, p, n, eval_seed(kind, p, j, family), family)


def eval_shots(code, p, chunks, n=10000, kind=0, procs=1, family="uniform"):
    """Fresh shots: chunks (an iterable of chunk numbers) of n shots each at p -> (det, obs) concatenated in order."""
    jobs = [(code, kind, p, j, n, family) for j in chunks]
    if procs > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            out = pool.map(_eval_chunk, jobs)
    else:
        out = [_eval_chunk(x) for x in jobs]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def _stream_worker(code, run, rank, sched, chunk, c0, step, q, family="uniform"):
    """Chunks c0, c0 + step, ...: `chunk` shots each at p = sched(c) (bit-packed, Stim's little-endian order)."""
    from bb.cascade_optim import p_schedule
    c = c0
    while True:
        t = time.time()
        p = p_schedule(c, *sched) if sched[0] else sched[2]
        det, obs = circuit_at(code, p, family).compile_detector_sampler(seed=train_seed(run, rank, c, family)).sample(
            chunk, separate_observables=True, bit_packed=True)
        q.put((c, p, det, obs, time.time() - t))
        c += step


class StimStream:
    """Training shots from `workers` spawned processes; chunk c (one optimizer step's shots on this rank) is sampled
    at p_schedule(c, *sched) from train_seed(run, rank, c), so the stream does not depend on the number of workers and
    a resumed run continues it exactly. sched = (steps, p1, p2, f1, f2); steps 0 = p2 throughout."""

    def __init__(self, code, run, rank, sched, chunk, workers, start=0, depth=4, family="uniform"):
        ctx = mp.get_context("spawn")
        self.qs = [ctx.Queue(maxsize=depth) for _ in range(workers)]
        self.ps = [ctx.Process(target=_stream_worker, args=(code, run, rank, sched, chunk, start + w, workers, q,
                                                            family), daemon=True) for w, q in enumerate(self.qs)]
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


@lru_cache(None)
def dem_dense(code, p, family="uniform"):
    """The circuit's undecomposed detector error model at p, identical mechanisms merged (qnd.circuits.dem_matrices):
    -> (q (F,) float64, M (F, D + k) uint8: the detectors then observables each mechanism flips)."""
    from qnd import circuits
    dem = circuit_at(code, p, family).detector_error_model(decompose_errors=False)
    H, L, q = circuits.dem_matrices(dem)
    M = np.concatenate([H.T.toarray(), L.T.toarray()], 1).astype(np.uint8)
    return q, M


class DemSampler:
    """GPU sampler of the same distribution: independent Bernoulli(q_f) mechanisms of Stim's detector error model,
    XORed into detectors and observables (one fp16 matrix product, then mod 2; exact for the uniform depolarizing,
    X/Z-flip and measurement-flip noise of these circuits, which Stim converts to independent mechanisms without
    approximation). Seeded per chunk from train_seed like StimStream; used only where the CPU samplers cannot keep up
    (the re-implementation's notes, section 4; not part of this release)."""

    def __init__(self, code, run, rank, sched, chunk, dev, start=0, family="uniform"):
        import torch
        self.torch, self.code, self.run, self.rank, self.sched, self.chunk, self.dev = torch, code, run, rank, sched, chunk, dev
        self.family = family
        self.c = start
        self.cache = {}
        self.gen = torch.Generator(device=dev)

    def _dem(self, p):
        if p not in self.cache:
            q, M = dem_dense(self.code, p, self.family)
            t = self.torch
            self.cache[p] = (t.as_tensor(q, dtype=t.float32, device=self.dev),
                             t.as_tensor(M, dtype=t.float16, device=self.dev))
        return self.cache[p]

    def next_unpacked(self, k):
        """-> (p, det (chunk, D) uint8, obs (chunk, k) uint8) on the device."""
        from bb.cascade_optim import p_schedule
        t = self.torch
        c = self.c
        p = p_schedule(c, *self.sched) if self.sched[0] else self.sched[2]
        q, M = self._dem(p)
        self.gen.manual_seed(train_seed(self.run, self.rank, c, self.family))
        e = (t.rand(self.chunk, len(q), generator=self.gen, device=self.dev) < q).to(t.float16)
        x = (e @ M).to(t.int32) & 1
        self.c += 1
        x = x.to(t.uint8)
        return p, x[:, :-k], x[:, -k:]


def env_record(code="bb72"):
    """Versions and a fingerprint of Stim's output on this host (1,024 shots at p = 0.003 from a fixed seed)."""
    import hashlib
    import platform
    import stim
    det, obs = circuit_at(code, BASE_P).compile_detector_sampler(seed=eval_seed(3, BASE_P, 0)).sample(
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
    code, p, chunk, secs, w = args
    t0 = time.time()
    c = circuit_at(code, p)
    t_load = time.time() - t0
    n, t_comp, t_samp = 0, 0.0, 0.0
    j = 0
    while t_comp + t_samp < secs:
        t = time.time()
        s = c.compile_detector_sampler(seed=eval_seed(3, p, 1000 * w + j))
        t_comp += time.time() - t
        t = time.time()
        s.sample(chunk, separate_observables=True, bit_packed=True)
        t_samp += time.time() - t
        n += chunk
        j += 1
    return n, t_comp, t_samp, t_load


def bench(code="bb72", p=0.0055, chunk=3328, workers=(1,), secs=20.0):
    """Stim's throughput on this host: per worker count, shots/s with a fresh compile per chunk (as StimStream does)
    and the compile share. -> list of dicts."""
    out = []
    for W in workers:
        with mp.get_context("spawn").Pool(W) as pool:
            t0 = time.time()
            res = pool.map(_bench_worker, [(code, p, chunk, secs, w) for w in range(W)])
            wall = time.time() - t0
        n = sum(r[0] for r in res)
        comp = sum(r[1] for r in res)
        samp = sum(r[2] for r in res)
        out.append({"workers": W, "chunk": chunk, "p": p, "shots": n, "shots_per_s": round(n / ((comp + samp) / W), 1),
                    "per_worker": round(n / (comp + samp), 1), "compile_share": round(comp / (comp + samp), 3),
                    "load_s": round(max(r[3] for r in res), 2), "wall": round(wall, 1)})
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "check", "env"])
    ap.add_argument("--code", default="bb72")
    ap.add_argument("--p", type=float, default=0.0055)
    ap.add_argument("--chunks", default="3328")
    ap.add_argument("--workers", default="1,8")
    ap.add_argument("--secs", type=float, default=20.0)
    a = ap.parse_args()
    if a.cmd == "bench":
        for ch in (int(x) for x in a.chunks.split(",")):
            for r in bench(a.code, a.p, ch, [int(x) for x in a.workers.split(",")], a.secs):
                print("SAMPLER " + json.dumps(r), flush=True)
    elif a.cmd == "check":
        print("circuit_at reproduces the pinned files at p =", check_circuit_at(a.code))
    else:
        print(json.dumps(env_record(a.code)))
