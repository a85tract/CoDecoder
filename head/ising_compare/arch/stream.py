"""Stage G1 (arch/README.md): the ordered true-label stream with every logical line's flip.

Same scheme as icmp.train.Stream(ordered=True) (worker A's RRStream): V = 8 virtual Stim streams with seeds
10^8 + 10^6 run_id + v, worker w owns the streams w, w + workers, ... and cycles them in order, one chunk of 8,192 shots
from each in turn, one queue per worker read round-robin. The sampled circuit is icmp.lines.line_circuit's augmented
copy (observables 1 ... d-1 = the parallel lines added over the final data measurements; no quantum operation
changed), whose seeded detector sampler gives exactly the original circuit's detectors and observable 0
(tests/test_lines.py; tests/test_arch_stream.py checks it against icmp.train.Stream at d = 9). So an arm that reads
this stream and a control that reads icmp.train's ordered stream with the same run id see the same shots in the same
order. next(k) -> (packed detectors (k, ceil(m / 8)) uint8, packed line flips (k, ceil(d / 8)) uint8, teacher labels).

Stage G3 (arch/README.md section 9): teacher_upto = K makes the samplers also decode the shots of the first K global
chunks with correlated PM (icmp.pm, the opponent's matcher on the verdict circuit) and return those labels (k,) uint8 as
the third item (None where a chunk has none); skip_chunks = K resumes the stream after K global chunks, any K: global
chunk i is worker i mod W's (i div W)-th chunk (the queues are read round-robin from queue K mod W), and worker w's
l-th chunk comes from its owned stream l mod (8 / W); each sampler draws and drops its streams' consumed chunks and
continues its cycle where it stood, so the next shots equal an uninterrupted stream's (tests/test_stream_resume.py)."""
import math
import multiprocessing as mp
import sys

import numpy as np

V = 8


def sampler_rr_lines(circuit_text, seeds, n, q, w=0, workers=1, skip_chunks=0, teacher_upto=0, d=0):
    import stim
    c = stim.Circuit(circuit_text)
    smps = [c.compile_detector_sampler(seed=s) for s in seeds]
    k = len(smps)
    local = max(0, -(-(skip_chunks - w) // workers))      # this worker's chunks among the first skip_chunks global ones
    for j, smp in enumerate(smps):                        # owned stream j served this worker's chunks l = j mod k
        for _ in range(max(0, -(-(local - j) // k))):
            smp.sample(n, separate_observables=True)
    mc = None
    if teacher_upto:
        from icmp import circuits, pm
        mc = pm.matcher(d, True, circuits.circuit(d, noise=circuits.NOISE_VERDICT))
    while True:
        det, obs = smps[local % k].sample(n, separate_observables=True)
        lab = pm.decode(mc, det, True) if mc is not None and local * workers + w < teacher_upto else None
        q.put((np.packbits(det, axis=1), np.packbits(obs, axis=1), lab))
        local += 1


def ramp_p(frac, p, ramp):
    """Training noise at a fraction frac of the run: ramp = (p0, f) raises it linearly from p0 to p over the first f of
    the samples, snapped down to the 11 levels p0 + i (p - p0) / 10 (like bb.train.p_of's grid); None or frac >= f: p."""
    if not ramp or frac >= ramp[1]:
        return p
    p0, f = ramp
    i = int(math.floor(10 * (frac / f) + 1e-9))
    return round(p0 + i * (p - p0) / 10, 12)


def sampler_ramp(d, p, ramp, total, n, base, q, w, workers, skip_chunks):
    """Global chunks c = skip_chunks + w, + workers, ...: n shots of the line circuit at noise ramp_p(c n / total), from a
    detector sampler compiled with seed base + c (per-chunk seeds: the order is the same whatever the worker count, and
    a resume at any chunk continues it exactly)."""
    from icmp.lines import line_circuit
    circ = {}
    c = skip_chunks + w
    while True:
        pc = ramp_p(c * n / total, p, ramp)
        if pc not in circ:
            circ[pc] = line_circuit(d, p=pc)[1]
        det, obs = circ[pc].compile_detector_sampler(seed=base + c).sample(n, separate_observables=True)
        q.put((c, np.packbits(det, axis=1), np.packbits(obs, axis=1)))
        c += workers


class RampLineStream:
    """A1's noise-ramp stream (revision/REVISION_PLAN.md 2.1, a head that does not train): LineStream's shots and line
    flips, but global chunk c is sampled at ramp_p(c chunk / total, p, ramp) from its own seed 10^8 + 10^6 stream_id + c
    (a1.seeds.surface_ramp_stream; c < 10^6). next(k) -> (packed detectors, packed line flips, None)."""

    def __init__(self, d, stream_id, workers, total, p, ramp, chunk=8192, skip_chunks=0):
        ctx = mp.get_context("spawn")
        self.base = 10**8 + 10**6 * stream_id
        assert total / chunk < 10**6 - 1, "the chunk seeds would leave the stream id's block"
        self.seeds = {"base": self.base, "rule": "base + global chunk", "ramp": list(ramp), "p": p}
        self.qs = [ctx.Queue(maxsize=4) for _ in range(workers)]
        self.ps = [ctx.Process(target=sampler_ramp, args=(d, p, tuple(ramp), total, chunk, self.base, q, w, workers,
                                                          skip_chunks), daemon=True) for w, q in enumerate(self.qs)]
        main_mod = sys.modules["__main__"]
        saved = main_mod.__dict__.get("__spec__"), main_mod.__dict__.get("__file__")
        main_mod.__spec__ = None
        main_mod.__dict__.pop("__file__", None)
        try:
            for pr in self.ps:
                pr.start()
        finally:
            main_mod.__spec__ = saved[0]
            if saved[1] is not None:
                main_mod.__file__ = saved[1]
        self.c0 = self.c = skip_chunks
        self.bx, self.by, self.have = [], [], 0

    def next(self, k):
        while self.have < k:
            c, x, y = self.qs[(self.c - self.c0) % len(self.qs)].get()
            assert c == self.c, (c, self.c)
            self.c += 1
            self.bx.append(x)
            self.by.append(y)
            self.have += len(x)
        X, Y = np.concatenate(self.bx), np.concatenate(self.by)
        self.bx, self.by, self.have = [X[k:]], [Y[k:]], len(X) - k
        return X[:k], Y[:k], None

    def close(self):
        LineStream.close(self)


class LineStream:
    def __init__(self, d, run_id, workers, chunk=8192, skip_chunks=0, teacher_upto=0):
        from icmp.lines import line_circuit
        assert V % workers == 0, f"workers must divide {V}"
        ctx = mp.get_context("spawn")
        text = str(line_circuit(d)[1])
        self.seeds = [10**8 + 10**6 * run_id + v for v in range(V)]
        self.qs = [ctx.Queue(maxsize=4) for _ in range(workers)]
        self.ps = [ctx.Process(target=sampler_rr_lines, args=(text, self.seeds[w::workers], chunk, q, w, workers,
                                                              skip_chunks, teacher_upto, d), daemon=True)
                   for w, q in enumerate(self.qs)]
        main_mod = sys.modules["__main__"]
        saved = main_mod.__dict__.get("__spec__"), main_mod.__dict__.get("__file__")
        main_mod.__spec__ = None                         # spawn: no init_main_from_name / _from_path in the children
        main_mod.__dict__.pop("__file__", None)
        try:
            for p in self.ps:
                p.start()
        finally:
            main_mod.__spec__ = saved[0]
            if saved[1] is not None:
                main_mod.__file__ = saved[1]
        self.turn, self.bx, self.by, self.bt, self.have = skip_chunks % workers, [], [], [], 0

    def next(self, k):
        while self.have < k:
            x, y, t = self.qs[self.turn].get()
            self.turn = (self.turn + 1) % len(self.qs)
            self.bx.append(x)
            self.by.append(y)
            self.bt.append(t if t is not None else np.full(len(x), 255, np.uint8))
            self.have += len(x)
        X, Y, T = np.concatenate(self.bx), np.concatenate(self.by), np.concatenate(self.bt)
        self.bx, self.by, self.bt, self.have = [X[k:]], [Y[k:]], [T[k:]], len(X) - k
        T = T[:k]
        return X[:k], Y[:k], (None if (T == 255).any() else T)

    def close(self):
        for q in self.qs:
            q.cancel_join_thread()
        for p in self.ps:
            if p.is_alive():
                p.terminate()
        for p in self.ps:
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join(5)
