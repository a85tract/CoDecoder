"""Fresh training shots from Stim sampler processes, in pools, and the noise scaling of a curriculum.

Pool k (pool_shots shots) is drawn by Stim's detector sampler with seed pool_seed(seed, k), from the training circuit
with every noise probability multiplied by the curriculum's scale for that pool (scale_noise). The pools arrive in
order whatever the number of sampler processes, so a run's shots depend on its seed, its curriculum and the Stim build
only. This module imports no torch: the sampler processes are forked before the trainer touches a GPU."""
import collections
import multiprocessing as mp

import numpy as np
import stim

_STATE = {}


def pool_seed(seed, k):
    return int(np.random.SeedSequence([seed, k]).generate_state(1, np.uint64)[0])


def scale_noise(circuit, s):
    """The circuit with every noise probability multiplied by s (0 < s <= 1): the arguments of every noisy gate,
    measurement flip probabilities included (stim.GateData.is_noisy_gate)."""
    if s == 1:
        return circuit
    assert 0 < s <= 1, s
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, scale_noise(inst.body_copy(), s)))
        elif stim.gate_data(inst.name).is_noisy_gate and inst.gate_args_copy():
            out.append(stim.CircuitInstruction(inst.name, inst.targets_copy(), [s * v for v in inst.gate_args_copy()],
                                              tag=inst.tag))
        else:
            out.append(inst)
    return out


def _init(text):
    _STATE.clear()
    _STATE["circuit"] = stim.Circuit(text)


def _pool(job):
    k, scale, seed, n = job
    if _STATE.get("scale") != scale:
        _STATE["scale"], _STATE["scaled"] = scale, scale_noise(_STATE["circuit"], scale)
    det, obs = _STATE["scaled"].compile_detector_sampler(seed=seed).sample(n, separate_observables=True,
                                                                           bit_packed=True)
    return k, det, obs


class Pools:
    """Iterator over (k, det, obs) for k = start, start + 1, ...: det and obs bit-packed (Stim's little-endian bit
    order), scale(k) the noise scale of pool k. At most `ahead` pools are sampled in advance."""

    def __init__(self, circuit, seed, pool_shots, scale, start=0, workers=4, ahead=None):
        self.seed, self.n, self.scale, self.start = seed, pool_shots, scale, start
        self.ahead = ahead or 2 * max(1, workers)
        self.pool = mp.get_context("fork").Pool(workers, _init, (str(circuit),)) if workers > 0 else None
        if self.pool is None:
            _init(str(circuit))

    def __iter__(self):
        jobs = ((k, self.scale(k), pool_seed(self.seed, k), self.n) for k in range(self.start, 1 << 62))
        if self.pool is None:
            yield from map(_pool, jobs)
            return
        q = collections.deque(self.pool.apply_async(_pool, (next(jobs),)) for _ in range(self.ahead))
        while True:
            r = q.popleft().get()
            q.append(self.pool.apply_async(_pool, (next(jobs),)))
            yield r

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
