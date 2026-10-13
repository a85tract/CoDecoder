"""Stim sampler processes for the true-label stream (icmp.train --phase truth). The parent passes the circuit as Stim
text, so a spawned worker imports only numpy and stim (the NVIDIA package would pull in torch, ~0.5 GB per worker)."""
import numpy as np
import stim


def sampler(circuit_text, seed, n, q):
    smp = stim.Circuit(circuit_text).compile_detector_sampler(seed=seed)
    while True:
        det, obs = smp.sample(n, separate_observables=True)
        q.put((np.packbits(det, axis=1), obs[:, 0].astype(np.float32)))


def sampler_rr(circuit_text, seeds, n, q):
    """Ordered variant (icmp.train --ordered; the RRStream design, ising-aux): one sampler per virtual-stream
    seed, cycled in order, one chunk of n shots from each in turn, so the chunk sequence of a worker does not depend on
    timing."""
    c = stim.Circuit(circuit_text)
    smps = [c.compile_detector_sampler(seed=s) for s in seeds]
    while True:
        for smp in smps:
            det, obs = smp.sample(n, separate_observables=True)
            q.put((np.packbits(det, axis=1), obs[:, 0].astype(np.float32)))
