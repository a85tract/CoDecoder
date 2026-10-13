"""PyMatching references and the global decoder of the Ising pipeline (SETTINGS §3): the official DEM
(decompose_errors=True, approximate_disjoint_errors=True); plain = Matching.from_detector_error_model(dem), as the
official evaluation; correlated = the same with enable_correlations=True (PyMatching 2.4 correlated matching)."""
import time

import numpy as np
import pymatching

from icmp import circuits


def matcher(d, correlated=False, c=None):
    dm = circuits.dem(c or circuits.circuit(d))
    if correlated:
        return pymatching.Matching.from_detector_error_model(dm, enable_correlations=True)
    return pymatching.Matching.from_detector_error_model(dm)


def decode(m, det, correlated=False):
    """(n,) uint8 predicted observable."""
    if correlated:
        out = m.decode_batch(det, enable_correlations=True)
    else:
        out = m.decode_batch(det)
    return np.asarray(out, dtype=np.uint8).reshape(len(det), -1)[:, 0]


def time_single_shot(m, syndromes, n_rounds, warmup=50, correlated=False, tails=False):
    """Mean single-shot `decode` time per round in µs, the official protocol (evaluation/logical_error_rate.py,
    _time_single_shot_latency_stim): warm-up, then one matcher.decode per syndrome, mean / n_rounds. Also the median."""
    kw = {"enable_correlations": True} if correlated else {}
    n = len(syndromes)
    for i in range(min(warmup, n)):
        m.decode(syndromes[i], **kw)
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        m.decode(syndromes[i], **kw)
        ts[i] = time.perf_counter() - t0
    if tails:                                            # per-shot tails over the actual shots (SETTINGS §0.5)
        return (float(ts.mean() / n_rounds * 1e6), float(np.median(ts) / n_rounds * 1e6),
                float(np.percentile(ts, 99) / n_rounds * 1e6), float(ts.max() / n_rounds * 1e6))
    return float(ts.mean() / n_rounds * 1e6), float(np.median(ts) / n_rounds * 1e6)
