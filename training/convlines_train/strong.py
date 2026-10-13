"""Strong decoders for convlines_train.tune: factories that take the stim.Circuit and return a callable
det (k, m) uint8 -> predicted observables (k, n_obs) uint8.

PyMatching (2.4.0) on the circuit's DEM with decompose_errors=True and approximate_disjoint_errors=True, as
convlines/circuits.py builds it for the studies: plain matching, and correlated matching (its two-pass mode that also
uses the X-Z correlations; the studies' strong decoder on the surface code). BP-OSD from the ldpc package, for codes
whose DEM has hyperedges (BB codes), if ldpc is installed. Any other decoder is a function of the same form, passed to
the tool as "module:function"."""
import numpy as np


def _dem(circuit, decompose=True):
    return circuit.detector_error_model(decompose_errors=decompose, approximate_disjoint_errors=True)


def pymatching(circuit):
    import pymatching as pm
    m = pm.Matching.from_detector_error_model(_dem(circuit))
    return lambda det: np.asarray(m.decode_batch(det), np.uint8).reshape(len(det), -1)


def pymatching_correlated(circuit):
    import pymatching as pm
    m = pm.Matching.from_detector_error_model(_dem(circuit), enable_correlations=True)
    return lambda det: np.asarray(m.decode_batch(det, enable_correlations=True), np.uint8).reshape(len(det), -1)


def bposd(circuit, max_iter=100, osd_order=10):
    """BP-OSD (ldpc 2: product-sum BP, OSD-CS of order osd_order) on the DEM's check and observable matrices."""
    from ldpc import BpOsdDecoder
    from scipy import sparse
    dem = _dem(circuit, decompose=False)
    rows, cols, obs_r, obs_c, prior = [], [], [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        k = len(prior)
        prior.append(inst.args_copy()[0])
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                rows.append(t.val)
                cols.append(k)
            elif t.is_logical_observable_id():
                obs_r.append(t.val)
                obs_c.append(k)
    H = sparse.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(dem.num_detectors, len(prior)))
    L = sparse.csr_matrix((np.ones(len(obs_r), np.uint8), (obs_r, obs_c)), shape=(dem.num_observables, len(prior)))
    dec = BpOsdDecoder(H, error_channel=prior, max_iter=max_iter, bp_method="product_sum", osd_method="osd_cs",
                       osd_order=osd_order)

    def run(det):
        out = np.zeros((len(det), dem.num_observables), np.uint8)
        for i, s in enumerate(det):
            out[i] = (L @ dec.decode(np.asarray(s, np.uint8))) % 2
        return out
    return run
