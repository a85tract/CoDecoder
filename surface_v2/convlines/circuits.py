"""The verdict circuits and the PyMatching references.

circuits/d{d}_legacy.stim is NVIDIA Ising-Decoding's MemoryCircuit (commit 3ce4379, tag 0.1.0) as its inference
pipeline builds it: rotated surface code, X-basis memory, code_rotation 'XV', all detectors plus boundary detectors,
d rounds, the legacy single-parameter noise (idle, single- and two-qubit gate error p, SPAM 2p/3) at p = 0.006. The
file is the circuit's Stim text; its sha256 is the circuit sha256 every study record uses (scripts/make_circuits.py
rebuilds the files from NVIDIA's code).

PyMatching (2.4.0) works on the DEM with decompose_errors=True, approximate_disjoint_errors=True; plain matching is
Matching.from_detector_error_model(dem), correlated matching the same with enable_correlations=True."""
import hashlib
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 0.006
CIRCUIT_SHA256 = {
    7: "159b46ed3db59cf0a8dcb3bb84b469904c8b1c2e7a3252c64297356b0001c03e",
    9: "19390af8855ca9d885dfb12ff5d852c76f69eacf9600bebb26c6ff3286b77351",
    11: "a357867a1d29dacc9608c70a526507e25ef5018bb91db33a8dccd9eccad99236",
    13: "86f1d9e436a788b3cc1cb141268f064ed5e4afe7c0f8da3639852bc8ec68397e",
    15: "aed0439da21ff92e2868fa60f8e1c36acff2fd23ea5416eb0508783f9d81a7e0",
    17: "5a17d28287e216cc0797acef8a6de96a6e217b7668e1b5417a850498a2a962f7",
}


def circuit(d):
    """The verdict circuit of distance d, checked against its registered sha256."""
    import stim
    path = os.path.join(ROOT, "circuits", f"d{d}_legacy.stim")
    text = open(path).read()
    assert hashlib.sha256(text.encode()).hexdigest() == CIRCUIT_SHA256[d], f"{path}: not the registered circuit"
    c = stim.Circuit(text)
    assert str(c) == text, f"{path}: the Stim text does not round-trip"
    return c


def dem(c, decompose=True):
    return c.detector_error_model(decompose_errors=decompose, approximate_disjoint_errors=True)


def matcher(d, correlated=False, c=None):
    import pymatching
    dm = dem(c or circuit(d))
    if correlated:
        return pymatching.Matching.from_detector_error_model(dm, enable_correlations=True)
    return pymatching.Matching.from_detector_error_model(dm)


def pm_decode(m, det, correlated=False):
    """(n,) uint8 predicted observable for detector bits det (n, m) uint8."""
    if len(det) == 0:
        return np.zeros(0, np.uint8)
    out = m.decode_batch(det, enable_correlations=True) if correlated else m.decode_batch(det)
    return np.asarray(out, dtype=np.uint8).reshape(len(det), -1)[:, 0]
