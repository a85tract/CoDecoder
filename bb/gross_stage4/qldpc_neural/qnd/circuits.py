"""Reference circuits and their detector error models (SETTINGS.md section 2).

Circuits come from the Relay-BP repository (github.com/trmue/relay, Apache-2.0) at the pinned revision
RELAY_REV, directory tests/testdata/<family>/. They are not copied into git: a compute host clones the
repository (scripts/fetch_relay.sh) and every load checks the file's sha256 against data/circuit_manifest.csv.

A DEM is compiled without decomposition (hyperedges kept, both detector sectors) into
  H  (detectors x faults, uint8 CSC)   detector incidence
  L  (observables x faults, uint8 CSC) observable incidence
  q  (faults,) float64                 fault priors
Mechanisms with identical (detectors, observables) are merged (independent XOR of their probabilities).
"""
import csv
import hashlib
import os
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import stim

RELAY_REV = "d185194ba0cb4101ced4340d82b2ee6d42f225f0"
RELAY_DIR = os.path.expanduser(os.environ.get("QND_RELAY", "~/work/relay"))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(HERE, "data", "circuit_manifest.csv")

# code name -> (n, k, d, rounds, polynomial strings as in the Relay file names)
CODES = {
    "bb72": (72, 12, 6, 6, "A=x^3+y+y^2,B=y^3+x+x^2"),
    "bb144": (144, 12, 12, 12, "A=x^3+y+y^2,B=y^3+x+x^2"),
}
# generated codes (ASSUMPTIONS Q29): rotated surface code, Z memory, d rounds, uniform circuit noise at p (the
# surface-code route's "L2" level); name "sc<d>"
SURFACE = {"sc7": 7}


def generated_name(code: str, basis: str, p: float) -> str:
    ps = f"{p:.6f}".rstrip("0")
    return f"generated:surface_code:rotated_memory_{basis.lower()},distance={SURFACE[code]},rounds={SURFACE[code]},p={ps}"


def generate(code: str, basis: str, p: float) -> stim.Circuit:
    if basis != "Z":
        raise ValueError("surface-code circuits are Z-basis memories here")
    d = SURFACE[code]
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


def relay_name(code: str, basis: str, p: float) -> str:
    """File name of a Relay-BP reference circuit; basis in {'X', 'Z', 'choi_XZ'}."""
    n, k, d, r, poly = CODES[code]
    ps = f"{p:.6f}".rstrip("0")
    return (f"circuit=bicycle_bivariate_{n}_{k}_{d}_memory_{basis},distance={d},rounds={r},"
            f"error_rate={ps},noise_model=uniform_circuit,basis=CX,{poly}.stim")


def relay_path(code: str, basis: str, p: float) -> str:
    return os.path.join(RELAY_DIR, "tests", "testdata", "bicycle_bivariate", relay_name(code, basis, p))


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@lru_cache(None)
def _manifest() -> dict:
    if not os.path.exists(MANIFEST):
        return {}
    return {r["file"]: r["sha256"] for r in csv.DictReader(open(MANIFEST))}


@lru_cache(None)
def load(code: str, basis: str, p: float, check: bool = True) -> stim.Circuit:
    if code in SURFACE:
        c = generate(code, basis, p)
        name = generated_name(code, basis, p)
        if check:
            want = _manifest().get(name)
            got = hashlib.sha256(str(c).encode()).hexdigest()
            if want is None:
                raise KeyError(f"{name} is not in {MANIFEST}")
            if got != want:
                raise ValueError(f"sha256 mismatch for {name}: {got} != manifest {want}")
        return c
    path = relay_path(code, basis, p)
    if check:
        want = _manifest().get(os.path.basename(path))
        if want is None:
            raise KeyError(f"{os.path.basename(path)} is not in {MANIFEST}")
        got = sha256(path)
        if got != want:
            raise ValueError(f"sha256 mismatch for {path}: {got} != manifest {want}")
    return stim.Circuit.from_file(path)


def dem_matrices(dem: stim.DetectorErrorModel):
    """Undecomposed DEM -> (H, L, q) with identical mechanisms merged; column order = first appearance."""
    cols = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
            elif t.is_separator():
                raise ValueError("decomposed DEM given; compile with decompose_errors=False")
        # a target listed twice cancels (XOR)
        key = (tuple(sorted(x for x in set(dets) if dets.count(x) % 2)),
               tuple(sorted(x for x in set(obs) if obs.count(x) % 2)))
        if key in cols:
            p0 = cols[key]
            cols[key] = p0 * (1 - p) + p * (1 - p0)
        else:
            cols[key] = p
    keys = list(cols)
    q = np.array([cols[k] for k in keys])

    def incidence(rows_of, nrows):
        indptr = np.zeros(len(keys) + 1, np.int64)
        idx = []
        for j, k in enumerate(keys):
            idx.extend(rows_of(k))
            indptr[j + 1] = len(idx)
        return sp.csc_matrix((np.ones(len(idx), np.uint8), np.array(idx, np.int64), indptr),
                             shape=(nrows, len(keys)))

    H = incidence(lambda k: k[0], dem.num_detectors)
    L = incidence(lambda k: k[1], dem.num_observables)
    return H, L, q


@lru_cache(None)
def problem(code: str, basis: str, p: float, sector: str = "all"):
    """(H, L, q, keep): DEM matrices and the indices of the kept detectors in the full circuit's detector list.
    sector 'all' keeps every detector (the network's input and the xyz teachers); sector 'basis' keeps only the
    detectors of the memory basis (Relay-BP's xyz=False decoding, the published baselines), selected by relay's
    testdata.filter_detectors_by_basis."""
    c = load(code, basis, p)
    if sector == "all":
        keep = np.arange(c.num_detectors)
    elif sector == "basis":
        f = filter_by_basis(c, basis)
        keep = kept_detectors(c, f)
        c = f
    else:
        raise ValueError(sector)
    H, L, q = dem_matrices(c.detector_error_model(decompose_errors=False))
    return H, L, q, keep


def kept_detectors(full: stim.Circuit, filtered: stim.Circuit) -> np.ndarray:
    """Indices of filtered's detectors in full (the filter only deletes DETECTOR instructions, so the kept ones
    appear in the same order with identical targets)."""
    fd = [(str(i.targets_copy()), tuple(i.gate_args_copy())) for i in full.flattened() if i.name == "DETECTOR"]
    gd = [(str(i.targets_copy()), tuple(i.gate_args_copy())) for i in filtered.flattened() if i.name == "DETECTOR"]
    keep, j = [], 0
    for i, d in enumerate(fd):
        if j < len(gd) and d == gd[j]:
            keep.append(i)
            j += 1
    if j != len(gd):
        raise ValueError("filtered detectors are not a subsequence of the full circuit's")
    return np.array(keep)


def filter_by_basis(c: stim.Circuit, basis: str) -> stim.Circuit:
    import sys
    sys.path.insert(0, os.path.join(RELAY_DIR, "tests"))
    from testdata import filter_detectors_by_basis
    return filter_detectors_by_basis(c, basis)
