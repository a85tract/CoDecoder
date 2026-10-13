"""Circuits of this route (SETTINGS §1): the official Ising-Decoding `MemoryCircuit` at commit 3ce4379, built exactly as
its inference datapipe builds them (data/datapipe_stim.py, QCDataPipePreDecoder_Memory_inference): rotated surface code,
code_rotation 'XV', get_all_detectors=True, add_boundary_detectors=True, n_rounds = d (the builder counts the final
perfect stabilizer round), noise NoiseModel.from_single_p(p) (the public 25-parameter mapping), legacy scalar slots set
to the placeholder the datapipe uses (they are unused when a noise model is given).

The official circuit has no detector or qubit coordinates. `geometry` derives them from the measurement records (the
latest ancilla measurement of each detector: its coordinate and round; boundary detectors get t = n_rounds) and
`annotated` returns the same circuit with DETECTOR(x, y, t) coordinates, for networks that read coordinates
(nemu.nets.Geometry). Sampling always uses the unannotated circuit; `dem_errors` shows the two have the same DEM.
"""
import hashlib
import os
import sys

import numpy as np
import stim

NV_CODE = os.environ.get("ICMP_NV", os.path.expanduser("~/work/ising/ext/ising_nv/code"))
NV_COMMIT = "3ce4379fa6cf475789007bde4b6a8206e8622992"
DISTANCES = (7, 9, 11, 13)
P_VERDICT = 0.006
NOISE_VERDICT = "legacy"                # ASSUMPTIONS I12


def _nv():
    if NV_CODE not in sys.path:
        sys.path.insert(0, NV_CODE)


def memory_circuit(d, p=P_VERDICT, basis="X", rounds=None, noise="25p"):
    """The official MemoryCircuit object (has .stim_circuit and .code). noise '25p': NoiseModel.from_single_p(p) with
    the datapipe's placeholders; 'legacy': the official legacy single-p path (noise_model=None, idle = single-qubit
    gate = two-qubit gate error p, SPAM 2p/3), as the inference datapipe builds it with cfg.test.noise_model='none'."""
    _nv()
    from qec.noise_model import NoiseModel
    from qec.surface_code.memory_circuit import MemoryCircuit
    if noise == "25p":
        nm = NoiseModel.from_single_p(p)
        ph = float(nm.get_max_probability())
    elif noise == "legacy":
        nm, ph = None, float(p)
    else:
        raise ValueError(noise)
    mc = MemoryCircuit(distance=d, idle_error=ph, sqgate_error=ph, tqgate_error=ph, spam_error=(2.0 / 3.0) * ph,
                       n_rounds=rounds or d, basis=basis, code_rotation="XV", get_all_detectors=True,
                       noise_model=nm, add_boundary_detectors=True)
    mc.set_error_rates()
    return mc


def circuit(d, p=P_VERDICT, basis="X", rounds=None, noise="25p") -> stim.Circuit:
    return memory_circuit(d, p, basis, rounds, noise).stim_circuit


def dem(c: stim.Circuit, decompose=True) -> stim.DetectorErrorModel:
    """The DEM as the official code builds it for PyMatching."""
    return c.detector_error_model(decompose_errors=decompose, approximate_disjoint_errors=True)


def sha(obj) -> str:
    return hashlib.sha256(str(obj).encode()).hexdigest()


def geometry(mc):
    """(m, 5) int array per detector: x, y (ancilla coordinate), t (round of its latest ancilla measurement, n_rounds for
    the boundary detectors), is_x (X-type check), boundary (compares final data measurements with the last ancilla)."""
    code = mc.code
    nd = len(code.data_qubits)
    xq = set(int(q) for q in code.xcheck_qubits)
    coord = {}
    for part in ("data", "syndrome_X", "syndrome_Z"):
        for q, v in code.code_dict[part].items():
            coord[int(q)] = tuple(int(round(c)) for c in v["coord"])
    meas_q, meas_r, cnt, rows = [], [], {}, []
    for inst in mc.stim_circuit.flattened():
        if inst.name == "DETECTOR":
            recs = [len(meas_q) + t.value for t in inst.targets_copy()]
            anc = [k for k in recs if meas_q[k] >= nd]
            k = max(anc)
            boundary = any(meas_q[r] < nd for r in recs)
            q = meas_q[k]
            rows.append((*coord[q], meas_r[k] + int(boundary), int(q in xq), int(boundary)))
        elif stim.gate_data(inst.name).produces_measurements:
            for t in inst.targets_copy():
                q = t.value
                meas_q.append(q)
                meas_r.append(cnt.get(q, 0))
                cnt[q] = cnt.get(q, 0) + 1
    g = np.array(rows, dtype=np.int64)
    assert len(g) == mc.stim_circuit.num_detectors
    return g


def annotated(mc) -> stim.Circuit:
    """The flattened circuit with DETECTOR(x, y, t) coordinates from `geometry` (same detectors, same order)."""
    g = geometry(mc)
    out, i = stim.Circuit(), 0
    for inst in mc.stim_circuit.flattened():
        if inst.name == "DETECTOR":
            x, y, t = (float(v) for v in g[i, :3])
            out.append(stim.CircuitInstruction("DETECTOR", inst.targets_copy(), [x, y, t]))
            i += 1
        else:
            out.append(inst)
    return out


def dem_errors(dm: stim.DetectorErrorModel):
    """{targets string: probability} of the error instructions, coordinates and order ignored; repeated instructions
    with the same targets (a REPEAT block's DEM keeps them apart, a flattened circuit's merges them) combined as
    independent flips, p = p1 (1 - p2) + p2 (1 - p1)."""
    out = {}
    for inst in dm.flattened():
        if inst.type == "error":
            k = " ".join(str(t) for t in inst.targets_copy())
            p, q = inst.args_copy()[0], out.get(k, 0.0)
            out[k] = p * (1 - q) + q * (1 - p)
    return out


def same_dem(a, b, tol=1e-12):
    ea, eb = dem_errors(a), dem_errors(b)
    return ea.keys() == eb.keys() and all(abs(ea[k] - eb[k]) <= tol for k in ea)
