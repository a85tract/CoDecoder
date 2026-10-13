"""Rebuild circuits/d{d}_legacy.stim and weights/surface/grid_d{d}.npz from NVIDIA's Ising-Decoding code and compare them
with the files in this repository. Needs a clone of NVIDIA/Ising-Decoding at commit 3ce4379 (ISING_DECODING_DIR) and
its Python requirements; nothing else in this repository needs NVIDIA's code.

Circuit: MemoryCircuit(distance=d, idle_error=p, sqgate_error=p, tqgate_error=p, spam_error=2p/3, n_rounds=d, basis='X',
code_rotation='XV', get_all_detectors=True, noise_model=None, add_boundary_detectors=True), set_error_rates(), p = 0.006:
the legacy single-parameter noise path, as NVIDIA's inference pipeline builds the circuits its released models were
trained on. The .stim file is the circuit's Stim text (str(circuit)).

Geometry (the network's grid): the official circuit has no coordinates, so each detector gets the coordinate of the
ancilla of its latest measurement record and that measurement's round (boundary detectors, which compare the final
data measurements with the last ancilla round, get the next round); (t, i, j) = (round, coordinates divided by their
lattice spacing). line_off = per detector and logical line j (data-qubit row j), the signed distance across the lines
in half-line units.
    ISING_DECODING_DIR=... python scripts/make_circuits.py [--write]"""
import argparse
import hashlib
import os
import sys

import numpy as np
import stim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 0.006
DISTANCES = (7, 9, 11, 13, 15, 17)


def memory_circuit(d, p=P):
    sys.path.insert(0, os.path.join(os.environ["ISING_DECODING_DIR"], "code"))
    from qec.surface_code.memory_circuit import MemoryCircuit
    mc = MemoryCircuit(distance=d, idle_error=p, sqgate_error=p, tqgate_error=p, spam_error=(2.0 / 3.0) * p,
                       n_rounds=d, basis="X", code_rotation="XV", get_all_detectors=True, noise_model=None,
                       add_boundary_detectors=True)
    mc.set_error_rates()
    return mc


def geometry(mc):
    """(m, 5) int: x, y (ancilla coordinate), t (round of its latest ancilla measurement; boundary detectors one more),
    is_x, boundary."""
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


def grid(g):
    idx = np.zeros((len(g), 3), np.int64)
    idx[:, 0] = g[:, 2] - g[:, 2].min()
    for a, col in ((1, 0), (2, 1)):
        v = np.unique(g[:, col])
        s = int(np.min(np.diff(v)))
        assert np.all((v - v.min()) % s == 0), f"coordinate {col} is not on a lattice of spacing {s}"
        idx[:, a] = (g[:, col] - v.min()) // s
    assert len({tuple(r) for r in idx}) == len(idx), "two detectors share a grid site"
    return idx


def line_offsets(mc, g, d):
    data = mc.code.code_dict["data"]
    co = np.array([data[q]["coord"] for q in sorted(data)], float)            # (d^2, 2)
    lines = co.reshape(d, d, 2)
    axis = 0 if np.allclose(lines[:, :, 0].std(1), 0) else 1
    assert np.allclose(lines[:, :, axis].std(1), 0), "a line's data qubits do not share one coordinate"
    pj = lines[:, 0, axis]
    s = float(np.median(np.abs(np.diff(pj))))
    pd = g[:, axis].astype(float)
    return np.rint(2 * (pd[:, None] - pj[None, :]) / s).astype(np.int64)


def build(d):
    mc = memory_circuit(d)
    g = geometry(mc)
    return str(mc.stim_circuit), dict(idx=grid(g).astype(np.int16), is_x=g[:, 3].astype(np.uint8),
                                      line_off=line_offsets(mc, g, d).astype(np.int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the files instead of comparing with them")
    a = ap.parse_args()
    ok = True
    for d in DISTANCES:
        text, geo = build(d)
        cpath = os.path.join(ROOT, "circuits", f"d{d}_legacy.stim")
        gpath = os.path.join(os.path.dirname(ROOT), "weights", "surface", f"grid_d{d}.npz")
        if a.write:
            open(cpath, "w").write(text)
            np.savez(gpath, **geo)
        same_c = open(cpath).read() == text
        z = np.load(gpath)
        same_g = all(np.array_equal(z[k], v) for k, v in geo.items())
        ok &= same_c and same_g
        print(f"d = {d}: circuit sha256 {hashlib.sha256(text.encode()).hexdigest()} "
              f"{'equal' if same_c else 'DIFFERENT'}; geometry {'equal' if same_g else 'DIFFERENT'}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
