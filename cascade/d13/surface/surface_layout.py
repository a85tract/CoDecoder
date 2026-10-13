"""The surface-code circuits of the ConvLines release and Cascade's grid on them.

Circuits. The release's verdict circuits (convlines_release circuits/d{d}_legacy.stim, read through its own loader
convlines.circuits.circuit, which checks the registered sha256): NVIDIA Ising-Decoding's MemoryCircuit, rotated
surface code, X-basis memory, R = d rounds, the legacy single-parameter noise at p = 0.006 (DEPOLARIZE1(p) idle,
DEPOLARIZE2(p) after the CX layers, X_ERROR / Z_ERROR(2p/3) after resets and before measurements; the last syndrome
round and the data readout carry none). The files exist at p = 0.006 only. circuit_at(d, p) sets every DEPOLARIZE
argument to p and every flip argument to 2p/3 after checking that each equals 0.006 or 0.004; check_circuit_at()
confirms that p = 0.006 gives the pinned text back exactly.

Grid. The release's grid file (weights/grid_d{d}.npz, checked against weights/MANIFEST.sha256: idx (D, 3) = (t, i, j)
per detector, is_x), the geometry
ConvLines reads: each detector at the ancilla of its latest measurement (coordinates divided by the lattice spacing,
so i, j = 0 .. d) and that measurement's round (the boundary detectors, which compare the final data measurement with
the last X round, at t = R). X places have i + j of one parity, Z the other (checked).

Observable support. The circuit has no coordinates, so the data qubits are placed from its gates: each data qubit
interacts (CX) with the ancillas of the plaquettes at the corners of its own cell, and data qubit (r, c) is the one
cell (r, c) .. (r + 1, c + 1) of the plaquette grid whose present plaquettes are exactly those ancillas (unique,
checked). OBSERVABLE_INCLUDE(0) lists d final data measurements; their (r, c) must form one line (checked)."""
import hashlib
import os
import sys
from functools import lru_cache

import numpy as np

P0 = 0.006
DEP = {"DEPOLARIZE1", "DEPOLARIZE2"}
FLIP = {"X_ERROR", "Z_ERROR"}


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def release_root():
    """The surface-code decoder's directory (convlines/, circuits/, timing/): this repository's evaluation/ unless
    CONVLINES_RELEASE names another copy."""
    root = os.path.expanduser(os.environ.get("CONVLINES_RELEASE", os.path.join(REPO, "evaluation")))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def pinned_circuit(d):
    release_root()
    from convlines import circuits
    return circuits.circuit(d)


def _rewrite(circ, p):
    import stim
    out = stim.Circuit()
    for inst in circ:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, _rewrite(inst.body_copy(), p), tag=inst.tag))
            continue
        args = inst.gate_args_copy()
        if inst.name in DEP or inst.name in FLIP:
            want = P0 if inst.name in DEP else 2 * P0 / 3
            if len(args) != 1 or abs(args[0] - want) > 1e-15:
                raise ValueError(f"not the legacy noise at p = {P0}: {inst}")
            args = [p if inst.name in DEP else 2 * p / 3]
        elif inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS") and args:
            raise ValueError(f"an instruction with arguments the rewrite does not know: {inst}")
        out.append(stim.CircuitInstruction(inst.name, inst.targets_copy(), args, tag=inst.tag))
    return out


@lru_cache(None)
def circuit_at(d, p):
    """The release's d circuit with the legacy noise at p (p = 0.006: the pinned file)."""
    return _rewrite(pinned_circuit(d), float(p))


def check_circuit_at(d):
    c = pinned_circuit(d)
    if str(circuit_at(d, P0)) != str(c):
        raise ValueError(f"circuit_at({d}, {P0}) differs from the pinned file")
    return hashlib.sha256(str(c).encode()).hexdigest()


class Layout:
    """Detector places of the d circuit on Cascade's (T, d + 1, d + 1) grid and the observable's data qubits.
    t, i, j, typ (0 X, 1 Z): (D,) per detector in the circuit's order; support: (d, 2) data-qubit (r, c) of observable 0."""

    def __init__(self, d):
        self.d = d
        release_root()
        from convlines import circuits as rc
        path = os.path.join(rc.WEIGHTS, f"grid_d{d}.npz")
        self.grid_sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
        manifest = dict(reversed(line.split()) for line in open(os.path.join(rc.REPO, "weights", "MANIFEST.sha256"))
                        if line.strip())
        if manifest.get(f"surface/grid_d{d}.npz") != self.grid_sha256:
            raise ValueError(f"{path}: not the grid of weights/MANIFEST.sha256")
        g = np.load(path)
        idx, is_x = g["idx"].astype(np.int64), g["is_x"].astype(np.int64)
        circ = pinned_circuit(d)
        self.circuit_sha256 = hashlib.sha256(str(circ).encode()).hexdigest()
        self.D = D = circ.num_detectors
        if idx.shape != (D, 3) or circ.num_observables != 1:
            raise ValueError("the grid file does not match the circuit")
        self.t, self.i, self.j = idx.T
        self.typ = 1 - is_x
        self.R = R = int(self.t.max())
        if self.t.min() != 0 or self.i.max() != d or self.j.max() != d or min(self.i.min(), self.j.min()) != 0:
            raise ValueError("unexpected grid extent")
        par = (self.i + self.j) % 2
        if len(np.unique(par[self.typ == 0])) != 1 or len(np.unique(par[self.typ == 1])) != 1 or \
                par[self.typ == 0][0] == par[self.typ == 1][0]:
            raise ValueError("X and Z places do not form a checkerboard")
        self.x_parity = int(par[self.typ == 0][0])
        node = (self.t * (d + 1) + self.i) * (d + 1) + self.j
        if len(np.unique(node)) != D:
            raise ValueError("two detectors share a place")
        self.per_layer = [(int(((self.t == t) & (self.typ == 0)).sum()), int(((self.t == t) & (self.typ == 1)).sum()))
                          for t in range(R + 1)]
        self.support = self._support(circ, idx)

    def _support(self, circ, idx):
        import stim
        d = self.d
        meas_q, anc_place, nbrs, obs_recs = [], {}, {}, []
        mid = set()                                      # qubits measured before the final data readout
        n_det = 0
        for inst in circ.flattened():
            name = inst.name
            if name == "DETECTOR":
                recs = [len(meas_q) + t.value for t in inst.targets_copy()]
                q = meas_q[max(r for r in recs if meas_q[r] in mid)]     # its latest ancilla measurement
                pl = (int(idx[n_det, 1]), int(idx[n_det, 2]))
                if anc_place.setdefault(q, pl) != pl:
                    raise ValueError(f"ancilla {q} has two places")
                n_det += 1
            elif name == "OBSERVABLE_INCLUDE":
                obs_recs += [len(meas_q) + t.value for t in inst.targets_copy()]
            elif name == "CX":
                tg = [t.value for t in inst.targets_copy()]
                for a, b in zip(tg[::2], tg[1::2]):
                    nbrs.setdefault(a, set()).add(b)
                    nbrs.setdefault(b, set()).add(a)
            elif stim.gate_data(name).produces_measurements:
                for t in inst.targets_copy():
                    if name in ("MR", "MRX"):
                        mid.add(t.value)
                    meas_q.append(t.value)
        data = meas_q[-d * d:]
        if len(set(data)) != d * d or set(data) & mid:
            raise ValueError("the last d^2 measurements are not the data readout")
        present = set(anc_place.values())
        place = {}
        for q in data:
            S = {anc_place[a] for a in nbrs.get(q, ()) if a in anc_place}
            cands = []
            for r in range(d):
                for c in range(d):
                    cell = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
                    if S == cell & present:
                        cands.append((r, c))
            if len(cands) != 1:
                raise ValueError(f"data qubit {q}: {len(cands)} cells match its ancillas {sorted(S)}")
            place[q] = cands[0]
        if len(set(place.values())) != d * d:
            raise ValueError("two data qubits share a cell")
        sup = np.array(sorted(place[meas_q[r]] for r in obs_recs), np.int64)
        if len(sup) != d or not (len(np.unique(sup[:, 0])) == 1 or len(np.unique(sup[:, 1])) == 1):
            raise ValueError(f"the observable is not one line of d data qubits: {sup.tolist()}")
        return sup

    def describe(self):
        return {"d": self.d, "R": self.R, "detectors": self.D, "grid": [self.R + 1, self.d + 1, self.d + 1],
                "x_parity": self.x_parity, "per_layer_xz": self.per_layer, "support": self.support.tolist(),
                "circuit_sha256": self.circuit_sha256, "grid_sha256": self.grid_sha256}
