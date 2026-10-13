"""A surface-code front end's geometry and training labels from a Stim circuit: any rotated-surface-code memory
experiment with one observable (your own circuit and noise model), with QUBIT_COORDS for its data qubits.

The ConvLines network (convlines/net.py) needs, for every detector, its grid site (t, i, j), its check type and its
signed distance to each of the d logical lines; training needs the flip of every line (the d rows of data qubits
parallel to the observable) and the GF(2) map A with y_j XOR y_obs = (D A)_j mod 2, D the detector bits. spec()
derives them:
  data qubits  the qubits measured exactly once (the final data measurement; so the circuit needs two rounds or more).
               The observable must read d of them, all on one line (one shared coordinate), and the d^2 data qubits
               must form d such lines of d.
  site         a detector's own coordinates (x, y, t) when every detector has three or more (Stim's generated circuits
               do); otherwise the coordinates of the qubit of its latest ancilla record and that qubit's measurement
               round, one more for a detector that reads data records (scripts/make_circuits.py's rule for NVIDIA's
               circuits, which carry no coordinates). (i, j) = (x, y) over their lattice spacing, t from 0.
  type         is_x = the check is of the observable's type (X-type in an X-basis memory): its site also has a
               detector that reads the final data measurement.
  line_off     round(2 (p_det - p_j) / s): p the coordinate the lines' qubits share, s the spacing of the lines.
  labels       the circuit with one more observable per line (its data qubits' final records); observable 0 stays.
  A            column j: every detector of the observable's type whose check lies between the observable's line and
               line j (the product of those checks over all rounds is the product of the two lines). It is checked
               against the sampled line flips: they must agree on every one of n_check fresh shots.
Measurements must be single-qubit (M, MR, MX, MRX, MY, MRY, MZ, MRZ)."""
import hashlib
import io

import numpy as np
import stim

SINGLE = {"M", "MR", "MX", "MRX", "MY", "MRY", "MZ", "MRZ"}


def _records(circuit):
    """(qubit of every measurement record, the record's index among that qubit's measurements)."""
    q_of, k_of, cnt = [], [], {}
    for inst in circuit.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            if inst.name not in SINGLE:
                raise ValueError(f"{inst.name}: only single-qubit measurements are supported")
            for t in inst.targets_copy():
                q_of.append(t.value)
                k_of.append(cnt.get(t.value, 0))
                cnt[t.value] = cnt.get(t.value, 0) + 1
    return np.array(q_of, np.int64), np.array(k_of, np.int64), cnt


def _detectors(circuit):
    """[(absolute records, coordinates)] per detector, and the records of observable 0 (a set, XOR semantics)."""
    dets, obs, n = [], set(), 0
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            dets.append(([n + t.value for t in inst.targets_copy()], inst.gate_args_copy()))
        elif inst.name == "OBSERVABLE_INCLUDE":
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    obs ^= {n + t.value}
        elif stim.gate_data(inst.name).produces_measurements:
            n += len(inst.targets_copy())
    return dets, obs


def _lattice(v):
    """Integer lattice positions of the values v (their spacing is the smallest gap)."""
    u = np.unique(np.round(v, 6))
    if len(u) == 1:
        return np.zeros(len(v), np.int64)
    s = float(np.min(np.diff(u)))
    pos = (np.round(v, 6) - u[0]) / s
    assert np.allclose(pos, np.round(pos), atol=1e-6), "detector coordinates are not on a lattice"
    return np.round(pos).astype(np.int64)


def spec(circuit, n_check=20_000, seed=0):
    """The front end's spec (a dict: d, geometry arrays, A, the lines and the circuit) of a surface-code circuit."""
    if circuit.num_observables != 1:
        raise ValueError(f"the circuit has {circuit.num_observables} observables; a surface-code memory has one")
    q_of, k_of, cnt = _records(circuit)
    dets, obs_recs = _detectors(circuit)
    data = sorted(q for q, c in cnt.items() if c == 1)
    dset = set(data)
    obs_q = sorted(int(q_of[r]) for r in obs_recs)
    d = len(obs_q)
    if not set(obs_q) <= dset or len(data) != d * d:
        raise ValueError(f"the observable reads {d} qubits but {len(data)} qubits are measured once; expected the d^2 "
                         "data qubits of a rotated surface code measured once at the end, with two rounds or more")
    coords = circuit.get_final_qubit_coordinates()
    missing = [q for q in data if q not in coords or len(coords[q]) < 2]
    if missing:
        raise ValueError(f"data qubits without (x, y) QUBIT_COORDS: {missing[:5]}")
    xy = {q: np.array(coords[q][:2], float) for q in data}
    oc = np.array([xy[q] for q in obs_q])
    shared = [ax for ax in (0, 1) if np.allclose(oc[:, ax], oc[0, ax])]
    if len(shared) != 1:
        raise ValueError("the observable's data qubits do not lie on one line of constant x or y")
    ax = shared[0]
    vals = np.unique(np.round([xy[q][ax] for q in data], 6))
    lines = [sorted(q for q in data if np.isclose(xy[q][ax], v)) for v in vals]
    if len(lines) != d or any(len(L) != d for L in lines):
        raise ValueError(f"the data qubits do not form {d} parallel lines of {d}")
    pj = vals.astype(float)
    gaps = np.diff(pj)
    assert np.allclose(gaps, gaps[0]), "the lines are not evenly spaced"
    s = float(gaps[0])
    obs_line = [L == obs_q for L in lines].index(True)
    rec_of = {int(q_of[r]): r for r in range(len(q_of)) if int(q_of[r]) in dset}       # a data qubit's one record

    m = len(dets)
    reads_data = np.array([any(int(q_of[r]) in dset for r in recs) for recs, _ in dets])
    if all(len(c) >= 3 for _, c in dets):
        site = np.array([c[:3] for _, c in dets], float)
    else:
        qc = circuit.get_final_qubit_coordinates()
        site = np.zeros((m, 3))
        for k, (recs, _) in enumerate(dets):
            anc = [r for r in recs if int(q_of[r]) not in dset]
            if not anc:
                raise ValueError(f"detector {k} reads no ancilla record")
            r = max(anc)
            q = int(q_of[r])
            if q not in qc or len(qc[q]) < 2:
                raise ValueError(f"ancilla qubit {q} has no (x, y) QUBIT_COORDS (and the detectors have none)")
            site[k] = (qc[q][0], qc[q][1], k_of[r] + int(reads_data[k]))
    t = site[:, 2] - site[:, 2].min()
    assert np.allclose(t, np.round(t)), "detector rounds are not integers"
    idx = np.stack([np.round(t).astype(np.int64), _lattice(site[:, 0]), _lattice(site[:, 1])], 1)
    if len({tuple(r) for r in idx}) != m:
        raise ValueError("two detectors share a grid site")
    obs_type = {tuple(np.round(site[k, :2], 6)) for k in np.flatnonzero(reads_data)}
    is_x = np.array([tuple(np.round(site[k, :2], 6)) in obs_type for k in range(m)])
    pd = site[:, ax]
    line_off = np.rint(2 * (pd[:, None] - pj[None, :]) / s).astype(np.int64)
    A = np.zeros((m, d), np.uint8)
    for j in range(d):
        lo, hi = sorted((pj[obs_line], pj[j]))
        A[:, j] = is_x & (pd > lo) & (pd < hi)
    sp = dict(kind="surface", d=d, num_detectors=m, idx=idx.astype(np.int16), is_x=is_x.astype(np.uint8),
              line_off=line_off.astype(np.int8), A=A, obs_line=obs_line, axis=ax,
              line_records=np.array([[rec_of[q] for q in L] for L in lines], np.int64),
              num_measurements=len(q_of), circuit=str(circuit))
    check(sp, n_check, seed)
    return sp


def label_circuit(sp):
    """(the circuit with one observable per line: 0 the original, then the other lines in order; the observable index
    of each line j)."""
    c = stim.Circuit(sp["circuit"])
    n = sp["num_measurements"]
    order = [sp["obs_line"]] + [j for j in range(sp["d"]) if j != sp["obs_line"]]
    for k, j in enumerate(order[1:], start=1):
        c.append(stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_rec(int(r) - n)
                                                                for r in sp["line_records"][j]], [k]))
    return c, np.argsort(order)


def check(sp, n, seed):
    """Every line observable deterministic, and g = D A mod 2 equal to y_j XOR y_obs on n fresh shots."""
    c, col = label_circuit(sp)
    c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)    # raises if not deterministic
    det, obs = c.compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    y = obs[:, col].astype(np.uint8)
    g = (det.astype(np.int64) @ sp["A"].astype(np.int64)) % 2
    bad = int((g != (y ^ y[:, [sp["obs_line"]]])).any(1).sum())
    if bad:
        raise ValueError(f"{bad} of {n} shots: the line flips are not the XOR of the observable-type detectors between "
                         "the lines; the circuit's detectors do not follow a standard memory experiment")


def geometry_npz(sp):
    """The geometry file convlines.net.Layout reads (idx, is_x, line_off), as bytes."""
    f = io.BytesIO()
    np.savez(f, idx=sp["idx"], is_x=sp["is_x"], line_off=sp["line_off"])
    return f.getvalue()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()
