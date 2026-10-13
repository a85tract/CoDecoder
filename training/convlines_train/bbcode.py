"""Your bivariate bicycle (BB) code in the code that trained the BB72 and gross networks (training/bb, frozen).

Those modules know two codes (bb72, bb144) and read Relay-BP's pinned circuit files. register() adds a code under its
own name with your circuit: qnd.bb.PARAMS and bb.code.ROUNDS get the code, and qnd.circuits.load and bb.code.load_circuit
return your circuit for that name (every other name still goes to the pinned files). The code is given in Bravyi et al.'s form,
(l, m, (a1, a2, a3), (b1, b2, b3)) with A = x^a1 + y^a2 + y^a3 and B = y^b1 + x^b2 + x^b3 (BB72: 6, 6, (3, 1, 2),
(3, 1, 2); gross: 12, 6, (3, 1, 2), (3, 1, 2)). The circuit must follow Relay-BP's conventions, which bb.code.Layout
and bb.code.translates check: an X-basis memory of R rounds; qubits L data (0 ... lm - 1), R data, X checks, Z checks,
site (a, b) at index a m + b of each block; one X detector per X check and round, Z detectors comparing consecutive
rounds, the final X detectors on the data measurement; the code's k logical X observables. Your noise model is free;
the network shares its weights across the torus, so it suits noise that is (nearly) the same on every site
(design.json records bb.design's check of the DEM's translation invariance).

design() runs bb.design for the registered code (the stem's taps: every co-firing offset of the DEM; the blocks':
the Tanner offsets) and checks the translate labels (bb.code.translates' GF(2) maps) against a circuit with one
observable per translate on fresh shots."""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root
FROZEN = os.path.join(ROOT, "training", "bb")
_USER = {}


def frozen():
    """(qnd.bb, qnd.circuits, bb.code, bb.net, bb.design) of training/bb."""
    for p in (os.path.join(FROZEN, "ising_compare"), os.path.join(FROZEN, "qldpc_neural")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from qnd import bb as qbb, circuits
    from bb import code as bcode, design, net as bnet
    return qbb, circuits, bcode, bnet, design


def parse(text):
    """'l,m,a1,a2,a3,b1,b2,b3' -> (l, m, (a1, a2, a3), (b1, b2, b3))."""
    v = [int(x) for x in text.split(",")]
    if len(v) != 8:
        raise ValueError("--bb needs l,m,a1,a2,a3,b1,b2,b3")
    return v[0], v[1], tuple(v[2:5]), tuple(v[5:8])


def register(params, circuit):
    """Register the code and its circuit; returns the name (user_ + the circuit's sha256 prefix) the frozen modules
    know them by."""
    qbb, circuits, bcode, _, _ = frozen()
    text = str(circuit)
    name = "user_" + hashlib.sha256(text.encode()).hexdigest()[:12]
    l, m = params[0], params[1]
    q_x = 2 * l * m                                             # the first X check
    rounds = sum(t.value == q_x for inst in circuit.flattened() if inst.name in bcode.MEAS
                 for t in inst.targets_copy() if t.is_qubit_target)
    if rounds < 2:
        raise ValueError(f"qubit {q_x} (the first X check in Relay-BP's order) is measured {rounds} times")
    qbb.PARAMS[name] = (l, m, tuple(params[2]), tuple(params[3]))
    bcode.ROUNDS[name] = rounds
    _USER[name] = text
    if not getattr(circuits.load, "user_codes", False):           # bb.code.Layout reads bb.code.load_circuit,
        pinned, pinned_at = circuits.load, bcode.load_circuit       # bb.design qnd.circuits.problem -> load
        import stim

        def load(code, basis, p, check=True):
            return stim.Circuit(_USER[code]) if code in _USER else pinned(code, basis, p, check)

        def load_circuit(code, p):
            return stim.Circuit(_USER[code]) if code in _USER else pinned_at(code, p)
        load.user_codes = True
        circuits.load, bcode.load_circuit = load, load_circuit
    return name


def design(name, out, label_shots=2000, seed=0):
    """bb.design's report for the registered code into out (JSON), plus the translate-label check; returns it."""
    _, _, bcode, _, design_mod = frozen()
    argv = sys.argv
    sys.argv = ["bb.design", "--code", name, "--p", "0", "--out", out, "--label-shots", "0"]
    try:
        design_mod.main()
    finally:
        sys.argv = argv
    rep = json.load(open(out))
    tr = bcode.translates(name, 0.0)
    k, G = tr["M"].shape[1], tr["M"].shape[0]
    aug, extra = bcode.translate_circuit(name, 0.0)
    det, obs = aug.compile_detector_sampler(seed=seed).sample(label_shots, separate_observables=True)
    y = (obs[:, :k].astype(np.int64) @ tr["Mcol"] + det.astype(np.int64) @ tr["Gdet"]) % 2
    cols = [kk * G + g for kk, g in extra]
    bad = int((y[:, cols] != obs[:, k:]).any(0).sum())
    rep["labels"] = dict(shots=label_shots, seed=seed, translates_checked=len(cols), translates_differing=bad)
    json.dump(rep, open(out, "w"), indent=1)
    if bad:
        raise ValueError(f"{bad} translate labels differ from the circuit's translated observables")
    return rep
