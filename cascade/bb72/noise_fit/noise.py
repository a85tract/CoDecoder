"""Alternative noise models on the frozen BB72 schedule (the noise-model fit, step 4).

The frozen circuit (relay-bp testdata at d185194, 'uniform_circuit': the noise model of Bravyi et al.,
arXiv:2308.07915) has five kinds of noise location, all at strength p:
  cx     DEPOLARIZE2 after every CX (2,592 per shot)
  idle   DEPOLARIZE1 on idle data qubits (792): in every round L data idle in step 1 (X-ancilla reset, CX R->Z), R data
         in step 7 (Z-ancilla measurement, CX X->L) and all data in step 8 (X-ancilla measurement, Z-ancilla reset);
         round 1 has no step-1 idle (data prepared), round 6 has a step-7 idle on the Z ancillas and none in step 8
  reset  X_ERROR after R, Z_ERROR after RX (ancilla resets every round and the data preparation; 504)
  meas   flip probability of every ancilla measurement (432)
  final  flip probability of the final data MX (72; 0 in the frozen circuit)
(depspam_fm adds DEPOLARIZE1 after every reset and before every measurement, keys rdep / mdep / fdep.)
A model rebuilds the circuit with the same gates, detectors and observables and each location's strength set to
factor x p (factor 0 drops the location). Every idle slot of this schedule falls in a step with an ancilla
measurement or reset, so SI1000's 'idle during measurement/reset = 2p' applies to all of them.

  python noise.py --check     the 'uniform' rebuild must give the frozen circuit's DEM exactly"""
import argparse
import os
import sys

import stim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("QND_PATH", os.path.join(HERE, "..", "..", "..", "qldpc_neural")))
from qnd import circuits  # noqa: E402

FACTORS = {
    "uniform": dict(cx=1, idle=1, reset=1, meas=1, final=0),       # the frozen circuit
    "uniform_fm": dict(cx=1, idle=1, reset=1, meas=1, final=1),    # + flips p on the final data measurement
    "noidle_fm": dict(cx=1, idle=0, reset=1, meas=1, final=1),     # no idle noise, noisy final measurement
    "spam2_fm": dict(cx=1, idle=1, reset=2, meas=2, final=2),      # resets and measurements at 2p
    "si1000": dict(cx=1, idle=2, reset=2, meas=5, final=5),        # SI1000 strengths (CX for CZ, no 1q gates here)
    # every reset followed and every measurement preceded by DEPOLARIZE1(p) as well as its flip p (final data
    # measurement included): an effective flip of about 5p/3 on each
    "depspam_fm": dict(cx=1, idle=1, reset=1, meas=1, final=1, rdep=1, mdep=1, fdep=1),
}


def build(model, p):
    f = FACTORS[model]
    src = circuits.load("bb72", "X", p)
    out = stim.Circuit()
    for ins in src.flattened():
        name, tg, args = ins.name, ins.targets_copy(), ins.gate_args_copy()
        if name == "DEPOLARIZE2":
            continue                                        # re-added after its CX
        if name == "CX":
            out.append("CX", tg)
            if f["cx"]:
                out.append("DEPOLARIZE2", tg, [f["cx"] * p])
        elif name == "DEPOLARIZE1":
            if f["idle"]:
                out.append("DEPOLARIZE1", tg, [f["idle"] * p])
        elif name in ("X_ERROR", "Z_ERROR"):                # each follows its R / RX in the frozen circuit
            if f["reset"]:
                out.append(name, tg, [f["reset"] * p])
            if f.get("rdep"):
                out.append("DEPOLARIZE1", tg, [f["rdep"] * p])
        elif name in ("M", "MX"):
            k = "meas" if args else "final"                 # the frozen circuit's only noiseless measurement is the final MX
            dep = f.get("mdep" if args else "fdep")
            if dep:
                out.append("DEPOLARIZE1", tg, [dep * p])
            out.append(name, tg, [f[k] * p] if f[k] else [])
        elif name in ("MZ", "MY", "MR", "MRX", "MRZ", "MRY", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "Y_ERROR", "E",
                      "ELSE_CORRELATED_ERROR"):
            raise ValueError(f"unexpected instruction {name}")
        else:
            out.append(ins)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        for p in (0.002, 0.003):
            frozen = circuits.load("bb72", "X", p).detector_error_model(decompose_errors=False)
            rebuilt = build("uniform", p).detector_error_model(decompose_errors=False)
            print(p, "uniform rebuild == frozen DEM:", rebuilt == frozen, frozen.num_errors)
            for m in FACTORS:
                c = build(m, p)
                d = c.detector_error_model(decompose_errors=False)
                print(f"  {m}: detectors {d.num_detectors} errors {d.num_errors} "
                      f"sum of priors {sum(i.args_copy()[0] for i in d.flattened() if i.type == 'error'):.4f}")


if __name__ == "__main__":
    main()
