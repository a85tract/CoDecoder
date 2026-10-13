"""Shots for Cascade (bb.cascade_eval): circuits at any p, the study's seed namespace and fresh evaluation shots. The
training streams of the original module (train_seed, the Stim stream and the DEM sampler of the training program, and
the sampler benchmark) are not part of this release.

Circuits. Relay-BP's BB72 X-memory files (qnd.circuits, revision d185194, sha256-pinned) exist on a p grid without 0.55 %,
Cascade's training point. Their noise is uniform: every noise channel and every noisy measurement carries the single
argument p. circuit_at(code, p) rewrites the arguments of the 0.003 file to p after checking that each equals 0.003;
check_circuit_at() confirms that this reproduces every pinned file of the code exactly (stim.Circuit equality).

Seeds (the study's own namespace, 2^55 + ..., disjoint from every documented namespace of ising_compare and qldpc_neural:
those lie below 2^33, at 2^40 + .., 2^41 + .., 2^42 + .. (< 2^43) and >= 2^62 for test sets):
  eval_seed(kind, p, chunk)    = 2^55 + 2e15 + 1e13 kind + 1e7 round(1e6 p) + chunk   (kind < 100, chunk < 1e7)
  kind 0 = evaluation sets, 1 = the training monitor, 2 = smoke and unit checks, 3 = sampler checks,
  4 = BN recalibration shots (bb.cascade_eval --bn-recal), 5 = the stage-2 fidelity pilot's evaluation,
  6 = the stage-2 adapter and S55 checks.
  The fixed monitoring block of C14, C6 and S55 is kind 1, chunks 100-119 at 0.3 % and 100-109 at 0.55 %.

Circuit families. Every function takes family = "uniform" (the default: the circuits and seeds above, unchanged) or
"depspam_fm", the noise-model fit's variant (bb/cascade/noise.py from ising-cascade-cal fb722bd4: the same gates,
detectors and observables, DEPOLARIZE1(p) after every reset and before every measurement on top of each flip p, and a
noisy final data measurement), built by noise.build on circuit_at(code, p), so at any p. Each family has its own seeds,
FAMILY_SEED[family] above the uniform ones:
  depspam_fm  eval_seed  = 2^55 + 5e15 + ...  in [2^55 + 5e15, 2^55 + 6e15)
disjoint from the uniform family's [2^55 + 1e15, 2^55 + 3e15) and from every other namespace. Kinds keep their
meaning; eval kinds 0 the frozen networks' rows, 1 the monitoring block of C14' and S55' (chunks 100-119 at 0.3 %,
100-109 at 0.55 %), 3 the family's sampler and BP+OSD checks, 6 its label check, 7 the routing thresholds' calibration
blocks."""
import multiprocessing as mp
import os
import time
from functools import lru_cache

import numpy as np

SEED_BASE = 2 ** 55
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E",
         "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"}
MEAS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP", "MXX", "MYY", "MZZ"}
BASE_P = 0.003
FAMILY_SEED = {"uniform": 0, "depspam_fm": 3 * 10 ** 15}      # circuit family -> seed offset (module docstring)
FAMILIES = tuple(FAMILY_SEED)


def eval_seed(kind, p, chunk, family="uniform"):
    assert 0 <= kind < 100 and 0 <= chunk < 10 ** 7 and 0 < p < 1
    return SEED_BASE + FAMILY_SEED[family] + 2 * 10 ** 15 + kind * 10 ** 13 + int(round(p * 1e6)) * 10 ** 7 + chunk


def _rewrite(circ, p0, p):
    import stim
    out = stim.Circuit()
    for inst in circ:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, _rewrite(inst.body_copy(), p0, p), tag=inst.tag))
            continue
        args = inst.gate_args_copy()
        if inst.name in NOISE or (inst.name in MEAS and args):
            if len(args) != 1 or args[0] != p0:
                raise ValueError(f"not uniform noise at p = {p0}: {inst}")
            args = [p]
        out.append(stim.CircuitInstruction(inst.name, inst.targets_copy(), args, tag=inst.tag))
    return out


@lru_cache(None)
def circuit_at(code, p, family="uniform"):
    """The code's X-memory circuit at noise p: the pinned 0.003 file with every noise argument set to p (uniform), or
    the noise-model fit's noise.build(family, p) on that circuit (depspam_fm)."""
    if family != "uniform":
        if family not in FAMILY_SEED or code != "bb72":
            raise ValueError(f"no circuit family {family} for {code}")
        from bb.cascade import noise                     # the noise-model fit's noise models (bb/cascade/noise.py)
        return noise.build(family, float(p), src=circuit_at(code, p))
    from qnd import circuits
    return _rewrite(circuits.load(code, "X", BASE_P), BASE_P, float(p))


def check_circuit_at(code="bb72"):
    """circuit_at(p) == the pinned file for every p of the code in the circuit manifest. -> the p checked."""
    from qnd import circuits
    n, k, d, r, _ = circuits.CODES[code]
    ok = []
    for name in circuits._manifest():
        if name.startswith(f"circuit=bicycle_bivariate_{n}_{k}_{d}_memory_X,"):
            p = float(name.split("error_rate=")[1].split(",")[0])
            if circuit_at(code, p) != circuits.load(code, "X", p):
                raise ValueError(f"circuit_at({code}, {p}) differs from the pinned file")
            ok.append(p)
    return sorted(ok)


def sample(code, p, n, seed, family="uniform"):
    """n shots of the circuit at p from one seed -> (det (n, D) uint8, obs (n, k) uint8)."""
    det, obs = circuit_at(code, p, family).compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def _eval_chunk(args):
    code, kind, p, j, n = args[:5]
    family = args[5] if len(args) > 5 else "uniform"
    return sample(code, p, n, eval_seed(kind, p, j, family), family)


def eval_shots(code, p, chunks, n=10000, kind=0, procs=1, family="uniform"):
    """Fresh shots: chunks (an iterable of chunk numbers) of n shots each at p -> (det, obs) concatenated in order."""
    jobs = [(code, kind, p, j, n, family) for j in chunks]
    if procs > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            out = pool.map(_eval_chunk, jobs)
    else:
        out = [_eval_chunk(x) for x in jobs]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def env_record(code="bb72"):
    """Versions and a fingerprint of Stim's output on this host (1,024 shots at p = 0.003 from a fixed seed)."""
    import hashlib
    import platform
    import stim
    det, obs = circuit_at(code, BASE_P).compile_detector_sampler(seed=eval_seed(3, BASE_P, 0)).sample(
        1024, separate_observables=True, bit_packed=True)
    return {"stim": stim.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "host": platform.node(), "cpu": _cpu_model(),
            "stim_fingerprint": hashlib.sha256(det.tobytes() + obs.tobytes()).hexdigest()[:16]}


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "env"])
    ap.add_argument("--code", default="bb72")
    a = ap.parse_args()
    if a.cmd == "check":
        print("circuit_at reproduces the pinned files at p =", check_circuit_at(a.code))
    else:
        print(json.dumps(env_record(a.code)))
