"""Two statistics helpers of bb.cascade_eval, copied verbatim from the training program (bb/cascade_train.py at
ising-cascade-net 64b2e25d), which is not part of this release: Eq. (1) of Cascade's paper and Wilson's interval."""
import math


def p_block_to_pl(pb, k=12, R=6):
    """Eq. (1) of the paper: the block failure rate -> P_L per logical qubit per round."""
    x = 2 * (1 - pb) ** (1 / k) - 1
    return (1 - x ** (1 / R)) / 2 if x > 0 else float("nan")


def wilson(f, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    ph = f / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)
