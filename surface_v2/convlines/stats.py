"""The paired statistics of the registered comparisons: on the same shots, the exact two-sided McNemar test on the
discordant shots and the interval of LER(a) - LER(b) by Newcombe's (1998) method 8 (Wilson score limits of each
marginal proportion combined with the signed, uncorrected phi). A decoder a is "better" than b when the (1 - alpha)
interval lies entirely below 0 and p < alpha; both studies used alpha = 0.025 (97.5 % intervals), and their claims
are intersection-union tests over all cells, so no multiplicity correction applies."""
import math

import numpy as np


def wilson(k, n, z):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe8(a, b, c, d, z):
    """p1 - p2 with p1 = (a + b) / n, p2 = (a + c) / n (a both fail, b only 1, c only 2, d neither) and its interval:
    Wilson limits of each marginal, combined with the signed uncorrected phi. -> (difference, lower, upper, phi)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if den == 0 else (a * d - b * c) / math.sqrt(den)
    dt = p1 - p2
    lo = dt - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = dt + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return dt, lo, hi, phi


def paired(fa, fb, alpha=0.025):
    """fa, fb: boolean failure arrays on the same shots. Failures, discordant counts, the exact McNemar p, the LER
    difference with its (1 - alpha) interval (diff_lo, diff_hi) and the 95 % one beside it, phi and the verdict."""
    from scipy.stats import binomtest, norm
    fa, fb = np.asarray(fa, bool), np.asarray(fb, bool)
    n = len(fa)
    both, oa, ob = int((fa & fb).sum()), int((fa & ~fb).sum()), int((~fa & fb).sum())
    disc = oa + ob
    p = binomtest(oa, disc, 0.5).pvalue if disc else 1.0
    cells = (both, oa, ob, n - both - oa - ob)
    dt, lo, hi, phi = newcombe8(*cells, z=float(norm.ppf(1 - alpha / 2)))
    _, lo95, hi95, _ = newcombe8(*cells, z=float(norm.ppf(0.975)))
    verdict = "better" if (hi < 0 and p < alpha) else ("worse" if (lo > 0 and p < alpha) else "not shown")
    return dict(shots=n, fail_a=int(fa.sum()), fail_b=int(fb.sum()), only_a=oa, only_b=ob, mcnemar_p=p, diff_ler=dt,
                diff_lo95=lo95, diff_hi95=hi95, verdict=verdict, alpha=alpha, diff_lo=lo, diff_hi=hi, phi=phi,
                method="Newcombe (1998) method 8, signed uncorrected phi")
