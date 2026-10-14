"""The Grace-basis conversion of SPEED-RT: one factor per reference decoder, fitted over every point of NVIDIA's Grace
(Neoverse-V2) PyMatching times, from the timing host's own times of the same decoders on the same circuits and noise,
measured in the stage's CPU session at the session's concurrency.

NVIDIA's values (Ising paper, arXiv 2604.12841 v1, Table VII: uncorrelated PyMatching; Table IX: correlated PyMatching),
microseconds per syndrome round, on the rotated surface-code memory circuits of NVIDIA's Ising-Decoding (d rounds, the
legacy single-parameter noise) at p = 0.3 % and 0.6 %. A shot holds d rounds, so per shot = per round x d.

For decoder x (plain PM "plain", correlated PM "corr") with points i = (d, p) of NVIDIA's table and the SLOWEST of the
point's three per-shot times t_x(i) on the timing host (the statistic that goes against the claim: a larger factor
favours our pairs):
    r_x(i) = G_x(i) x d / t_x(i)                     the point's Grace-over-Xeon ratio (a diagnostic)
    f_x    = exp(mean_i ln r_x(i))                   the least-squares fit in log space (the geometric mean)
    s_x    = standard deviation of ln r_x(i) (n - 1)  the spread
One common factor (the geometric mean over every point of both decoders) when |ln f_plain - ln f_corr| <= max(s_plain,
s_corr), and then every CPU time takes it; otherwise plain-PM terms take f_plain, correlated-PM terms f_corr, and every
other CPU decoder (kind "other": Relay-BP, full-DEM Relay-BP, ldpc BP-OSD, Decoder Switching) the geometric mean
sqrt(f_plain x f_corr). Which CPU term is which kind is fixed by the stage that uses the conversion (its registration
lists them). Residual of a point = ln G_x(i) - ln(f x t_x(i) / d) with
the factor the point's decoder takes. Pure functions: no file, clock or console."""
import math

NVIDIA_US_PER_ROUND = {                      # (decoder, d, p): microseconds per round
    ("plain", 13, 0.003): 3.38, ("plain", 13, 0.006): 9.97,
    ("plain", 21, 0.003): 13.41, ("plain", 21, 0.006): 29.95,
    ("plain", 31, 0.003): 28.78, ("plain", 31, 0.006): 91.06,
    ("corr", 5, 0.003): 1.15, ("corr", 5, 0.006): 1.78,
    ("corr", 9, 0.003): 3.35, ("corr", 9, 0.006): 7.51,
    ("corr", 13, 0.003): 9.14, ("corr", 13, 0.006): 21.51,
    ("corr", 17, 0.003): 24.12, ("corr", 17, 0.006): 50.63,
    ("corr", 21, 0.003): 49.75, ("corr", 21, 0.006): 92.27,
    ("corr", 31, 0.003): 133.31, ("corr", 31, 0.006): 270.83,
}
DECODERS = ("plain", "corr")
KINDS = ("plain", "corr", "other")


def points():
    """Every (decoder, d, p) the timing host must time, in a fixed order."""
    return sorted(NVIDIA_US_PER_ROUND)


def _mean(xs):
    return math.fsum(xs) / len(xs)


def _sd(xs):
    m = _mean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fit(xeon_us_per_shot):
    """xeon_us_per_shot: {(decoder, d, p): the slowest of the point's three per-shot times in us on the timing host}
    for every point of points() -> {points: [...], factors: {plain, corr, common}, spread: {plain, corr}, common: bool,
    applied: {plain, corr, other}, residuals: {...}}. A missing point raises: the fit is defined on every point."""
    missing = [k for k in points() if k not in xeon_us_per_shot]
    if missing:
        raise ValueError(f"the fit needs every point; missing {missing}")
    rows, logs = [], {x: [] for x in DECODERS}
    for k in points():
        dec, d, p = k
        g = NVIDIA_US_PER_ROUND[k] * d
        t = xeon_us_per_shot[k]
        if not t > 0:
            raise ValueError(f"{k}: time {t}")
        lr = math.log(g / t)
        logs[dec].append(lr)
        rows.append(dict(decoder=dec, d=d, p=p, grace_us_per_shot=g, xeon_us_per_shot=t, ratio=g / t))
    f = {x: math.exp(_mean(logs[x])) for x in DECODERS}
    s = {x: _sd(logs[x]) for x in DECODERS}
    allogs = logs["plain"] + logs["corr"]
    f["common"] = math.exp(_mean(allogs))
    common = abs(math.log(f["plain"]) - math.log(f["corr"])) <= max(s["plain"], s["corr"])
    applied = {x: (f["common"] if common else f[x]) for x in DECODERS}
    applied["other"] = f["common"] if common else math.exp((math.log(f["plain"]) + math.log(f["corr"])) / 2)
    for r in rows:
        r["residual_ln"] = math.log(r["grace_us_per_shot"]) - math.log(applied[r["decoder"]] * r["xeon_us_per_shot"])
    return dict(points=rows, factors=f, spread=s, common=common, applied=applied,
                rule="common when |ln f_plain - ln f_corr| <= max(s_plain, s_corr); else plain and corr their own factor "
                     "and other CPU decoders sqrt(f_plain x f_corr)")


def to_grace(us, kind, result):
    """A timing-host CPU time (us) of kind "plain" (plain PM), "corr" (correlated PM) or "other" (every other CPU
    decoder) on the Grace basis."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {KINDS}")
    return us * result["applied"][kind]
