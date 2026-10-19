"""Local test of arb2_analysis.py on synthetic data (no GPU, no detector rows): brute-force checks of the J, P and T
sweeps on small random sets, then the whole analysis on a fake read directory in arb2_read.py's layout. Never used for
any result.   PYTHONPATH=RELEASE/cascade/d13:CODE python test_synth.py OUTDIR LOCK"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np

import arb2_analysis as an

rng = np.random.default_rng(7)


def disagreement_rows(n):
    """c, z with S = c > 0 != B = z > 0, magnitudes with ties, zeros and exact ratios on purpose."""
    S = rng.random(n) < 0.5
    ac = np.round(np.abs(rng.normal(0, 2, n)), 1)
    az = np.round(np.abs(rng.normal(0, 2, n)), 1)
    ac[rng.random(n) < 0.1] = 0.0
    az[rng.random(n) < 0.1] = 0.0
    c = np.where(S, ac, -ac)
    z = np.where(S, -az, az)
    ok = ((c > 0) != (z > 0))
    return c[ok], z[ok]


def brute_j(c, z, g):
    ac, az = np.abs(c), np.abs(z)
    r = np.unique(np.r_[0.0, ac[az > 0] / az[az > 0]])
    ws = sorted(set(np.r_[r, np.sqrt(r[:-1] * r[1:]), r / 2, 2 * r, r * 1.0001, r * 0.9999, 1.0]) | {np.inf})
    return max(int(g[an.j_switch(c, z, float(w))].sum()) for w in ws)


def brute_t(x, y, g):
    tx = np.r_[np.unique(x), np.inf]
    ty = np.r_[-np.inf, np.unique(y)]
    return max(int(g[(x < tc) & (y > ts)].sum()) for tc, ts in itertools.product(tx, ty))


def brute_p(x, y, g):
    best = 0
    for b in [0.0] + an.GRID:
        u = x - b * y
        for t in np.r_[np.unique(u), np.inf]:
            best = max(best, int(g[u < t].sum()))
    return best


def sweeps():
    for trial in range(60):
        c, z = disagreement_rows(int(rng.integers(5, 70)))
        g = rng.choice([-1, 1], size=len(c), p=[0.6, 0.4])
        t = an.fit_j(c, z, g)
        bj = brute_j(c, z, g)
        assert t["gain"] == bj, (trial, t, bj)
        assert int(g[an.j_switch(c, z, t["w"])].sum()) == t["gain"], (trial, t)
        # J switches iff |c| > w |z| on these rows
        for w in (0.0, 0.5, 1.0, 3.0):
            assert np.array_equal(an.j_switch(c, z, w), np.abs(c) > w * np.abs(z)), (trial, w)
        x, y = np.abs(z), np.abs(c)
        t = an.fit_t(x, y, g)
        assert t["gain"] == brute_t(x, y, g), (trial, t)
        assert int(g[an.t_switch(x, y, t)].sum()) == t["gain"], (trial, t)
        if trial < 15:
            t = an.fit_p(x, y, g)
            assert t["gain"] == brute_p(x, y, g), (trial, t)
            assert int(g[an.p_switch(x, y, t)].sum()) == t["gain"], (trial, t)
    # ties keep the larger w: two equal optima
    c = np.array([3.0, -1.0])
    z = np.array([-1.0, 4.0])          # ratios 3 and 0.25
    g = np.array([1, 1])
    t = an.fit_j(c, z, g)
    assert t["gain"] == 2 and t["w"] < 0.25, t
    g = np.array([1, -1])
    t = an.fit_j(c, z, g)
    assert t["gain"] == 1 and 0.25 < t["w"] < 3, t
    print("sweeps: J, P and T equal brute force on 60 random sets (P on 15)")


def fake_read(out, p, n_chunks):
    d = os.path.join(out, "read", f"p{p:.4f}")
    os.makedirs(d, exist_ok=True)
    for s in range(0, n_chunks, 10):
        e = min(s + 10, n_chunks)
        n = (e - s) * 10000
        obs = (rng.random(n) < 0.3).astype(np.uint8)
        hard = rng.random(n) < 0.15
        sgn = 2.0 * obs - 1
        ac = np.where(hard, np.abs(rng.normal(0, 4, n)), np.abs(rng.normal(12, 3, n)))
        az = np.where(hard, np.abs(rng.normal(0, 2, n)), np.abs(rng.normal(9, 3, n)))
        flipS = hard & (rng.random(n) < 0.15)
        flipB = hard & (rng.random(n) < 0.08)
        c = (np.where(flipS, -sgn, sgn) * ac).astype(np.float32)
        z = (np.where(flipB, -sgn, sgn) * az).astype(np.float32)
        z = z.astype(np.float16).astype(np.float32)
        meta = {"p": p, "kind": 11, "chunks": [s, e], "seeds": [0] * (e - s), "digests": [""] * (e - s)}
        np.savez(os.path.join(d, f"part_{s:05d}-{e:05d}.npz"), obs=obs, det=np.zeros((n, 273), np.uint8),
                 chunk=np.repeat(np.arange(s, e), 10000).astype(np.uint16), c=c, sdec=(c > 0).astype(np.uint8),
                 z=z, bdec=(z > 0).astype(np.uint8), meta=np.array(json.dumps(meta)))


def whole(out, lock):
    for p in (0.006, 0.0035):
        fake_read(out, p, 10 if p == 0.006 else 12)
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "arb2_analysis.py"),
                        "--read", os.path.join(out, "read"), "--lock", lock, "--out", os.path.join(out, "analysis"),
                        "--allow-partial"], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0
    s = json.load(open(os.path.join(out, "analysis", "summary.json")))
    for p, q in s["p"].items():
        e = q["counts"]["eval"]
        assert e["composition"] - e["oracle_repairs"] == e["oracle_fail"]
        for nm in an.RULES:
            ev = q["rules"][nm]["eval"]
            assert ev["failures"] == e["composition"] - ev["net"], (p, nm)
            assert ev["net"] == ev["repairs"] - ev["breaks"]
            assert ev["repairs"] <= e["oracle_repairs"]
        o = q["reference"]["routed_oracle"]
        assert o["breaks"] == 0 and o["repairs"] == e["oracle_repairs"] and o["failures"] == e["oracle_fail"]
        al = q["reference"]["always_network_on_disagreement"]
        assert al["repairs"] == e["oracle_repairs"] and al["breaks"] == e["routed_only_S"]
        k = q["j_curve"]["w"].index(float("inf"))
        assert q["j_curve"]["fit_gain"][k] == 0 and q["j_curve"]["eval_gain"][k] == 0
        assert max(q["j_curve"]["fit_gain"]) <= q["rules"]["J"]["fit"]["gain"]
    print("whole analysis on a fake read: consistent; choice", s["choice"]["carried_forward"],
          "fit scores", {k: round(v, 4) for k, v in s["choice"]["fit_scores"].items()})


if __name__ == "__main__":
    sweeps()
    whole(sys.argv[1], sys.argv[2])
