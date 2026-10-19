"""Local test of arb_analysis.py on a synthetic read (no GPU, no detector rows): brute-force checks of the A1 and A2
sweeps on small random data, then the whole analysis on a fake read directory with the files of arb_read.py and
arb_scores.py. Never used for any result.   python test_synth.py OUTDIR"""
import itertools
import json
import os
import sys

import numpy as np

import arb_analysis as an

rng = np.random.default_rng(1)


def brute_a1(x, y, g):
    tx = np.r_[np.unique(x), np.inf]
    ty = np.r_[-np.inf, np.unique(y)]
    best = 0
    for tc, ts in itertools.product(tx, ty):
        best = max(best, int(g[(x < tc) & (y > ts)].sum()))
    return best


def brute_a2(dC, dN, g):
    best = 0
    for b in [0.0] + an.GRID:
        u = dC - b * dN
        for t in np.r_[np.unique(u), np.inf]:
            best = max(best, int(g[u < t].sum()))
    return best


def sweeps():
    for trial in range(40):
        n = int(rng.integers(5, 60))
        x = np.round(rng.normal(size=n), 1)          # ties on purpose
        y = np.round(rng.normal(size=n), 1)
        g = rng.choice([-1, 0, 1], size=n, p=[0.5, 0.2, 0.3])
        t = an.fit_a1(x, y, g)
        assert t["gain"] == brute_a1(x, y, g), (trial, t)
        assert int(g[an.a1_switch(x, y, t)].sum()) == t["gain"], (trial, t)
        dC, dN = np.abs(rng.normal(size=n)) * 3, np.abs(rng.normal(size=n)) * 2
        t = an.fit_a2(dC, dN, g)
        assert t["gain"] == brute_a2(dC, dN, g), (trial, t)
        assert int(g[an.a2_switch(dC, dN, t)].sum()) == t["gain"], (trial, t)
    print("sweeps: A1 and A2 equal brute force on 40 random sets")


def fake_read(out, p, n_chunks=40, chunk=10000):
    n = n_chunks * chunk
    obs = (rng.random((n, 12)) < 0.02).astype(np.uint8)
    hard = rng.random(n) < 0.08                                    # hard shots: both decoders unsure
    # C14': logits whose sign is obs, flipped on a few bits of hard shots, small |z| there
    sgn = 2.0 * obs - 1
    flipC = hard[:, None] & (rng.random((n, 12)) < 0.06)
    mag = np.abs(rng.normal(4, 1.5, (n, 12)))
    mag[flipC] = np.abs(rng.normal(0.3, 0.3, flipC.sum()))
    zc = (np.where(flipC, -sgn, sgn) * mag).astype(np.float32)
    zc = zc.astype(np.float16).astype(np.float32)                   # exactly representable in float16, as bf16 mostly is
    bitsC = (zc > 0).astype(np.uint8)
    # S55'-deep: answer obs, flipped on hard shots with probability; margin small on hard shots
    flipN = hard[:, None] & (rng.random((n, 12)) < 0.05)
    bitsN = obs ^ flipN.astype(np.uint8)
    margin = np.where(hard, np.abs(rng.normal(3, 3, n)), np.abs(rng.normal(20, 5, n))).astype(np.float32)
    fC, fN = (bitsC != obs).any(1), (bitsN != obs).any(1)
    tau1 = an.TAU1[p]
    routed = margin.astype(np.float64) < tau1
    differ = (bitsC != bitsN).any(1)
    rows = np.nonzero(routed & differ)[0].astype(np.int64)
    pw = 1 << np.arange(12)
    yN = bitsN[rows].astype(np.int64) @ pw
    sc = rng.normal(0, 1, (len(rows), 4096)).astype(np.float32)
    sc[np.arange(len(rows)), yN] = sc.max(1) + margin[rows]
    # the margin must be top1 - top2 of the scores, as the engine's is
    s2 = sc.copy()
    s2[np.arange(len(rows)), yN] = -np.inf
    margin[rows] = sc[np.arange(len(rows)), yN] - s2.max(1)
    routed = margin.astype(np.float64) < tau1
    assert np.array_equal(np.nonzero(routed & differ)[0], rows)
    tag = f"p{p:.5f}"
    c0 = 6433102
    np.savez(os.path.join(out, f"{tag}.npz"), chunks=np.array([c0, c0 + n_chunks]), models=np.array(["C14p.pt", "S55p_deep.pt"]),
             pred0=np.packbits(bitsC, axis=1), conf0=np.abs(zc).min(1), fail0=fC, z0=zc.astype(np.float16),
             pred1=np.packbits(bitsN, axis=1), conf1=margin, fail1=fN, cls1=(bitsN.astype(np.int64) @ pw).astype(np.uint16),
             obs=np.packbits(obs, axis=1), chunk=np.repeat(np.arange(c0, c0 + n_chunks), chunk).astype(np.int32),
             row=np.tile(np.arange(chunk), n_chunks).astype(np.int32))
    json.dump({"p": p, "kind": 14, "chunks": [c0, c0 + n_chunks], "models": [{"kind": "cascade"}, {"kind": "convlines"}]},
              open(os.path.join(out, f"{tag}.json"), "w"))
    np.save(os.path.join(out, f"{tag}_z32.npy"), zc)
    json.dump({"float16_equals_read_z": True, "minabs_equals_read_conf_bitwise": True, "sha256": "test"},
              open(os.path.join(out, f"{tag}_z32.json"), "w"))
    np.save(os.path.join(out, f"{tag}_scores.npy"), sc)
    np.save(os.path.join(out, f"{tag}_scores_rows.npy"), rows)
    json.dump({"pass": True, "scores_sha256": "test"}, open(os.path.join(out, f"{tag}_scores.json"), "w"))
    return len(rows)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sweeps()
    for p in (0.002, 0.003):
        print(f"fake read at {p}: {fake_read(out, p):,} disagreement rows", flush=True)


if __name__ == "__main__":
    main()
