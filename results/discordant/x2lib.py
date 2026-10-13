"""X2 (REVISION_PLAN.md 2.11): helpers for the discordant-shot analysis of ConvLines against correlated PyMatching on
the registered surface-code tests: study 2's T2 at d = 13 and R17's T6 at d = 17.

Inputs, all read-only: the circuits, test-set manifests and network grids of this repository (evaluation/circuits,
evaluation/testsets, results/r17/run, weights/surface) and the per-shot records of the two test reads, which are
archived on the cluster and not published ($X2_ARCHIVE). The shots are rebuilt from the manifests' seeds and
checked byte for byte against the manifests' sha256 before any record is joined to them.

Matching quantities. PyMatching 2.4 on the circuit's DEM (decompose_errors, approximate_disjoint_errors), as the
studies ran it. For the class margin the boundary of the X-type detectors is split: every boundary edge on the side of
logical line 0 goes to an extra detector A instead (the other side stays the PyMatching boundary). A correction that
ends an even (odd) number of chains on that side lies in one homology class (the other), so decoding with A = 0 and
A = 1 gives the least weight of a correction in each class, plain or correlated (enable_correlations), and the signed
gap W(wrong class) - W(true class) in nats: positive where matching's least-weight class is the true one."""
import hashlib
import io
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))      # the repository root
ARCH = os.environ.get("X2_ARCHIVE", "<archive>")   # the archive of per-shot records (not published)
OUT = os.environ.get("X2_OUT", os.path.join(HERE, "out"))
CHUNK = 10_000
REF_EVERY = 32                     # the reference sample: every 32nd shot of every chunk (313 per chunk)

TESTS = {
    13: dict(set="T2", manifest="evaluation/testsets/T2.json", split="test2", preds=f"{ARCH}/ising/out/s2/preds/d13",
             network="d13_b5_r775", tau=1.8486117124557495, q=0.02,
             registered=dict(failures=5627, routed=19405, cpm=6549)),
    17: dict(set="T6", manifest="results/r17/run/t6_manifest.json", split="test6", preds=f"{ARCH}/ising_r17/out/preds/d17",
             network="d17_b5_r787", tau=4.539173126220703, q=0.08,
             registered=dict(failures=2708, routed=79652, cpm=2943)),
}
PARTS = ("ours_deployed", "pm_corr", "pm")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    with open(path, "rb") as fh:
        return sha_bytes(fh.read())


def manifest(d):
    """{chunk j: (file name, seed, sha256)} and the circuit's sha256 of the test set at d."""
    t = TESTS[d]
    m = json.load(open(os.path.join(REPO, t["manifest"])))
    files = {}
    for name, e in m["files"].items():
        if name.startswith(f"d{d}_p0.006_X_{t['split']}_"):
            files[int(e["j"])] = (name, int(e["seed"]), e["sha256"])
    csha = m["circuit_sha256"]
    return files, (csha[str(d)] if isinstance(csha, dict) else csha)


def circuit(d, want):
    import stim
    text = open(os.path.join(REPO, "evaluation", "circuits", f"d{d}_legacy.stim")).read()
    assert sha_bytes(text.encode()) == want, f"d = {d}: circuit text is not the registered one"
    return stim.Circuit(text)


def npz_bytes(arrays):
    """convlines.testset.write_npz's bytes (stored members in name order, fixed timestamps), in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.asarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o444 << 16
            z.writestr(zi, b.getvalue())
    return buf.getvalue()


def rebuild(c, seed, want):
    """One test chunk from its seed -> (det (n, m) uint8, obs (n,) uint8), its file bytes checked against want."""
    det, obs = c.compile_detector_sampler(seed=seed).sample(CHUNK, separate_observables=True)
    det = det.astype(np.uint8)
    arrays = dict(det=np.packbits(det, axis=1), obs=obs[:, 0].astype(np.uint8), seed=np.int64(seed))
    got = sha_bytes(npz_bytes(arrays))
    assert got == want, f"seed {seed}: rebuilt chunk sha256 {got[:16]} is not the manifest's {want[:16]}"
    return det, arrays["obs"]


def records(d, j, chunk_sha):
    """The test read's per-shot records of chunk j: {part: npz arrays}, each file checked against its provenance
    record (output sha256) and bound to the chunk (input sha256)."""
    out = {}
    for part in PARTS:
        base = os.path.join(TESTS[d]["preds"], part, f"c{j:03d}")
        prov = json.load(open(base + ".prov.json"))
        assert prov["part"] == part and int(prov["chunk"]) == j and int(prov["d"]) == d, f"{base}: provenance"
        assert prov["input_sha256"] == chunk_sha, f"{base}: read another chunk"
        assert sha_file(base + ".npz") == prov["output_sha256"], f"{base}.npz: not the recorded output"
        z = np.load(base + ".npz")
        out[part] = {k: z[k] for k in z.files}
    return out


class Geometry:
    """The network grid of the release (weights/surface/grid_d{d}.npz): per detector the site (t, i, j), X type and the
    signed distance to every logical line (half-line units). X-type detectors lie between data rows k and k + 1,
    k = 0 .. d - 2; bdist = min(k + 1, d - 1 - k) is the number of data rows a chain must cross to reach the nearer
    X boundary (1 = next to it)."""

    def __init__(self, d):
        g = np.load(os.path.join(REPO, "weights", "surface", f"grid_d{d}.npz"))
        self.d = d
        self.t, self.i, self.j = (g["idx"][:, a].astype(np.int64) for a in range(3))
        self.is_x = g["is_x"].astype(bool)
        off = g["line_off"].astype(np.int64)                  # (m, d)
        self.off0, self.offl = off[:, 0], off[:, d - 1]
        x = self.is_x
        assert np.all(np.abs(self.off0[x]) % 2 == 1), "an X-type detector sits on a data row"
        assert len(np.unique(np.sign(self.off0[x]))) == 1, "X-type detectors on both sides of line 0"
        k = (np.abs(self.off0) - 1) // 2                         # row gap index for X-type detectors
        assert np.all(k[x] <= d - 2) and np.all(np.abs(self.offl[x]) == 2 * (d - 1 - k[x]) - 1), "line geometry"
        self.k = np.where(x, k, -1)
        self.bdist = np.where(x, np.minimum(k + 1, d - 1 - k), -1)
        self.m = len(self.t)


def dem_parts(dem):
    """[(p, [components: (detectors, flips_obs)])] of a decomposed DEM's error instructions (no repeat blocks)."""
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps, dets, obs = [], [], False
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append((dets, obs))
                dets, obs = [], False
            elif t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs = obs ^ True
        comps.append((dets, obs))
        out.append((inst.args_copy()[0], comps))
    return out


def split_dem(dem, geo):
    """The DEM with the X boundary on line 0's side moved to a new detector A = geo.m (see the module text), and the
    checks that the split is clean: every X-type boundary component sits next to line 0 or line d - 1."""
    import stim
    A = geo.m
    new = stim.DetectorErrorModel()
    n_a = n_b = 0
    for p, comps in dem_parts(dem):
        targets = []
        for ci, (dets, obs) in enumerate(comps):
            if ci:
                targets.append(stim.target_separator())
            dd = list(dets)
            if len(dd) == 1 and geo.is_x[dd[0]]:
                k = geo.k[dd[0]]
                assert k in (0, geo.d - 2), f"X boundary component at row gap {k}"
                if k == 0:
                    dd.append(A)
                    n_a += 1
                else:
                    n_b += 1
            targets += [stim.target_relative_detector_id(x) for x in dd]
            if obs:
                targets.append(stim.target_logical_observable_id(0))
        new.append("error", p, targets)
    return new, dict(components_to_A=n_a, components_other_side=n_b)


def edge_kind(geo, u, v):
    """0 time-like (same site, consecutive rounds), 1 space-like (same round), 2 space-time (another site, another
    round), 3 boundary (v < 0 or v = A)."""
    if v < 0 or v >= geo.m or u < 0 or u >= geo.m:
        return 3
    if geo.t[u] == geo.t[v]:
        return 1
    if geo.i[u] == geo.i[v] and geo.j[u] == geo.j[v]:
        return 0
    return 2


def graph_edges(dem, geo):
    """The distinct graphlike detector pairs (u < v) of the DEM's components, X-type only, with their kind."""
    seen = set()
    for _, comps in dem_parts(dem):
        for dets, _ in comps:
            if len(dets) == 2 and geo.is_x[dets[0]] and geo.is_x[dets[1]]:
                seen.add((min(dets), max(dets)))
    e = np.array(sorted(seen), dtype=np.int64)
    kind = np.array([edge_kind(geo, u, v) for u, v in e], dtype=np.int8)
    return e, kind


def edge_set(edges):
    """{(u, v)} mod 2 of a decode_to_edges_array result (boundary -1 kept)."""
    s = set()
    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        key = (int(min(u, v)), int(max(u, v))) if u >= 0 and v >= 0 else (int(max(u, v)), -1)
        s ^= {key}
    return s
