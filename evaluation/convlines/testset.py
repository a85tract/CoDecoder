"""The test sets T2 (study 2, d = 7, 9, 11, 13), T3 (study 3, d = 15) and T5 (study 4, d = 17): seeds, sampling, the file
format, the check against the manifests in testsets/, and the validation shots the timing runs used.

A test set is a list of chunks of 10,000 shots. Chunk j at distance d holds the shots of the verdict circuit
(convlines.circuits) drawn by Stim's detector sampler with seed(d, j):

    det, obs = circuit.compile_detector_sampler(seed=seed(d, j)).sample(10_000, separate_observables=True)

and is stored as d{d}_p0.006_X_{split}_{j:03d}.npz (split test2 / test3 / test4) with the members det (np.packbits of the
detector bits along each shot, big-endian), obs (observable 0, uint8) and seed (int64), written byte-deterministically
(zip members stored uncompressed in name order, timestamp 1980-01-01, numpy's .npy format), so the file's sha256 is a
function of the shots. The manifest lists every chunk with its seed and sha256.

Seeds. The registration fixed the rule before the test set existed; its inputs were unknown until after the freeze:
    seed(d, j) = 2^62 + (int.from_bytes(sha256(TAG || A || R || bytes([d]) || j.to_bytes(2, "big"))[:8], "big") mod 2^62)
TAG = "ising_compare study 2 T2\n" (T3: "... study 3 T3\n", T5: "... study 4 T4\n"), A = the bytes of the study's approval
record (made with the freeze commit; for T5 the record of the re-registration S4-R3; its sha256 is in the manifest), R = the 32 randomness bytes of the drand mainnet beacon round fixed by
the freeze commit's time (the first round at least 600 s later; round, signature and randomness are in the manifest
and can be checked against any drand endpoint: randomness = sha256(signature)). The manifests record every seed, so
regenerating a chunk needs only the manifest; `seeds` re-derives them when A is at hand.

Bit-for-bit regeneration needs Stim 1.16.0 and numpy 2.2.6 on x86-64 (Stim's output depends on its build and SIMD
path; all three sets were sampled on x86-64 CPUs with AVX2).

    python -m convlines.testset regen --set T2 --d 9 --chunks 0,99 --out DIR    (sample, write, compare with the manifest)
    python -m convlines.testset check --set T2 --dir DIR                         (re-hash the chunk files in DIR)
    python -m convlines.testset seeds --set T2 --approval FILE                   (re-derive every seed from A and R)
    python -m convlines.testset val --d 9 --chunks 0-19 --out DIR                (the validation shots used for timing)

SURF-v2 (the surface networks retrained without line supervision; surf_v2/REGISTRATION.md) adds three sets of the
same circuits, file format and chunk size, d = 7-17, each with its manifest in testsets/:
  sv2cal   calibration, seed 81,000,000 + 10,000 d + j, chunks 0-19 (200,000 shots per d): fitting tau and choosing q
  sv2dry   dry runs only, seed 89,000,000 + 10,000 d + j, chunks 0-3 (40,000 shots per d): never in a result
  sv2test  the test, chunks 0-199 at d = 7 and 0-99 at d = 9-17, seed(d, j) of convlines.beacon from the drand round
           fixed by the GitHub push of the freeze commit (beacon.json, checked again before any shot is drawn)
draw samples a set once on an x86-64 host with AVX2 (Stim 1.16.0, numpy 2.2.6), writes each chunk file once (never
over an existing file) and the set's manifest (file, d, j, seed, sha256; circuits, host, versions, the code's commit
and the beacon record's sha256) to --manifest, which is then committed as testsets/<set>.json before any read.
    python -m convlines.testset draw --set sv2cal --d all --out DIR --manifest DIR/sv2cal.json
    python -m convlines.testset draw --set sv2test --d all --beacon surf_v2/beacon.json --out DIR --manifest DIR/sv2test.json"""
import argparse
import hashlib
import io
import json
import os
import time
import zipfile

import numpy as np

from convlines import circuits

ROOT = circuits.ROOT
CHUNK = 10_000
FLOOR = 2 ** 62
SETS = {"T2": "testsets/T2.json", "T3": "testsets/T3.json", "T5": "testsets/T5.json",
        "sv2cal": "testsets/sv2cal.json", "sv2dry": "testsets/sv2dry.json", "sv2test": "testsets/sv2test.json"}
SV2_DISTANCES = (7, 9, 11, 13, 15, 17)
SV2 = {"sv2cal": dict(base=81_000_000, chunks={d: 20 for d in SV2_DISTANCES}),
       "sv2dry": dict(base=89_000_000, chunks={d: 4 for d in SV2_DISTANCES}),
       "sv2test": dict(base=None, chunks={d: 200 if d == 7 else 100 for d in SV2_DISTANCES})}
# validation shots (timing and development only): val_legacy at d = 7-13, V15 at d = 15, V17b at d = 17
# (seed = base + 10,000 d + j)
VAL_BASE = {7: 40_000_000, 9: 40_000_000, 11: 40_000_000, 13: 40_000_000, 15: 65_000_000, 17: 69_000_000}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def manifest(name):
    return json.load(open(os.path.join(ROOT, SETS[name])))


def parse_range(s):
    out = []
    for part in str(s).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def seed(tag, approval_bytes, randomness, d, j):
    h = hashlib.sha256(tag + approval_bytes + bytes.fromhex(randomness) + bytes([d]) + j.to_bytes(2, "big")).digest()
    return FLOOR + int.from_bytes(h[:8], "big") % FLOOR


def name(split, d, j):
    return f"d{d}_p{circuits.P}_X_{split}_{j:03d}.npz"


def write_npz(path, arrays):
    """np.savez's layout with deterministic bytes (stored members in name order, fixed timestamps), written once."""
    with open(path, "xb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o444 << 16
            z.writestr(zi, buf.getvalue())


def sample(c, s, shots=CHUNK):
    """{det, obs, seed} of one chunk."""
    det, obs = c.compile_detector_sampler(seed=s).sample(shots, separate_observables=True)
    return dict(det=np.packbits(det.astype(np.uint8), axis=1), obs=obs[:, 0].astype(np.uint8), seed=np.int64(s))


def load(path, d, sha256=None):
    """(det (n, m) uint8, obs (n,) uint8) of a chunk file, checked against sha256 when given."""
    raw = open(path, "rb").read()
    assert sha256 is None or sha256_bytes(raw) == sha256, f"{path}: sha256 differs from the manifest"
    z = np.load(io.BytesIO(raw))
    m = circuits.circuit(d).num_detectors
    return np.unpackbits(z["det"], axis=1)[:, :m], z["obs"].astype(np.uint8)


def load_chunks(name_, d, chunks, shot_dir):
    """Concatenated chunks of a test set, each hash-checked against the manifest."""
    man = manifest(name_)
    dets, obs = [], []
    for j in chunks:
        f = name(man["split"], d, j)
        det, ob = load(os.path.join(shot_dir, f), d, man["files"][f]["sha256"])
        dets.append(det)
        obs.append(ob)
    return np.concatenate(dets), np.concatenate(obs)


def regen(name_, d, chunks, out):
    man = manifest(name_)
    c = circuits.circuit(d)
    assert man["circuit_sha256"][str(d)] == circuits.CIRCUIT_SHA256[d]
    os.makedirs(out, exist_ok=True)
    ok = True
    for j in chunks:
        f = name(man["split"], d, j)
        e = man["files"][f]
        path = os.path.join(out, f)
        write_npz(path, sample(c, e["seed"]))
        got = sha256_bytes(open(path, "rb").read())
        same = got == e["sha256"]
        ok &= same
        print(f"{f}: seed {e['seed']} sha256 {got} {'== manifest' if same else '!= manifest ' + e['sha256']}", flush=True)
    return ok


def check(name_, shot_dir):
    man = manifest(name_)
    have = sorted(f for f in os.listdir(shot_dir) if f in man["files"])
    bad = [f for f in have if sha256_bytes(open(os.path.join(shot_dir, f), "rb").read()) != man["files"][f]["sha256"]]
    print(f"{len(have)} of {len(man['files'])} {name_} chunk files present in {shot_dir}; {len(bad)} differ: {bad}")
    return not bad


def seeds(name_, approval_path):
    man = manifest(name_)
    if name_ in SV2:                                    # SURF-v2: the base, or the beacon's randomness in the manifest
        b = None
        if name_ == "sv2test":
            from convlines import beacon as B
            b = B.load(approval_path)                   # the beacon record (checked) in place of an approval record
            assert sha256_bytes(open(approval_path, "rb").read()) == man["beacon"]["record_sha256"], \
                "not the beacon record the manifest names"
        bad = [f for f, e in man["files"].items() if sv2_seed(name_, e["d"], e["j"], b) != e["seed"]]
        print(f"{len(man['files']) - len(bad)} of {len(man['files'])} seeds re-derived; differing: {bad}")
        return not bad
    appr = open(approval_path, "rb").read()
    assert sha256_bytes(appr) == man["approval_sha256"], "not the approval record the manifest names"
    b = man["beacon"]
    assert sha256_bytes(bytes.fromhex(b["signature"])) == b["randomness"], "randomness is not sha256(signature)"
    tag = man["tag"].encode()
    bad = [f for f, e in man["files"].items() if seed(tag, appr, b["randomness"], e["d"], e["j"]) != e["seed"]]
    print(f"{len(man['files']) - len(bad)} of {len(man['files'])} seeds re-derived; differing: {bad}")
    return not bad


def val(d, chunks, out):
    """Validation chunks (not test data): seeds VAL_BASE[d] + 10,000 d + j, the same sampler and format."""
    c = circuits.circuit(d)
    os.makedirs(out, exist_ok=True)
    for j in chunks:
        write_npz(os.path.join(out, name("val", d, j)), sample(c, VAL_BASE[d] + 10_000 * d + j))
    print(f"d{d}: {len(chunks)} validation chunks in {out}")


def sv2_seed(name_, d, j, beacon=None):
    """The seed of chunk j at distance d of a SURF-v2 set: base + 10,000 d + j, or the beacon's for sv2test."""
    assert 0 <= j < SV2[name_]["chunks"][d], f"{name_} has no chunk {j} at d = {d}"
    if name_ == "sv2test":
        from convlines import beacon as B
        return B.seed(beacon["randomness"], d, j)
    return SV2[name_]["base"] + 10_000 * d + j


def draw(name_, ds, out, manifest_path, beacon_path=None):
    """Draw every chunk of set name_ at the distances ds once into out, and write the set's manifest (never over an
    existing file). sv2test needs the beacon record, checked first (convlines.beacon.load)."""
    import stim
    from convlines import provenance as P
    h = P.host()
    assert h["machine"] == "x86_64" and "avx2" in h["simd"], f"the sets are drawn on x86-64 with AVX2 only: {h}"
    assert (stim.__version__, np.__version__) == ("1.16.0", "2.2.6"), (stim.__version__, np.__version__)
    assert not os.path.exists(manifest_path), f"{manifest_path} exists"
    b = None
    if name_ == "sv2test":
        from convlines import beacon as B
        assert beacon_path, "sv2test needs --beacon"
        b = B.load(beacon_path)
    else:
        assert not beacon_path, f"{name_}'s seeds do not come from a beacon"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(out, exist_ok=True)
    files = {}
    for d in ds:
        c = circuits.circuit(d)
        for j in range(SV2[name_]["chunks"][d]):
            f, s = name(name_, d, j), sv2_seed(name_, d, j, b)
            write_npz(os.path.join(out, f), sample(c, s))
            files[f] = dict(d=d, j=j, seed=s, sha256=sha256_bytes(open(os.path.join(out, f), "rb").read()))
        print(f"{name_} d{d}: {SV2[name_]['chunks'][d]} chunks drawn", flush=True)
    man = dict(test_set=name_, split=name_, what="SURF-v2 " + {"sv2cal": "calibration set (fitting tau and choosing q)",
               "sv2dry": "dry set (dry runs only; never in a result)", "sv2test": "test set"}[name_],
               p=circuits.P, basis="X", chunk_shots=CHUNK, chunks={str(d): f"0-{SV2[name_]['chunks'][d] - 1}" for d in ds},
               shots={str(d): CHUNK * SV2[name_]["chunks"][d] for d in ds},
               circuit_sha256={str(d): circuits.CIRCUIT_SHA256[d] for d in ds},
               seed_rule=("seed(d, j) = convlines.beacon.seed(beacon.randomness, d, j): " + b["seed_rule"]) if b else
               f"{SV2[name_]['base']:,} + 10,000 d + j", files=files, versions=dict(stim=stim.__version__, numpy=np.__version__),
               host=h, release_commit=P.commit(), code_digest=P.code_digest(), started=started,
               ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if b:
        man["beacon"] = dict(record_sha256=sha256_bytes(open(beacon_path, "rb").read()), round=b["round"],
                             round_time_unix=b["round_time"], randomness=b["randomness"], signature=b["signature"],
                             freeze_commit=b["freeze_commit"], push_time_unix=b["push_time"], tags=b["tags"])
    with open(manifest_path, "x") as fh:
        json.dump(man, fh, indent=1)
    print(f"{len(files)} chunk files; manifest {manifest_path} sha256 {sha256_bytes(open(manifest_path, 'rb').read())}")
    return True


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("regen")
    r.add_argument("--set", required=True, choices=sorted(SETS))
    r.add_argument("--d", type=int, required=True)
    r.add_argument("--chunks", required=True, help="e.g. 0,99 or 0-99")
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--set", required=True, choices=sorted(SETS))
    c.add_argument("--dir", required=True)
    s = sub.add_parser("seeds")
    s.add_argument("--set", required=True, choices=sorted(SETS))
    s.add_argument("--approval", default="", help="the approval record (sv2test: the beacon record; sv2cal, sv2dry: none)")
    v = sub.add_parser("val")
    v.add_argument("--d", type=int, required=True)
    v.add_argument("--chunks", default="0-19")
    v.add_argument("--out", required=True)
    w = sub.add_parser("draw")
    w.add_argument("--set", required=True, choices=sorted(SV2))
    w.add_argument("--d", default="all", help="all, or a comma list of distances")
    w.add_argument("--beacon", default="", help="sv2test: the beacon record (convlines.beacon fetch)")
    w.add_argument("--out", required=True)
    w.add_argument("--manifest", required=True, help="where the manifest is written (then committed to testsets/)")
    a = ap.parse_args()
    if a.cmd == "regen":
        ok = regen(a.set, a.d, parse_range(a.chunks), a.out)
    elif a.cmd == "check":
        ok = check(a.set, a.dir)
    elif a.cmd == "seeds":
        ok = seeds(a.set, a.approval)
    elif a.cmd == "draw":
        ds = SV2_DISTANCES if a.d == "all" else tuple(int(x) for x in a.d.split(","))
        ok = draw(a.set, ds, a.out, a.manifest, a.beacon or None)
    else:
        val(a.d, parse_range(a.chunks), a.out)
        ok = True
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
