"""SPEED-RT's draw of the timing sets (SPEED_RT.md, "The timing shots"), once, on Alpine (x86-64 with AVX2, Stim
1.16.0, numpy 2.2.6, as every earlier test set): the beacon record checked first, every chunk sampled from its seed
(speed_rt/sets.py) with the samplers and file formats of the original tests, each file written once, and one manifest
per kind with every chunk's set, index, seed and sha256.

  surface  srt-surf-d<d>: Stim's detector sampler on the verdict circuit of d (surface_v2/convlines: circuits.circuit,
           testset.sample and write_npz), chunk j as testset.name("srt", d, j)
  ref      srt-ref-d<d>-p<p>: the Grace fit's reference points, one chunk of 10,000 shots each: the same sampler on NVIDIA's
           legacy circuit of (d, p) (speed_rt/circuits/ref_circuits.json, speed_rt/gref.py), chunk file gref.shots_file
The sets are those of the surface stage's sets file (speed_rt/sets_surf.json; --sets-file); the stage's beacon record
names it, or names stage 2's union sets file (speed_rt/sets_s2.json), which must hold these sets unchanged. The BB sets (speed_rt/bb/draw.sh) and the Cascade sets (no files) belong to their own stages.

    python -m speed_rt.draw --beacon speed_rt/beacon.json --kind surface --out SHOTS --manifest OUT/srt-surf.json
    python -m speed_rt.draw --beacon speed_rt/beacon.json --kind ref --out REFSHOTS --manifest OUT/srt-ref.json
    python -m speed_rt.draw --beacon DRY.json --dry --kind surface --out DRYSHOTS --manifest DRYOUT/srt-surf.json
    python -m speed_rt.draw view --shots SHOTS --out VIEW [--ds 7,9,11]   (the timing modules' --val directory: one
                                  symbolic link per timing chunk, named as the validation chunk it replaces; --ds: a
                                  pod role's distances)
    python -m speed_rt.draw check --dir DIR --manifest M [--ds 7,9,11] [--beacon B [--dry]]   (every chunk of the manifest
                                  present with its sha256; with --beacon, every seed recomputed from the record)

The surface timing modules (surface_v2/timing: gpu_term, deployed, cpu_term, ising_trt) read the 20 validation chunks of
a distance from a directory by file name. VIEW gives each chunk of srt-surf-d<d> the name of validation chunk j, so the
modules run unchanged on the timing set; the digest of the shots that every routing mask records binds their records
to it."""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "surface_v2"))

from speed_rt import beacon, sets  # noqa: E402

SETS_FILE = os.path.join(HERE, "sets_surf.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def draw_surface(S, seeds, todo, out):
    """The chunks of the surface sets (S: the stage's sets); per d also the digest of the whole set in chunk order
    (timing.deployed's shots_sha256, the digest every routing mask records and every timed process checks), which binds
    the session's records to this draw."""
    import numpy as np
    from convlines import circuits, testset
    from timing import deployed
    files, digest, prefix = {}, {}, {}
    n6 = json.load(open(os.path.join(HERE, "config_surf.json")))["items"]["G6"]["shots"]
    for s in todo:
        e = S[s]
        c = circuits.circuit(e["d"])
        for j, sd in enumerate(seeds[s]):
            f = testset.name("srt", e["d"], j)
            testset.write_npz(os.path.join(out, f), testset.sample(c, sd, e["chunk_shots"]))
            files[f] = dict(set=s, d=e["d"], j=j, seed=sd, sha256=sha256_file(os.path.join(out, f)))
        sh = [testset.load(os.path.join(out, testset.name("srt", e["d"], j)), e["d"]) for j in range(e["chunks"])]
        digest[str(e["d"])] = deployed.shots_sha256(np.concatenate([x[0] for x in sh]), np.concatenate([x[1] for x in sh]))
        prefix[str(e["d"])] = deployed.shots_sha256(sh[0][0][:n6], sh[0][1][:n6])          # G6's shots (BP-OSD)
        print(f"{s}: {e['chunks']} chunks, shots_sha256 {digest[str(e['d'])][:16]}", flush=True)
    return files, dict(circuit_sha256={str(S[s]["d"]): circuits.CIRCUIT_SHA256[S[s]["d"]] for s in todo},
                       shots_sha256=digest, chunk0_prefix=dict(shots=n6, sha256=prefix))


def draw_ref(S, seeds, todo, out):
    """The reference sets (S: the stage's sets): one chunk each, on NVIDIA's legacy circuit of (d, p), with the
    circuit's sha256 (speed_rt/circuits/ref_circuits.json) and the digest of the shots (timing.deployed.shots_sha256)."""
    import numpy as np
    from convlines import testset
    from timing import deployed
    from speed_rt import gref
    files, digest, circ = {}, {}, {}
    for s in todo:
        e = S[s]
        assert e["chunks"] == 1, f"{s}: a reference set is one chunk"
        c, csha = gref.circuit(e["d"], e["p"])
        f = gref.shots_file(e["d"], e["p"])
        arrays = testset.sample(c, seeds[s][0], e["chunk_shots"])
        testset.write_npz(os.path.join(out, f), arrays)
        files[f] = dict(set=s, d=e["d"], p=e["p"], j=0, seed=seeds[s][0], sha256=sha256_file(os.path.join(out, f)))
        digest[s] = deployed.shots_sha256(np.unpackbits(arrays["det"], axis=1)[:, :c.num_detectors], arrays["obs"])
        circ[s] = dict(d=e["d"], p=e["p"], circuit_sha256=csha, detectors=int(c.num_detectors))
        print(f"{s}: {e['chunk_shots']} shots, circuit {csha[:12]}, shots_sha256 {digest[s][:16]}", flush=True)
    return files, dict(circuits=circ, shots_sha256=digest)


def view(S, shots, out, ds=None):
    from convlines import testset
    os.makedirs(out, exist_ok=False)
    n = 0
    for s, e in sorted(S.items()):
        if e["kind"] != "surface" or (ds is not None and e["d"] not in ds):
            continue
        for j in range(e["chunks"]):
            src = os.path.join(os.path.abspath(shots), testset.name("srt", e["d"], j))
            assert os.path.exists(src), f"{src} is missing"
            os.symlink(src, os.path.join(out, testset.name("val", e["d"], j)))
            n += 1
    print(f"VIEW {out}: {n} links")


def check(dirname, manifest, ds=None, beacon_path=None, sets_file=SETS_FILE, dry=False):
    """Every chunk of the manifest (of the distances ds, when given) present in dirname with the manifest's sha256, and no
    other chunk file; with a beacon record, also every chunk's seed recomputed from it (speed_rt/sets.py)."""
    man = json.load(open(manifest))
    want = {f: e for f, e in sorted(man["files"].items()) if ds is None or e["d"] in ds}
    bad = [f for f, e in want.items()
           if not os.path.exists(os.path.join(dirname, f)) or sha256_file(os.path.join(dirname, f)) != e["sha256"]]
    extra = sorted(set(os.listdir(dirname)) - set(want))
    assert want and not bad and not extra, f"{len(bad)} chunks missing or different ({bad[:3]}), extra files {extra[:3]}"
    if beacon_path:
        b = beacon.load(beacon_path, dry_ok=dry)
        seeds = sets.all_seeds(b["randomness"], sets_file)
        wrong = [f for f, e in man["files"].items() if seeds[e["set"]][e["j"]] != e["seed"]]
        assert not wrong and man["beacon"]["randomness"] == b["randomness"], f"seeds differ from the record: {wrong[:3]}"
    print(f"CHECK {dirname}: {len(want)} chunks equal to {os.path.basename(manifest)}"
          + ("; every seed recomputed from the beacon record" if beacon_path else ""))


def ints(text):
    return None if not text else [int(x) for x in text.split(",")]


def main():
    if sys.argv[1:2] == ["check"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("what")
        ap.add_argument("--dir", required=True)
        ap.add_argument("--manifest", required=True)
        ap.add_argument("--ds", default="")
        ap.add_argument("--beacon", default="", help="recompute every seed from this record")
        ap.add_argument("--dry", action="store_true", help="the beacon record is a dry one")
        ap.add_argument("--sets-file", default=SETS_FILE)
        a = ap.parse_args()
        return check(a.dir, a.manifest, ints(a.ds), a.beacon or None, a.sets_file, a.dry)
    if sys.argv[1:2] == ["view"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("what")
        ap.add_argument("--shots", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--ds", default="")
        ap.add_argument("--sets-file", default=SETS_FILE)
        a = ap.parse_args()
        return view(sets.sets(a.sets_file), a.shots, a.out, ints(a.ds))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beacon", required=True)
    ap.add_argument("--dry", action="store_true", help="a dry record (a past round): dry runs only")
    ap.add_argument("--kind", required=True, choices=("surface", "ref"))
    ap.add_argument("--sets-file", default=SETS_FILE, help="the stage's sets file (default speed_rt/sets_surf.json)")
    ap.add_argument("--sets", nargs="*", default=None, help="a subset of the kind's sets (dry runs only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    import numpy as np
    import stim
    from convlines import provenance as P
    h = P.host()
    assert h["machine"] == "x86_64" and "avx2" in h["simd"], f"the sets are drawn on x86-64 with AVX2 only: {h}"
    assert (stim.__version__, np.__version__) == ("1.16.0", "2.2.6"), (stim.__version__, np.__version__)
    assert not os.path.exists(a.manifest), f"{a.manifest} exists"
    b = beacon.load(a.beacon, dry_ok=a.dry)
    assert bool(b.get("dry")) == a.dry, "--dry and the record disagree"
    S = sets.sets(a.sets_file)
    rel = os.path.relpath(os.path.abspath(a.sets_file), ROOT)
    if not a.dry and b.get("sets_file") != rel:    # stage 2's one record names the union file speed_rt/sets_s2.json (STAGE2_common C8)
        assert b.get("sets_file"), "the beacon record names no sets file"
        U = sets.sets(os.path.join(ROOT, b["sets_file"]))
        assert all(U.get(s) == e for s, e in S.items()), f"the beacon record names {b['sets_file']}, which does not hold {rel}'s sets unchanged"
    todo = sorted(s for s, e in S.items() if e["kind"] == a.kind)
    if a.sets is not None:
        assert a.dry and set(a.sets) <= set(todo), "a subset of sets is for dry runs only"
        todo = sorted(a.sets)
    seeds = sets.all_seeds(b["randomness"], a.sets_file)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(a.out, exist_ok=True)
    files, extra = (draw_surface if a.kind == "surface" else draw_ref)(S, seeds, todo, a.out)
    man = dict(what=f"SPEED-RT timing sets, kind {a.kind}" + (" (DRY: never a timing input)" if a.dry else ""),
               dry=a.dry, sets=todo, sets_file=rel, sets_file_sha256=sha256_file(a.sets_file), rule=sets.RULE,
               tags={s: sets.tag(s).decode() for s in todo},
               beacon=dict(record_sha256=sha256_file(a.beacon), round=b["round"], randomness=b["randomness"],
                           freeze_commit=b.get("freeze_commit"), push_time_unix=b.get("push_time")),
               files=files, versions=dict(stim=stim.__version__, numpy=np.__version__), host=h,
               release_commit=P.commit(os.path.join(ROOT)), started=started,
               ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    with open(a.manifest, "x") as fh:
        json.dump(man, fh, indent=1)
    print(f"{len(files)} files; manifest {a.manifest} sha256 {sha256_file(a.manifest)}")


if __name__ == "__main__":
    main()
