"""The Decoder Switching stage's draw (speed_rt/gapsw/sets_gs.json), once, on Alpine (x86-64 with AVX2, Stim 1.16.0, numpy
2.2.6, as every earlier draw): the stage's beacon record checked first, every chunk sampled from its seed (speed_rt/sets.py
with this stage's sets file) with SURF-v2's sampler and file format (convlines.testset.sample, write_npz), each file written
once, and one manifest with every chunk's set, index, seed and sha256, the digest of every set and every circuit's sha256.

    python -m speed_rt.gapsw.draw_gs --beacon speed_rt/gapsw/beacon.json --out SHOTS --manifest OUT/srt-gs.json
    python -m speed_rt.gapsw.draw_gs --beacon DRY.json --dry --out SHOTS --manifest OUT/srt-gs.json [--sets NAME...] [--chunks N]
    python -m speed_rt.gapsw.draw_gs check --dir SHOTS --manifest OUT/srt-gs.json     (every file present with its sha256, no other)

--sets and --chunks (a subset of sets, the first N chunks of each set) are for dry runs only. The draw holds the six DS sets srt-gs-d7 ... d17
only: the stage's Grace reference sets are the surface family's (speed_rt/sets_surf.json)."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from speed_rt.gapsw import gslib as G  # noqa: E402


def check(dirname, manifest):
    man = json.load(open(manifest))
    bad = [f for f, e in sorted(man["files"].items())
           if not os.path.exists(os.path.join(dirname, f)) or G.sha256_file(os.path.join(dirname, f)) != e["sha256"]]
    extra = sorted(set(os.listdir(dirname)) - set(man["files"]))
    assert not bad and not extra, f"{len(bad)} chunks missing or different ({bad[:3]}), extra files {extra[:3]}"
    print(f"CHECK {os.path.basename(os.path.normpath(dirname))}: {len(man['files'])} chunks equal to {os.path.basename(manifest)}")


def main():
    if sys.argv[1:2] == ["check"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("what")
        ap.add_argument("--dir", required=True)
        ap.add_argument("--manifest", required=True)
        a = ap.parse_args()
        return check(a.dir, a.manifest)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beacon", required=True)
    ap.add_argument("--dry", action="store_true", help="a dry record (a past round): dry runs only")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    import numpy as np
    import stim
    from convlines import provenance as P
    from convlines import testset
    from speed_rt import beacon, sets
    h = P.host()
    assert h["machine"] == "x86_64" and "avx2" in h["simd"], f"the sets are drawn on x86-64 with AVX2 only: {h}"
    assert (stim.__version__, np.__version__) == ("1.16.0", "2.2.6"), (stim.__version__, np.__version__)
    assert not os.path.exists(a.manifest), f"{a.manifest} exists"
    assert a.dry or (a.sets is None and a.chunks is None), "--sets and --chunks are for dry runs only"
    b = beacon.load(a.beacon, dry_ok=a.dry)
    assert bool(b.get("dry")) == a.dry, "--dry and the record disagree"
    allsets = G.sets()
    todo = sorted(allsets) if a.sets is None else sorted(a.sets)
    assert set(todo) <= set(allsets), "unknown set"
    seeds = sets.all_seeds(b["randomness"], G.SETS_FILE)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(a.out, exist_ok=True)
    files, digest, circ = {}, {}, {}
    for s in todo:
        e = allsets[s]
        d, p = e["d"], e["p"]
        c = G.circuit(d, p)
        n = e["chunks"] if a.chunks is None else a.chunks     # dry: fewer chunks only
        dets, obs = [], []
        for j in range(n):
            f = G.ds_file(d, j)
            z = testset.sample(c, seeds[s][j], e["chunk_shots"])
            testset.write_npz(os.path.join(a.out, f), z)
            files[f] = dict(set=s, d=d, p=p, j=j, seed=seeds[s][j], sha256=G.sha256_file(os.path.join(a.out, f)))
            det, ob = G.load_chunk(os.path.join(a.out, f), c.num_detectors)
            dets.append(det)
            obs.append(ob)
        digest[s] = G.shots_sha256(np.concatenate(dets), np.concatenate(obs))
        circ[s] = G.circuit_sha256(d, p)
        print(f"{s}: {n} chunks, shots_sha256 {digest[s][:16]}", flush=True)
    man = dict(what="SPEED-RT Decoder Switching stage: timing sets" + (" (DRY: never a timing input)" if a.dry else ""),
               dry=a.dry, sets=todo, rule=sets.RULE, tags={s: sets.tag(s).decode() for s in todo},
               beacon=dict(record_sha256=G.sha256_file(a.beacon), round=b["round"], randomness=b["randomness"],
                           freeze_commit=b.get("freeze_commit"), push_time_unix=b.get("push_time")),
               files=files, shots_sha256=digest, circuit_sha256=circ, versions=dict(stim=stim.__version__, numpy=np.__version__),
               host=h, release_commit=P.commit(G.ROOT), started=started, ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(a.manifest, "x") as fh:
        json.dump(man, fh, indent=1)
    print(f"{len(files)} files; manifest {os.path.basename(a.manifest)} sha256 {G.sha256_file(a.manifest)}")


if __name__ == "__main__":
    main()
