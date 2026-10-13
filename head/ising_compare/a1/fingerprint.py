"""A1 on rented hosts (revision/A1_POD_RUN.md, gate finding 4): this host's sampler fingerprint, taken before training.
Stim's output for a seed may depend on the CPU and the runtime, so the arms of one seed train only on hosts whose
fingerprints are equal, and each run records its own stream (arch.train's and bb.train's stream_chain) so the equality
can be shown afterwards.
  surface  the first 4 pools (16 chunks of 8,192 shots) of seed S's training stream (arch.stream.LineStream, stream id
           a1.seeds.surface_stream(S)), chained as arch.train's stream record: its value equals the STREAM line of
           every surface run of seed S that started on this host;
  bb72     the first 16 chunks of seed S's BB72 stream (bb.train.Stream, p = 0.3 %, stream id a1.seeds.bb_stream(S)),
           chained as bb.train's stream record (its STREAM line);
  sets     the sha256 of the validation files the stage-1 records read (d = 9 val_legacy chunks 0-19; BB72 dev p = 0.3 %
           chunks 0-19 and monitor chunk 0), each also checked against the repository's manifests by the loaders;
  host     the CPU model and the flags that select Stim's SIMD path, the Stim, NumPy and Python versions.
'fingerprint' = sha256 of the surface, bb72 and sets values (not host). Output: one JSON line.
    python -m a1.fingerprint --seed 1 [--workers 2]"""
import argparse
import hashlib
import json
import os
import platform

import numpy as np

from a1 import seeds as a1seeds

CHUNKS = range(20)


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def surface_chain(seed, workers, pools=4, pool=32 * 1024):
    from arch.stream import LineStream
    from arch.train import chain_next
    src = LineStream(9, a1seeds.surface_stream(seed), workers)
    chain = "0" * 64
    try:
        for _ in range(pools):
            X, Y, _ = src.next(pool)
            chain = chain_next(chain, X, Y)
    finally:
        src.close()
    return chain


def bb_chain(seed, workers, chunks=16):
    from bb.train import Stream
    src = Stream("bb72", 0.003, a1seeds.bb_stream(seed), workers, 2e8, None)
    chain = "0" * 64
    try:
        for i in range(chunks):
            c, d, o = src.next()
            assert c == i
            chain = hashlib.sha256(bytes.fromhex(chain) + d.tobytes() + o.tobytes()).hexdigest()
    finally:
        src.close()
    return chain


def sets_sha():
    from bb import shots as bbs
    from icmp import shots
    shots.load(shots.VAL, 9, CHUNKS)                         # sha-checked against data/shot_manifest.csv
    bbs.load_many("bb72", 0.003, "dev", CHUNKS)              # sha-checked against data/bb/shot_manifest.csv
    bbs.load("bb72", 0.003, "monitor", 0)
    files = [os.path.join(shots.SHOT_DIR, shots.name(shots.VAL, 9, j)) for j in CHUNKS]
    files += [os.path.join(bbs.SHOTS, bbs.name("bb72", 0.003, "dev", j)) for j in CHUNKS]
    files += [os.path.join(bbs.SHOTS, bbs.name("bb72", 0.003, "monitor", 0))]
    return {os.path.basename(f): sha_file(f) for f in files}


def host():
    import stim
    flags = set()
    model = ""
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name") and not model:
            model = line.split(":", 1)[1].strip()
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
    return {"cpu": model, "simd": sorted(f for f in flags if f in ("sse2", "avx", "avx2", "avx512f", "avx512bw")),
            "stim": stim.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "node": platform.node()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    rec = {"seed": a.seed, "surface_chain_16": surface_chain(a.seed, a.workers),
           "bb72_chain_16": bb_chain(a.seed, a.workers)}
    ss = sets_sha()
    rec["sets_sha256"] = hashlib.sha256(json.dumps(ss, sort_keys=True).encode()).hexdigest()
    rec["fingerprint"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
    rec["host"] = host()
    rec["sets"] = ss
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
