"""SPEED-RT stage 3 (STAGE3_batched.md section 3): the CPU session's inputs in the tmpfs, untimed, before the first window.

    python -m speed_rt.s3.prep3 --surf SHOTS --bb SHOTS --val VAL --decide DIR --manifests DIR --data DATA

DATA/shots/<point>/: the point's timing chunks (the draw's files, each sha256-checked against the manifests);
DATA/decide/: the decision passes' decide_<point>.npz (sha256-checked against decide_<point>.json);
DATA/warm/warm_<point>.npy: the last 20 rows of the point's validation chunk (VAL/<point>/, stage 2's chunk 0), the
decoders' untimed warm-up rows."""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(os.path.dirname(HERE))
POINTS = {"surf-d7": "s3-surf-d7", "surf-d9": "s3-surf-d9", "surf-d11": "s3-surf-d11", "surf-d13": "s3-surf-d13",
          "surf-d15": "s3-surf-d15", "surf-d17": "s3-surf-d17", "bb72-p2": "s3-bb72-p2", "bb72-p3": "s3-bb72-p3",
          "gross-p4": "s3-gross-p4"}
BBSET = {("bb72", "0.002"): "s3-bb72-p2", ("bb72", "0.003"): "s3-bb72-p3", ("bb144", "0.004"): "s3-gross-p4"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("surf", "bb", "val", "decide", "manifests", "data"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    man = {}
    m = json.load(open(os.path.join(a.manifests, "srt-s3-surf.json")))
    for f, e in m["files"].items():
        man[f] = (e["set"], e["sha256"], a.surf)
    for path in sorted(glob.glob(os.path.join(a.manifests, "manifest_s3_*.csv"))):
        for r in csv.DictReader(open(path)):
            man[r["file"]] = (BBSET[(r["code"], r["p"])], r["sha256"], a.bb)
    for pt, st in POINTS.items():
        od = os.path.join(a.data, "shots", pt)
        os.makedirs(od, exist_ok=True)
        n = 0
        for f, (s, h, src) in sorted(man.items()):
            if s != st:
                continue
            shutil.copyfile(os.path.join(src, f), os.path.join(od, f))
            assert sha256_file(os.path.join(od, f)) == h, f"{f}: sha256 differs from the manifest"
            n += 1
        dj = os.path.join(a.decide, f"decide_{pt}.json")
        if os.path.exists(dj):
            d = json.load(open(dj))
            os.makedirs(os.path.join(a.data, "decide"), exist_ok=True)
            shutil.copyfile(os.path.join(a.decide, f"decide_{pt}.npz"), os.path.join(a.data, "decide", f"decide_{pt}.npz"))
            assert sha256_file(os.path.join(a.data, "decide", f"decide_{pt}.npz")) == d["npz_sha256"], pt
            assert sorted(tuple(x) for x in d["files"]) == sorted((f, h) for f, (s, h, _) in man.items() if s == st), pt
        vf = sorted(glob.glob(os.path.join(a.val, pt, "*.npz")))[0]
        if pt.startswith("surf"):
            sys.path.insert(0, os.path.join(REL, "surface_v2"))
            from convlines import testset
            x, _ = testset.load(vf, int(pt.split("-d")[1]))
        else:
            z = np.load(vf)
            x = np.unpackbits(z["det"], axis=1, count=int(z["D"]))[:int(z["n"])]
        os.makedirs(os.path.join(a.data, "warm"), exist_ok=True)
        np.save(os.path.join(a.data, "warm", f"warm_{pt}.npy"), np.ascontiguousarray(x[-20:], np.uint8))
        print(f"{pt}: {n} chunks, decide {'yes' if os.path.exists(dj) else 'no'}, warm from {os.path.basename(vf)}")


if __name__ == "__main__":
    main()
