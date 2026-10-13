"""Compare an original D1 / X2 output with a replay's:
    python rt_d1_compare.py json A B OUT.json     the loaded JSON values equal; byte-identical; up to 20 differing paths
    python rt_d1_compare.py text A B OUT.json     the files are byte-identical
    python rt_d1_compare.py npz A B OUT.json      every array of the .npz files equal (NaN equal), the file bytes, and the sha256s
Exit status 1 when the values differ."""
import hashlib
import json
import sys

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def diffs(a, b, path="", out=None):
    out = [] if out is None else out
    if len(out) >= 20:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing in {'A' if k not in a else 'B'}")
            else:
                diffs(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diffs(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} != {str(b)[:60]}")
    return out


def main():
    mode, pa, pb, out = sys.argv[1:5]
    r = {"mode": mode, "A": pa, "B": pb, "sha256_A": sha(pa), "sha256_B": sha(pb)}
    r["bytes_identical"] = r["sha256_A"] == r["sha256_B"]
    if mode == "json":
        a, b = json.load(open(pa)), json.load(open(pb))
        r["values_equal"] = a == b
        r["differences"] = diffs(a, b)
        r["pass"] = r["values_equal"]
    elif mode == "text":
        r["pass"] = r["bytes_identical"]
    else:
        za, zb = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
        r["keys_equal"] = sorted(za.files) == sorted(zb.files)
        r["arrays_equal"] = {k: bool(np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f")) for k in za.files
                             if k in zb.files}
        r["pass"] = r["keys_equal"] and all(r["arrays_equal"].values())
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("mode", "pass", "bytes_identical")}))
    sys.exit(0 if r["pass"] else 1)


main()
