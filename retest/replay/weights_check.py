"""Every network file a re-test read loads against its copy in this repository (RETEST.md section 8): equal bytes
(sha256), or, where the repository holds a copy in another container (an inference copy, safetensors, a checkpoint
without training state), equal tensors: the copy's floating-point tensors and those of one entry of the read file
(the EMA weights, say) correspond one to one by name, shape, dtype and bytes (the entry is recorded).
    python weights_check.py PAIRS_JSON PUBLIC_ROOT OUT_JSON"""
import hashlib
import json
import sys

import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
      "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def safetensors(path):
    """The safetensors format read directly (8-byte little-endian header length, JSON header, raw little-endian data),
    so that no extra package is needed."""
    blob = open(path, "rb").read()
    n = int.from_bytes(blob[:8], "little")
    head, base = json.loads(blob[8:8 + n]), 8 + n
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        t = torch.frombuffer(bytearray(blob[base + a:base + b]), dtype=torch.uint8)
        out[k] = t.view(DT[v["dtype"]]).reshape(v["shape"]) if b > a else torch.empty(v["shape"], dtype=DT[v["dtype"]])
    return out


def tensors(path):
    """name -> tensor of every tensor in the file (nested dicts flattened with '/'; a 'module.' prefix dropped)."""
    if path.endswith(".safetensors"):
        obj = safetensors(path)
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    out = {}

    def walk(x, pre):
        if isinstance(x, torch.Tensor):
            out[pre] = x
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{pre}/{k}" if pre else str(k))
    walk(obj, "")
    return out


def raw(t):
    """The tensor's bytes, as uint8 (bit-for-bit comparison, NaN included)."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def leaf(name):
    return name.split("/")[-1].removeprefix("module.").removeprefix("_orig_mod.")


def compare(read, pub):
    """Equal iff, for one top-level entry of the read file (its whole content if flat), its floating-point tensors and
    the copy's correspond one to one by name (prefixes dropped), shape, dtype and bytes."""
    a, b = tensors(read), tensors(pub)
    fb = {leaf(k): v for k, v in b.items() if v.is_floating_point()}
    groups = {}
    for k, v in a.items():
        if v.is_floating_point():
            groups.setdefault(k.split("/")[0] if "/" in k else "", {})[leaf(k)] = v
    rep = {"copy_tensors": len(fb), "read_groups": {g: len(x) for g, x in groups.items()}, "equal": False}
    for g, x in groups.items():
        if set(x) == set(fb) and all(x[k].shape == v.shape and x[k].dtype == v.dtype and torch.equal(raw(x[k]), raw(v))
                                     for k, v in fb.items()):
            rep.update(equal=True, read_entry=g or "(flat)")
            break
    return rep


if __name__ == "__main__":
    pairs, root, rep = json.load(open(sys.argv[1])), sys.argv[2], []
    for x in pairs:
        r = dict(x)
        r["read_sha256"], r["public_sha256"] = sha(x["read"]), sha(f"{root}/{x['public']}")
        if r["read_sha256"] == r["public_sha256"]:
            r["check"], r["ok"] = "bytes equal", True
        else:
            r.update(compare(x["read"], f"{root}/{x['public']}"))
            r["check"], r["ok"] = "tensors", r["equal"]
        rep.append(r)
        print(r["name"], r["check"], r["ok"], flush=True)
    json.dump(rep, open(sys.argv[3], "w"), indent=1)
    print("ALL_OK" if all(x["ok"] for x in rep) else "NOT_ALL_OK", flush=True)
