"""Runs on an Alpine login node (Python 3 standard library only, no numpy there). From the re-test's eager read of D1
(retest/out/d1/<p>/read/p<p>.npz, stored uncompressed), writes a tar stream to stdout with, per p:
  <tag>/all_{fail0,fail1,conf1}.npy   every shot, copied byte for byte from the read's members
  <tag>/sel_<member>.npy              the routed shots whose two decisions differ (conf1 in float64 < D1's tau1 and
                                      packed pred0 != pred1), every per-shot member of the read except nothing
  <tag>/sel_rows.npy                  their shot indices in the read (int64)
  <tag>/info.json                     the read's sha256, the member list, counts
The selection is arb_scores.py's: routed = conf1.astype(float64) < tau1, differ = (pred0 != pred1).any(1).

  python3 alpine_extract_bb.py <cluster-scratch>/retest/out/d1 > bb_sel.tar"""
import array
import ast
import hashlib
import io
import json
import struct
import sys
import tarfile
import zipfile

TAU1 = {"0.002": 7.3331888389587405, "0.003": 7.60439167022705}
ALL = ("fail0", "fail1", "conf1")
SEL = ("det", "z0", "pred0", "pred1", "obs", "conf0", "conf1", "cls1", "fail0", "fail1", "chunk", "row")


def members(path):
    """name -> (absolute offset of the array data, descr, shape, npy header bytes) for an uncompressed npz."""
    out = {}
    with zipfile.ZipFile(path) as zf, open(path, "rb") as fh:
        for info in zf.infolist():
            assert info.compress_type == 0, info.filename
            fh.seek(info.header_offset)
            lh = fh.read(30)
            assert lh[:4] == b"PK\x03\x04"
            n, m = struct.unpack("<HH", lh[26:30])
            start = info.header_offset + 30 + n + m
            fh.seek(start)
            magic = fh.read(8)
            assert magic[:6] == b"\x93NUMPY"
            hl = struct.unpack("<H", fh.read(2))[0] if magic[6] == 1 else struct.unpack("<I", fh.read(4))[0]
            hdr = ast.literal_eval(fh.read(hl).decode("latin1"))
            assert not hdr["fortran_order"]
            out[info.filename[:-4]] = (fh.tell(), hdr["descr"], tuple(hdr["shape"]), info.file_size, start)
    return out


def npy_bytes(descr, shape, raw):
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (descr, repr(tuple(shape)))
    pad = 64 - (10 + len(hdr) + 1) % 64
    hdr = (hdr + " " * pad + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr + raw


def item(descr, shape):
    k = int(descr[2:]) if descr[1] in "fiu" else 1
    for s in shape[1:]:
        k *= s
    return k


def add(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tf.addfile(ti, io.BytesIO(data))


def main():
    base = sys.argv[1]
    tf = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
    for p, tau1 in TAU1.items():
        tag = "p%.5f" % float(p)
        path = "%s/%s/read/%s.npz" % (base, p, tag)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        mem = members(path)
        N = mem["conf1"][2][0]
        with open(path, "rb") as fh:
            def raw(name):
                off, descr, shape, _, _ = mem[name]
                fh.seek(off)
                return fh.read(item(descr, shape) * shape[0])
            for name in ALL:                                            # whole members, byte for byte
                _, _, _, size, start = mem[name]
                fh.seek(start)
                add(tf, "%s/all_%s.npy" % (tag, name), fh.read(size))
            conf = array.array("f")
            conf.frombytes(raw("conf1"))
            p0, p1 = raw("pred0"), raw("pred1")
            assert mem["pred0"][2] == (N, 2) and conf.itemsize == 4 and sys.byteorder == "little"
            rows = [i for i in range(N) if conf[i] < tau1 and p0[2 * i:2 * i + 2] != p1[2 * i:2 * i + 2]]
            routed = sum(1 for c in conf if c < tau1)
            del conf, p0, p1
            add(tf, "%s/sel_rows.npy" % tag, npy_bytes("<i8", (len(rows),), array.array("q", rows).tobytes()))
            for name in SEL:
                off, descr, shape, _, _ = mem[name]
                k = item(descr, shape)
                buf = bytearray()
                blk = 65536
                j = 0
                for lo in range(0, N, blk):                             # sequential blocks, rows picked in order
                    hi = min(N, lo + blk)
                    if j >= len(rows) or rows[j] >= hi:
                        continue
                    fh.seek(off + lo * k)
                    b = fh.read((hi - lo) * k)
                    while j < len(rows) and rows[j] < hi:
                        r = rows[j] - lo
                        buf += b[r * k:(r + 1) * k]
                        j += 1
                assert j == len(rows)
                add(tf, "%s/sel_%s.npy" % (tag, name), npy_bytes(descr, (len(rows),) + shape[1:], bytes(buf)))
        info = {"p": float(p), "tau1": tau1, "npz": path, "npz_sha256": h.hexdigest(), "shots": N, "routed": routed,
                "rows": len(rows), "members": {k: [v[1], list(v[2])] for k, v in mem.items()}}
        add(tf, "%s/info.json" % tag, json.dumps(info, indent=1).encode())
        sys.stderr.write("%s: %d shots, routed %d, routed with different decisions %d\n" % (tag, N, routed, len(rows)))
    tf.close()


if __name__ == "__main__":
    main()
