"""Phase D1 (cascade_compare/phase_d/PHASE_D1_PLAN.md): D1's fixed inputs and guards, shared by its scripts.

Nothing here imports torch, so the CPU jobs can import it. D1 reads phase C's files (cascade_w4) and milestone 2's
(cascade_batchopt) and never writes them; its own outputs live under A."""
import hashlib
import json
import os

import numpy as np

import pc_common as pc

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(HERE))      # the deployed tree ($T), whose DEPLOYED and DEPLOYED_SHA name it
A = "<cluster-scratch>/cascade_d1"
LOCK = os.path.join(HERE, "d1_locked_depspam_fm.json")
FAMILY = "depspam_fm"
B, S = "C14p", "S55p_deep"                          # Cascade (C14') and the small network (S55'-deep)
C4B_READ = pc.W4 + "/out/check/depspam_fm/read"     # C4b's frozen read (kind 12, chunks 10-19), read only
CHECK_FIRST = os.path.join(os.path.dirname(HERE), "phase_c", "out", "lock_1e8d1c6d", "check_depspam_fm.json")
COMPILED_PATHS = {"single": "compiled@1", "batched": "compiled@4096"}
NET = "64b2e25d"                                    # the Cascade tree's commit (deploy.sh)


def c0_of(F):
    """The block's first chunk (PHASE_D1_PLAN.md 2.3): int(F[:7], 16) mod 9,000,000, F the full sha of the freeze."""
    if len(F) != 40 or any(ch not in "0123456789abcdef" for ch in F):
        raise SystemExit(f"F {F!r}: not a full commit sha")
    return int(F[:7], 16) % 9_000_000


def tree():
    """(full sha, DEPLOYED line) of the deployed tree this process runs in ('' when absent)."""
    out = []
    for f in ("DEPLOYED_SHA", "DEPLOYED"):
        q = os.path.join(TREE, f)
        out.append(open(q).read().strip() if os.path.exists(q) else "")
    return tuple(out)


def tree_tag(F):
    """The DEPLOYED line of a clean tree deployed at F (deploy.sh)."""
    return f"net {NET} d1 {F}"


def lock(path=LOCK):
    return json.load(open(path))


def config_at(L, p):
    """(index, configuration) of the lock's configuration at p."""
    xs = [(i, c) for i, c in enumerate(L["configs"]) if abs(c["p"] - p) < 1e-12]
    if len(xs) != 1:
        raise SystemExit(f"the lock has {len(xs)} configurations at p = {p}")
    return xs[0]


def n_chunks(L, p):
    return int(L["chunks"][str(p)])


def first_lock(L, path=None):
    """The first lock (phase C, 1e8d1c6d), refused unless its file has the sha256 the D1 lock names."""
    path = path or L["source"]["alpine_file"]
    if pc.sha256_file(path) != L["source"]["sha256"]:
        raise SystemExit(f"{path}: not the first lock (sha256 {L['source']['sha256'][:16]})")
    return json.load(open(path))


def configs_copied(L, first):
    """The lock's configurations are exactly the first lock's configurations named in its source (2 and 7)."""
    return [first["configs"][i] for i in L["source"]["configs"]] == L["configs"]


def unpack_det(packed):
    """Detector rows saved by bb.cascade_eval --save-det (packbits, little-endian bit order) -> (n, 432) uint8."""
    return np.unpackbits(packed, axis=1, count=432, bitorder="little")


def chunk_digests(det_packed, chunks, meta):
    """Every chunk's digest recomputed from the saved rows (as bb.cascade_eval computes it: the first 16 hex digits of
    the sha256 of the rows packed in numpy's default bit order) against the read's chunk_meta -> (ok, n_chunks)."""
    want = pc.det_digests(meta)
    order = [c["chunk"] for c in meta["chunk_meta"]]
    lo = 0
    for j in order:
        hi = lo + pc.CHUNK
        if not np.all(chunks[lo:hi] == j):
            return False, 0
        d = unpack_det(det_packed[lo:hi])
        if hashlib.sha256(np.packbits(d, axis=1).tobytes()).hexdigest()[:16] != want[j]:
            return False, 0
        lo = hi
    return lo == len(det_packed), len(order)


def u32(x):
    """A float32 array's bit patterns (bitwise comparisons of confidences and margins)."""
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
