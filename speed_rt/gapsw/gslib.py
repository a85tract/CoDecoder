"""Shared pieces of the Decoder Switching stage (speed_rt/gapsw): the stage's constants, its sets and circuits, the chunk
files and their digests, and the record helpers. Reads SURF-v2's released code (surface_v2/convlines, timing and
surf_v2/gapsw.py, which this stage never changes) and speed_rt/grace_fit.py, all unchanged.

Names. DS sets srt-gs-d<d> (d = 7 ... 17, the verdict circuits at p = 0.6 %, 5 chunks of 10,000 shots, files
d<d>_p0.006_X_gs_<j>.npz); chunk files are SURF-v2's format (convlines.testset: packed detector bits, observable, seed).
This family has no Grace reference points of its own: the stage's one Grace fit is the surface family's (its 12 sets srt-ref-*,
timed by speed_rt/gref.py); analysis_gs.py reads that fit's file."""
import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "surface_v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SETS_FILE = os.path.join(HERE, "sets_gs.json")
DS_DISTANCES = (7, 9, 11, 13, 15, 17)
REPS = 3
CHUNK = 10_000
MAX_ATTEMPTS = 3
PROBE_D = 9                                   # the host probe's distance (plain PM per block on validation chunk 0)
PROBE_FILE = "d9_p0.006_X_val_000.npz"        # SURF-v2's validation chunk 0 (convlines.testset.name("val", 9, 0))
PROBE_SHOTS_SHA256 = "72a5ae9e57e201490399e64d377abc1b2425251585ca226b905e3381d795bc1b"   # timing.deployed.shots_sha256 of that chunk
PROBE_FILE_SHA256 = "49d1bea92e79979f0e1c0c42895d323febcca5a5557710edf1254179c8186740"     # sha256 of the file (910,686 bytes)
GAPSW_SHA256 = "7c652b0c966af77911b20acb61a940d00780e8d5eb329e01e9f658aa6e3b1f94"   # surf_v2/gapsw.py, never changed


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def sets():
    return json.load(open(SETS_FILE))["sets"]


def ds_set(d):
    return f"srt-gs-d{d}"


def ds_file(d, j):
    return f"d{d}_p0.006_X_gs_{j:03d}.npz"


def circuit(d, p=0.006):
    """The stim circuit of the verdict circuit of d = 7 ... 17 at p = 0.6 % (convlines.circuits)."""
    from convlines import circuits
    assert p == 0.006 and d in DS_DISTANCES, (d, p)
    return circuits.circuit(d)


def circuit_sha256(d, p=0.006):
    from convlines import circuits
    assert p == 0.006 and d in DS_DISTANCES, (d, p)
    return circuits.CIRCUIT_SHA256[d]


def shots_sha256(det, obs):
    from timing import deployed
    return deployed.shots_sha256(det, obs)


def load_chunk(path, ndet, sha256=None):
    """(det (n, ndet) uint8, obs (n,) uint8) of a chunk file (convlines.testset's format), checked against sha256."""
    raw = open(path, "rb").read()
    assert sha256 is None or hashlib.sha256(raw).hexdigest() == sha256, f"{path}: sha256 differs from the manifest"
    import io
    z = np.load(io.BytesIO(raw))
    return np.unpackbits(z["det"], axis=1)[:, :ndet], z["obs"].astype(np.uint8)


def manifest(path):
    return json.load(open(path))


def load_ds(shots, man, d, chunks=None):
    """The 5 chunks of srt-gs-d<d> in order, each checked against the manifest's sha256, and the digest of the whole set
    (checked against the manifest's). -> det, obs, digest. chunks: dry runs read fewer (their own digest, not checked)."""
    ndet = circuit(d).num_detectors
    n = sets()[ds_set(d)]["chunks"] if chunks is None else chunks
    dets, obs = [], []
    for j in range(n):
        f = ds_file(d, j)
        a, b = load_chunk(os.path.join(shots, f), ndet, man["files"][f]["sha256"])
        dets.append(a)
        obs.append(b)
    det, ob = np.ascontiguousarray(np.concatenate(dets)), np.concatenate(obs)
    dg = shots_sha256(det, ob)
    if chunks is None:
        assert dg == man["shots_sha256"][ds_set(d)], f"d = {d}: the set's digest is not the manifest's"
    return det, ob, dg


def pin(core):
    os.sched_setaffinity(0, {core})
    return sorted(os.sched_getaffinity(0)) == [core]


def cpu_model():
    try:
        return open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
    except (OSError, IndexError):
        return platform.processor()


def cgroup_quota():
    for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        if os.path.exists(f):
            try:
                return open(f).read().strip()
            except OSError:
                pass
    return None


def env_record():
    import pymatching
    import stim
    return dict(cpu=cpu_model(), cpus_allowed=len(os.sched_getaffinity(0)), cpus_online=os.cpu_count(),
                cgroup_cpu=cgroup_quota(), python=sys.version.split()[0], numpy=np.__version__, stim=stim.__version__,
                pymatching=pymatching.__version__, node=platform.node(), omp=os.environ.get("OMP_NUM_THREADS"))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel_argv(argv):
    """argv as a record keeps it: every argument that is an absolute path (or `--opt=/abs/path`) is kept by its file name only. Records
    and manifests name files relatively, never by a host's or cluster's absolute path, so that publication changes no hash (UPDATE 29, item 1)."""
    def one(x):
        if os.path.isabs(x):
            return os.path.basename(os.path.normpath(x))
        k, eq, v = x.partition("=")
        return k + eq + os.path.basename(os.path.normpath(v)) if eq and os.path.isabs(v) else x
    return [one(x) for x in argv]


def write_once(path, obj):
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
