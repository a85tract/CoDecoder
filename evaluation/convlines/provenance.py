"""What a record says about the code and the host that made it (SURF-v2's reads, draws and timing records).

code_digest() is the sha256 of the read path's source files (convlines/*.py, timing/*.py and timing/d7's sources,
each as "sha256  relative path" in path order): every SURF-v2 read records it, and the analysis refuses a read whose
digest is not the one the freeze recorded. commit() is the deployed commit (the DEPLOYED file a pinned deployment
writes, else `git rev-parse HEAD` of a checkout)."""
import glob
import hashlib
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READ_PATH = ("convlines/*.py", "timing/*.py", "timing/d7/*.py", "timing/d7/*.cu", "timing/d7/*.cuh",
             "timing/d7/*.cc", "timing/d7/build.sh")
SIMD = ("sse2", "sse4_1", "sse4_2", "avx", "avx2", "avx512f", "avx512bw", "avx512vl", "asimd", "sve")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def code_files(root=ROOT):
    return sorted({os.path.relpath(p, root) for pat in READ_PATH for p in glob.glob(os.path.join(root, pat))})


def code_digest(root=ROOT):
    lines = "".join(f"{sha_file(os.path.join(root, f))}  {f}\n" for f in code_files(root))
    return hashlib.sha256(lines.encode()).hexdigest()


def commit(root=ROOT):
    f = os.path.join(root, "DEPLOYED")
    if os.path.exists(f):
        return open(f).read().strip()
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def host():
    flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
    cpu = flags.split("model name")[1].split("\n")[0].strip(": \t") if "model name" in flags else platform.processor()
    words = set(flags.split())
    return dict(node=platform.node(), machine=platform.machine(), cpu=cpu, simd=sorted(s for s in SIMD if s in words),
                python=sys.version.split()[0])


def versions():
    import numpy as np
    out = dict(numpy=np.__version__)
    for m in ("stim", "pymatching", "torch", "triton", "scipy"):
        if m in sys.modules or m in ("stim", "pymatching"):
            try:
                out[m] = __import__(m).__version__
            except ImportError:
                out[m] = None
    return out
