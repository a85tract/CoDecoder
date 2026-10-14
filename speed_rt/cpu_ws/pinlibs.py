"""Keeps every shared object a timed process can load from the distro's disk (the ELF loader, libc, libm, libz, libcrypto,
libstdc++, ...) locked in memory, so that no timed window reads the HDD-backed ext4.vhdx even when the page cache is
dropped (the dry run saw 71 reads of about 6 MB in a window that came 20 minutes after the previous one).
The files: `ldd` of the tmpfs interpreter, of every extension module of its standard library and of the venv, and of the
clock probe, outside /dev/shm, and EXTRA (uname, git and a few other programs and files, with their libraries). They
are mapped read-only and locked with mlockall(MCL_CURRENT); the process then sleeps.
prep.sh env starts it detached and writes its pid to /dev/shm/srt/pinlibs.pid; it prints the files it holds.
    python pinlibs.py [ROOT]          (ROOT: /dev/shm/srt)"""
import ctypes
import glob
import mmap
import os
import subprocess
import sys
import time


# programs and files the timed code may touch outside the tmpfs: platform.processor() runs `uname -p` (cpu_term, gref,
# gapsw), provenance.commit runs git when the tree has no DEPLOYED file, and the loader's cache, zone and locale files
EXTRA = ["/usr/bin/uname", "/usr/bin/git", "/usr/bin/dash", "/usr/bin/bash", "/usr/bin/nproc", "/usr/bin/cat",
         "/usr/bin/lscpu", "/etc/ld.so.cache", "/etc/localtime", "/etc/passwd", "/etc/nsswitch.conf",
         "/usr/lib/locale/locale-archive", "/usr/lib/x86_64-linux-gnu/gconv/gconv-modules.cache"] + \
    glob.glob("/usr/lib/locale/C.utf8/*") + glob.glob("/usr/lib/locale/C.utf8/LC_MESSAGES/*")      # LC_ALL=C.UTF-8


def deps(root):
    objs = [f"{root}/pybase/bin/python3.12", f"{root}/cpu_ws/clk"] + [f for f in EXTRA if os.path.isfile(f)]
    objs += glob.glob(f"{root}/pybase/lib/python3.12/lib-dynload/*.so")
    objs += glob.glob(f"{root}/venv/lib/python3.12/site-packages/**/*.so*", recursive=True)
    out = set()
    for f in objs:
        if not os.path.isfile(f):
            continue
        r = subprocess.run(["ldd", f], capture_output=True, text=True)
        for tok in r.stdout.split():
            if tok.startswith("/") and not tok.startswith("/dev/shm") and os.path.isfile(tok):
                out.add(os.path.realpath(tok))
    out.update(os.path.realpath(f) for f in EXTRA if os.path.isfile(f))
    return sorted(out)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/srt"
    files, maps, total = deps(root), [], 0
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        size = os.fstat(fd).st_size
        if size:
            maps.append(mmap.mmap(fd, size, prot=mmap.PROT_READ))
            total += size
        os.close(fd)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.mlockall(1) != 0:                                          # MCL_CURRENT: fault in and lock every mapping
        raise SystemExit(f"mlockall failed: errno {ctypes.get_errno()}")
    print(f"pinned {len(files)} files, {total / 1e6:.1f} MB: " + " ".join(files), flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
