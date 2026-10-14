#!/bin/bash
# The C libraries in the tmpfs: every shared object outside /dev/shm that the tmpfs interpreter, its extension modules
# and the venv's extension modules link (ldd), copied to ROOT/lib under its soname. With LD_LIBRARY_PATH=ROOT/lib
# (cfg_ws.json "env") a timed process loads every library from the tmpfs; only the ELF loader named in the
# interpreter's header stays the distro's file, locked in memory by pinlibs.py. (Rewriting that header with patchelf
# made Ubuntu's python3.12 crash at start, so the interpreter stays byte-identical to the distro's.) The check at the
# end prints what a process that imports every timed module maps outside the tmpfs (expected: the loader and the
# C.UTF-8 locale files, which pinlibs.py locks too).
#    bash tmpfs_libs.sh [ROOT]          (ROOT: /dev/shm/srt; run by prep.sh env)
set -euo pipefail
R=${1:-${SRT_TMPFS:-/dev/shm/srt}}
PYB=$R/pybase/bin/python3.12
cmp -s /usr/bin/python3.12 "$PYB" || cp --remove-destination /usr/bin/python3.12 "$PYB"
mkdir -p "$R/lib"
mapfile -t objs < <(find "$R/pybase/lib/python3.12/lib-dynload" "$R/venv/lib/python3.12/site-packages" -name '*.so*' -type f)
for f in "$PYB" "${objs[@]}"; do ldd "$f" 2>/dev/null || true; done \
  | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}' | sort -u \
  | while read -r name path; do
      case "$path" in /dev/shm/*) continue ;; esac
      [ -e "$R/lib/$name" ] || cp -L "$path" "$R/lib/$name"
    done
rm -f "$R/lib/ld-linux-x86-64.so.2"
echo "tmpfs libs: $(ls "$R/lib" | wc -l) files, $(du -sh "$R/lib" | cut -f1)"
LD_LIBRARY_PATH=$R/lib LC_ALL=C.UTF-8 TZ=UTC0 "$R/venv/bin/python" - <<'EOF'
import bz2, ctypes, hashlib, json, lzma, ssl                                   # noqa: F401 (the standard library's C parts)
import numpy, scipy.sparse, stim, pymatching, relay_bp, ldpc, sinter             # noqa: F401 (the timed code's)
maps = {ln.split()[-1] for ln in open("/proc/self/maps") if len(ln.split()) >= 6}
out = sorted(p for p in maps if p.startswith("/") and not p.startswith("/dev/shm/"))
print("mapped outside the tmpfs:", " ".join(out) if out else "none")
EOF
