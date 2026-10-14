#!/bin/bash
# SPEED-RT CPU half on the Windows CPU host: the pinned Python environment in the tmpfs, cached on the distro's disk.
# Runs inside the WSL2 distro as root:  bash setup_env.sh build | sync | restore | check  (sync: install PKGS into an existing venv, re-tar)
#   build    a tools venv on disk with uv (from PyPI; the host reaches PyPI but not GitHub, so uv cannot fetch a
#            standalone CPython), then a copy of Ubuntu 24.04's CPython 3.12.3 (/usr/bin/python3.12 and
#            /usr/lib/python3.12) into /dev/shm/srt/pybase, a venv on it at /dev/shm/srt/venv, the paper's pinned
#            packages, and a tar of both at $QRT/cache/env.tar (QRT: ${QRT:-$HOME/qrt})
#   restore  the distro's tmpfs lives only while the distro runs; after a restart, unpack the cache to the same paths
#   check    print the interpreter, its prefix and every package version (the session's record)
# The interpreter, its standard library and every package then live in the tmpfs; only the C libraries the
# interpreter links (libc, libm, libz, libssl, ...) come from the disk, once, at the warm pass, and stay in the page
# cache. harness.py's disk check (/proc/diskstats around every window) rejects a window with any block-device read.
# The pods and the paper's studies used Python 3.12.3 (uv) and 3.12.14 (M4); this host's 3.12.3 is the pods' version.
set -euo pipefail
R=${SRT_TMPFS:-/dev/shm/srt}       # SRT_TMPFS: another tmpfs root (the Linux host: /dev/shm/srt_box)
QRT=${QRT:-$HOME/qrt}
CACHE=$QRT/cache
TOOLS=$QRT/tools
PKGS=(numpy==2.2.6 scipy==1.18.1 stim==1.16.0 PyMatching==2.4.0 relay-bp==0.2.2 ldpc==2.4.1 stimbposd==0.2.0
      sinter==1.16.0 beliefmatching==0.2.0)  # sinter: imported by Relay-BP's tests/testdata helper (the BB circuits);
                                             # beliefmatching: a dependency of stimbposd, pinned here
mkdir -p "$R" "$CACHE"

case "${1:-check}" in
build)
  [ -x "$TOOLS/bin/uv" ] || { python3.12 -m venv "$TOOLS" && "$TOOLS/bin/pip" install -q uv; }
  rm -rf "$R/pybase" "$R/venv"
  mkdir -p "$R/pybase/bin" "$R/pybase/lib"
  cp /usr/bin/python3.12 "$R/pybase/bin/python3.12"
  cp -a /usr/lib/python3.12 "$R/pybase/lib/python3.12"
  "$TOOLS/bin/uv" venv -q --python "$R/pybase/bin/python3.12" "$R/venv"
  UV_CACHE_DIR=$QRT/uvcache "$TOOLS/bin/uv" pip install -q --python "$R/venv/bin/python" "${PKGS[@]}" ${EXTRA_PKGS:-}
  tar -C / -cf "$CACHE/env.tar" "${R#/}/pybase" "${R#/}/venv"
  bash "$0" check
  ;;
sync)
  UV_CACHE_DIR=$QRT/uvcache "$TOOLS/bin/uv" pip install -q --python "$R/venv/bin/python" "${PKGS[@]}" ${EXTRA_PKGS:-}
  tar -C / -cf "$CACHE/env.tar" "${R#/}/pybase" "${R#/}/venv"
  bash "$0" check
  ;;
restore)
  [ -x "$R/venv/bin/python" ] || tar -C / -xf "$CACHE/env.tar"
  bash "$0" check
  ;;
check)
  "$R/venv/bin/python" -I - <<'EOF'
import importlib.metadata as md, platform, sys, sysconfig
print("python", sys.version.split()[0], sys.executable, "prefix", sys.base_prefix, "stdlib", sysconfig.get_paths()["stdlib"])
print(" ".join(f"{p}=={md.version(p)}" for p in ("numpy", "scipy", "stim", "PyMatching", "relay-bp", "ldpc", "stimbposd", "sinter", "beliefmatching")))
EOF
  ;;
*) echo "usage: $0 build|restore|check" >&2; exit 2 ;;
esac
