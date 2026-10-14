#!/bin/bash
# pack.sh LABEL > tree.tgz: the Decoder Switching stage's deployable tree as a tar.gz on stdout: the files the stage reads
# (surface_v2 without its weights, speed_rt's sets.py, beacon.py, grace_fit.py and
# gapsw/), SUMS (sha256 of every file) and DEPLOYED (LABEL: the commit for a registered session, a dry label otherwise).
# The pod's relay (`pod.sh K untar DEST`) and Alpine (`tar xzf -`) unpack it; both check SUMS.
set -euo pipefail
L=${1:?LABEL (the commit of the freeze, or dry-<n>)}
R=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d); trap 'rm -rf $T' EXIT
cd $R
find surface_v2 speed_rt/sets.py speed_rt/beacon.py speed_rt/grace_fit.py speed_rt/gapsw -type f \
  ! -path '*/__pycache__/*' ! -path 'surface_v2/weights/*' ! -name '*.pyc' | LC_ALL=C sort > $T/files
xargs sha256sum < $T/files > $T/SUMS
echo "$L" > $T/DEPLOYED
tar czf - -C $T SUMS DEPLOYED -C $R -T $T/files
