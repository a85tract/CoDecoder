#!/bin/bash
# The re-test's content pin of an original deployed tree (RETEST.md sections 3.2 and 3.3; review finding 3), run on the
# login node before each D1 and d = 13 job is submitted; its output line goes into the next commit. Python's byte-code
# caches (__pycache__, *.pyc), which the runs themselves write into the trees, are not pinned.
#   bash retest/jobs/tree_pin.sh TREE LIST SHA256    LIST = the list on the cluster (retest/pins/ holds its copy, with
#                                                    the job ids in log names masked); SHA256 = LIST's sha256 (RETEST.md)
set -euo pipefail
T=$1; L=$2; H=$3
[ "$(sha256sum < $L | cut -c1-64)" = "$H" ] || { echo "PIN_FAIL $T: $L is not the frozen list"; exit 1; }
(cd $T && sha256sum --quiet --strict -c $L) || { echo "PIN_FAIL $T: a file differs from the list"; exit 1; }
[ "$(cd $T && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort)" = "$(cut -c67- $L | LC_ALL=C sort)" ] \
  || { echo "PIN_FAIL $T: a file was added or removed"; exit 1; }
echo "PIN_OK $(cat $T/DEPLOYED) files $(wc -l < $L) list $H $(date -u +%FT%TZ)"
