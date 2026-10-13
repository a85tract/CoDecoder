#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 1 --mem=4G -t 00:20:00 -J rt_trees
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# The Cascade and head trees that run (Alpine) against this repository's copies (RETEST.md section 7, item 6).
set -euo pipefail
T=${T:-<cluster-scratch>/retest/code/dry}; P=${PUB:-<cluster-scratch>/retest/code/public}; Y=<cluster-scratch>
<cluster-scratch>/retest/bin/pycpu $T/retest/replay/tree_compare.py ${OUT:-$Y/retest/replay/tree_compare.json} \
  d1=$Y/cascade_d1/code.20261011_152939:$P/cascade/bb72 \
  d13=$Y/cascade_w5/code.20261012_060258:$P/cascade/d13 \
  head=$Y/rev_a1/code_at/2e69738e74be71c5cfbcb6661ecfa28a54048877:$P/head
