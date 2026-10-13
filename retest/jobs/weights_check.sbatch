#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 2 --mem=16G -t 00:30:00 -J rt_weights
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# Network identity check (RETEST.md section 8): every network file a re-test read loads against this repository's copy.
set -euo pipefail
T=${T:-<cluster-scratch>/retest/code/dry}; PUB=${PUB:-<cluster-scratch>/retest/code/public}
<cluster-scratch>/retest/bin/pycu128 $T/retest/replay/weights_check.py $T/retest/replay/weights_pairs.json $PUB \
  ${OUT:-<cluster-scratch>/retest/replay/weights_check.json} | tail -3
