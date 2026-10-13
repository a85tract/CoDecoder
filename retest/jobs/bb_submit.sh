#!/bin/bash
# Table I's Alpine jobs in order (RETEST.md section 3.1), run on Alpine's login node after the seeds exist:
#   bash retest/jobs/bb_submit.sh [dry]
# draw (host check first) -> after it: BB72's network reads (full RTX PRO 6000) and BB72's CPU reads (one job per
# error rate: sector Relay-BP, BP-OSD, full-DEM Relay-BP). The gross reads go to Derecho and the H100 pod with the
# draw's shots (RETEST.md section 3.1); each analysis (bb_analysis.sbatch) is submitted once its reads are complete.
set -euo pipefail
MODE=${1:-run}; T=${T:-<cluster-scratch>/retest/code/freeze}; J=$T/retest/jobs; E="ALL,MODE=$MODE,T=$T"
d=$(sbatch --parsable --export=$E $J/bb_draw.sbatch); echo "draw $d"
n=$(sbatch --parsable --dependency=afterok:$d --export=$E $J/bb72_net.sbatch); echo "bb72 networks $n"
for x in "0.003 0-19" "0.002 0-29"; do set -- $x
  c=$(sbatch --parsable --dependency=afterok:$d --export=$E,TREE=bb72,CODE=bb72,P=$1,SPLIT=test,CHUNKS=$2,DEC=relay+bposd+relay_full \
      -t 02:00:00 $J/bb_cpu.sbatch); echo "bb72 p $1 CPU reads $c"
done
