#!/bin/bash
#SBATCH -J rt_d13_analysis
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --nice=1000
#SBATCH -t 01:00:00
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# The re-test's analysis (eval/jobs/e8_check.sh's last two steps): sv_route confirm on the new read directory, then
# sv_route report with the original e7 rows (pipes.jsonl, bitwise_c7.json) and speed.json, through the shim. It draws
# no shot (patch on or off changes nothing); it needs no GPU (identity ANALYSIS: venv, packages, code), so it runs on a
# CPU node with the unstaged venv. confirm needs about 11 GB of memory and 15 s; both write once.
#   RT=... RT_OUT=... sbatch $RT/retest/jobs/d13_analysis.sh        (RT_BEACON / RT_SEEDS optional here)
# Overrides, for the replay on the original reads: RT_CONF (default $RT_OUT/d13/conf), RT_ROUTE (default $RT_OUT/d13/route).
set -euo pipefail
source ${RT:?RT: the deployed freeze tree (contains retest/)}/retest/jobs/d13_env.sh
: "${RT_OUT:?}"
CONF=${RT_CONF:-$RT_OUT/d13/conf} ROUTE=${RT_ROUTE:-$RT_OUT/d13/route}
PY="apptainer exec -B <cluster-scratch-root> $I/pyfull.sif $V/bin/python"
SV_VENV_STAGED="not staged: the analysis needs no GPU; the venv is read in place at its own path"; export SV_VENV_STAGED
source $T/surface/eval/jobs/common.sh       # E=$R/eval, LOCK, GATE, run
mkdir -p $ROUTE
echo "analysis of $CONF -> $ROUTE"
run $PY $SHIM surface.sv_route confirm --lock $LOCK --gate $GATE --conf $CONF --out $ROUTE/confirm_d13.json
run $PY $SHIM surface.sv_route report --lock $LOCK --confirm $ROUTE/confirm_d13.json --speed $E/speed.json \
    --pipes $E/pipes.jsonl --bitwise $E/bitwise_c7.json --out $ROUTE/SURFACE_RESULT_retest.md
date; echo JOBDONE
