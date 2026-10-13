#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1 -c 2 --mem=40G -t 01:00:00 --nice=1000
#SBATCH -J d1rt_x2replay
#SBATCH -o <cluster-scratch>/retest/logs/d1/%x_%j.out
# The X2 adapter's replay on D1's original records, in X2's own job environment (revision/x2/job.sbatch).
set -euo pipefail
R0=<cluster-scratch>/ising
export RT=${RT:-<cluster-scratch>/retest/code/dry} OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYX="apptainer exec -B <cluster-scratch-root> -B <cluster-projects> $R0/py.sif $R0/venv/bin/python"
echo "node $(hostname) job ${SLURM_JOB_ID:-?} RT=$RT"
source $RT/retest/replay/d1/replay_x2.sh
date; echo JOBDONE
