#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1 -c 2 --mem=40G -t 01:00:00 --nice=1000
#SBATCH -J d1rt_x2
#SBATCH -o <cluster-scratch>/retest/logs/d1/%x_%j.out
# X2's BB72 analysis on the re-test's reads (exploratory), as X2's own job (revision/x2/job.sbatch: acpu -c 2 --mem=40G
# -t 01:00:00, py.sif and its venv). After d1_confirm.sbatch: sbatch --dependency=afterok:<confirm> d1_x2.sbatch
set -euo pipefail
R=<cluster-scratch>/ising
export RT=${RT:-<cluster-scratch>/retest/code/freeze}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYX="apptainer exec -B <cluster-scratch-root> -B <cluster-projects> $R/py.sif $R/venv/bin/python"
echo "node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} job ${SLURM_JOB_ID:-?} RT=$RT"
source $RT/retest/jobs/d1_x2_rt.sh
date; echo JOBDONE
