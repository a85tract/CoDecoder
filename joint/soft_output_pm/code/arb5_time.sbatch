#!/bin/bash
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/qec_arb5/logs/%x_%j.out
# ARB5 (exploratory): the gaps' CPU time on the paper's CPU (EPYC 7713), one process on one pinned core, the distances in
# turn, on the first 10,000 routed shots of each at the registered q (q = 2 % at d = 11 and 13, where nothing is routed).
#   sbatch -p acpu --qos=cpu-normal --constraint=epyc-7713 -c 4 --mem=16G -t 00:45:00 -J arb5t arb5_time.sbatch
set -euo pipefail
S=/scratch/alpine/${USER:?}; W=$S/qec_arb5; E=$S/surf_v2/eval; I=$S/ising
T=$E/code_at/ce8780e7357877a91ede59e1ef7ed7bdfd02280e
PY="apptainer exec -B /scratch/alpine $I/pyfull.sif $S/release_r/venv_ising/bin/python"
export PYTHONPATH=$T OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $W/out_7713
echo "node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} $(grep -m1 'model name' /proc/cpuinfo)"; date -Is
cd $W/code
$PY arb5_alpine.py time --d 7 9 11 13 15 17 --q '{"7": 0.02, "9": 0.06, "11": 0.02, "13": 0.02, "15": 0.01, "17": 0.01}' \
  --shots9 $W/shots_d9s --core 1 --out $W/out --save $W/out_7713
date -Is; echo JOBDONE
