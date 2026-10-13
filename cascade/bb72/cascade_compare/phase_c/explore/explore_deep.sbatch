#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J explore_deep
#SBATCH -o <cluster-scratch>/cascade_explore/logs/%x_%j.out
# The exploratory analysis of EXPLORE_DEEP.md as one acpu job (16 cores): explore_deep.py on the tree that ship.sh put
# at $X/run. It reads phase C's development and confirmation reads under cascade_w4/out and writes only $X/run/out.
set -euo pipefail
X=<cluster-scratch>/cascade_explore
I=<cluster-scratch>/ising
V=$(readlink -f $X/run)
T=$V/code
mkdir -p $V/out
cd $V
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       PYTHONDONTWRITEBYTECODE=1
echo "node $(hostname) cpus $SLURM_CPUS_ON_NODE run $V $(cat $T/DEPLOYED)"
apptainer exec -B <cluster-scratch> $I/pyfull.sif $I/venv_gpu/bin/python $T/cascade_compare/phase_c/explore/explore_deep.py \
  --lock1 $V/in/lock1 --lock2 $V/in/lock2 --confirm $V/in/confirm --out $V/out/explore_deep.json \
  --tables $V/out/explore_deep_tables.md --procs ${SLURM_CPUS_PER_TASK:-16}
date; echo JOBDONE
