#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/cascade_w4/logs/%x_%j.out
# One phase-C CPU job (acpu; QEC's cap: 80 cores at a time): sources $CMDFILE in the tree deployed at the job's start,
# with $PY (the torch venv, CPU) and $PYR (bb's CPU venv with relay-bp) in the pyfull container, $PC as in gpu.sbatch.
set -euo pipefail
R=<cluster-scratch>/cascade_w4
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$(readlink $R/code)
PC=$T/cascade_compare/phase_c
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$B/relay \
       PC_TREE="$(cat $T/DEPLOYED)"
PY="apptainer exec -B <cluster-scratch> $I/pyfull.sif $I/venv_gpu/bin/python"
PYR="apptainer exec -B <cluster-scratch> $I/pyfull.sif $B/venv/bin/python"
echo "node $(hostname) cpus $SLURM_CPUS_ON_NODE tree $PC_TREE cmdfile $CMDFILE"
source "$CMDFILE"
date; echo JOBDONE
