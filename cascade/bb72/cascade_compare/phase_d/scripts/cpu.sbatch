#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/cascade_d1/logs/%x_%j.out
# One phase-D1 CPU job (acpu): sources $CMDFILE in the deployed tree $T (D1_TREE, else $A/code), with $PY (the torch
# venv, CPU) in the pyfull container and $PC, $PD as in gpu.sbatch.
set -euo pipefail
A=<cluster-scratch>/cascade_d1
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=${D1_TREE:-$(readlink -f $A/code)}
PC=$T/cascade_compare/phase_c
PD=$T/cascade_compare/phase_d
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC:$PD:$T/cascade_compare/batchopt OMP_NUM_THREADS=1 \
       PYTHONUNBUFFERED=1 QND_RELAY=$B/relay PC_TREE="$(cat $T/DEPLOYED)"
PY="apptainer exec -B <cluster-scratch> $I/pyfull.sif $I/venv_gpu/bin/python"
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $PC_TREE cmdfile $CMDFILE"
source "$CMDFILE"
date; echo JOBDONE
