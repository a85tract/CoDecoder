#!/bin/bash
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/cascade_w4/logs/%x_%j.out
# One phase-C GPU job: sources $CMDFILE in the tree deployed at the job's start ($T), with
#   $PY   bb's venv_cu128 (torch 2.11 for CUDA 12.8) in the pyfull container, read only: the one venv of every GPU
#         step, which runs on a full RTX PRO 6000 (PHASE_C_PLAN.md 5.1; pc_common.gpu_env refuses any other card)
#   $PYR  bb's CPU venv (relay-bp 0.2.2, no torch) in the same container, read only
#   $PC   the phase-C scripts ($T/cascade_compare/phase_c), on PYTHONPATH with the Cascade tree's ising_compare and qnd
# PC_GPU_DRIVER, the driver's version from nvidia-smi, is recorded by every GPU step and named in the lock.
set -euo pipefail
R=<cluster-scratch>/cascade_w4
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$(readlink $R/code)
PC=$T/cascade_compare/phase_c
V=$B/venv_cu128
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$B/relay \
       TRITON_CACHE_DIR=$R/triton_cache TORCHINDUCTOR_CACHE_DIR=$R/inductor_cache PC_TREE="$(cat $T/DEPLOYED)" \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm --format=csv,noheader
PY="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $V/bin/python"
PYR="apptainer exec -B <cluster-scratch> $I/pyfull.sif $B/venv/bin/python"
echo "node $(hostname) cpus $SLURM_CPUS_ON_NODE tree $PC_TREE venv $V driver $PC_GPU_DRIVER cmdfile $CMDFILE"
source "$CMDFILE"
date; echo JOBDONE
