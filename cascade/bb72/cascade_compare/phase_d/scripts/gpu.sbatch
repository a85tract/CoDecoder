#!/bin/bash
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/cascade_d1/logs/%x_%j.out
# One phase-D1 GPU job (PHASE_D1_PLAN.md 5.3) on a full RTX PRO 6000 (submit.sh asks for -p artxpro6000
# --gres=gpu:rtx_pro_6000:1): sources $CMDFILE in the deployed tree $T (D1_TREE, which submit.sh pins, else $A/code), with
#   $PY       bb's venv_cu128 (torch 2.11 for CUDA 12.8) in the pyfull container, read only, as phase C's gpu.sbatch
#   $PC, $PD  phase C's and D1's scripts, on PYTHONPATH with the Cascade tree's ising_compare and qnd and batchopt
# Triton's and Inductor's caches are this job's own copy ($A/jobcache/<job>, removed at the end) of $A/cache, the seed
# that deploy.sh copied once from phase C's caches: every job starts from the binaries and autotuning choices phase C
# ran. PC_GPU_DRIVER, the driver's version from nvidia-smi, is recorded by every GPU step.
set -euo pipefail
A=<cluster-scratch>/cascade_d1
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=${D1_TREE:-$(readlink -f $A/code)}
PC=$T/cascade_compare/phase_c
PD=$T/cascade_compare/phase_d
V=$B/venv_cu128
JC=$A/jobcache/${SLURM_JOB_ID:-manual}
mkdir -p $JC && cp -r $A/cache/triton $JC/triton && cp -r $A/cache/inductor $JC/inductor
trap 'rm -rf $JC' EXIT
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC:$PD:$T/cascade_compare/batchopt OMP_NUM_THREADS=1 \
       PYTHONUNBUFFERED=1 QND_RELAY=$B/relay TRITON_CACHE_DIR=$JC/triton TORCHINDUCTOR_CACHE_DIR=$JC/inductor \
       PC_TREE="$(cat $T/DEPLOYED)" PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm --format=csv,noheader
PY="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $V/bin/python"
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $PC_TREE venv $V driver $PC_GPU_DRIVER cmdfile $CMDFILE"
source "$CMDFILE"
date; echo JOBDONE
