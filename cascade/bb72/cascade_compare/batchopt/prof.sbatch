#!/bin/bash
#SBATCH -J w6prof
#SBATCH -p artxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:55:00
#SBATCH --nice=200
#SBATCH -o <cluster-scratch>/cascade_batchopt/logs/%x_%j.out
# The profile on a full RTX PRO 6000 (batchopt/PROFILE.md): prof_run.sh in the tree $A/code (batchopt/deploy.sh), with
# bb's venv_cu128 in the pyfull container as phase C's gpu.sbatch runs it, CUDA 13.1's Nsight tools bound into the
# container, the outputs in $A/out/<job>. Below phase C's jobs in priority (--nice) and under an hour, so that backfill
# starts it only where it delays none of them. The formal run's directory is not written.
set -uo pipefail
A=<cluster-scratch>/cascade_batchopt
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
CU=<cluster-sw>/cuda/13.1
T=$(readlink $A/code)
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       QND_RELAY=$B/relay TRITON_CACHE_DIR=$A/cache/triton TORCHINDUCTOR_CACHE_DIR=$A/cache/inductor \
       PC_TREE="$(cat $T/DEPLOYED)"
export RUN="apptainer exec --nv -B <cluster-scratch> -B $CU $I/pyfull.sif" PYBIN=$B/venv_cu128/bin/python \
       O=$A/out/${SLURM_JOB_ID:-manual} NSYS=$CU/bin/nsys NCU=$CU/bin/ncu
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm --format=csv,noheader
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $PC_TREE out $O"
cd $T/ising_compare
bash $T/cascade_compare/batchopt/prof_run.sh
date; echo JOBDONE
