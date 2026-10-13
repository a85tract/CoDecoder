#!/bin/bash
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH --mem-per-cpu=3840
#SBATCH -o <cluster-scratch>/ising_arch/logs/%x_%j.out
# Stage G1 (arch/README.md): a GPU job; evaluates the command lines in the file CMD (relative to scripts/alpine_arch/)
# with $PY = the container python of the studies' venv_gpu (torch 2.11.0+cu126, Triton 3.6.0) and $PYT = the same
# container with venv_trt (TensorRT 10.14.1). Through submit_any.sh (first copy to start wins):
#   submit_any.sh arch_NAME $A/code/ising_compare/scripts/alpine_arch/gpu_job.sbatch ah200:gpu:h200_3g.71gb:1 -- \
#     -c 4 --mem=30G -t 00:30:00 -o $A/logs/arch_NAME_%j.out --export=ALL,CMD=args/X.txt
set -euo pipefail
R=<cluster-scratch>/ising
A=<cluster-scratch>/ising_arch
cd $(readlink $A/code)/ising_compare     # the tree itself, not the moving `code` link (apptainer binds the resolved cwd)
export PYTHONPATH=.:../neural_emulation OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       ICMP_NV=$R/ext/ising_nv/code ICMP_SHOTS=$R/shots ICMP_CKPT=$A/ckpt ICMP_OUT=$A/out \
       TORCHINDUCTOR_CACHE_DIR=$A/inductor_cache TRITON_CACHE_DIR=$A/triton_cache
mkdir -p $A/out $A/ckpt $A/art $A/preds
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} job ${SLURM_JOB_ID:-?} code $(pwd)"
VENV=$R/venv_gpu
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader)  # captured: `nvidia-smi | grep -q` under pipefail failed when
if [[ $GPUS == *"RTX PRO 6000"* ]]; then                     # grep quit first (MIG nodes list 4 cards); sm_120: the cu128 build
  VENV=<cluster-scratch>/ising_levers/venv_cu128
  export TORCHINDUCTOR_CACHE_DIR=$A/inductor_cache_cu128 TRITON_CACHE_DIR=$A/triton_cache_cu128
fi
PY="apptainer exec --nv -B <cluster-scratch> $R/pyfull.sif $VENV/bin/python"
PYT="apptainer exec --nv -B <cluster-scratch> $R/pyfull.sif $R/venv_trt/bin/python"
echo "python $VENV"
eval "$(cat scripts/alpine_arch/$CMD)"
date; echo JOBDONE
