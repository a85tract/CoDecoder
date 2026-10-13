#!/bin/bash
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/cascade_w5/logs/%x_%j.out
# One Cascade-surface GPU job (W5): sources $CMDFILE (bash lines using $PY, $TORCHRUN and $MM) in the deployed tree, with
# $PY = ising_compare's torch 2.11 venv in the pyfull container (venv_gpu, CUDA 12.6; on a Blackwell card bb's
# venv_cu128, the same torch built for CUDA 12.8), read only. $MM = the minutes a training segment may run before it
# pauses with a checkpoint (surface_train --max-minutes): until the job's end less 10 minutes.
#   sbatch -p ah200 --gres=gpu:h200:1 -c 16 --mem=120G -J NAME -t 01:00:00 \
#          --export=ALL,CMDFILE=<cluster-scratch>/cascade_w5/jobs/NAME.sh gpu.sbatch
set -euo pipefail
R=<cluster-scratch>/cascade_w5
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=${TREE:-$(readlink $R/code)}            # the tree pinned at submission (submit.sh), else the one deployed at the start
cd $T
export PYTHONPATH=$T OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CONVLINES_RELEASE=$T/release CASCADE_ROOT=$R \
       TRITON_CACHE_DIR=$R/triton_cache TORCHINDUCTOR_CACHE_DIR=$R/inductor_cache \
       NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}     # single-node DDP: NVLink / PCIe and shared memory only
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
MM=$(( (END - $(date +%s)) / 60 - 10 ))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in *Blackwell*) V=$B/venv_cu128 ;; *) V=$I/venv_gpu ;; esac
PY="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $V/bin/python"
TORCHRUN="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $V/bin/torchrun"
echo "node $(hostname) cpus $SLURM_CPUS_ON_NODE tree $(cat $T/DEPLOYED) venv $V cmdfile $CMDFILE; MM=$MM"
source "$CMDFILE"
date; echo JOBDONE
