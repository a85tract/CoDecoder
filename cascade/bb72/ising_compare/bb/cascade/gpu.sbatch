#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o <cluster-scratch>/cascade_w2/logs/%x_%j.out
# The curve sets (bb/cascade/CASCADE_CURVES.md): one GPU job. Evaluates bb/cascade/args/$CMD with $PY = ising_compare's torch venv
# (venv_gpu, torch 2.11.0 for CUDA 12.6; on a Blackwell card the BB sub-route's venv_cu128, the same torch for CUDA
# 12.8; both read only) in the pyfull container.
#   sbatch -p aa100 --qos=gpu-normal --gres=gpu:1 -J NAME -t HH:MM:SS --export=ALL,CMD=NAME.txt bb/cascade/gpu.sbatch
set -euo pipefail
W=<cluster-scratch>/cascade_w2
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$(readlink $W/code)
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$B/relay W2_DIR=$W \
       BB_CKPT=$B/ckpt CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in *Blackwell*) V=$B/venv_cu128 ;; *) V=$I/venv_gpu ;; esac
echo "node $(hostname) tree $(cat ../DEPLOYED) venv $V cmd $CMD"
PY="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $V/bin/python"
eval "$(cat bb/cascade/args/$CMD)"
date; echo JOBDONE
