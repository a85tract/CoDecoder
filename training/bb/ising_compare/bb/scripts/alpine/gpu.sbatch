#!/bin/bash
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o <cluster-scratch>/bb/logs/%x_%j.out
# One GPU job of the BB sub-route on Alpine (stage 2: at most 3 GPUs and 64 cores at a time, the study lead's cap):
# the commands in bb/scripts/alpine/args/$CMD with $PY = ising_compare's torch venv (venv_gpu, CUDA 12.6, read only; on
# a Blackwell card, whose sm_120 it lacks, $R/venv_cu128: the same torch 2.11.0 built for CUDA 12.8, args/s2venv.txt)
# and $PYQ = its cudaq-qec venv (plus $R/pyq_extra: sinter and relay_bp, which qnd.circuits' basis filter imports and
# that venv lacks; args/s2setup.txt provides them), both in the pyfull container. $MM = the minutes a training segment may run
# before it pauses with a checkpoint (bb.train --max-minutes): until the job's end or the drain time $DRAIN
# (default: 04:30 PDT on 2026-10-07, one hour before Alpine's maintenance), whichever comes first, less 10 minutes
# (bb/PLAN.md 2.8).
#   sbatch -p aa100 --gres=gpu:a100-40gb:1 -J NAME -t HH:MM:SS [--time-min HH:MM:SS] --export=ALL,CMD=NAME.txt bb/scripts/alpine/gpu.sbatch
#   sbatch -p artxpro6000 --gres=gpu:rtx_pro_6000:1 ...   (a full RTX PRO 6000: the registered GPU, PLAN 2.8 / Q9)
set -euo pipefail
R=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$(readlink $R/code)                    # pin this job to the tree deployed at its start (a redeploy moves the symlink)
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$R/relay BB_SHOTS=$R/shots \
       BB_OUT=$R/out BB_PREDS=$R/preds BB_CKPT=$R/ckpt TRITON_CACHE_DIR=$R/triton_cache
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
DR=$(date -d "${DRAIN:-2026-10-07T04:30:00-07:00}" +%s)
NOW=$(date +%s)
[ $DR -gt $NOW ] && [ $DR -lt $END ] && END=$DR
MM=$(( (END - NOW) / 60 - 10 ))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in *Blackwell*) V=$R/venv_cu128 ;; *) V=$I/venv_gpu ;; esac
echo "node $(hostname) tree $(cat ../DEPLOYED) venv $V cmd $CMD; training segments pause after $MM min ($(date -d @$END))"
PY="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $V/bin/python"
PYQ="env PYTHONPATH=$PYTHONPATH:$R/pyq_extra apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $I/venv_cudaq/bin/python"
eval "$(cat bb/scripts/alpine/args/$CMD)"
date; echo JOBDONE
