#!/bin/bash
#SBATCH -J srt_r3
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH --nice=1000
# R3: the multi-shot engine (M2): the networks C3 needs at batch 4096 (S55 on both engines, S55'-deep on the new one, the
# Cascade networks C14 and C14'; 15 processes of m2.py nets) and the pipelines C4 needs (depspam_fm configs 2 and 7 batched
# on both engines, Cascade alone batched: 18 processes of m2.py pipeline), 3 repetitions each, under the gate. The
# originals' windows at the other batch sizes and the single-shot, chain and tail variants feed no printed number.
# Environment: common.sh (the seed caches are the M2 caches). Estimate on the RTX PRO 6000 (dry jobs rtx_r3 and rtx_r3b,
# scaled): about 1.3 min venv staging + 12 min for the 15 net processes (each Cascade network process takes about 100 s, a ConvLines
# network 11-40 s) + 7 min for the 18 pipelines = 21 min, plus 0.5-1.7 min per replaced process.
source "${SRT_ROOT:?}/speed_rt/cascade/common.sh"
$PY $CAS/drive.py r3 --tree $T --out $OUT ${DRIVE_ARGS:-}
date; echo JOBDONE
