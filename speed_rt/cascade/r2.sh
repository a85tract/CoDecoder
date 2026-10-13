#!/bin/bash
#SBATCH -J srt_r2
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 00:35:00
#SBATCH --nice=1000
# R2: phase C's pipelines at the claim points (the part of CJ39 that feeds C6 and C7): depspam_fm configs -1, 2, 3, 7, 8,
# single and batched, 3 repetitions = 36 processes of pc_speed.py, rep-major as speed_pipe.sh runs them, under the gate.
# Needs, besides common.sh's variables (the seed caches are phase C's):
#   W4_LOCK       phase C's lock of the depspam_fm setting (locked_depspam_fm.json of Amendment 1)
#   SPEED_JSON    the frozen speed components (speed_components.json; the lock names its sha256)
# 24 CPUs: the batched Relay-BP chains use 16 worker processes. Estimate on the RTX PRO 6000 (dry job rtx_r2, scaled; the original
# 36-process subset took 15.4 min): about 1.5 min venv staging + 18 min = 19.5 min, plus 0.5-1 min per replaced process.
source "${SRT_ROOT:?}/speed_rt/cascade/common.sh"
: "${W4_LOCK:?} ${SPEED_JSON:?}"
$PY $CAS/drive.py r2 --tree $T --out $OUT --w4-lock $W4_LOCK --speed $SPEED_JSON ${DRIVE_ARGS:-}
date; echo JOBDONE
