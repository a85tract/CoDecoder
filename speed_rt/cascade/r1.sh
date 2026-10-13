#!/bin/bash
#SBATCH -J srt_r1
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 00:35:00
#SBATCH --nice=1000
# R1: the follow-up test of Cascade on BB72 (D1-S): 36 pipelines of D1's tree, 3 repetitions, rep-major as d1_speed.sh runs
# them, each a fresh process with the original arguments, under the gate of drive.py (replacement at most twice).
# Environment: common.sh (sourced from SRT_ROOT). Full RTX PRO 6000, one card (the harness refuses any other). Estimate on the RTX PRO 6000 (dry job rtx_r1,
# scaled to 3 reps, 20,000 / 500,000 shots; the original job took 14.6 min): about 2 min venv staging + 17 min = 19 min, plus
# 0.5-1 min per replaced process.
source "${SRT_ROOT:?}/speed_rt/cascade/common.sh"
$PY $CAS/drive.py r1 --tree $T --out $OUT ${DRIVE_ARGS:-}
date; echo JOBDONE
