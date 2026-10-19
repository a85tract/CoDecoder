#!/bin/bash
#SBATCH -J arb3_d13
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 00:50:00
#SBATCH -o <cluster-scratch>/arb3/logs/rtx_d13_%j.out
# ARB3's timing of the joint decision at d = 13 (unregistered), RTX PRO 6000: timing/d13_job.sh with Alpine's paths.
# The caller sets PHASE (prep | timed) and W13.
X=<cluster-scratch>/arb3
export R=$X/rt/release PY=$X/venv/bin/python OUTD=$X/out/d13_rtx CARD=rtx K=$X/kc_d13_rtx D13ROOT=$X/d13root \
  DRY=$X/rt/timing/dry
bash $X/rt/timing/d13_job.sh
