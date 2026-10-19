#!/bin/bash
#SBATCH -J arb3_bb
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:15:00
#SBATCH -o <cluster-scratch>/arb3/logs/rtx_bb_%j.out
# ARB3's timing of the joint decision on BB72 (unregistered), RTX PRO 6000: timing/bb_job.sh with Alpine's paths.
X=<cluster-scratch>/arb3
export R=$X/rt/release PY=$X/venv/bin/python OUT=$X/out/bb_rtx_$SLURM_JOB_ID CARD=rtx JC=/tmp/arb3_jc_$SLURM_JOB_ID \
  W4=$X/w4_$SLURM_JOB_ID RELAY=<cluster-scratch>/bb/relay WB="0.002=6.309573444801933,0.003=3.758374042884441"
trap 'rm -rf $JC' EXIT
mkdir -p $X/out
bash $X/rt/timing/bb_job.sh
