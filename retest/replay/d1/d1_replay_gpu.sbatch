#!/bin/bash
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1 -c 16 --mem=96G -t 00:40:00 --nice=1000
#SBATCH -J d1rt_replread
#SBATCH -o <cluster-scratch>/retest/logs/d1/%x_%j.out
# A D1 replay on one full RTX PRO 6000, in D1's GPU environment (jobs/d1_gpu_job.sh); STEP names the script to run:
#   sbatch --export=ALL,RT=...,STEP=../replay/d1/replay_read_off.sh d1_replay_gpu.sbatch
set -euo pipefail
: "${STEP:?}"; export P=0.002
export RT=${RT:-<cluster-scratch>/retest/code/dry}
source $RT/retest/jobs/d1_gpu_job.sh
