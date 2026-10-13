#!/bin/bash
#SBATCH -J rt_d13_e5
#SBATCH -p artxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH --nice=1000
#SBATCH -t 03:15:00
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# The re-test's e5: the confirmation's batched reads at both p on fresh kind-11 shots (d13_read.sh). The original e5
# (AJ32) ran 2:06:48 on one full RTX PRO 6000 with these resources and a 4:00:00 limit.
#   RT=<cluster-scratch>/retest/code/freeze RT_BEACON=.../run/beacon.json RT_SEEDS=.../run/seeds.json \
#   RT_OUT=<cluster-scratch>/retest/out sbatch $RT/retest/jobs/d13_e5.sbatch      (RESUME=1 resumes)
set -euo pipefail
READ=e5
source ${RT:?}/retest/jobs/d13_read.sh
date; echo JOBDONE
