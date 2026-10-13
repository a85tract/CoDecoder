#!/bin/bash
#SBATCH -J rt_d13_e6
#SBATCH -p artxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH --nice=1000
#SBATCH -t 02:10:00
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# The re-test's e6: the confirmation's batch-1 reads at both p on fresh kind-11 shots (d13_read.sh). The original e6
# (AJ33) ran 1:22:03 on one full RTX PRO 6000 with these resources and a 2:30:00 limit.
#   RT=... RT_BEACON=... RT_SEEDS=... RT_OUT=... sbatch $RT/retest/jobs/d13_e6.sbatch      (RESUME=1 resumes)
set -euo pipefail
READ=e6
source ${RT:?}/retest/jobs/d13_read.sh
date; echo JOBDONE
