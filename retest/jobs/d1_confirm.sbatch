#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1 -c 4 --mem=32G -t 01:00:00 --nice=1000
#SBATCH -J d1rt_confirm
#SBATCH -o <cluster-scratch>/retest/logs/d1/%x_%j.out
# D1 re-test analysis: d1_confirm.py through the seed shim on the two reads (D1's D1-A job: acpu -c 4 --mem=32G, 32 s).
# After R1 and R2: RT_BEACON=... RT_SEEDS=... sbatch --dependency=afterok:<R1>:<R2> d1_confirm.sbatch
set -euo pipefail
export STEP=d1_confirm_rt.sh
export RT=${RT:-<cluster-scratch>/retest/code/freeze}
source $RT/retest/jobs/d1_cpu_job.sh
