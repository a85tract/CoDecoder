#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1 -c 2 --mem=8G -t 00:20:00 --nice=1000
#SBATCH -J d1rt_guard
#SBATCH -o <cluster-scratch>/retest/logs/d1/%x_%j.out
# The guard replay (d1_guard_rt.sh against D1's d1_guard.sh), in D1's cpu.sbatch environment for its python.
set -euo pipefail
export STEP=../replay/d1/guard_tests.sh RT=${RT:-<cluster-scratch>/retest/code/dry}
source $RT/retest/jobs/d1_cpu_job.sh
