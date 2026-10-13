#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1 -c 4 --mem=32G -t 00:30:00 --nice=1000
#SBATCH -J d1rt_replay
#SBATCH -o <cluster-scratch>/retest/logs/d1/%x_%j.out
# A D1 replay job on a CPU node (replay checks, RETEST.md test D1), in D1's cpu.sbatch environment. STEP names the script of
# retest/jobs (or a path relative to it) that runs: sbatch --export=ALL,RT=...,STEP=../replay/d1/replay_seeds.sh d1_replay.sbatch
set -euo pipefail
: "${STEP:?}"; export RT=${RT:-<cluster-scratch>/retest/code/dry}
source $RT/retest/jobs/d1_cpu_job.sh
