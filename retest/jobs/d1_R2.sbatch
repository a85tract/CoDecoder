#!/bin/bash
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1 -c 16 --mem=96G -t 03:00:00 --nice=1000
#SBATCH -J d1rt_r2
#SBATCH -o <cluster-scratch>/retest/logs/d1/%x_%j.out
# D1 re-test read R2: p = 0.3 %, one full RTX PRO 6000 as D1's R2 (D1's job: -c 16 --mem=96G, 1.29 / 2.00 GPU-h; this
# walltime is 1.5 times that). The eager read through the seed shim, the seed check, the compiled reads, as d1_read_rt.sh.
# Submit with the paths exported (RT_BEACON, RT_SEEDS: required; RT, RT_OUT = the root of all tests: defaults of the real run):
#   RT_BEACON=.../run/beacon.json RT_SEEDS=.../run/seeds.json sbatch --parsable d1_R2.sbatch
set -euo pipefail
: "${RT_BEACON:?RT_BEACON, the beacon record}" "${RT_SEEDS:?RT_SEEDS, the seeds file}"
export P=0.003
export RT=${RT:-<cluster-scratch>/retest/code/freeze}
source $RT/retest/jobs/d1_gpu_job.sh
