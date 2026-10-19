#!/bin/bash
#SBATCH -J arb3_bench
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o <cluster-scratch>/arb3/logs/rtx_bench_%j.out
# ARB3: the joint step's cost measured directly on the RTX PRO 6000 (timing/bench_job.sh with Alpine's paths).
X=<cluster-scratch>/arb3
export B=$X/rt R=$X/rt/release PY=$X/venv/bin/python W4=$X/w4_AJ52 RELAY=<cluster-scratch>/bb/relay \
  D13ROOT=$X/d13root DLOCK=$X/out/d13_rtx/dry_lock.json OUTB=$X/out/bench_rtx KB=/tmp/arb3_kb_$SLURM_JOB_ID
trap 'rm -rf $KB' EXIT
bash $X/rt/timing/bench_job.sh
