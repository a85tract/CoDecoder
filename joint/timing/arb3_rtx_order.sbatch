#!/bin/bash
#SBATCH -J arb3_order
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o <cluster-scratch>/arb3/logs/rtx_order_%j.out
# ARB3: the RTX single-mode clock question. In job AJ52 every joint process ran at 2,295 MHz and every composition
# process but the first at 2,422 MHz, the joint twin always right after the composition. Here the order is joint ->
# composition -> joint, single mode, one repetition per p (bb_job.sh with --reps 1 --only ...; not part of the tables).
X=<cluster-scratch>/arb3
export R=$X/rt/release PY=$X/venv/bin/python CARD=rtx JC=/tmp/arb3_jco_$SLURM_JOB_ID W4=$X/w4_AJ52 \
  RELAY=<cluster-scratch>/bb/relay WB="0.002=6.309573444801933,0.003=3.758374042884441"
trap 'rm -rf $JC' EXIT
for P in 0.002 0.003; do
  C=$([ $P = 0.002 ] && echo 0 || echo 1)
  OUT=$X/out/order_rtx_$SLURM_JOB_ID/p${P}_A DRIVE_ARGS="--reps 1 --only p$P.c$C.single.base.joint" \
    bash $X/rt/timing/bb_job.sh
  OUT=$X/out/order_rtx_$SLURM_JOB_ID/p${P}_B DRIVE_ARGS="--reps 1 --only p$P.c$C.single.base" \
    bash $X/rt/timing/bb_job.sh
done
