#!/bin/bash
#SBATCH -p artxpro6000 --qos=gpu-normal --gres=gpu:rtx_pro_6000:1 --nice=1000 -N 1 -n 1 -c 8 --mem=32G -t 00:45:00
#SBATCH -J rt_bb72net -o <cluster-scratch>/retest/logs/%x_%j.out
# BB72's network reads (RETEST.md section 3.1): r11 and r12 at p 0.003 (BB72-3, 20 chunks), r12 at p 0.002 (BB72-2,
# 30 chunks), the fused engine at batch 1 on a full RTX PRO 6000 (the policies' own engine signature: the commit guard
# passes without --gpu). Stage 3's commands (data/bb/stage3/pod/s3_reads.sh) without --gpu.
set -euo pipefail
source ${T:-<cluster-scratch>/retest/code/freeze}/retest/jobs/bb_env.sh
bb72; POL=data/bb/policies; W=$T/weights/bb
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$PYG $S bb.route infer --split test --code bb72 --p 0.003 --tag r11 --chunks 0-19 --policy $POL/bb72_p0.003_rep1.json --ckpt $W/r11_final_ema.pt | tail -1 | cut -c1-200
$PYG $S bb.route infer --split test --code bb72 --p 0.003 --tag r12 --chunks 0-19 --policy $POL/bb72_p0.003_rep2.json --ckpt $W/r12_final_ema.pt | tail -1 | cut -c1-200
$PYG $S bb.route infer --split test --code bb72 --p 0.002 --tag r12 --chunks 0-29 --policy $POL/bb72_p0.002_rep2.json --ckpt $W/r12_final_ema.pt | tail -1 | cut -c1-200
echo NETREAD_END
