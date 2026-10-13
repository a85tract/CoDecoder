#!/bin/bash
#SBATCH -p artxpro6000 --qos=gpu-normal --gres=gpu:rtx_pro_6000:1 --nice=1000 -N 1 -n 1 -c 8 --mem=32G -t 00:45:00
#SBATCH -J rt_bb72plat -o <cluster-scratch>/retest/logs/%x_%j.out
# Platform check of BB72's network reads (RETEST.md section 6): r11 and r12 read on Alpine's full RTX PRO 6000 (the
# re-test's platform) on the original BB72-3 and BB72-2 test sets (regenerated from stage 3's beacon by
# bb_replay.sbatch, their arrays equal to the published ones), patch off, against the published H100 predictions and
# margins (test_outputs_BB72-{3,2}.npz): every decision and every margin bit equal.
set -euo pipefail
T=${T:-<cluster-scratch>/retest/code/dry}; R=${R:-<cluster-scratch>/retest/replay/bb}
O=<cluster-scratch>/retest/platform/bb72; rm -rf $O && mkdir -p $O
unset RT_SEEDS RT_BEACON RT_STAGE RT_CELLS RT_DRY_SHOTS
export BB_SHOTS=$R/shots BB_PREDS=$O/preds RT_MANIFEST=$R/rows_bb72.csv QND_RELAY=<cluster-scratch>/bb/relay
PYG=<cluster-scratch>/retest/bin/pycu128; S=$T/retest/shim/rt_bb.py
cd $T/bb/bb72/ising_compare && export PYTHONPATH=.:../qldpc_neural; POL=data/bb/policies; W=$T/weights/bb
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$PYG $S bb.route infer --split test --code bb72 --p 0.003 --tag r11 --chunks 0-19 --policy $POL/bb72_p0.003_rep1.json --ckpt $W/r11_final_ema.pt | tail -1 | cut -c1-120
$PYG $S bb.route infer --split test --code bb72 --p 0.003 --tag r12 --chunks 0-19 --policy $POL/bb72_p0.003_rep2.json --ckpt $W/r12_final_ema.pt | tail -1 | cut -c1-120
$PYG $S bb.route infer --split test --code bb72 --p 0.002 --tag r12 --chunks 0-29 --policy $POL/bb72_p0.002_rep2.json --ckpt $W/r12_final_ema.pt | tail -1 | cut -c1-120
$PYG $T/retest/replay/bb_replay.py platform r11,r12 $O/platform_bb72.json
