#!/bin/bash
# Fallback for BB72's network reads on the H100 (RETEST.md section 3.1): stage 3's commands exactly
# (bb/bb72/.../stage3/pod/s3_reads.sh: --gpu names the H100), in stage 3's environment (pod_s3venv.sh). MODE=check: on
# the original test sets (regenerated on Alpine), against the published reads; MODE=run (or dry): the re-test sets.
source <pod-home>/rt/env.sh
set -uo pipefail
MODE=$1; D=$W/$MODE; GPU="NVIDIA H100 80GB HBM3"
export BB_SHOTS=$D/shots BB_PREDS=$D/preds; mkdir -p $D/preds
cd $W/code/bb/bb72/ising_compare && export PYTHONPATH=.:../qldpc_neural RT_MANIFEST=$D/manifest_bb72.csv
POL=data/bb/policies; K=$W/code/weights/bb
$PY3 $S bb.route infer --split test --code bb72 --p 0.003 --tag r11 --chunks 0-19 --policy $POL/bb72_p0.003_rep1.json --ckpt $K/r11_final_ema.pt --gpu "$GPU" | tail -1 | cut -c1-90
$PY3 $S bb.route infer --split test --code bb72 --p 0.003 --tag r12 --chunks 0-19 --policy $POL/bb72_p0.003_rep2.json --ckpt $K/r12_final_ema.pt --gpu "$GPU" | tail -1 | cut -c1-90
$PY3 $S bb.route infer --split test --code bb72 --p 0.002 --tag r12 --chunks 0-29 --policy $POL/bb72_p0.002_rep2.json --ckpt $K/r12_final_ema.pt --gpu "$GPU" | tail -1 | cut -c1-90
[ "$MODE" = check ] && $PY3 $X platform r11,r12 $D/platform_bb72_h100.json
echo "BB72NET_END $(date +%T)"
