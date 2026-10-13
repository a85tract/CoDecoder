#!/bin/bash
# The pod's reads (RETEST.md section 3.1), blind. MODE=check: the platform check, the gross network through fusedt on
# the original test4 chunks (regenerated on Alpine from stage 4's beacon, its replay manifest) against the published
# predictions and margins, bit for bit. MODE=run (or dry): the gross network's test4 read (stage 4's command), then
# cudaq-qec's reads, one whole read each: gross test4 0-14 (gross_stage4 tree, as stage 4) and BB72 test 0-19 at
# p 0.003 and 0-29 at p 0.002 (bb72 tree, as stage 3). Shots and manifests: Alpine's draw, copied to $W/$MODE.
source <pod-home>/rt/env.sh
set -uo pipefail
MODE=$1; D=$W/$MODE
export BB_SHOTS=$D/shots BB_PREDS=$D/preds
mkdir -p $D/preds $D/rows
CK=$W/code/weights/bb/g2_D12k113c10_final_ema.pt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd $W/code/bb/gross_stage4/ising_compare && export PYTHONPATH=.:../qldpc_neural RT_MANIFEST=$D/manifest_gross.csv
$PY $S bb.stage4_gross infer --split test4 --chunks 0-14 --ckpt $CK | tail -1
echo "NETREAD_END $(date +%T)"
if [ "$MODE" = check ]; then $PY $X platform g2_D12k113c10_fusedt_s4 $D/platform_gross.json; exit $?; fi
$PYQ $S bb.gpu_bposd read --code bb144 --p 0.004 --split test4 --chunks 0-14 --out $D/rows/gpu_bposd_gross.csv 2>&1 | tail -1 | cut -c1-160
cd $W/code/bb/bb72/ising_compare && export PYTHONPATH=.:../qldpc_neural RT_MANIFEST=$D/manifest_bb72.csv
$PYQ $S bb.gpu_bposd read --code bb72 --p 0.003 --split test --chunks 0-19 --out $D/rows/gpu_bposd_bb72.csv 2>&1 | tail -1 | cut -c1-160
$PYQ $S bb.gpu_bposd read --code bb72 --p 0.002 --split test --chunks 0-29 --out $D/rows/gpu_bposd_bb72.csv 2>&1 | tail -1 | cut -c1-160
echo "READS_END $(date +%T)"
