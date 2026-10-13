#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 4 --mem=32G -t 00:45:00 -J rt_bbanalysis
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# The BB analyses, each once (RETEST.md section 3.1): WHICH=bb72 -> the CPU-term rows and parameter copies (--prepare,
# the read GPU named: READ_GPU, default the RTX PRO 6000; the H100 fallback keeps the frozen field), then stage 3's
# analysis and Amendment S's; WHICH=gross -> the same for stage 4.
set -euo pipefail
source ${T:-<cluster-scratch>/retest/code/freeze}/retest/jobs/bb_env.sh
if [ "$WHICH" = bb72 ]; then
  bb72; A=$OUT/stage3; [ -e $A/results.json ] && { echo "$A/results.json exists: the analysis runs once"; exit 1; }
  export RT_STAGE=$A RT_CELLS=$A/cells.json
  G=${READ_GPU:-$RTX}                        # the GPU BB72's network reads ran on (the H100 fallback: stage 3's GPU)
  if [ "$G" = "NVIDIA H100 80GB HBM3" ]; then $PYC $S --prepare $A; else $PYC $S --prepare $A --gpu "$G"; fi
  $PYC $S bb.stage3 analyze --out $A > $A/analysis.log
  $PYC $S bb.stage3 analyze --amend $A/amendment_S.json --out $A > $A/analysis_amendment_S.log
  mkdir -p $A/outputs; $PYC $S bb/scripts/stage3_pack.py $T/bb/bb72 $A/outputs > $A/pack.log    # every decoder's per-shot outputs, as published
else
  gross; A=$OUT/stage4; [ -e $A/results.json ] && { echo "$A/results.json exists: the analysis runs once"; exit 1; }
  export RT_STAGE=$A RT_CELLS=$A/cell.json
  $PYC $S --prepare $A
  $PYC $S bb.stage4_gross analyze --out $A > $A/analysis.log
  mkdir -p $A/outputs; $PYC $S bb/scripts/stage4_gross_pack.py $T/bb/gross_stage4 $A/outputs > $A/pack.log
fi
ls $A | tr '\n' ' '; echo; echo ANALYSIS_END $WHICH
