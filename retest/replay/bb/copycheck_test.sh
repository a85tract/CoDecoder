#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 2 --mem=8G -t 00:15:00 -J rt_copycheck
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# rt_bb.py verify_copies (review finding 8): stage 3's analysis through the shim refuses parameter copies that differ
# from the frozen files beyond the CPU-term rows and the GPU field. On a copy of the dry run's prepared directory.
set -uo pipefail
MODE=dry; source ${T:-<cluster-scratch>/retest/code/freeze}/retest/jobs/bb_env.sh; bb72
B=<cluster-scratch>/retest/dry/bb_fixA/stage3; W=<cluster-scratch>/retest/dry/copycheck
t() {   # t NAME PYTHON-EDIT: a fresh copy of B with its rows paths in W (as --prepare W writes them), one edit, analysis
  rm -rf $W && mkdir -p $W && cp $B/cells.json $B/amendment_S.json $B/rows_cpu_retest.csv $B/rows_cpu_S_retest.csv $W/
  /usr/bin/python3 -c "import json; W='$W'
c = json.load(open(W + '/cells.json')); c['rows']['cpu'] = W + '/rows_cpu_retest.csv'; json.dump(c, open(W + '/cells.json', 'w'), indent=1)
a = json.load(open(W + '/amendment_S.json')); a['rows_cpu'] = W + '/rows_cpu_S_retest.csv'; json.dump(a, open(W + '/amendment_S.json', 'w'), indent=1)"
  /usr/bin/python3 -c "import json; W='$W'; $2"
  o=$(RT_STAGE=$W RT_CELLS=$W/cells.json $PYC $S bb.stage3 analyze --out $W 2>&1 | tail -1); r=$?
  echo "case $1: exit $r: ${o:0:200}"
}
t baseline_unchanged "pass"
t threshold_changed "c=json.load(open(W+'/cells.json')); c['cells']['BB72-3F']['tau']+=1e-9; json.dump(c,open(W+'/cells.json','w'))"
t rows_outside_stage "c=json.load(open(W+'/cells.json')); c['rows']['cpu']='/tmp/rows_cpu_retest.csv'; json.dump(c,open(W+'/cells.json','w'))"
t gpu_not_checked "c=json.load(open(W+'/cells.json')); c['gpu']='NVIDIA A100-SXM4-80GB'; json.dump(c,open(W+'/cells.json','w'))"
t gpu_frozen_wrong "c=json.load(open(W+'/cells.json')); c['gpu_frozen']='NVIDIA RTX PRO 6000 Blackwell Server Edition'; json.dump(c,open(W+'/cells.json','w'))"
t amendment_changed "a=json.load(open(W+'/amendment_S.json')); a['title']+=' '; json.dump(a,open(W+'/amendment_S.json','w'))"
rm -rf $W
