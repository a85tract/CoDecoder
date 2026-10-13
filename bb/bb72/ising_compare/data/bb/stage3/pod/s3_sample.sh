#!/bin/bash
# BB72 stage 3: sample the two test splits on the pod (STAGE3_PLAN.md section 2), after the beacon record exists
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out
$PY -m bb.stage3 hostcheck --out $O/hostcheck_sampling.json > /dev/null || { echo HOSTCHECK_FAIL; exit 1; }
$PY -m bb.shots --code bb72 --p 0.003 --split test --chunks 0-19 --rows $O/test_rows.csv | cut -c1-60 | tail -2
$PY -m bb.shots --code bb72 --p 0.002 --split test --chunks 0-29 --rows $O/test_rows.csv | cut -c1-60 | tail -2
$PY -c "import sys; print(sorted(m for m in sys.modules if m.startswith('stim._')))" 
echo "SAMPLE_END $(date +%T) rows $(($(wc -l < $O/test_rows.csv) - 1))"
