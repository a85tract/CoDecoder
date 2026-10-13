#!/bin/bash
# gross stage 3: sample the test split on the pod (STAGE3_GROSS_PLAN.md section 2), after the beacon record exists
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out
$PY -m bb.stage3_gross hostcheck --out $O/hostcheck_sampling.json > /dev/null || { echo HOSTCHECK_FAIL; exit 1; }
$PY -m bb.shots --code bb144 --p 0.004 --split test --chunks 0-29 --rows $O/test_rows.csv | tail -1 | cut -c1-40
echo "SAMPLE_END $(date +%T) rows $(($(wc -l < $O/test_rows.csv) - 1))"
