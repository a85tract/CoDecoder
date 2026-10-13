#!/bin/bash
# gross stage 4: sample the test4 split on the pod (STAGE4_GROSS_PLAN.md section 2), after the beacon record is in the tree
source <pod-home>/g7/env.sh
cd $W/code/ising_compare
O=$W/out
$PY -m bb.stage3_gross hostcheck --out $O/hostcheck_sampling.json > /dev/null || { echo HOSTCHECK_FAIL; exit 1; }
$PY -m bb.shots --code bb144 --p 0.004 --split test4 --chunks 0-14 --rows $O/test4_rows.csv | tail -1 | cut -c1-40
echo "SAMPLE_END $(date +%T) rows $(($(wc -l < $O/test4_rows.csv) - 1))"
