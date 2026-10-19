#!/bin/bash
# ARB4 on the pod: ARB's setup (pod_setup.sh: D1's venv pins, environment record), then per p the score export on the
# reduced reads (run_scores.py -> arb_scores.py, bitwise checks against the read), then the frozen A3 (a3_pod.py).
# Bundle unpacked under $W: release/ (94aaaa7f, weights left out), code/, relay_min/, ckpt/, read/<tag>/.
set -uo pipefail
export W=/workspace/arb4
source $W/code/pod_env.sh
bash $W/code/pod_setup.sh || { echo "SETUP_FAILED $?"; exit 2; }
for pt in 0.002:p0.00200 0.003:p0.00300; do
  p=${pt%%:*}; tag=${pt##*:}
  echo "== scores $p start $(date -Is)"
  $PY $W/code/run_scores.py --read $W/read/$tag --p $p --ckpt $W/ckpt/S55_dfm_deep/final_ema.pt --lock $LOCK \
    > $W/out/logs/scores_$p.log 2>&1
  echo "== scores $p exit $? $(date -Is)"
done
echo "== a3 start $(date -Is)"
$PY $W/code/a3_pod.py $W/read > $W/out/logs/a3.log 2>&1
echo "== a3 exit $? $(date -Is)"
echo "== POD_RUN_DONE $(date -Is)"
