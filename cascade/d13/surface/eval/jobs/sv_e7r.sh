# e7r, the clean repeat of e7's eight timings below 97 % of the clock (RESULTS.md section 3.1): C2 batched at both p,
# reps 1-3, and C4 single at 0.35 %, reps 2 and 3, in e7's rep-major order, on the pinned tree, the lock and its
# kernel caches (common.sh: R/eval), with e7's clock sampler. Each pipeline runs unchanged under winmark.py (outside
# the tree), which records its timed window exactly. Outputs in R/eval_retime, not R/eval.
CK=$R/clocks; mkdir -p $CK; CKF=$CK/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 100 > ${CKF}_clk.csv 2>&1 & CW1=$!
nvidia-smi --query-gpu=timestamp,clocks_event_reasons.active --format=csv,noheader -lms 100 > ${CKF}_why.csv 2>&1 & CW2=$!
( while sleep 2; do echo "$(date '+%Y/%m/%d %H:%M:%S.%3N') $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"; done ) > ${CKF}_apps.txt 2>&1 & CW3=$!
source $T/surface/eval/jobs/common.sh
X=$R/eval_retime WM=$R/retime/winmark.py
mkdir -p $X
echo "winmark $(sha256sum $WM)"
for rep in 1 2 3; do for p in 0.006 0.0035; do for c in C1 C4 C5 C2; do for m in single batched; do
  case $c.$m.$p.$rep in C2.batched.*|C4.single.0.0035.[23]) ;; *) continue ;; esac
  run $PY $WM $X/marks.jsonl pipeline $D --config $c --p $p --mode $m --rep $rep --lock $LOCK --stage c7 \
      --out $X/pipes.jsonl --save $X/pipes
done; done; done; done
kill $CW1 $CW2 $CW3 2>/dev/null || true
