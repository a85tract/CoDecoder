# Stage 2, C14' (run C14_dfm, run id 201): C14's recipe and stopping rule (w3b_C14.sh) in the depspam_fm circuit
# family (--family depspam_fm: the noise-model fit's variant at every p of the curriculum, the family's own seeds):
# Cascade L 14, H 512, the literal paper version (--res skip --mup-base 0), batch 3,328
# in two micro-batches of 1,664, the paper's schedule over --steps 80000 (budget_ema.pt at step 80,000), then the floor
# LR and p = 0.55 % up to --max-steps 160000; the fixed monitor block (the family's seed kind 1: chunks 100-119 at
# 0.3 %, 100-109 at 0.55 %) at every 1e7 samples; past the budget it stops when the EMA failures fell by less than 2 %
# at both p over the last 3e7 samples (--converge 3e7:0.02). Chained segments (afterany), each resuming last.pt.
# The first segment first runs question 1's evaluation (once: $Q/done) on this GPU: the frozen r11 and r12 (the release
# weights) through the adapter on fresh rows of the family, kind 0 (chunks of 10,000): 0.2 and 0.3 % chunks 0-29 with
# the detector rows kept for the routing, the calibration blocks (kind 7, chunks 0-9 at 0.2 and 0.3 %), 0.25 % and
# 0.35-0.55 % chunks 0-29, 0.15 % chunks 0-99, 0.1 % chunks 0-299; then the training, its minutes recomputed.
T=$R/code.20261009_042038    # every segment runs tree c69800c1 (checked by an Alpine job), whatever is deployed later
cd $T/ising_compare; export PYTHONPATH=$T/ising_compare:$T/qldpc_neural; echo "pinned tree $(cat ../DEPLOYED)"
Q=$R/out/w3c/q1
if [ ! -f $Q/done ]; then
  mkdir -p $Q; ok=0
  E="$PY -m bb.cascade_eval --family depspam_fm --ckpt $B/ckpt/r11/final_ema.pt,$B/ckpt/r12/final_ema.pt --procs 8"
  $E --kind 0 --p 0.002,0.003 --chunks 0:30 --save $Q/test --save-det --out $Q/test_p020_p030.json || ok=1
  $E --kind 7 --p 0.002,0.003 --chunks 0:10 --save $Q/cal --out $Q/cal_p020_p030.json || ok=1
  $E --kind 0 --p 0.0025,0.0035,0.004,0.0045,0.005,0.0055 --chunks 0:30 --save $Q/test --out $Q/test_p025_p055.json || ok=1
  $E --kind 0 --p 0.0015 --chunks 0:100 --save $Q/test --out $Q/test_p015.json || ok=1
  $E --kind 0 --p 0.001 --chunks 0:300 --save $Q/test --out $Q/test_p010.json || ok=1
  if [ $ok = 0 ]; then date > $Q/done; fi
  MM=$(( (END - $(date +%s)) / 60 - 10 )); echo "question 1 evaluation ended (ok=$ok); MM=$MM"
fi
RUN=C14_dfm
if [ -f $R/ckpt/$RUN/final_ema.pt ]; then echo "$RUN has ended (final_ema.pt exists)"; else
  RES=""; [ -f $R/ckpt/$RUN/last.pt ] && RES="--resume"
  $PY -m bb.cascade_train --run $RUN --run-id 201 --family depspam_fm --L 14 --H 512 --res skip --mup-base 0 --micro 2 \
      --compile --workers 2 --log-every 200 --max-steps 160000 --monitor 0.003:100:120,0.0055:100:110 --monitor-every 1e7 \
      --converge 3e7:0.02 --max-minutes $MM $RES || echo "exit $?"
fi
