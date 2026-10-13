# Stage 2, C14 (run id 101): Cascade L 14, H 512 with the recipe the fidelity pilot chose:
# the literal paper version, the skip-scaled residual h / sqrt(2L) + f(h) (--res skip) and the quoted peak rates
# without width scaling (--mup-base 0). One GPU, two micro-batches of 1,664 shots per step (BN statistics per
# micro-batch, as on two GPUs without SyncBN; at L 14 one pass of 3,328 shots needs 104 GB and fits only a full
# H200), the batch 3,328 per optimiser step.
# The paper's schedule over --steps 80000 (the fixed budget: budget_ema.pt at step 80,000), then the floor LR (1/10) and
# p = 0.55 % up to --max-steps 160000 (twice the budget). The fixed monitor block (seed kind 1: chunks 100-119 at 0.3 % =
# 200,000 shots, 100-109 at 0.55 % = 100,000) is evaluated (EMA and raw) at every 1e7 samples; past the budget the run
# stops when the EMA failures fell by less than 2 % at both p over the last 3e7 samples (--converge 3e7:0.02).
# Chained segments (afterany): each resumes last.pt; a segment after the end (final_ema.pt) does nothing.
T=$R/code.20261009_001049    # every segment runs tree b142837e (the checked one), whatever is deployed later
cd $T/ising_compare; export PYTHONPATH=$T/ising_compare:$T/qldpc_neural; echo "pinned tree $(cat ../DEPLOYED)"
RUN=C14
if [ -f $R/ckpt/$RUN/final_ema.pt ]; then echo "$RUN has ended (final_ema.pt exists)"; else
  RES=""; [ -f $R/ckpt/$RUN/last.pt ] && RES="--resume"
  $PY -m bb.cascade_train --run $RUN --run-id 101 --L 14 --H 512 --res skip --mup-base 0 --micro 2 --compile --workers 2 --log-every 200 \
      --max-steps 160000 --monitor 0.003:100:120,0.0055:100:110 --monitor-every 1e7 --converge 3e7:0.02 \
      --max-minutes $MM $RES || echo "exit $?"
fi
