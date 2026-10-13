# Stage 2, S55 (run id 110): ConvLines-BB with r11/r12's architecture and recipe (bb/scripts/alpine/args/s2_train.txt:
# --samples 2e8 --eval-every 1e7 --workers 6 --monitor 0 --compile, bb.train's defaults) at p = 0.55 %, with Cascade's
# curriculum (0.2 % for the first 1 % of the budget, a linear anneal over the next 1 %), shots from the training stage's namespace
# (--w3-stream), the same fixed monitor block as C14 and C6 (--w3-monitor), budget_ema.pt at 2e8 samples, then the floor LR
# (2 %) up to 4e8 samples with the same stopping rule (--converge 3e7:0.02). Chained segments as C14's.
T=$R/code.20261009_001049    # every segment runs tree b142837e (the checked one), whatever is deployed later
cd $T/ising_compare; export PYTHONPATH=$T/ising_compare:$T/qldpc_neural; echo "pinned tree $(cat ../DEPLOYED)"
RUN=S55
export BB_CKPT=$R/ckpt BB_OUT=$R/out/bb
if [ -f $R/ckpt/$RUN/final_ema.pt ]; then echo "$RUN has ended (final_ema.pt exists)"; else
  RES=""; [ -f $R/ckpt/$RUN/last.pt ] && RES="--resume"
  $PY -m bb.train --run $RUN --run-id 110 --code bb72 --p 0.0055 --design data/bb/design_bb72.json --samples 2e8 \
      --eval-every 1e7 --workers 6 --monitor 0 --compile --w3-stream --p-curriculum 0.002:0.01:0.01 \
      --w3-monitor 0.003:100:120,0.0055:100:110 --extend-to 4e8 --converge 3e7:0.02 --max-minutes $MM $RES || echo "exit $?"
fi
