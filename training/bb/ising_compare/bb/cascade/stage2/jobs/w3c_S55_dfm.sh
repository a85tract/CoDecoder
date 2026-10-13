# Stage 2, S55' (run S55_dfm, run id 210): S55's recipe and stopping rule (w3b_S55.sh) in the depspam_fm circuit
# family (--family depspam_fm: the noise-model fit's variant, the family's own seeds): ConvLines-BB
# with r11/r12's architecture and recipe (--samples 2e8 --eval-every 1e7 --workers 6 --monitor 0 --compile, bb.train's
# defaults) at p = 0.55 %, Cascade's curriculum (0.2 % for the first 1 % of the budget, a linear anneal over the next
# 1 %), shots from the training stage's namespace of the family (--w3-stream), the fixed monitor block of C14' (the family's kind 1,
# --w3-monitor), budget_ema.pt at 2e8 samples, then the floor LR (2 %) up to 4e8 samples with --converge 3e7:0.02.
# Chained segments as C14''s.
T=$R/code.20261009_042038    # every segment runs tree c69800c1 (checked by an Alpine job), whatever is deployed later
cd $T/ising_compare; export PYTHONPATH=$T/ising_compare:$T/qldpc_neural; echo "pinned tree $(cat ../DEPLOYED)"
RUN=S55_dfm
export BB_CKPT=$R/ckpt BB_OUT=$R/out/bb
if [ -f $R/ckpt/$RUN/final_ema.pt ]; then echo "$RUN has ended (final_ema.pt exists)"; else
  RES=""; [ -f $R/ckpt/$RUN/last.pt ] && RES="--resume"
  $PY -m bb.train --run $RUN --run-id 210 --family depspam_fm --code bb72 --p 0.0055 --design data/bb/design_bb72.json \
      --samples 2e8 --eval-every 1e7 --workers 6 --monitor 0 --compile --w3-stream --p-curriculum 0.002:0.01:0.01 \
      --w3-monitor 0.003:100:120,0.0055:100:110 --extend-to 4e8 --converge 3e7:0.02 --max-minutes $MM $RES || echo "exit $?"
fi
