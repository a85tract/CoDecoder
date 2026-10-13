# Stage 2, S55'-deep (run S55_dfm_deep, run id 211): S55' (run 210) grown from 3 to 6 trunk blocks of width 128,
# then trained in the depspam_fm family with S55's recipe and stopping rule (w3c_S55_dfm.sh plus --blocks 6 --init).
# bb.train --init loads S55''s final EMA network (ckpt/S55_dfm/final_ema.pt: stopped by the rule at 2e8 samples) into
# the 6-block network: blocks 1-3 copied, blocks 4-6 new with zero up-projections (bb.net's Block initialisation), so the
# grown network starts as S55' itself; the step-0 check (every decision equal and every score within 1e-4 on the
# monitoring block's 0.55 % shots) must pass, or the run refuses to start. Fresh optimiser, schedule and EMA (--init);
# the rest is S55''s: width 128, bottleneck 32, 4 heads, batch 1,024, AdamW 1e-3, warm-up 500, cosine to 2 %, EMA 0.999,
# workers 6, p = 0.55 % with Cascade's curriculum from 0.2 % (the first 1 % of the budget, then a linear anneal over the
# next 1 %), shots from the training stage's namespace of the family (run id 211), the fixed monitoring block of C14' and S55',
# budget_ema.pt at 2e8 samples, then the floor LR up to 4e8 samples with --converge 3e7:0.02. Chained segments as S55''s,
# the first after S55''s last segment (afterany on that segment's job); without S55''s final_ema.pt a segment does nothing.
T=$R/code.20261009_042038    # every segment runs tree c69800c1 (checked by an Alpine job), as S55'
cd $T/ising_compare; export PYTHONPATH=$T/ising_compare:$T/qldpc_neural; echo "pinned tree $(cat ../DEPLOYED)"
RUN=S55_dfm_deep
INIT=$R/ckpt/S55_dfm/final_ema.pt
export BB_CKPT=$R/ckpt BB_OUT=$R/out/bb
if [ -f $R/ckpt/$RUN/final_ema.pt ]; then echo "$RUN has ended (final_ema.pt exists)"
elif [ ! -f $INIT ]; then echo "S55' has no final_ema.pt ($INIT): nothing to grow, not started"
else
  RES=""; [ -f $R/ckpt/$RUN/last.pt ] && RES="--resume"
  echo "init $INIT sha256 $(sha256sum < $INIT | cut -c1-16)"
  $PY -m bb.train --run $RUN --run-id 211 --family depspam_fm --code bb72 --p 0.0055 --design data/bb/design_bb72.json \
      --samples 2e8 --eval-every 1e7 --workers 6 --monitor 0 --compile --w3-stream --p-curriculum 0.002:0.01:0.01 \
      --w3-monitor 0.003:100:120,0.0055:100:110 --extend-to 4e8 --converge 3e7:0.02 --blocks 6 --init $INIT \
      --max-minutes $MM $RES || echo "exit $?"
fi
