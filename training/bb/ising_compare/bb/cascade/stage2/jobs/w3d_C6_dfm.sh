# Stage 2, C6' (run C6_dfm, run id 202) on the Windows workstation's RTX A6000 (job C6p, started by its launcher
# with this file): C6's network and recipe (w3b_C6.sh: Cascade L 6, H 512, the literal paper recipe --res skip --mup-base 0,
# two micro-batches of 1,664 shots) in the depspam_fm family, with C14''s curriculum and stopping rule (w3c_C14_dfm.sh):
# the paper's schedule over 80,000 steps (budget_ema.pt at 2.66e8 samples), then the floor LR at 0.55 % up to 160,000
# steps, the family's fixed monitoring block (kind 1, chunks 100-119 at 0.3 %, 100-109 at 0.55 %) every 1e7 samples,
# and --converge 3e7:0.02. No Slurm limit here: one process for the whole run (--max-minutes 0), last.pt every 30
# minutes. The loop resumes last.pt after a pause (exit 3) and retries once after a non-finite loss (exit 4), as the
# next Slurm segment does on Alpine; a second non-finite loss or any other error ends the job. Started again after an
# interruption (same command), it resumes last.pt; once final_ema.pt exists it does nothing. Training runs through
# wsl_train.py, the WSL2 launcher (Inductor's autotuner keeps its argument copies on the GPU, since
# WSL2 refuses pinned host allocations of 1 GiB; the pinned code is unchanged).
T=$Q/code.c69800c1    # tree c69800c1, C14''s and S55''s
cd $T/ising_compare; export PYTHONPATH=$T/ising_compare:$T/qldpc_neural; echo "pinned tree $(cat ../DEPLOYED)"
RUN=C6_dfm
A="--run $RUN --run-id 202 --family depspam_fm --L 6 --H 512 --res skip --mup-base 0 --micro 2 --compile --workers 2 \
   --log-every 200 --max-steps 160000 --monitor 0.003:100:120,0.0055:100:110 --monitor-every 1e7 --converge 3e7:0.02 \
   --ckpt-minutes 30 --max-minutes 0"
nf=0
for n in 1 2 3 4 5 6; do
  if [ -f $R/ckpt/$RUN/final_ema.pt ]; then echo "$RUN has ended (final_ema.pt exists)"; exit 0; fi
  RES=""; [ -f $R/ckpt/$RUN/last.pt ] && RES="--resume"
  echo "== attempt $n $(date -Is) $RES"
  $PY $Q/bin/wsl_train.py bb.cascade_train $A $RES; rc=$?
  echo "== attempt $n exit $rc $(date -Is)"
  case $rc in
    0) exit 0 ;;
    3) ;;
    4) nf=$((nf + 1)); [ $nf -lt 2 ] || { echo "a second non-finite loss: stopped (report it)"; exit 4; } ;;
    *) exit $rc ;;
  esac
done
exit 5
