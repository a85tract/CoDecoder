# Stage 2, C6' (run C6_dfm, run id 202) continued on Alpine after its part on the Windows workstation's RTX A6000 (the
# workstation's job C6p, stopped right after a 30-minute last.pt). The same tree (c69800c1) and the same
# arguments as w3d_C6_dfm.sh, except two that cascade_train's FREE list allows to change: --max-minutes (until the Slurm
# job's end less 10 minutes: the segment pauses with last.pt, exit 3, and the next chained segment resumes it) and the
# plain launcher (python -m; wsl_train.py's Inductor patch exists only for WSL2's pinned-memory limit and changes neither
# kernels nor numerics). --resume continues the moved last.pt: the step, the shot stream (seeds by step), the schedule,
# the EMA, Muon's and Lion's state and the monitoring log that the stopping rule (--converge 3e7:0.02) reads. It never
# starts from scratch: without last.pt it writes ATTENTION and stops.
# The end step: once the run has ended (final_ema.pt, exit 0), copy final_ema.pt to $W4/final_ema.pt (through a
# temporary name), check the copy's sha256 against the original, write $W4/SHA256SUMS and DONE in the run directory. A
# segment that leaves the run unable to go on (a pause or a non-finite loss with no next segment queued, a second
# non-finite loss, any other error) writes ATTENTION there with the reason. A segment after DONE does nothing.
T=$R/code.20261009_042038    # tree c69800c1 (C6''s code.c69800c1 on the Windows workstation; C14''s and S55''s)
cd $T/ising_compare; export PYTHONPATH=$T/ising_compare:$T/qldpc_neural; echo "pinned tree $(cat ../DEPLOYED)"
RUN=C6_dfm; D=$R/ckpt/$RUN; W4=<cluster-scratch>/cascade_w4/ckpt/$RUN; LOG=$R/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.out
attn() { echo "$(date -Is) job $SLURM_JOB_ID: $1" | tee -a $D/ATTENTION; }
nxt() { squeue -h -u <user> -n $SLURM_JOB_NAME -t PENDING -o %i 2>/dev/null | head -1 || true; }
if [ -f $D/DONE ]; then echo "$RUN has ended (DONE exists)"
elif [ ! -f $D/last.pt ]; then mkdir -p $D; attn "no last.pt in $D: not started (C6' never starts from scratch)"
elif [ -f $D/nonfinite ] && [ "$(wc -l < $D/nonfinite)" -ge 2 ]; then attn "two non-finite losses in $D/nonfinite: not started"
else
  rc=0
  if [ ! -f $D/final_ema.pt ]; then
    $PY -m bb.cascade_train --run $RUN --run-id 202 --family depspam_fm --L 6 --H 512 --res skip --mup-base 0 --micro 2 \
        --compile --workers 2 --log-every 200 --max-steps 160000 --monitor 0.003:100:120,0.0055:100:110 \
        --monitor-every 1e7 --converge 3e7:0.02 --ckpt-minutes 30 --max-minutes $MM --resume || rc=$?
    echo "== cascade_train exit $rc $(date -Is)"
  fi
  case $rc in
    0) if [ ! -f $D/final_ema.pt ]; then attn "exit 0 without final_ema.pt"
       else
         s=$(sha256sum < $D/final_ema.pt | cut -d' ' -f1); mkdir -p $W4
         if cp $D/final_ema.pt $W4/final_ema.pt.tmp && mv $W4/final_ema.pt.tmp $W4/final_ema.pt \
            && [ "$(sha256sum < $W4/final_ema.pt | cut -d' ' -f1)" = "$s" ]; then
           echo "$s  final_ema.pt" > $W4/SHA256SUMS
           { echo "C6' ($RUN, run id 202) ended $(date -Is) in job $SLURM_JOB_ID"
             grep -E '^\{"(rule|status)"' $LOG | tail -2 || true
             echo "final_ema.pt sha256 $s, copied to $W4/final_ema.pt with the same sha256 ($W4/SHA256SUMS)"; } > $D/DONE
           cat $D/DONE
         else attn "the copy of final_ema.pt (sha256 $s) to $W4 failed or differs"; fi
       fi ;;
    3) if [ -n "$(nxt)" ]; then echo "paused; the next segment $(nxt) resumes last.pt"
       else attn "paused at the segment's end with no next segment queued: submit one more with this job file"; fi ;;
    4) echo "$(date -Is) job $SLURM_JOB_ID" >> $D/nonfinite
       if [ "$(wc -l < $D/nonfinite)" -ge 2 ]; then attn "a second non-finite loss: stopped (cancel the remaining segments by id; report it)"
       elif [ -n "$(nxt)" ]; then echo "non-finite loss; the next segment $(nxt) retries from the last good last.pt"
       else attn "a non-finite loss and no next segment queued: submit one more with this job file (it retries from last.pt)"; fi ;;
    *) attn "cascade_train exit $rc (see $LOG)" ;;
  esac
fi
