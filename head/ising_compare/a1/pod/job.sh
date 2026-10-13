#!/bin/bash
# job.sh CODE ARM SEED [RAMP]: one A1 training run on a pod, then its validation record and the list of its outputs.
#  1. a1/jobs/args/train_surface.txt or train_bb72.txt as written (2e8 samples from the seed's registered trunk file,
#     the not-trained check at 2e7, RAMP=1 the ramp rerun), under job.sbatch's set -euo pipefail, in its environment
#     with the pod's paths (env.sh; squeue answers from shim/); the args files resume from the run's own checkpoints
#     when the job starts again (the surface's newest state_p*.pt, BB72's last.pt). $WORKERS sampler workers a run
#     (default 2: as fast as 8 on the H100s, revision/A1_POD_RUN.md section 3; the stream is the same for any count);
#  2. a1.podval: the validation record of final_ema.pt (val_legacy d = 9 or BB72 dev, 200,000 shots; never test),
#     identity-checked, renamed into place only when its batch-one check passes;
#  3. $A/done/<run>.sha256: the sha256 of every output the pull loop copies (the run directory, the trainer's csv or
#     log, the job's log, the record, the pod's fingerprint), then $A/done/<run>.ok.
set -u
source /workspace/a1/code/ising_compare/a1/pod/env.sh
CODE=$1; export ARM=$2 S=$3 RAMP=${4:-0} WORKERS=${WORKERS:-2}
pre=a1; [ "$RAMP" = 1 ] && pre=a1ramp
run=${pre}_${ARM}_s$S
case $CODE in surface) CMD=train_surface.txt; C=$ICMP_CKPT ;; bb72) CMD=train_bb72.txt; C=$BB_CKPT ;; *) exit 2 ;; esac
[ -f $A/done/$run.ok ] && { echo "$run has its record"; exit 0; }
export SLURM_JOB_ID=pod-$(cat $A/POD_NAME 2>/dev/null || hostname)
echo "job $run on $SLURM_JOB_ID gpu '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)' tree $(cat $A/code/DEPLOYED) cmd $CMD workers $WORKERS $(date -Is)"
( set -euo pipefail; eval "$(cat a1/jobs/args/$CMD)" ) || { echo "TRAIN FAILED rc=$? $(date -Is)"; exit 1; }
[ -f $C/$run/final_ema.pt ] || { echo "TRAIN FAILED: no final_ema.pt $(date -Is)"; exit 1; }
echo "trained $(date -Is)"
$PY -m a1.podval --code $CODE --ckpt $C/$run/final_ema.pt --run $run --arm $ARM --seed $S --out $A/val/$run.npz ||
  { echo "VALREC FAILED $(date -Is)"; exit 1; }
echo "JOB OK $run $(date -Is)"                       # the job log's last line, before its sha256 is taken
cd $A && { find ckpt/$CODE/$run -type f ! -name '*.tmp'; ls val/$run.npz logs/$run.out fingerprint.json 2>/dev/null
           ls out/$CODE/train_logs/$run.csv 2>/dev/null; } |
  LC_ALL=C sort | xargs sha256sum > done/$run.sha256.tmp && mv done/$run.sha256.tmp done/$run.sha256 && touch done/$run.ok
