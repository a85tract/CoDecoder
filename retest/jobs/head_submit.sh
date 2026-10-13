#!/bin/bash
# jobs/head_submit.sh STEP: submits the output-head comparison's jobs on Alpine (run on the login node, after the beacon).
# Environment (RETEST.md section head): RT = the deployed freeze tree (real run <cluster-scratch>/retest/code/freeze),
# RT_BEACON and RT_SEEDS = the beacon record and the seeds file (real run <cluster-scratch>/retest/run/{beacon,seeds}.json),
# RT_OUT = where the run writes (real run <cluster-scratch>/retest/out; the head run's files are under $RT_OUT/head).
# The dry run uses the dry seeds (retest/dry/{beacon,seeds}_dry.json), RT_DRY=1 and its own RT, RT_OUT. Every job has
# --nice=1000 (also an #SBATCH line of head_job.sbatch). Prints the job ids, one line per job.
#   draw   the test-set draw (acpu): 100 surface + 50 BB72 chunks, about 1 minute. The plan runs it, then commits the draw
#          manifests, before any read is submitted.
#   reads  after that commit: the 14 surface reads (two H200 2g.35gb slices, 7 networks each) and the 9 BB72 reads (one
#          RTX PRO 6000 1g.24gb slice) on the new test sets, the two fallbacks (acpu), and the two analyses, each
#          depending on its code's reads and fallback (afterok; the analysis jobs are killed if one fails). AFTER=<job id>
#          adds a dependency of the reads and fallbacks on that job (the dry run: the draw). RELAY_CPUS (default 32)
#          sets the Relay-BP job's cores and RELAY_T its time limit (default 01:30:00): 0.130 s per shot, 18.1 core-hours in all,
#          so 32 cores take 35-50 minutes, 16 cores about 70 (then RELAY_T=02:00:00), 64 cores about 18.
#   all    draw, then reads after it (the dry run's driver)
set -euo pipefail
for v in RT RT_BEACON RT_SEEDS RT_OUT; do [ -n "${!v:-}" ] || { echo "$v is not set"; exit 1; }; done
J=$RT/retest/jobs/head_job.sbatch
X="ALL,RT=$RT,RT_BEACON=$RT_BEACON,RT_SEEDS=$RT_SEEDS,RT_OUT=$RT_OUT${RT_DRY:+,RT_DRY=1}"
D=${AFTER:+-d afterok:$AFTER}
sub() { sbatch --parsable --nice=1000 "$@"; }
CPU="-p acpu --qos=cpu-normal"
draw() {
  sub $CPU -c 4 --mem=16G -t 00:30:00 -J head_draw --export=$X,CMD=head_draw.sh $J
}
reads() {
  local s0 s1 b pm rl
  s0=$(sub $D -p ah200 --qos=gpu-normal --gres=gpu:h200_2g.35gb:1 -c 8 --mem=48G -t 01:00:00 -J head_reads_s0 \
         --export=$X,CMD=head_reads.sh,CODE=surface,PARTS=2,PART=0 $J)
  s1=$(sub $D -p ah200 --qos=gpu-normal --gres=gpu:h200_2g.35gb:1 -c 8 --mem=48G -t 01:00:00 -J head_reads_s1 \
         --export=$X,CMD=head_reads.sh,CODE=surface,PARTS=2,PART=1 $J)
  b=$(sub $D -p artxpro6000 --qos=gpu-normal --gres=gpu:rtx_pro_6000_1g.24gb:1 -c 8 --mem=48G -t 00:45:00 -J head_reads_b \
        --export=$X,CMD=head_reads.sh,CODE=bb72 $J)
  pm=$(sub $D $CPU -c 32 --mem=64G -t 00:30:00 -J head_pm --export=$X,CMD=head_fallback_pm.sh $J)
  rl=$(sub $D $CPU -c ${RELAY_CPUS:-32} --mem=48G -t ${RELAY_T:-01:30:00} -J head_relay --export=$X,CMD=head_fallback_relay.sh $J)
  echo "reads_s0 $s0"; echo "reads_s1 $s1"; echo "reads_bb72 $b"; echo "pm $pm"; echo "relay $rl"
  echo "analysis_surface $(sub --kill-on-invalid-dep=yes -d afterok:$s0:$s1:$pm $CPU -c 4 --mem=32G -t 00:30:00 -J head_an_s \
        --export=$X,CMD=head_analysis.sh,CODE=surface $J)"
  echo "analysis_bb72 $(sub --kill-on-invalid-dep=yes -d afterok:$b:$rl $CPU -c 4 --mem=32G -t 00:30:00 -J head_an_b \
        --export=$X,CMD=head_analysis.sh,CODE=bb72 $J)"
}
case "${1:-}" in
  draw) echo "draw $(draw)" ;;
  reads) reads ;;
  all) a=$(draw); echo "draw $a"; AFTER=$a; D="-d afterok:$a"; reads ;;
  *) echo "usage: head_submit.sh draw|reads|all"; exit 1 ;;
esac
