#!/bin/bash
# ARB3's timing of the joint decision at d = 13 (unregistered; TIMING.md): one drive.py r4 job on this machine's card with
# --joint-both --only .C5. .C2.: per repetition and p (0.6 %, 0.35 %), Cascade alone (C2), the composition (C5) and C5
# with the joint step, single and batched, one fresh process each, 3 repetitions, gate on, no burn. As stage 1 built the
# H100's d = 13 caches (h100_pod.sh kcache r4): a dry lock, an untimed build pass with the dry seeds (SRT_KCACHE_BUILD),
# verify passes until one finds the caches unchanged, then the timed run on the joint lock derived from that session.
# PHASE=prep runs the untimed part (the joint step's kernel takes w as a run-time argument, so the caches do not depend on
# w; prep's passes use W13 = 1.0); PHASE=timed checks that the caches are still the settled session's and runs the timed
# job with the frozen W13 into $OUTD/drive_<stamp>; PHASE=all does both.
# Caller: R (the release root, patched), PY, OUTD (the run's dir), CARD (rtx|h100), K (the kernel-cache dir, kept),
# D13ROOT (CASCADE_ROOT), W13 (the joint w per p), DRY (the dir with beacon_dry.json and seeds_dry_arb3.json), PHASE.
set -euo pipefail
: "${R:?} ${PY:?} ${OUTD:?} ${CARD:?} ${K:?} ${D13ROOT:?} ${W13:?} ${DRY:?} ${PHASE:?}"
T=$R/cascade/d13; CAS=$R/speed_rt/cascade; LOCK=$T/out/locked_d13.json
[ $PHASE = timed ] || [ ! -e "$OUTD" ] || { echo "$OUTD exists"; exit 7; }
mkdir -p $OUTD $K/triton $K/inductor $D13ROOT/ckpt/CS13
ln -sf $R/weights/cascade_d13/CS13/final_ema.pt $D13ROOT/ckpt/CS13/final_ema.pt
cd $T
export PYTHONPATH=$T OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CONVLINES_RELEASE=$R/evaluation \
  CASCADE_ROOT=$D13ROOT NCCL_IB_DISABLE=1 TRITON_CACHE_DIR=$K/triton TORCHINDUCTOR_CACHE_DIR=$K/inductor SRT_BURN_S=0 \
  SV_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" SRT_CARD=$CARD \
  SRT_JOINT_VENV=1 SRT_SETS_FILE=$CAS/sets_arb3.json
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.default_limit --format=csv,noheader
echo "host $(hostname) cpu $(lscpu | grep 'Model name' | sed 's/.*: *//') W13 $W13 out $OUTD $(date -Is)"
JA=(--joint-w13 "$W13" --joint-both --only .C5. .C2.)
sess() { if [ $CARD = h100 ]; then echo "SRT_SESSION=$1"; else echo "SRT_SESSION="; fi; }
if [ $PHASE != timed ]; then
# untimed: the dry lock, session 0, its joint lock, the build pass (dry seeds, digest check off)
$PY $CAS/srtlib.py dry-lock --lock $LOCK --out $OUTD/dry_lock.json --chunks 2 --single 2000 --batched 20000
$PY $CAS/srt_d13.py session --out $OUTD/session_0.json
$PY $CAS/srt_d13.py joint-lock --lock $OUTD/dry_lock.json --session $OUTD/session_0.json --out $OUTD/lock_0.json
t0=$(date +%s)
env $(sess $OUTD/session_0.json) SRT_DRY=1 SRT_KCACHE_BUILD=1 SRT_BEACON=$DRY/beacon_dry.json SRT_SEEDS=$DRY/seeds_dry_arb3.json \
  $PY $CAS/drive.py r4 --tree $T --out $OUTD/build --lock $OUTD/lock_0.json --card $CARD --gate off --reps 1 "${JA[@]}"
echo "build pass in $(( $(date +%s) - t0 )) s"
ok=""
for i in 1 2 3; do
  $PY $CAS/srt_d13.py session --out $OUTD/session_$i.json
  $PY $CAS/srt_d13.py joint-lock --lock $OUTD/dry_lock.json --session $OUTD/session_$i.json --out $OUTD/lock_$i.json
  if env $(sess $OUTD/session_$i.json) SRT_DRY=1 SRT_BEACON=$DRY/beacon_dry.json SRT_SEEDS=$DRY/seeds_dry_arb3.json \
      $PY $CAS/drive.py r4 --tree $T --out $OUTD/verify_$i --lock $OUTD/lock_$i.json --card $CARD --gate off --reps 1 \
      "${JA[@]}"; then ok=$i; break; fi
  echo "verify pass $i changed the caches or failed"
done
[ -n "$ok" ] || { echo "the d = 13 caches did not settle"; exit 1; }
cp $OUTD/session_$ok.json $OUTD/session_settled.json
echo "caches settled after verify pass $ok: $(grep -A3 '"kcache"' $OUTD/session_settled.json | tr -d ' \n')"
fi
[ $PHASE != prep ] || { date -Is; echo PREPDONE; exit 0; }
# timed: the caches must still be the settled session's; the study's lock with the session's code, caches (and venv; on
# the H100 card and venv); the beacon's new seeds
S=$OUTD/session_settled.json; ST=$(date +%Y%m%d_%H%M%S)
$PY $CAS/srt_d13.py session --out $OUTD/session_now_$ST.json
cmp -s <(grep -A2 '"kcache"' $OUTD/session_now_$ST.json) <(grep -A2 '"kcache"' $S) || { echo "the caches are not the settled session's"; exit 1; }
$PY $CAS/srt_d13.py joint-lock --lock $LOCK --session $S --out $OUTD/lock_timed_$ST.json
unset SRT_DRY SRT_KCACHE_BUILD
env $(sess $S) SRT_BEACON=$CAS/beacon.json SRT_SEEDS=$CAS/seeds_arb3.json \
  $PY $CAS/drive.py r4 --tree $T --out $OUTD/drive_$ST --lock $OUTD/lock_timed_$ST.json --card $CARD "${JA[@]}"
date -Is; echo JOBDONE
