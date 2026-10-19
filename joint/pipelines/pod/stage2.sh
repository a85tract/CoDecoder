#!/bin/bash
# After the BB72 checks: the d = 13 checks (w = 1.0, a placeholder until the parallel run's fit) and the joint step's
# micro-benchmarks (checks/bench_joint.py), one process at a time.
source "$(dirname "$0")/pod_env.sh"
while pgrep -f "pod/run_bb.sh" > /dev/null; do sleep 10; done
[ "${BENCH_ONLY:-0}" = 1 ] || { bash $W/pod/run_d13.sh 0.006 1.0; bash $W/pod/run_d13.sh 0.0035 1.0; }
mkdir -p $W/out/bench
cd $BB/ising_compare
for P in 0.002 0.003; do
  WJ=$([ $P = 0.002 ] && echo 6.309573444801933 || echo 3.758374042884441)
  FR=$([ $P = 0.002 ] && echo 0.041 || echo 0.055)
  PYTHONPATH=$BBPATH $PY $W/pod/podrun.py $W/checks/bench_joint.py bb --p $P --w $WJ --frac $FR \
    --out $W/out/bench/bb_$P.json > $W/out/logs/bench_bb_$P.log 2>&1
  rc=$?; echo "$(date -Is) bench bb $P rc $rc" >> $W/out/status.txt
done
cd $R/cascade/d13
for P in 0.006 0.0035; do
  PYTHONPATH=$R/cascade/d13 CONVLINES_RELEASE=$R/evaluation CASCADE_ROOT=$W/d13root $PY $W/pod/podrun13.py \
    $W/checks/bench_joint.py d13 --p $P --w 1.0 --lock $W/out/d13/dry_lock.json --out $W/out/bench/d13_$P.json \
    > $W/out/logs/bench_d13_$P.log 2>&1
  rc=$?; echo "$(date -Is) bench d13 $P rc $rc" >> $W/out/status.txt
done
echo "$(date -Is) STAGE2_DONE" >> $W/out/status.txt
