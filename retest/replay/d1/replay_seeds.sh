# Replay checks 2 and 3 of the shim: patch on with the dry seeds (main process, spawn workers, other kinds), and patch off
# (the original function, the original seeds in the workers). STEP of d1_cpu_job.sh. See rt_d1_replay.py.
set +e; set -uo pipefail
R=${R:-<cluster-scratch>/retest/replay/d1}; A=<cluster-scratch>/cascade_d1
PYS="$PY $RT/retest/shim/rt_d1.py"; X=$RT/retest/replay/d1/rt_d1_replay.py
DRY_B=<cluster-scratch>/retest/dry/beacon_dry.json; DRY_S=<cluster-scratch>/retest/dry/seeds_dry.json
unset RT_SEEDS RT_BEACON
RT_BEACON=$DRY_B RT_SEEDS=$DRY_S $PYS $X $R/seeds_on.json --procs 4; st1=$?
$PYS $X $R/seeds_off.json --procs 4 --original-read $A/out/read; st2=$?
echo "seeds_on exit $st1, seeds_off exit $st2"
[ $st1 = 0 ] && [ $st2 = 0 ]
