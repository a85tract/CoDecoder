# Replay check 1 for the eager read: the shim as an identity launcher (RT_SEEDS unset) around D1's own curves.sh / pc_eval.py,
# on D1's first chunk (c0 = 6,433,102) at p = 0.2 % and 0.3 %, compared with the original reads' first 10,000 rows: every array
# (detector rows, observables, both networks' predictions, confidences, logits, failures) and the chunk's seed and digest.
# Then the resume path of the read job: the eager read of the dry run (read/, with its seed trace) is copied to a new directory
# without its compiled read, and d1_read_rt.sh runs there again (dry seeds, 1 chunk): it must pass the guard, skip the eager read,
# pass the seed check on the existing read and write the compiled read, leaving the eager read byte for byte as it was.
# STEP of d1_gpu_job.sh (one full RTX PRO 6000, D1's GPU environment).
set +e
R=${R:-<cluster-scratch>/retest/replay/d1}; A=<cluster-scratch>/cascade_d1
F=310ce8e0f8bfb9d3b97b27186547a836d29fc9ca; C0=$(( 16#${F:0:7} % 9000000 ))
unset RT_SEEDS RT_BEACON
rm -rf $R/read_off; mkdir -p $R/read_off
for P in 0.002 0.003; do
  ( PY="$PY $RT/retest/shim/rt_d1.py"; SETTING=depspam_fm; KIND=14; PGROUPS="$P@$C0:$((C0 + 1))"; OUT=$R/read_off; TAGS="C14p S55p_deep"
    source $PC/jobs/curves.sh ) > $R/read_off_$P.log 2>&1; echo "read_off $P exit $?"
done
$PY $RT/retest/replay/d1/replay_read_off.py $R/read_off $A/out/read $R/replay_read_off.json

# resume path
DRY_B=<cluster-scratch>/retest/dry/beacon_dry.json; DRY_S=<cluster-scratch>/retest/dry/seeds_dry.json
SRC=<cluster-scratch>/retest/dry/out/d1; RES=<cluster-scratch>/retest/dry/out_resume/d1
rm -rf $RES; mkdir -p $RES/0.002; cp -r $SRC/0.002/read $RES/0.002/read; rm -f $RES/0.002/read/seedcheck_0.002.json
before=$(cd $RES/0.002/read && sha256sum p0.00200.npz p0.00200.json block_0.002.json | sha256sum | cut -c1-64)
( export RT_OUT=<cluster-scratch>/retest/dry/out_resume RT_BEACON=$DRY_B RT_SEEDS=$DRY_S RT_DRY_CHUNKS=1 P=0.002; source $RT/retest/jobs/d1_read_rt.sh ) > $R/resume.log 2>&1; st=$?
after=$(cd $RES/0.002/read && sha256sum p0.00200.npz p0.00200.json block_0.002.json | sha256sum | cut -c1-64)
$PY - $R $st $before $after $RES <<'PYEOF'
import json, os, sys
R, st, before, after, res = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
log = open(f"{R}/resume.log").read()
r = {"exit": st, "guard_passed": "guard passed" in log, "eager_read_skipped": "reads: setting" not in log,
     "seedcheck_passed": '"pass": true' in log, "compiled_written": os.path.exists(f"{res}/0.002/compiled/done_0.002"),
     "eager_files_unchanged": before == after, "seedcheck_record": os.path.exists(f"{res}/0.002/read/seedcheck_0.002.json")}
r["pass"] = all(v for k, v in r.items() if k != "exit") and st == 0
json.dump(r, open(f"{R}/replay_resume.json", "w"), indent=1)
print(json.dumps(r))
PYEOF
