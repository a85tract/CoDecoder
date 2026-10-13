# Replay check 1 for D1's analysis, and the shim's negative control. STEP of d1_cpu_job.sh (D1's cpu.sbatch environment).
# (1) d1_confirm.py through rt_d1.py with the patch off (RT_SEEDS unset), on D1's original reads, compiled reads and
#     timing rows, into a new directory: its d1_result.json and d1_tables.md must equal run_1's of D1's own run.
# (2) the same with the dry seeds switched on: the original reads' seeds are not the beacon's, so the analysis must refuse
#     (exit 3, "a chunk's seed is not eval_seed", at both p) and write nothing.
set +e; set -uo pipefail
R=${R:-<cluster-scratch>/retest/replay/d1}; A=<cluster-scratch>/cascade_d1
F=310ce8e0f8bfb9d3b97b27186547a836d29fc9ca; LOCK=$PD/d1_locked_depspam_fm.json
PYS="$PY $RT/retest/shim/rt_d1.py"; CMP="$PY $RT/retest/replay/d1/rt_d1_compare.py"
DRY_B=<cluster-scratch>/retest/dry/beacon_dry.json; DRY_S=<cluster-scratch>/retest/dry/seeds_dry.json
unset RT_SEEDS RT_BEACON
rm -rf $R/confirm_orig $R/confirm_negctl; mkdir -p $R/confirm_orig $R/confirm_negctl
args="--lock $LOCK --F $F --read $A/out/read --compiled $A/out/compiled --speed $A/out/speed/rows.jsonl"
$PYS $PD/d1_confirm.py $args --out $R/confirm_orig > $R/confirm_orig.log 2>&1; st1=$?
$CMP json $A/out/confirm/run_1/d1_result.json $R/confirm_orig/run_1/d1_result.json $R/compare_d1_result.json; c1=$?
$CMP text $A/out/confirm/run_1/d1_tables.md $R/confirm_orig/run_1/d1_tables.md $R/compare_d1_tables.json; c2=$?
RT_BEACON=$DRY_B RT_SEEDS=$DRY_S $PYS $PD/d1_confirm.py $args --out $R/confirm_negctl > $R/confirm_negctl.log 2>&1; st2=$?
nref=$(grep -c "is not eval_seed" $R/confirm_negctl.log); nrun=$(ls -d $R/confirm_negctl/run_* 2>/dev/null | wc -l)
$PY - $R $st1 $c1 $c2 $st2 $nref $nrun <<'PYEOF'
import json, sys
R, st1, c1, c2, st2, nref, nrun = sys.argv[1], *map(int, sys.argv[2:])
res = {"patch_off_analysis_exit": st1, "d1_result_json_equal": c1 == 0, "d1_tables_md_byte_identical": c2 == 0,
       "compare_d1_result": json.load(open(f"{R}/compare_d1_result.json")),
       "negative_control_patch_on_exit": st2, "negative_control_refusals_of_seed_check": nref, "negative_control_runs_written": nrun,
       "negative_control_expected": "exit 3, one 'is not eval_seed' refusal per p, nothing written"}
res["pass"] = st1 == 0 and c1 == 0 and c2 == 0 and st2 == 3 and nref == 2 and nrun == 0
json.dump(res, open(f"{R}/replay_confirm.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("pass", "d1_result_json_equal", "d1_tables_md_byte_identical", "negative_control_patch_on_exit")}))
PYEOF
