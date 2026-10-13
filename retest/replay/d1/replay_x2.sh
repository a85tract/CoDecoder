# Replay check 1 for the X2 adapter: STEP of an X2-environment job (d1_replay_x2.sbatch): PYX is X2's python.
# (1) rt_x2.py check: the NPZ_SHA and REG derived from D1's committed records (npz_sha256.txt, d1_result.json) equal the ones
#     bb_d1.py hard-codes;  (2) rt_x2.py run on D1's original reads and compiled reads: the per-shot files, summary.json and fig.npz
#     equal X2's published outputs (REV-x2, Alpine x2/out/bb; summary.json also the hub's revision/x2/data/bb/summary.json).
set +e; set -uo pipefail
R=${R:-<cluster-scratch>/retest/replay/d1}; A=<cluster-scratch>/cascade_d1
X=<cluster-scratch>/x2/out/bb; REC=$RT/retest/replay/d1/d1_orig_npz_sha256.txt; RES=$A/out/confirm/run_1/d1_result.json
CMP="$PYX $RT/retest/replay/d1/rt_d1_compare.py"
rm -rf $R/x2_orig; mkdir -p $R/x2_orig
X2_OUT=$R/x2_orig $PYX $RT/retest/shim/rt_x2.py check --result $RES --npz-sha $REC > $R/x2_check.json; st0=$?
$PYX $RT/retest/shim/rt_x2.py run --inputs $A/out --result $RES --npz-sha $REC --out $R/x2_orig > $R/x2_orig.log 2>&1; st1=$?
st=0
for f in p0.00200.npz p0.00300.npz fig.npz; do $CMP npz $X/$f $R/x2_orig/bb/$f $R/compare_x2_$f.json; st=$((st + $?)); done
$CMP json $X/summary.json $R/x2_orig/bb/summary.json $R/compare_x2_summary.json.json; st=$((st + $?))
$PYX - $R $st0 $st1 $st <<'PYEOF'
import json, sys
R, st0, st1, st = sys.argv[1], *map(int, sys.argv[2:])
res = {"check_exit": st0, "check": json.load(open(f"{R}/x2_check.json")), "run_exit": st1,
       "compare": {f: json.load(open(f"{R}/compare_x2_{f}.json")) for f in ("p0.00200.npz", "p0.00300.npz", "fig.npz", "summary.json")}}
res["compare"] = {k: {x: v[x] for x in ("pass", "bytes_identical", "sha256_A", "sha256_B")} for k, v in res["compare"].items()}
res["pass"] = st0 == 0 and st1 == 0 and st == 0
json.dump(res, open(f"{R}/replay_x2.json", "w"), indent=1)
print(json.dumps({"pass": res["pass"], "check": st0, "run": st1, "compare_failures": st}))
PYEOF
