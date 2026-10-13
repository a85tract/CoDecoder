# Replay check 2 (resume and repeat protection) and the gate's finding 4 for D1's guards: d1_guard_rt.sh against D1's own
# d1_guard.sh, on D1's deployed tree (read only) and stub trees. Runs in a CPU job (d1_replay_guard.sbatch); writes
# $R/guard.json. D1's guard with an empty SEARCH (its kind-14 freshness test emptied) is the reference for every check
# that d1_guard_rt keeps: both must give the same status on the same input.
set +e; set -uo pipefail
A=<cluster-scratch>/cascade_d1; T=$A/code.20261011_152939; PD=$T/cascade_compare/phase_d
F=310ce8e0f8bfb9d3b97b27186547a836d29fc9ca
R=${R:-<cluster-scratch>/retest/replay/d1}; W=$R/guard_work
DRY_B=<cluster-scratch>/retest/dry/beacon_dry.json; DRY_S=<cluster-scratch>/retest/dry/seeds_dry.json
rm -rf $W; mkdir -p $W/empty $W/dry $W/real $W/tree_dirty $W/tree_other
echo '{"round": 7000001}' > $W/beacon_real.json; echo '{"beacon_round": 7000001}' > $W/seeds_real.json
echo '{"round": 7000001, "other": 1}' > $W/beacon_real3.json
printf '%s' $F > $W/tree_dirty/DEPLOYED_SHA; printf 'net 64b2e25d d1 %s+dirty' $F > $W/tree_dirty/DEPLOYED
printf '%s' 0000000000000000000000000000000000000000 > $W/tree_other/DEPLOYED_SHA; printf 'net 64b2e25d d1 0000000000000000000000000000000000000000' > $W/tree_other/DEPLOYED
source $PD/jobs/d1_guard.sh
source ${RT:?}/retest/jobs/d1_guard_rt.sh
: > $W/cases.tsv
# case NAME GUARD(orig|rt) EXPECTED_STATUS PATTERN(in the message) TREE P O BEACON SEEDS [DRYCHUNKS]
case_() {
  local name=$1 g=$2 want=$3 pat=$4 tree=$5 p=$6 o=$7 b=$8 s=$9 dc=${10:-} out st
  if [ $g = orig ]; then out=$(SEARCH=$W/empty d1_guard $F $p $tree $o 2>&1); st=$?
  else out=$(RT_BEACON=$b RT_SEEDS=$s RT_DRY_CHUNKS=$dc d1_guard_rt $F $p $tree $o 2>&1); st=$?; fi
  echo "$out" | grep -q -F -- "$pat" || st="$st(pattern '$pat' not in the message)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" $g "$st" $want "$(echo $out | cut -c1-200)" >> $W/cases.tsv
}
mk() { mkdir -p $1/read $1/compiled; }
# 1. the freshness test: D1's guard refuses on D1's own tree and outputs because the original reads name kind 14
out=$(SEARCH=$A d1_guard $F 0.002 $T $W/dry/o1 2>&1); st=$?
echo "$out" | grep -q -F -- "name kind 14" || st="$st(pattern not in the message)"
printf '%s\t%s\t%s\t%s\t%s\n' kind14_freshness_original_guard orig "$st" 1 "$(echo $out | cut -c1-200)" >> $W/cases.tsv
# 2. every check that stays: same status from both guards
for g in orig rt; do
  mk $W/dry/ok;       case_ fresh_dir_passes $g 0 'guard passed' $T 0.002 $W/dry/ok $DRY_B $DRY_S
  mk $W/dry/wrongtree; case_ tree_not_F_refused $g 1 'not F' $W/tree_other 0.002 $W/dry/wrongtree $DRY_B $DRY_S
  mk $W/dry/dirty;    case_ tree_not_clean_refused $g 1 'not a clean tree' $W/tree_dirty 0.002 $W/dry/dirty $DRY_B $DRY_S
  mk $W/dry/both_$g;  touch $W/dry/both_$g/read/done_0.002 $W/dry/both_$g/compiled/done_0.002
                      case_ completed_read_refused $g 1 'already done' $T 0.002 $W/dry/both_$g $DRY_B $DRY_S
  mk $W/dry/readonly_$g; touch $W/dry/readonly_$g/read/done_0.002
                      case_ eager_done_only_continues $g 0 'guard passed' $T 0.002 $W/dry/readonly_$g $DRY_B $DRY_S
done
# 3. the re-test's additions
case_ original_outputs_refused rt 1 "inside D1's original outputs" $T 0.002 $A/out $DRY_B $DRY_S
case_ original_outputs_subdir_refused rt 1 "inside D1's original outputs" $T 0.002 $A/out/dry_x $DRY_B $DRY_S
mk $W/dry/blk; echo "{\"seeds_sha256\": \"$(sha256sum $DRY_S | cut -c1-64)\", \"beacon_sha256\": \"$(sha256sum $DRY_B | cut -c1-64)\"}" > $W/dry/blk/read/block_0.002.json
case_ block_same_seeds_continues rt 0 'guard passed' $T 0.002 $W/dry/blk $DRY_B $DRY_S
mk $W/dry/blk2; echo "{\"seeds_sha256\": \"$(printf x | sha256sum | cut -c1-64)\", \"beacon_sha256\": \"$(sha256sum $DRY_B | cut -c1-64)\"}" > $W/dry/blk2/read/block_0.002.json
case_ block_other_seeds_refused rt 1 'other seeds' $T 0.002 $W/dry/blk2 $DRY_B $DRY_S
case_ seeds_round_differs_from_beacon_refused rt 1 'seeds file is of round' $T 0.002 $W/dry/r1 $DRY_B $W/seeds_real.json
case_ dry_seeds_outside_dry_dir_refused rt 1 'under /dry/' $T 0.002 $W/real/r2 $DRY_B $DRY_S
case_ dry_chunks_with_dry_seeds_passes rt 0 'guard passed' $T 0.002 $W/dry/r3 $DRY_B $DRY_S 1
case_ real_seeds_real_dir_passes rt 0 'guard passed' $T 0.002 $W/real/r4 $W/beacon_real.json $W/seeds_real.json
case_ real_seeds_with_dry_chunks_refused rt 1 'RT_DRY_CHUNKS is set' $T 0.002 $W/real/r5 $W/beacon_real.json $W/seeds_real.json 1
case_ real_seeds_dry_dir_refused rt 1 'cannot write under /dry/' $T 0.002 $W/dry/r6 $W/beacon_real.json $W/seeds_real.json
case_ missing_beacon_refused rt 1 'must name files' $T 0.002 $W/dry/r7 $W/nonexistent.json $DRY_S
# 4. a real-seeds block record is continued only under the same seeds
mk $W/real/blk3; echo "{\"seeds_sha256\": \"$(sha256sum $W/seeds_real.json | cut -c1-64)\", \"beacon_sha256\": \"$(sha256sum $W/beacon_real.json | cut -c1-64)\"}" > $W/real/blk3/read/block_0.002.json
case_ real_block_same_continues rt 0 'guard passed' $T 0.002 $W/real/blk3 $W/beacon_real.json $W/seeds_real.json
case_ real_block_other_beacon_refused rt 1 'other seeds or another beacon' $T 0.002 $W/real/blk3 $W/beacon_real3.json $W/seeds_real.json
${PY:-python3} - $W/cases.tsv $R/guard.json <<'PYEOF'
import json, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
cases = [{"case": n, "guard": g, "status": s, "expected": int(w), "message": m, "ok": s == w} for n, g, s, w, m in rows]
pairs = {}
for c in cases:
    pairs.setdefault(c["case"], {})[c["guard"]] = c["status"]
same = {k: v["orig"] == v["rt"] for k, v in pairs.items() if len(v) == 2}
res = {"cases": cases, "kept_checks_same_status_as_d1_guard": same, "pass": all(c["ok"] for c in cases) and all(same.values()),
       "note": "the original guard is run with an empty SEARCH (its kind-14 freshness test emptied); kind14_freshness_original_guard "
               "runs it with D1's own directory as SEARCH"}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps({"pass": res["pass"], "cases": len(cases), "failed": [c["case"] for c in cases if not c["ok"]]}))
sys.exit(0 if res["pass"] else 1)
PYEOF
