# The dry run's analysis (a STEP of d1_cpu_job.sh): (1) d1_confirm_rt.sh as it will run on the real reads: on the dry reads
# (1 chunk per p) it must refuse, because the lock's blocks are 1,100 and 1,700 chunks (exit 3); (2) d1_confirm.py through
# the shim with a copy of D1's lock whose blocks are 1 chunk and the timing rows re-bound to that copy (a dry-only
# derivation: only "chunks" and the rows' locked_sha256 change), to the tables. Not a result: 10,000 shots.
set +e
ROOT=${RT_OUT:?}/d1; F=310ce8e0f8bfb9d3b97b27186547a836d29fc9ca
PYS="$PY $RT/retest/shim/rt_d1.py"
( source $RT/retest/jobs/d1_confirm_rt.sh ) > $ROOT/dry_confirm_refusal.log 2>&1; echo "d1_confirm_rt.sh exit $? (expected 3)"
tail -4 $ROOT/dry_confirm_refusal.log | cut -c1-220
rm -rf $ROOT/dry_lock $ROOT/confirm_dry; mkdir -p $ROOT/dry_lock
$PY - $PD/d1_locked_depspam_fm.json <cluster-scratch>/cascade_d1/out/speed/rows.jsonl $ROOT/dry_lock <<'PYEOF'
import hashlib, json, sys
lock, rows, out = sys.argv[1:4]
L = json.load(open(lock))
L["chunks"] = {k: 1 for k in L["chunks"]}
open(f"{out}/lock.json", "w").write(json.dumps(L, indent=1))
sha = hashlib.sha256(open(f"{out}/lock.json", "rb").read()).hexdigest()
with open(f"{out}/rows.jsonl", "w") as fh:
    for line in open(rows):
        r = json.loads(line)
        if r.get("role") == "pipeline":
            r["locked_sha256"] = sha
        fh.write(json.dumps(r) + "\n")
open(f"{out}/lock_sha256.txt", "w").write(sha + "\n")
print("dry lock sha256", sha)
PYEOF
$PYS $PD/d1_confirm.py --lock $ROOT/dry_lock/lock.json --F $F --read $ROOT/confirm_in/read --compiled $ROOT/confirm_in/compiled \
     --speed $ROOT/dry_lock/rows.jsonl --out $ROOT/confirm_dry > $ROOT/dry_confirm_tables.log 2>&1; echo "dry d1_confirm.py exit $?"
tail -6 $ROOT/dry_confirm_tables.log | cut -c1-220
