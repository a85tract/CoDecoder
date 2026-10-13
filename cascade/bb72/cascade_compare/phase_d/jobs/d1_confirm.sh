# D1-A (PHASE_D1_PLAN.md section 3), on acpu after D1-R1, D1-R2 and D1-S: d1_confirm.py; each run writes
# O/confirm/run_<k> and keeps the earlier ones. Needs F. Optional: O (default $A/out), LOCK (default the tree's).
O=${O:-$A/out}; : "${F:?F, the freeze commit}"
$PY $PD/d1_confirm.py --lock ${LOCK:-$PD/d1_locked_depspam_fm.json} --F $F --read $O/read --compiled $O/compiled \
    --speed $O/speed/rows.jsonl --out $O/confirm
