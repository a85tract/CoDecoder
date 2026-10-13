# D1-P (PHASE_D1_PLAN.md 2.4), on acpu before the freeze: d1_power.py on the tree's lock -> OP/power_<job>.json
OP=${OP:-$A/out/power}; mkdir -p $OP
$PY $PD/d1_power.py --lock ${LOCK:-$PD/d1_locked_depspam_fm.json} --out $OP/power_${SLURM_JOB_ID:-manual}.json
