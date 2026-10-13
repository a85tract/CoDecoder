# Phase C step C4b (PHASE_C_PLAN.md 2.4), after the locks and before any confirmation read: per setting, every network
# on CHECK_CHUNKS fresh chunks of seed kind KIND (12) at the claim points, chunks 10-19 (the screen of C3 read chunks
# 0-9 there), through curves.sh (as in C1: ConvLines on the fused engine, the depth-rule networks compiled; detector
# rows kept); the same rows read eagerly by the depth-rule networks; then pc_speed.py check: the frozen timed paths of
# every network alone and of every locked configuration against these reads (a ConvLines read must be bitwise its
# timed paths). The reads run in subshells, so nothing they set leaks into the job.
# Optional: KIND, CK_GROUPS (the reads' groups, default pc_common's), CKD (reads; default $R/out/check), LOCKD, SP,
# SETTINGS_.
CKD=${CKD:-$R/out/check}; CKK=${KIND:-12}; CKG=${CK_GROUPS:-$($PY $PC/pc_common.py groups check)}
CKT="$($PY $PC/pc_common.py tags) $($PY $PC/pc_common.py tags late)"; CKE=$($PY $PC/pc_common.py tags compiled)
for S in ${SETTINGS_:-uniform depspam_fm}; do
  ( SETTING=$S; KIND=$CKK; PGROUPS=$CKG; OUT=$CKD/$S/read; TAGS=$CKT; source $PC/jobs/curves.sh )
  [ -n "$CKE" ] && ( export PC_EAGER=1; SETTING=$S; KIND=$CKK; PGROUPS=$CKG; OUT=$CKD/$S/eager; TAGS=$CKE
                     source $PC/jobs/curves.sh )
  $PY $PC/pc_speed.py check --read $CKD/$S/read ${CKE:+--eager $CKD/$S/eager} --locked ${LOCKD:-$R/out}/locked_$S.json \
      --speed ${SP:-$R/out/speed}/speed_components.json --out ${LOCKD:-$R/out}/check_$S.json
done
