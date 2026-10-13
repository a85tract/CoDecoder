# Phase C, Relay-BP on the confirmation block and the one analysis of it (PHASE_C_PLAN.md section 4), per setting.
for S in ${SETTINGS_:-uniform depspam_fm}; do
  LOCKED=${LOCKD:-$R/out}/locked_$S.json; READ=${CONF:-$R/out/conf}/$S; OUTR=${RELAYC:-$R/out/relay_conf}/$S; MODE=conf
  source $PC/jobs/relay.sh
  $PY $PC/pc_route.py confirm --conf $READ --relay $OUTR --locked $LOCKED --out ${LOCKD:-$R/out}/confirm_$S.json
done
