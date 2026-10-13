# Phase C step C2 (PHASE_C_PLAN.md section 4.2): Relay-BP on the routed union of every candidate, both settings.
# Optional: DEV and RELAYD (default $R/out/dev and $R/out/relay_dev).
for S in uniform depspam_fm; do
  MODE=dev; READ=${DEV:-$R/out/dev}/$S; OUTR=${RELAYD:-$R/out/relay_dev}/$S; source $PC/jobs/relay.sh
done
unset MODE READ OUTR
