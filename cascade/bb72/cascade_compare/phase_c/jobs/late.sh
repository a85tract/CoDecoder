# Phase C step C1b (PHASE_C_PLAN.md section 3): C6' on the development block of both settings, once its weights are on
# Alpine. The same seeds give the same rows; pc_report refuses to pair them if a chunk's detector digest differs.
# KIND and PGROUPS keep curves.sh's defaults (kind 10, the development block) unless the caller sets them.
for SETTING in uniform depspam_fm; do
  TAGS=${LATE_TAGS:-C6p}; OUT=${LATE_OUT:-$R/out/dev_late}/$SETTING; source $PC/jobs/curves.sh
done
unset TAGS OUT
