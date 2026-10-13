# Phase C, the routing rule on the development block (PHASE_C_PLAN.md section 4): one lock per setting. Optional:
# DEV, RELAYD, SPEEDC, LOCKD (directories / files), SETTINGS_ (default both), PSEL (dry runs only: a subset of PS).
for S in ${SETTINGS_:-uniform depspam_fm}; do
  $PY $PC/pc_route.py select --dev ${DEV:-$R/out/dev}/$S --relay ${RELAYD:-$R/out/relay_dev}/$S \
      --speed ${SPEEDC:-$R/out/speed/speed_components.json} ${PSEL:+--p $PSEL} --out ${LOCKD:-$R/out}/locked_$S.json
done
