# The development reads (SURFACE_PLAN 4.4; kind 10: chunks 0-99 at 0.6 %, 0-999 at 0.35 %): S on the engine at 4,096,
# B on its frozen amortized path at b*, M's decode timed shot by shot on this node's CPU; then the rule and the lock
# (select). The lock is committed only after the gate (e3, e4).
source $T/surface/eval/jobs/common.sh
if [ -n "${DRY:-}" ]; then DV="0.006:2000-2020 0.0035:2000-2050"
  SEL="--dev-chunks 0.006:2000-2020,0.0035:2000-2050 --dry-conf 0.006:5,0.0035:10"
else DV="0.006:0-100 0.0035:0-1000"; SEL=""; fi
for x in $DV; do
  run $PY -m surface.sv_read $D --stage dev --p ${x%%:*} --chunks ${x#*:} --comp SB,Bb,MS --speed $E/speed.json \
      --out $E/dev
done
run $PY -m surface.sv_route select $D --dev $E/dev --speed $E/speed.json $SEL --out $LOCK
