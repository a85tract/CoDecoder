# Phase C reads (PHASE_C_PLAN.md sections 3-4): networks on the same fresh rows of SETTING through pc_eval.py (the
# ConvLines networks on their fused engine; the Cascade networks through bb.cascade_eval's BF16 evaluation backend, C6
# and C6' under torch.compile unless PC_EAGER=1 is exported), per-shot outputs and detector rows kept. Sourced by a job
# file that sets SETTING and may set
#   KIND    seed kind (default 10: the curves and routing development block)
#   OUT     output directory (default $R/out/dev/$SETTING)
#   PGROUPS  'p,p,...@a:b' chunk groups (default the development block of pc_common.DEV_CHUNKS)
#   TAGS    the networks (default every one whose checkpoint exists, LATE ones excepted)
# A group whose marker OUT/done_<p,...> exists is skipped, so a resubmitted job continues.
KIND=${KIND:-10}; OUT=${OUT:-$R/out/dev/$SETTING}
PGROUPS=${PGROUPS:-$($PY $PC/pc_common.py groups dev)}
TAGS=${TAGS:-$($PY $PC/pc_common.py tags)}
CK=$($PY $PC/pc_common.py paths $TAGS)
mkdir -p $OUT
echo "reads: setting $SETTING kind $KIND networks $TAGS groups $PGROUPS -> $OUT"
for g in $PGROUPS; do
  ps=${g%%@*}; ch=${g##*@}
  [ -f "$OUT/done_$ps" ] && { echo "skip $ps (done)"; continue; }
  $PY $PC/pc_eval.py --family $SETTING --ckpt $CK --prec bf16 --kind $KIND --p $ps --chunks $ch --procs 8 \
      --save $OUT --save-det --out $OUT/rows_$ps.json && touch "$OUT/done_$ps"
done
