# Sourced by every job file of the evaluation (surface/eval/jobs/e*.sh), inside scripts/gpu.sbatch (R, T, I, V, PY set).
# DRY=1 RUN=name: a dry run (seed kind 13 only, small sizes, $R/eval_dry/$RUN); otherwise the evaluation ($R/eval),
# which runs only in the tree pinned at submission (scripts/submit.sh exports TREE) and deployed from a clean commit
# (finding 3 of the stage gate of 2026-10-12; every process also compares its code's digest with the lock's).
# One kernel cache for every job of a run (pinned caches, item 8); SV_GPU_DRIVER goes into every record.
if [ -n "${DRY:-}" ]; then E=$R/eval_dry/${RUN:?RUN names the dry run}; D=--dry; else E=$R/eval; D=; fi
if [ -z "${DRY:-}" ]; then
  [ "${TREE:-}" = "$T" ] || { echo "the evaluation runs in the tree pinned at submission (TREE), not in $T"; exit 1; }
  case "$(cat $T/DEPLOYED)" in *+dirty*) echo "$T was deployed from a dirty checkout"; exit 1 ;; esac
fi
mkdir -p $E
export TRITON_CACHE_DIR=$E/kcache/triton TORCHINDUCTOR_CACHE_DIR=$E/kcache/inductor
export SV_GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
LOCK=$E/locked_d13.json GATE=$E/gate_d13.json
# bb's venv, copied once per job to the node's local disk and bound over its own path in the container: the same
# interpreter, sys.prefix and files (item 10's identity does not change), but imports read the local disk. On GPFS a
# cold import of torch took 140 s and torch.compile's imports minutes more (2026-10-11), in every fresh process, and a
# compute node read GPFS at 14-16 MB/s. The copy comes from $R/venv/<venv>.tar.zst (one file, 3.4 GB, made on a login
# node) when it exists, else file by file; it is refused if its site-packages listing differs from the venv's.
if [ -z "${SV_VENV_STAGED:-}" ]; then
  VL=/tmp/sv_venv_$SLURM_JOB_ID VT=$R/venv/$(basename $V).tar.zst
  t0=$(date +%s); mkdir -p $VL; trap "rm -rf $VL" EXIT; trap "exit 143" TERM
  if [ -f $VT ]; then zstd -dcq $VT | tar -C $VL -xf -; SRC="$VT ($(stat -c '%s bytes, %y' $VT))"
  else tar -C $(dirname $V) -cf - $(basename $V) | tar -C $VL -xf -; SRC=$V; fi
  SP=lib/python3.12/site-packages
  if [ "$(ls $V/$SP | sha256sum)" != "$(ls $VL/$(basename $V)/$SP | sha256sum)" ]; then
    echo "the staged venv's site-packages differ from $V: rebuild $VT"; exit 1
  fi
  cp $I/pyfull.sif $VL/pyfull.sif                   # the container too (352 MB), read at every process start
  export SV_VENV_STAGED="$V staged from $SRC, container from $I/pyfull.sif, in $(( $(date +%s) - t0 )) s"
  PY="apptainer exec --nv -B <cluster-scratch> -B $VL/$(basename $V):$V $VL/pyfull.sif $V/bin/python"
  echo "venv staged: $SV_VENV_STAGED"
fi
run() {  # a step of the job: its output (warnings dropped, lines cut at 1,500 characters) and its exit status, which
         # run returns (finding 1): under the job's set -e a failed step ends the job as failed, so afterok never follows
  echo "== $(date +%H:%M:%S) $*"
  set +e
  "$@" 2>&1 | grep --line-buffered -vE 'UserWarning|warnings.warn' | awk '{print substr($0, 1, 1500); fflush()}'
  local rc=${PIPESTATUS[0]}
  set -e
  echo "== rc $rc $(date +%H:%M:%S)"
  return $rc
}
refuse() {  # the dry runs' refusal checks: a step that must fail; its last lines are shown, and success fails the job
  echo "== $(date +%H:%M:%S) must refuse: $*"
  set +e
  "$@" 2>&1 | grep -vE 'UserWarning|warnings.warn' | tail -n 2 | cut -c1-700
  local rc=${PIPESTATUS[0]}
  set -e
  if [ $rc -eq 0 ]; then echo "== NOT REFUSED"; return 1; fi
  echo "== refused, rc $rc"
}
lockchunks() {  # the lock's confirmation chunks at p, as a-b
  $PY -c "import json; q = json.load(open('$LOCK'))['p']['$1']; print('%d-%d' % tuple(q['conf_chunks']))"
}
