#!/bin/bash
# The first H100 pod (phase_c/H100_TIMING.md): phase C's speed protocol on a rented H100, a second column for speed only. Runs
# detached on the pod as root, in the tree $Q/code (deployed by scripts/h100_deploy.sh):
#   DEADLINE=<epoch> setsid nohup bash $Q/code/cascade_compare/phase_c/jobs/h100.sh < /dev/null > $Q/h100.log 2>&1 &
# The steps, every process fresh, the networks and configurations interleaved rep-major as C3 and C7 run them:
#   nets       every network alone on its frozen paths (pc_h100.py nets), REPS processes each
#   outputs    the H100 half of the screen: C4b's check shots through every frozen path (pc_h100.py outputs)
#   pipelines  every locked configuration at the claim points and Cascade alone, both modes, REPS processes each
#              (pc_h100.py pipeline = pc_speed.pipeline; 20,000 shots single, 500,000 batched; Relay-BP on this CPU)
#   summarize  h100_speed.json (pc_h100.py summarize; the screen's verdicts join once C4b's read exists)
# The lock and the frozen paths are the primary run's ($R/out, copied from Alpine read-only; their sha256 are checked
# by every process). No process starts after DEADLINE (epoch seconds). Every process appends "epoch step status" to
# $O/progress, which the pull loop turns into the heartbeat. At the end $O/SHA256SUMS covers every result file, then
# $O/END: complete, or how many processes the deadline skipped.
set -uo pipefail
Q=${Q:-/opt/h100t}; T=$(readlink -f $Q/code); PC=$T/cascade_compare/phase_c; O=${O:-$Q/out}
R=<cluster-scratch>/cascade_w4
REPS=${REPS:-3}; DEADLINE=${DEADLINE:?epoch seconds after which no process starts}
PY=$Q/venv/bin/python
SP=$R/out/speed/speed_components.json
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 BB_PIDNS=1 \
       QND_RELAY=<cluster-scratch>/bb/relay TRITON_CACHE_DIR=$Q/cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$Q/cache/inductor PC_TREE="$(cat $T/DEPLOYED)" \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
mkdir -p $O $Q/cache/triton $Q/cache/inductor
cd $T/ising_compare
skipped=0
prog() { echo "$(date +%s) $*" >> $O/progress; }
go() {  # go STEP CMD...: one fresh process, unless the deadline has passed
  local s=$1; shift
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then skipped=$((skipped + 1)); prog "$s skipped (deadline)"; return; fi
  if "$@"; then prog "$s ok"; else prog "$s exit=$?"; fi
}
$PY - > $O/env.json <<'EOF'
import json, os, platform, subprocess
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print(json.dumps({"card_role": "H100 (secondary, speed only)", "host": platform.node(),
                  "gpu": sh("nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,"
                            "power.limit --format=csv,noheader"),
                  "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"), "cpus": os.cpu_count(),
                  "python": platform.python_version(), "tree": os.environ["PC_TREE"],
                  "venv_freeze_sha256": sh("sha256sum /opt/h100t/venv_freeze.txt")[:64],
                  "deadline": int(os.environ["DEADLINE"]), "reps": int(os.environ.get("REPS", "3"))}, indent=1))
EOF
prog "start tree=$PC_TREE deadline=$DEADLINE"
for rep in $(seq 1 $REPS); do
  for t in C14 C6 S55 C14p S55p S55p_deep r11 r12 C6p; do
    case $t in C14p|C6p|S55p|S55p_deep) S=depspam_fm;; *) S=uniform;; esac
    go "nets $t rep $rep" $PY $PC/pc_h100.py nets --net $t --rep $rep --locked $R/out/locked_$S.json --speed $SP \
       --out $O/rows.jsonl
  done
done
for S in uniform depspam_fm; do
  go "outputs $S" $PY $PC/pc_h100.py outputs --locked $R/out/locked_$S.json --speed $SP --out $O/outputs/$S
done
for rep in $(seq 1 $REPS); do
  for S in uniform depspam_fm; do
    L=$R/out/locked_$S.json
    for p in 0.002 0.003; do
      for i in -1 $($PY -c "import json; L = json.load(open('$L')); print(' '.join(str(i) for i, c in enumerate(L['configs']) if c['p'] == $p))"); do
        for m in single batched; do
          [ $m = single ] && N=20000 || N=500000
          go "pipeline $S $p $i $m rep $rep" $PY $PC/pc_h100.py pipeline --locked $L --config $i --p $p --mode $m \
             --rep $rep --n $N --speed $SP --out $O/rows.jsonl
        done
      done
    done
  done
done
$PY $PC/pc_h100.py summarize --rows $O/rows.jsonl --speed $SP --env $O/env.json \
    --locked $R/out/locked_uniform.json $R/out/locked_depspam_fm.json --out $O/h100_speed.json > $O/summary.txt 2>&1 \
  || prog "summarize exit=$?"
prog "end skipped=$skipped"
(cd $O && find . -type f ! -name SHA256SUMS ! -name END -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
if [ $skipped = 0 ]; then echo complete > $O/END; else echo "deadline: $skipped processes skipped" > $O/END; fi
