#!/bin/bash
# The second H100 pod (H100_TIMING.md, "Amendment 1"): the configurations Amendment 1 changed at the claim points, timed on a
# rented H100 as jobs/h100.sh timed the first lock, with Cascade alone beside them in every (setting, p) group; then the
# H100 half of the screen on the new lock, the summary, and the batched-decoding profile of ising-cascade-batchopt.
# Runs detached on the pod as root, in the tree $Q/code (scripts/h100_deploy.sh with Q=/opt/h100b INPUTS=1) and, for
# the profile, $Q/batchopt/code (batchopt/pod_deploy.sh with Q=/opt/h100b/batchopt):
#   DEADLINE_T=<epoch> DEADLINE=<epoch> HARD=<epoch> setsid nohup bash $Q/code/cascade_compare/phase_c/jobs/h100_a1.sh \
#       < /dev/null > $Q/h100b.log 2>&1 &
# The steps, every process fresh, in jobs/h100.sh's order (its "nets" are not rerun):
#   outputs    the H100 half of the screen on the new lock (pc_h100.py outputs, per setting)
#   pipelines  per rep (REPS = 3, rep-major), setting and claim point: Cascade alone (--config -1) and every changed
#              configuration (CH below, RUN.md "The new lock"; checked against the two locks' picks first), single
#              (20,000 shots) and batched (500,000), pc_h100.py pipeline = pc_speed.pipeline on Amendment 1's lock
#   summarize  h100_speed.json on the new lock (the screen's verdicts join on Alpine, h100_screen_a1.sbatch)
#   prof       batchopt/prof_run.sh's steps (prof_batched.py; ncu and nsys through prof_run.sh) into $O/prof
# No pipeline or outputs process starts after DEADLINE_T, no profile step after DEADLINE, and HARD cuts a running
# profile step. Every process appends "epoch step status" to $O/progress, which the pull loop turns into the heartbeat.
# At the end $O/SHA256SUMS covers every result file, then $O/END.
set -uo pipefail
Q=${Q:-/opt/h100b}; T=$(readlink -f $Q/code); PC=$T/cascade_compare/phase_c; O=${O:-$Q/out}
R=<cluster-scratch>/cascade_w4; LD=$R/out/amend1
REPS=${REPS:-3}; DEADLINE_T=${DEADLINE_T:?}; DEADLINE=${DEADLINE:?}; HARD=${HARD:?}
PY=$Q/venv/bin/python
SP=$R/out/speed/speed_components.json
declare -A CH=([uniform 0.002]="2 3" [uniform 0.003]="7 8" [depspam_fm 0.002]="2 3 4" [depspam_fm 0.003]="7 9")
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 BB_PIDNS=1 \
       QND_RELAY=<cluster-scratch>/bb/relay TRITON_CACHE_DIR=$Q/cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$Q/cache/inductor PC_TREE="$(cat $T/DEPLOYED)" \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
mkdir -p $O $Q/cache/triton $Q/cache/inductor
cd $T/ising_compare
skipped=0
prog() { echo "$(date +%s) $*" >> $O/progress; }
go() {  # go STEP CMD...: one fresh process, unless DEADLINE_T has passed
  local s=$1; shift
  if [ "$(date +%s)" -ge "$DEADLINE_T" ]; then skipped=$((skipped + 1)); prog "$s skipped (deadline)"; return; fi
  if "$@"; then prog "$s ok"; else prog "$s exit=$?"; fi
}
# The changed configurations: those whose pick (S, B, q1, r, q, score, thresholds) differs between the locks
for S in uniform depspam_fm; do
  for p in 0.002 0.003; do
    got=$($PY -c "import json, sys
o, n = (json.load(open(f))['configs'] for f in sys.argv[1:3]); p = float(sys.argv[3])
K = ('type', 'p', 'S', 'B', 'q1', 'r', 'q', 'score', 'tau')
assert [(c['type'], c['p']) for c in o] == [(c['type'], c['p']) for c in n]
print(' '.join(str(i) for i, c in enumerate(n) if c['p'] == p and any(c.get(k) != o[i].get(k) for k in K)))" \
      $R/out/locked_$S.json $LD/locked_$S.json $p)
    [ "$got" = "${CH[$S $p]}" ] && continue
    prog "changed configurations $S $p: '$got', expected '${CH[$S $p]}' exit=1"
    (cd $O && sha256sum progress > SHA256SUMS); echo "stopped: the changed configurations differ" > $O/END; exit 1
  done
done
$PY - > $O/env.json <<'EOF'
import json, os, platform, subprocess
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
R = "<cluster-scratch>/cascade_w4/out"
print(json.dumps({"card_role": "H100 (secondary, speed only)", "host": platform.node(),
                  "gpu": sh("nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,"
                            "power.limit --format=csv,noheader"),
                  "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"), "cpus": os.cpu_count(),
                  "python": platform.python_version(), "tree": os.environ["PC_TREE"],
                  "venv_freeze_sha256": sh("sha256sum /opt/h100b/venv_freeze.txt")[:64],
                  "lock": f"Amendment 1 (629ca1c4), {R}/amend1",
                  "locked_sha256": {s: sh(f"sha256sum {R}/amend1/locked_{s}.json")[:64] for s in ("uniform", "depspam_fm")},
                  "changed": {"uniform": {"0.002": [2, 3], "0.003": [7, 8]},
                              "depspam_fm": {"0.002": [2, 3, 4], "0.003": [7, 9]}},
                  "deadline_timing": int(os.environ["DEADLINE_T"]), "deadline": int(os.environ["DEADLINE"]),
                  "reps": int(os.environ.get("REPS", "3"))}, indent=1))
EOF
prog "start tree=$PC_TREE deadline_t=$DEADLINE_T deadline=$DEADLINE hard=$HARD"
for S in uniform depspam_fm; do
  go "outputs $S" $PY $PC/pc_h100.py outputs --locked $LD/locked_$S.json --speed $SP --out $O/outputs/$S
done
for rep in $(seq 1 $REPS); do
  for S in uniform depspam_fm; do
    for p in 0.002 0.003; do
      for i in -1 ${CH[$S $p]}; do
        for m in single batched; do
          [ $m = single ] && N=20000 || N=500000
          go "pipeline $S $p $i $m rep $rep" $PY $PC/pc_h100.py pipeline --locked $LD/locked_$S.json --config $i --p $p \
             --mode $m --rep $rep --n $N --speed $SP --out $O/rows.jsonl
        done
      done
    done
  done
done
$PY $PC/pc_h100.py summarize --rows $O/rows.jsonl --speed $SP --env $O/env.json \
    --locked $LD/locked_uniform.json $LD/locked_depspam_fm.json --out $O/h100_speed.json > $O/summary.txt 2>&1 \
  && prog "summarize ok" || prog "summarize exit=$?"
# The profile: batchopt/prof_run.sh's steps in its order, in the batchopt tree, each in a fresh process
BT=$(readlink -f $Q/batchopt/code); BP=$BT/cascade_compare/batchopt
NCU=${NCU:-$(command -v ncu || ls /usr/local/cuda/bin/ncu 2>/dev/null || true)}; NSYS=${NSYS:-$(command -v nsys || true)}
export PYTHONPATH=$BT/ising_compare:$BT/qldpc_neural:$BT/cascade_compare/phase_c PC_TREE="$(cat $BT/DEPLOYED)" \
       PYBIN=$PY RUN="" NCU NSYS
OP=$O/prof; mkdir -p $OP; cd $BT/ising_compare
for s in shots env work s55 c14 pipe report ncu nsys report; do
  now=$(date +%s)
  if [ $now -ge $DEADLINE ] || { [ $s = ncu ] && [ $((HARD - now)) -lt 2400 ]; }; then
    skipped=$((skipped + 1)); prog "prof $s skipped (deadline)"; continue
  fi
  case $s in
    ncu|nsys) O=$OP STEPS=$s timeout -k 30 $((HARD - now)) bash $BP/prof_run.sh > $Q/prof_step.log 2>&1; rc=$? ;;
    *) echo "== $s start $(date +%T)" > $Q/prof_step.log
       timeout -k 30 $((HARD - now)) $PY $BP/prof_batched.py $s --out $OP >> $Q/prof_step.log 2>&1; rc=$?
       echo "== $s exit=$rc $(( $(date +%s) - now )) s" >> $Q/prof_step.log ;;
  esac
  cat $Q/prof_step.log >> $Q/prof.log
  bad=$(grep -oE '^== [a-z0-9_]+ exit=[1-9][0-9]*' $Q/prof_step.log | tr '\n' ' ')
  if [ $rc -ne 0 ] || [ -n "$bad" ]; then prog "prof $s exit=$rc ${bad}"; else prog "prof $s ok"; fi
done
cp $Q/prof.log $OP/prof.log 2>/dev/null
prog "end skipped=$skipped"
(cd $O && find . -type f ! -name SHA256SUMS ! -name END -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
if [ $skipped = 0 ]; then echo complete > $O/END; else echo "deadline: $skipped processes skipped" > $O/END; fi
