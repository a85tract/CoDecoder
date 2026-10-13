#!/bin/bash
# Milestone 2's H100 session (M2.md section 9), after m2_pod_deploy.sh: the venv once (Alpine's bb/venv_cu128 pins,
# req_cu128.txt, and relay-bp 0.2.2 as the first H100 pod's h100_setup_venv.sh installs it; `m2_pod_run.sh venv` builds only
# that), then m2_run.sh's steps with the pod's paths, detached so that a dropped ssh session does not stop it:
#   setsid nohup bash /opt/b2m2/code/cascade_compare/batchopt/m2_pod_run.sh < /dev/null > /opt/b2m2/run.log 2>&1 &
# STEPS (default: screen ptx tailnets tail nets pipe report), REPS (3), TAIL_B (512), TAIL_PATHS (compiled@512, as the
# RTX PRO 6000's tailnets job ran it), PTX_ARGS (default: S55 at the six tiers' batches, as the RTX PRO 6000's formal
# job CJ55 ran it) and the other *_ARGS of m2_run.sh pass through. Every timed process refuses any card but the
# H100 80GB HBM3 (M2_CARD=H100) and labels its rows "H100 (secondary, speed only)"; BB_PIDNS=1 is
# bb.timing.gpu_watch's rule for a container's PID namespace (the second H100 pod's run). The outputs go to $Q/out/<time> ($Q/run.out
# names it, $Q/run.pid holds this script's pid) with env/ (review finding 6: the card and MIG mode, the CPU, its quota
# and the Relay-BP workers, the driver, the power limits, nvidia-smi's clock samples every 10 s through the run, the
# ptxas Triton calls, the packages against the pins, the code's and the inputs' sha256, every check and exit code).
# The first failed step ends the run (m2_run.sh) and leaves FAILED (first line: the reason); END is written only after
# every output STEPS makes is there, after SHA256SUMS.
set -uo pipefail
Q=${Q:-/opt/b2m2}; T=$(readlink -f $Q/code); B=$T/cascade_compare/batchopt
PYBIN=${PYBIN:-$Q/venv/bin/python}
venv() {   # once; venv_freeze.txt marks a finished build
  [ -x $PYBIN ] && [ -s $Q/venv_freeze.txt ] && return 0
  local PIP="$Q/venv/bin/pip --disable-pip-version-check"
  rm -rf $Q/venv && python3.12 -m venv $Q/venv && $PIP install -q pip==25.0.1 && $PIP install -q --index-url \
    https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r $B/req_cu128.txt \
    && $PIP install -q --no-deps relay-bp==0.2.2 && $Q/venv/bin/pip freeze > $Q/venv_freeze.txt
}
if [ "${1:-}" = venv ]; then venv && echo "venv ok" && exit 0; echo "venv failed"; exit 1; fi

O=${O:-$Q/out/$(date +%Y%m%d_%H%M%S)}; E=$O/env
mkdir -p $E $Q/cache/triton $Q/cache/inductor
echo $O > $Q/run.out.tmp && mv -f $Q/run.out.tmp $Q/run.out && echo $$ > $Q/run.pid.tmp && mv -f $Q/run.pid.tmp $Q/run.pid
SAMPLER=""; trap '[ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null' EXIT
sums() { [ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null; SAMPLER=""; sleep 1
         (cd $O && find . -type f ! -name SHA256SUMS ! -name END ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
          > SHA256SUMS); }
fail() { echo "FAILED: $1"; sums; echo "$1" > $O/FAILED; exit 1; }
venv || fail "venv failed"
export PYBIN RUN="" O REFS=<cluster-scratch>/cascade_batchopt/out/m2_refs \
       STEPS=${STEPS:-"screen ptx tailnets tail nets pipe report"} TAIL_B=${TAIL_B:-512} REPS=${REPS:-3} \
       TAIL_PATHS=${TAIL_PATHS:-compiled@512} PTX_ARGS=${PTX_ARGS:-"--net S55 --batches 4096,1024,256,64,16,1"} \
       M2_CARD=H100 BB_PIDNS=1
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c:$B \
       OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=<cluster-scratch>/bb/relay \
       TRITON_CACHE_DIR=$Q/cache/triton TORCHINDUCTOR_CACHE_DIR=$Q/cache/inductor PC_TREE="$(cat $T/DEPLOYED)" \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.limit --format=csv,noheader
echo "tree $PC_TREE out $O python $PYBIN ($($PYBIN -V 2>&1)) steps $STEPS cpus $(env -u OMP_NUM_THREADS nproc)" \
     "cpu $(lscpu | sed -n 's/^Model name: *//p')"

# --- the record (review finding 6), before the first step
{ date -Is; echo "tree $PC_TREE"; nvidia-smi -L
  nvidia-smi --query-gpu=name,pci.bus_id,driver_version,vbios_version,mig.mode.current,mig.mode.pending,memory.total,\
clocks.max.sm,clocks.max.mem,clocks.applications.graphics,power.limit,enforced.power.limit,power.default_limit,\
power.max_limit,persistence_mode,compute_mode,ecc.mode.current,pstate --format=csv
  lscpu | grep -E '^(Model name|CPU\(s\)|On-line|Thread|Core|Socket|NUMA node\(s\)|CPU max MHz)'
  echo "nproc $(env -u OMP_NUM_THREADS nproc); cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo -);" \
       "Relay-BP workers (pc_common.NCPU_RELAY) $(sed -n 's/^NCPU_RELAY = \([0-9]*\).*/\1/p' \
       $T/cascade_compare/phase_c/pc_common.py); OMP_NUM_THREADS $OMP_NUM_THREADS"
  free -g | head -2; uname -srm; sed -n 's/^PRETTY_NAME=//p' /etc/os-release
  echo "python $($PYBIN -V 2>&1); $($PYBIN -m pip --version | cut -d' ' -f1-2)"
  env | grep -E '^(STEPS|REPS|TAIL_[A-Z]*|[A-Z]*_ARGS|[A-Z]*_EXTRA|M2_CARD|BB_PIDNS|OMP_NUM_THREADS|PYTHONPATH|O|REFS|'\
'TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR|QND_RELAY|PC_GPU_DRIVER|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)=' | sort
} > $E/run_env.txt 2>&1
nvidia-smi -q > $E/nvidia_smi_q.txt 2>&1
$PYBIN - > $E/ptxas_version.txt 2>&1 <<'PY'
import subprocess, torch, triton
from triton.backends.nvidia import compiler as c
cc = torch.cuda.get_device_capability()
t = c.get_ptxas(cc[0] * 10 + cc[1])
print(f"arch sm_{cc[0]}{cc[1]}; torch {torch.__version__}; CUDA {torch.version.cuda}; triton {triton.__version__}")
print(torch.cuda.get_device_properties(0))
print(f"ptxas Triton calls: {t.path} (release {t.version})")
print(subprocess.run([t.path, "--version"], capture_output=True, text=True).stdout.strip())
PY
$PYBIN -m pip freeze --all > $E/pip_freeze.txt 2>&1
(cd $T && find . -type f ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum) > $E/code_sha256.txt
(cd / && xargs -d '\n' -a $Q/inputs.list sha256sum) > $E/pod_inputs.sha256 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,clocks.max.sm,power.draw,enforced.power.limit,temperature.gpu,\
temperature.memory,utilization.gpu,clocks_event_reasons.active --format=csv -l 10 > $E/clock_samples.csv 2>&1 &
SAMPLER=$!
{ echo "inputs at the run's start = the deploy's list: $(cmp -s $E/pod_inputs.sha256 $Q/inputs_deploy.sha256 \
    && echo ok || echo DIFFER)"
  $PYBIN - $B/req_cu128.txt $E/pip_freeze.txt <<'PY'
import re, sys
n = lambda s: re.sub(r"[-_.]+", "-", s.split("==")[0].strip().lower()) + "==" + s.split("==")[1].strip()
req, fr = ({n(x) for x in open(f) if "==" in x and not x.lstrip().startswith("#")} for f in sys.argv[1:3])
print("venv pins (req_cu128.txt) installed:", "ok" if req <= fr else "MISSING " + " ".join(sorted(req - fr)))
PY
} > $E/checks.txt 2>&1
cat $E/checks.txt
grep -q DIFFER $E/checks.txt && fail "the inputs on the pod differ from the deploy's sha256 list"

# --- the steps
cd $T/ising_compare
bash $B/m2_run.sh 2>&1 | tee $O/m2_run.log; rc=${PIPESTATUS[0]}
nvidia-smi -q > $E/nvidia_smi_q_end.txt 2>&1

# --- every check and exit code, then the outputs STEPS must have made
$PYBIN - $O >> $E/checks.txt 2>&1 <<'PY'
import collections, json, os, re, sys
o = sys.argv[1]
jl = lambda f: [json.loads(x) for x in open(os.path.join(o, f))] if os.path.exists(os.path.join(o, f)) else []
for x in open(os.path.join(o, "m2_run.log")):
    if re.match(r"== .+ exit=-?\d+ ", x) or x.startswith("== run stopped") or x.startswith("== unknown"):
        print("step", x[3:].strip())
for w in ("nets", "pipe"):
    d = jl(f"drive_{w}.jsonl")
    if d:
        runs = [r for r in d if "rc" in r]
        c = collections.Counter((r["tag"], r["rc"]) for r in runs)
        print(f"drive {w}: {len(runs)} processes, (tag, exit code) counts {dict(sorted(c.items()))};",
              "clock-flagged groups", [r["clock_flagged_groups"] for r in d if "clock_flagged_groups" in r])
s = jl("screen.jsonl")
if s:
    bad = [(r["net"], r["cfg"], r.get("batch", r.get("buffers_n"))) for r in s if not r.get("ok") and not r.get("control")]
    print(f"screen: {len(s)} rows, {sum(1 for r in s if 'error' in r)} raised, {len(bad)} non-control rows differ"
          f" (bitwise {'yes' if not bad else 'no'}): {bad[:40]}")
for f in sorted(os.listdir(o)):
    if f.startswith("tail_") and f.endswith(".json"):
        t = json.load(open(os.path.join(o, f)))
        print(f, {k: t.get(k) for k in ("ok_bounds", "ok_queue", "ok_tolerance")})
PY
missing=""
for s in $STEPS; do
  case $s in
    screen) want=screen.jsonl ;; ptx) want=ptx.json ;; tailnets) want=tailnets.jsonl ;;
    tail) want=$(for x in ${TAIL_SETTINGS:-uniform depspam_fm}; do echo tail_$x.json; done) ;;
    nets) want="nets.jsonl drive_nets.jsonl" ;; pipe) want="pipe.jsonl drive_pipe.jsonl" ;;
    report) want="summary.md summary.json" ;; *) want="" ;;
  esac
  for f in $want; do [ -s $O/$f ] || missing="$missing $f"; done
  [ $s = ptx ] && { ls $O/ptx/*.ptx.gz > /dev/null 2>&1 || missing="$missing ptx/*.ptx.gz"; }
done
for f in run_env.txt nvidia_smi_q.txt nvidia_smi_q_end.txt ptxas_version.txt pip_freeze.txt code_sha256.txt \
         pod_inputs.sha256 clock_samples.csv checks.txt; do [ -s $E/$f ] || missing="$missing env/$f"; done
echo "m2_run.sh exit=$rc; missing outputs:${missing:- none}" | tee -a $E/checks.txt
[ $rc -eq 0 ] || fail "m2_run.sh exit=$rc: $(grep -h '^== run stopped' $O/m2_run.log | tail -1 | cut -c4-)"
[ -z "$missing" ] || fail "missing outputs:$missing"
sums && echo done > $O/END && echo "END $(date -Is)"
