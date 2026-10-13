#!/bin/bash
# h100_pod.sh MODE [ROUND...]: SPEED-RT's Cascade timings on an H100 80GB HBM3 pod (the registered second card: rounds r1-r4 of
# drive.py exactly as on the RTX PRO 6000, gate on, same pipelines and shots, each round one job on one card), on the pod,
# after the caller has deployed the pinned trees, the inputs, the seeds and the public tree speed_rt/ + surface_v2/ (the
# caller's relay copies them to their paths and checks every file's sha256 at its destination). The pod has outbound
# internet for pip and for uv's CPython (setup only).
#   setup       once: Python 3.12.3 (uv) and the BB harness venv with the pins of the D1 tree's req_cu128.txt + relay-bp 0.2.2
#               (--no-deps); pip freeze; checks (versions, the card, the pins installed)
#   prescreen   the pod's pre-screen, mechanical, on the idle pod: surf_v2.gpucheck's burst measured five times, each judged by the
#               window rule of these rounds (median SM clock >= 97 % of the maximum; every sample's clock-event reasons within
#               GPU idle 0x1, applications clocks 0x2 and SW power cap 0x4; the enforced power limit equals the default; under
#               64 MiB in use before and no other process on the GPU), all five must pass; CPU steal at most 1 % over 5 s on
#               the container's CPUs; container CPU quota at least 20 CPUs (16 worker processes plus the main process); every
#               sample is kept; records CPU model, quota, driver, power limit, maximum SM clock; exit 1 = replace the pod
#   kcache [R..]  untimed, with the DRY seeds: the pod's own Triton / Inductor caches (the lock pins an sm_120 cache; the H100 builds
#               its own): for r1-r3 the seed cache $Q/cache/bb (a drive.py dry pass of each round, gate off), for r4 the d = 13
#               cache $Q/cache/d13 and the session record $Q/session/d13_session.json (srt_d13.py h100-session) after a dry pass
#               whose every process found the cache unchanged. Default rounds: r1 r2 r3 r4
#   run R..     rounds r1..r4 in the order given, each one drive.py job on one card (--card h100, gate on, at most 3 attempts per
#               slot), into $Q/out/ROUND_STAMP/{env/, drive/, drive.log, SHA256SUMS, END}; a second round on the same pod
#               starts after the first has ended (a lock refuses two at once). r1-r3 copy the seed cache into a job-local
#               directory as the RTX jobs do; r4 runs the d = 13 pipelines on a lock derived from the study's lock and the
#               session (srt_d13.py h100-lock) and checks the cache digest at every process's start and end.
# Variables (the caller's; paths are the pod's):
#   Q            the pod's root (default /opt/srt): $Q/venv, $Q/pub (the public tree), $Q/cache, $Q/session, $Q/out
#   D1_TREE, D13_TREE   the pinned trees (r1-r3, r4);  D13_ROOT (CASCADE_ROOT: ckpt/CS13/final_ema.pt);  D13_LOCK the study's lock
#   W4_LOCK, SPEED_JSON   phase C's depspam_fm lock and the speed components (r2);  RELAY_ROOT   QND_RELAY
#   INPUTS_LIST, INPUTS_SHA   the input files' absolute paths and their sha256 on the cluster (sha256sum format), checked first
#   SRT_SEEDS, SRT_BEACON (SRT_DRY=1 for a dry record)   as srtlib.py, for run;  SRT_DRY_SEEDS, SRT_DRY_BEACON   for kcache
#   DRIVE_ARGS   extra drive.py arguments (dry runs: --reps 1 --n-single 2000 --n-batched 20000 --force-bad ...)
#   DRY_LOCK_ARGS   srtlib.py dry-lock arguments of the d = 13 dry runs (default --chunks 2 --single 2000 --batched 20000)
#   SRT_BURN_S   seconds of burn before each window (default 0: registered jobs never burn);  PYVER (default 3.12.3)
set -uo pipefail
MODE=${1:-}; [ $# -gt 0 ] && shift
Q=${Q:-/opt/srt}; PYVER=${PYVER:-3.12.3}; H100="NVIDIA H100 80GB HBM3"
SRT_ROOT=${SRT_ROOT:-$Q/pub}; CAS=$SRT_ROOT/speed_rt/cascade; PY=${PY:-$Q/venv/bin/python}
STAMP=$(date +%Y%m%d_%H%M%S); DRY_LOCK_ARGS=${DRY_LOCK_ARGS:---chunks 2 --single 2000 --batched 20000}
need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "h100_pod.sh $MODE: $v is not set"; exit 2; }; done; }
die() { echo "FAILED: $*"; exit 1; }
smi() { nvidia-smi --query-gpu=$1 --format=csv,noheader,nounits | head -1; }
SAMPLER=""; trap '[ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null' EXIT

# ----------------------------------------------------------------------------------------------------------------- setup
setup() {
  need D1_TREE
  local REQ=$D1_TREE/cascade_compare/batchopt/req_cu128.txt PIP="$Q/venv/bin/pip --disable-pip-version-check"
  mkdir -p $Q/setup
  if ! { [ -x $PY ] && [ -s $Q/venv_freeze.txt ]; }; then
    command -v uv > /dev/null || python3 -m pip install -q uv || die "no uv"
    rm -rf $Q/venv
    uv venv -q --seed --python $PYVER $Q/venv && $PIP install -q pip==25.0.1 && $PIP install -q --index-url \
      https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r $REQ &&
      $PIP install -q --no-deps relay-bp==0.2.2 && $Q/venv/bin/pip freeze > $Q/venv_freeze.txt || die "venv build"
  fi
  cp $Q/venv_freeze.txt $Q/setup/pip_freeze.txt
  $PY - $REQ $Q/venv_freeze.txt $PYVER <<'PY' || die "setup checks"
import re, sys
n = lambda s: re.sub(r"[-_.]+", "-", s.split("==")[0].strip().lower()) + "==" + s.split("==")[1].strip().split("+")[0]
req, fr = ({n(x) for x in open(f) if "==" in x and not x.lstrip().startswith("#")} for f in sys.argv[1:3])
miss = sorted(req - fr)
import platform, torch, triton, numpy, stim, pymatching, relay_bp                    # noqa: E401
ok = not miss and platform.python_version() == sys.argv[3] and torch.cuda.is_available()
print(f"python {platform.python_version()} torch {torch.__version__} triton {triton.__version__} numpy {numpy.__version__} "
      f"stim {stim.__version__} pymatching {pymatching.__version__} card {torch.cuda.get_device_name(0)}")
print("pins installed:", "ok" if not miss else "MISSING " + " ".join(miss))
sys.exit(0 if ok else 1)
PY
  [ "$(smi name)" = "$H100" ] || die "the card is not an $H100"
  echo "setup ok: $Q/venv"
}

# -------------------------------------------------------------------------------------------------------------- prescreen
prescreen() {
  local P=$Q/prescreen_$STAMP k; mkdir -p $P
  echo "pre-screen of $(hostname) -> $P"
  { echo "card: $(nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.limit,enforced.power.limit,power.max_limit --format=csv,noheader)"
    lscpu | grep -E '^(Model name|CPU\(s\)|Socket|Thread)'; } > $P/host.txt
  cat $P/host.txt
  $PY - $P <<'PY' || die "PRE-SCREEN FAILED: CPU steal or quota (see $P/cpu.json)"
import json, os, sys, time
cpus = sorted(os.sched_getaffinity(0))
def snap():
    d = {}
    for ln in open("/proc/stat"):
        if ln.startswith("cpu") and ln[3].isdigit():
            f = ln.split()
            d[int(f[0][3:])] = [int(x) for x in f[1:]]
    return d
a = snap(); time.sleep(5); b = snap()
tot = steal = 0
for c in cpus:
    d = [y - x for x, y in zip(a[c], b[c])]
    tot += sum(d[:8]); steal += d[7]                                  # user nice system idle iowait irq softirq steal
quota = None
try:
    q, per = open("/sys/fs/cgroup/cpu.max").read().split()
    quota = None if q == "max" else int(q) / int(per)
except OSError:
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        quota = q / int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read()) if q > 0 else None
    except OSError:
        pass
cpu_quota = quota if quota is not None else float(len(cpus))
res = dict(cpus_visible=len(cpus), steal_pct_5s=100.0 * steal / max(tot, 1), quota_cpus=quota, quota_or_cpus=cpu_quota,
           steal_ok=100.0 * steal / max(tot, 1) <= 1.0, quota_ok=cpu_quota >= 20.0)
json.dump(res, open(sys.argv[1] + "/cpu.json", "w"), indent=1)
print(f"CPU: {len(cpus)} visible, steal {res['steal_pct_5s']:.3f} % over 5 s (limit 1 %), quota {quota} CPUs (limit 20): "
      f"{'ok' if res['steal_ok'] and res['quota_ok'] else 'FAILED'}")
sys.exit(0 if res["steal_ok"] and res["quota_ok"] else 1)
PY
  for k in 1 2 3 4 5; do
    [ $k = 1 ] || sleep 6
    local pl0; pl0=$(nvidia-smi --query-gpu=enforced.power.limit,power.default_limit --format=csv,noheader,nounits | head -1)
    PYTHONPATH=$SRT_ROOT/surface_v2 $PY -m surf_v2.gpucheck --when prescreen --out $P/check_$k.json > $P/check_$k.log 2>&1
    # gpucheck's own pass/fail (it exits 1 on its stricter throttle test) is not used; its measurement is judged here
    $PY - $P/check_$k.json "$pl0" $P/verdict_$k.json <<'PY' || die "PRE-SCREEN FAILED at GPU check $k of 5: replace this pod before anything runs on it"
import json, sys
d = json.load(open(sys.argv[1]))
lim, default = (float(x) for x in sys.argv[2].replace(" ", "").split(","))
c, ck = d["clocks"], d["checks"]
reasons = sorted({r[2] for r in c.get("rows", []) if r})
bad = [r for r in reasons if int(r, 16) & ~0x7]                 # anything but GPU idle 0x1, applications clocks 0x2, SW power cap 0x4
v = dict(telemetry=ck["telemetry"], sm_median_ge_97pct=ck["sm_median"], reasons_within_0x1_0x2_0x4=ck["telemetry"] and not bad,
         power_limit_is_default=lim == default, alone=ck["alone"], reasons_seen=reasons, sm_median=c.get("sm_median"),
         sm_max_clock=c.get("sm_max_clock"), sm_min=c.get("sm_min"), power_max_w=c.get("power_max"), enforced_w=lim, default_w=default,
         power_capped_rows=sum(1 for r in c.get("rows", []) if r and int(r[2], 16) & 4), rows=c.get("samples"))
v["passed"] = all(v[k] for k in ("telemetry", "sm_median_ge_97pct", "reasons_within_0x1_0x2_0x4", "power_limit_is_default", "alone"))
json.dump(v, open(sys.argv[3], "w"), indent=1)
print(f"GPU check: median SM {v['sm_median']} of {v['sm_max_clock']} MHz (min {v['sm_min']}), reasons {reasons}, power-capped rows "
      f"{v['power_capped_rows']} of {v['rows']}, power {v['power_max_w']} W max, limit {lim:g}/{default:g} W, alone {v['alone']}: "
      f"{'PASSED' if v['passed'] else 'FAILED'}")
sys.exit(0 if v["passed"] else 1)
PY
  done
  echo "PRE-SCREEN PASSED: 5 of 5 GPU checks, CPU steal and quota ($P)"
}

# --------------------------------------------------------------------------------------------------- environments of jobs
lock_pod() { mkdir -p $Q; exec 9> $Q/run.lock; flock -n 9 || die "another round or cache build is running on this pod"; }
gpu_idle() {
  [ "$(smi name)" = "$H100" ] || die "the card is not an $H100"
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || die "the card is not idle (compute apps listed)"
}
bb_env() {   # $1 = triton cache dir, $2 = inductor cache dir; the environment of common.sh (the RTX jobs), on the pod
  local T=$D1_TREE
  case "$(cat $T/DEPLOYED)" in *+dirty*) die "$T was deployed from a dirty checkout" ;; esac
  cd $T/ising_compare
  export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c:$T/cascade_compare/phase_d:$T/cascade_compare/batchopt \
         OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 QND_RELAY=$RELAY_ROOT TRITON_CACHE_DIR=$1 \
         TORCHINDUCTOR_CACHE_DIR=$2 PC_TREE="$(cat $T/DEPLOYED)" PC_GPU_DRIVER="$(smi driver_version)" SRT_BURN_S=${SRT_BURN_S:-0}
}
d13_env() {  # the environment of r4.sh; the caches are the session's own
  local T=$D13_TREE
  case "$(cat $T/DEPLOYED)" in *+dirty*) die "$T was deployed from a dirty checkout" ;; esac
  cd $T
  export PYTHONPATH=$T OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CONVLINES_RELEASE=$T/release \
         CASCADE_ROOT=$D13_ROOT NCCL_IB_DISABLE=1 TRITON_CACHE_DIR=$Q/cache/d13/triton TORCHINDUCTOR_CACHE_DIR=$Q/cache/d13/inductor \
         SRT_BURN_S=${SRT_BURN_S:-0} SV_GPU_DRIVER="$(smi driver_version)"
}
drive_bb() {  # drive_bb ROUND OUT ARGS...
  local r=$1 o=$2; shift 2
  local x=""; [ $r = r2 ] && x="--w4-lock $W4_LOCK --speed $SPEED_JSON"
  $PY $CAS/drive.py $r --tree $D1_TREE --out $o --card h100 $x "$@"
}

# ----------------------------------------------------------------------------------------------------------------- kcache
kcache() {
  need D1_TREE D13_TREE D13_ROOT D13_LOCK RELAY_ROOT W4_LOCK SPEED_JSON SRT_DRY_SEEDS SRT_DRY_BEACON
  local rounds="${*:-r1 r2 r3 r4}" r K=$Q/kcache_$STAMP
  lock_pod; gpu_idle; mkdir -p $K $Q/session
  export SRT_SEEDS=$SRT_DRY_SEEDS SRT_BEACON=$SRT_DRY_BEACON SRT_DRY=1          # a cache build never draws timing shots
  for r in $rounds; do
    echo "== kcache $r $(date -Is)"
    case $r in
      r1|r2|r3) mkdir -p $Q/cache/bb/triton $Q/cache/bb/inductor
        ( bb_env $Q/cache/bb/triton $Q/cache/bb/inductor && drive_bb $r $K/$r --gate off --reps 1 --n-single 2000 \
            --n-batched 20000 ) > $K/$r.log 2>&1 || die "cache build $r (see $K/$r.log)"
        tail -n 2 $K/$r.log ;;
      r4) ( d13_kcache $K ) > $K/r4.log 2>&1 || die "cache build r4 (see $K/r4.log)"
        tail -n 4 $K/r4.log ;;
      *) die "round $r" ;;
    esac
  done
  echo "kcache done: $K"
}
d13_kcache() {  # the d = 13 cache: build with the digest check off, then passes that must find it unchanged
  local K=$1 i S L rc
  d13_env; mkdir -p $Q/cache/d13/triton $Q/cache/d13/inductor
  local DL=$K/d13_dry_lock.json
  $PY $CAS/srtlib.py dry-lock --lock $D13_LOCK --out $DL $DRY_LOCK_ARGS || return 1
  S=$K/session_0.json; $PY $CAS/srt_d13.py h100-session --out $S || return 1
  L=$K/d13_lock_0.json; $PY $CAS/srt_d13.py h100-lock --lock $DL --session $S --out $L || return 1
  SRT_SESSION=$S SRT_KCACHE_BUILD=1 $PY $CAS/drive.py r4 --tree $D13_TREE --out $K/r4_build --lock $L --card h100 --gate off \
    --reps 1 || return 1
  for i in 1 2 3; do
    S=$K/session_$i.json; $PY $CAS/srt_d13.py h100-session --out $S || return 1
    L=$K/d13_lock_$i.json; $PY $CAS/srt_d13.py h100-lock --lock $DL --session $S --out $L || return 1
    SRT_SESSION=$S $PY $CAS/drive.py r4 --tree $D13_TREE --out $K/r4_verify_$i --lock $L --card h100 --gate off --reps 1
    rc=$?
    if [ $rc = 0 ]; then
      [ ! -e $Q/session/d13_session.json ] || mv $Q/session/d13_session.json $Q/session/d13_session.json.before_$STAMP
      cp $S $Q/session/d13_session.json && chmod a-w $Q/session/d13_session.json
      echo "d13 cache complete after $i verify pass(es): $(grep -A3 '"kcache"' $S | tr -d ' \n') -> $Q/session/d13_session.json"
      return 0
    fi
    echo "verify pass $i changed the cache or failed (exit $rc): its session record is taken again"
  done
  return 1
}

# -------------------------------------------------------------------------------------------------------------------- run
job_r4() {   # job_r4 OUT ENV: the d = 13 round on the derived lock; the caches must still be the session's
  local O=$1 E=$2 L=$D13_LOCK S=$Q/session/d13_session.json
  [ -s $S ] || { echo "no d = 13 session: run kcache first"; return 1; }
  d13_env
  if [ "${SRT_DRY:-}" = 1 ]; then
    $PY $CAS/srtlib.py dry-lock --lock $D13_LOCK --out $E/d13_lock_dry.json $DRY_LOCK_ARGS || return 1
    L=$E/d13_lock_dry.json
  fi
  $PY $CAS/srt_d13.py h100-session --out $E/session_now.json || return 1
  cmp -s <(grep -A2 '"kcache"' $E/session_now.json) <(grep -A2 '"kcache"' $S) ||
    { echo "the caches are not the session's: $(grep -A2 '"kcache"' $E/session_now.json | tr -d ' \n') against $(grep -A2 '"kcache"' $S | tr -d ' \n')"; return 1; }
  $PY $CAS/srt_d13.py h100-lock --lock $L --session $S --out $E/d13_lock_h100.json || return 1
  SRT_SESSION=$S $PY $CAS/drive.py r4 --tree $D13_TREE --out $O/drive --lock $E/d13_lock_h100.json --card h100 ${DRIVE_ARGS:-}
}
job() {   # job ROUND: one registered round
  local r=$1 O=$Q/out/${1}_$(date +%Y%m%d_%H%M%S) rc A=${DRIVE_ARGS:-}; local E=$O/env
  mkdir -p $E; echo "== $r -> $O"
  gpu_idle
  { date -Is; echo "round $r card h100 host $(hostname)"; nvidia-smi -L
    nvidia-smi --query-gpu=name,pci.bus_id,driver_version,vbios_version,mig.mode.current,memory.total,clocks.max.sm,power.limit,enforced.power.limit,power.default_limit,power.max_limit,persistence_mode,compute_mode,pstate --format=csv
    lscpu | grep -E '^(Model name|CPU\(s\)|On-line|Thread|Core|Socket|CPU max MHz)'
    echo "nproc $(env -u OMP_NUM_THREADS nproc); cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo -)"
    free -g | head -2; uname -srm; echo "python $($PY -V 2>&1)"
    echo "inputs: $INPUTS_LIST"; env | grep -E '^(SRT_[A-Z_]*|DRIVE_ARGS|DRY_LOCK_ARGS|D1_TREE|D13_[A-Z]*|W4_LOCK|SPEED_JSON|RELAY_ROOT|Q)=' | sort
  } > $E/run_env.txt 2>&1
  cp $Q/venv_freeze.txt $E/pip_freeze.txt
  for t in D1_TREE D13_TREE; do (cd ${!t} && find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' | LC_ALL=C sort | xargs -d '\n' sha256sum) > $E/code_sha256_$t.txt; done
  (cd $SRT_ROOT && find speed_rt surface_v2 -type f ! -path '*/__pycache__/*' ! -name '*.pyc' | LC_ALL=C sort | xargs -d '\n' sha256sum) > $E/srt_sha256.txt
  cp $INPUTS_LIST $E/inputs.list; cp $INPUTS_SHA $E/inputs.sha256
  (cd / && sha256sum --quiet -c $INPUTS_SHA) > $E/inputs_check.txt 2>&1 || { cat $E/inputs_check.txt | head -3; finish $O 1 "the inputs differ from the deploy's sha256 list"; return 1; }
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,enforced.power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
    --format=csv -l 2 > $E/clock_samples.csv 2>&1 &
  SAMPLER=$!
  case $r in
    r1|r2|r3)
      local JC=$Q/tmp/${r}_$STAMP; mkdir -p $JC/triton $JC/inductor
      cp -r $Q/cache/bb/triton/. $JC/triton 2> /dev/null; cp -r $Q/cache/bb/inductor/. $JC/inductor 2> /dev/null
      ( bb_env $JC/triton $JC/inductor && drive_bb $r $O/drive $A ) 2>&1 | tee $O/drive.log; rc=${PIPESTATUS[0]}
      rm -rf $JC ;;
    r4) ( job_r4 $O $E ) 2>&1 | tee $O/drive.log; rc=${PIPESTATUS[0]} ;;
    *) die "round $r" ;;
  esac
  finish $O $rc "drive.py exit $rc"
}
finish() {   # finish OUTDIR RC WHY: sums, END or FAILED
  local O=$1 rc=$2
  [ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null; SAMPLER=""; sleep 1
  nvidia-smi -q > $O/env/nvidia_smi_q_end.txt 2>&1
  [ $rc = 0 ] || echo "$3" > $O/FAILED
  (cd $O && find . -type f ! -name SHA256SUMS ! -name END ! -name FAILED -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  if [ $rc = 0 ]; then echo done > $O/END; echo "END $(date -Is) $O"; else echo "FAILED: $3 ($O)"; fi
}
run() {
  need D1_TREE D13_TREE D13_ROOT D13_LOCK RELAY_ROOT W4_LOCK SPEED_JSON INPUTS_LIST INPUTS_SHA SRT_SEEDS SRT_BEACON
  [ $# -gt 0 ] || die "run needs rounds (r1 r2 r3 r4)"
  lock_pod; mkdir -p $Q/out $Q/tmp
  local r; for r in "$@"; do job $r || return 1; done
}

case "$MODE" in
  setup) setup ;;
  prescreen) prescreen ;;
  kcache) kcache "$@" ;;
  run) run "$@" ;;
  *) echo "usage: h100_pod.sh setup | prescreen | kcache [ROUND...] | run ROUND..."; exit 2 ;;
esac
