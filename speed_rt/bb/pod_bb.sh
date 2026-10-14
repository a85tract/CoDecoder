#!/bin/bash
# SPEED-RT BB stage: the GPU session on one rented H100 80GB HBM3 pod (root, internet, uv), from a deployed copy of the public tree whose
# DEPLOYED file holds the frozen commit. The pod times GPU work only (CURRENT RULES of 13 October 15:45): every CPU part of the BB
# comparisons (sector Relay-BP, full-DEM Relay-BP, ldpc BP-OSD, our pairs' Relay-BP terms) is timed on winserver by the CPU worker
# (speed_rt/bb/cpu_items.json, speed_rt/bb/cpu_item.sh), from the routing reads this session makes. One role, one session ("gpu"):
#   BB72    r11 and r12 (GPU term and handoff, 5 fresh processes each), CUDA-Q BP-OSD at p = 0.3 % and 0.2 % (3 each)
#   gross   g2 (GPU term and handoff, 3), CUDA-Q BP-OSD on the gross code (3), graph32 (the fp32 CUDA graph, 5, descriptive)
#   reads   the untimed GPU pass on the timing shots (the routing reads): r12 on BB72 p = 0.3 % chunks 0-19 and p = 0.2 % chunks 0-29,
#           g2 on the gross code chunks 0-14: the margins the CPU worker routes its CPU terms with
# Modes (each of the first four is safe to repeat; the timed block is written once per output directory):
#   pod_bb.sh setup          the environments from their locks (s3venv: BB72's GPU items and CUDA-Q; g2venv: g2 and graph32; cqvenv: CUDA-Q on the
#                            gross code; hcvenv: the pre-screen's GPU checks and the attempt log); each venv's freeze must equal its lock
#   pod_bb.sh ensure         setup only when an environment differs from its lock (SETUP_NOT_NEEDED otherwise)
#   pod_bb.sh prescreen      stage 1's rule (speed_rt/bb/prescreen_bb.py, SPEED_RT.md 1.5): five GPU checks on the idle card, CPU steal at most
#                            1 % over 5 s, container quota at least 20 CPUs -> $SRT_OUT/prescreen; exit 4: the pod failed, replace it
#   pod_bb.sh reads          the untimed GPU pass, blind, on the timing shots -> $SRT_OUT/preds (65 npz files, 11 MB), preds.sha256: the hand-off
#   pod_bb.sh timed          the timed block: rounds r = 1 .. 5, each round every item with a slot in it, in a fixed order. A slot (one process
#                            of one item in one round) has at most three attempts; each attempt is preceded by the idle-card check (bb.timing
#                            idle: two samples 100 ms apart show 0 % utilisation and under 1,024 MiB used; 20 tries 3 s apart) and runs inside
#                            speed_rt.watch (pinned to the timed core; the clock and throttle record, no other compute process; the modules' own
#                            97 % clock floor inside their timed spans and exactly one compute application). The host numbers of the window
#                            are recorded and decide nothing. An attempt is accepted, rejected (a check refused it) or failed (it did not finish
#                            its work, attempts.py). A slot with three attempts and no accepted one is not established (the session goes on); a
#                            third attempt that fails stops the session (exit 6). Exit codes: 0 every slot established, 1 some slot not
#                            established, 6 stopped, 2 refused to start.
# Records in $SRT_OUT: rows.csv, attempts.json, watch/, logs/, session.json (written at the end), pod_env.txt, prescreen/, preds/, preds.sha256.
# Environment, set by the caller (all paths on the pod):
#   SRT_W        work directory (default /workspace/srt)          SRT_REL    the deployed copy of the public tree (default $SRT_W/rel)
#   SRT_DRAW     the draw copied from Alpine (default $SRT_W/draw): shots/, manifest_bb72.csv, manifest_gross.csv
#   QND_RELAY    the Relay-BP clone at d185194 (default $SRT_W/relay): tests/testdata holds the circuits
#   SRT_OUT      this session's outputs (default $SRT_W/out_gpu)  SRT_SESSION  the session label written into every row (default srt-bb-gpu)
# Dry runs only (never set in a registered session): SRT_DRY_NEED=N counted processes per item instead of 5 / 3; SRT_DRY_HOLD=1 a second CUDA
#   process sits on the card during the first GPU attempt; SRT_DRY_FAILFIRST=ITEM the first attempt of ITEM fails (exit 7).
set -uo pipefail
MODE=${1:-}; ROLE=${2:-gpu}
W=${SRT_W:-/workspace/srt}; REL=${SRT_REL:-$W/rel}; D=${SRT_DRAW:-$W/draw}; OUT=${SRT_OUT:-$W/out_gpu}
PRE=$OUT/prescreen
export QND_RELAY=${QND_RELAY:-$W/relay} BB_PIDNS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=$W/uvcache \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BB_SESSION=${SRT_SESSION:-srt-bb-gpu} BB_SHOTS=$D/shots BB_PREDS=$OUT/preds
unset SRT_BEACON SRT_DRY SRT_DRY_SHOTS SRT_ORIGINAL_SEEDS      # timing steps never draw shots (the shim refuses a test seed)
PY=$W/g2venv/bin/python; PYQ=$W/cqvenv/bin/python; PY3=$W/s3venv/bin/python; HC=$W/hcvenv/bin/python
T72=$REL/bb/bb72/ising_compare; TG=$REL/bb/gross_stage4/ising_compare; SHIM=$REL/speed_rt/bb/srt_bb.py
ATT=$REL/speed_rt/bb/attempts.py; WATCH=$REL/speed_rt/watch.py
K=$REL/weights/bb; CKG=$K/g2_D12k113c10_final_ema.pt; GPU="NVIDIA H100 80GB HBM3"
L4=$TG/data/bb/stage4_gross/pod; L3=$T72/data/bb/stage3/pod; LHC=$REL/speed_rt/bb/hcvenv.lock
R=$OUT/rows.csv; A=$OUT/attempts.json
pkgs() { grep -v '^#' "$1" | grep -v '^-e \|@ file' | sort; }
lock() { [ "$(uv pip freeze -p "$1" | sort)" = "$(pkgs "$2")" ] || { echo "LOCK_REFUSED $1 against $2"; return 1; }; }

[ "$ROLE" = gpu ] || { echo "usage: pod_bb.sh setup|ensure|prescreen|reads|timed [gpu]"; exit 2; }
ITEMS="r11 r12 cq72p3 cq72p2 g2 cqg graph32"; MAXR=5          # the round order; an item has a slot in rounds 1 .. its need (r11, r12, graph32 5; the rest 3)
tree_ok() { [ -s "$REL/bb/bb72/DEPLOYED" ] && [ -s "$REL/DEPLOYED" ] && [ -d "$T72" ] && [ -d "$TG" ] && [ -f "$SHIM" ] && [ -f "$WATCH" ] \
              && [ -f "$ATT" ] && [ -f "$REL/speed_rt/bb/prescreen_bb.py" ] && [ -d "$REL/surface_v2/surf_v2" ] && [ -d "$QND_RELAY/tests/testdata" ] \
              || { echo "no deployed tree, DEPLOYED file or relay clone under $REL"; return 1; }; }

# ---- setup ----------------------------------------------------------------------------------------------------------------------
setup() {
  tree_ok || exit 2
  set -x
  command -v uv > /dev/null || pip install -q uv
  command -v gcc > /dev/null && command -v git > /dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential git; }
  uv python install 3.12.14 || exit 2
  local bad=0
  rm -rf "$W/g2venv" && uv venv -q --python 3.12.14 "$W/g2venv" || exit 2
  uv pip install -q -p "$PY" --no-deps --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple -r <(pkgs "$L4/env.lock") || exit 3
  rm -rf "$W/cqvenv" && uv venv -q --python 3.12.14 "$W/cqvenv" || exit 2
  uv pip install -q -p "$PYQ" --no-deps -r <(pkgs "$L4/cqvenv.lock") || { echo CUDAQ_FAIL; bad=1; }
  rm -rf "$W/s3venv" && uv venv -q --python 3.12.14 "$W/s3venv" || exit 2
  uv pip install -q -p "$PY3" --no-deps --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple -r <(pkgs "$L3/pod_pip_freeze.txt") || exit 3
  rm -rf "$W/hcvenv" && uv venv -q --python 3.12.14 "$W/hcvenv" || exit 2          # torch (the g2 lock's), safetensors, PyMatching: the pre-screen's GPU checks and the attempt log
  uv pip install -q -p "$HC" --no-deps --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple -r <(pkgs "$LHC") || exit 3
  set +x
  mkdir -p "$OUT"
  lock "$PY" "$L4/env.lock" || bad=1
  lock "$PYQ" "$L4/cqvenv.lock" || bad=1
  lock "$PY3" "$L3/pod_pip_freeze.txt" || bad=1
  lock "$HC" "$LHC" || bad=1
  for v in g2venv cqvenv s3venv hcvenv; do [ -x "$W/$v/bin/python" ] && uv pip freeze -p "$W/$v/bin/python" > "$OUT/freeze_$v.txt"; done
  $PY -c "import torch, triton; x = torch.randn(256, 256, device='cuda'); assert torch.isfinite(x @ x).all(); print('ENV g2venv torch', torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), 'triton', triton.__version__, torch.cuda.get_device_name(0))"
  $PY3 -c "import torch, triton; x = torch.randn(256, 256, device='cuda'); assert torch.isfinite(x @ x).all(); print('ENV s3venv torch', torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), 'triton', triton.__version__, torch.cuda.get_device_name(0))"
  $PYQ -c "import importlib.metadata as m, cudaq_qec; print('ENV cqvenv cudaq-qec', m.version('cudaq-qec-cu12'))"
  $PY3 -c "import cudaq_qec" 2> /dev/null && echo "s3venv has cudaq_qec" || { echo "s3venv lacks cudaq_qec"; bad=1; }
  $HC -c "import torch, safetensors; print('ENV hcvenv torch', torch.__version__, torch.cuda.get_device_name(0))"
  echo "SETUP_END $(date -u +%T) bad $bad"
  exit $bad
}

ensure() {   # setup only when an environment differs from its lock (a pod that already holds them saves the ten minutes)
  tree_ok || exit 2
  local ok=1
  lock "$PY" "$L4/env.lock" > /dev/null 2>&1 || ok=0
  lock "$PYQ" "$L4/cqvenv.lock" > /dev/null 2>&1 || ok=0
  lock "$PY3" "$L3/pod_pip_freeze.txt" > /dev/null 2>&1 || ok=0
  lock "$HC" "$LHC" > /dev/null 2>&1 || ok=0
  if [ "$ok" = 1 ]; then
    mkdir -p "$OUT"; for v in g2venv cqvenv s3venv hcvenv; do [ -x "$W/$v/bin/python" ] && uv pip freeze -p "$W/$v/bin/python" > "$OUT/freeze_$v.txt"; done
    echo "SETUP_NOT_NEEDED $(date -u +%T)"; exit 0
  fi
  echo "ensure: the environments are not complete: running setup"
  setup
}

# ---- gate -------------------------------------------------------------------------------------------------------------------------
gate() {   # the pod and the environments are the registered ones, or nothing is read or timed
  tree_ok || exit 2
  local g; g=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  [ "$g" = "$GPU" ] && [ "$(nvidia-smi -L | wc -l)" = 1 ] || { echo "GPU_REFUSED $g"; exit 1; }
  lock "$PY" "$L4/env.lock" || exit 1
  lock "$PYQ" "$L4/cqvenv.lock" || exit 1
  lock "$PY3" "$L3/pod_pip_freeze.txt" || exit 1
  lock "$HC" "$LHC" || exit 1
  REL=$REL "$HC" - <<'PY' || { echo "WEIGHTS_REFUSED"; exit 1; }
import hashlib, json, os, sys
R = os.environ["REL"]
c3 = json.load(open(f"{R}/bb/bb72/ising_compare/data/bb/stage3/cells.json"))["cells"]
c4 = json.load(open(f"{R}/bb/gross_stage4/ising_compare/data/bb/stage4_gross/cell.json"))
want = {"r11": c3["BB72-3"]["net_sha256"], "r12": c3["BB72-3F"]["net_sha256"], "g2_D12k113c10": c4["net_sha256"]}
bad = 0
for n, w in want.items():
    h = hashlib.sha256(open(f"{R}/weights/bb/{n}_final_ema.pt", "rb").read()).hexdigest()
    bad += h != w
    print("weights", n, h[:12], "ok" if h == w else "DIFFERENT")
sys.exit(1 if bad else 0)
PY
  mkdir -p "$OUT"
  { echo "== $(date -u +%FT%TZ) $1 commit $(cat "$REL/DEPLOYED") tree $(cat "$REL/bb/bb72/DEPLOYED")"
    nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,mig.mode.current --format=csv,noheader
    echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) visible $(env -u OMP_NUM_THREADS nproc) cpuset $(awk '/Cpus_allowed_list/{print $2}' /proc/self/status) quota $(cat /sys/fs/cgroup/cpu.max 2> /dev/null) kernel $(uname -r)"
    echo "freeze sha256: $(for v in g2venv cqvenv s3venv hcvenv; do printf '%s ' "$(sha256sum "$OUT/freeze_$v.txt" 2> /dev/null | cut -c1-16)"; done)"; } >> "$OUT/pod_env.txt"
}

prescreen() {
  gate "prescreen"
  [ ! -e "$PRE" ] || { echo "$PRE exists: the pre-screen is written once per output directory"; exit 2; }
  PYTHONPATH=$REL/surface_v2:$REL "$HC" "$REL/speed_rt/bb/prescreen_bb.py" --out "$PRE" 2>&1 | tail -9
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "PRESCREEN_FAILED: replace the pod"; exit 4; }
  echo "PRESCREEN_END $(date -u +%T)"
}

# ---- reads: the untimed GPU pass (the hand-off to the CPU worker) ---------------------------------------------------------------------
reads() {
  gate reads
  [ ! -e "$OUT/preds" ] || { echo "$OUT/preds exists: the reads are written once"; exit 2; }
  mkdir -p "$OUT/preds" "$OUT/logs"
  local pc p c
  for pc in "0.003 0-19" "0.002 0-29"; do set -- $pc; p=$1; c=$2
    ( cd "$T72" && export PYTHONPATH=.:../qldpc_neural SRT_MANIFEST=$D/manifest_bb72.csv
      $PY3 "$SHIM" bb.route infer --split test --code bb72 --p "$p" --tag r12 --chunks "$c" --policy "data/bb/policies/bb72_p${p}_rep2.json" \
        --ckpt "$K/r12_final_ema.pt" --gpu "$GPU" ) > "$OUT/logs/read_bb72_p$p.log" 2>&1
    echo "bb72 p$p read rc $?: $(tail -1 "$OUT/logs/read_bb72_p$p.log" | cut -c1-100)"
  done
  ( cd "$TG" && export PYTHONPATH=.:../qldpc_neural SRT_MANIFEST=$D/manifest_gross.csv
    $PY "$SHIM" bb.stage4_gross infer --split test4 --chunks 0-14 --ckpt "$CKG" ) > "$OUT/logs/read_gross.log" 2>&1
  echo "gross read rc $?: $(tail -1 "$OUT/logs/read_gross.log" | cut -c1-100)"
  ( cd "$OUT/preds" && sha256sum ./*.npz > ../preds.sha256 )
  echo "READS_END $(date -u +%T) files $(ls "$OUT/preds" | wc -l) (65 expected)"
  [ "$(ls "$OUT/preds" | wc -l)" = 65 ]
}

# ---- the timed block --------------------------------------------------------------------------------------------------------------
need_of() { local n=$1; if [ -n "${SRT_DRY_NEED:-}" ] && [ "$SRT_DRY_NEED" -lt "$n" ]; then n=$SRT_DRY_NEED; fi; echo "$n"; }

item_spec() {   # item_spec ITEM ROUND: TREE, PYI, NEED, MAN, RULE and the command ARGS (the python arguments)
  local i=$1 r=$2 p
  RULE=throttle; TREE=$T72; PYI=$PY3; MAN=$D/manifest_bb72.csv
  case "$i" in
    r11|r12) NEED=5; ARGS=(-m bb.latency --ckpt "$K/${i}_final_ema.pt" --engine fused --p 0.003 --proc "$r" --out "$R") ;;
    cq72p3|cq72p2) p=0.003; [ "$i" = cq72p2 ] && p=0.002; NEED=3
      ARGS=("$SHIM" bb.gpu_bposd time --code bb72 --p "$p" --split test --chunks 0 --batches 1 --reps1 300 --out "$OUT/gpu_bposd_detail.csv" --timing-out "$R" --proc "$r") ;;
    g2) TREE=$TG; PYI=$PY; MAN=$D/manifest_gross.csv; NEED=3; ARGS=(-m bb.fast lat --ckpt "$CKG" --kind fusedt --proc "$r" --out "$R") ;;
    graph32) TREE=$TG; PYI=$PY; MAN=$D/manifest_gross.csv; NEED=5; ARGS=(-m bb.latency --ckpt "$CKG" --engine graph32 --p 0.004 --proc "$r" --out "$R") ;;
    cqg) TREE=$TG; PYI=$PYQ; MAN=$D/manifest_gross.csv; NEED=3
      ARGS=("$SHIM" bb.gpu_bposd time --code bb144 --p 0.004 --split test4 --chunks 0 --batches 1 --reps1 300 --out "$OUT/gpu_bposd_detail.csv" --timing-out "$R" --proc "$r") ;;
    *) echo "unknown item $i"; exit 2 ;;
  esac
  NEED=$(need_of "$NEED")
}

idle() {   # bb.timing idle: 20 attempts, 3 s apart; the module appends its check to the rows
  local i
  for i in $(seq 1 20); do
    ( cd "$TREE" && PYTHONPATH=.:../qldpc_neural "$PYI" -m bb.timing idle --out "$R" ) > /dev/null 2>&1 && return 0
    sleep 3
  done
  return 1
}

rel() { local t=$1; t=${t//"$OUT"\//}; t=${t//"$REL"\//}; t=${t//"$W"\//}; echo "${t//"$OUT"/.}"; }     # a record names files relative to the output directory or the tree, never by an absolute path
record() { "$HC" "$ATT" record --file "$A" --item "$ITEM" --round "$ROUND" --attempt "$N" --kind gpu --log "logs/$(basename "$LOGF")" --cmd-text "$(rel "$ITEM: ${ARGS[*]}")" "$@"; }

run_attempt() {   # run_attempt ITEM ROUND N -> OUTCOME accepted | rejected | failed
  ITEM=$1; ROUND=$2; N=$3
  local tag=${ITEM}_r${ROUND}_a${N} hold="" skip rec cmd rc
  item_spec "$ITEM" "$ROUND"
  rec=$OUT/watch/$tag.json; LOGF=$OUT/logs/$tag.log
  if ! idle; then
    echo "IDLE_FAIL $tag"
    record --rows "$R" --not-started "the card was not idle (bb.timing idle failed 20 times)"; rc=$?
    OUTCOME=rejected; return 0
  fi
  if [ -n "${SRT_DRY_HOLD:-}" ] && [ ! -e "$OUT/.dry_hold" ]; then   # dry only: a second CUDA process on the card
    : > "$OUT/.dry_hold"
    "$HC" -c "import torch, time; x = torch.zeros(1, device='cuda'); print('HOLDING', flush=True); time.sleep(600)" > "$OUT/logs/holder.log" 2>&1 &
    hold=$!
    for _ in $(seq 1 60); do grep -q HOLDING "$OUT/logs/holder.log" 2> /dev/null && break; sleep 1; done
  fi
  skip=$("$HC" "$ATT" rows-count --rows "$R")
  [ ! -e "$rec" ] || mv "$rec" "$rec.orphan.$(date +%s)"
  cmd=("$PYI" "${ARGS[@]}")
  if [ "${SRT_DRY_FAILFIRST:-}" = "$ITEM" ] && [ ! -e "$OUT/.dry_failfirst" ]; then : > "$OUT/.dry_failfirst"; cmd=(bash -c "echo dry failure; exit 7"); fi
  ( cd "$TREE" && export PYTHONPATH=.:../qldpc_neural SRT_MANIFEST=$MAN
    exec "$HC" "$WATCH" run --core "$CORE" --rule "$RULE" --out "$rec" -- "${cmd[@]}" ) > "$LOGF" 2>&1
  record --rows "$R" --skip "$skip" --watch "$rec" >> "$OUT/logs/attempts.log"; rc=$?
  tail -1 "$OUT/logs/attempts.log"
  [ -z "$hold" ] || { kill "$hold" 2> /dev/null; wait "$hold" 2> /dev/null; }
  case $rc in 0) OUTCOME=accepted ;; 10) OUTCOME=rejected ;; *) OUTCOME=failed ;; esac
}

slot() {   # slot ITEM ROUND: up to three attempts; stops the session when the third one fails
  local n
  for n in 1 2 3; do
    run_attempt "$1" "$2" "$n"
    [ "$OUTCOME" != accepted ] || return 0
  done
  if [ "$OUTCOME" = failed ]; then
    echo "SLOT_STOPPED $1 round $2: the third attempt failed"
    finish_session 6 false
  fi
  echo "SLOT_NOT_ESTABLISHED $1 round $2"
}

finish_session() {   # finish_session EXITCODE REACHED_END: session.json, then exit
  local code=$1 reached=$2
  OUT=$OUT CODE=$code REACHED=$reached BB_SESSION=$BB_SESSION START=$T_START CORE=${CORE:-} "$HC" - <<'PY'
import json, os, sys, time
O = os.environ["OUT"]
att = json.load(open(f"{O}/attempts.json"))["attempts"] if os.path.exists(f"{O}/attempts.json") else []
items = {}
for e in att:
    items.setdefault(e["item"], {}).setdefault(e["round"], []).append(e["outcome"])
slots = {}
for item, rs in sorted(items.items()):
    slots[item] = {str(r): ("established" if "accepted" in o else ("stopped" if len(o) >= 3 and o[2] == "failed" else ("exhausted" if len(o) >= 3 else "open")))
                   for r, o in sorted(rs.items())}
bad = [f"{i} round {r}: {s}" for i, rs in slots.items() for r, s in rs.items() if s != "established"]
pre = json.load(open(f"{O}/prescreen/prescreen.json")) if os.path.exists(f"{O}/prescreen/prescreen.json") else {}
doc = dict(role="gpu", session=os.environ.get("BB_SESSION"), start=os.environ["START"], core=os.environ["CORE"], prescreen_passed=bool(pre.get("passed")),
           reached_end=os.environ["REACHED"] == "true", not_established=bad, slots=slots)
doc["qualified"] = bool(doc["reached_end"] and doc["prescreen_passed"])
doc["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
code = int(os.environ["CODE"])
if code == 0 and bad:
    code = 1                                           # reached its end, but a slot is not established
doc["exit_code"] = code
json.dump(doc, open(f"{O}/session.json", "x"), indent=1, sort_keys=True)
print("SESSION", json.dumps({k: doc[k] for k in ("role", "reached_end", "prescreen_passed", "qualified", "not_established", "exit_code")}))
sys.exit(code)
PY
  exit $?
}

timed() {
  gate "timed"
  [ -f "$PRE/prescreen.json" ] && grep -q '"passed": true' "$PRE/prescreen.json" || { echo "no passed pre-screen in $PRE"; exit 2; }
  [ -d "$OUT/preds" ] && [ -s "$OUT/preds.sha256" ] || { echo "no reads in $OUT/preds: run reads first"; exit 2; }
  [ ! -e "$A" ] && [ ! -e "$OUT/session.json" ] || { echo "$OUT already holds a timed session: a session is written once; a rerun uses a new output directory"; exit 2; }
  CORE=${SRT_CORE:-$("$HC" -c "import json; print(json.load(open('$PRE/prescreen.json'))['core'])")}
  taskset -c "$CORE" true 2> /dev/null || { echo "core $CORE is not usable in this container"; exit 2; }
  OFF=$(PYTHONPATH=$REL "$HC" -c "import sys; from speed_rt import watch; print(watch.off_cores(int(sys.argv[1])))" "$CORE")
  [ -z "$OFF" ] || taskset -pc "$OFF" $$ > /dev/null          # this script and everything it starts stay off the timed core and its SMT siblings
  mkdir -p "$OUT/watch" "$OUT/logs"
  T_START=$(date -Is)
  nvidia-smi -q > "$OUT/nvidia_smi_q_timed.txt" 2> /dev/null
  echo "TIMED_START core $CORE $T_START"
  local r item
  for r in $(seq 1 "$MAXR"); do
    echo "== round $r $(date -u +%T)"
    for item in $ITEMS; do
      item_spec "$item" "$r"
      [ "$r" -le "$NEED" ] || continue
      slot "$item" "$r"
    done
  done
  "$HC" "$ATT" summary --file "$A"
  echo "TIMED_END $(date -u +%T)"
  finish_session 0 true
}

case "$MODE" in
  setup) setup ;;
  ensure) ensure ;;
  prescreen) prescreen ;;
  reads) reads ;;
  timed) timed ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
