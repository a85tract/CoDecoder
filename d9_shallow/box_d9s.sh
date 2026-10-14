#!/bin/bash
# D9-SHALLOW's CPU session on the Linux workstation (PLAN.md section 5), run once, detached, after the authors' box settings
# (STAGE2_common C1: governor performance, EPP performance, boost on, CPUs 16-23 offline, other accounts on CCD1).
# The speed re-test's harness and rules, unchanged: speed_rt/cpu_ws/ as deployed in the tmpfs for stage 2 (its sha256 list
# must be 2fd2afa5df2be2cd and equal this tree's), cfg_box.json, the window rule (every reference value within 1.10 x the
# session reference, each segment's largest over smallest at most 1.06, no block read by our processes, every process
# exit 0), 8 slots on CCD0, fillers, at most 3 attempts a slot, the session reference gate (<= 1.10 x 17.306 us).
#   bash d9_shallow/box_d9s.sh TREE_TGZ COMMIT IN SESSION
# TREE_TGZ: `git archive` of COMMIT (commit B) holding surface_v2 (without weights), speed_rt and d9_shallow (without
# weights); IN: the hand-off folder holding the unit d9s (low/val_d9_b1w64_sh.npz, low/HANDOFF_MASKS.json, the 20 timing
# chunks under shots/, d9s_timing.json, SHA256SUMS, DONE written last). Steps: deploy the tree (cpu_ws/prep.sh code),
# pin the libraries, import the unit (SHA256SUMS checked), the timing view, the reference shots' check, topo, warm pass,
# session reference and its gate, the plan (d9_shallow/plan_d9s.py), harness run, export (export.py --only surface).
set -uo pipefail
TGZ=$1; COMMIT=$2; IN=$3; S=$4; EXPECT=2fd2afa5df2be2cd; REF0=17.306
export SRT_TMPFS=/dev/shm/srt_box SRT_CFG=cfg_box.json QRT=$HOME/srt_box_io/qrt_$4
R=$SRT_TMPFS; PY=$R/venv/bin/python; RUN=$R/runs/$S; OUT=$HOME/srt_box_io/export_$S
log() { echo "[$(date -Is)] $*"; }
[ -e "$RUN" ] && { log "run directory $RUN exists; refusing"; exit 2; }
mkdir -p "$QRT/logs"
log "=== box_d9s start: session $S, commit $COMMIT, host $(hostname), CPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
dep=$(cd "$R/cpu_ws" && sha256sum $(find . -maxdepth 1 -type f ! -name clk | LC_ALL=C sort) | sha256sum | cut -c1-16)
[ "$dep" = "$EXPECT" ] || { log "deployed cpu_ws sha256 list $dep != $EXPECT; stopping"; exit 4; }
mkdir -p /tmp/d9s_tree_$S && tar -C /tmp/d9s_tree_$S -xzf "$TGZ"
got=$(cd /tmp/d9s_tree_$S/speed_rt/cpu_ws && sha256sum $(find . -maxdepth 1 -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
[ "$got" = "$EXPECT" ] || { log "the tree's cpu_ws sha256 list $got != $EXPECT; stopping"; exit 4; }
bash "$R/cpu_ws/prep.sh" code "$TGZ" "$COMMIT" | tail -2
cmp "$R/code/speed_rt/testsets/srt-ref.json" "$R/data/ref/srt-ref.json" || { log "srt-ref.json differs"; exit 4; }
(cd "$R/handoff/surface_shots" && sha256sum --quiet -c SHA256SUMS) || { log "the reference shots fail SHA256SUMS"; exit 4; }
$PY -c "import sys, numpy, scipy, stim, pymatching; v = (sys.version.split()[0], numpy.__version__, scipy.__version__, stim.__version__, pymatching.__version__); print('env', v); assert v == ('3.12.3', '2.2.6', '1.18.1', '1.16.0', '2.4.0')" || exit 4
mkdir -p "$RUN"
if ! { [ -e "$R/pinlibs.pid" ] && kill -0 "$(cat "$R/pinlibs.pid")" 2>/dev/null; }; then
  setsid -f taskset -c 8-15,24-31 bash -c "echo \$\$ > $R/pinlibs.pid; exec $PY $R/cpu_ws/pinlibs.py $R > $RUN/pinlibs.log 2>&1 < /dev/null"
  sleep 8
fi
log "pinlibs pid $(cat "$R/pinlibs.pid")"
export LD_LIBRARY_PATH=$R/lib LC_ALL=C.UTF-8 TZ=UTC0
cd "$R/cpu_ws"
[ -e "$IN/d9s/DONE" ] || { log "no hand-off unit $IN/d9s/DONE"; exit 4; }
taskset -c 8-15,24-31 $PY harness.py import --cfg cfg_box.json --units d9s --log "$RUN/handoff" --handoff-from "$IN" --wait 0 | tail -2 || exit 4
V=$R/data/surf_d9s/view; rm -rf "$R/data/surf_d9s" && mkdir -p "$V"
for j in $(seq -w 0 19); do ln -s "$R/handoff/d9s/shots/d9_p0.006_X_d9stiming_0$j.npz" "$V/d9_p0.006_X_val_0$j.npz"; done
[ "$(ls "$V" | wc -l)" = 20 ] && [ -e "$R/handoff/d9s/low/val_d9_b1w64_sh.npz" ] || { log "view or mask missing"; exit 4; }
$PY harness.py topo --cfg cfg_box.json --out "$RUN/topo.json" | cut -c1-300
$PY harness.py sessionref --cfg cfg_box.json --out "$RUN/warm" --repeats 1 | tail -1
$PY harness.py sessionref --cfg cfg_box.json --out "$RUN/sref" | tail -1
ok=$($PY -c "import json; s=json.load(open('$RUN/sref/sessionref.json')); print(int(s['disk_delta']['passed'] and s['reference_us'] <= 1.10 * $REF0))")
[ "$ok" = 1 ] || { log "session reference failed its disk rule or exceeds 1.10 x $REF0 us; stopping (exit 3)"; exit 3; }
$PY "$R/code/d9_shallow/plan_d9s.py" --cpu-ws "$R/cpu_ws" --out "$RUN/plan.json" | tail -10
$PY -u harness.py run --cfg cfg_box.json --plan "$RUN/plan.json" --sessionref "$RUN/sref/sessionref.json" --out "$RUN/win" --handoff-from "$IN"
log "=== harness run ended"
taskset -c 8-15,24-31 $PY export.py "$RUN" "$OUT" --only surface | tail -3
log "=== box_d9s done: export in $OUT"
