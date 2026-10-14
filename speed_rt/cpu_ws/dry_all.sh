#!/bin/bash
# SPEED-RT CPU half: the whole dry run on the Windows CPU host, detached from the login session (dry shots only, never a
# timing set).
# Run inside the WSL2 distro:  setsid -f bash dry_all.sh CODE_TGZ RELAY_TGZ DRY_BEACON RUN > LOG 2>&1 < /dev/null
# Steps: environment, code, Relay testdata, validation chunks; then the three dry draws in parallel (surface + the
# reference points with synthetic masks; Decoder Switching with 4 chunks per set and its masks; BB with 10,000 shots
# per chunk and synthetic predictions); the dry hand-off folder (masks, synthetic Ising residuals, the BB draw and
# predictions) in the Windows profile's speedrt_in_dry, which the harness imports at a window boundary as it would the
# real one; the topology record, a warm pass, the session reference, and one repetition of every part whose
# preparation succeeded, in timed windows (plans.py --dry; MAXATT attempts per process, default 1).
# Summary: RUN/summary.txt; the records are also saved to the Windows profile's speedrt_out\<RUN's name>.
set -uo pipefail
CODE=$1; RELAY=$2; BEACON=$3; RUN=$4
H=$(dirname "$(readlink -f "$0")")
P="bash $H/prep.sh"
mkdir -p "$RUN"
ok() { echo "$1" >> "$RUN/prep_ok"; }
$P env && $P code "$CODE" && $P relay "$RELAY" && $P val || { echo "base preparation failed"; exit 1; }
PY=/dev/shm/srt/venv/bin/python
"$PY" -c "import sinter, stimbposd" 2>/dev/null || bash /dev/shm/srt/cpu_ws/setup_env.sh sync
( $P surface "$BEACON" --dry && ok surface ) &
( $P gs "$BEACON" --dry 4 && ok gs ) &
( $P bb "$BEACON" --dry 10000 && ok bb ) &
wait
HD=$("$PY" /dev/shm/srt/cpu_ws/harness.py winpath '%USERPROFILE%\speedrt_in_dry')    # through a live interop socket
grep -qx surface "$RUN/prep_ok" && grep -qx bb "$RUN/prep_ok" && $P handoff_dry "$HD" && ok handoff
cd /dev/shm/srt/cpu_ws
parts=""
grep -qx surface "$RUN/prep_ok" && parts="$parts,gref"
grep -qx gs "$RUN/prep_ok" && parts="$parts,gs_fe,gs_term"
grep -qx handoff "$RUN/prep_ok" && parts="$parts,surf_base,surf_term,surf_share,bb_warm,bb"
parts=${parts#,}
echo "parts prepared: $parts"
export LD_LIBRARY_PATH=/dev/shm/srt/lib LC_ALL=C.UTF-8 TZ=UTC0
"$PY" harness.py topo --out "$RUN/topo.json"
"$PY" harness.py sessionref --cfg cfg_ws.json --out "$RUN/warm" --repeats 1 > /dev/null    # the warm pass (untimed)
"$PY" harness.py sessionref --cfg cfg_ws.json --out "$RUN/sref" --repeats 3
"$PY" plans.py --parts "$parts" --reps 1 --dry --out "$RUN/plan.json"
"$PY" plans.py --show "$RUN/plan.json" | tail -3
"$PY" harness.py run --cfg cfg_ws.json --plan "$RUN/plan.json" --sessionref "$RUN/sref/sessionref.json" --out "$RUN/win" \
  --handoff-from "$HD" --handoff-wait 600 --max-attempts "${MAXATT:-1}" | tee "$RUN/summary.txt"
echo "=== dry_all done $(date -Is)" | tee -a "$RUN/summary.txt"
