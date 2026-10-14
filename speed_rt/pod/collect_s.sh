#!/bin/bash
# collect_s.sh DEST low=DIR high=DIR cpu=DIR TESTSETS: puts the surface stage's two pod sessions, the CPU worker's records and the draw's manifests into the layout
# that speed_rt/analysis_surf.py reads (cp only, no network). DIR is a pod's $W as pulled (pod.sh pod2alpine of session, masks and
# res into one directory): DIR/session (attempts.json, META.json, DONE.json, recheck_*, every record), DIR/masks, DIR/res; the
# shots (srt, ref, view) are not copied. cpu=DIR is the CPU worker's return (winserver): cpu_session.json, cpu_attempts.json and cpu/ (section 1.14),
# copied to DEST/cpu. TESTSETS is the directory holding srt-surf.json and srt-ref.json (the draw's manifests).
# When a session was run again (the failure rule's whole-session rerun), pass only the one that counts; the other keeps its own
# directory outside DEST and is published beside the records. DEST is created; a part that exists is refused.
# DEST/COLLECTED.sha256 lists every file copied. Environment: POD_WORK = the pods' working directory (the path their records name).
set -euo pipefail
: "${POD_WORK:?POD_WORK: the working directory of the pods, as their records name it}"
[ $# -eq 5 ] || { sed -n 2,9p "$0"; exit 2; }
DEST=$1; shift; mkdir -p "$DEST"; DEST=$(cd "$DEST" && pwd)
for a in "$@"; do
  case "$a" in
  low=*|high=*) r=${a%%=*}; S=${a#*=}
    [ -f "$S/session/attempts.json" ] && [ -f "$S/session/META.json" ] || { echo "$S: no session/attempts.json and META.json"; exit 2; }
    [ ! -e "$DEST/$r" ] || { echo "$DEST/$r exists"; exit 2; }
    mkdir -p "$DEST/$r"; cp -r "$S/session" "$S/masks" "$S/res" "$DEST/$r/"
    grep -q "\"role\": \"$r\"" "$S/session/META.json" || { echo "$S: META.json names another role"; exit 2; }
    echo "$S -> $DEST/$r" ;;
  cpu=*) C=${a#*=}
    [ -f "$C/cpu_session.json" ] && [ -f "$C/cpu_attempts.json" ] && [ -d "$C/cpu" ] || { echo "$C: no cpu_session.json, cpu_attempts.json and cpu/"; exit 2; }
    [ ! -e "$DEST/cpu" ] || { echo "$DEST/cpu exists"; exit 2; }
    mkdir -p "$DEST/cpu"; cp -r "$C/cpu_session.json" "$C/cpu_attempts.json" "$C/cpu" "$DEST/cpu/"; echo "$C -> $DEST/cpu" ;;
  *) T=$a
    [ -f "$T/srt-surf.json" ] && [ -f "$T/srt-ref.json" ] || { echo "$T: no srt-surf.json and srt-ref.json"; exit 2; }
    [ ! -e "$DEST/testsets" ] || { echo "$DEST/testsets exists"; exit 2; }
    mkdir -p "$DEST/testsets"; cp "$T/srt-surf.json" "$T/srt-ref.json" "$DEST/testsets/" ;;
  esac
done
# No absolute path in a hashed record (GATE UPDATE 29, item 1): the pods' working directory and TensorRT's install directory become tokens in the copied text
# records (commands and process lists that the pod's processes wrote), before anything is hashed; the analysis hashes these copies, which are the published
# records, so publication changes no hash. Any other absolute path (a user's home, a cluster's scratch) stops the collection: the writer must record relative paths.
find "$DEST" -type f \( -name '*.json' -o -name '*.log' -o -name '*.sha256' -o -name '*.txt' \) -print0 | xargs -0 -r sed -i -e "s#$POD_WORK#<pod-work>#g" -e "s#/usr/src/tensorrt#<trt-dir>#g"
bad=$(grep -rIoE '(^|[^A-Za-z0-9_<>.-])/(workspace|root|home|scratch|projects|mnt|Users|tmp)/[^" ,]*|[A-Za-z]:\\\\Users[^" ,]*' "$DEST" 2>/dev/null | head -n 5 | cut -c1-200 || true)
[ -z "$bad" ] || { echo "absolute paths remain (file:path):"; echo "$bad"; exit 2; }
( cd "$DEST" && find . -type f ! -name COLLECTED.sha256 | sort | xargs sha256sum > COLLECTED.sha256 )
echo "collected $(wc -l < "$DEST/COLLECTED.sha256") files into $DEST"
