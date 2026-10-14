#!/bin/bash
# D9-SHALLOW's analysis inputs (PLAN.md section 6): RECORDS from commit B's tree (TREE), the pod's pulled files (POD:
# shots/, session/, reads/ as pod_run.sh wrote them, checked against POD/PULL.sha256) and the CPU session's export
# (CPU = the export's surface/records/cpu). RECORDS.sha256 beside RECORDS lists every file.
#   bash d9_shallow/collect_d9s.sh TREE POD CPU RECORDS
set -euo pipefail
T=$1; P=$2; C=$3; O=$4
[ ! -e "$O" ] || { echo "$O exists"; exit 1; }
(cd "$P" && sha256sum --quiet -c PULL.sha256) || { echo "the pod's files fail PULL.sha256"; exit 1; }
mkdir -p "$O/testsets" "$O/shots" "$O/pod" "$O/reads" "$O/cpu"
cp "$T"/d9_shallow/testsets/beacon_d9s.json "$T"/d9_shallow/testsets/d9s_test.json "$T"/d9_shallow/testsets/d9s_timing.json "$O/testsets/"
cp -r "$P/shots/d9stest" "$P/shots/d9stiming" "$O/shots/"
cp -r "$P/session/." "$O/pod/"
cp "$P"/reads/d9stest_*.npz "$P"/reads/d9stest_*.json "$O/reads/"
cp -r "$C/." "$O/cpu/"
(cd "$O" && find . -type f | LC_ALL=C sort | xargs sha256sum) > "$O.sha256"
echo "$O: $(wc -l < "$O.sha256") files"
