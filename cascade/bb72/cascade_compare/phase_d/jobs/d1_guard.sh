# The read job's guards (PHASE_D1_PLAN.md 2.2 and 2.3), sourced by d1_read.sh and by the dry runs (perturbed inputs).
# d1_guard F P T O returns 1, saying why, when
#   - the tree T is not F: its DEPLOYED_SHA is not F, or its DEPLOYED is not "net 64b2e25d d1 F" (a clean tree at F);
#   - a JSON record under SEARCH, outside D1's read directories (O/read, O/compiled) and outside code trees and
#     caches, names seed kind 14 ("kind" or "kind_seed");
#   - the read at P is already done in both parts (O/read/done_P and O/compiled/done_P).
# SEARCH defaults to the audit's directories (PHASE_D1_PLAN.md appendix) and D1's own.
D1_SEARCH_DEFAULT="$(for d in cascade_w1 cascade_w2 cascade_w3 cascade_w4 cascade_w5 cascade_batchopt cascade_explore \
  cascade_h100b cascade_h100t cascade_d1; do printf '<cluster-scratch>/%s ' $d; done)"
d1_guard() {
  local F=$1 P=$2 T=$3 O=$4 sha tag hits
  sha=$(cat "$T/DEPLOYED_SHA" 2>/dev/null || true)
  tag=$(cat "$T/DEPLOYED" 2>/dev/null || true)
  if [ "$sha" != "$F" ]; then echo "GUARD: the deployed tree is '$sha', not F = $F"; return 1; fi
  if [ "$tag" != "net 64b2e25d d1 $F" ]; then echo "GUARD: the deployed tree is '$tag', not a clean tree at F"; return 1; fi
  hits=$(for d in ${SEARCH:-$D1_SEARCH_DEFAULT}; do
           [ -d "$d" ] && grep -rlE --include='*.json' --exclude-dir='code.*' --exclude-dir=cache --exclude-dir=jobcache \
             '"kind(_seed)?": *14([^0-9]|$)' "$d" 2>/dev/null
         done | grep -v -e "^$O/read/" -e "^$O/compiled/" | head -5 || true)
  if [ -n "$hits" ]; then echo "GUARD: records outside D1's read directories name kind 14: $hits"; return 1; fi
  if [ -f "$O/read/done_$P" ] && [ -f "$O/compiled/done_$P" ]; then echo "GUARD: the read at $P is already done"; return 1; fi
  echo "guard passed: tree $F, clean; no other record names kind 14; the read at $P is not done"
  return 0
}
