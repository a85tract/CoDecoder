# The re-test's read guards for D1 (RETEST.md test D1; review finding 4), sourced by d1_read_rt.sh and by the replay.
# They are the guards of D1's jobs/d1_guard.sh with its kind-14 freshness search removed (the beacon replaces it):
# d1_guard_rt F P T O returns 1, saying why, when
#   - the tree T is not F: its DEPLOYED_SHA is not F, or its DEPLOYED is not "net 64b2e25d d1 F" (a clean tree at F)
#     [d1_guard.sh, unchanged];
#   - the read at P is already done in both parts (O/read/done_P and O/compiled/done_P)   [d1_guard.sh, unchanged,
#     now applied to the re-test's output directory O];
#   - O lies inside D1's original outputs (the original reads are never written to);
#   - O/read/block_P.json exists and names other seeds or another beacon than RT_SEEDS / RT_BEACON (a partial read is only
#     continued under the same seeds);
#   - dry and real seeds are mixed: the dry beacon (drand round 6,540,000, a past round) needs an O under /dry/ and
#     RT_DRY_CHUNKS may only be set with it; a real beacon needs RT_DRY_CHUNKS unset and an O outside /dry/.
# Removed: the search of the audited Alpine directories for JSON records naming seed kind 14 (D1's original reads name
# it, so the test would refuse every re-test read; the drand round, drawn after the freeze is public, takes its place).
RT_DRY_ROUND=6540000
d1_guard_rt() {
  local F=$1 P=$2 T=$3 O=$4 sha tag rounds round seeds_round ssha bsha blk
  sha=$(cat "$T/DEPLOYED_SHA" 2>/dev/null || true)
  tag=$(cat "$T/DEPLOYED" 2>/dev/null || true)
  if [ "$sha" != "$F" ]; then echo "GUARD: the deployed tree is '$sha', not F = $F"; return 1; fi
  if [ "$tag" != "net 64b2e25d d1 $F" ]; then echo "GUARD: the deployed tree is '$tag', not a clean tree at F"; return 1; fi
  case "$(readlink -m "$O")/" in                      # <cluster-scratch-root> is a symlink: compare resolved paths
    "$(readlink -m <cluster-scratch>/cascade_d1)"/*) echo "GUARD: $O is inside D1's original outputs"; return 1;;
  esac
  if [ -f "$O/read/done_$P" ] && [ -f "$O/compiled/done_$P" ]; then echo "GUARD: the read at $P is already done"; return 1; fi
  if [ ! -f "${RT_BEACON:-}" ] || [ ! -f "${RT_SEEDS:-}" ]; then echo "GUARD: RT_BEACON and RT_SEEDS must name files"; return 1; fi
  rounds=$(${PY:-python3} -c "import json, sys; print(json.load(open(sys.argv[1]))['round'], json.load(open(sys.argv[2]))['beacon_round'])" \
           "$RT_BEACON" "$RT_SEEDS") || { echo "GUARD: cannot read the beacon and seeds files"; return 1; }
  round=${rounds% *}; seeds_round=${rounds#* }
  if [ "$round" != "$seeds_round" ]; then echo "GUARD: the seeds file is of round $seeds_round, the beacon's is $round"; return 1; fi
  if [ "$round" = "$RT_DRY_ROUND" ]; then
    case "$O" in */dry/*) ;; *) echo "GUARD: the dry seeds need an output directory under /dry/, not $O"; return 1;; esac
  else
    if [ -n "${RT_DRY_CHUNKS:-}" ]; then echo "GUARD: RT_DRY_CHUNKS is set with the real seeds (round $round)"; return 1; fi
    case "$O" in */dry/*) echo "GUARD: the real seeds (round $round) cannot write under /dry/: $O"; return 1;; esac
  fi
  blk="$O/read/block_$P.json"
  if [ -f "$blk" ]; then
    ssha=$(sha256sum "$RT_SEEDS" | cut -c1-64); bsha=$(sha256sum "$RT_BEACON" | cut -c1-64)
    if ! grep -q "\"seeds_sha256\": \"$ssha\"" "$blk" || ! grep -q "\"beacon_sha256\": \"$bsha\"" "$blk"; then
      echo "GUARD: $blk names other seeds or another beacon: a read is never continued under other seeds"; return 1
    fi
  fi
  echo "guard passed: tree $F, clean; the read at $P is not done; seeds of round $round; output $O"
  return 0
}
