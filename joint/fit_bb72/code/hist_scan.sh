#!/bin/bash
# Seeds used anywhere since retest/historical_seeds.txt was frozen (RETEST.md section 2): every 19-digit integer in
# [2^62, 2^63) in (1) every seed, beacon or manifest file on every branch of the research repository <research-repo> and (2) every text
# file of the release clone (weights excluded). Output: seeds/hist_extra.txt (sorted, unique) and its counts on stderr.
set -euo pipefail
HUB=${HUB:-$HOME/<research-repo>}; REL=${REL:-$HOME/arb/release}; OUT=${1:-$HOME/arb/seeds/hist_extra.txt}
filt() { grep -ohE '(^|[^0-9])[0-9]{19}([^0-9]|$)' | grep -oE '[0-9]{19}' | python3 -c "
import sys
lo, hi = 2 ** 62, 2 ** 63
for x in sorted({int(l) for l in sys.stdin if l.strip()}):
    if lo <= x < hi: print(x)"; }
for b in $(git -C "$HUB" for-each-ref --format='%(refname:short)' refs/heads); do
  git -C "$HUB" ls-tree -r "$b" | awk -F'\t' 'tolower($2) ~ /seed|beacon|manifest/ {split($1, a, " "); print a[3]}'
done | sort -u | git -C "$HUB" cat-file --batch | filt > /tmp/arb_hist_hub.txt
( cd "$REL" && grep -rIl --exclude-dir=weights --exclude-dir=.git -E '[0-9]{19}' . | xargs -r cat ) | filt > /tmp/arb_hist_rel.txt
sort -u /tmp/arb_hist_hub.txt /tmp/arb_hist_rel.txt > "$OUT"
echo "hub $(wc -l < /tmp/arb_hist_hub.txt) release $(wc -l < /tmp/arb_hist_rel.txt) union $(wc -l < "$OUT")" >&2
