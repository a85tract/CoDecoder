#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 2 --mem=8G -t 01:00:00 -J rt_histscan
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# Historical seeds on Alpine (RETEST.md section 2): every 19-digit integer in [2^62, 2^63) of every manifest, seeds or
# beacon file of this account's Alpine records (venvs, site-packages and .git excluded), for retest/historical_seeds.txt.
# The re-test's own dry-run and replay records were scanned with the same rule apart (retest/replay/hist/dry.files).
set -uo pipefail
O=<cluster-scratch>/retest/replay/hist_alpine
find <cluster-scratch> <cluster-projects> -maxdepth 7 \( -name "*manifest*" -o -name "*seeds*" -o -name "beacon*.json" \) \
  -type f -size -20M 2>/dev/null | grep -v -E "/(venv[^/]*|site-packages|\.git|retest)/" > $O.files
wc -l < $O.files
tr '\n' '\0' < $O.files | xargs -0 grep -ohE '\b[0-9]{19}\b' 2>/dev/null | sort -u > $O.raw
/usr/bin/python3 -c "
xs = {int(x) for x in open('$O.raw').read().split()}
ys = sorted(x for x in xs if 2**62 <= x < 2**63)
open('$O.txt', 'w').write(''.join(f'{x}\n' for x in ys)); print(len(xs), len(ys))"
