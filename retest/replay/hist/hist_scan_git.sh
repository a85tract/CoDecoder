#!/bin/bash
# Historical seeds in the research hub (every branch) and the release (RETEST.md section 2), with the file-name rule of
# retest/jobs/hist_scan.sbatch (manifest, seeds or beacon files): every 19-digit integer in [2^62, 2^63).
#   bash hist_scan_git.sh HUB_CLONE RELEASE_TREE OUT     -> OUT/{hub,release}.{files,txt}
set -euo pipefail
H=$1; R=$2; O=$3; mkdir -p $O; : > $O/hub.files; : > $O/hub.raw
P=(':(glob)**/*manifest*' ':(glob)**/*seeds*' ':(glob)**/beacon*.json')
for b in $(git -C $H for-each-ref --format='%(refname)' refs/remotes/origin | grep -v '/HEAD$'); do
  git -C $H ls-tree -r --name-only $b | grep -E '(^|/)[^/]*(manifest|seeds)[^/]*$|(^|/)beacon[^/]*\.json$' \
    | sed "s#^#${b#refs/remotes/origin/}:#" >> $O/hub.files || true
  git -C $H grep -h -o -E '\b[0-9]{19}\b' $b -- "${P[@]}" >> $O/hub.raw || true
done
(cd $R && find . -type f \( -name "*manifest*" -o -name "*seeds*" -o -name "beacon*.json" \) -not -path "./.git/*" | LC_ALL=C sort) > $O/release.files
(cd $R && tr '\n' '\0' < $O/release.files | xargs -0 grep -ohE '\b[0-9]{19}\b' || true) > $O/release.raw
for s in hub release; do
  python3 -c "
xs = sorted({int(x) for x in open('$O/$s.raw').read().split() if 2**62 <= int(x) < 2**63})
open('$O/$s.txt', 'w').write(''.join(f'{x}\n' for x in xs))
print('$s', 'files', sum(1 for _ in open('$O/$s.files')), 'seeds', len(xs))"
  rm $O/$s.raw
done
