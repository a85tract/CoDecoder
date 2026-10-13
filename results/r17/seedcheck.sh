#!/bin/bash
# seedcheck.sh REPO... : R17's test-set seeds (PLAN_R17.md section 2) against every branch of the given repositories.
# T6's seeds are 91,000,000 + 10,000 * 17 + j, j = 0..99 (91,170,000-91,170,099). The script searches every text file
# of every branch for any integer of the 91 M namespace (91,000,000-91,999,999) written as digits (91xxxxxx), with
# underscores (91_xxx_xxx) or commas (91,xxx,xxx), in exponent form (9.1e7, 91e6) or as a base (91 M, 91M,
# 91 * 10**6), and prints each hit with its repository, branch and line. It then prints the smallest seed of every
# registered test-set manifest it finds (T2-T5 are drawn at 2^62 or above). R17's own directory (ising_compare/r17,
# which defines the seeds) is left out. git only; the expected result is "hits: 0".
#   bash ising_compare/r17/seedcheck.sh ~/work/repo ~/work/release ~/work/paper.git
# It ran on the research repositories before T6 was drawn. In this repository, T6's manifest and the documents that
# describe it hold such numbers by design.
set -uo pipefail
B='(^|[^0-9A-Za-z_.])'                  # a number starts here: not inside a word, a hex string or a decimal
P="${B}91[0-9]{6}([^0-9]|\$)"
P+="|${B}91_[0-9]{3}_[0-9]{3}([^0-9]|\$)"
P+="|(^|[^0-9A-Za-z_.,])91,[0-9]{3},[0-9]{3}([^0-9,]|\$)"          # not one field of a comma-separated list
P+="|${B}(9\\.1[eE]\\+?0?7|91[eE]\\+?0?6|91\\.0[eE]\\+?0?6)([^0-9A-Za-z_]|\$)"
P+="|${B}91 ?M([^A-Za-z0-9]|\$)"
P+="|${B}91 ?\\* ?(10 ?\\*\\* ?6|1_?000_?000|1e6)"
hits=0; refs=0
for R in "$@"; do
  for ref in $(git -C "$R" for-each-ref --format='%(refname)' refs/heads refs/remotes | grep -v '/HEAD$'); do
    refs=$((refs + 1))
    out=$(git -C "$R" grep -I -n -E -e "$P" "$ref" -- . ':(exclude)ising_compare/r17' 2>/dev/null)
    [ -n "$out" ] || continue
    n=$(printf '%s\n' "$out" | wc -l); hits=$((hits + n))
    printf '%s\n' "$out" | cut -c1-220 | sed "s#^#$(basename "$R") #"
  done
done
echo "repositories: $#; branches searched: $refs; hits: $hits"
for R in "$@"; do
  for ref in $(git -C "$R" for-each-ref --format='%(refname)' refs/heads refs/remotes | grep -v '/HEAD$'); do
    git -C "$R" ls-tree -r --name-only "$ref" | grep -E '(^|/)(T[2-5]|t[45]_manifest)\.json$' | while read -r f; do
      m=$(git -C "$R" show "$ref:$f" | grep -oE '"seed": *[0-9]+' | grep -oE '[0-9]+$' | sort -n | head -1)
      [ -n "$m" ] && echo "$(basename "$R") ${ref#refs/remotes/} $f: smallest seed $m"
    done
  done
done | sort -u -k3,3 -k1,1 | awk '!seen[$3]++'
echo "2^62 = 4611686018427387904"
