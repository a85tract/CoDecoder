"""SPEED-RT stage 3: the records' sanitization for publication (the release's SANITIZE.md), before the analysis runs, so
that the published records reproduce the outputs. In every text file (.json, .jsonl, .txt, .csv, .log) under DIR: the
workstation's host name -> "workstation", each other account name -> "user<k>" (k by first appearance in sorted file
order), home paths -> "~". Nothing else changes.

    python -m speed_rt.s3.sanitize3 DIR --host NAME --users NAME[,NAME...]"""
import argparse
import os
import re

HOMES = os.path.dirname(os.path.expanduser("~"))        # the directory that holds the accounts' home directories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--host", required=True)
    ap.add_argument("--users", required=True, help="a regular expression of account names")
    a = ap.parse_args()
    users, n = {}, 0
    pat = re.compile(a.users)
    for root, _, files in sorted(os.walk(a.dir)):
        for f in sorted(files):
            if not f.endswith((".json", ".jsonl", ".txt", ".csv", ".log")):
                continue
            p = os.path.join(root, f)
            s = open(p, encoding="utf-8", errors="surrogateescape").read()
            t = s.replace(a.host, "workstation")
            t = re.sub(re.escape(HOMES) + r"/[A-Za-z0-9_.-]+", "~", t)

            def sub(m):
                u = m.group(0)
                if u not in users:
                    users[u] = f"user{len(users) + 1}"
                return users[u]
            t = pat.sub(sub, t)
            if t != s:
                open(p, "w", encoding="utf-8", errors="surrogateescape").write(t)
                n += 1
    print(f"{n} files changed; {len(users)} account names replaced")


if __name__ == "__main__":
    main()
