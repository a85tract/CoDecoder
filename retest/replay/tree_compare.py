"""The original deployed trees the Cascade and head re-tests run from, against this repository's copies (RETEST.md
section 7, item 6): for every file of the original tree that the copy holds at the same relative path, equal bytes, or
the number of changed lines (the copies replace paths, host, account and tool names by placeholders; the changed
lines themselves are not recorded, since they hold what the copies replaced).
    python tree_compare.py OUT_JSON NAME=ORIGINAL_TREE:COPY_DIR ...      (stdlib only)"""
import difflib
import hashlib
import json
import os
import sys


def files(root):
    out = {}
    for d, sub, fs in os.walk(root):
        sub[:] = sorted(s for s in sub if s not in ("__pycache__", ".git"))
        for f in fs:
            p = os.path.join(d, f)
            if os.path.isfile(p) and not f.endswith(".pyc"):
                out[os.path.relpath(p, root)] = p
    return out


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


rep = {}
for arg in sys.argv[2:]:
    name, _, pair = arg.partition("=")
    orig, copy = pair.split(":")
    fo, fc = files(orig), files(copy)
    same, differ, only_orig = [], {}, []
    for rel, p in sorted(fo.items()):
        if rel not in fc:
            only_orig.append(rel)
            continue
        if sha(p) == sha(fc[rel]):
            same.append(rel)
            continue
        try:
            a, b = open(p).read().splitlines(), open(fc[rel]).read().splitlines()
            d = [x for x in difflib.unified_diff(a, b, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---")]
            differ[rel] = {"original_sha256": sha(p), "copy_sha256": sha(fc[rel]), "lines_changed": len(d)}
        except UnicodeDecodeError:
            differ[rel] = {"original_sha256": sha(p), "copy_sha256": sha(fc[rel]), "binary": True}
    rep[name] = {"original": orig, "copy": copy, "original_files": len(fo), "in_copy": len(fo) - len(only_orig),
                 "identical": len(same), "differ": differ, "only_in_original_count": len(only_orig)}
    print(name, len(fo), "files;", len(same), "identical,", len(differ), "differ,", len(only_orig), "not in the copy",
          flush=True)
json.dump(rep, open(sys.argv[1], "w"), indent=1)
