"""X2 adapter for D1's re-test (RETEST.md, test D1; review finding 7). It runs X2's BB72 analysis, hub branch
ising-rev-x2 commit 20179987 (revision/x2/bb_d1.py: per-shot features of D1's eager read; revision/x2/bb_join.py: the
tables and fig.npz), from the byte-identical copies in retest/x2/ (SOURCE.txt lists their sha256; refused if they differ).
The analysis is exploratory: it runs after the confirmatory d1_confirm.py and is reported apart from its verdicts.

What the adapter changes in bb_d1.py: only the names that tie it to D1's original files, set before bb_d1.main() runs.
  ARCH      the directory holding read/p0.00{2,3}00.{npz,json} and compiled/p0.00{2,3}00.{npz,json} (the re-test's
            confirm_in, or the original cascade_d1/out for the replay)
  NPZ_SHA   the sha256 of each eager read's npz: from a committed record, D1's npz_sha256.txt ("sha256  path" lines) or
            the re-test's confirm/npz_sha256.txt, in place of the two original hashes
  REG       the outcome counts bb_d1 asserts on the eager path (routed, S55'-deep fails, C14' fails, kept breaks,
            repairs, both-fail): from d1_result.json of the confirmatory analysis (points.<p>.paths.eager), in place of the
            original counts
  OUT       $X2_OUT/bb (set from the environment, as bb_d1 does)
Unchanged: the eager path, TAU1 (checked equal to the lock's tau1 in d1_result.json), the circuit and its sha256 check, the
detector layout, the chunk-digest check, every feature and the percentile / binning code of bb_join.py.

    python rt_x2.py run --inputs DIR --result d1_result.json --npz-sha npz_sha256.txt --out X2_OUT   bb_d1 run, then bb_join
    python rt_x2.py check --result d1_result.json --npz-sha npz_sha256.txt                           the derived NPZ_SHA and REG
                                                                                                      against bb_d1's originals"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X2 = os.path.join(os.path.dirname(HERE), "x2")
PS = (0.002, 0.003)
F = "310ce8e0f8bfb9d3b97b27186547a836d29fc9ca"
LOCK_SHA = "3ab39941df356883be395ef7bbf39672fdb546729187bd89fa523be22714f96c"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frozen_sources():
    """retest/x2/SOURCE.txt's sha256 of each copied file, checked against the files."""
    want = {}
    for line in open(os.path.join(X2, "SOURCE.txt")):
        if not line.startswith("#") and line.strip():
            want[line.split()[1]] = line.split()[0]
    for name, h in want.items():
        if sha256(os.path.join(X2, name)) != h:
            raise SystemExit(f"rt_x2: {name} is not the byte-identical copy recorded in x2/SOURCE.txt")
    return want


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(X2, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def derive(result, npz_sha, lock_sha=LOCK_SHA):
    """(NPZ_SHA, REG, tau1) of the committed records: d1_result.json and the npz sha256 list."""
    r = json.load(open(result))
    if r["F"] != F or r["lock_sha256"] != lock_sha:
        raise SystemExit(f"rt_x2: {result} is not an analysis under D1's F and lock")
    reg, tau = {}, {}
    for p in PS:
        pt = r["points"][str(p)]
        e = pt["paths"]["eager"]
        if e["S_alone_vs_C"]["fail_b"] != e["C"]["fail"]:
            raise SystemExit(f"rt_x2: {result}: C14' failures at {p} differ between C and S_alone_vs_C")
        reg[p] = dict(routed=e["routed"], S=e["S_alone_vs_C"]["fail_a"], C=e["C"]["fail"], breaks=e["kept_breaks"],
                      repairs=e["kept_repairs"], both=e["kept_both"])
        tau[p] = pt["tau1"]
    sha = {}
    for line in open(npz_sha):
        if line.strip():
            h, path = line.split()
            for p in PS:
                if path.endswith(f"read/p{p:.5f}.npz"):
                    sha[p] = h
    if sorted(sha) != list(PS):
        raise SystemExit(f"rt_x2: {npz_sha} lacks a read npz sha256 at both p")
    return sha, reg, tau


def adapted(result, npz_sha, inputs, out, lock_sha=LOCK_SHA):
    os.environ["X2_OUT"] = out
    src = frozen_sources()
    d1 = load("bb_d1")
    sha, reg, tau = derive(result, npz_sha, lock_sha)
    if {p: d1.TAU1[p] for p in PS} != tau:
        raise SystemExit(f"rt_x2: bb_d1's TAU1 {d1.TAU1} is not the lock's tau1 {tau} in {result}")
    info = {"sources": src, "tau1": tau, "NPZ_SHA": {str(p): v for p, v in sha.items()},
            "REG": {str(p): v for p, v in reg.items()}, "result": os.path.abspath(result),
            "npz_sha256": os.path.abspath(npz_sha), "inputs": os.path.abspath(inputs) if inputs else None,
            "NPZ_SHA_equals_bb_d1_original": sha == d1.NPZ_SHA, "REG_equals_bb_d1_original": reg == d1.REG}
    d1.NPZ_SHA, d1.REG = sha, reg
    if inputs:
        d1.ARCH = os.path.abspath(inputs)
    return d1, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "check"))
    ap.add_argument("--result", required=True)
    ap.add_argument("--npz-sha", required=True)
    ap.add_argument("--inputs")
    ap.add_argument("--out")
    ap.add_argument("--lock-sha", default=LOCK_SHA, help="the lock the analysis ran under (D1's; the dry run's own lock)")
    a = ap.parse_args()
    if a.what == "check":
        d1, info = adapted(a.result, a.npz_sha, None, os.environ.get("X2_OUT", "out"), a.lock_sha)
        print(json.dumps(info, indent=1))
        sys.exit(0 if info["NPZ_SHA_equals_bb_d1_original"] and info["REG_equals_bb_d1_original"] else 1)
    if not (a.inputs and a.out):
        raise SystemExit("rt_x2 run: --inputs and --out are required")
    d1, info = adapted(a.result, a.npz_sha, a.inputs, a.out, a.lock_sha)
    os.makedirs(os.path.join(a.out, "bb"), exist_ok=True)
    json.dump(info, open(os.path.join(a.out, "bb", "adapter.json"), "w"), indent=1)
    print("rt_x2: " + json.dumps({k: info[k] for k in ("NPZ_SHA", "REG", "tau1")}), flush=True)
    sys.argv = ["bb_d1.py", "run"]
    d1.main()
    join = load("bb_join")
    join.main()


if __name__ == "__main__":
    main()
