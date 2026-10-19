"""ARB3's pod launcher for the d = 13 scripts (checks and smoke timings only, never a registered run):

    python podrun13.py MODULE ARGS...          e.g. surface.sv_speed pipeline ... --dry --joint-w W
    python podrun13.py --derive LOCK OUT       the pod's dry lock

Changes from outside the tree, and nothing else:
  - the card: surface.sv_common.GPU (a full RTX PRO 6000) is the pod's card (POD_GPU or nvidia-smi's name; here an H100
    80GB HBM3, the RTX PRO 6000 being sold out).
  - the kernel caches are not pinned on the pod: sv_common.kcache returns the lock's digest, with the real one under
    "real" (the lock's digest is an sm_120 cache's; the pod builds its own as it goes).
--derive writes the pod's dry lock: the study's lock with "dry": true (dry kind 13 only, as sv_read and sv_speed require
of a dry lock), its identity fields gpu, torch, real_executable, prefix, packages_sha256 and code replaced by this
process's (sv_common.record: the pod's venv and the modified tree), "pod_derivation" saying so; thresholds, frozen
paths, nets and the c7 plan are the lock's."""
import json
import os
import runpy
import subprocess
import sys


def install():
    from surface import sv_common as sc
    if getattr(sc.kcache, "_pod", False):
        return sc
    sc.GPU = os.environ.get("POD_GPU") or subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    real = sc.kcache
    lock = None
    if "--lock" in sys.argv:
        lock = json.load(open(sys.argv[sys.argv.index("--lock") + 1]))

    def kcache():
        r = real()
        return {**(lock["kcache"] if lock else r), "real": r["digest"]}
    kcache._pod = True
    sc.kcache = kcache
    return sc


def derive(lock_path, out):
    import hashlib
    sc = install()
    _, gpu = sc.setup_gpu()
    rec = sc.record(gpu)
    L = json.load(open(lock_path))
    keys = ("gpu", "torch", "real_executable", "prefix", "packages_sha256", "code")
    old = {k: L["identity"].get(k) for k in keys}
    for k in keys:
        L["identity"][k] = rec[k]
    L["dry"] = True
    L["pod_derivation"] = {"lock_sha256": hashlib.sha256(open(lock_path, "rb").read()).hexdigest(), "replaced": keys,
                           "was": old, "why": "ARB3's pod checks: dry kind 13, the pod's card and venv, the modified tree"}
    with open(out, "x") as fh:
        json.dump(L, fh, indent=1, sort_keys=True)
    print(f"podrun13: dry lock {out}: {json.dumps({k: str(rec[k])[:60] for k in keys})}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--derive":
        derive(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) < 2:
            raise SystemExit("usage: podrun13.py MODULE ARGS... | --derive LOCK OUT")
        mod = sys.argv[1]
        sys.argv = [mod] + sys.argv[2:]
        install()
        if mod.endswith(".py"):                                  # a script (checks/bench_joint.py)
            sys.path.insert(0, os.path.dirname(os.path.abspath(mod)))
            runpy.run_path(mod, run_name="__main__")
        else:
            runpy.run_module(mod, run_name="__main__", alter_sys=True)
