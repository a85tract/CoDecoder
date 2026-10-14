"""The record of a surface pod's plan builds (speed_rt/pod/run_s.sh build), so that the analysis can check that the plans were built
from F_2's files, also when the builds ran before F_2 (STAGE2_surface, the builds):
  tree      every file of the deployed tree the builds read, by relative path -> sha256: the import closure of
            surface_v2/timing/ising_trt.py and speed_rt/ising_d5.py (static: every import statement, function bodies included),
            speed_rt/pod/run_s.sh, this file, NVIDIA's legacy circuits (surface_v2/circuits/d*_legacy.stim) and the studies'
            test-eval tables (surface_v2/results/s*_test_eval.csv). analysis_s.qualifies checks each against its own tree.
  external  NVIDIA's Ising-Decoding with its checkpoints (the SUMS file the setup checked), the validation chunks 0, trtexec's version
  built     every file the builds wrote under ONNX and WORK (ONNX, plans, prep/build/check records), by name -> sha256
    python -m speed_rt.build_inputs --rel REL --onnx ONNX --work WORK --val VAL --nv NV --trtexec-help FILE --out OUT.json
Relative paths and names only; deterministic for the same files."""
import argparse
import ast
import glob
import hashlib
import json
import os

ENTRY = ("surface_v2/timing/ising_trt.py", "speed_rt/ising_d5.py")
EXTRA = ("speed_rt/pod/run_s.sh", "speed_rt/build_inputs.py")
DATA = ("surface_v2/circuits/d*_legacy.stim", "surface_v2/results/s*_test_eval.csv")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def closure(rel):
    """The tree's Python files reachable from ENTRY through import statements (roots surface_v2/ and the tree, as PYTHONPATH)."""
    roots = [os.path.join(rel, "surface_v2"), rel]

    def resolve(mod):
        for r in roots:
            f = os.path.join(r, *mod.split(".")) + ".py"
            if os.path.isfile(f):
                return f
        return None

    def imports(path):
        root = next(r for r in roots if path.startswith(r + os.sep))
        pkg = os.path.relpath(os.path.dirname(path), root).replace(os.sep, ".").split(".")
        out = set()
        for n in ast.walk(ast.parse(open(path).read(), path)):
            if isinstance(n, ast.Import):
                out |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    base = ".".join(pkg[:len(pkg) - n.level + 1] + ([base] if base else []))
                out |= {base} | {f"{base}.{a.name}" for a in n.names}
        return out

    seen, todo = set(), [os.path.join(rel, f) for f in ENTRY]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        todo += [g for g in map(resolve, imports(f)) if g and g not in seen]
    return {os.path.relpath(f, rel) for f in seen}


def files_under(d, prefix, skip=()):
    out = {}
    for base, _, names in os.walk(d):
        for n in names:
            p = os.path.join(base, n)
            if os.path.isfile(p) and os.path.abspath(p) not in skip:
                out[prefix + "/" + os.path.relpath(p, d)] = sha256_file(p)
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("rel", "onnx", "work", "val", "nv", "trtexec-help", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    tree = closure(a.rel) | set(EXTRA)
    for g in DATA:
        tree |= {os.path.relpath(f, a.rel) for f in glob.glob(os.path.join(a.rel, g))}
    dep = os.path.join(a.rel, "DEPLOYED")
    rec = dict(what="the plan builds' inputs and outputs (speed_rt/build_inputs.py)",
               deployed=open(dep).read().strip() if os.path.exists(dep) else None,
               tree={f: sha256_file(os.path.join(a.rel, f)) for f in sorted(tree)},
               external=dict(ising_nv_SUMS=sha256_file(os.path.join(a.nv, "SUMS")),
                             validation_chunk0={os.path.basename(f): sha256_file(f)
                                                for f in sorted(glob.glob(os.path.join(a.val, "*_val_000.npz")))},
                             trtexec=open(a.trtexec_help).readline().strip()),
               built={**files_under(a.onnx, "onnx"), **files_under(a.work, "work", {os.path.abspath(a.out)})})
    with open(a.out + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(a.out + ".tmp", a.out)
    print(f"BUILD_INPUTS {a.out}: {len(rec['tree'])} tree files, {len(rec['built'])} built files, deployed {rec['deployed']}")


if __name__ == "__main__":
    main()
