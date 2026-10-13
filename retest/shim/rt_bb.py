"""Re-test shim for the frozen BB code (RETEST.md section 3.1). It runs one command of bb72/ising_compare (Table I's
BB72 columns: stage 3 and amendment S) or gross_stage4/ising_compare (the gross column: stage 4), whichever tree is on
PYTHONPATH, and changes only these names before the command runs, each only when its variable is set:
  RT_SEEDS (with RT_BEACON)  bb.shots.seed_of on the test splits (bb72 "test" at p 0.003 and 0.002, gross_stage4
                             "test4"): chunk j -> retest/seeds.py chunk_seeds(set)[j] (sets BB72-3, BB72-2, gross);
                             every other split and code goes to the frozen function; bb.shots.BEACON, BEACON_GROSS4
                             name the re-test's beacon record
  RT_MANIFEST                bb.shots.MANIFEST: the re-test's shot manifest; shots.load checks every chunk against it
  RT_STAGE, RT_CELLS         bb.stage3.STAGE3, CELLS (bb72 tree) or bb.stage4_gross.STAGE4G, CELL (gross_stage4 tree):
                             the re-test's frozen-parameter copies, written by --prepare
  RT_DRY_SHOTS               dry runs only (refused unless RT_BEACON is a dry record): chunks of this many shots
The splits keep their names, so the frozen commands, records and analyses read them unchanged; the re-test's shots,
predictions and outputs live in their own directories (BB_SHOTS, BB_PREDS). With none of the variables set the shim
only launches the command (the replays of RETEST.md section 6).
    python retest/shim/rt_bb.py MODULE ARGS...       e.g. rt_bb.py bb.shots --code bb72 --p 0.003 --split test ...
    python retest/shim/rt_bb.py SCRIPT.py ARGS...    e.g. the pack scripts, bb/scripts/stage3_pack.py
    python retest/shim/rt_bb.py --prepare OUT [--gpu NAME|auto]   the re-test's CPU-term rows and frozen-parameter copies"""
import csv
import importlib
import json
import os
import sys

RT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIED_GPUS = ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 80GB HBM3")     # RETEST.md section 6
SETS = {("test", "bb72", 0.003): "BB72-3", ("test", "bb72", 0.002): "BB72-2", ("test4", "bb144", 0.004): "gross"}


def setup():
    from bb import shots
    if os.environ.get("RT_SEEDS"):
        sys.path.insert(0, RT)
        import seeds
        table = {k: seeds.chunk_seeds(v) for k, v in SETS.items()}     # each seed checked against RT_BEACON
        frozen = shots.seed_of

        def seed_of(split, code, p, j):
            k = (split, code, round(p, 6))
            if k in table:
                if not 0 <= j < len(table[k]):
                    raise ValueError(f"{k}: chunk {j} is outside the re-test set {SETS[k]}")
                return table[k][j]
            if split in ("test", "test4"):
                raise ValueError(f"{k}: no re-test set")
            return frozen(split, code, p, j)
        shots.seed_of = seed_of
        for name in ("BEACON", "BEACON_GROSS", "BEACON_GROSS4"):
            if hasattr(shots, name):
                setattr(shots, name, os.environ["RT_BEACON"])
    if os.environ.get("RT_DRY_SHOTS"):
        if json.load(open(os.environ["RT_BEACON"])).get("dry") is not True:
            raise SystemExit("RT_DRY_SHOTS with a beacon that is not a dry record")
        shots.sample.__defaults__ = (int(os.environ["RT_DRY_SHOTS"]),)
    if os.environ.get("RT_MANIFEST"):
        shots.MANIFEST = os.environ["RT_MANIFEST"]
    if os.environ.get("RT_STAGE"):
        if hasattr(shots, "TEST4_TAG"):                     # gross_stage4's tree
            from bb import stage4_gross as s4
            s4.STAGE4G, s4.CELL = os.environ["RT_STAGE"], os.environ["RT_CELLS"]
        else:
            from bb import stage3 as s3
            s3.STAGE3, s3.CELLS = os.environ["RT_STAGE"], os.environ["RT_CELLS"]
    return shots


def cputerm_rows(out):
    """CPU term = the re-test's routed fraction x each registered process's mean fallback time per routed shot
    (the frozen rule's other factor), computed as value_ms = registered value_ms x (re-test fraction / registered
    fraction): registered value_ms / registered fraction is that process's full-precision mean per routed shot, and
    with the registered shots the row is reproduced bit for bit. Reads the frozen records of the tree on PYTHONPATH;
    every other row is copied unchanged. Writes <rows>_retest.csv in OUT; the original rows files stay as they are."""
    from bb import route
    shots = setup()
    d = os.path.join(shots.HERE, "data", "bb")
    if os.path.exists(os.path.join(d, "stage4_gross", "cell.json")):            # gross_stage4's tree
        c = json.load(open(os.path.join(d, "stage4_gross", "cell.json")))
        spec = [(c["rows"]["cpu"], {n: ("test4", x["tau"], c["tag"], c["code"], c["p"], c["chunks"])
                                    for n, x in c["cells"].items()})]
    else:                                                                       # bb72's tree: stage 3 and amendment S
        c = json.load(open(os.path.join(d, "stage3", "cells.json")))
        am = json.load(open(os.path.join(d, "stage3", "amendment_S.json")))

        def cell_spec(cs):
            return {n: ("test", x["tau"], x["tag"]) + tuple(c["splits"][x["split"]][k] for k in ("code", "p", "chunks"))
                    for n, x in cs.items() if x.get("fallback")}
        spec = [(c["rows"]["cpu"], cell_spec(c["cells"])), (am["rows_cpu"], cell_spec(am["cells"]))]
    files, report = {}, {}
    for rel, cs in spec:
        rows = list(csv.DictReader(open(os.path.join(shots.HERE, rel))))
        for r in rows:
            note = json.loads(r["note"]) if r.get("note") else {}
            if r["role"] != "cpu_term" or r["name"] not in cs or note.get("split") != cs[r["name"]][0]:
                continue                                  # opponents, GPU rows and pre-freeze (cal) rows: unchanged
            sp, tau, tag, code, p, ch = cs[r["name"]]
            margin = route._load_net(code, p, sp, shots.parse_range(ch), tag)[1]
            k = int((margin < tau).sum())
            frac = k / len(margin)
            v, f = float(r["value_ms"]), float(note["routed_fraction"])
            note.update(registered_value_ms=v, registered_routed_fraction=f, retest_routed=k, retest_shots=len(margin),
                        retest_routed_fraction=frac, estimate="from the registered timing session",
                        rule="value_ms = registered value_ms x (retest_routed_fraction / registered_routed_fraction) "
                             "(RETEST.md section 4)")
            r["value_ms"] = repr(v * (frac / f))
            r["note"] = json.dumps(note)
            report[r["name"]] = {"routed": k, "shots": len(margin), "fraction": frac}
        dst = os.path.join(out, os.path.basename(rel).replace(".csv", "_retest.csv"))
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        files[rel] = dst
    return files, report


def read_gpu():
    """The GPU of BB72's network reads, from their own engine signatures: every read and chunk of the cells must name
    the same GPU, one of the two platforms checked in RETEST.md section 6."""
    from bb import route, shots
    c = json.load(open(os.path.join(shots.HERE, "data", "bb", "stage3", "cells.json")))
    reads = {(x["split"], x["tag"]) for x in c["cells"].values()}
    gpus = {route._load_net(c["splits"][sp]["code"], c["splits"][sp]["p"], "test",
                            shots.parse_range(c["splits"][sp]["chunks"]), tag)[2]["gpu"] for sp, tag in sorted(reads)}
    if len(gpus) != 1 or not gpus <= set(VERIFIED_GPUS):
        raise SystemExit(f"--gpu auto: BB72's network reads ran on {sorted(gpus)}")
    return gpus.pop()


def verify_copies():
    """The parameter copies of RT_STAGE differ from the tree's frozen files only where --prepare changes them: the
    CPU-term rows (a file in RT_STAGE) and, in bb72's cells.json, the GPU field (one of VERIFIED_GPUS, the frozen value
    kept as gpu_frozen)."""
    from bb import shots
    d, st = os.path.join(shots.HERE, "data", "bb"), os.environ["RT_STAGE"]
    if os.path.exists(os.path.join(d, "stage4_gross", "cell.json")):
        pairs = [("stage4_gross/cell.json", os.environ["RT_CELLS"])]
    else:
        pairs = [("stage3/cells.json", os.environ["RT_CELLS"]), ("stage3/amendment_S.json", os.path.join(st, "amendment_S.json"))]
    for rel, cp in pairs:
        f, c = json.load(open(os.path.join(d, rel))), json.load(open(cp))
        rows = c["rows"]["cpu"] if "rows" in f else c["rows_cpu"]
        if os.path.dirname(os.path.abspath(rows)) != os.path.abspath(st):
            raise SystemExit(f"{cp}: its CPU-term rows {rows} are not the derived rows in {st}")
        if "rows" in f:
            c["rows"] = dict(c["rows"], cpu=f["rows"]["cpu"])
        else:
            c["rows_cpu"] = f["rows_cpu"]
        if "gpu_frozen" in c:
            if c.pop("gpu_frozen") != f["gpu"] or c["gpu"] not in VERIFIED_GPUS:
                raise SystemExit(f"{cp}: the GPU field is not a checked platform with the frozen value kept")
            c["gpu"] = f["gpu"]
        if c != f:
            raise SystemExit(f"{cp} differs from the frozen {rel} beyond the CPU-term rows and the GPU field")


def prepare(out, gpu=""):
    """The re-test's frozen-parameter copies in OUT: the tree's cells.json and amendment_S.json (bb72) or cell.json
    (gross_stage4) with exactly two changes: the CPU-term rows file -> the derived rows (cputerm_rows), and, with --gpu,
    the field the analysis compares with every read's engine signature ("gpu", bb72's cells.json) -> the re-test's
    read GPU, the frozen value kept as gpu_frozen (--gpu auto: the GPU the reads' own signatures name, read_gpu; the
    frozen GPU leaves the field as it is). Run once, after the network reads, before the analysis."""
    os.makedirs(out, exist_ok=True)
    files, report = cputerm_rows(out)
    from bb import shots
    d = os.path.join(shots.HERE, "data", "bb")
    if os.path.exists(os.path.join(d, "stage4_gross", "cell.json")):
        if gpu:
            raise SystemExit("--gpu: the gross read runs on the frozen engine (cell.json engine, an H100)")
        c = json.load(open(os.path.join(d, "stage4_gross", "cell.json")))
        c["rows"]["cpu"] = files[c["rows"]["cpu"]]
        json.dump(c, open(os.path.join(out, "cell.json"), "w"), indent=1)
    else:
        c = json.load(open(os.path.join(d, "stage3", "cells.json")))
        am = json.load(open(os.path.join(d, "stage3", "amendment_S.json")))
        c["rows"]["cpu"], am["rows_cpu"] = files[c["rows"]["cpu"]], files[am["rows_cpu"]]
        if gpu == "auto":
            gpu = read_gpu()
            gpu = "" if gpu == c["gpu"] else gpu
        if gpu:
            c["gpu_frozen"], c["gpu"] = c["gpu"], gpu
        json.dump(c, open(os.path.join(out, "cells.json"), "w"), indent=1)
        json.dump(am, open(os.path.join(out, "amendment_S.json"), "w"), indent=1)
    print(json.dumps({"rows": files, "routed": report, "gpu": gpu or "frozen"}), flush=True)


def main():
    if sys.argv[1] == "--prepare":
        return prepare(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--gpu" else "")
    setup()
    if os.environ.get("RT_STAGE"):
        verify_copies()
    if sys.argv[1].endswith(".py"):                        # a script (the pack scripts of bb/scripts/), no pools
        import runpy
        sys.argv = sys.argv[1:]
        return runpy.run_path(sys.argv[0], run_name="__main__")
    mod = importlib.import_module(sys.argv[1])
    sys.argv = [sys.argv[1]] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()
