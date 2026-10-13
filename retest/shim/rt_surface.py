"""Re-test shim for the Cascade test on the surface code at d = 13 (RETEST.md section d = 13). It runs one command of
the pinned tree's `surface` package (cascade_w5/code.20261012_060258, code digest b15af28f3e162378) and changes one
thing before the command runs: surface.surface_data.eval_seed, for kind 11 (the confirmation block) only.
    python rt_surface.py MODULE ARGS...    e.g. rt_surface.py surface.sv_read --stage conf --p 0.006 --chunks 0-100 ...
                                                rt_surface.py surface.sv_route confirm --lock ... --conf DIR ...
    python rt_surface.py --check-dry       exit 0 iff the seeds in use are the dry ones (a past drand round)
The patch is active iff RT_SEEDS is set (the seeds file; RT_BEACON, the beacon record, must be set too). Without it
this file is an identity launcher: eval_seed stays the pinned tree's function object.
Active: eval_seed(13, 11, p, j) = seeds.chunk_seeds("d13-0.006" | "d13-0.0035")[j] (retest/seeds.py: each seed of the
file recomputed from the beacon record, an error on any mismatch) for p = 0.006 | 0.0035; any other kind goes to the
pinned function unchanged (its own range checks run first for kind 11 too); a kind-11 request outside the two sets
(another d or p, a chunk beyond the set) raises, so a read can never fall back to the original seeds of kind 11.
The patch is installed at import, at the top level of this file: the pinned tree's spawned pools (surface_data.
stream_shots' sampling pool, sv_read's M workers: mp.get_context("spawn")) re-run the main script as __mp_main__ in
every worker before a task is unpickled, so every process draws the same seeds; the target runs only under
`if __name__ == "__main__"`. The workers inherit RT_SEEDS / RT_BEACON through the environment.
Dry switches (refused unless the seeds in use are the dry ones: the beacon record says "dry": true and the seeds file
names the beacon's round):
  RT_DRY_CONFIRM=a-b   sv_route confirm on a dry read of chunks a .. b-1 at p = 0.006 only: load_read is cut to that
                       range, the family to 20 hypotheses and sc.PS to (0.006,). Nothing else changes."""
import hashlib
import importlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RETEST = os.path.dirname(HERE)
KIND_CONF, D = 11, 13
SETS = {6000: "d13-0.006", 3500: "d13-0.0035"}        # round(1e6 p) -> the seeds set of the re-test (retest/tags.json)


def _seeds_module():
    """retest/seeds.py loaded by path under its own name: no sys.path entry, so no module name can be shadowed."""
    spec = importlib.util.spec_from_file_location("rt_seeds", os.path.join(RETEST, "seeds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_dry():
    """True iff RT_BEACON is a dry record and RT_SEEDS is a seeds file of that record's round."""
    try:
        b, s = json.load(open(os.environ["RT_BEACON"])), json.load(open(os.environ["RT_SEEDS"]))
    except (KeyError, OSError, ValueError):
        return False
    return b.get("dry") is True and s.get("beacon_round") == b.get("round")


def table():
    """{round(1e6 p): seeds of that p's set}, checked against the beacon (seeds.chunk_seeds)."""
    if not os.environ.get("RT_BEACON"):
        raise SystemExit("rt_surface: RT_SEEDS is set but RT_BEACON is not")
    rs = _seeds_module()
    return {k: rs.chunk_seeds(name) for k, name in SETS.items()}


def install():
    """Replace surface_data.eval_seed for kind 11 when RT_SEEDS is set -> the table, or None when inactive."""
    if not os.environ.get("RT_SEEDS"):
        return None
    from surface import surface_data as sd
    if getattr(sd.eval_seed, "rt_table", None) is not None:
        return sd.eval_seed.rt_table
    orig, tbl = sd.eval_seed, table()

    def eval_seed(d, kind, p, chunk):
        base = orig(d, kind, p, chunk)                     # the pinned function's own checks and value
        if kind != KIND_CONF:
            return base
        k = int(round(p * 1e6))
        if d != D or k not in tbl or not 0 <= int(chunk) < len(tbl[k]):
            raise ValueError(f"rt_surface: kind 11 has seeds for d = {D}, p in 0.006 / 0.0035 and the sets' chunks only,"
                             f" not d = {d}, p = {p}, chunk = {chunk}")
        return tbl[k][int(chunk)]

    eval_seed.__wrapped__, eval_seed.rt_table = orig, tbl
    sd.eval_seed = eval_seed
    return tbl


def dry_confirm(spec):
    """RT_DRY_CONFIRM=a-b: see the module text. Refused unless the seeds are the dry ones."""
    if not is_dry():
        raise SystemExit("rt_surface: RT_DRY_CONFIRM is refused: the seeds in use are not the dry ones")
    a, b = (int(x) for x in spec.split("-"))
    from surface import sv_common as sc
    orig = sc.load_read
    sc.PS, sc.M_FAMILY = (0.006,), 20

    def load_read(d, p, comps, chunks=None, want=None, complete=None):
        return orig(d, p, comps, range(a, b), want, complete)
    sc.load_read = load_read


tbl = install()                                       # top level: the main process and every spawned worker


def banner():
    if tbl is None:
        print("rt_surface: patch OFF (RT_SEEDS unset): the pinned tree's eval_seed", file=sys.stderr, flush=True)
        return
    sha = {SETS[k]: hashlib.sha256(",".join(map(str, v)).encode()).hexdigest()[:16] for k, v in tbl.items()}
    print(f"rt_surface: patch ON for kind 11 (d = 13): {json.dumps(sha)} chunks {json.dumps({SETS[k]: len(v) for k, v in tbl.items()})} "
          f"first seeds {tbl[6000][0]} {tbl[3500][0]}; RT_SEEDS {os.environ['RT_SEEDS']} (dry: {is_dry()})",
          file=sys.stderr, flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--check-dry":
        if not is_dry():
            raise SystemExit("rt_surface: the seeds in use are not the dry ones")
        print("rt_surface: dry seeds", flush=True)
        return
    if os.environ.get("RT_DRY_CONFIRM") and not is_dry():
        raise SystemExit("rt_surface: RT_DRY_CONFIRM is refused: the seeds in use are not the dry ones")
    banner()
    if os.environ.get("RT_DRY_CONFIRM"):
        dry_confirm(os.environ["RT_DRY_CONFIRM"])
    mod = importlib.import_module(sys.argv[1])
    sys.argv = [sys.argv[1]] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()
