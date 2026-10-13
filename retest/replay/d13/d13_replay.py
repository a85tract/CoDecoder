"""Replay checks of the d = 13 shim (RETEST.md section d = 13), CPU only, run on Alpine in the evaluation's venv:
  python d13_replay.py compare ORIG_DIR NEW_DIR OUT.json
      replay 1: the analysis outputs the shim's `sv_route confirm` / `report` wrote into NEW_DIR from the original reads
      (confirm_d13.json, SURFACE_RESULT_retest.md and .json) against the original e8's in ORIG_DIR (confirm_d13.json,
      SURFACE_RESULT_draft.md and .json); JSON equality with the paths that differ listed.
  python rt_surface.py d13_replay seeds OUT.json
      replays 2 and 3 (PYTHONPATH must list this directory so spawned workers can import this module): patch ON when
      RT_SEEDS is set (the dry seeds), OFF otherwise. The expected seeds are computed here from the seeds file and the
      rule, not through the shim. ON: kind 11 in the main process, in surface_data's spawned sampling pool
      (stream_shots), in sv_read's spawned M workers (_m_task) and in the resume path's expression (sv_read line 211)
      draws the dry seeds, seen as the sha256 digests of the sampled shots; every other kind equals the pinned function;
      kind 11 outside the sets raises. OFF: the pinned function object, kind 11 = its formula.
  python d13_replay.py seeds-control OUT.json main-only|none      (no shim; RT_SEEDS set, PYTHONPATH as above)
      negative controls: the same probe with the patch installed in the main process only (as a shim that patched inside
      main() would), or not at all. The probe must fail exactly where the workers or the main process draw other seeds.
  python rt_surface.py d13_replay verify OUTDIR OUT.json
      the chunk digests a read stored (obs_*.npz) against a direct sample of the seeds of RT_SEEDS (a few chunks per part)."""
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys

SETS = {0.006: "d13-0.006", 0.0035: "d13-0.0035"}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(path)
    return out


def compare(orig, new, out):
    res = {}
    pairs = {"confirm": ("confirm_d13.json", "confirm_d13.json"),
             "report_md": ("SURFACE_RESULT_draft.md", "SURFACE_RESULT_retest.md"),
             "report_json": ("SURFACE_RESULT_draft.md.json", "SURFACE_RESULT_retest.md.json")}
    for k, (fo, fn) in pairs.items():
        po, pn = os.path.join(orig, fo), os.path.join(new, fn)
        r = {"orig": po, "new": pn, "sha256_orig": sha(po), "sha256_new": sha(pn)}
        r["bytes_equal"] = r["sha256_orig"] == r["sha256_new"]
        if fo.endswith(".json") or fo.endswith(".md.json"):
            a, b = json.load(open(po)), json.load(open(pn))
            r["json_differs_at"] = diff(a, b)
        res[k] = r
    c = res["confirm"]
    res["pass"] = (not set(c["json_differs_at"]) - {"/secs"}) and res["report_md"]["bytes_equal"] and \
        not res["report_json"]["json_differs_at"]
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"pass": res["pass"], "confirm_differs_at": c["json_differs_at"],
                      "report_md_bytes_equal": res["report_md"]["bytes_equal"],
                      "report_json_differs_at": res["report_json"]["json_differs_at"]}), flush=True)
    return 0 if res["pass"] else 1


def formula(d, kind, p, chunk):
    """The pinned eval_seed's formula (surface_data.py:36-38), written out again."""
    return 2 ** 56 + 3 * 10 ** 16 + d * 10 ** 15 + kind * 10 ** 13 + int(round(p * 1e6)) * 10 ** 7 + chunk


def expected_seeds():
    """{p: [seed of native chunk j]}: the seeds file's entries, each recomputed here from the beacon record and
    tags.json by the rule (2^62 + first 8 bytes of sha256(R || TAG || index as 2 bytes) mod 2^62)."""
    f, b = json.load(open(os.environ["RT_SEEDS"])), json.load(open(os.environ["RT_BEACON"]))
    tags = json.load(open(os.path.join(os.environ["RT"], "retest", "tags.json")))
    r = bytes.fromhex(b["randomness"])
    out = {}
    for p, name in SETS.items():
        s, tag = f["sets"][name], tags["tags"][tags["sets"][name]["tag"]].encode()
        calc = [2 ** 62 + int.from_bytes(hashlib.sha256(r + tag + j.to_bytes(2, "big")).digest()[:8], "big") % 2 ** 62
                for j in s["indices"]]
        assert calc == s["seeds"] and f["beacon_round"] == b["round"], name
        out[p] = calc
    return out


def task_m(a):
    """sv_read._m_task in a spawned worker: (worker pid, p, chunk, digest of the shots it sampled from its seed)."""
    from surface import sv_read
    r = sv_read._m_task(a)
    return os.getpid(), a[0], a[2], r[1]


def seeds(out, control=""):
    from surface import surface_data as sd
    from surface import sv_common as sc
    on = bool(os.environ.get("RT_SEEDS"))
    res = {"mode": ("control " + control) if control else ("on" if on else "off"), "pid": os.getpid(), "checks": {}}
    ck = res["checks"]
    pin = getattr(sd.eval_seed, "__wrapped__", sd.eval_seed)
    exp = expected_seeds() if on else None

    def dig(p, seed, n=sc.CHUNK):
        det, obs = sd.sample(sc.D, p, n, seed)
        return sc.digest(det, obs)
    # 1. the function object and kind 11 in the main process
    patched = getattr(sd.eval_seed, "rt_table", None) is not None
    ck["function"] = {"patched": patched, "expect_patched": on, "pass": patched == on}
    pj = {0.006: [0, 1, 2, 50, 99], 0.0035: [0, 1, 2, 1000, 2999]}
    bad = []
    for p, js in pj.items():
        for j in js:
            want = exp[p][j] if on else formula(13, 11, p, j)
            if sd.eval_seed(13, 11, p, j) != want or (on and want == formula(13, 11, p, j)):
                bad.append((p, j))
    ck["main_kind11"] = {"chunks_checked": sum(len(v) for v in pj.values()), "bad": bad, "pass": not bad}
    if on:                       # every chunk of both sets
        bad = [(p, j) for p in SETS for j in range(len(exp[p])) if sd.eval_seed(13, 11, p, j) != exp[p][j]]
        ck["main_kind11_all_chunks"] = {"chunks_checked": sum(len(v) for v in exp.values()), "bad": bad[:5],
                                        "pass": not bad}
        ck["seeds_distinct_from_original"] = {"pass": all(exp[p][j] != formula(13, 11, p, j)
                                                          for p in SETS for j in range(len(exp[p])))}
    else:
        bad = [(p, j) for p, n in ((0.006, 100), (0.0035, 3000)) for j in range(n)
               if sd.eval_seed(13, 11, p, j) != formula(13, 11, p, j)]
        ck["off_kind11_formula_all_chunks"] = {"bad": bad[:5], "pass": not bad}
    # 2. every other kind, replay 3: the patched function equals the pinned one and the formula
    n, bad = 0, []
    for kind in [k for k in range(100) if k != 11]:
        for d in (3, 13, 17, 29):
            for p in (0.000001, 0.0035, 0.006, 0.007, 0.009999):
                for ch in (0, 1, 99, 2999, 10 ** 7 - 1):
                    n += 1
                    if not (sd.eval_seed(d, kind, p, ch) == pin(d, kind, p, ch) == formula(d, kind, p, ch)):
                        bad.append((d, kind, p, ch))
    for kind in (1, 10, 12, 13):                 # the evaluation's other kinds at d = 13 and the re-test's p
        for p in (0.006, 0.0035):
            for ch in range(0, 3000, 7):
                n += 1
                if sd.eval_seed(13, kind, p, ch) != pin(13, kind, p, ch):
                    bad.append((13, kind, p, ch))
    ck["other_kinds_unchanged"] = {"values_compared": n, "bad": bad[:5], "pass": not bad}
    # kind 11 outside the sets raises (ON); the pinned function's own checks still run
    if on:
        refused = []
        for a in ((17, 11, 0.006, 0), (13, 11, 0.007, 0), (13, 11, 0.006, 100), (13, 11, 0.0035, 3000)):
            try:
                sd.eval_seed(*a)
                refused.append((a, "not refused"))
            except ValueError:
                pass
        try:
            sd.eval_seed(13, 11, 0.012, 0)           # the pinned assertion (p < 0.01)
            refused.append(((13, 11, 0.012, 0), "pinned check skipped"))
        except AssertionError:
            pass
        ck["kind11_outside_sets_refused"] = {"bad": refused, "pass": not refused}
    # 3. spawned sampling pool (surface_data.stream_shots) and the main process's _eval_chunk
    chunks = [0, 1, 2]
    want = lambda p, j: exp[p][j] if on else formula(13, 11, p, j)
    bad, rows = [], []
    for p in SETS:
        for j, det, obs in sd.stream_shots(sc.D, p, chunks, n=sc.CHUNK, kind=11, procs=2):
            got, ref = sc.digest(det, obs), dig(p, want(p, j))
            rows.append({"p": p, "chunk": j, "digest": got})
            if got != ref:
                bad.append((p, j))
    ck["spawned_sampling_pool"] = {"rows": rows, "bad": bad, "pass": not bad}
    d0 = sd._eval_chunk((sc.D, 11, 0.006, 0, 200))
    ck["main_eval_chunk"] = {"pass": sc.digest(*d0) == sc.digest(*sd.sample(sc.D, 0.006, 200, want(0.006, 0)))}
    if on:                                       # the pool really draws the dry seed, not the original
        orig_dig = dig(0.006, formula(13, 11, 0.006, 0))
        ck["pool_differs_from_original_seed"] = {"pass": rows[0]["digest"] != orig_dig}
    # 4. the resume path's draw (sv_read.prime_queues, line 211) and M's workers (sv_read._m_task, spawned)
    src = open(os.path.join(os.path.dirname(sc.__file__), "sv_read.py")).read().split("\n")
    sites = {f"sv_read.py:{i + 1}": ln.strip() for i, ln in enumerate(src) if re.search(r"\beval_seed\(", ln)}
    ck["resume_and_worker_call_sites"] = {"sites": sites, "pass": len(sites) == 2 and
                                          all("sc.sdata.eval_seed(" in v for v in sites.values())}
    j = 1
    det, obs = sc.sdata.sample(sc.D, 0.006, sc.CHUNK, sc.sdata.eval_seed(sc.D, 11, 0.006, int(j)))   # line 211's expression
    ck["resume_expression"] = {"pass": sc.digest(det, obs) == dig(0.006, want(0.006, j))}
    tasks = [(0.006, 11, 0, False, False, False), (0.006, 11, 1, False, False, False),
             (0.0035, 11, 0, False, False, False)]
    with mp.get_context("spawn").Pool(2) as pool:
        got = pool.map(task_m, tasks, chunksize=1)
    bad = [(p, j) for pid, p, j, dg in got if dg != dig(p, want(p, j))]
    ck["spawned_m_workers"] = {"rows": [{"pid": pid, "p": p, "chunk": j, "digest": dg} for pid, p, j, dg in got],
                               "worker_pids_differ_from_main": all(pid != os.getpid() for pid, *_ in got),
                               "bad": bad, "pass": not bad and all(pid != os.getpid() for pid, *_ in got)}
    res["all_pass"] = all(v["pass"] for v in ck.values())
    failed = sorted(k for k, v in ck.items() if not v["pass"])
    if control:        # the probe must see exactly the draws that differ
        res["failed"] = failed
        must = {"main-only": {"spawned_sampling_pool", "pool_differs_from_original_seed", "spawned_m_workers"},
                "none": {"function", "main_kind11", "main_kind11_all_chunks", "kind11_outside_sets_refused",
                         "spawned_sampling_pool", "main_eval_chunk", "pool_differs_from_original_seed",
                         "resume_expression", "spawned_m_workers"}}[control]
        res["control_expected_failures"] = sorted(must)
        res["control_ok"] = set(failed) == must
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"mode": res["mode"], "all_pass": res["all_pass"], "failed": failed,
                      **({"control_ok": res["control_ok"]} if control else {})}), flush=True)
    return (0 if res["control_ok"] else 1) if control else (0 if res["all_pass"] else 1)


def seeds_control(out, how):
    """Negative control (module text). main-only: the shim file is loaded here, so its top-level install() patches this
    process only; the spawned workers re-run __main__ (this file), which does not import the shim."""
    if how == "main-only":
        import importlib.util
        spec = importlib.util.spec_from_file_location("rt_shim_main_only",
                                                      os.path.join(os.environ["RT"], "retest", "shim", "rt_surface.py"))
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return seeds(out, how)


def verify(outdir, out, per_part=5):
    """Chunk digests stored in OUTDIR/p<p>/obs_*.npz against a direct sample of the seeds of RT_SEEDS."""
    from surface import surface_data as sd
    from surface import sv_common as sc
    exp = expected_seeds()
    rows, bad = [], []
    for p in SETS:
        base = os.path.join(outdir, sc.p_tag(p))
        if not os.path.isdir(base):
            continue
        for f in sorted(x for x in os.listdir(base) if x.startswith("obs_") and x.endswith(".npz")):
            c0, c1 = (int(x) for x in f[4:-4].split("-"))
            meta, _ = sc.load_npz(os.path.join(base, f))
            pick = sorted({c0 + i for i in list(range(per_part)) + [(c1 - c0) // 2] + [c1 - c0 - 1 - i for i in range(per_part)]
                           if 0 <= i < c1 - c0})
            for j in pick:
                det, obs = sd.sample(sc.D, p, sc.CHUNK, exp[p][j])
                ok = sc.digest(det, obs) == meta["digests"][j - c0]
                rows.append({"p": p, "file": f, "chunk": j, "equal": ok})
                if not ok:
                    bad.append((p, f, j))
    res = {"outdir": outdir, "chunks_checked": len(rows), "bad": bad, "pass": bool(rows) and not bad, "rows": rows}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"chunks_checked": len(rows), "bad": bad, "pass": res["pass"]}), flush=True)
    return 0 if res["pass"] else 1


def main():
    cmd, a = sys.argv[1], sys.argv[2:]
    rc = {"compare": lambda: compare(*a), "seeds": lambda: seeds(*a), "seeds-control": lambda: seeds_control(*a),
          "verify": lambda: verify(*a)}[cmd]()
    sys.exit(rc)


if __name__ == "__main__":
    main()
