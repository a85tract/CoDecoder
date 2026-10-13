"""Replay checks of the output-head re-test's shim (rt_a1.py; review finding 10), run by jobs/head_replay.sh on Alpine.
Modes (python head_replay.py MODE ...; `probe`, `kinds` and `relay` run inside the shim's patch, see the job script):
  compare-analysis ORIG NEW CODE READS_DIR OUT   check 1: the shim's identity mode reproduced A1's analysis (ORIG) in NEW
  probe OUT                       check 2: seeds seen by the main process and by spawned workers (module run by rt_a1.py)
  check-probe PROBE BEACON SEEDS TAGS OUT        the probe's seeds against a direct computation of the re-test's rule
  check-draw SETS_DIR BEACON SEEDS TAGS OUT      the drawn chunks: stored seed, rows, and the arrays against stim sampled with
                                                 the seed the rule gives
  check-resume LOG_KEPT LOG_RECOVERED LOG_OFF CHUNKS_BEFORE CHUNKS_AFTER OUT    the draw's resume path
  kinds OUT                       check 3: the patched seed function against the original, every other kind (module run by
                                  rt_a1.py, patch on or off)
  check-kinds ON OFF MANIFEST OUT the kinds' records: patch on equals the original off the test kinds; patch off equals
                                  A1's own seeds and A1's rows.csv rows
  check-relay DIR OUT             check 2: the Relay-BP fallback's per-shot seeds are the shot's index in its set
  guards LOG... / assemble        the shim's refusals; the three result files
The direct computation is written out here from the rule's text (RETEST.md section 2), not imported from retest/seeds.py."""
import csv
import hashlib
import json
import os
import sys
import time

A = "<cluster-scratch>/rev_a1"
OLD = {"surface": ("test", lambda j: 72_090_000 + j), "bb72": ("a1test", lambda j: 2_199_523_255_552 + j)}
NEW_SETS = {"surface": ("test", "head-surface", "surface_d9_p0.006"), "bb72": ("a1test", "head-bb72", "bb72_p0.003")}
SHA16 = {"surface": {"analysis.json": "10efcd2a", "comparisons.csv": "6dabba9e", "pair_curves.csv": "fd62a616"},
         "bb72": {"analysis.json": "55ef1f85", "comparisons.csv": "d315de5a", "pair_curves.csv": "6cb1c181"}}
VERDICTS = {"surface": {"R1": "worse", "R2": "worse", "R3": "inconclusive", "U1": "better", "U2": "inconclusive",
                        "W1": "worse", "W2": "inconclusive"},
            "bb72": {"R1": "worse", "R2": "worse", "U1": "inconclusive", "U2": "better", "W1": "worse"}}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def put(out, obj):
    json.dump(obj, open(out, "w"), indent=1, default=str)
    print(os.path.basename(out), "ok" if obj.get("ok") else "FAILED", flush=True)
    return 0 if obj.get("ok") else 1


def rule_seed(beacon, tag, j):
    """RETEST.md section 2: 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62."""
    r = bytes.fromhex(beacon["randomness"])
    assert hashlib.sha256(bytes.fromhex(beacon["signature"])).digest() == r
    return 2 ** 62 + int.from_bytes(hashlib.sha256(r + tag.encode() + j.to_bytes(2, "big")).digest()[:8], "big") % 2 ** 62


def expected(beacon_path, seeds_path, tags_path):
    """{code: [seed of native chunk 0, 1, ...]} and the beacon indices each takes, from the rule and the seeds file."""
    b, s, t = json.load(open(beacon_path)), json.load(open(seeds_path)), json.load(open(tags_path))
    out = {}
    for code, (kind, name, tagkey) in NEW_SETS.items():
        f = s["sets"][name]
        tag = t["tags"][tagkey]
        assert f["tag"] == tag and len(f["indices"]) == f["chunks"]
        out[code] = {"seeds": [rule_seed(b, tag, j) for j in f["indices"]], "indices": f["indices"], "j0": f["j0"],
                     "dropped": f["dropped"], "file_seeds": f["seeds"]}
    return out


# ---------------------------------------------------------------- check 1
def reads_meta(reads, out):
    import glob
    import numpy as np
    m = {os.path.basename(f): json.loads(str(np.load(f)["meta"])) for f in sorted(glob.glob(f"{reads}/*.npz"))
         if not f.endswith((".failed.npz", ".tmp.npz"))}
    json.dump(m, open(out, "w"), indent=1)
    return len(m)


def compare_analysis(orig, new, code, reads, out):
    files = {}
    for f in sorted(os.listdir(orig)):
        if f == "reads_meta.json":                    # written by analysis_main.txt's last line: the same line here
            reads_meta(reads, os.path.join(new, f))
        p, q = os.path.join(orig, f), os.path.join(new, f)
        e = {"equal_bytes": os.path.exists(q) and sha(p) == sha(q), "sha256_orig": sha(p)[:16]}
        if not e["equal_bytes"] and os.path.exists(q) and f.endswith(".json"):
            a, b = json.load(open(p)), json.load(open(q))
            e["json_equal"] = a == b
            if a != b:
                e["differing_keys"] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        files[f] = e
    same = {f for f, e in files.items() if e["equal_bytes"] or e.get("json_equal")}
    nets = {}
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(os.environ["RT"], "retest", "head", "check_taus.py"), code,
                        os.path.join(new, "pair_curves.csv"), os.path.join(os.environ["RT"], "retest", "head", "taus.csv"),
                        os.path.join(new, "taus_check.json")], capture_output=True, text=True)
    taus = json.load(open(os.path.join(new, "taus_check.json")))
    rows = list(csv.DictReader(open(os.path.join(new, "comparisons.csv"))))
    verd = {(r_["id"], r_["q"]): r_["verdict"] for r_ in rows}
    verdicts_ok = all(verd[(i, q)] == v for i, v in VERDICTS[code].items() for q in {r_["q"] for r_ in rows}) \
        and len(verd) == 2 * len(VERDICTS[code])
    sha16 = {f: (sha(os.path.join(new, f))[:8] == want) for f, want in SHA16[code].items()}
    return put(out, {"check": "1: a1.analyze through the shim, patch off, on the original reads", "code": code,
                     "files": files, "files_compared": len(files), "files_equal": len(same),
                     "taus": {k: v for k, v in taus.items() if k not in ("missing", "extra", "differ")},
                     "verdicts_equal_A1_RESULT": verdicts_ok, "verdicts": {f"{i}@{q}": v for (i, q), v in sorted(verd.items())},
                     "sha256_prefix_of_published_files_equal": sha16,
                     "ok": len(same) == len(files) and taus["ok"] and verdicts_ok and all(sha16.values())})


# ---------------------------------------------------------------- check 2
def _probe_one(_):
    from a1 import sets
    if _ is not None:
        time.sleep(0.5)                              # a worker task takes a while, so that every worker gets tasks
    return {"pid": os.getpid(), "patched": hasattr(sets, "rt_original_seed_of"), "sets_dir": sets.SETS,
            "surface_test": [sets.seed_of("surface", "test", j) for j in range(100)],
            "bb72_a1test": [sets.seed_of("bb72", "a1test", j) for j in range(50)]}


def probe(out):
    import multiprocessing as mp
    res = {"parent": _probe_one(None)}
    with mp.get_context("spawn").Pool(3) as pool:
        res["workers"] = pool.map(_probe_one, range(9), chunksize=1)
    json.dump(res, open(out, "w"), indent=1)
    print("PROBE", out, len({w["pid"] for w in res["workers"]}), "worker processes", flush=True)


def check_probe(probe_path, beacon, seeds, tags, out):
    p, exp = json.load(open(probe_path)), expected(beacon, seeds, tags)
    procs = [p["parent"]] + p["workers"]
    per = [{"pid": x["pid"], "patched": x["patched"],
            "surface_test_equal": x["surface_test"] == exp["surface"]["seeds"],
            "bb72_a1test_equal": x["bb72_a1test"] == exp["bb72"]["seeds"],
            "old_seeds_absent": not (set(x["surface_test"]) & {OLD["surface"][1](j) for j in range(100)})
            and not (set(x["bb72_a1test"]) & {OLD["bb72"][1](j) for j in range(50)})} for x in procs]
    workers = {x["pid"] for x in p["workers"]}
    ok = all(all(v for k, v in e.items() if k != "pid") for e in per) and len(workers) >= 2 and p["parent"]["pid"] not in workers
    return put(out, {"check": "2: the main process and spawned workers draw the intended seeds (shim launcher, dry seeds)",
                     "processes": per, "worker_pids": sorted(workers), "indices": {c: [min(e["indices"]), max(e["indices"])]
                                                                                    for c, e in exp.items()},
                     "dropped": {c: e["dropped"] for c, e in exp.items()}, "ok": ok})


def check_draw(sets_dir, beacon, seeds, tags, out, chunks=(0, 1)):
    import numpy as np
    from a1 import sets as a1sets
    exp, res, ok = expected(beacon, seeds, tags), [], True
    rows = {r["file"]: r for r in csv.DictReader(open(os.path.join(sets_dir, "rows.csv")))}
    for code, (kind, _, _) in NEW_SETS.items():
        circ = a1sets.circuit(code)
        for j in chunks:
            f = a1sets.name(code, kind, j)
            z = np.load(os.path.join(sets_dir, f))
            want = exp[code]["seeds"][j]
            det, obs = circ.compile_detector_sampler(seed=want).sample(10_000, separate_observables=True)
            e = {"file": f, "beacon_index": exp[code]["indices"][j], "expected_seed": want, "stored_seed": int(z["seed"]),
                 "row_seed": int(rows[f]["seed"]), "row_sha256_equal": rows[f]["sha256"] == sha(os.path.join(sets_dir, f)),
                 "det_equal_direct_sampling": bool((np.packbits(det.astype(np.uint8), axis=1) == z["det"]).all()),
                 "obs_equal_direct_sampling": bool((np.packbits(obs.astype(np.uint8), axis=1) == z["obs"]).all()),
                 "not_the_original_seed": int(z["seed"]) != OLD[code][1](j), "freeze": str(z["freeze"])}
            e["ok"] = (e["expected_seed"] == e["stored_seed"] == e["row_seed"] and e["row_sha256_equal"]
                       and e["det_equal_direct_sampling"] and e["obs_equal_direct_sampling"] and e["not_the_original_seed"])
            ok &= e["ok"]
            res.append(e)
    return put(out, {"check": "2: the set-generation main process draws the intended seeds", "chunks": res, "ok": ok})


def check_resume(kept, recovered, off, before, after, out):
    k, r, o = (open(x).read() for x in (kept, recovered, off))
    res = {"rerun_keeps_every_chunk": k.count("KEPT ") == 4 and "DRAWN " not in k,
           "rows_lost_recovered_from_bytes": r.count("RECOVERED ") == 4 and "DRAWN " not in r,
           "patch_off_refuses_the_chunks": "is not chunk" in o and "'seed'" in o,
           "chunk_sha256_unchanged": open(before).read() == open(after).read() and len(open(before).read().split()) == 8,
           "patch_off_message": [ln for ln in o.splitlines() if "is not chunk" in ln][:2]}
    res["ok"] = all(v for kk, v in res.items() if kk != "patch_off_message")
    return put(out, {"check": "2: the draw's resume path (KEPT, RECOVERED) uses the intended seeds", **res})


def check_relay(d, out):
    seeds, workers, errors = [], {}, []
    for f in sorted(os.listdir(d)):
        if f.startswith("seeds_"):
            lines = [ln.split() for ln in open(os.path.join(d, f)) if ln.strip()]
            workers[f[6:-4]] = len(lines)
            seeds += [int(s) for _, s in lines]
    info = json.load(open(os.path.join(d, "relay_result.json")))
    want = info["expected_seeds"]
    wk = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.startswith("worker_")]
    ok = (sorted(seeds) == sorted(want) and len(workers) >= 2 and info["answers_equal_across_layouts"]
          and all(w["patched"] and w["surface_test_0"] == info["surface_test_0"] for w in wk) and len(wk) >= 2)
    return put(out, {"check": "2: the Relay-BP fallback's per-shot decoder seeds are the shot's index in its set (frozen)",
                     "shots": len(want) // 2, "decodes_per_shot": 2, "native_chunks": [0, 1], "recorded_seeds": len(seeds), "workers_that_decoded": workers,
                     "seeds_equal_shot_index_10000c_plus_r": sorted(seeds) == sorted(want),
                     "beacon_indices_in_play": info["beacon_indices"], "answers_equal_across_layouts": info["answers_equal_across_layouts"],
                     "processes_with_the_patch": [{"pid": w["pid"], "patched": w["patched"]} for w in wk], "ok": ok})


def relay(d):
    import numpy as np
    from a1 import fallback, sets
    os.makedirs(d, exist_ok=True)
    det, obs, digest = sets.load("bb72", "a1test", [0, 1])
    take = np.r_[0:100, 10_000:10_100]
    idx = sets.index([0, 1])[take]
    a = fallback.relay(det[take], idx, 4, part=50)
    b = fallback.relay(det[take], idx, 2, part=200)
    same = all(bool((x == y).all()) for x, y in zip(a[:1], b[:1]))
    json.dump({"expected_seeds": [int(x) for x in idx] * 2, "answers_equal_across_layouts": same,
               "beacon_indices": json.load(open(os.environ["RT_SEEDS"]))["sets"]["head-bb72"]["indices"][:2],
               "surface_test_0": sets.seed_of("surface", "test", 0)}, open(os.path.join(d, "relay_result.json"), "w"))
    print("RELAY replay", len(idx), "shots, answers equal across layouts", same, flush=True)


def install_recorder(d):
    """Replay only: records every seed a RelayDecoderF32 is built with, per process, and notes the shim's patch."""
    import relay_bp
    from a1 import sets
    os.makedirs(d, exist_ok=True)
    orig = relay_bp.RelayDecoderF32

    def rec(*a, **k):
        with open(os.path.join(d, f"seeds_{os.getpid()}.txt"), "a") as fh:
            fh.write(f"{os.getpid()} {k['seed']}\n")
        return orig(*a, **k)
    relay_bp.RelayDecoderF32 = rec
    json.dump({"pid": os.getpid(), "patched": hasattr(sets, "rt_original_seed_of"),
               "surface_test_0": sets.seed_of("surface", "test", 0)}, open(os.path.join(d, f"worker_{os.getpid()}.json"), "w"))


# ---------------------------------------------------------------- check 3
def kinds(out):
    from a1 import seeds as a1seeds
    from a1 import sets
    from bb import shots
    patched = hasattr(sets, "rt_original_seed_of")
    rec = {}
    for code, kl in (("surface", list(a1seeds.SURFACE_SETS)), ("bb72", list(shots.SPLIT) + ["test4"])):
        for kind in kl:
            for j in list(range(100)) + [9_999]:
                try:
                    v = sets.seed_of(code, kind, j)
                except Exception as e:                       # (test4 and test: beacon-bound splits of the BB tree)
                    v = f"{type(e).__name__}"
                rec[f"{code}|{kind}|{j}"] = v
    # the names the shim must leave alone
    other = {"a1.seeds.surface_seed": a1seeds.surface_seed("test", 9, 5), "bb.shots.seed_of(a1test)": shots.seed_of("a1test", "bb72", 0.003, 5),
             "bb.shots.seed_of(a1cal)": shots.seed_of("a1cal", "bb72", 0.003, 5)}
    json.dump({"patched": patched, "seeds": rec, "other": other, "sets_dir": sets.SETS}, open(out, "w"))
    print("KINDS", len(rec), "seeds, patched", patched, flush=True)


def check_kinds(on, off, manifest, out, beacon, seeds, tags):
    a, b = json.load(open(on)), json.load(open(off))
    exp = expected(beacon, seeds, tags)
    new = {("surface", "test"): exp["surface"]["seeds"], ("bb72", "a1test"): exp["bb72"]["seeds"]}
    same, differ, wrong_new, wrong_old = 0, [], [], []
    for key, v in a["seeds"].items():
        code, kind, j = key.split("|")
        j = int(j)
        if (code, kind) in new:
            want = new[(code, kind)][j] if j < len(new[(code, kind)]) else None
            if want is None:
                wrong_new.append(key) if v != "ValueError" else None          # beyond the set: refused, not an old seed
            elif v != want or v == b["seeds"][key]:
                wrong_new.append(key)
        elif v == b["seeds"][key]:
            same += 1
        else:
            differ.append(key)
    # patch off: the original function; A1's formulas and A1's rows.csv rows
    rows = list(csv.DictReader(open(manifest)))
    man_bad = [r["file"] for r in rows if b["seeds"].get(f"{r['code']}|{r['kind']}|{r['chunk']}") != int(r["seed"])]
    for code, (kind, f) in OLD.items():
        wrong_old += [f"{code}|{kind}|{j}" for j in range(100 if code == "surface" else 50) if b["seeds"][f"{code}|{kind}|{j}"] != f(j)]
    other_same = a["other"] == b["other"]
    ok = (a["patched"] and not b["patched"] and not differ and not wrong_new and not man_bad and not wrong_old and other_same)
    return put(out, {"check": "3: every other seed kind is unchanged; the test kinds with the patch off are A1's seeds",
                     "patch_on_flag": a["patched"], "patch_off_flag": b["patched"], "values_compared": len(a["seeds"]),
                     "other_kinds_equal_original": same, "other_kinds_differ": differ, "test_kinds_wrong_with_patch_on": wrong_new,
                     "test_kinds_off_not_A1": wrong_old, "rows_checked": len(rows), "rows_unequal": man_bad,
                     "bb_and_a1_seed_functions_equal": other_same, "sets_dir_on": a["sets_dir"], "sets_dir_off": b["sets_dir"], "ok": ok})


def guards(logs, out):
    res = {}
    for spec in logs:
        name, code, path = spec.split(":", 2)
        txt = open(path).read()
        res[name] = {"exit_code": int(code), "refused_by_shim": "rt_a1:" in txt, "message": [ln for ln in txt.splitlines() if "rt_a1:" in ln][:1]}
    want_refused = [n for n in res if not n.startswith("control")]
    ok = all(res[n]["refused_by_shim"] and res[n]["exit_code"] != 0 for n in want_refused) and \
        all(not res[n]["refused_by_shim"] for n in res if n.startswith("control"))
    return put(out, {"check": "the shim's guards", "runs": res, "ok": ok})


def assemble(work, rep):
    """The three result files of replay/head/ from the parts in WORK."""
    part = lambda n: json.load(open(os.path.join(work, n)))
    r1 = {"description": "check 1: A1's analysis reproduced through the shim with the patch off (review finding 10)",
          "surface": part("r1_surface.json"), "bb72": part("r1_bb72.json")}
    r1["ok"] = r1["surface"]["ok"] and r1["bb72"]["ok"]
    r2 = {"description": "check 2: spawned workers, the main process and the resume path use the intended seeds; Relay-BP "
                         "keeps its per-shot seeds; the shim's guards (dry seeds)",
          "probe": part("r2a.json"), "draw": part("r2b.json"), "resume": part("r2c.json"), "relay": part("r2d.json"),
          "guards": part("r2f.json")}
    r2["ok"] = all(r2[k]["ok"] for k in ("probe", "draw", "resume", "relay", "guards"))
    r3 = part("r3.json")
    for name, r in (("replay1_analysis.json", r1), ("replay2_seeds.json", r2), ("replay3_kinds.json", r3)):
        json.dump(r, open(os.path.join(rep, name), "w"), indent=1, default=str)
        print(name, "ok" if r["ok"] else "FAILED", flush=True)
    return 0 if r1["ok"] and r2["ok"] and r3["ok"] else 1


def main():
    mode, a = sys.argv[1], sys.argv[2:]
    if mode == "assemble":
        sys.exit(assemble(*a))
    if mode == "compare-analysis":
        sys.exit(compare_analysis(*a))
    if mode == "probe":
        return probe(*a)
    if mode == "kinds":
        return kinds(*a)
    if mode == "relay":
        return relay(*a)
    fn = {"check-probe": check_probe, "check-draw": check_draw, "check-resume": check_resume, "check-kinds": check_kinds,
          "check-relay": check_relay, "guards": guards}[mode]
    sys.exit(fn(*a) if mode != "guards" else fn(a[:-1], a[-1]))


if __name__ == "__main__":
    main()
