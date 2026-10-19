"""ARB at d = 13: fresh seeds (PLAN.md step 1). RETEST.md section 2's formula and collision rule with two new TAGs.

    seed(TAG, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62

R is the 32 randomness bytes of the re-test's beacon record (release retest/run/beacon.json, drand round 6,547,616),
reused as is, as ARB on BB72 did: the new TAGs alone make the seeds new. Sets (one TAG per p):
    arb-d13-0.006    TAG "convlines arb surface d13 p0.006\\n"    400 chunks of 10,000 shots, beacon indices from 0
    arb-d13-0.0035   TAG "convlines arb surface d13 p0.0035\\n"   3,000 chunks, beacon indices from 0
Collision rule (RETEST.md section 2, unchanged): a chunk takes the smallest index no earlier chunk took; an index whose
seed equals a seed already taken or a historical seed is dropped (recorded) and the next index is tried. Historical
seeds: retest/historical_seeds.txt (7,034) and seeds/hist_extra.txt (code/hist_scan.sh: every 19-digit integer in
[2^62, 2^63) in any seed, beacon or manifest file on every branch of the research repository and in every text file of the release; it
holds the re-test's own seeds, its dry-run and replay seeds, and ARB's BB72 seeds of the research repository's branch arb-exp). The
arithmetic seeds of the d = 13 comparison (surface_data.eval_seed, below 2^57) cannot collide with these. Standard
library only.

    python arb2_seeds.py make  RELEASE OUT.json      write the seeds (refuses to overwrite)
    python arb2_seeds.py check RELEASE OUT.json      recompute everything and compare (exit 1 on any difference)
A shim reads a set with chunk_seeds(name, path, release): every seed recomputed from the beacon, any difference raises."""
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS_DIR = os.path.join(os.path.dirname(HERE), "seeds")
TAGS = {"arb-d13-0.006": "convlines arb surface d13 p0.006\n", "arb-d13-0.0035": "convlines arb surface d13 p0.0035\n"}
CHUNKS = {"arb-d13-0.006": 400, "arb-d13-0.0035": 3000}
RULE = ("seed(TAG, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62; "
        "R = the 32 randomness bytes of the re-test's beacon (release retest/run/beacon.json); collision rule of "
        "RETEST.md section 2 against retest/historical_seeds.txt and seeds/hist_extra.txt")
OLD_TAGS = ("ising_compare study 2 T2", "study 3 T3", "study 4 T4", "ising_compare bb test BB72-3", "bb test BB72-2",
            "bb test G-4", "bb test4 G-4R G-4F",                 # RETEST.md section 2: TAGs used before the re-test
            "convlines arb bb72 depspam_fm p0.002\n", "convlines arb bb72 depspam_fm p0.003\n")   # ARB on BB72


def _rt_seeds(release):
    spec = importlib.util.spec_from_file_location("rt_seeds", os.path.join(release, "retest", "seeds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def historical(release, extra=None):
    extra = extra or os.path.join(SEEDS_DIR, "hist_extra.txt")
    base = os.path.join(release, "retest", "historical_seeds.txt")
    h0 = {int(x) for x in open(base).read().split()}
    h1 = {int(x) for x in open(extra).read().split()}
    return h0 | h1, {"retest/historical_seeds.txt": {"sha256": sha256_file(base), "seeds": len(h0)},
                     "seeds/hist_extra.txt": {"sha256": sha256_file(extra), "seeds": len(h1)}, "union": len(h0 | h1)}


def build(release):
    rt = _rt_seeds(release)
    beacon_path = os.path.join(release, "retest", "run", "beacon.json")
    b = json.load(open(beacon_path))
    r = rt.randomness(b)                                       # checks randomness = sha256(signature)
    hist, hinfo = historical(release)
    old = set(rt.TAGS["tags"].values()) | set(OLD_TAGS)
    if set(TAGS.values()) & old:
        raise SystemExit("a TAG was used before")
    taken_seed, sets = set(), {}
    for name in sorted(TAGS):
        tag = TAGS[name].encode()
        j, idx, sd, dropped = 0, [], [], []
        while len(idx) < CHUNKS[name]:
            x = rt.seed(r, tag, j)
            if x in hist or x in taken_seed:
                dropped.append(j)
            else:
                idx.append(j)
                sd.append(x)
                taken_seed.add(x)
            j += 1
        sets[name] = {"tag": TAGS[name], "j0": 0, "chunks": CHUNKS[name], "indices": idx, "seeds": sd,
                      "dropped": dropped}
    every = [x for s in sets.values() for x in s["seeds"]]
    assert len(set(every)) == len(every) and not set(every) & hist
    assert all(2 ** 62 <= x < 2 ** 63 for x in every)
    return {"what": "ARB at d = 13, fresh seeds (PLAN.md step 1; code/arb2_seeds.py)",
            "beacon_file": "retest/run/beacon.json", "beacon_sha256": sha256_file(beacon_path),
            "beacon_round": b["round"], "randomness": b["randomness"], "rule": RULE, "historical": hinfo,
            "old_tags_checked": sorted(old), "sets": sets}


def chunk_seeds(name, path, release):
    """The set's seeds for native chunks 0 .. chunks - 1, each recomputed from the beacon; raises on any difference."""
    rt = _rt_seeds(release)
    f = json.load(open(path))
    b = json.load(open(os.path.join(release, "retest", "run", "beacon.json")))
    r = rt.randomness(b)
    s = f["sets"][name]
    if (f["beacon_round"] != b["round"] or f["randomness"] != b["randomness"] or s["tag"] != TAGS[name]
            or len(s["seeds"]) != CHUNKS[name] or len(s["indices"]) != CHUNKS[name]):
        raise ValueError(f"{path}: not the seeds of {name}")
    if any(rt.seed(r, s["tag"].encode(), j) != x for j, x in zip(s["indices"], s["seeds"])):
        raise ValueError(f"{path}: a seed of {name} differs from the beacon's")
    return list(s["seeds"])


def main():
    cmd, release, out = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
    if cmd == "make":
        if os.path.exists(out):
            raise SystemExit(f"{out} exists: the seeds are written once")
        t = build(release)
        json.dump(t, open(out, "w"), indent=1)
    elif cmd == "check":
        t, f = build(release), json.load(open(out))
        bad = sorted(k for k in set(t) | set(f) if t.get(k) != f.get(k))
        print(json.dumps({"check": out, "equal": not bad, "differ": bad}))
        sys.exit(1 if bad else 0)
    else:
        raise SystemExit(__doc__)
    s = t["sets"]
    print(json.dumps({k: {"tag": v["tag"], "chunks": len(v["seeds"]), "dropped": v["dropped"], "first": v["seeds"][0],
                          "last": v["seeds"][-1]} for k, v in s.items()} | {"historical": t["historical"]}))


if __name__ == "__main__":
    main()
