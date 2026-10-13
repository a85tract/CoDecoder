"""The re-test's seeds (RETEST.md section 2). Stage 3's formula (bb72/ising_compare/bb/shots.py test_seed):
    seed(TAG, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62
with R the 32 randomness bytes of the beacon record (retest/beacon.json), one TAG per code and error rate, and the test
sets of tags.json. Collision rule (frozen): the sets of one TAG take their chunks in order of j0; a set's next chunk
takes the smallest beacon index j >= j0 that no earlier chunk took; if seed(TAG, j) equals a seed already taken or a
historical seed (historical_seeds.txt: every 19-digit integer of every manifest, seed and beacon file of the research
hub's branches and the release, and of the Alpine records listed in RETEST.md), j is dropped (recorded) and the next
index is tried. Every seed from a TAG lies in [2^62, 2^63); D1's, d = 13's and A1's arithmetic seeds lie below 2^57.
Standard library only, so every shim and every host can import it.
    python retest/seeds.py BEACON OUT              every set's beacon indices and seeds (OUT, JSON)
    python retest/seeds.py --check-stage3 BEACON ROWS   recomputes stage 3's published seeds (the formula's check)
A shim reads the seeds of a set with chunk_seeds(name): the file RT_SEEDS, every entry checked against RT_BEACON."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = json.load(open(os.path.join(HERE, "tags.json")))
HISTORICAL = os.path.join(HERE, "historical_seeds.txt")
FLOOR = 2 ** 62


def seed(randomness, tag, j):
    """randomness: 32 bytes; tag: bytes; j: 0 <= j < 2^16."""
    if len(randomness) != 32 or not 0 <= j < 2 ** 16:
        raise ValueError((len(randomness), j))
    h = hashlib.sha256(randomness + tag + j.to_bytes(2, "big")).digest()
    return FLOOR + int.from_bytes(h[:8], "big") % FLOOR


def randomness(beacon):
    """The 32 randomness bytes of a beacon record (a path or a loaded dict)."""
    b = json.load(open(beacon)) if isinstance(beacon, str) else beacon
    r = bytes.fromhex(b["randomness"])
    if len(r) != 32 or hashlib.sha256(bytes.fromhex(b["signature"])).digest() != r:
        raise ValueError("the beacon's randomness is not sha256(signature)")
    return r


def tag_of(name):
    return TAGS["tags"][TAGS["sets"][name]["tag"]].encode()


def historical():
    return {int(x) for x in open(HISTORICAL).read().split()}


def assign(beacon, hist):
    """Every set's beacon indices and seeds under the collision rule."""
    r = randomness(beacon)
    taken_seed, taken_idx, out = set(), {}, {}
    for name, s in sorted(TAGS["sets"].items(), key=lambda kv: (kv[1]["tag"], kv[1]["j0"])):
        tag = tag_of(name)
        taken = taken_idx.setdefault(tag, set())
        j, idx, sd, dropped = s["j0"], [], [], []
        while len(idx) < s["chunks"]:
            if j not in taken:
                taken.add(j)
                x = seed(r, tag, j)
                if x in hist or x in taken_seed:
                    dropped.append(j)
                else:
                    idx.append(j)
                    sd.append(x)
                    taken_seed.add(x)
            j += 1
        out[name] = {"tag": tag.decode(), "j0": s["j0"], "chunks": s["chunks"], "indices": idx, "seeds": sd,
                     "dropped": dropped}
    return {n: out[n] for n in TAGS["sets"]}


def chunk_seeds(name, path=None, beacon=None):
    """Seeds of the set's native chunks 0 .. chunks - 1, from the seeds file (RT_SEEDS); each recomputed from the beacon
    record (RT_BEACON)."""
    path, beacon = path or os.environ["RT_SEEDS"], beacon or os.environ["RT_BEACON"]
    f, b = json.load(open(path)), json.load(open(beacon))
    s = f["sets"][name]
    r = randomness(b)
    if f["beacon_round"] != b["round"] or s["tag"].encode() != tag_of(name) or len(s["seeds"]) != s["chunks"]:
        raise ValueError(f"{path}: not the seeds of {name} under {beacon}")
    if any(seed(r, s["tag"].encode(), j) != x for j, x in zip(s["indices"], s["seeds"])):
        raise ValueError(f"{path}: a seed of {name} differs from the beacon's")
    return list(s["seeds"])


def main():
    if sys.argv[1] == "--check-stage3":                    # stage 3's TAGs on its beacon against its manifest rows
        import csv
        b, rows = json.load(open(sys.argv[2])), list(csv.DictReader(open(sys.argv[3])))
        tags = {k: v.encode() for k, v in b["tags"].items()}
        bad = [r["file"] for r in rows if seed(randomness(b), tags[f"{r['code']}_p{r['p']}"], int(r["chunk"]))
               != int(r["seed"])]
        print(f"stage 3 seeds recomputed: {len(rows) - len(bad)} of {len(rows)} equal", flush=True)
        sys.exit(1 if bad else 0)
    if sys.argv[1] == "--verify":                          # the seeds file against the rule, recomputed whole
        b, f, hist = json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), historical()
        with open(HISTORICAL, "rb") as fh:
            hsha = hashlib.sha256(fh.read()).hexdigest()
        want = {"beacon_round": b["round"], "randomness": b["randomness"], "rule": TAGS["rule"],
                "historical": {"file": "retest/historical_seeds.txt", "sha256": hsha, "seeds": len(hist)},
                "sets": assign(b, hist)}
        bad = sorted(k for k in set(want) | set(f) if want.get(k) != f.get(k))
        print(json.dumps({"verify": sys.argv[3], "beacon_round": b["round"], "equal": not bad, "differ": bad}), flush=True)
        sys.exit(1 if bad else 0)
    if os.path.exists(sys.argv[2]):
        raise SystemExit(f"{sys.argv[2]} exists: the seeds are written once")
    b, hist = json.load(open(sys.argv[1])), historical()
    t = assign(b, hist)
    every = [x for v in t.values() for x in v["seeds"]]
    assert len(set(every)) == len(every) and not set(every) & hist
    with open(HISTORICAL, "rb") as fh:
        hsha = hashlib.sha256(fh.read()).hexdigest()
    json.dump({"beacon_round": b["round"], "randomness": b["randomness"], "rule": TAGS["rule"],
               "historical": {"file": "retest/historical_seeds.txt", "sha256": hsha, "seeds": len(hist)},
               "sets": t}, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({"chunks": len(every), "dropped": {k: v["dropped"] for k, v in t.items() if v["dropped"]},
                      "historical": len(hist)}), flush=True)


if __name__ == "__main__":
    main()
