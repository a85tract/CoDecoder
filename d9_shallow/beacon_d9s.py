"""D9-SHALLOW's beacon record (PLAN.md section 3): SURF-v2's fetch and checks unchanged (surface_v2/convlines/beacon.py:
drand mainnet, four endpoints, two agreeing answers fetched after the round's time, randomness R = SHA-256(signature),
32 bytes), with this test's two tags and seed_tag:

  test    b"convlines d9-shallow test d9\\n",   j = 0..99: 100 chunks of 10,000 shots (1,000,000 test shots)
  timing  b"convlines d9-shallow timing d9\\n", j = 0..19: 20 chunks of 10,000 shots (200,000 timing shots)
  seed(set, j) = convlines.beacon.seed_tag(R, tag, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || tag || j as
  2 bytes big-endian)) mod 2^62

T is the time of the earliest record in GitHub's activity list of the repository whose `after` is the freeze commit F;
the round is the first drand round at least 600 s after T. The record keeps the whole activity list, and every check
derives T, the round and every seed from it again.
    python d9_shallow/beacon_d9s.py fetch --commit F --activity activity.json --push-source "$(cat push_source.txt)" \\
        --out d9_shallow/testsets/beacon_d9s.json
    python d9_shallow/beacon_d9s.py check --beacon d9_shallow/testsets/beacon_d9s.json [--commit F]"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "surface_v2"))
from convlines import beacon as B  # noqa: E402  (SURF-v2's, unchanged)

TAGS = {"test": "convlines d9-shallow test d9\n", "timing": "convlines d9-shallow timing d9\n"}
CHUNKS = {"test": 100, "timing": 20}
RULE = ("seed(set, j) = convlines.beacon.seed_tag(randomness, tags[set] as UTF-8 bytes, j) = 2^62 + int.from_bytes("
        "SHA-256(R || tag || j.to_bytes(2, 'big'))[:8], 'big') mod 2^62; R = bytes.fromhex(randomness) = SHA-256(signature); "
        "j = 0..99 (test), 0..19 (timing)")
OWN = ("tags", "seed_rule", "seeds", "chunks")


def seeds(randomness):
    """{set: [seed of chunk j for j in 0..n-1]} from the round's randomness (64 hex)."""
    return {s: [B.seed_tag(randomness, TAGS[s].encode(), j) for j in range(CHUNKS[s])] for s in sorted(TAGS)}


def check(b, commit=None):
    """SURF-v2's checks (chain, round, two agreeing fetches after the round's time, randomness = SHA-256(signature)),
    T as the earliest activity record of F, the round as the first at least 600 s after T, and this test's tags, rule
    and seeds re-derived from the randomness."""
    assert b.get("activity") and b.get("freeze_commit") and b.get("push_time") is not None, "not a freeze record"
    assert commit is None or b["freeze_commit"] == commit, f"the record names {b['freeze_commit']}, not {commit}"
    B.check({k: v for k, v in b.items() if k not in OWN})
    assert len(bytes.fromhex(b["randomness"])) == 32, "R is not 32 bytes"
    t, first = B.push_activity(b["activity"], b["freeze_commit"])
    assert t == b["push_time"] and first == b["push_activity"], "T is not the earliest activity record of F"
    assert b["round"] == B.beacon_round(t), "the round is not the first at least 600 s after T"
    assert b["tags"] == TAGS and b["seed_rule"] == RULE and b["chunks"] == CHUNKS, "not this test's tags or seed rule"
    s = seeds(b["randomness"])
    assert b["seeds"] == s, "a seed does not re-derive from the randomness"
    allv = [x for v in s.values() for x in v]
    assert len(set(allv)) == len(allv), "two chunks share a seed"
    return b


def load(path, commit=None):
    return check(json.load(open(path)), commit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--commit", required=True)
    f.add_argument("--activity", required=True)
    f.add_argument("--push-source", required=True)
    f.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--beacon", required=True)
    c.add_argument("--commit", default=None)
    a = ap.parse_args()
    if a.cmd == "fetch":
        assert len(a.commit) == 40, "the full sha of the freeze commit"
        assert not os.path.exists(a.out), f"{a.out} exists: the beacon is fetched once"
        act = json.load(open(a.activity))
        t, first = B.push_activity(act, a.commit)
        r = B.beacon_round(t)
        print(f"push {datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()} (activity {first['id']}, "
              f"{first.get('activity_type')} of {first.get('ref')}): round {r} at "
              f"{datetime.datetime.fromtimestamp(B.round_time(r), datetime.timezone.utc).isoformat()}", flush=True)
        b = B.fetch(r)
        b.update(freeze_commit=a.commit, push_time=t, push_activity=first, activity=act, push_source=a.push_source,
                 delay_s=B.DELAY, tags=TAGS, chunks=CHUNKS, seed_rule=RULE, seeds=seeds(b["randomness"]))
        check(b, a.commit)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "x") as fh:
            json.dump(b, fh, indent=1)
    b = load(a.out if a.cmd == "fetch" else a.beacon, getattr(a, "commit", None))
    print(json.dumps({k: b.get(k) for k in ("round", "round_time", "randomness", "freeze_commit", "push_time")}), flush=True)


if __name__ == "__main__":
    main()
