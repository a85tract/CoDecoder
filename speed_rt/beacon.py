"""SPEED-RT's beacon record (SPEED_RT.md, "The timing shots"), with SURF-v2's fetch and checks unchanged
(surface_v2/convlines/beacon.py: drand mainnet, four endpoints, two agreeing answers fetched after the round's time,
randomness = SHA-256(signature)); only the sets' tags are SPEED-RT's (speed_rt/sets.py).

T is the time of the earliest record in GitHub's activity list of the repository whose `after` is the freeze commit F
(GitHub's own record of the push); the round is the first one at least 600 s after T. The record keeps the whole
activity list as GitHub returned it, and every check derives T and the round from it again.
    python -m speed_rt.beacon fetch --commit F --activity activity.json --push-source "$(cat push_source.txt)" \\
        --out speed_rt/beacon.json [--sets speed_rt/cascade/sets_cas.json]   (a stage's own sets file)
    python -m speed_rt.beacon check --beacon speed_rt/beacon.json [--commit F]
    python -m speed_rt.beacon dry --round R --out DRY.json     (a past round, for dry runs only: never a timing seed)"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "surface_v2"))
from convlines import beacon as B  # noqa: E402  (SURF-v2's, unchanged)

from speed_rt import sets  # noqa: E402


ROOT = os.path.dirname(HERE)


def tags(sets_file=None):
    """The tags of a stage's sets; sets_file is the sets file's path in the repository (default speed_rt/config.json)."""
    return {s: sets.tag(s).decode() for s in sorted(sets.sets(sets_file and os.path.join(ROOT, sets_file)))}


def check(b, commit=None):
    """SURF-v2's checks (chain, round fixed by T, two agreeing fetches after the round's time, randomness), T as the
    earliest activity record of the freeze commit, and SPEED-RT's tags and seed rule."""
    assert not b.get("dry"), "a dry record holds no timing seeds"
    assert b.get("activity") and b.get("freeze_commit") and b.get("push_time") is not None, "not a freeze record"
    assert commit is None or b["freeze_commit"] == commit, f"the record names {b['freeze_commit']}, not {commit}"
    B.check({k: v for k, v in b.items() if k not in ("tags", "seed_rule", "sets_file")})
    t, first = B.push_activity(b["activity"], b["freeze_commit"])
    assert t == b["push_time"] and first == b["push_activity"], "T is not the earliest activity record of F"
    assert b["round"] == B.beacon_round(t), "the round is not the first at least 600 s after T"
    assert b["tags"] == tags(b.get("sets_file")) and b["seed_rule"] == sets.RULE, \
        "the record's tags or seed rule are not SPEED-RT's"
    return b


def load(path, commit=None, dry_ok=False):
    b = json.load(open(path))
    if b.get("dry"):
        assert dry_ok, f"{path} is a dry record"
        B.check({k: v for k, v in b.items() if k not in ("tags", "seed_rule", "dry", "sets_file")})
        return b
    return check(b, commit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--commit", required=True)
    f.add_argument("--activity", required=True)
    f.add_argument("--push-source", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--sets", default=None, help="the stage's sets file, a path in the repository (stage 1: "
                   "speed_rt/cascade/sets_cas.json; default speed_rt/config.json)")
    c = sub.add_parser("check")
    c.add_argument("--beacon", required=True)
    c.add_argument("--commit", default=None)
    d = sub.add_parser("dry")
    d.add_argument("--round", type=int, required=True)
    d.add_argument("--out", required=True)
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
                 delay_s=B.DELAY, tags=tags(a.sets), seed_rule=sets.RULE)
        if a.sets:
            b["sets_file"] = a.sets
        check(b, a.commit)
        with open(a.out, "x") as fh:
            json.dump(b, fh, indent=1)
    elif a.cmd == "dry":
        assert B.round_time(a.round) < B.round_time(B.beacon_round(datetime.datetime.now().timestamp())) - 86400, \
            "a dry round is at least a day old"
        b = B.fetch(a.round, wait=False)
        b.update(dry=True, tags=tags(), seed_rule=sets.RULE, note="dry runs only; never a timing seed")
        with open(a.out, "x") as fh:
            json.dump(b, fh, indent=1)
    b = load(a.out if a.cmd in ("fetch", "dry") else a.beacon, getattr(a, "commit", None), dry_ok=a.cmd == "dry")
    print(json.dumps({k: b.get(k) for k in ("round", "round_time", "randomness", "freeze_commit", "push_time", "dry")}),
          flush=True)


if __name__ == "__main__":
    main()
