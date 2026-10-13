"""The re-test's beacon (RETEST.md section 2): the first drand mainnet round whose time is at least 600 s after the
GitHub push time of the freeze commit, fetched from at least two endpoints that agree on round, signature and previous
signature; randomness = SHA-256(signature). Stage 3 and 4's code does the fetch and the checks (bb72/ising_compare/icmp/
t2.py: fetch_beacon, check_beacon, beacon_round); only the reference time differs: GitHub's push time.
    python retest/beacon.py --push-time 2026-10-12T22:00:00Z --commit HASH --out retest/beacon.json
    python retest/beacon.py --dry-round R --out DRY.json     (a past round, for the dry runs only)"""
import argparse
import calendar
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bb", "bb72", "ising_compare"))
from icmp import t2  # noqa: E402


def unix(s):
    """GitHub's push time, ISO 8601 in UTC ('Z' or '+00:00', seconds, an optional fraction) -> UNIX seconds, a fraction
    rounded up, so that the round is at least 600 s after the push. Python 3.6 (Alpine's login node) has no
    datetime.fromisoformat."""
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|\+00:00)", s.strip())
    if not m:
        raise SystemExit(f"--push-time {s!r}: need YYYY-MM-DDTHH:MM:SS[.f]Z (UTC)")
    return calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) + bool(m.group(2) and int(m.group(2)[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-time", default="", help="GitHub's push time of the freeze commit, ISO 8601, UTC")
    ap.add_argument("--commit", default="", help="the freeze commit (recorded)")
    ap.add_argument("--dry-round", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: the beacon is fetched once")
    tags = json.load(open(os.path.join(HERE, "tags.json")))
    if a.dry_round:
        b = t2.fetch_beacon(a.dry_round, wait=False)
        b.update(dry=True, note="a past round, for the dry runs only; never a test seed")
    else:
        ref = unix(a.push_time)
        b = t2.fetch_beacon(t2.beacon_round(ref))        # waits for the round's time, then fetches every endpoint
        t2.check_beacon(b, freeze_time=ref)               # the round is the first at least 600 s after the push
        b.update(freeze_commit=a.commit, push_time=a.push_time, reference_time=ref, delay_s=t2.DELAY)
    b.update(tags=tags["tags"], seed_rule=tags["rule"])
    json.dump(b, open(a.out, "w"), indent=1)
    print(json.dumps({k: b.get(k) for k in ("round", "round_time", "randomness", "push_time", "reference_time")}),
          flush=True)


if __name__ == "__main__":
    main()
