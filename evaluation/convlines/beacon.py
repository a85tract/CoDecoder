"""The public randomness behind a test set's seeds: a drand mainnet beacon round fixed by the time the frozen commit was
public, as in the BB72 and gross tests (bb/BB.md, "How the test was run"; the same chain, endpoints and checks).

The round is the first drand mainnet round whose time is at least 600 s after the moment the freeze commit was pushed
to GitHub (the push time recorded from GitHub, UNIX seconds). Chunk j of distance d then has the seed

    seed(d, j) = 2^62 + (int.from_bytes(SHA-256(R || TAG(d) || j as 2 bytes big-endian)[:8], "big") mod 2^62)

with R the round's 32 randomness bytes (randomness = SHA-256(signature)) and one TAG per distance (TAGS below), the BB
stage-3 formula. The record (beacon.json) holds the chain, the round, the answers of every endpoint asked (at least
two must agree on round, signature and previous signature, each fetched after the round's time), the freeze commit,
the push time and its source, the delay and the tags.

    python -m convlines.beacon fetch --commit SHA --push-time T --push-source TEXT --out surf_v2/beacon.json
    python -m convlines.beacon check --beacon surf_v2/beacon.json             (the record again, offline)"""
import argparse
import datetime
import hashlib
import json
import platform
import time

CHAIN = dict(chain_hash="8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce",
             public_key="868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801"
                        "af31",
             genesis_time=1595431050, period=30, scheme="pedersen-bls-chained", beacon_id="default")
ENDPOINTS = ("https://api.drand.sh", "https://api2.drand.sh", "https://api3.drand.sh", "https://drand.cloudflare.com")
AGENT = "convlines-surf-v2-beacon/1"     # drand.cloudflare.com refuses Python's default User-Agent (HTTP 403)
DELAY = 600                              # seconds from the GitHub push of the freeze commit to the round's time, at least
FLOOR = 2 ** 62
DISTANCES = (7, 9, 11, 13, 15, 17)
TAGS = {d: f"convlines surf-v2 test d{d}\n".encode() for d in DISTANCES}
SEED_RULE = "2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG(d) || j as 2 bytes big-endian)) mod 2^62"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def beacon_round(push_time):
    """The first round whose time is at least DELAY seconds after push_time (UNIX seconds)."""
    g, p = CHAIN["genesis_time"], CHAIN["period"]
    return max(1, -(-(int(push_time) + DELAY - g) // p) + 1)


def round_time(r):
    return CHAIN["genesis_time"] + (r - 1) * CHAIN["period"]


def seed_tag(randomness, tag, j):
    """BB stage 3's formula: 2^62 + (first 8 bytes of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62."""
    r = bytes.fromhex(randomness)
    assert len(r) == 32, "the beacon randomness must be 32 bytes"
    assert 0 <= j < 65536
    h = hashlib.sha256(r + tag + j.to_bytes(2, "big")).digest()
    return FLOOR + int.from_bytes(h[:8], "big") % FLOOR


def seed(randomness, d, j):
    """The test seed of chunk j at distance d from the round's randomness (hex)."""
    return seed_tag(randomness, TAGS[d], j)


def fetch(r, endpoints=ENDPOINTS, get=None, clock=time.time, wait=True):
    """Round r from every endpoint, after its time: the record with the fetch log (icmp.t2.fetch_beacon's rule)."""
    import urllib.request
    get = get or (lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}),
                                                     timeout=20).read())
    while wait and clock() < round_time(r) + 2:
        time.sleep(min(30.0, round_time(r) + 2 - clock()))
    assert clock() >= round_time(r), f"round {r} is not released before {round_time(r)}"
    log = []
    for ep in endpoints:
        url = f"{ep}/{CHAIN['chain_hash']}/public/{r}"
        t0 = clock()
        try:
            body = get(url)
            j = json.loads(body)
            log.append(dict(url=url, t=t0, ok=True, body_sha256=sha256_bytes(body), round=j.get("round"),
                            signature=j.get("signature"), previous_signature=j.get("previous_signature"),
                            randomness=j.get("randomness")))
        except Exception as e:                                       # recorded; two agreeing answers are required
            log.append(dict(url=url, t=t0, ok=False, error=repr(e)[:300]))
    b = dict(version=1, chain=CHAIN, round=r, round_time=round_time(r), fetched=log, fetched_by=platform.node())
    good = [x for x in log if x["ok"]]
    assert len(good) >= 2, f"fewer than two endpoints answered round {r}: {log}"
    b.update(signature=good[0]["signature"], previous_signature=good[0]["previous_signature"],
             randomness=sha256_bytes(bytes.fromhex(good[0]["signature"])))
    check(b)
    return b


def check(b, push_time=None):
    """The record: this chain, the round fixed by the push time (when given, else the record's own), two or more
    fetches after the round's time agreeing on round, signature and previous signature, randomness = SHA-256(signature),
    and (when present) the tags and the seed rule of this module."""
    assert b["chain"] == CHAIN, "another drand chain"
    pt = b.get("push_time") if push_time is None else push_time
    if pt is not None:
        assert b["round"] == beacon_round(pt), \
            f"round {b['round']} is not the first at least {DELAY} s after the push ({beacon_round(pt)})"
    assert b["round_time"] == round_time(b["round"])
    good = [x for x in b["fetched"] if x.get("ok")]
    assert len(good) >= 2, "fewer than two endpoint fetches"
    for x in good:
        assert x["round"] == b["round"] and x["signature"] == b["signature"] and \
            x["previous_signature"] == b["previous_signature"], f"{x['url']}: another answer"
        assert x["t"] >= b["round_time"], f"{x['url']}: fetched before the round's time"
        assert x.get("randomness") in (None, b["randomness"]), f"{x['url']}: randomness differs"
    assert b["randomness"] == sha256_bytes(bytes.fromhex(b["signature"])), "randomness is not sha256(signature)"
    if "tags" in b:
        assert b["tags"] == {str(d): t.decode() for d, t in TAGS.items()} and b["seed_rule"] == SEED_RULE, \
            "the record's tags or seed rule are not this module's"
    return b


def load(path, push_time=None):
    b = json.load(open(path))
    assert b.get("push_time") is not None and b.get("freeze_commit"), f"{path}: no push time or freeze commit"
    return check(b, push_time)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--commit", required=True, help="the freeze commit (full sha) that was pushed")
    f.add_argument("--push-time", type=int, required=True, help="GitHub's push time of the freeze commit, UNIX seconds")
    f.add_argument("--push-source", required=True, help="where the push time was read (e.g. the GitHub event's id)")
    f.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--beacon", required=True)
    a = ap.parse_args()
    if a.cmd == "fetch":
        assert len(a.commit) == 40, "the full sha of the freeze commit"
        r = beacon_round(a.push_time)
        print(f"push {datetime.datetime.fromtimestamp(a.push_time).astimezone().isoformat()}: round {r} at "
              f"{datetime.datetime.fromtimestamp(round_time(r)).astimezone().isoformat()}", flush=True)
        b = fetch(r)
        b.update(freeze_commit=a.commit, push_time=a.push_time, push_source=a.push_source, delay_s=DELAY,
                 tags={str(d): t.decode() for d, t in TAGS.items()}, seed_rule=SEED_RULE)
        check(b, a.push_time)
        with open(a.out, "x") as fh:
            json.dump(b, fh, indent=1)
    b = load(a.out if a.cmd == "fetch" else a.beacon)
    print(json.dumps({k: b[k] for k in ("round", "round_time", "randomness", "freeze_commit", "push_time")}), flush=True)


if __name__ == "__main__":
    main()
