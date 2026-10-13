"""SPEED-RT's timing sets and their seeds (SPEED_RT.md, "The timing shots"). Standard library only, so every host
(the pods, Alpine's login node, the Mac's system Python) can import it.

    seed(set, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG(set) || j as 2 bytes big-endian)) mod 2^62
    TAG(set) = "convlines speed-rt <set>" followed by a newline

R is the 32 randomness bytes of the drand round in speed_rt/beacon.json (SHA-256 of the round's signature). This is the
formula of the BB72, gross and SURF-v2 tests (bb72/ising_compare/bb/shots.py test_seed; surface_v2/convlines/beacon.py
seed_tag). Chunk j of a set is the j-th seed of its TAG; config.json's "sets" gives every set's kind, size and chunking.
    python speed_rt/sets.py BEACON OUT [SETS]   every set's seeds (OUT, JSON; the draw's input, checked again by it);
                                                SETS: the stage's sets file (default config.json)"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = 2 ** 62
RULE = ("seed(set, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG(set) || j as 2 bytes big-endian)) "
        "mod 2^62; TAG(set) = 'convlines speed-rt <set>' + newline; R = the beacon round's 32 randomness bytes")


def sets(path=None):
    """{set: its entry} of a stage's sets file (default config.json; stage 1, Cascade: cascade/sets_cas.json): kind,
    chunks, chunk_shots, and per kind d / code / p."""
    return json.load(open(path or os.path.join(HERE, "config.json")))["sets"]


def tag(name):
    return f"convlines speed-rt {name}\n".encode()


def seed(randomness, name, j):
    """randomness: the round's randomness, 64 hex characters; 0 <= j < 2^16."""
    r = bytes.fromhex(randomness)
    if len(r) != 32 or not 0 <= j < 2 ** 16:
        raise ValueError((len(r), j))
    h = hashlib.sha256(r + tag(name) + j.to_bytes(2, "big")).digest()
    return FLOOR + int.from_bytes(h[:8], "big") % FLOOR


def all_seeds(randomness, path=None):
    """{set: [seed of chunk 0, 1, ...]}; every seed of every set distinct (asserted)."""
    out = {s: [seed(randomness, s, j) for j in range(e["chunks"])] for s, e in sorted(sets(path).items())}
    flat = [x for v in out.values() for x in v]
    assert len(set(flat)) == len(flat), "two chunks share a seed"
    return out


def main():
    b = json.load(open(sys.argv[1]))
    path = sys.argv[3] if len(sys.argv) > 3 else None
    assert len(b["randomness"]) == 64, "no 32-byte randomness in the beacon record"
    out = dict(rule=RULE, beacon_round=b["round"], randomness=b["randomness"], dry=bool(b.get("dry")),
               tags={s: tag(s).decode() for s in sets(path)}, seeds=all_seeds(b["randomness"], path))
    with open(sys.argv[2], "x") as fh:
        json.dump(out, fh, indent=1)
    print(f"{sum(len(v) for v in out['seeds'].values())} seeds of {len(out['seeds'])} sets -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
