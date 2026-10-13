"""Study 2's test set T2 (SETTINGS §0.8.2; the study-2 draft review, items 1-3): the beacon, the seeds, the
sampler, the manifest, its check and the hash-checked loader. T2 exists only after the freeze commit, the claim, the
beacon and the sampling reservation.

Seeds. seed(d, j) = 2^62 + (first 8 bytes of sha256(TAG || approval bytes || R || d || j) mod 2^62), with the approval
record's committed bytes, R the 32 randomness bytes of the drand mainnet beacon (League of Entropy, chain 8990e7a9...,
30 s rounds) at the round fixed by the freeze commit: the first round whose time is at least 10 minutes after the
freeze commit's time, released only after the immutable freeze, a single public draw that nobody here controls. d is
one byte, j two bytes big-endian (fixed widths after the approval's bytes, so the encoding is unambiguous). Every seed
is at least 2^62; every earlier seed namespace lies below 2^33 (data/study2/seed_namespaces.json); the seeds are checked
distinct.

Files: as T1's (packed detector bits, the observable, the seed) but written byte-deterministically (stored zip
members, fixed timestamps), once, never overwritten; the same seeds on the same Stim build and SIMD path give the same
bytes, and the manifest records the host and its SIMD flags. The sampler computes no decoder and no statistic of the
shots. It runs only with a sampling ticket: the claim's beacon and its one reservation (an atomic mkdir in the claim
directory, naming the output directory); before sampling it checks the frozen code, circuits and versions and writes a
start record, then an end or a failure record (the claim's log; the acceptance needs exactly one start and one end).
    python -m icmp.t2 beacon --freeze-time T --out beacon.json      (the Linux workstation: waits for the round, fetches it twice)
    python -m icmp.t2 sample --ticket TICKET                         (after the claim, the beacon, the reservation)
    python -m icmp.t2 check --manifest M --approval A --beacon B --freeze F [--reservation R] [--dir DIR]"""
import argparse
import datetime
import hashlib
import io
import json
import os
import platform
import sys
import time

import numpy as np

TAG = b"ising_compare study 2 T2\n"
SPLIT = "test2"
CHUNK = 10_000                     # as T1's (icmp.shots.CHUNK); this module imports neither Stim nor icmp.circuits at
P = 0.006                          # load time (circuits.P_VERDICT), so the claim's acceptance runs where Stim is absent
FLOOR = 2 ** 62
NAMESPACES_BELOW = 2 ** 33         # every earlier seed namespace lies below this (data/study2/seed_namespaces.json)
DISTANCES = (7, 9, 11, 13)
SIMD = ("sse2", "sse4_1", "sse4_2", "avx", "avx2", "avx512f", "avx512bw", "avx512vl", "asimd", "sve")
DRAND = dict(chain_hash="8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce",
             public_key="868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801"
                        "af31",
             genesis_time=1595431050, period=30, scheme="pedersen-bls-chained", beacon_id="default")
ENDPOINTS = ("https://api.drand.sh", "https://api2.drand.sh", "https://api3.drand.sh", "https://drand.cloudflare.com")
AGENT = "ising_compare-study2-beacon/1"   # drand.cloudflare.com refuses Python's default User-Agent (HTTP 403)
DELAY = 600                        # seconds between the freeze commit's time and the beacon round's time, at least
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(open(path, "rb").read())


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---- the beacon -------------------------------------------------------------------------------------------------------
def beacon_round(freeze_time):
    """The first drand mainnet round whose time is at least DELAY seconds after freeze_time (UNIX seconds)."""
    g, p = DRAND["genesis_time"], DRAND["period"]
    return max(1, -(-(int(freeze_time) + DELAY - g) // p) + 1)


def round_time(r):
    return DRAND["genesis_time"] + (r - 1) * DRAND["period"]


def fetch_beacon(r, endpoints=ENDPOINTS, get=None, clock=time.time, wait=True):
    """Round r of the chain from every endpoint (after its time): the record with the fetch log. At least two
    endpoints must answer with the same round, signature and previous signature; randomness = sha256(signature)."""
    import urllib.request
    get = get or (lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}),
                                                     timeout=20).read())
    while wait and clock() < round_time(r) + 2:
        time.sleep(min(30.0, round_time(r) + 2 - clock()))
    assert clock() >= round_time(r), f"round {r} is not released before {round_time(r)}"
    log = []
    for ep in endpoints:
        url = f"{ep}/{DRAND['chain_hash']}/public/{r}"
        t0 = clock()
        try:
            body = get(url)
            j = json.loads(body)
            log.append(dict(url=url, t=t0, ok=True, body_sha256=sha256_bytes(body), round=j.get("round"),
                            signature=j.get("signature"), previous_signature=j.get("previous_signature"),
                            randomness=j.get("randomness")))
        except Exception as e:                                       # recorded; two agreeing answers are required
            log.append(dict(url=url, t=t0, ok=False, error=repr(e)[:300]))
    b = dict(version=1, chain=DRAND, round=r, round_time=round_time(r), fetched=log, fetched_by=platform.node())
    good = [x for x in log if x["ok"]]
    assert len(good) >= 2, f"fewer than two endpoints answered round {r}: {log}"
    b.update(signature=good[0]["signature"], previous_signature=good[0]["previous_signature"],
             randomness=sha256_bytes(bytes.fromhex(good[0]["signature"])))
    check_beacon(b)
    return b


def check_beacon(b, freeze_time=None):
    """The beacon record: this chain, the round fixed by the freeze commit's time (when given), two or more fetches
    after the round's time agreeing on round, signature and previous signature, randomness = sha256(signature)."""
    assert b["chain"] == DRAND, "another drand chain"
    if freeze_time is not None:
        assert b["round"] == beacon_round(freeze_time), \
            f"round {b['round']} is not the first at least {DELAY} s after the freeze commit ({beacon_round(freeze_time)})"
    assert b["round_time"] == round_time(b["round"])
    good = [x for x in b["fetched"] if x.get("ok")]
    assert len(good) >= 2, "fewer than two endpoint fetches"
    for x in good:
        assert x["round"] == b["round"] and x["signature"] == b["signature"] and \
            x["previous_signature"] == b["previous_signature"], f"{x['url']}: another answer"
        assert x["t"] >= b["round_time"], f"{x['url']}: fetched before the round's time"
        assert x.get("randomness") in (None, b["randomness"]), f"{x['url']}: randomness differs"
    assert b["randomness"] == sha256_bytes(bytes.fromhex(b["signature"])), "randomness is not sha256(signature)"
    return b


# ---- seeds ------------------------------------------------------------------------------------------------------------
def seed(approval_bytes, randomness, d, j):
    assert d in DISTANCES and 0 <= j < 10_000 and len(bytes.fromhex(randomness)) == 32
    h = hashlib.sha256(TAG + approval_bytes + bytes.fromhex(randomness) + bytes([d]) + j.to_bytes(2, "big")).digest()
    return FLOOR + int.from_bytes(h[:8], "big") % FLOOR


def seeds(approval_bytes, randomness, chunks):
    """{(d, j): seed}, every one at least 2^62 (above every earlier namespace), below 2^63 and all distinct."""
    s = {(d, j): seed(approval_bytes, randomness, d, j) for d, js in chunks.items() for j in js}
    v = list(s.values())
    assert min(v) >= FLOOR > NAMESPACES_BELOW and max(v) < 2 ** 63 and len(set(v)) == len(v), "T2 seeds not disjoint"
    return s


def name(d, j):
    return f"d{d}_p{P}_X_{SPLIT}_{j:03d}.npz"            # as icmp.shots.name: d7_p0.006_X_test2_000.npz


def circuit(d):
    from icmp import circuits
    return circuits.circuit(d, noise=circuits.NOISE_VERDICT)


def parse_chunks(spec):
    """{"7": "0-199", "9": "0-99", ...} (the freeze's route.t2.chunks) -> {7: [0, ..., 199], ...}."""
    out = {}
    for d, s in spec.items():
        js = []
        for part in str(s).split(","):
            a, _, b = part.strip().partition("-")
            js += list(range(int(a), int(b or a) + 1))
        out[int(d)] = js
    assert set(out) <= set(DISTANCES) and all(js and len(set(js)) == len(js) for js in out.values()), spec
    return out


# ---- files ------------------------------------------------------------------------------------------------------------
def write_npz(path, arrays):
    """np.savez's layout with deterministic bytes (stored members in name order, fixed timestamps), written once."""
    import zipfile
    with open(path, "xb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o444 << 16
            z.writestr(zi, buf.getvalue())


def host():
    info = dict(node=platform.node(), machine=platform.machine(), platform=platform.platform())
    try:
        txt = open("/proc/cpuinfo").read()
        model = [ln.split(":", 1)[1].strip() for ln in txt.splitlines() if ln.startswith(("model name", "Model"))]
        flags = set(next(ln for ln in txt.splitlines() if ln.startswith(("flags", "Features"))).split(":", 1)[1].split())
        info.update(cpu=model[0] if model else "", simd=sorted(flags & set(SIMD)))
    except (OSError, StopIteration):
        info.update(cpu="", simd=[])
    return info


def sample(approval_bytes, beacon, chunks, out_dir, ident):
    """Writes every chunk once into out_dir (which must not exist) and returns the manifest. ident: the claim's
    freeze_sha256, approval_ref, approval_sha256, beacon_sha256, reservation (id, sha256) and ticket_sha256."""
    import stim
    from icmp import circuits
    s = seeds(approval_bytes, beacon["randomness"], chunks)
    os.makedirs(out_dir)                                     # a second run into the same directory stops here
    started = now()
    files = {}
    for d, js in sorted(chunks.items()):
        c = circuit(d)
        for j in js:
            det, obs = c.compile_detector_sampler(seed=s[(d, j)]).sample(CHUNK, separate_observables=True)
            path = os.path.join(out_dir, name(d, j))
            write_npz(path, dict(det=np.packbits(det.astype(np.uint8), axis=1), obs=obs[:, 0].astype(np.uint8),
                                 seed=np.int64(s[(d, j)])))
            files[name(d, j)] = dict(d=d, j=j, seed=s[(d, j)], shots=CHUNK, sha256=sha256_file(path))
            print(name(d, j), files[name(d, j)]["sha256"][:16], flush=True)
    return dict(version=2, study="2", split=SPLIT, tag=TAG.decode(), ident=ident,
                beacon=dict(round=beacon["round"], randomness=beacon["randomness"]),
                chunks={str(d): js for d, js in sorted(chunks.items())},
                circuit_sha256={str(d): circuits.sha(circuit(d)) for d in sorted(chunks)}, files=files,
                host=host(), versions=dict(stim=stim.__version__, numpy=np.__version__, python=sys.version.split()[0]),
                code_sha256={"icmp/t2.py": sha256_file(__file__)}, started=started, ended=now())


def check(man, approval_bytes, beacon_bytes, fz, freeze_sha256, reservation_bytes=None, shot_dir=None):
    """The manifest against the claim's records (the acceptance; every T2 read repeats it): study, split and tag; the
    claimed freeze, approval and beacon; the reservation (when given); exactly the frozen chunks with the derived
    seeds; the frozen circuits, Stim and numpy versions and icmp/t2.py; the host recorded; with shot_dir every file
    re-hashed. Raises on any difference."""
    r = fz["route"]
    b = json.loads(beacon_bytes)
    i = man["ident"]
    assert (man["version"], man["study"], man["split"], man["tag"]) == (2, "2", SPLIT, TAG.decode()), \
        "not a study-2 T2 manifest"
    assert i["freeze_sha256"] == freeze_sha256 and r["study"] == "2", "the manifest names another freeze"
    assert i["approval_sha256"] == sha256_bytes(approval_bytes), "the manifest names another approval"
    assert i["beacon_sha256"] == sha256_bytes(beacon_bytes) and man["beacon"] == dict(
        round=b["round"], randomness=b["randomness"]), "the manifest names another beacon"
    if reservation_bytes is not None:
        res = json.loads(reservation_bytes)
        assert i["reservation"] == dict(id=res["id"], sha256=sha256_bytes(reservation_bytes)), \
            "the manifest names another sampling reservation"
    chunks = parse_chunks(r["t2"]["chunks"])
    assert {int(d): js for d, js in man["chunks"].items()} == chunks, "the chunks differ from the freeze's"
    s = seeds(approval_bytes, b["randomness"], chunks)
    want = {name(d, j): (d, j) for d, js in chunks.items() for j in js}
    assert set(man["files"]) == set(want), "the files differ from the frozen chunks"
    for f, e in man["files"].items():
        d, j = want[f]
        assert (e["d"], e["j"], e["seed"], e["shots"]) == (d, j, s[(d, j)], CHUNK), f"{f}: entry differs"
        assert len(e["sha256"]) == 64 and int(e["sha256"], 16) >= 0, f
        if shot_dir is not None:
            assert sha256_file(os.path.join(shot_dir, f)) == e["sha256"], f"{f}: hash mismatch"
    assert man["circuit_sha256"] == {str(d): fz["distances"][str(d)]["circuit_sha256"] for d in chunks}, \
        "the manifest's circuits are not the frozen ones"
    assert man["versions"]["stim"] == r["versions"]["stim"] and man["versions"]["numpy"] == r["versions"]["numpy"], \
        "the sampler's Stim or numpy is not the frozen one"
    assert man["code_sha256"]["icmp/t2.py"] == r["code_sha256"]["icmp/t2.py"], "the sampler is not the frozen icmp/t2.py"
    assert man["host"].get("node") and "simd" in man["host"], "the sampling host is not recorded"
    return dict(files=len(want), randomness=b["randomness"])


def load(man, d, chunks, shot_dir, packed=False):
    """Concatenated T2 chunks, each sha256-checked against the (accepted) manifest and loaded from the very bytes
    hashed; fails closed."""
    dets, obs = [], []
    for j in chunks:
        f = name(d, j)
        assert f in man["files"], f"{f}: not in the manifest"
        raw = open(os.path.join(shot_dir, f), "rb").read()
        assert sha256_bytes(raw) == man["files"][f]["sha256"], f"{f}: hash mismatch"
        z = np.load(io.BytesIO(raw))
        assert int(z["seed"]) == man["files"][f]["seed"], f"{f}: seed differs from the manifest"
        dets.append(z["det"])
        obs.append(z["obs"])
    det = np.concatenate(dets)
    if not packed:
        det = np.unpackbits(det, axis=1)[:, :circuit(d).num_detectors]
    return det, np.concatenate(obs)


# ---- the sampling run -------------------------------------------------------------------------------------------------
def ticket_inputs(ticket):
    """A sampling ticket's records, checked against each other: (approval bytes, freeze, freeze sha256, claim, beacon
    bytes, reservation bytes). The claim's beacon, its reservation and the reservation's one consumption naming this
    very ticket (made by the claim when it issued the one sampling ticket); no T2 manifest accepted yet."""
    claim = json.load(open(os.path.join(ticket, "claim.json")))
    fz_raw = open(os.path.join(ticket, "freeze.json"), "rb").read()
    appr = open(os.path.join(ticket, "approved.json"), "rb").read()
    bec = open(os.path.join(ticket, "beacon.json"), "rb").read()
    res = open(os.path.join(ticket, "reservation.json"), "rb").read()
    fh = sha256_bytes(fz_raw)
    fz = json.loads(fz_raw)
    assert fz["route"]["study"] == "2" and claim["freeze_sha256"] == fh, "the ticket's freeze is not the claim's"
    assert json.loads(appr)["freeze_sha256"] == fh, "the approval names another freeze"
    assert claim["approval_sha256"] == sha256_bytes(appr), "the approval differs from the claim's"
    assert claim.get("beacon_sha256") == sha256_bytes(bec), "the beacon is not the claim's"
    check_beacon(json.loads(bec), claim["freeze_commit_time"])
    r = json.loads(res)
    assert (r["freeze_sha256"], r["approval_sha256"], r["beacon_sha256"]) == (
        fh, claim["approval_sha256"], claim["beacon_sha256"]), "the reservation is not this claim's"
    assert "t2_manifest_sha256" not in claim and not os.path.exists(os.path.join(ticket, "t2_manifest.json")), \
        "a reading ticket: the claim has accepted a T2 manifest already"
    assert os.path.exists(os.path.join(ticket, "consumed.json")), "not the claim's sampling ticket (no consumption)"
    con = json.load(open(os.path.join(ticket, "consumed.json")))
    assert (con["reservation_id"], con["reservation_sha256"], con["ticket_sha256"]) == (
        r["id"], sha256_bytes(res), sha256_file(os.path.join(ticket, "claim.json"))), \
        "the consumption names another reservation or ticket"
    return appr, fz, fh, claim, bec, res


def frozen_checks(fz, chunks):
    """Before sampling: the frozen code (every file the freeze pins, icmp/t2.py among them), circuits and versions."""
    import stim
    from icmp import circuits
    r = fz["route"]
    assert "icmp/t2.py" in r["code_sha256"], "the freeze does not pin icmp/t2.py"
    bad = sorted(f for f, h in r["code_sha256"].items() if sha256_file(os.path.join(HERE, f)) != h)
    assert not bad, f"files differ from the frozen code: {bad}"
    assert stim.__version__ == r["versions"]["stim"] and np.__version__ == r["versions"]["numpy"], \
        f"Stim {stim.__version__} / numpy {np.__version__} are not the frozen versions"
    for d in chunks:
        assert circuits.sha(circuit(d)) == fz["distances"][str(d)]["circuit_sha256"], f"d{d}: not the frozen circuit"


def run(ticket, preflight=False):
    """The sampling run of a sampling ticket: frozen checks, a start record, the files, the manifest, an end record
    (or a failure record, and the error propagates). preflight: every check of the ticket, the tree and the output on
    this host, without the launch (made after it) and without writing anything (s2_claim.sh launch runs it first)."""
    from icmp import s2guard, testguard
    appr, fz, fh, claim, bec, res = ticket_inputs(ticket)
    r = json.loads(res)
    L = None if preflight else s2guard.require_launch(ticket, "t2 sample", r["id"])   # only in the launched job
    assert not [x for x in testguard.read_log(os.path.join(ticket, "reads.log"))
                if str(x.get("what", "")).startswith("t2 sample")], "this ticket's sampling has started already"
    out = s2guard.canonical(r["out"])            # absolute and canonical: the same directory from any working directory
    chunks = parse_chunks(fz["route"]["t2"]["chunks"])
    frozen_checks(fz, chunks)
    s2guard.check_runtime(fz)
    assert os.path.isdir(os.path.dirname(out)), f"{out}: its parent is not on this host (the reserved host's path)"
    assert not os.path.exists(out), f"{out} exists: T2 is sampled once, into the reserved directory"
    if preflight:
        print("PREFLIGHT OK: t2 sample", ticket, "->", out, flush=True)
        return None
    ident = dict(freeze_sha256=fh, approval_ref=claim["approval_ref"], approval_sha256=claim["approval_sha256"],
                 beacon_sha256=claim["beacon_sha256"], reservation=dict(id=r["id"], sha256=sha256_bytes(res)),
                 ticket_sha256=testguard.sha256_file(os.path.join(ticket, "claim.json")),
                 launch=dict(cluster=L["cluster"], job=L["job"]))
    start = testguard.now()
    marker = s2guard.start_once(L)                 # the one start of this job's run, outside the ticket
    testguard.record(ticket, f"t2 sample start {r['id']}", start, {
        "reservation.json": sha256_bytes(res), "launch.json": sha256_file(os.path.join(ticket, "launch.json")),
        "start_marker": marker})
    try:
        man = sample(appr, json.loads(bec), chunks, out, ident)
        mpath = os.path.join(out, "t2_manifest.json")
        with open(mpath, "x") as fh_out:
            json.dump(man, fh_out, indent=1)
            fh_out.write("\n")
        for f in os.listdir(out):
            os.chmod(os.path.join(out, f), 0o444)
        os.chmod(out, 0o555)
    except BaseException as e:
        testguard.record(ticket, f"t2 sample failed {r['id']}", start, {"error": repr(e)[:300]})
        raise
    s2guard.check_runtime(fz)
    testguard.record(ticket, f"t2 sample end {r['id']}", start, {"t2_manifest.json": sha256_file(mpath)})
    print("manifest", mpath, sha256_file(mpath), len(man["files"]), "files", flush=True)
    return mpath


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("beacon")
    b.add_argument("--freeze-time", type=int, required=True, help="the freeze commit's committer time (UNIX seconds)")
    b.add_argument("--out", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--ticket", required=True)
    s.add_argument("--preflight", action="store_true", help="every check, nothing written (before the launch)")
    c = sub.add_parser("check")
    c.add_argument("--manifest", required=True)
    c.add_argument("--approval", required=True, help="the approval record's bytes (the file at its commit)")
    c.add_argument("--beacon", required=True)
    c.add_argument("--freeze", required=True)
    c.add_argument("--reservation", default=None)
    c.add_argument("--dir", default=None, help="re-hash every file there too")
    a = ap.parse_args()
    if a.cmd == "beacon":
        rec = fetch_beacon(beacon_round(a.freeze_time))
        with open(a.out, "x") as fh:
            json.dump(rec, fh, indent=1)
            fh.write("\n")
        print("beacon round", rec["round"], "randomness", rec["randomness"], flush=True)
    elif a.cmd == "sample":
        run(a.ticket, a.preflight)
    else:
        fz_raw = open(a.freeze, "rb").read()
        res = open(a.reservation, "rb").read() if a.reservation else None
        r = check(json.load(open(a.manifest)), open(a.approval, "rb").read(), open(a.beacon, "rb").read(),
                  json.loads(fz_raw), sha256_bytes(fz_raw), res, a.dir)
        print("T2 manifest OK:", r["files"], "files")


if __name__ == "__main__":
    main()
