"""A1's evaluation sets (revision/REVISION_PLAN.md 2.1, "Data"). Each chunk is drawn once on Alpine, stored packed with
its seed and its circuit's sha256, and listed with the file's sha256 in a manifest (rows printed and appended to
--rows; the real sets' rows go to data/a1/sets_manifest.csv in git); load() reads chunks back only through a manifest
row (sha256-checked, fail closed).
  surface  kinds cal (71 M, 20 chunks), test (72 M, 100 chunks), dry (79 M): chunk j = 10,000 shots of the d = 9 verdict
           circuit (icmp.circuits.circuit(9, noise=NOISE_VERDICT), p = 0.6 %), seed a1.seeds.surface_seed(kind, 9, j);
           obs = observable 0
  bb72     kinds a1cal (20 chunks), a1test (50), a1dry: chunk j = 10,000 shots of the registered BB72 circuit at p = 0.3 %
           (qnd.circuits.load('bb72', 'X', 0.003), sha-checked), seed bb.shots.seed_of(kind, 'bb72', 0.003, j); obs = the
           12 observables
The dry kinds are synthetic stand-ins for the dry runs; cal and test are drawn only after the freeze: draw refuses them
without --freeze COMMIT (the frozen commit of the arms and the analysis, recorded in every row).
Publication (gate finding 5 on revision/A1_READ_PLAN.md): a chunk is written and synced under a hidden temporary name
and then linked to its name, which never replaces a file, so a chunk file exists only complete; the chunk itself
records every fact of its row (code, kind, chunk, seed, circuit sha256, freeze, host, stim). Its row is then made from
the published bytes (row_of: every member read back, the shot count and widths, the seed, circuit and freeze checked)
and --rows is rewritten whole under a temporary name and renamed into place, so it is never torn. Run again with the
same chunks and FREEZE after an interruption, draw keeps every chunk that has its row (its bytes checked against it),
makes the missing row of a published chunk from its validated bytes, and draws only the chunks never published: no
chunk is ever drawn twice. A row without its file, bytes that differ from their row, or a row of another freeze stop it.
    python -m a1.sets draw --code surface --kind dry --chunks 0-9 --rows ROWS.csv
    python -m a1.sets draw --code bb72 --kind a1dry --chunks 0-9 --rows ROWS.csv"""
import argparse
import csv
import hashlib
import io
import os
import platform

import numpy as np

from a1 import seeds as a1seeds

SETS = os.path.expanduser(os.environ.get("A1_SETS", "~/work/sets"))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(HERE, "data", "a1", "sets_manifest.csv")
FIELDS = ["file", "sha256", "shots", "seed", "code", "kind", "chunk", "circuit_sha256", "freeze", "host", "stim"]
DRY = {"surface": "dry", "bb72": "a1dry"}
P = {"surface": 0.006, "bb72": 0.003}
CHUNK = 10_000                      # shots per chunk


def name(code, kind, j):
    return f"a1_{code}_{kind}_{j:03d}.npz"


def seed_of(code, kind, j):
    if code == "surface":
        return a1seeds.surface_seed(kind, 9, j)
    from bb import shots
    return shots.seed_of(kind, "bb72", 0.003, j)


def circuit(code):
    if code == "surface":
        from icmp import circuits
        return circuits.circuit(9, noise=circuits.NOISE_VERDICT)
    from qnd import circuits
    return circuits.load("bb72", "X", 0.003)


def draw(code, kind, j, circ, n=CHUNK):
    det, obs = circ.compile_detector_sampler(seed=seed_of(code, kind, j)).sample(n, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def index(chunks, n=CHUNK):
    """The shots' indices in their set, in load()'s order: n c + r for row r of chunk c (a shot keeps its index in any
    subset of the chunks)."""
    return np.concatenate([n * c + np.arange(n) for c in chunks]) if len(chunks) else np.zeros(0, np.int64)


def publish(code, kind, j, circ, csha, freeze, stim_version):
    """Draw chunk j and publish it: written and synced under a hidden temporary name, then linked to the chunk's name
    (os.link never replaces a file: FileExistsError). Returns (det, obs)."""
    f = os.path.join(SETS, name(code, kind, j))
    det, obs = draw(code, kind, j, circ)
    tmp = os.path.join(SETS, f".{name(code, kind, j)}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez(fh, det=np.packbits(det, axis=1), obs=np.packbits(obs, axis=1), n=len(det), m=det.shape[1],
                 k=obs.shape[1], seed=seed_of(code, kind, j), circuit_sha256=csha, code=code, kind=kind, chunk=j,
                 freeze=freeze, host=platform.node(), stim=stim_version)
        fh.flush()
        os.fsync(fh.fileno())
    os.link(tmp, f)
    os.unlink(tmp)
    return det, obs


def row_of(code, kind, j, circ, csha, freeze):
    """The manifest row of published chunk j, made from its saved bytes, which must be the chunk's: every member reads
    back (the zip CRCs), CHUNK shots of the circuit's widths, packed, and the chunk's own record of code, kind, chunk,
    seed, circuit sha256 and freeze equal to what they must be. A mismatch raises ValueError (nothing is redrawn)."""
    f = os.path.join(SETS, name(code, kind, j))
    blob = open(f, "rb").read()
    with np.load(io.BytesIO(blob)) as z:
        v = {key: z[key] for key in z.files}
    m, k = circ.num_detectors, circ.num_observables
    want = {"n": CHUNK, "m": m, "k": k, "seed": seed_of(code, kind, j), "circuit_sha256": csha, "code": code,
            "kind": kind, "chunk": j, "freeze": freeze}
    got = {key: (v[key].item() if key in v else None) for key in want}
    bad = {key: (got[key], want[key]) for key in want if got[key] != want[key]}
    for key, w in (("det", m), ("obs", k)):
        if key not in v or v[key].dtype != np.uint8 or v[key].shape != (CHUNK, (w + 7) // 8):
            bad[key] = (None if key not in v else (str(v[key].dtype), v[key].shape), ("uint8", (CHUNK, (w + 7) // 8)))
    if bad:
        raise ValueError(f"{os.path.basename(f)} is not chunk {j} of {code} {kind} under freeze {freeze!r} "
                         f"(got, want): {bad}")
    return {"file": os.path.basename(f), "sha256": hashlib.sha256(blob).hexdigest(), "shots": got["n"],
            "seed": got["seed"], "code": code, "kind": kind, "chunk": j, "circuit_sha256": csha, "freeze": freeze,
            "host": v["host"].item() if "host" in v else "", "stim": v["stim"].item() if "stim" in v else ""}


def write_rows(path, rows):
    """Every row, written whole under a temporary name and renamed over `path` (never a torn file)."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def manifest(paths=None):
    """{file: row} from the git manifest and any further rows files (A1_SETS_ROWS, ':'-separated, or paths)."""
    rows = {}
    extra = paths if paths is not None else [x for x in os.environ.get("A1_SETS_ROWS", "").split(":") if x]
    for f in [MANIFEST] + list(extra):
        if os.path.exists(f):
            rows.update({r["file"]: r for r in csv.DictReader(open(f))})
    return rows


def load(code, kind, chunks, rows=None):
    """(det (n, m) uint8, obs (n, k) uint8, digest): the chunks in order, each sha256-checked against its manifest row;
    digest = sha256 of the rows' file hashes in order (what a read records)."""
    rows = manifest() if rows is None else rows
    dets, obss, hs = [], [], []
    for j in chunks:
        f = name(code, kind, j)
        r = rows.get(f)
        if r is None:
            raise FileNotFoundError(f"{f} is not in a manifest")
        blob = open(os.path.join(SETS, f), "rb").read()
        if hashlib.sha256(blob).hexdigest() != r["sha256"]:
            raise ValueError(f"sha256 mismatch for {f}")
        z = np.load(os.path.join(SETS, f))
        n = int(z["n"])
        dets.append(np.unpackbits(z["det"], axis=1, count=int(z["m"]))[:n])
        obss.append(np.unpackbits(z["obs"], axis=1, count=int(z["k"]))[:n])
        hs.append(r["sha256"])
    return np.concatenate(dets), np.concatenate(obss), hashlib.sha256(",".join(hs).encode()).hexdigest()


def parse_range(s):
    out = []
    for part in str(s).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    import stim
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draw"])
    ap.add_argument("--code", choices=["surface", "bb72"], required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--rows", required=True, help="the manifest rows' CSV file (kept rows and new ones, rewritten whole)")
    ap.add_argument("--freeze", default="", help="the frozen commit (required for every kind but the dry one)")
    a = ap.parse_args()
    ok = {"surface": ("cal", "test", "dry"), "bb72": ("a1cal", "a1test", "a1dry")}[a.code]
    assert a.kind in ok, f"kind {a.kind} is not one of {ok}"
    if a.kind != DRY[a.code] and not a.freeze:
        raise SystemExit("cal and test sets are drawn after the freeze only: give --freeze COMMIT")
    os.makedirs(SETS, exist_ok=True)
    circ = circuit(a.code)
    csha = hashlib.sha256(str(circ).encode()).hexdigest()
    rows = list(csv.DictReader(open(a.rows))) if os.path.exists(a.rows) else []
    have = {r["file"]: r for r in rows}
    count = {"KEPT": 0, "RECOVERED": 0, "DRAWN": 0}
    for j in parse_range(a.chunks):
        fn = name(a.code, a.kind, j)
        f = os.path.join(SETS, fn)
        r = have.get(fn)
        if r is not None:                      # drawn and recorded: check it, never draw it again
            if not os.path.exists(f):
                raise SystemExit(f"{fn} has a row but no file: it is never drawn again")
            if hashlib.sha256(open(f, "rb").read()).hexdigest() != r["sha256"]:
                raise SystemExit(f"{fn}: its bytes differ from its row")
            if r["freeze"] != a.freeze:
                raise SystemExit(f"{fn} was drawn under freeze {r['freeze']!r}, not {a.freeze!r}")
            count["KEPT"] += 1
            print("KEPT", fn, r["seed"], r["sha256"][:12], flush=True)
            continue
        if os.path.exists(f):                  # published, its row lost: the row from the validated bytes
            how, note = "RECOVERED", "row made from the saved bytes"
        else:
            det, obs = publish(a.code, a.kind, j, circ, csha, a.freeze, stim.__version__)
            how, note = "DRAWN", f"fired/shot {det.mean() * det.shape[1]:.1f} flip rate {obs.any(1).mean():.4f}"
        row = row_of(a.code, a.kind, j, circ, csha, a.freeze)
        rows.append(row)
        have[fn] = row
        write_rows(a.rows, rows)
        count[how] += 1
        print(how, row["file"], row["seed"], row["sha256"][:12], note, flush=True)
    print(f"DRAW {a.code} {a.kind}: " + ", ".join(f"{k.lower()} {v}" for k, v in count.items()), flush=True)


if __name__ == "__main__":
    main()
