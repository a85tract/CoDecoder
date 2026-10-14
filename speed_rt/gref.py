"""The Grace fit's reference timings of the surface stage (the stage's section, "The Grace conversion"): NVIDIA's own
PyMatching decoders, plain and correlated, on NVIDIA's legacy memory circuits at the points of the Ising paper's Tables
VII and IX (speed_rt/grace_fit.py lists them: plain PM at d = 13, 21, 31, correlated PM at d = 5, 9, 13, 17, 21, 31, each
at p = 0.3 % and 0.6 %), timed on the Linux workstation's Ryzen, one fresh single-threaded process per (decoder, point, repetition).

circuits  builds the 12 reference circuits (d = 5, 9, 13, 17, 21, 31 x p = 0.003, 0.006) with the repository's circuit
          builder from NVIDIA's code (evaluation/scripts/make_circuits.py `memory_circuit`: Ising-Decoding 3ce4379's
          MemoryCircuit, d rounds, X basis, code_rotation "XV", all detectors plus boundary detectors, the legacy
          single-parameter noise: idle, single- and two-qubit gate error p, SPAM 2p/3) -> circuits/ref_d<d>_p<3|6>.stim
          and circuits/ref_circuits.json (every file's sha256). A p = 0.006 circuit that the repository already holds
          (surface_v2/circuits/d<d>_legacy.stim, d = 9, 13, 17) must equal it byte for byte. Without --write it
          compares the files with a rebuild. Needs ISING_DECODING_DIR (a clone of NVIDIA/Ising-Decoding at 3ce4379).
time      one reference process: the circuit file (its sha256 checked against ref_circuits.json), the detector error
          model with decompose_errors=True and approximate_disjoint_errors=True (surface_v2/convlines/circuits.dem, the
          release's), plain PyMatching (Matching.from_detector_error_model) or correlated PyMatching (the same with
          enable_correlations=True), the point's drawn chunk (speed_rt/draw.py, srt-ref-d<d>-p<p>: 10,000 shots, file
          checked against the manifest by sha256 when --manifest is given), 50 untimed warm-up decodes, then one
          perf_counter reading around each matcher.decode(syndrome[, enable_correlations=True]) (surface_v2/timing/
          cpu_term.py `loop`, unchanged), pinned to the session's timed core (--core). The record: the mean per shot in
          microseconds (the value the fit uses; a shot is d rounds), the median and 99th percentile, the failures
          against the drawn observable (a sanity figure), the digests of circuit and shots, the host and the versions;
          written once, never over an existing file.
    python -m speed_rt.gref circuits [--write]
    python -m speed_rt.gref time --dec corr|plain --d 21 --p 0.006 --shots DIR --rep 1 --core C --out DIR [--manifest M]
                                 [--limit N]
-> DIR/ref_<dec>_d<d>_p<3|6>_rep<r>.json"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "surface_v2"))
CIRC = os.path.join(HERE, "circuits")
MANIFEST = os.path.join(CIRC, "ref_circuits.json")
DS = (5, 9, 13, 17, 21, 31)
PS = (0.003, 0.006)
WARMUP = 50
SHOTS = 10000


def ptag(p):
    """0.003 -> "3", 0.006 -> "6" (the p in tenths of a percent)."""
    return str(int(round(p * 1000)))


def set_name(d, p):
    return f"srt-ref-d{d}-p{ptag(p)}"


def shots_file(d, p, j=0):
    return f"ref_d{d}_p{ptag(p)}_{j:03d}.npz"


def circuit_file(d, p):
    return f"ref_d{d}_p{ptag(p)}.stim"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def manifest():
    return json.load(open(MANIFEST))


def circuit(d, p, man=None):
    """The reference circuit of (d, p), checked against ref_circuits.json's sha256 (and round-tripped through Stim's
    text, as surface_v2/convlines/circuits.circuit does)."""
    import stim
    e = (man or manifest())["circuits"][set_name(d, p)]
    path = os.path.join(CIRC, e["file"])
    text = open(path).read()
    assert sha256_bytes(text.encode()) == e["sha256"], f"{path}: not the registered circuit"
    c = stim.Circuit(text)
    assert str(c) == text, f"{path}: the Stim text does not round-trip"
    return c, e["sha256"]


def _builder():
    spec = importlib.util.spec_from_file_location("make_circuits", os.path.join(ROOT, "evaluation", "scripts",
                                                                                 "make_circuits.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_circuits(a):
    builder = _builder()
    from convlines import circuits as C
    os.makedirs(CIRC, exist_ok=True)
    out, ok = {}, True
    for d in DS:
        for p in PS:
            text = str(builder.memory_circuit(d, p).stim_circuit)
            name = set_name(d, p)
            f = os.path.join(CIRC, circuit_file(d, p))
            sha = sha256_bytes(text.encode())
            if a.write:
                with open(f, "w") as fh:
                    fh.write(text)
            same = os.path.exists(f) and open(f).read() == text
            repo = None
            if p == 0.006 and d in C.CIRCUIT_SHA256:                 # the repository's own circuit of this (d, p)
                repo = C.CIRCUIT_SHA256[d] == sha
            ok &= same and repo is not False
            out[name] = dict(d=d, p=p, file=circuit_file(d, p), sha256=sha, bytes=len(text.encode()),
                             repo_circuit_equal=repo)
            print(f"{name}: sha256 {sha} {'equal' if same else 'DIFFERENT'} to {circuit_file(d, p)}"
                  + ("" if repo is None else f"; the repository's d{d}_legacy.stim {'equal' if repo else 'DIFFERENT'}"),
                  flush=True)
    if a.write:
        with open(MANIFEST, "w") as fh:
            json.dump(dict(what="SPEED-RT surface stage: the Grace fit's reference circuits (NVIDIA Ising-Decoding "
                                "3ce4379 MemoryCircuit, legacy single-parameter noise, built by speed_rt/gref.py "
                                "circuits from evaluation/scripts/make_circuits.py)", circuits=out), fh, indent=1,
                      sort_keys=True)
            fh.write("\n")
    else:
        ok &= json.load(open(MANIFEST))["circuits"] == out
    raise SystemExit(0 if ok else 1)


def cmd_time(a):
    import numpy as np
    import pymatching
    import stim
    from convlines import circuits as C, provenance
    from timing import cpu_term as CT                      # the release's loop and pin, unchanged
    corr = a.dec == "corr"
    man = manifest()
    c, csha = circuit(a.d, a.p, man)
    t0 = time.perf_counter()
    dm = C.dem(c)
    m = pymatching.Matching.from_detector_error_model(dm, enable_correlations=True) if corr else \
        pymatching.Matching.from_detector_error_model(dm)
    build_s = time.perf_counter() - t0
    f = os.path.join(a.shots, shots_file(a.d, a.p))
    raw = open(f, "rb").read()
    fsha = sha256_bytes(raw)
    if a.manifest:
        e = json.load(open(a.manifest))["files"][shots_file(a.d, a.p)]
        assert e["sha256"] == fsha, f"{f}: sha256 differs from the manifest"
    z = np.load(io.BytesIO(raw))
    det = np.unpackbits(z["det"], axis=1)[:, :c.num_detectors]
    obs = z["obs"].astype(np.uint8)
    if a.limit:
        det, obs = det[:a.limit], obs[:a.limit]
    det = np.ascontiguousarray(det, dtype=np.uint8)
    assert det.shape[1] == m.num_detectors, f"{f}: not d = {a.d}'s detectors"
    rec = dict(what="ref", decoder=a.dec, d=a.d, p=a.p, set=set_name(a.d, a.p), rep=a.rep, core=a.core,
               cpu=platform.processor() or platform.machine(), python=sys.version.split()[0], numpy=np.__version__,
               stim=stim.__version__, pymatching=pymatching.__version__, load1=os.getloadavg()[0],
               start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code_digest=provenance.code_digest(),
               gref_sha256=sha256_file(os.path.abspath(__file__)), circuit_sha256=csha, shots_file=shots_file(a.d, a.p),
               shots_file_sha256=fsha, limit=a.limit or None, detectors=int(c.num_detectors),
               dem_errors=int(dm.num_errors), build_s=build_s)
    try:
        rec["cpu"] = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
    except (OSError, IndexError):
        pass
    rec["pinned"] = CT.pin(a.core)
    w0, c0 = time.perf_counter(), time.process_time()
    ts, dec = CT.loop(m, det, corr, WARMUP)
    rec.update(loop_wall_s=time.perf_counter() - w0, loop_cpu_s=time.process_time() - c0, shots=len(det),
               per_shot_us=float(ts.mean()), median_us=float(np.median(ts)), p99_us=float(np.percentile(ts, 99)),
               max_us=float(ts.max()), failures=int((dec != obs[:len(dec)]).sum()),
               end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"{rec['set']} {a.dec} rep {a.rep}: {rec['per_shot_us']:.3f} us per shot (median {rec['median_us']:.3f}, "
          f"{rec['failures']} failures of {rec['shots']}; build {build_s:.1f} s, loop {rec['loop_wall_s']:.1f} s)",
          flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"ref_{a.dec}_d{a.d}_p{ptag(a.p)}_rep{a.rep}.json"), "x") as fh:
        json.dump(rec, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("circuits")
    c.add_argument("--write", action="store_true")
    t = sub.add_parser("time")
    t.add_argument("--dec", choices=("plain", "corr"), required=True)
    t.add_argument("--d", type=int, required=True, choices=DS)
    t.add_argument("--p", type=float, required=True, choices=PS)
    t.add_argument("--shots", required=True, help="the directory of the drawn reference chunks")
    t.add_argument("--rep", type=int, required=True)
    t.add_argument("--core", type=int, default=-1)
    t.add_argument("--out", required=True)
    t.add_argument("--manifest", default="", help="the draw's srt-ref.json: the chunk's sha256 is checked against it")
    t.add_argument("--limit", type=int, default=0, help="dry runs only: the first N shots")
    a = ap.parse_args()
    {"circuits": cmd_circuits, "time": cmd_time}[a.what](a)


if __name__ == "__main__":
    main()
