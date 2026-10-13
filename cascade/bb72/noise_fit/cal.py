"""The noise-model fit (exploratory, not registered): does our BB72 setting match Cascade's Fig. 2(a) (Gu, Bonilla Ataides, Lukin,
Yelin, arXiv:2604.08358)? Fresh shots of our circuits decoded by the reference decoders whose configuration the paper
states.

Circuits: the frozen BB72 X-basis memory over 6 rounds, qnd.circuits.load('bb72', 'X', p) (relay-bp's testdata at
d185194, sha256-checked against qldpc_neural/data/circuit_manifest.csv, uniform circuit noise), or the same schedule
under an alternative noise model (noise.py). Every decoder sees the full DEM (all 432 detectors,
decompose_errors=False) except the *_sector decoders.

Shots: chunk j of point i holds CHUNK = 1,000 shots sampled by stim with seed 2^50 + 10^9 + 10^6 i + j (POINTS lists
i). Decoders reading the same chunk on the same host read the same shots (stim's samples depend on the host's SIMD
width, so the host is recorded).

Decoders
  tesseract     tesseract-decoder's own 'tesseract-short-beam' (make_tesseract_sinter_decoders_dict(), "the parameters
                used in the Tesseract paper"): det_beam 15, beam_climbing, 16 detector orders (DetectorOrderMethod.Index,
                seed 2384753), pqlimit 200,000, no_revisit_dets, merge_errors, det_penalty 0
  bposd         stimbposd.BPOSD(dem) with its defaults (product-sum BP, 30 iterations, OSD-CS order 60)
  relay_full    the frozen BB72 Relay-BP (qnd.teachers 'relay': relay-bp 0.2.2 RelayDecoderF32, alpha 1, gamma0 0.15,
                pre_iter 80, 300 sets x 60 iterations, stop after 5 solutions) on the full DEM (other noise
                models: the same parameters on the rebuilt circuit's full DEM, class RelayFull)
  relay_sector  the same on the X-detector sector (relay's filter_detectors_by_basis)          [uniform model only]
  tesseract_sector, bposd_sector   diagnostic: Tesseract / stimbposd as above on that X-detector sector (252 detectors)
  tesseract_bfs  diagnostic: the short-beam preset with DetectorOrderMethod.BFS orders (same seed) instead of Index

  python cal.py --point 0 --dec tesseract --chunks 0-9 --out runs/tesseract_i0_a.jsonl
One JSON line per chunk (shots, failures, the failing shot indices, flagged shots, decode seconds) after a header line
(versions, configuration, circuit and DEM digests)."""
import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time

import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QND_PATH", os.path.join(HERE, "..", "..", "..", "qldpc_neural")))
from qnd import circuits  # noqa: E402

SEED0 = 2**50 + 10**9
CHUNK = 1000
# point index i -> (noise model, p); models other than 'uniform' are rebuilt by noise.py on the same schedule
POINTS = {0: ("uniform", 0.002), 1: ("uniform", 0.003), 2: ("uniform_fm", 0.002), 3: ("uniform_fm", 0.003),
          4: ("noidle_fm", 0.002), 5: ("noidle_fm", 0.003), 6: ("spam2_fm", 0.002), 7: ("spam2_fm", 0.003),
          8: ("si1000", 0.002), 9: ("si1000", 0.003), 10: ("depspam_fm", 0.002), 11: ("depspam_fm", 0.003)}
TESSERACT_SHORT_BEAM = dict(det_beam=15, beam_climbing=True, no_revisit_dets=True, merge_errors=True, pqlimit=200_000,
                            num_det_orders=16, det_order="Index", seed=2384753, det_penalty=0.0)


def circuit(model, p):
    if model == "uniform":
        return circuits.load("bb72", "X", p)
    import noise
    return noise.build(model, p)            # noise.py --check: its 'uniform' rebuild gives the frozen DEM exactly


def seed_of(i, j):
    return SEED0 + 10**6 * i + j


def parse_chunks(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


class RelayFull:
    """qnd.teachers.Decoder('relay', ..., 'all') for any circuit: the same RelayDecoderF32 parameters on this circuit's
    full DEM (qnd.circuits.dem_matrices, identical mechanisms merged), same decode interface."""

    def __init__(self, c, kw):
        import relay_bp
        H, L, q = circuits.dem_matrices(c.detector_error_model(decompose_errors=False))
        self.L = L.tocsr().astype(np.uint8)
        self.dec = relay_bp.RelayDecoderF32(H.tocsr().astype(np.uint8), error_priors=q.astype(np.float64), **kw)

    def decode(self, det):
        n = len(det)
        out = np.zeros((n, self.L.shape[0]), np.uint8)
        secs = np.zeros(n)
        for i in range(n):
            si = np.ascontiguousarray(det[i])
            t = time.perf_counter()
            e = self.dec.decode(si)
            secs[i] = time.perf_counter() - t
            out[i] = (self.L @ e.astype(np.uint8)) % 2
        return out, secs, np.ones(n, bool)


class Dec:
    """decode(det (n, D) bool) -> (pred (n, k) bool, flag (n,) bool, secs (n,))."""

    def __init__(self, name, model, p, c):
        self.name = name
        self.keep = None                                    # the decoder's detectors (None = all)
        if name in ("tesseract_sector", "bposd_sector"):    # diagnostic: the X-detector sector only, as relay_sector
            f = circuits.filter_by_basis(c, "X")
            self.keep = circuits.kept_detectors(c, f)
            c = f
        dem = c.detector_error_model(decompose_errors=False)
        self.dem = dem
        kind = name.split("_")[0] if name.startswith(("tesseract", "bposd")) else name
        if kind == "tesseract":
            from tesseract_decoder import tesseract as T
            from tesseract_decoder import utils as U
            cfg = dict(TESSERACT_SHORT_BEAM)
            if name == "tesseract_bfs":                     # diagnostic: the preset with BFS detector orders
                cfg["det_order"] = "BFS"
            cfg["det_order_method"] = getattr(U.DetectorOrderMethod, cfg.pop("det_order"))
            self.dec = T.TesseractConfig(dem, **cfg).compile_decoder()
            self.config = {k: str(v) if k == "det_order_method" else v for k, v in cfg.items()}
        elif kind == "bposd":
            import stimbposd
            from stimbposd import config as sc
            self.dec = stimbposd.BPOSD(dem)
            self.config = dict(max_bp_iters=sc.DEFAULT_MAX_BP_ITERS, bp_method=sc.DEFAULT_BP_METHOD,
                               osd_order=sc.DEFAULT_OSD_ORDER, osd_method=sc.DEFAULT_OSD_METHOD,
                               pcm_shape=list(self.dec._matrices.check_matrix.shape))
        elif name in ("relay_full", "relay_sector"):
            from qnd import teachers
            sector = "all" if name == "relay_full" else "basis"
            if model == "uniform":
                self.dec = teachers.Decoder("relay", "bb72", "X", p, sector)
            elif sector == "all":
                self.dec = RelayFull(c, teachers.CONFIGS["relay"][1])
            else:
                raise ValueError("sector Relay-BP runs on the frozen circuits only")
            self.config = dict(teacher="relay", sector=sector, kw={k: str(v) for k, v in teachers.CONFIGS["relay"][1].items()})
        else:
            raise ValueError(name)

    def decode(self, det):
        n = len(det)
        if self.name.startswith("relay"):
            pred, secs, _ = self.dec.decode(det.astype(np.uint8))   # teachers.Decoder reports no Relay convergence
            return pred.astype(bool), np.zeros(n, bool), secs
        pred = np.zeros((n, self.dem.num_observables), bool)
        flag = np.zeros(n, bool)
        secs = np.zeros(n)
        for s in range(n):
            x = det[s] if self.keep is None else det[s][self.keep]
            t = time.perf_counter()
            if self.name.startswith("tesseract"):
                pred[s] = self.dec.decode(x)
                flag[s] = self.dec.low_confidence_flag
            else:
                pred[s] = self.dec.decode(x.astype(np.uint8))
                flag[s] = not self.dec._bposd.converge
            secs[s] = time.perf_counter() - t
        return pred, flag, secs


def versions():
    import importlib.metadata as md
    out = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "host": socket.gethostname()}
    for pkg in ("stim", "tesseract-decoder", "stimbposd", "ldpc", "relay-bp", "numpy", "scipy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", type=int, required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sinter-check", type=int, default=0, help="Tesseract only: also decode the first N shots of each "
                    "chunk through the package's sinter 'tesseract-short-beam' and count disagreements")
    a = ap.parse_args()
    model, p = POINTS[a.point]
    c = circuit(model, p)
    dec = Dec(a.dec, model, p, c)
    sinter_dec = None
    if a.sinter_check:
        import tesseract_decoder as td
        sinter_dec = td.make_tesseract_sinter_decoders_dict()["tesseract-short-beam"].compile_decoder_for_dem(dem=dec.dem)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        head = dict(header=True, point=a.point, model=model, p=p, dec=a.dec, config=dec.config, chunk_size=CHUNK,
                    seed_rule="2^50 + 10^9 + 10^6 point + chunk", versions=versions(),
                    circuit_sha256=hashlib.sha256(str(c).encode()).hexdigest(),
                    dem=dict(detectors=dec.dem.num_detectors, errors=dec.dem.num_errors,
                             observables=dec.dem.num_observables), argv=sys.argv, t=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        f.write(json.dumps(head) + "\n")
        f.flush()
        for j in parse_chunks(a.chunks):
            seed = seed_of(a.point, j)
            det, obs = c.compile_detector_sampler(seed=seed).sample(CHUNK, separate_observables=True)
            t0 = time.time()
            pred, flag, secs = dec.decode(det)
            fail = np.any(pred != obs, axis=1)
            row = dict(point=a.point, model=model, p=p, dec=a.dec, chunk=j, seed=seed, shots=CHUNK,
                       failures=int(fail.sum()), fail_idx=np.flatnonzero(fail).tolist(), flagged=int(flag.sum()),
                       flagged_failures=int((flag & fail).sum()), secs=float(secs.sum()), max_secs=float(secs.max()),
                       wall=time.time() - t0, det_events=float(det.sum(1).mean()),
                       obs_digest=hashlib.sha256(np.packbits(obs).tobytes()).hexdigest()[:16])
            if sinter_dec is not None:
                m = min(a.sinter_check, CHUNK)
                packed = np.packbits(det[:m], axis=1, bitorder="little")
                sp = np.unpackbits(sinter_dec.decode_shots_bit_packed(bit_packed_detection_event_data=packed), axis=1,
                                   bitorder="little", count=obs.shape[1]).astype(bool)
                row["sinter_check"] = dict(shots=m, disagree=int(np.any(sp != pred[:m], axis=1).sum()))
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
