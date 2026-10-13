"""Classical decoders: the teachers and the verdict references (SETTINGS.md section 5, brief §2.3).

Each decoder maps the detector bits of its sector to a fault estimate e; the predicted observables are L e mod 2.
All run single-threaded; decode() times every shot separately (seconds of wall time on one core).
"""
import time

import numpy as np

from qnd import circuits

# name -> (library, parameters). 'sector' is the detector set the decoder reads ('all' = both, 'basis' = xyz=False).
CONFIGS = {
    # teacher candidates (brief §2.3)
    "bposd_cs10": ("ldpc.BpOsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=0.75, max_iter=30,
                                              schedule="parallel", osd_method="osd_cs", osd_order=10)),
    "bplsd0": ("ldpc.BpLsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=0.75, max_iter=30,
                                          schedule="parallel", lsd_method="lsd_cs", lsd_order=0)),
    # Relay-BP with the parameters its authors tuned for the gross code (examples/BivariateBicycleCodeAnalysis.ipynb)
    "relay": ("relay_bp.RelayDecoderF32", dict(alpha=1.0, gamma0=0.15000000000000002, pre_iter=80, num_sets=300,
                                               set_max_iter=60, gamma_dist_interval=(-0.22628432386414646,
                                                                                     0.6216020925981884),
                                               stop_nconv=5)),
    # stronger BP-OSD references (brief §2.3)
    "bposd_cs60_ms60": ("ldpc.BpOsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=0.75, max_iter=60,
                                                   schedule="parallel", osd_method="osd_cs", osd_order=60)),
    # the published configuration's BP (min-sum, scale 1.0, 80 iterations) with OSD-CS order 60
    "bposd_ms80_cs60": ("ldpc.BpOsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=1.0, max_iter=80,
                                                   schedule="parallel", osd_method="osd_cs", osd_order=60)),
    # lower-iteration / lower-order challengers for criterion (b) (review #32.6)
    "bposd_ms80_osd0": ("ldpc.BpOsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=1.0, max_iter=80,
                                                   schedule="parallel", osd_method="osd0", osd_order=0)),
    "bposd_ms30_osd0": ("ldpc.BpOsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=1.0, max_iter=30,
                                                   schedule="parallel", osd_method="osd0", osd_order=0)),
    "bposd_ms30_cs10": ("ldpc.BpOsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=1.0, max_iter=30,
                                                   schedule="parallel", osd_method="osd_cs", osd_order=10)),
    "bposd_gu": ("ldpc.BpOsdDecoder", dict(bp_method="product_sum", max_iter=30, schedule="parallel",
                                            osd_method="osd_cs", osd_order=60)),
    # the published configuration of Relay-BP's sweep (stimbposd defaults: ldpc ms_scaling_factor 1.0)
    "published_bposd": ("ldpc.BpOsdDecoder", dict(bp_method="minimum_sum", ms_scaling_factor=1.0, max_iter=80,
                                                   schedule="parallel", osd_method="osd_cs", osd_order=10)),
    # uncorrelated MWPM on the decomposed DEM (surface-code teacher / reference, ASSUMPTIONS Q29)
    "mwpm": ("pymatching.Matching", dict()),
    # the same through stimbposd itself (beliefmatching's check matrices), i.e. the published code path
    "published_stimbposd": ("stimbposd.BPOSD", dict(max_bp_iters=80, bp_method="minimum_sum", osd_order=10,
                                                    osd_method="osd_cs")),
}


def describe(name: str, sector: str) -> str:
    import ldpc
    import relay_bp  # noqa: F401
    lib, kw = CONFIGS[name]
    return f"{name}: {lib}({', '.join(f'{k}={v}' for k, v in kw.items())}) on DEM sector={sector}, ldpc {ldpc.__version__}"


class Decoder:
    def __init__(self, name: str, code: str, basis: str, p: float, sector: str):
        self.name, self.sector = name, sector
        H, L, q, keep = circuits.problem(code, basis, p, sector)
        self.keep, self.L = keep, L.tocsr().astype(np.uint8)
        lib, kw = CONFIGS[name]
        H = H.tocsr()
        if lib == "ldpc.BpOsdDecoder":
            from ldpc import BpOsdDecoder
            self.dec = BpOsdDecoder(H, error_channel=list(q), omp_thread_count=1, **kw)
        elif lib == "ldpc.BpLsdDecoder":
            from ldpc import BpLsdDecoder
            self.dec = BpLsdDecoder(H, error_channel=list(q), omp_thread_count=1, **kw)
        elif lib == "relay_bp.RelayDecoderF32":
            import relay_bp
            self.dec = relay_bp.RelayDecoderF32(H.astype(np.uint8), error_priors=q.astype(np.float64), **kw)
        elif lib == "pymatching.Matching":
            import pymatching
            c = circuits.load(code, basis, p)
            self.dec = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
            self.keep = np.arange(c.num_detectors)
        elif lib == "stimbposd.BPOSD":
            import stimbposd
            c = circuits.load(code, basis, p)
            if sector == "basis":
                c = circuits.filter_by_basis(c, basis)
            self.dec = stimbposd.BPOSD(c.detector_error_model(decompose_errors=False), **kw)
        else:
            raise ValueError(lib)
        self.lib = lib

    def decode(self, det: np.ndarray):
        """det: (n, D_all) uint8 full-circuit detectors -> (obs_pred (n, k) uint8, seconds (n,), converged (n,))."""
        s = det[:, self.keep]
        n = len(s)
        if self.lib == "pymatching.Matching":           # batch decode; per-shot time = batch time / n
            t = time.perf_counter()
            out = self.dec.decode_batch(s).astype(np.uint8)
            return out, np.full(n, (time.perf_counter() - t) / max(n, 1)), np.ones(n, bool)
        out = np.zeros((n, self.L.shape[0]), np.uint8)
        secs = np.zeros(n)
        conv = np.zeros(n, bool)
        for i in range(n):
            si = np.ascontiguousarray(s[i])
            t = time.perf_counter()
            e = self.dec.decode(si)
            secs[i] = time.perf_counter() - t
            if self.lib == "stimbposd.BPOSD":   # returns the predicted observables
                out[i] = e
                conv[i] = bool(self.dec._bposd.converge)
            else:
                out[i] = (self.L @ e.astype(np.uint8)) % 2
            if self.lib.startswith("ldpc"):
                conv[i] = bool(self.dec.converge)
        return out, secs, conv
