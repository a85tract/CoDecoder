"""A chunk's digest (sv_common.digest of a fresh Stim draw) in a worker process: importable, so that a spawn pool can
run it (arb2_seedcheck.py). Detector rows never leave the worker."""
from surface import surface_data as sdata
from surface import sv_common as sc


def chunk_digest(args):
    """(p, seed) -> sc.digest of the 10,000 shots of the d = 13 circuit at p drawn from seed."""
    p, seed = args
    det, obs = sdata.sample(sc.D, p, sc.CHUNK, int(seed))
    return sc.digest(det, obs)
