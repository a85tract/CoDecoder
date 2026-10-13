"""RelayFull (cal.py, used for the alternative noise models) must decode exactly as qnd.teachers.Decoder('relay', ...,
'all') on the frozen circuit: 300 shots at p = 0.003 from seed 2^50 + 10^9 + 10^6 + 999 (point 1, chunk 999, used only
here)."""
import numpy as np

import cal
from qnd import teachers

c = cal.circuit("uniform", 0.003)
det, obs = c.compile_detector_sampler(seed=cal.seed_of(1, 999)).sample(300, separate_observables=True)
det = det.astype(np.uint8)
a = teachers.Decoder("relay", "bb72", "X", 0.003, "all").decode(det)[0]
b = cal.RelayFull(c, teachers.CONFIGS["relay"][1]).decode(det)[0]
print("RelayFull == teachers.Decoder on 300 shots:", bool((a == b).all()), "failures", int(np.any(a != obs, 1).sum()))
