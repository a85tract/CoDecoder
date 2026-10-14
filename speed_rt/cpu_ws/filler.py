"""The harness's filler (speed_rt/cpu_ws/harness.py): plain PyMatching on validation chunk 0 at d = 9, decoded over and
over, untimed, until the harness kills it. It keeps a slot's core busy after the slot's timed process has ended, so that
the window's other timed processes run at the same load to their end.
    python filler.py VALDIR        (cwd: the checkout root; PYTHONPATH=<root>/surface_v2:<root>)"""
import os
import sys

import numpy as np
from convlines import circuits, testset

det, _ = testset.load(os.path.join(sys.argv[1], testset.name("val", 9, 0)), 9)
det = np.ascontiguousarray(det, dtype=np.uint8)
m = circuits.matcher(9)
while True:
    for s in det:
        m.decode(s)
