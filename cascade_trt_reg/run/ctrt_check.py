"""CASCADE-TRT-REG's decision check (reported, not a gate): code/trt_check.py unchanged, on this test's beacon shots. After the
changes of ctrt_common.py (this test's sets), srtlib's shot streams are installed for BB72 and d = 13 (SRT_SEEDS, SRT_BEACON;
every sampled chunk's digest goes to SRT_TRACE), and the check samples seed kind 12, srtlib's beacon seeds (chunks 10-19 of
each Cascade's two sets, trt_check.CHUNKS), in place of the exploratory run's kind 47; its record's "kind" says 12.

    python ctrt_check.py ARGS...    trt_check.py's arguments, e.g. --net bb --engine trt --plans PL --ckpt CK --save F.npz

The spawned sampling workers re-run this file's top level, so each installs the same streams before it samples."""
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctrt_common as cc  # noqa: E402,F401  (this test's sets for srtlib)
import srtlib  # noqa: E402

FACTS = {"bb": srtlib.install_bb(), "d13": srtlib.install_d13()}
import trt_alone as ta  # noqa: E402  (code/: the check's shots come from its shots())
import trt_check as tk  # noqa: E402

ta.KIND = 12
ta.shots = functools.partial(ta.shots, kind=12)

if __name__ == "__main__":
    print(f"ctrt_check: beacon seeds for kind 12, sets {FACTS['bb']['sets']} {FACTS['d13']['sets']}", file=sys.stderr,
          flush=True)
    tk.main()
