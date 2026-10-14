"""CASCADE-TRT-REG's launcher of the d = 13 Cascade timing processes (R4): stage 1's srt_d13.py (speed_rt/cascade/) and its
main() unchanged (srtlib's beacon seeds and trace, the clock window around the pipeline's timed part, SRT_CARD's card
admission, on the H100 the session record and the lock derived from it), with the changes of ctrt_common.py made first:
this test's sets, the Cascade engine from CTRT_ENGINE, and the public tree's patches of surface.sv_common (identity reduced
to GPU and torch, the lock's cache digest).

    python ctrt_d13.py pipeline --config C5 --p 0.006 --mode single --rep 1 --lock LOCK --stage c7 --out ROWS --save DIR

Why the top level matters: as in srt_d13.py, the pipeline's spawn pools re-run this file as __mp_main__ in every worker, which
imports ctrt_common (this test's sets) and srt_d13 (srtlib.install_d13 at its top level) before it samples."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctrt_common as cc  # noqa: E402  (before srt_d13: srtlib then reads this test's sets)
import srt_d13  # noqa: E402  (stage 1's launcher: srtlib.install_d13() at import, SRT_CARD)

cc.INFO["seeds"] = srt_d13.FACTS


def main(argv):
    if not argv or argv[0] != "pipeline":
        raise SystemExit("usage: ctrt_d13.py pipeline ... (sv_speed's arguments, as srt_d13.py)")
    print(f"ctrt_d13: Cascade engine {cc.ENGINE}, card {srt_d13.CARD}", file=sys.stderr, flush=True)
    sv = importlib.import_module("surface.sv_speed")
    cc.patch_d13(sv, argv)
    srt_d13.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
