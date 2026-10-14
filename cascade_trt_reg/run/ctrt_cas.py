"""CASCADE-TRT-REG's launcher of the BB72 Cascade timing processes (R1, R2): stage 1's srt_cas.py (speed_rt/cascade/), whose
main() this repeats, with the changes of ctrt_common.py made first (this test's sets for srtlib's beacon seeds, trace and
clock windows; the Cascade engine from CTRT_ENGINE; the public tree's patches, applied after the harness is imported).

    python ctrt_cas.py SCRIPT.py ARGS...      as srt_cas.py: SCRIPT.py = pc_speed.py or d1_m2.py of the tree, with its own
                                              arguments (drive's slots give them)

Why the top level matters: the harnesses sample in spawn pools, which re-run the parent's __main__ (this file) in every
worker before a task is unpickled, so each worker imports ctrt_common (this test's sets) and srt_cas (srtlib.install_bb at
its top level) and patches its own copy of bb.cascade_data before it samples."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctrt_common as cc  # noqa: E402  (before srt_cas: srtlib then reads this test's sets)
import srt_cas  # noqa: E402  (stage 1's launcher: srtlib.install_bb() at import, SRT_CARD)

cc.INFO["seeds"] = srt_cas.FACTS


def main(argv):
    if not argv or not argv[0].endswith(".py") or not os.path.isfile(argv[0]) or \
            os.path.basename(argv[0])[:-3] not in ("pc_speed", "d1_m2"):
        raise SystemExit("usage: ctrt_cas.py SCRIPT.py ARGS... (SCRIPT = pc_speed.py or d1_m2.py)")
    script = os.path.abspath(argv[0])
    stem = os.path.basename(script)[:-3]
    print(f"ctrt_cas: {stem}, Cascade engine {cc.ENGINE}, beacon seeds for kind {srt_cas.FACTS['kind']} "
          f"(sets {srt_cas.FACTS['sets']}), card {srt_cas.CARD}", file=sys.stderr, flush=True)
    sys.argv = [script] + list(argv[1:])
    sys.path.insert(0, os.path.dirname(script))
    mod = importlib.import_module(stem)
    cc.patch_bb(stem, mod)
    if srt_cas.CARD == "h100":
        srt_cas.admit_h100()
    srt_cas.ENTRY[stem](mod)()


if __name__ == "__main__":
    main(sys.argv[1:])
