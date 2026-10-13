"""Launcher of SPEED-RT's Cascade timing processes on BB72 (R1-R3 and G10): one command of the pinned tree's harnesses
(pc_speed.py, m2.py, d1_m2.py; ising_compare, qldpc_neural and the harness folders on PYTHONPATH), unchanged, with the
three changes of srtlib.py made first: beacon seeds for kind 12, a sha256 trace of every sampled chunk, and a burn plus an
in-process clock window in front of every timed window.

    python srt_cas.py SCRIPT.py ARGS...      SCRIPT.py as the job files call it, e.g.
        srt_cas.py $D1_TREE/cascade_compare/phase_d/d1_m2.py pipeline --setting depspam_fm --config -1 --p 0.002 ...
        srt_cas.py $D1_TREE/cascade_compare/phase_c/pc_speed.py pipeline --locked LOCK --config 2 --p 0.002 ...
        srt_cas.py $D1_TREE/cascade_compare/batchopt/m2.py nets --net S55 --engine new --rep 1 --out ROWS ...
    python srt_cas.py --check                the patch installs and the seeds are the beacon's (prints the facts)

Why the script runs as an imported module and not as __main__: the harnesses sample shots and decode in spawn pools
(Python's "spawn" start method, which re-runs the parent's __main__ file as __mp_main__ in every worker before a task is
unpickled). Here __main__ is this file, so every worker executes the top level below and patches its own copy of
bb.cascade_data before it samples, and the pools' functions (pc_speed._relay_init, m2.*) pickle by module name.
Needs SRT_SEEDS and SRT_BEACON (srtlib.py); the SRT_* variables of srtlib.py are read from the environment, so the
workers inherit them.

Card: SRT_CARD = rtx (default: the harnesses' own guard, which admits the RTX PRO 6000 only) or h100 (drive.py --card h100):
then, after the harness is imported, phase C's card guard (pc_common.gpu_env, which every GPU step of the BB harnesses
calls through pc_speed.setup) admits exactly "NVIDIA H100 80GB HBM3" (a MIG slice's name differs, so it is refused) and
labels every row with the card ("card", "card_role"); nothing else changes."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srtlib  # noqa: E402

FACTS = srtlib.install_bb()                    # at import: the main process and every spawned worker
CARD = os.environ.get("SRT_CARD", "rtx").lower()
H100 = "NVIDIA H100 80GB HBM3"
if CARD not in ("rtx", "h100"):
    raise SystemExit(f"srt_cas: SRT_CARD={CARD!r}, not rtx or h100")
# the harnesses' entry points: script stem -> function of the imported module (d1_m2 replaces m2's lock reader on import
# and leaves m2.main to be called)
ENTRY = {"pc_speed": lambda m: m.main, "m2": lambda m: m.main, "d1_m2": lambda m: m.m2.main}


def admit_h100():
    """pc_common.gpu_env for the H100 session: the same record as the original (gpu, driver, torch) plus the card's label;
    any other card, a MIG slice included, is refused."""
    import pc_common

    def gpu_env(dev):
        import torch
        name = torch.cuda.get_device_name(dev)
        if name != H100:
            raise SystemExit(f"GPU {name!r}: SRT_CARD=h100 admits an {H100} only")
        return {"gpu": name, "driver": os.environ.get("PC_GPU_DRIVER", ""), "torch": torch.__version__,
                "card": "h100", "card_role": "H100 (registered, speed)"}

    pc_common.gpu_env = gpu_env                            # pc_speed.setup() looks it up here at every call


def main(argv):
    if argv == ["--check"]:
        print(f"srt_cas: eval_seed patched for kind {FACTS['kind']}: sets {FACTS['sets']}, card {CARD}", flush=True)
        return
    if not argv or not argv[0].endswith(".py") or not os.path.isfile(argv[0]) or \
            os.path.basename(argv[0])[:-3] not in ENTRY:
        raise SystemExit(f"usage: srt_cas.py SCRIPT.py ARGS... (SCRIPT in {sorted(k + '.py' for k in ENTRY)}) | --check")
    script = os.path.abspath(argv[0])
    stem = os.path.basename(script)[:-3]
    print(f"srt_cas: {stem} with beacon seeds for kind {FACTS['kind']} (sets {len(FACTS['sets'])}), card {CARD}, "
          f"burn {srtlib.burn_seconds():g} s before every timed window", file=sys.stderr, flush=True)
    sys.argv = [script] + list(argv[1:])
    sys.path.insert(0, os.path.dirname(script))             # as `python SCRIPT.py` puts it
    mod = importlib.import_module(stem)
    if CARD == "h100":                                      # after the import: m2.py's own M2_CARD guard must not win
        admit_h100()
    ENTRY[stem](mod)()


if __name__ == "__main__":
    main(sys.argv[1:])
