"""Replay launcher for the Relay-BP seed check (jobs/head_replay.sh; replay only). Imports the shim (its patch installs on
import, as in every spawned worker), wraps relay_bp.RelayDecoderF32 so that every process records the seed it builds a
decoder with, and runs a1.fallback.relay (the function a1.fallback's `relay` command calls) on 200 dry shots: the first
100 of native chunks 0 and 1. Spawned workers re-run this file as __mp_main__: the imports and the recorder are at top level,
the run only under __main__. Run with bb's venv (relay-bp 0.2.2); RT_SEEDS, RT_BEACON, RT_OUT, RT_DRY and RT_REPLAY_RECORD set."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "shim"))
sys.path.insert(0, HERE)
import rt_a1                                  # noqa: E402,F401  (the patch, at import)
import head_replay                            # noqa: E402

head_replay.install_recorder(os.environ["RT_REPLAY_RECORD"])

if __name__ == "__main__":
    head_replay.relay(os.environ["RT_REPLAY_RECORD"])
