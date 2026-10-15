# Checks and verification

Part of the [CoDecoder release](../README.md).

## Checks already run

- The Cascade comparison on BB72 was checked from fresh clones on the cluster (`cascade/bb72/REPRODUCTION_CHECK.md`):
  every manifest and archived per-shot file matched; the networks' GPU check passed; of 192 recomputed numbers, the 70
  that have a record all equal it, and of the 146 compared with the paper, one was not as printed (5.1 against 5.2, a
  speed-up's low end), which the paper has since corrected.
- R17's record check (`results/r17/REPRODUCTION_CHECK.md`) and a full re-run of the surface-code reproduction steps on a
  rented H100 (`results/reproduction/REPRODUCTION.md`).
- RETEST's replay, seed and platform checks before its freeze (`RETEST.md` section 6, records in `retest/replay/`).
- SPEED-RT's reruns: on 14 October 2026 (UTC) the commands of [How to verify](SPEED.md#how-to-verify) ran from a fresh clone of
  `main` at `e0a9377` on a Linux x86-64 machine with Python 3.12.3 and numpy 1.26.4. Both beacon records passed the
  offline check, stage 1's seeds file was re-derived byte for byte, and every output of the four analyses was
  byte-identical to the published one.
- The verification of this repository's first commit from a fresh clone (next section).

## Verification of this repository

On 12 October 2026 this repository's first commit, `ab5cf49`, was cloned fresh on the Alpine cluster (CU Boulder
Research Computing) and checked there with the commands above.

- Installation: a new virtual environment from `requirements.txt` with torch 2.11.0+cu128, which brought Triton 3.6.0
  and cuDNN 9.19, plus relay-bp 0.2.2 and ldpc 2.4.1. Every pin installed as listed. The nodes' Python was 3.12.12
  (pinned: 3.12.14), and the GPU node needed `C_INCLUDE_PATH` for the Python headers (Installation).
- Weights: all 45 files match `weights/MANIFEST.sha256`, and all 38 networks build, load strictly and have the
  parameter counts of `weights/networks.csv`. `convlines_train.check_frozen` confirmed that the frozen training
  programs rebuild the 16 networks of the surface chain, BB72, the gross code and the Cascade comparison on BB72.
- Code and documents: every Python file compiles, and the 42 bash blocks of `README.md`, `results/surface/SURFACE.md`,
  `bb/BB.md`, `cascade/bb72/CASCADE.md` and `training/TRAINING.md` parse.
- Records: `timing.composite` reproduced every registered composite time and speed cell. `recompute_numbers.py`
  recomputed the 192 numbers of the Cascade comparison on BB72: none of the 70 that have a record differs from it, and
  of the 146 compared with the paper's values when the script was written, the one not as printed is the 5.1 that the
  paper now prints. `results_tables.py` printed the d = 13 tables, and the three figure scripts redrew every figure.
- Surface code, on a dry set (T2 at d = 9, chunks 0 and 99): both chunks were rebuilt with the manifest's sha256, and the
  reference decoder failed 94 and 124 times, as the earlier release's verification recorded. On chunk 0,
  `convlines.evaluate` gave ours 94 failures against 233 for plain PM and 124 for correlated PM (better than both), and
  the deployed Triton engine on an H200 MIG slice gave the same 94 failures, with the routing of every shot equal to the
  published mask.
- SURF-v2's options in `evaluation/`: `convlines.evaluate --save` wrote every decoder's per-shot arrays and record for
  the same chunk, on the CPU and through the deployed engine, with the same counts and the network's sha256 equal to the
  manifest's; `convlines.beacon` checked BB72's beacon record offline and re-derived its 50 test seeds.
- BB72, on a dry set (test chunk 0 at p = 0.3 %, 10,000 shots): the shots were rebuilt from their seed, and the second
  network's read on an H200 MIG slice, through the commit guard, matched the published per-shot outputs exactly: all
  10,000 decisions equal and every margin bitwise equal.
- Training: `training/smoke.sh` (4·10^4 samples) and `convlines_train.train --bb` on BB72 (2·10^4 samples) ran on a
  CPU; the frozen BB program (16 steps of `bb.train`) and the frozen surface program (20 steps of the d = 7 stage, with
  `torch.compile`) ran on an H200 MIG slice.

## What is not included

- The test shots, and the full per-shot outputs of every test read except the BB tests' ([BB codes](REPRODUCE.md#bb-codes)). The
  shots are rebuilt from the manifests' seeds. The per-shot outputs of the surface-code tests (SURF-v2 included), the
  Cascade tests and the output-head study are archived by the authors.
- SPEED-RT's timing shots. Stage 2's are rebuilt from the beacon's seeds and checked against the manifests in
  `speed_rt/testsets/`, `speed_rt/bb/` and `speed_rt/gapsw/`; stage 1's processes sampled their chunks during the run
  and recorded each chunk's seed and sha256.
- The fp32 module of the d = 17 surface network: it is released as its deploy tables, which the deployed engine reads.
- NVIDIA's Ising-Decoding code and checkpoints, and Relay-BP's repository: fetch them from their public releases.
- The research project's internal plans that the frozen trees cite but that no result depends on, the review texts,
  and the logs of runs that no table reads.
