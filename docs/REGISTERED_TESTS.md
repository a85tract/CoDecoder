# Registered tests

Part of the [CoDecoder release](../README.md).

## D9-SHALLOW: the network at d = 9

SPEED-RT found plain PM faster than SURF-v2's three-block network at d = 9, 26.297 against 28.927 µs. At small
distances the network's depth sets its time, so a one-block network of width 64 (`d9_b1w64_sh`, 90,467 parameters)
was trained on the same training shots in the same order and chosen by a rule fixed before its results existed.
D9-SHALLOW (`d9_shallow/PLAN.md`) then registered a test of its composition with correlated PM at q = 6 % against plain and
correlated PM on 1,000,000 fresh shots, and timed the composition in its own session on an H100 and the same workstation,
with a Grace factor fitted in that session (1.5423). The composition failed 11,978 times against 27,028 for plain PM and
12,802 for correlated PM, in 21.234 µs per shot against 26.297 and 63.916 µs (`d9_shallow/RESULTS.md`). Under the
publication rule frozen in its plan, this composition is the paper's d = 9 row, and SURF-v2's three-block network stays in
the records.

## CASCADE-TRT-REG: Cascade on TensorRT

CASCADE-TRT-REG (`cascade_trt_reg/RULES.md`) re-timed every Cascade comparison of stage 1 with Cascade on the faster
of two engines, stage 1's compiled path and a TensorRT 10.16.1.11 engine (BF16, builder level 3) in a CUDA graph. It
timed Cascade alone on both engines, three fresh processes each, on an H100 80GB HBM3 SXM and on an RTX PRO 6000
Blackwell Server Edition of the cluster's partition `artxpro6000`, the host its rules name, and ran each composition on the
engine with the lower Cascade-alone time. A composition is the slowest of its three processes and Cascade alone the fastest
of its three. The paper's appendix "Our Cascades on TensorRT" reports it.

- TensorRT ran Cascade alone 3.1 to 4.7 times faster per single shot on BB72 and 1.3 to 1.4 times faster at d = 13.
  In batches it ran the d = 13 Cascade 1.8 and 1.1 times faster on the two cards and the BB72 Cascade at batch 4,096
  slower, so the batched BB72 comparisons keep the compiled path.
- Every composition still took less time than Cascade alone on its faster engine: per single shot 2.53 to 4.20 times less on
  the RTX PRO 6000 and 2.38 to 4.33 times less on the H100, and in batches 3.15 to 16.36 and 2.42 to 8.16 times less.
- The frozen analysis withholds the three RTX single-shot comparisons at p = 0.3 %, because some of Cascade alone's
  three TensorRT processes were not established within three attempts (two of the three for the comparisons with the
  six-block network, one of the three for the three-block network).
- The engines decide alike: on fresh shots at most 47 of 200,000 batched decisions differ, the failure counts differ
  by at most 3 per comparison, and every exact McNemar p is at least 0.375.

Its registration is in the table of the next section. `cascade_trt_reg/RESULTS.md`, `cascade_trt_reg/results_alpine/`
(`tables.md`, `numbers.json`) and `cascade_trt_reg/records_alpine/` give the results by card, the decision checks and
the engines' sha256; the frozen analysis ran twice on the records with byte-identical outputs. An earlier RTX PRO 6000
session on rented RunPod pods, run while the cluster had no free card, stays published in `cascade_trt_reg/records/`
and `cascade_trt_reg/results/`; the paper does not use it.

## Registered tests and how to verify them

RETEST and SURF-v2, the two accuracy tests, each fixed everything their results depend on (the networks, their
thresholds, the rule that makes the test seeds, and the analysis) in a freeze commit and pushed it to GitHub before any
test shot existed. GitHub records the push in the repository's activity list. The test seeds come from the first round
of the drand public randomness beacon (mainnet, one round every 30 s) at least 600 s after that push, so nobody could
know the test shots when the freeze was made. Each test set was then drawn once, read once by every decoder and analysed
once by the frozen analysis.

RETEST (`RETEST.md`) re-ran the BB tests of Table I, both Cascade comparisons and the output-head comparison on fresh
shots, with the networks, thresholds, decoders and analyses frozen for their original runs; it timed nothing again, and
SPEED-RT re-timed its speed numbers.
SURF-v2 (`surface_v2/surf_v2/REGISTRATION.md`) tested the surface-code networks retrained without line supervision
(trained on the observable's flip alone) at d = 7 to 17 against eight baselines, with the deployed networks read
beside them. It also registered a secondary comparison with gap switching
(`surface_v2/surf_v2/GAPSW_REGISTRATION.md`), in which the composition failed less often than gap switching at every distance.

| test | freeze commit | GitHub push record of the freeze | beacon round | analysis commit | results |
|---|---|---|---|---|---|
| RETEST | `dd332c5` | `push` to `main` from `ab5cf49` to `dd332c5`, 2026-10-13T02:34:48Z, record 46295757992 | 6547616, released 2026-10-13T02:45:00Z; the record, the seeds and the draw manifests are in `390f2da` | one per test: `7c1aae0`, `d92e011`, `a3bd87d`, `4a85d65`, `87a1742`, `301e9e7`; the summary `3cc1258` | `RETEST_RESULTS.md`; each test's records and analysis outputs in `retest/run/` |
| SURF-v2 | `f94f4ed` | `branch_creation` of `surf-v2` at `f94f4ed`, 2026-10-13T11:49:59Z, record 46339453746 | 6548726, released 2026-10-13T12:00:00Z; the record and the test manifest are in `ce8780e` | `88d14de`; gap switching `3b41af7` | `surface_v2/surf_v2/analysis/tables.md`, `surface_v2/surf_v2/gapsw/analysis/gapsw_tables.md` |
| D9-SHALLOW | `31300ee` | `push` to `main`, 2026-10-14T10:17:57Z, record 46454694871 | 6551422; the record and the shot manifests are in `445e693` (`d9_shallow/testsets/beacon_d9s.json`) | `68947ee` | `d9_shallow/RESULTS.md`, `d9_shallow/results/` |
| CASCADE-TRT-REG | `55a3464` | `push` to `main`, 2026-10-14T17:39:38Z, record 46500379570 | 6552306; the record and the seed manifest are in `3896ac6` (`cascade_trt_reg/beacon_ctrt.json`) | `cc2d76b` | `cascade_trt_reg/RESULTS.md`, `cascade_trt_reg/results/` |

- The push records are GitHub's JSON as its API returned it: `retest/run/github_push_activity.json`, and
  `surface_v2/surf_v2/beacon.json`, which keeps the record together with the whole activity list.
- `RETEST.md` as frozen is `git show dd332c5:RETEST.md`. The summary commit `3cc1258` recorded in it the run's one
  deviation, the job shape of the gross-code reads, and corrected or completed several commands and descriptions;
  `git diff dd332c5 3cc1258 -- RETEST.md` shows every change. After `f94f4ed`, SURF-v2's commits only add files.
- `surface_v2/surf_v2/REGISTRATION.md` names the branch `surf-v2`, to which the freeze was pushed (the
  `branch_creation` record above). `main` holds all of that branch's commits, and the branch was deleted on 14 October
  2026; GitHub's activity list keeps the `branch_creation` record that step 1 below reads.
- The plans of the earlier registered tests are listed in the Table XII row of [Tables and figures](REPRODUCE.md#tables-and-figures).
- SPEED-RT follows the same beacon rule for its timing shots. Its commits, push records and rounds are in
  [Stages, freezes and beacons](SPEED.md#stages-freezes-and-beacons), and steps 1 to 3 below apply to them as well;
  [How to verify](SPEED.md#how-to-verify) checks its two beacon records offline.

### Checking a registration

The steps check SURF-v2 from `surface_v2/` (paths are relative to it) in the pinned environment of
[Installation](REPRODUCE.md#installation), with the GitHub CLI `gh` and network access to drand. Each step ends with the same check
for RETEST, run from the repository root.

1. Read the push record.

   ```bash
   cd surface_v2
   gh api 'repos/a85tract/CoDecoder/activity?per_page=100' > activity.json    # one page: under 100 records
   jq '.[] | select(.after == "f94f4edb843b4a0b4762c556d79782cf2a2167e2") | {id, activity_type, ref, timestamp}' activity.json
   ```

   The earliest record whose `after` is the freeze commit gives the push time T, so a later push of the same commit
   cannot move it: record 46339453746, `branch_creation` of `refs/heads/surf-v2`, 2026-10-13T11:49:59Z, the same
   record as `push_activity` in `surf_v2/beacon.json`. RETEST: select `dd332c5082f44c41d486327400d90c73c77c926a`, which
   gives record 46295757992, `push` to `refs/heads/main`, 2026-10-13T02:34:48Z.

2. Derive the drand round: the first round whose time, 1595431050 + 30 (round − 1) in UNIX seconds, is at least
   T + 600. T = 1791892199 (`date -u -d 2026-10-13T11:49:59Z +%s`) gives round 6548726, released at 1791892800
   (12:00:00 UTC). RETEST: T = 1791858888 gives round 6547616, released at 1791859500 (02:45:00 UTC).
   `convlines.beacon.beacon_round` computes the same.

3. Check the round against drand's public API. `convlines.beacon fetch` derives T and the round from your copy of the
   activity list, fetches the round from four drand endpoints, requires at least two answers that agree, checks them
   and writes a new record, whose round, signature and randomness must equal the committed record's:

   ```bash
   python -m convlines.beacon fetch --commit f94f4edb843b4a0b4762c556d79782cf2a2167e2 --activity activity.json \
       --push-source "verification" --out beacon_check.json
   jq -c '{round, randomness, signature}' surf_v2/beacon.json beacon_check.json
   ```

   The randomness is the SHA-256 of the round's signature, `930cdbaf...` for round 6548726, and drand serves the round
   without our code at
   `https://api.drand.sh/8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce/public/6548726`. RETEST:
   `python retest/beacon.py --push-time 2026-10-13T02:34:48Z --commit dd332c5082f44c41d486327400d90c73c77c926a --out
   rt_beacon_check.json` fetches and checks round 6547616, to compare with `retest/run/beacon.json` (randomness
   `299266d1...`).

4. Re-derive the seeds and check them against the test manifest.

   ```bash
   python -m convlines.beacon check --beacon surf_v2/beacon.json
   python -m convlines.testset seeds --set sv2test --approval surf_v2/beacon.json
   ```

   The first checks the committed record offline: the chain, T as the earliest activity record of the freeze commit,
   the round, the endpoints' agreement, and randomness = SHA-256(signature). The second checks that
   `testsets/sv2test.json` names this record by its sha256 and re-derives all its seeds, 200 chunks of 10,000 shots at
   d = 7 and 100 at each other distance ("700 of 700 seeds re-derived"). RETEST: `python retest/seeds.py --verify
   retest/run/beacon.json retest/run/seeds.json` recomputes every set's seeds from the beacon, `retest/tags.json` and
   the historical seeds and prints `"equal": true`, and it prints the same with step 3's `rt_beacon_check.json` in
   place of the committed record. The draw manifests `retest/run/bb/manifest_bb72.csv`, `manifest_gross.csv` and
   `retest/run/head/rows.csv` list each chunk's seed and sha256.

5. Regenerate a test chunk and compare its sha256.

   ```bash
   python -m convlines.testset regen --set sv2test --d 9 --chunks 0 --out shots/sv2test
   ```

   It prints `d9_p0.006_X_sv2test_000.npz: seed 5867115360100734495 sha256 8c7f5d84... == manifest`. Bit-for-bit
   regeneration needs Stim 1.16.0 and numpy 2.2.6 on an x86-64 CPU with AVX2. RETEST's BB chunks are rebuilt through its
   shim, as `retest/jobs/bb_draw.sbatch` does: `retest/shim/rt_bb.py bb.shots ...` with `RT_BEACON` and `RT_SEEDS` set
   to `retest/run/beacon.json` and `retest/run/seeds.json`; the BB circuits come from Relay-BP's repository
   (Installation).

6. Rerun the analysis and compare it byte for byte.

   ```bash
   python -m surf_v2.analysis --freeze surf_v2/freeze.json --reads READS... --shots shots/sv2test \
       --cal-shots shots/sv2cal --sessions SESSION --cpu CPU --out run1
   diff -r run1 surf_v2/analysis
   ```

   Nothing in the outputs depends on the time, the host or the output directory, so `diff` must report only
   `sessions.log`, a one-line note of the timing session's pod and GPU that the analysis does not write. The test and
   calibration sets are rebuilt with `regen` (`--set sv2cal` for the calibration set). The other inputs (every decoder's
   per-shot reads, the timing session and the CPU-term records) are archived by the authors and are not in this
   repository; `surf_v2/analysis/analysis.json` lists each of them with its sha256 under `inputs`, and
   `surf_v2/args/alpine_analysis.txt` holds the command as it ran. The gap-switching analysis runs the same way
   (`surf_v2/args/alpine_gapsw_analysis.txt`), and `RETEST.md` section 3 gives each RETEST analysis command.

### NVIDIA's Ising decoders

The tables' Ising M1 and Ising M4 are NVIDIA's public pre-decoders Ising-Decoder-SurfaceCode-1-Fast (receptive field 9)
and Ising-Decoder-SurfaceCode-1-Accurate (receptive field 13). Each is followed by plain or by correlated matching on
the detectors it leaves, which gives the rows Ising M1 / M4 + PM and Ising M1 / M4 + correlated PM.

- Accuracy comes from Ising-Decoding's official PyTorch numerics at commit 3ce4379: NVIDIA's own inference module, with
  fp32 weights and TF32 as NVIDIA's runner enables it (`surface_v2/convlines/opponents.py`). SURF-v2 read all four Ising
  rows, with plain and with correlated PM, on every test shot (`surface_v2/surf_v2/args/pod_e1_test.txt`); their
  failures are in `surface_v2/surf_v2/analysis/tables.md` and `family_A.csv`.
- Speed comes from SPEED-RT stage 2 (`speed_rt/STAGE2_surface.md` section 1.13). For each distance 12 TensorRT 10.14.1
  plans were built, six per model: FP16 and FP8 at optimisation levels 3 and 5, and NVIDIA's own build of each
  precision. The pod for d = 7, 9 and 11 built its plans after stage 2's freeze and then timed them. For d = 13, 15 and
  17 an earlier H100 pod built the plans before the freeze, from tree files that the frozen analysis checks against the
  frozen tree, saved every build output with its sha256 and was released. The timing pod, with the same GPU model and
  the same TensorRT, CUDA and torch versions, loaded the 134 saved files (the 133 outputs that
  `speed_rt/s2/surface/records/high/session/BUILD_INPUTS.json` lists with their sha256, and that file) after checking
  each file's sha256, and re-ran every plan's check, as section 1.13.7 registers. Every plan was checked on validation
  shots against the PyTorch numerics above before it was timed, an FP8 plan only if every convolution the recipe
  quantized ran an FP8 kernel. The plans were timed at batch 1 on the first timing shot, with CUDA-graph replays and
  with NVIDIA's `trtexec` command, and the fastest accepted repetition of any validated plan counts as Ising's GPU time.
  The composite adds the PM share on the detectors Ising leaves, timed on the Linux workstation and put on the Grace
  basis (rows S1 and S10 of `speed_rt/s2/surface/results/table.md`). SURF-v2's own speed comparisons used the earlier
  registered times of `results/surface/SURFACE.md` (section 4; `surface_v2/surf_v2/config.json`, `registered_times_us`),
  which SPEED-RT supersedes.

An Ising row's accuracy and its time thus come from two numerics of the same model, and no single run measured both.
The earlier surface-code tests also read every timed TensorRT plan for accuracy (`results/surface/s*_test_eval.csv`,
`r17_test_eval.csv`).
