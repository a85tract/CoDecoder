# ARB at d = 13: arbitration on the routed shots of ConvLines + Cascade (surface code), exploratory

Exploratory, as PLAN.md fixes it: nothing was registered and nothing here is a claim. Every weight and threshold, and
the choice of the rule to carry forward, come from the fit half (even chunks). Every number below is from the
evaluation half (odd chunks) unless a table says otherwise.

## Summary

On fresh shots (4,000,000 at p = 0.6 % and 30,000,000 at p = 0.35 %; evaluation halves of 2,000,000 and 15,000,000),
the three rules gain little at d = 13, though the routed oracle leaves room for much more:

- J (joint decision) cuts the composition's failures from 10,064 to 9,949 at 0.6 % (529 repairs, 414 breaks, net +115,
  1.14 % fewer failures [0.54, 1.74], McNemar p = 0.0002), and from 1,178 to 1,172 at 0.35 % (35 repairs, 29 breaks,
  net +6, 0.51 % [-0.86, 1.88], p = 0.53). Its fitted weights are w = 7.866 at 0.6 % and w = 13.84 at 0.35 %.
- P and T cut failures by 0.58 % and 0.59 % at 0.6 % (p = 0.022 and 0.017) and by 0.68 % and 0.59 % at 0.35 %
  (p = 0.38 and 0.59). At 0.6 % both fits switch only where Cascade's |z| is below about 0.11, almost regardless of |c|
  (P: a = 0.107, b = 0.001, the grid's smallest nonzero b; T: t_C = 0.109, t_S = 0.072).
- The routed oracle would cut failures by 30.2 % and 29.2 %. The rules recover 1.7 to 3.8 % of its repairs, against
  about 60 % on BB72. On the disagreement rows the network is wrong about three times as often as Cascade (9,423
  against 3,040 at 0.6 %, 1,163 against 344 at 0.35 %).
- The fit-half rule carries P forward (mean fit-half relative reduction: P 1.64 %, T 1.42 %, J 1.23 %). On the
  evaluation half J's net is about twice P's and T's at 0.6 % (+115 against +58 and +59) and about the same at 0.35 %
  (+6 against +8 and +7).
- J's weight: the gain over w is a broad plateau. For w between 6 and 15 (16 grid values) the evaluation net stays at
  +97 to +125 at 0.6 % and 0 to +11 at 0.35 % (fit half: +35 to +72 and +7 to +19). The evaluation halves' own best
  w are 8.35 (+127) and 4.27 (+15). One w for both p (a diagnostic: the fit half's best mean) is w = 10.59, with
  evaluation nets of +111 (1.10 %) and +11 (0.93 %).
- For sizing a confirmation (rough arithmetic only): J's evaluation net per discordant shot is 0.12 at 0.6 %
  (115 of 943), so about 1,100,000 fresh shots would give 80 % power at the two-sided 5 % level. At 0.35 % it is 0.09
  on the evaluation half and 0.23 on the fit half, which puts the same power at about 210,000,000 or 26,000,000 shots.
- The reads reproduce the comparison's per-category rates (paper Table I): Cascade alone fails 5,065 and 78.1 times
  per 10^6 shots (Table I: 4,936 and 78.5) and the composition 5,042 and 76.7 (4,917 and 77.5). Across the four counts
  at both p every |z| against the re-test is at most 1.65, and the routed shares are 9.98 % and 1.00 %.

Under PLAN.md step 6 the authors decide on the registered confirmation and its frozen weight.

## Results per p (PLAN.md step 5)

Columns: failures of each decoder on the evaluation half. Repairs and breaks are the shots where the rule's change
fixes or spoils the composition's answer. The net improvement per 10^4 shots carries Newcombe's 95 % interval for
paired proportions (method 10, `sv_common.newcombe10`, the release's equivalent of `bb.route.paired`). The relative
reduction is the net divided by the composition's failures. Its interval is the net's interval divided by the same
count, which treats that count as fixed, so it is approximate. The routed oracle's repairs are the evaluation half's
routed shots where only Cascade fails. With one logical bit, exactly one decoder is right on every routed shot where
the two decisions differ, so the oracle breaks nothing. The McNemar test is exact and two-sided against the
composition (`sv_common.mcnemar_p`). The last two rows are references: the routed oracle picks whichever of the two
answers is right, and the other row always takes the network's answer on the disagreement rows.

### p = 0.6 % (evaluation half: odd chunks, 2,000,000 shots; tau1 = 5.242445659637451)

| decoder | failures | repairs | breaks | net (repairs - breaks) per 10^4 shots, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|
| Cascade alone | 10,110 | | | | | | | |
| network alone | 16,447 | | | | | | | |
| composition (the comparison's C5) | 10,064 | | | | | | | |
| J (joint decision) | 9,949 | 529 | 414 | +0.575 [+0.274, +0.877] | 1.14 % [0.54, 1.74] | 3.8 % | 0.000202 | w = 7.866 (step [7.865678, 7.865811)) (+77) |
| P (preference) | 10,006 | 338 | 280 | +0.29 [+0.0459, +0.535] | 0.58 % [0.09, 1.06] | 1.9 % | 0.0218 | a = 0.1069, b = 0.001 (+89) |
| T (two thresholds) | 10,005 | 327 | 268 | +0.295 [+0.0554, +0.535] | 0.59 % [0.11, 1.06] | 1.9 % | 0.0173 | t_C = 0.1091, t_S = 0.07241 (+90) |
| reference: always the network on the disagreement rows | 16,447 | 3,040 | 9,423 | -31.9 [-33, -30.8] | -63.42 % [-65.60, -61.26] | -210.0 % | < 1e-300 |  |
| reference: routed oracle | 7,024 | 3,040 | 0 | +15.2 [+14.7, +15.7] | 30.21 % [29.14, 31.29] | 100.0 % | < 1e-300 |  |

Fit half (even chunks, 2,000,000 shots; composition 10,103 failures, routed oracle repairs 3,094): J 518 repairs, 441 breaks, net +77 (0.76 %); P 340 repairs, 251 breaks, net +89 (0.88 %); T 335 repairs, 245 breaks, net +90 (0.89 %).

Evaluation half: routed 199,753 (9.99 %), disagreement rows 12,463 (only Cascade fails 3,040, only the network fails 9,423); routed oracle failures 7,024. Fit half: disagreement rows 12,380 (only Cascade fails 3,094, only the network fails 9,286).

J diagnostics: the evaluation half's own best w = 8.346 (step [8.344573, 8.347169), evaluation net +127); the fitted w's evaluation net +115.

Per-category counts of the whole read (4,000,000 shots) beside the re-test (paper Table I, single mode) and the comparison's original single-mode counts (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | ARB rate | re-test | re-test rate | z | original | original rate | z |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C2) | 20,260 | 5065.0 | 4,936 | 4936.0 | 1.63 | 4,963 | 4963.0 | 1.29 |
| composition (C5) | 20,167 | 5041.8 | 4,917 | 4917.0 | 1.58 | 4,936 | 4936.0 | 1.34 |
| only the composition fails | 75 | 18.8 | 21 | 21.0 | -0.459 | 13 | 13.0 | 1.23 |
| only Cascade alone fails | 168 | 42.0 | 40 | 40.0 | 0.277 | 40 | 40.0 | 0.277 |
| routed share | 9.977 % | | 9.98 % | | | 9.96 % | | |

Whole read: network alone 32,742, only the network fails 18,784, only Cascade fails 6,302, both 13,958, routed 399,064, routed oracle failures 14,033.

### p = 0.35 % (evaluation half: odd chunks, 15,000,000 shots; tau1 = 8.111059150695802)

| decoder | failures | repairs | breaks | net (repairs - breaks) per 10^4 shots, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|
| Cascade alone | 1,195 | | | | | | | |
| network alone | 1,997 | | | | | | | |
| composition (the comparison's C5) | 1,178 | | | | | | | |
| J (joint decision) | 1,172 | 35 | 29 | +0.004 [-0.00674, +0.0148] | 0.51 % [-0.86, 1.88] | 1.7 % | 0.532 | w = 13.84 (step [13.53909, 14.15232)) (+19) |
| P (preference) | 1,170 | 36 | 28 | +0.00533 [-0.00539, +0.0161] | 0.68 % [-0.69, 2.05] | 2.3 % | 0.382 | a = -0.08824, b = 0.1122 (+27) |
| T (two thresholds) | 1,171 | 66 | 59 | +0.00467 [-0.0101, +0.0195] | 0.59 % [-1.29, 2.49] | 2.0 % | 0.592 | t_C = 0.5645, t_S = 1.87 (+22) |
| reference: always the network on the disagreement rows | 1,997 | 344 | 1,163 | -0.546 [-0.597, -0.496] | -69.52 % [-76.06, -63.13] | -238.1 % | 5.77e-104 |  |
| reference: routed oracle | 834 | 344 | 0 | +0.229 [+0.206, +0.254] | 29.20 % [26.19, 32.39] | 100.0 % | 5.58e-104 |  |

Fit half (even chunks, 15,000,000 shots; composition 1,124 failures, routed oracle repairs 346): J 50 repairs, 31 breaks, net +19 (1.69 %); P 49 repairs, 22 breaks, net +27 (2.40 %); T 68 repairs, 46 breaks, net +22 (1.96 %).

Evaluation half: routed 150,042 (1.00 %), disagreement rows 1,507 (only Cascade fails 344, only the network fails 1,163); routed oracle failures 834. Fit half: disagreement rows 1,487 (only Cascade fails 346, only the network fails 1,141).

J diagnostics: the evaluation half's own best w = 4.271 (step [4.241655, 4.301133), evaluation net +15); the fitted w's evaluation net +6.

Per-category counts of the whole read (30,000,000 shots) beside the re-test (paper Table I, single mode) and the comparison's original single-mode counts (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | ARB rate | re-test | re-test rate | z | original | original rate | z |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C2) | 2,344 | 78.1 | 2,354 | 78.5 | -0.146 | 2,406 | 80.2 | -0.9 |
| composition (C5) | 2,302 | 76.7 | 2,325 | 77.5 | -0.338 | 2,373 | 79.1 | -1.04 |
| only the composition fails | 1 | 0.0 | 0 | 0.0 | 1 | 0 | 0.0 | 1 |
| only Cascade alone fails | 43 | 1.4 | 29 | 1.0 | 1.65 | 33 | 1.1 | 1.15 |
| routed share | 1.002 % | | 1.00 % | | | 1.00 % | | |

Whole read: network alone 3,916, only the network fails 2,305, only Cascade fails 733, both 1,611, routed 300,484, routed oracle failures 1,612.

Carried forward (fit half only): P; mean fit-half relative reduction over the two p: J 1.23 %, P 1.64 %, T 1.42 %.

One w for both p (diagnostic): w = 10.59, mean fit-half relative reduction 1.04 %; p = 0.6 %: fit net +48, evaluation net +111 (1.10 %); p = 0.35 %: fit net +18, evaluation net +11 (0.93 %).

## How it ran

- Pod: one RunPod pod (secure cloud, 2.49 USD/h), one RTX PRO 6000 Blackwell Server Edition. `sv_common`'s GPU
  guard passed, so nothing was lifted. Driver 580.159.04 (the comparison's speed screen: 590.48.01), AMD EPYC 9554 host
  with 256 vCPUs. A first request for a pod (07:12 UTC) had failed within 7 seconds; a second request at 07:36:53 UTC
  gave the pod at 07:37:04 UTC (00:37:04 PDT). DONE was written at 09:44:00 UTC. The pod last answered ssh at
  09:51:53 UTC and refused connections from 09:52:08 UTC, so it ran about 2 h 15 min (about 5.6 USD).
- Timeline (UTC): setup 07:38:21-07:39:19; dry run 07:39:49-07:40:30 (its analysis step re-run by hand at about
  07:47, see "What changed"); read at 0.6 % 07:41:03-07:55:13 (844.5 s for the read, then its seed check and the
  dry-row comparison); read at 0.35 % 07:55:13-09:39:57 (6,278.6 s); analysis 09:39:57-09:40:07; the run's manifest
  09:40:34. The parts were copied back one by one during the reads (08:19-09:41), and a final pass checked every file
  (09:42-09:43).
- Speed: Cascade's graph at batch 16 took 193 us per shot at 0.6 % (770 s for 4,000,000 shots) and 191 us at 0.35 %
  (5,739 s for 30,000,000), against 204.7 us in the comparison's screen; the engine took 16.5 us per shot (16.2).
- The link: the first copy test (07:45-07:53 UTC) ran at 0.3 to 0.5 MB/s, because the Linux workstation's own uplink was saturated by
  other accounts (about 24 MB/s outbound), which slows every download to it. At that rate the 9.7 GB of parts
  would not have come back within the budget, so `code/arb2_slim.py` (every per-shot array except the detector rows)
  was prepared and briefly used. From about 08:18 UTC the link carried 14 to 18 MB/s; the slim copies were then deleted
  on both sides, and the full parts came back instead.
- Environment: the comparison's venv pins (release `cascade/d13/surface/pod/requirements_cu128.txt`). `uv pip freeze`
  equals the list apart from pip itself, which the comparison's own check also leaves out, and the packages' sha256
  (6fd0f593...) equals the identity in the comparison's speed screen. Python 3.12.14, torch 2.11.0+cu128, CUDA 12.8,
  cuDNN 9.19.0, triton 3.6.0, numpy 2.2.6, scipy 1.18.1, stim 1.16.0. Records: `out/env.json`, `out/logs/setup.log`,
  `out/logs/venv_freeze.txt`.
- Code and data: the release at 94aaaa7f (this repository, cloned to `~/work/arb2/release`, clean). The
  weights equal the release's `weights/MANIFEST.sha256` (`cascade_d13/CS13/final_ema.pt`,
  `surface/d13_b5_r775.safetensors`, `surface/grid_d13.npz`); `sc.load_S` checks d13_b5_r775's sha256 (cbd2b84a...)
  and `sc.load_B` the source checkpoint's (5351b27b...). Stim's fingerprint on the pod is the release's
  (21335d99f1b86f0b), and the circuit at 0.6 % passes the release's pinned check (`surface_layout.check_circuit_at`,
  86f1d9e4...).
- Routing: the lock's C5 tau1, 5.242445659637451 at 0.6 % and 8.111059150695802 at 0.35 %, read from the release's
  sanitized lock `cascade/d13/out/locked_d13.json` (sha256 455eb2a6...); the analysis refuses any other value.
- Reads: `code/arb2_read.py` through the shim, one p per run: chunks of 10,000 shots, parts of 100 chunks, 4 spawn
  sampling workers at most 8 chunks ahead; S in batches of 4,096 carried across chunks; Cascade at batch 16 in a CUDA
  graph of its uncompiled module (the dry-run rule, below).

## Seeds, and the check that the rows are new

- TAGs, new (neither in the re-test's `retest/tags.json`, nor among the TAGs RETEST.md section 2 lists, nor ARB's two
  BB72 TAGs): `convlines arb surface d13 p0.006\n` (set arb-d13-0.006, 400 chunks of 10,000 shots) and
  `convlines arb surface d13 p0.0035\n` (set arb-d13-0.0035, 3,000 chunks).
- Formula: RETEST.md section 2's, seed(TAG, j) = 2^62 + (first 8 bytes of SHA-256(R || TAG || j as 2 bytes)) mod 2^62.
  R is the re-test's beacon randomness (`retest/run/beacon.json`, drand round 6,547,616, 299266d1...), reused as ARB
  on BB72 did; the new TAGs alone make the seeds new.
- Collision rule: RETEST.md section 2's, unchanged, against 17,534 historical seeds: `retest/historical_seeds.txt`
  (7,034, sha256 f2c7eb44..., the list RETEST.md names) and `seeds/hist_extra.txt` (17,534, from `code/hist_scan.sh`:
  every 19-digit integer in [2^62, 2^63) in any seed, beacon or manifest file on every branch of the research repository and in every
  text file of the release; it holds the 7,034, the re-test's 6,115 seeds and ARB's 2,800). No index was dropped.
  Seeds file: `seeds/arb2_seeds.json` (sha256 b869cd7d...; `code/arb2_seeds.py check` recomputes it). First seeds:
  7125939203913419922 (arb-d13-0.006) and 5068027980986011364 (arb-d13-0.0035).
- Mapping: `code/arb2_shim.py` is `retest/shim/rt_surface.py`'s patch with ARB's sets. It replaces
  `surface.surface_data.eval_seed` at the top level, so the main process and every spawned sampling worker (which
  re-runs the shim as `__mp_main__`) use it. Kind 11 (the comparison's confirmation kind) at d = 13, chunk j gets ARB's
  seed j; any other kind-11 request raises; every other kind is unchanged; there is no identity mode.
- Checks (`code/arb2_seedcheck.py`; records `out/read/seedcheck_<p>.json` and `out/dry/read/seedcheck_<p>.json`). Both
  full reads and the dry run passed:
  - Every chunk's recorded seed is ARB's seed of its set and the patched `eval_seed`'s (400 and 3,000 chunks). All
    lie in [2^62, 2^63), and none is a seed of the re-test or of the comparison's confirmation block.
  - The shim's trace shows exactly one call per chunk by the main process and one draw by a sampling worker, both
    with ARB's seed: 800 and 6,000 records, 4 workers.
  - At 8 sampled chunks per p (chunks 0 and 1 in the dry run), the saved detector rows and observables equal a fresh
    draw from the recorded seed, and the recorded digest is the fresh draw's. Each digest differs from that of the
    same chunk number in the comparison's confirmation block (kind 11, its arithmetic seed), in the re-test's block and
    in the comparison's development block (kind 10).
  - For chunk 0 of each read and chunks 0 and 1 of the dry run, the comparison's own seed reproduces on the pod the
    digest that the re-test's replay drew on the cluster (`retest/replay/d13/replay_seeds_off.json`). The pod's Stim
    therefore draws the comparison's rows from the comparison's seeds, and ARB's rows are new because its seeds are.
  - Every chunk digest within each read is distinct.

## Checks of the outputs

- Cascade's path (PLAN.md's rule): the dry run ran all 40,000 of its shots both ways. The graph's logits equal those of
  the direct eager calls bit for bit, and the eager calls took 24 to 25 % longer (5.1 s against 4.1 s and 5.0 s against
  4.0 s per 20,000 shots), so the full reads used the graph. In the full reads, the first 1,024 shots of every part ran
  again by direct eager calls: 4,096 and 30,720 shots, with 0 differing logits and decisions.
- The analysis's inputs: every chunk from 0 to n - 1 holds 10,000 shots, the parts are contiguous, every logit is
  finite, and the engine's decision equals c > 0 and Cascade's equals z > 0 on every shot (`input_checks` in
  `summary.json`). Each rule's fit-half gain, recomputed from its fitted parameters (J's from c + w z in float64),
  equals its sweep's.
- Reproducibility (`read/drycompare_<p>.json`): the dry run's chunks 0 and 1 equal the full reads' same chunks bit for
  bit (observables, detector rows, chunk numbers, c, the engine's decisions, z, Cascade's decisions, seeds, digests).
- Copy-back: all 85 files of the pod's `out/` (9,736,824,237 bytes) are in `~/work/arb2/out` with the pod's sizes and
  sha256 (`code/pull_parts.py` recomputed the pod's sha256 for every file and compared it with the local copy;
  manifests `out/pod_out.sha256` and `out/pod_out.sizes`). The run's own `manifest.sha256` (78 files) agrees with them,
  and the code that ran on the pod equals `~/work/arb2/code` (sha256).

## What changed in the release's code, and why

No file of the release was edited; the clone at 94aaaa7f stays clean, and the pod ran a byte-equal subset of it (all
555 files of the bundle compared with the clone). Everything ARB adds is a wrapper in `code/` that loads the release's
modules unchanged:

- `arb2_shim.py`: the seed mapping (above). It is needed because the sampling workers compute seeds with the
  arithmetic `eval_seed`, so new TAG strings alone change no shot.
- `arb2_read.py`: the read. It builds both decoders through `sv_common` exactly as `sv_read.py` builds them: S is
  `sc.load_S` on the deployed engine (`sc.s_fn`, `timing.engine` FROZEN) in batches of 4,096 consecutive shots carried
  across chunk boundaries (`sc.Queue` over `sc.Graph` at `sc.S_BATCH`); B is `sc.load_B(dev, "final")` and
  `sc.b_fn(model, "graph")`, the uncompiled module under bf16 autocast, at batch 16. Shots come from
  `surface_data.stream_shots` (4 spawn workers, at most 8 chunks ahead). It saves per shot the observable, the packed
  detectors, the chunk, c, the engine's decision, z and Cascade's decision, and per chunk its seed and digest.
- Cascade's path. The d = 13 comparison has no separate eager accuracy read: it read and timed Cascade through
  torch.compile plus a CUDA graph at batch 1 and 16. PLAN.md therefore takes "its eager accuracy path" to be the
  uncompiled module at the comparison's batch of 16, replayed in one CUDA graph of that module (`sc.Graph`, the
  comparison's screened path graph@16) if the dry run shows the graph's logits equal to direct eager calls bit for bit
  and the eager calls more than 10 % slower. Both held (below), so the reads used the graph, and every part re-ran its
  first 1,024 shots by direct eager calls.
- `arb2_seedcheck.py` and `arb2_drycompare.py`: the checks above.
- `arb2_analysis.py`: PLAN.md steps 3 to 5, run on the pod. Detector rows are never read. The J, P and T sweeps are
  exact (checked against brute force by `test_synth.py`); statistics come from `sv_common.newcombe10` and
  `sv_common.mcnemar_p`.
- One change during the run: the analysis script's diagnostic "one w for both p" divided by the composition's fit-half
  failures without a guard, which raised ZeroDivisionError in the dry run (20,000 shots: no fit-half failure at
  0.35 %). The guard (0 when that count is 0, as the per-rule relative reduction already did) was uploaded at about
  07:47 UTC, during the first read and before any count of the reads had been looked at. No fit, rule, choice or
  evaluation formula changed, and the full reads have hundreds of fit-half failures at each p, so the guard never
  acts on them. The dry run's analysis step was then re-run by hand on the same dry rows (`out/dry/analysis/`);
  `out/logs/dry.log` still ends with the original traceback. `arb2_tables.py` was written during the run and only
  formats `summary.json`.
- Other files: `arb2_seeds.py` and `hist_scan.sh` (seeds), `arb2_digest.py` (one chunk's digest), `pod_env.sh`,
  `pod_setup.sh`, `pod_dry.sh`, `pod_read.sh` and `pod_run.sh` (the pod), `pull_parts.py`, `sync_full.sh`,
  `sync_loop.sh` and `final_pull.sh` (the copy back during and after the reads, with verification), `arb2_tables.py`
  (the tables above from `summary.json`), `arb2_slim.py` and `sync_slim.sh` (the fallback for a slow link; none of
  their outputs was kept, see "How it ran"), `poll.sh` (written by the stopped session, superseded by `sync_loop.sh`),
  `test_synth.py` (local tests on synthetic data, no results).

## Definitions (as `code/arb2_analysis.py` implements them)

- Decisions: the network S decides 1 iff c > 0 (the engine's own decision, checked equal to c > 0 on every shot), and
  Cascade B iff z > 0 (checked likewise). Routed: |c| (float32, compared in float64) strictly below the lock's C5
  tau1, as `sv_common.routed`. The composition answers B on routed shots and S elsewhere.
- Halves: fit = even chunk numbers, evaluation = odd.
- Disagreement rows: routed shots where S and B differ. The rules act only there; everywhere else the composition's
  answer stands. On a disagreement row exactly one decoder is right: taking the network's answer repairs (+1) when
  Cascade is wrong and breaks (-1) when the network is wrong.
- J (joint decision): answer 1 iff c + w z > 0, Cascade's answer on a tie. On a disagreement row this takes the
  network's answer iff |c| > w |z|, so the fit-half gain is a step function of w with steps at the rows' ratios
  |c| / |z|. The sweep is exact over every step; the fitted w is the geometric midpoint of the best step (r1 / 2 on the
  first step, 2 rm on the last finite one), ties keep the larger w, and the answers are recomputed from c + w z in
  float64 (the fit-half gain is asserted equal).
- P (preference): take the network's answer iff |z| < a + b |c|; b in {0} and 10^(k/40), k = -120..120, the best a
  for each b exactly by cumulative counts; ties keep the smaller b, then the smaller a.
- T (two thresholds): take the network's answer iff |z| < t_C and |c| > t_S, exact over both thresholds; ties keep the
  fewest switches.
- Objective on the fit half: the composition's failures (repairs minus breaks); "never switch" (gain 0) is always
  available. Carried forward: the rule with the largest mean over the two p of its fit-half relative reduction; ties
  go to fewer parameters, then to the order J, P, T; none if that mean is not positive.

## Files

- `~/work/arb2/RESULTS.md` (this file), `~/work/arb2/PLAN.md`, `~/work/arb2/code/`, `~/work/arb2/seeds/` and the small outputs below are
  committed to branch arb-d13 of the research repository.
- `~/work/arb2/out/` (9.7 GB, copied from the pod and verified; the npz files are not committed):
  - `read/p0.0060/part_*.npz` (4 files, 1.1 GB) and `read/p0.0035/part_*.npz` (30 files, 8.0 GB), one per 1,000,000
    shots: obs, det (packed detector rows, (n, 273) uint8, `np.packbits` big bit order), chunk, c (float32), sdec,
    z (float32) and bdec, with each chunk's seed and digest in `meta`. Their sha256 are in each read's journal
    (`read/p<p>/progress.jsonl`) and in `pod_out.sha256`.
  - `read/p<p>/read_*.json` (settings, identity, timings, the eager checks, each decoder's failures alone),
    `read/seedcheck_<p>.json`, `read/drycompare_<p>.json` and `read/trace_<p>/` (the shim's trace).
  - `analysis/summary.json` (every number above, with J's whole curve over w on both halves), `analysis/tables.md`
    (the tables above, rendered on the pod by `code/arb2_tables.py`) and `analysis/disagree_<p>.npz` (per disagreement
    row: shot index, chunk, half, both answers, truth, gain, c, z and each rule's switch).
  - `dry/` (the dry run, two chunks per p, with its own analysis; never reported), `logs/` (setup, dry run, run, venv
    freeze), `env.json`, `status.txt`, `manifest.sha256`, `pod_out.sha256` and `pod_out.sizes`.

## Not done, and limits

- No registered test and no claim (PLAN.md step 6). The evaluation half is a single exploratory look, and its intervals
  and p-values are not corrected for the three rules or the two p.
- No timing. Only the comparison's uncompiled Cascade module at batch 16 (in a CUDA graph, checked against direct eager
  calls) was read; the comparison's compiled single-shot and batched paths were not run on these shots, and the chain
  to correlated PyMatching is not part of this experiment.
- Skipped to save time, as the handoff asked: the optional comparison of every chunk digest of the reads with every
  chunk of the comparison's confirmation block, the re-test's block and the development block (`OLD_ALL`). The seeds
  are new by the collision rule, and 8 sampled chunks per p differ from those blocks' chunks of the same number.
- The pod's driver (580.159.04) differs from the comparison's (590.48.01).
- R comes from the re-test's beacon, not a new one; the seeds are new because the TAGs are.
