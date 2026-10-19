# ARB results: arbitration on the routed shots of CoDecoder + Cascade (BB72), exploratory

Exploratory, as PLAN.md fixes it: nothing was registered and nothing here is a claim. Every threshold, weight and the
choice of the arbiter to carry forward come from the fit half (even chunks). Every number below is from the evaluation
half (odd chunks) unless a table says otherwise.

## Summary

On fresh shots (11,000,000 at p = 0.2 % and 17,000,000 at p = 0.3 %, of which 5,500,000 and 8,500,000 are in the
evaluation half), choosing between the two answers on the routed shots where they differ recovers much of what the
routed oracle could add:

- A3 (joint score, the arbiter carried forward by the fit-half rule) cuts the composition's failures from 4,053 to
  3,329 at 0.2 % (742 repairs, 18 breaks, net +724, -17.9 %) and from 37,888 to 32,504 at 0.3 % (5,763 repairs,
  379 breaks, net +5,384, -14.2 %). That is 61 % and 59 % of the routed oracle's repairs. Against Cascade alone
  (4,097 and 37,998 failures) the A3 composition fails 18.7 % and 14.5 % less often. The composition itself fails
  1.1 % and 0.3 % less often than Cascade on these shots.
- A2 (preference) comes close with far fewer breaks: -17.6 % (721 repairs, 9 breaks) and -13.0 % (5,109 repairs,
  171 breaks). A1m reaches -16.1 % and -11.0 %, and A1s -11.0 % and -6.8 %.
- Fit-half and evaluation-half gains agree closely (A3: +719 and +724 at 0.2 %, +5,306 and +5,384 at 0.3 %). A3's
  fitted weight sits on the flat part of its curve: the evaluation half's own best w would add only 27 and 7 to the
  net.
- A3 also repairs shots that both networks fail: 283 and 2,055 of its repairs are third answers (neither network's),
  against 4 and 105 third-answer breaks.
- The reads reproduce D1's per-category rates: across the 13 counts at both p, every |z| is at most 1.9.

Under PLAN.md step 6, a registered test on fresh shots may follow when an arbiter pays off. This report does not make
that decision.

## Results per p (PLAN.md step 5)

Columns: failures of each decoder on the evaluation half. Repairs and breaks are the shots the arbiter's change fixes
and spoils relative to the composition. The net improvement per 10^4 shots carries Newcombe's 95 % interval for paired
proportions (`bb.route.paired`). The relative reduction is net divided by the composition's failures; its interval is
the net interval divided by the same count, so it treats that count as fixed (approximate). The routed oracle's
repairs are the evaluation half's routed shots where only C14' fails. The McNemar test is exact and two-sided against
the composition; "< 1e-300" means scipy's binomtest underflowed to 0. The two reference rows are not arbiters: the
routed oracle chooses the right one of the two answers, and the last row always takes S55'-deep's answer on the
disagreement rows.

### p = 0.2 %: evaluation half (odd chunks), 5,500,000 shots

| decoder | failures | repairs | breaks | net | net per 10^4 shots [95 % CI] | relative failure reduction [approx. 95 % CI] | share of the routed oracle | exact McNemar p vs composition |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C14') | 4,097 | | | | | | | |
| composition (S55'-deep, 2 % routed to C14') | 4,053 | | | | | | | |
| A1m (min abs logit) | 3,400 | 838 | 185 | +653 | 1.19 [1.07, 1.30] | 16.1 % [14.6, 17.7] | 55.2 % | 9.2e-100 |
| A1s (softplus score) | 3,607 | 702 | 256 | +446 | 0.81 [0.70, 0.92] | 11.0 % [9.5, 12.5] | 37.7 % | 1.2e-48 |
| A2 (preference) | 3,341 | 721 | 9 | +712 | 1.29 [1.20, 1.39] | 17.6 % [16.3, 18.9] | 60.1 % | 5.5e-200 |
| A3 (joint score) | 3,329 | 742 | 18 | +724 | 1.32 [1.22, 1.42] | 17.9 % [16.5, 19.2] | 61.1 % | 3.1e-193 |
| routed oracle (reference) | 2,869 | 1,184 | 0 | +1,184 | 2.15 [2.03, 2.28] | 29.2 % | 100.0 % | < 1e-300 |
| always S55'-deep on the disagreement rows (reference) | 5,204 | 1,184 | 2,335 | -1,151 | -2.09 [-2.31, -1.88] | -28.4 % | -97.2 % | 3.1e-85 |

A3's third-answer changes (answers that are neither network's): 714, of which 283 repairs, 4 breaks and 427 neutral
(both networks and A3 wrong). Its two-answer changes: 459 repairs, 14 breaks. Every arbiter has breaks, so the 3/N
bound does not apply (3/N_eval would be 5.5e-07).

Disagreement rows of the evaluation half: 4,530 of 110,268 routed shots (only C14' fails 1,184, only S55'-deep fails
2,335, both fail 1,011). The routed oracle keeps 2,869 failures.

Fitted on the fit half (even chunks, 5,500,000 shots, composition 4,018 failures, 4,507 disagreement rows):

| arbiter | parameters | fit-half repairs | breaks | net | relative reduction |
|---|---|---|---|---|---|
| A1m | t_C = 0.2173, t_S = 0.00059 | 808 | 192 | +616 | 15.33 % |
| A1s | t_C = -2.263, t_S = 0.00059 | 660 | 254 | +406 | 10.10 % |
| A2 | a = -4.281, b = 1.059 | 707 | 8 | +699 | 17.40 % |
| A3 | w = 6.31 | 733 | 14 | +719 | 17.89 % |

### p = 0.3 %: evaluation half (odd chunks), 8,500,000 shots

| decoder | failures | repairs | breaks | net | net per 10^4 shots [95 % CI] | relative failure reduction [approx. 95 % CI] | share of the routed oracle | exact McNemar p vs composition |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C14') | 37,998 | | | | | | | |
| composition (S55'-deep, 10 % routed to C14') | 37,888 | | | | | | | |
| A1m (min abs logit) | 33,705 | 5,875 | 1,692 | +4,183 | 4.92 [4.72, 5.12] | 11.0 % [10.6, 11.5] | 46.0 % | < 1e-300 |
| A1s (softplus score) | 35,312 | 3,919 | 1,343 | +2,576 | 3.03 [2.86, 3.20] | 6.8 % [6.4, 7.2] | 28.3 % | 4.0e-288 |
| A2 (preference) | 32,950 | 5,109 | 171 | +4,938 | 5.81 [5.64, 5.98] | 13.0 % [12.7, 13.4] | 54.3 % | < 1e-300 |
| A3 (joint score) | 32,504 | 5,763 | 379 | +5,384 | 6.33 [6.15, 6.52] | 14.2 % [13.8, 14.6] | 59.2 % | < 1e-300 |
| routed oracle (reference) | 28,786 | 9,102 | 0 | +9,102 | 10.71 [10.49, 10.93] | 24.0 % | 100.0 % | < 1e-300 |
| always S55'-deep on the disagreement rows (reference) | 56,092 | 9,102 | 27,306 | -18,204 | -21.42 [-21.86, -20.98] | -48.0 % | -200.0 % | < 1e-300 |

A3's third-answer changes: 6,082, of which 2,055 repairs, 105 breaks and 3,922 neutral. Its two-answer changes:
3,708 repairs, 274 breaks. Every arbiter has breaks, so the 3/N bound does not apply (3/N_eval would be 3.5e-07).

Disagreement rows of the evaluation half: 47,202 of 849,558 routed shots (only C14' fails 9,102, only S55'-deep fails
27,306, both fail 10,794). The routed oracle keeps 28,786 failures.

Fitted on the fit half (even chunks, 8,500,000 shots, composition 37,985 failures, 46,982 disagreement rows):

| arbiter | parameters | fit-half repairs | breaks | net | relative reduction |
|---|---|---|---|---|---|
| A1m | t_C = 0.2231, t_S = 0.00049 | 5,835 | 1,701 | +4,134 | 10.88 % |
| A1s | t_C = -3.132, t_S = 7.6e-05 | 3,828 | 1,392 | +2,436 | 6.41 % |
| A2 | a = -1.086, b = 0.5012 | 5,049 | 182 | +4,867 | 12.81 % |
| A3 | w = 3.758 | 5,688 | 382 | +5,306 | 13.97 % |

### The arbiter carried forward

The rule was fixed in `code/arb_analysis.py` before any read: the largest mean over the two p of the fit-half relative
failure reduction, ties to fewer parameters, then the order A1m, A1s, A2, A3, and none if the best mean is not
positive. Fit-half means: A1m 13.11 %, A1s 8.26 %, A2 15.10 %, A3 15.93 %. Carried forward: A3, with
w = 6.31 at 0.2 % and w = 3.76 at 0.3 % (the grid is 10^(k/40), k = -120..120, plus 0 and infinity). A1's fitted
margin threshold t_S is close to 0 at both p, so A1 acts as a threshold on Cascade's confidence alone.

A3's gain on each half across w (from `a3_curve` in `out/analysis/summary.json`):

| w | 0 (always S55'-deep) | 0.1 | 1 | 3.16 | 10 | 31.6 | 100 |
|---|---|---|---|---|---|---|---|
| fit / evaluation net at 0.2 % | -1,279 / -1,151 | -266 / -164 | +641 / +665 | +705 / +737 | +678 / +689 | +498 / +510 | +238 / +247 |
| fit / evaluation net at 0.3 % | -18,067 / -18,204 | -7,333 / -7,332 | +4,083 / +4,066 | +5,283 / +5,383 | +4,753 / +4,828 | +3,208 / +3,239 | +1,493 / +1,547 |

### Per-category counts of the whole reads beside D1's original reads

D1's counts are PLAN.md's table and X2's `bb_d1.py` (REG). The derived rows follow from those: C14' failures on
routed shots, composition failures, routed-oracle failures and repairs. z is the difference of the two rates over its
pooled standard error (independent samples).

| count | ARB 0.2 % | D1 0.2 % | z | ARB 0.3 % | D1 0.3 % | z |
|---|---|---|---|---|---|---|
| C14' fails (C) | 8,163 | 7,922 | +1.90 | 76,110 | 76,730 | -1.59 |
| S55'-deep fails (S) | 10,501 | 10,348 | +1.06 | 112,144 | 112,636 | -1.04 |
| only S55'-deep fails | 4,776 | 4,735 | +0.42 | 54,480 | 54,491 | -0.03 |
| only C14' fails | 2,438 | 2,309 | +1.87 | 18,446 | 18,585 | -0.72 |
| both fail | 5,725 | 5,613 | +1.05 | 57,664 | 58,145 | -1.42 |
| routed | 220,654 | 220,058 | +0.91 | 1,701,204 | 1,701,544 | -0.19 |
| kept breaks | 16 | 19 | -0.51 | 98 | 81 | +1.27 |
| kept repairs | 108 | 86 | +1.58 | 335 | 342 | -0.27 |
| kept, both fail | 266 | 238 | +1.25 | 2,032 | 1,977 | +0.87 |
| C14' fails on routed shots | 7,789 | 7,598 | +1.54 | 73,743 | 74,411 | -1.74 |
| composition fails | 8,071 | 7,855 | +1.71 | 75,873 | 76,469 | -1.53 |
| routed oracle fails | 5,741 | 5,632 | +1.02 | 57,762 | 58,226 | -1.36 |
| routed oracle repairs | 2,330 | 2,223 | +1.59 | 18,111 | 18,243 | -0.69 |

The model times of the reads are also D1's: C14' 1,722 s and S55'-deep 72 s at 0.2 % (D1: 1,712 s and 69 s);
2,643 s and 109 s at 0.3 % (D1: 2,639 s and 107 s).

## How it ran

- Pod: one RunPod pod (secure cloud, 2.49 USD/h), one RTX PRO 6000 Blackwell Server Edition (pc_eval's GPU guard
  passed), driver 580.159.04 (D1: 590.48.01), AMD EPYC Genoa host. Pod start 2026-10-17 01:46:33 UTC
  (18:46:33 PDT on 16 October). DONE written at 03:22:46 UTC. The pod last answered ssh at 03:29:48
  UTC and refused connections from 03:30:18 UTC, so it ran about 1 h 43 min (about 4.3 USD).
- Timeline (UTC): setup 01:48:27-01:54:39, dry run 01:54:43-01:57:46, read at 0.2 % 01:58:23-02:29:25, read at
  0.3 % 02:29:25-03:16:00 (each with its seed check, score export and dry-row comparison), analysis
  03:16:00-03:16:14. Copy-back with verification: the 0.2 % and dry outputs 02:30:44-02:32:39, during the second
  read; the rest 03:17:08-03:21:40.
- Environment: D1's venv pins exactly (release `phase_c/scripts/h100_req_cu128.txt`, the pins of Alpine's
  bb/venv_cu128; `uv pip freeze` equals the list): Python 3.12.14, torch 2.11.0+cu128, triton 3.6.0, cuDNN 9.19.0,
  numpy 2.2.6, scipy 1.18.1, stim 1.16.0. Record: `out/env.json`, `out/logs/setup.log`, `out/logs/venv_freeze.txt`.
- Code: the release at 94aaaa7f (this repository, cloned to `~/work/arb/release`, clean). The fused engine's
  sources are byte for byte those of D1's reads (`pc_fused.py` 257561171055fba4 and `bb/fused.py` 195afee3087c0ee4, as
  D1's read signature records them). Weights `weights/cascade_bb72/C14p.pt` (sha256 ca803910...) and `S55p_deep.pt`
  (160d598b...) are equal to the release's `weights/MANIFEST.sha256`. D1's reads name other file hashes (dd0c3b14...,
  051370c3...): the release re-saved the checkpoints without their cluster paths, and SANITIZE.md states that no
  tensor changed. The per-category counts and model times above agree with D1's.
- Circuits: Relay-BP's BB72 X-memory file at d185194 (the one file `circuit_at` reads; its sha256 equals the release's
  manifest), made into depspam_fm by the release's code. The circuit sha256 at 0.2 % and 0.3 % (dcaae19e...,
  647fda02...) are the ones D1's reads record. Stim's fingerprint (`cascade_data.env_record`) is fbd24a267773fa8a on
  the pod and on the Linux workstation.
- Reads: D1's command, run through the seed shim: `pc_eval.py --family depspam_fm --ckpt C14p.pt,S55p_deep.pt --prec
  bf16 --kind 14 --p P --chunks c0:c0+n --procs 8 --save --save-det`, with c0 = 6,433,102 and n = 1,100 or 1,700
  chunks of 10,000 shots. C14' runs eagerly under bf16 autocast in batches of 8,192 and 1,808 per chunk. S55'-deep runs
  on the fused engine in calls of 4,096. Sampling uses 8 spawn workers at most 16 chunks ahead (D1's bounded prefetch).
- Routing: S55'-deep's margin below D1's frozen tau1 (the lock: 7.3331888389587405 at 0.2 %, 7.60439167022705 at
  0.3 %) sends a shot to C14', as in `pc_route.masks`.

## Seeds, and the check that the rows are new

- TAGs, new (neither in the re-test's `tags.json` nor among the TAGs RETEST.md section 2 lists):
  `convlines arb bb72 depspam_fm p0.002\n` (set arb-0.002, 1,100 chunks) and
  `convlines arb bb72 depspam_fm p0.003\n` (set arb-0.003, 1,700 chunks).
- Formula: RETEST.md section 2's, seed(TAG, j) = 2^62 + (first 8 bytes of SHA-256(R || TAG || j as 2 bytes)) mod 2^62.
  R is reused from the re-test's own beacon (`retest/run/beacon.json`, drand round 6,547,616, randomness
  299266d1...); the new TAGs alone make the seeds new.
- Collision rule: RETEST.md section 2, unchanged, against 14,734 historical seeds. These are
  `retest/historical_seeds.txt` (7,034, sha256 f2c7eb44..., the list RETEST.md names) and `seeds/hist_extra.txt`
  (`code/hist_scan.sh`: every 19-digit integer in [2^62, 2^63) in any seed, beacon or manifest file on any branch of
  the research repository and in any text file of the release). The scan holds all 7,034 plus the re-test's own seeds and its dry-run
  and replay seeds. No index was dropped. Seeds file: `seeds/arb_seeds.json` (sha256 5719c8cb...;
  `arb_seeds.py check` recomputes it). First seeds: 7800278565957140538 (arb-0.002) and 7639215094236668584
  (arb-0.003).
- Mapping: `code/arb_shim.py` is `retest/shim/rt_d1.py` with ARB's sets. It patches `bb.cascade_data.eval_seed` at the
  top level, so the main process and every spawn worker (which re-runs the shim as `__mp_main__`) use the new
  function. Kind 14, depspam_fm, chunk c0 + j gets ARB's seed j; any other kind-14 chunk raises; there is no identity
  mode.
- Checks (`code/arb_seedcheck.py`, adapted from the re-test's `d1_seedcheck_rt.py`; records
  `out/read/seedcheck_<p>.json` and `out/dry/read/seedcheck_<p>.json`). Both full reads and the dry run passed:
  - Every chunk's recorded seed is ARB's seed and the patched `eval_seed`'s.
  - The shim's trace shows, for each chunk, exactly one draw by the main process and one by a sampling worker, both
    with ARB's seed: 2,200 and 3,400 records, 8 workers.
  - At 8 sampled chunks per p (2 in the dry run) the saved detector rows equal a fresh draw from the recorded seed.
  - Every chunk digest of the reads is distinct, and none equals any of the 5,600 digests recorded by D1's reads and
    the re-test's reads, at either p (and so none equals the digest of the same chunk number).
  - At every sampled chunk, a draw from D1's arithmetic seed reproduces D1's recorded digest on the pod, while ARB's
    rows differ. Stim on the pod therefore draws D1's rows from D1's seeds, and the difference comes from the seeds.
  The same comparison ran on the Linux workstation before the pod, on chunks c0 to c0 + 2 at both p: D1's digests reproduced,
  ARB's were new.

## Checks of the outputs

- Float32 logits (`p<p>_z32.json`): float16 of them equals the read's `z0`, min |logit| equals the read's `conf0` bit
  for bit, and none is non-finite, at both p.
- Score export (`p<p>_scores.json`): 9,037 rows at 0.2 % and 94,184 at 0.3 %, every routed shot whose two decisions
  differ. On every row, bit for bit: the read's own engine (`pc_fused.Batched`) rerun on these rows gives the read's
  decisions and margins; the exporting engine gives the same decisions, margins and per-program top two
  (m1, i1, m2); the exported scores give back each program's top two, the decision (first argmax over the 4,096
  classes) and the margin (top one minus top two in fp32). Mismatches: 0 in every check.
- Reproducibility (`drycompare_<p>.json`): the dry run read the first two chunks of each block on their own. Its
  detector rows, observables, decisions, margins, logits (float16 and float32) and exported scores equal the full
  read's same rows bit for bit.
- Copy-back: all 84 files of the pod's `out/` are in `~/work/arb/out` with the pod's sizes and sha256
  (`code/pull_parts.py` recomputed the pod's sha256 for every file and compared it with the local copy; manifests
  `out/pod_out.sha256` and `out/pod_out.sizes`). The code that ran on the pod equals `~/work/arb/code` (sha256 compared).

## What changed in the release's code, and why

No file of the release was edited; the clone at 94aaaa7f stays clean. Everything ARB adds is a wrapper in `~/work/arb/code`
that loads the release's modules unchanged:

- `arb_shim.py`: `retest/shim/rt_d1.py` with ARB's seed sets (above). It is needed because pc_eval's spawn workers
  compute seeds with the arithmetic `eval_seed`, so new TAG strings alone change no shot.
- `arb_read.py`: runs `pc_eval.py` as D1 did. It imports pc_eval, whose patches of `bb.cascade_eval.Predictor` give the
  fused engine and the GPU guard, then calls `cascade_eval.main()`. It adds one output: C14''s 12 logits in float32 for
  every shot (`p<p>_z32.npy`), taken from the same call's output before cascade_eval casts them to float16. The
  float16 copy rounds logits below 6e-5 in magnitude.
- `arb_fused.py` and `arb_scores.py`: S55'-deep's 4,096 class scores on the routed shots where the two decisions
  differ. pc_fused's vote kernel computes every class score but keeps only the top two of each program of 128 classes,
  so the scores never reach memory. `arb_fused.Scored` is `pc_fused.Batched` whose vote kernel is copied line for line
  with one added store of the scores. Every other kernel, table, tile and launch option is pc_fused's. `arb_scores.py`
  reruns this engine on the saved detector rows of those shots, in calls of 4,096 as in the read, and checks it bit
  for bit (above).
- `arb_seedcheck.py`: the re-test's seed check with ARB's sets, plus the comparison with D1's and the re-test's
  recorded chunk digests.
- `arb_drycompare.py`: the dry read against the same rows of the full read.
- `arb_analysis.py`: PLAN.md steps 3 to 5, run on the pod. Detector rows are never read; features are computed once
  on the disagreement rows; the A1 and A2 sweeps use cumulative counts (exact over every threshold, checked against
  brute force by `test_synth.py`); A3 runs on the GPU in float64. Statistics come from `bb.route.paired` (exact
  McNemar and Newcombe's interval, the functions of D1's analysis).
- Other files: `arb_seeds.py` and `hist_scan.sh` (seeds); `pod_env.sh`, `pod_setup.sh`, `pod_dry.sh`, `pod_read.sh`
  and `pod_run.sh` (the pod); `pull_parts.py`, `pull_back.sh` and `poll.sh` (copy-back, verification, polling);
  `results_tables.py` (this file's tables from `summary.json`); `test_synth.py` and `test_cpu_seedcheck.py` (local tests
  on fake data, no results).
- One change during the run: the analysis script gained fit-half repair and break counts, the approximate interval
  of the relative reduction and the fit-half table (sha256 9a097750...). It was uploaded at 02:02 UTC, during the
  first read, before the analysis step and before any count of ARB's reads had been looked at. The fitting, the
  selection rule and every evaluation formula are unchanged.

## Definitions (as `code/arb_analysis.py` implements them)

- Halves: fit = even chunk numbers, evaluation = odd (c0 is even, so chunk c0 + j is in the fit half when j is even).
- Disagreement rows: routed shots where C14''s and S55'-deep's 12-bit decisions differ. Arbiters act only there;
  everywhere else the composition's answer stands. On a disagreement row, choosing S55'-deep's answer repairs (+1)
  when only C14' fails, breaks (-1) when only S55'-deep fails, and counts zero when both fail.
- A1m and A1s: take S55'-deep's answer when Cascade's confidence < t_C and the margin > t_S. The confidence is min |z|
  (A1m) or -sum softplus(-|z|) (A1s), computed from the float32 logits. The sweep is exact over both thresholds,
  placed at midpoints between neighbouring fit-half values; ties keep the fewest switches.
- A2: take S55'-deep's answer when dC < a + b dN. Here dC is the sum of |z| over the differing bits, and dN is
  S55'-deep's score of its answer minus its score of C14''s answer. b runs over {0} and 10^(k/40), k = -120..120; for
  each b, the best a comes exactly from cumulative counts.
- A3: answer argmax_y [s(y) + w (y . z)] over the 4,096 classes (class bit j = observable j). w runs over {0},
  10^(k/40) for k = -120..120 and infinity (= the composition); ties keep the larger w.

## Files

- `~/work/arb/RESULTS.md` (this file), `~/work/arb/PLAN.md`, `~/work/arb/code/`, `~/work/arb/seeds/`; committed to branch arb-exp of the research repository.
- `~/work/arb/out/` (5.6 GB, copied from the pod and verified; per-shot files not committed):
  - `read/p0.00200.npz` (1.14 GB) and `read/p0.00300.npz` (1.77 GB): D1's read layout from bb.cascade_eval (model 0
    C14', model 1 S55'-deep): obs, det (packed), chunk, row, pred<i>, conf<i>, fail<i>, z0 (float16) and cls1.
  - `read/p<p>_z32.npy` (528 MB, 816 MB): C14''s logits in float32 for every shot, rows in the npz's order.
  - `read/p<p>_scores.npy` (148 MB, 1.54 GB) and `read/p<p>_scores_rows.npy`: S55'-deep's 4,096 class scores (fp32)
    on the disagreement rows, with their shot indices.
  - `read/p<p>.json`, `rows_<p>.json`, `p<p>_z32.json`, `p<p>_scores.json`, `seedcheck_<p>.json`,
    `drycompare_<p>.json` and `trace_<p>/`: signatures, chunk seeds and digests, totals, check records and the
    shim's trace.
  - `analysis/summary.json` (every number above), `analysis/tables.md` (the pod's own tables) and
    `analysis/disagree_<p>.npz` (per disagreement row: half, both answers, truth, gain, logits, the features, and each
    arbiter's answer).
  - `dry/` (the dry run: two chunks per p, its own analysis; never reported), `logs/` (setup, dry run, run),
    `env.json`, `status.txt`, `manifest.sha256`, and `pod_out.sha256` and `pod_out.sizes` (the copy-back manifests).
- sha256 (first 16 hex digits): `read/p0.00200.npz` 581321ad3ede2a67, `read/p0.00300.npz` 04e46135898e32de,
  `p0.00200_z32.npy` a2b4209c52db1761, `p0.00300_z32.npy` 170247934cb92dc2, `p0.00200_scores.npy` cfae7a6333d89bf1,
  `p0.00300_scores.npy` b7db1ab37fa9a056. The full list is `out/pod_out.sha256`.

## Not done, and limits

- No registered test, and no claim (PLAN.md step 6). The evaluation half is a single exploratory look; its intervals
  and p-values are not corrected for the four arbiters or the two p.
- No timing. "Cheap" is not measured: A2 and A3 need S55'-deep's class scores, which the deployed fused engine does not
  write out (`arb_fused.Scored` adds the store, untimed), plus a 4,096 x 12 product per disagreement shot for A3.
- Only D1's eager C14' path was read. D1's compiled single-shot and batched paths (`d1_compiled.py`) were not run on
  these shots.
- A2 and A3 use the PLAN's model of C14''s 12 logits as independent per-bit log-odds (log P_C(y) = y . z + const).
  Whether that model is calibrated as a joint probability was not checked.
- The pod's driver (580.159.04) differs from D1's (590.48.01); the weight files differ from D1's in their hash but not
  in their tensors (SANITIZE.md). The rates and model times agree with D1's.
- R comes from the re-test's beacon, not a new one; the seeds are new because the TAGs are.
