# R17: the d = 17 re-registration on T6, result

Written 2026-10-10. Branch `ising-r17`; plan `PLAN_R17.md`; commands `R17_READY.md`.
Amended the same day after the review of these results: section 6 now records that BP-OSD's T6 read overlapped the
CPU job (finding 1).

The one-time analysis (Alpine job J30, outputs in `run/final/`, commit 4e56fb2b) ran once and ended with exit 0.
All 16 cells hold, and accuracy at q = 8 % is preserved within the registered tolerance. Under PLAN_R17.md section 8
this is outcome 1: the paper's d = 17 rows become R17's.

Terms used below:

| term | meaning |
|---|---|
| shot | one simulated run of the d = 17 memory experiment (rotated surface code, 17 rounds, p = 0.006) |
| failure | a shot whose decoded logical outcome is wrong |
| LER | logical error rate: failures divided by shots; given here as failures per million shots |
| tau, q | ours sends a shot to correlated PyMatching when \|c\| < tau (c is the network's logit); q is the share of calibration shots that a tau routes. R17's tau is 4.539173126220703 (q = 8 %), study 4's registered tau 5.121974468231201 (q = 10 %) |
| PM | PyMatching 2.4.0, plain or with correlations ("correlated PM") |
| M1, M4 | NVIDIA's Ising pre-decoders "Fast" and "Accurate", each followed by plain or correlated PM |
| row names | `fp16` / `fp8`: the TensorRT plan's precision; `L3` / `L5`: TensorRT's optimizer level; `B1`: batch 1; `torch`: NVIDIA's PyTorch module; `_corr`: followed by correlated PM instead of plain PM |
| BP-OSD | belief propagation with ordered-statistics decoding (CUDA-Q QEC 0.8.0), two arms: min-LLR (10 iterations, min-LLR initialisation, OSD-CS order 10) and BP 60 (60 iterations, final-LLR initialisation) |
| paired statistics | `icmp.stats2.paired`: the exact two-sided McNemar test (p) and Newcombe's method-8 interval at 97.5 % for LER(ours) - LER(row), alpha = 0.025. A row is "better" when p < 0.025 and the whole interval lies below 0 |
| discordant shots | shots that one decoder gets wrong and the other right; McNemar's test uses only these |

## 1. Accuracy on T6

T6 has 1,000,000 new shots (seeds 91,170,000 + j, j = 0..99), drawn by sample job J24 after the freeze commit
47dc748f was pushed (manifest `run/t6_manifest.json`, sha256 7860cceb...). Both BP-OSD arms read its first 200,000
shots (chunks 0-19).

### 1.1 Ours

| row | tau | routed to correlated PM on T6 | failures | per million |
|---|---|---|---|---|
| ours at q = 8 % (R17's registration) | 4.539173126220703 | 7.97 % (79,652 shots) | 2,708 of 1,000,000 | 2,708 |
| ours at q = 10 % (study 4's tau), same read | 5.121974468231201 | 10.08 % (100,754 shots) | 2,698 of 1,000,000 | 2,698 |
| ours at q = 8 % on the BP-OSD subset | 4.539173126220703 | | 502 of 200,000 | 2,510 |

On every shot, the analysis checked both thresholds against the frozen routing semantics and confirmed that each shot
routed at q = 8 % is also routed at q = 10 %.

### 1.2 Every opponent row against ours at q = 8 %

Differences are LER(ours) - LER(row) in failures per million shots, so a negative value means ours fails less often.
"Only ours" and "only row" count the discordant shots. A McNemar p printed as "< 1e-300" underflowed to 0.0 in double
precision.

| cell | row | shots | row's failures | ours | only ours / only row | difference per million | 97.5 % interval | McNemar p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| plain PM | `pm` | 1,000,000 | 15,658 | 2,708 | 1,159 / 14,109 | -12,950.0 | [-13,227.2, -12,676.3] | < 1e-300 | better |
| correlated PM | `pm_corr` | 1,000,000 | 2,943 | 2,708 | 285 / 520 | -235.0 | [-299.1, -171.6] | 9.9e-17 | better |
| Ising M1 + PM | `M1_fp16_L3_B1` | 1,000,000 | 12,917 | 2,708 | 1,469 / 11,678 | -10,209.0 | [-10,466.6, -9,954.5] | < 1e-300 | better |
| | `M1_fp16_L5_B1` | 1,000,000 | 12,954 | 2,708 | 1,470 / 11,716 | -10,246.0 | [-10,504.0, -9,991.2] | < 1e-300 | better |
| | `M1_fp8_L3_B1` | 1,000,000 | 12,840 | 2,708 | 1,479 / 11,611 | -10,132.0 | [-10,389.0, -9,878.1] | < 1e-300 | better |
| | `M1_fp8_L5_B1` | 1,000,000 | 12,886 | 2,708 | 1,472 / 11,650 | -10,178.0 | [-10,435.3, -9,923.8] | < 1e-300 | better |
| | `M1_torch` | 1,000,000 | 12,961 | 2,708 | 1,470 / 11,723 | -10,253.0 | [-10,511.0, -9,998.1] | < 1e-300 | better |
| Ising M4 + PM | `M4_fp16_L3_B1` | 1,000,000 | 8,109 | 2,708 | 1,590 / 6,991 | -5,401.0 | [-5,609.6, -5,194.9] | < 1e-300 | better |
| | `M4_fp16_L5_B1` | 1,000,000 | 8,110 | 2,708 | 1,584 / 6,986 | -5,402.0 | [-5,610.4, -5,196.0] | < 1e-300 | better |
| | `M4_fp8_L3_B1` | 1,000,000 | 8,072 | 2,708 | 1,585 / 6,949 | -5,364.0 | [-5,572.0, -5,158.5] | < 1e-300 | better |
| | `M4_fp8_L5_B1` | 1,000,000 | 8,070 | 2,708 | 1,588 / 6,950 | -5,362.0 | [-5,570.0, -5,156.4] | < 1e-300 | better |
| | `M4_torch` | 1,000,000 | 8,106 | 2,708 | 1,586 / 6,984 | -5,398.0 | [-5,606.4, -5,192.0] | < 1e-300 | better |
| Ising M1 + correlated PM | `M1_fp16_L3_B1_corr` | 1,000,000 | 12,509 | 2,708 | 1,476 / 11,277 | -9,801.0 | [-10,054.7, -9,550.3] | < 1e-300 | better |
| | `M1_fp16_L5_B1_corr` | 1,000,000 | 12,502 | 2,708 | 1,480 / 11,274 | -9,794.0 | [-10,047.8, -9,543.3] | < 1e-300 | better |
| | `M1_fp8_L3_B1_corr` | 1,000,000 | 12,405 | 2,708 | 1,492 / 11,189 | -9,697.0 | [-9,950.0, -9,447.0] | < 1e-300 | better |
| | `M1_fp8_L5_B1_corr` | 1,000,000 | 12,452 | 2,708 | 1,485 / 11,229 | -9,744.0 | [-9,997.4, -9,493.7] | < 1e-300 | better |
| | `M1_torch_corr` | 1,000,000 | 12,524 | 2,708 | 1,476 / 11,292 | -9,816.0 | [-10,069.9, -9,565.2] | < 1e-300 | better |
| Ising M4 + correlated PM | `M4_fp16_L3_B1_corr` | 1,000,000 | 7,939 | 2,708 | 1,597 / 6,828 | -5,231.0 | [-5,437.7, -5,026.8] | < 1e-300 | better |
| | `M4_fp16_L5_B1_corr` | 1,000,000 | 7,952 | 2,708 | 1,589 / 6,833 | -5,244.0 | [-5,450.6, -5,039.8] | < 1e-300 | better |
| | `M4_fp8_L3_B1_corr` | 1,000,000 | 7,902 | 2,708 | 1,584 / 6,778 | -5,194.0 | [-5,399.9, -4,990.5] | < 1e-300 | better |
| | `M4_fp8_L5_B1_corr` | 1,000,000 | 7,877 | 2,708 | 1,589 / 6,758 | -5,169.0 | [-5,374.7, -4,965.7] | < 1e-300 | better |
| | `M4_torch_corr` | 1,000,000 | 7,955 | 2,708 | 1,590 / 6,837 | -5,247.0 | [-5,453.7, -5,042.7] | < 1e-300 | better |
| min-LLR BP-OSD | `bposd_minllr` | 200,000 | 2,114 | 502 | 299 / 1,911 | -8,060.0 | [-8,593.0, -7,541.4] | 7.7e-287 | better |
| BP 60 BP-OSD | `bposd_bp60` | 200,000 | 3,141 | 502 | 307 / 2,946 | -13,195.0 | [-13,839.3, -12,566.9] | < 1e-300 | better |

All 24 rows are better, so all eight accuracy cells hold. The narrowest is correlated PM: 235 fewer failures per
million, with the interval's end nearest zero at -171.6 per million.

## 2. Preservation: q = 8 % against q = 10 % on the same T6 shots

| item | value | plain meaning |
|---|---|---|
| failures | 2,708 at q = 8 %, 2,698 at q = 10 %, of 1,000,000 | the same read decoded at both thresholds |
| discordant shots | 113 fail only at q = 8 %, 103 only at q = 10 % | the 216 shots on which the thresholds disagree |
| difference LER(q = 8 %) - LER(q = 10 %) | +10.0 per million | q = 8 % failed 10 more times in a million shots |
| 97.5 % Newcombe interval | [-23.3, +43.3] per million | the range of differences T6 is consistent with |
| margin delta | 81.69 per million (8.1688 x 10^-5) | half of study 4's established advantage over correlated PM; about 3.04 % of study 4's LER |
| rule | preserved iff the interval's upper end < delta | 43.3 < 81.7 per million |
| result | preserved | |
| McNemar p | 0.54 | no evidence of a difference in either direction |

The margin is a tolerance. A pass shows that T6 excludes a loss of more than 81.7 failures per million shots (about
3.04 % of study 4's error rate); a smaller loss is still possible. The stated power (0.994 and 0.554) assumed V17b's discordance rate,
59 discordant shots per 200,000 (295 per million). T6 had 216 per million, so its interval (half-width 33.3 per
million) came out narrower than the planned 38.5.

## 3. Composite time

Ours = (a) GPU term + (b) CPU term on the Grace basis + (c) handoff, in µs per shot, as study 4 registered it. The
Grace basis is study 4's reference CPU: a CPU time measured on another node is scaled by 714.05 µs (correlated PM per
block at d = 17 on that basis) over the same node's correlated-PM time.

| term | value | source | measured in R17? |
|---|---|---|---|
| (a) GPU term | 66.6595 µs | study 4's phase A (`phase_a.json`, `cands.d17_b5_r787.a_counted`); engine FROZEN, because E3 ended NO-GO | no, inherited |
| (b) CPU term, Grace basis | 66.8160 µs | R17 CPU job J23 | yes |
| (c) handoff | 1.5432 µs | study 4's estimate (`phase_a.json`, `c_handoff`) | no, inherited |
| ours | 135.0187 µs (135.02) | | |

(b) was measured on one exclusive Alpine amem node (mem-node-01, AMD EPYC 7713, load average 1.25 at the start),
on core 8, in three interleaved repetitions, each in a fresh process.

| quantity | repetition 1 | 2 | 3 | used |
|---|---|---|---|---|
| term at q = 8 % (15,759 routed V17b shots; every decision equal to the development record) | 71.140 µs | 69.182 µs | 73.038 µs | max 73.038 µs |
| node's correlated PM per block (V17b chunk 0) | 780.54 µs | 785.61 µs | 763.09 µs | median 780.54 µs |
| term at q = 10 %, timed beside (19,957 routed shots; reported only) | 87.778 µs | 89.130 µs | 89.312 µs | max 89.312 µs |

(b) = 73.038 µs x 714.05 / 780.54 = 73.038 µs x 0.91482 = 66.82 µs, where 714.05 µs is the registered Grace-basis
time of correlated PM at d = 17. The q = 10 % term on the same node converts to 81.70 µs, so lowering q saved 14.89 µs
of CPU term on one node. Study 4's composite was 151.35 µs, with (b) = 83.14 µs from its own session; R17's composite
is 16.33 µs lower.

The opponents' times are study 4's registered constants. Margins are the opponent's time minus ours. The sensitivity
column replaces (a) with E3's same-session FROZEN median, 67.2015 µs (0.54 µs above (a)), which gives ours 135.56 µs.

| opponent | registered time | margin | margin with +0.54 µs | speed cell |
|---|---|---|---|---|
| plain PM | 325.23 µs | 190.21 µs | 189.67 µs | holds |
| correlated PM | 714.05 µs | 579.03 µs | 578.49 µs | holds |
| Ising M1 + PM | 161.33 µs | 26.31 µs | 25.77 µs | holds |
| Ising M4 + PM | 165.97 µs | 30.95 µs | 30.41 µs | holds |
| Ising M1 + correlated PM | 297.51 µs | 162.49 µs | 161.95 µs | holds |
| Ising M4 + correlated PM | 275.79 µs | 140.78 µs | 140.23 µs | holds |
| min-LLR BP-OSD | 444,502.44 µs | 444,367.43 µs | 444,366.88 µs | holds |
| BP 60 BP-OSD | 619,209.00 µs | 619,073.98 µs | 619,073.44 µs | holds |

The narrowest margin is Ising M1 + PM: 26.31 µs, or 25.77 µs with the sensitivity. No margin is smaller than
0.54 µs, so no speed cell depends on it. The terms come from different sessions: (a) from study 4's GPU session,
(b) from this Alpine job, and the opponents' times from study 4's sessions of 2026-10-06 and 2026-10-07.

## 4. The gates

| gate | result |
|---|---|
| step 1: the rehearsal's GPU step on the pod (dry pass pod20261010094236) | pins, ours_val, hopper and the read of T6dry each exit 0; 20 of 20 Ising rows with 0 of 200,000 decisions different from S4-R3's H100 read; M1 and M4 torch `residual_equal` and `frame_equal` True (`PASSED False`, as every dry hopper prints) |
| step 2: dry inventory J21 and dry analysis J22 | both exit 0; eight opponent lines, the preservation line and `R17 DRY claim (all 16 cells): NOT SHOWN` on 20,000 validation shots (no weight) |
| step 3: freeze commit 47dc748f | pushed at 02:52 PDT; `git diff --stat ef5c042e 47dc748f` over r17.py, r17.json, the policy, `alpine/`, pod.sh and deploy.sh printed nothing |
| T6 drawn after the freeze | sample job J24 started at 02:52:56 PDT, after the push |
| the T6 read's own checks (pod20261010100258) | pins with no file missing or different; ours_val PASS (582 failures, 19,957 routed, 20 V17b chunks equal to the development record); hopper `PASSED True` with the same 20 rows and torch checks; then the read of T6 (22 parts) |
| inventory before analysis | inventory J29 (4,709 files, sha256 3d2cc97a...) committed as f6779a84 and pushed at 04:38:27 PDT; the analysis began at 04:38:40 and checked every input against it |

## 5. Jobs, commits and the pod session

Times are PDT on 2026-10-10. The Alpine work directory is `$B = <cluster-scratch>/ising_r17`.

| step | where | id | time | result |
|---|---|---|---|---|
| venvs (frozen pod recipe) | pod | | 02:31-02:42 | torch 2.11.0+cu126, Triton 3.6.0, TensorRT 10.14.1.48.post1, modelopt 0.47.0, Stim 1.16.0, PyMatching 2.4.0, numpy 2.2.6 |
| push of the rehearsal inputs | Alpine to pod | | 02:31-02:34 | 7,314 files, sha256 equal |
| GPU dry pass | pod | pod20261010094236 | 02:42:37-02:48:43 | passed (section 4); 943 files pulled, sha256 equal |
| dry inventory, dry analysis | Alpine acpu | J21, J22 | 02:50-02:51 | exit 0 |
| freeze commit | hub | 47dc748f | 02:52 | deployed to `$B/r17` (114 files) |
| CPU term | Alpine amem, exclusive | J23 | 02:52:56-02:56:54 | (b) = 66.82 µs |
| sample: T5 chunks 0-19 again, then T6 | Alpine acpu | J24 | 02:52:56-02:53:49 | T6 manifest sha256 7860cceb... |
| manifest commit | hub | 214360ed | 02:54 | |
| push with T6 | Alpine to pod | | 02:54-03:02 | 7,482 files, sha256 equal |
| GPU read of T6 | pod | pod20261010100258 | 03:02:58-03:34:26 | exit 0; 5,451 files pulled, sha256 equal |
| BP-OSD share, chunks 0-6 | Alpine A100 80GB | J25 | 02:55-04:13 | min-LLR 33.58 shots/s, BP 60 27.19 shots/s |
| BP-OSD share, chunks 7-13 | Alpine A100 80GB | J26 | 02:55-04:37 | 23.80 and 22.11 shots/s |
| BP-OSD share, chunks 14-19 | Alpine A100 80GB | J27 | 02:55-04:25 | 23.32 and 21.38 shots/s |
| merge | Alpine acpu | J28 | 04:37 | both arms joined, 200,000 shots each |
| inventory | Alpine acpu | J29 | 04:37-04:38 | 4,709 files |
| inventory commit | hub | f6779a84 | 04:38 | deployed to `$B/r17` (116 files) |
| one-time analysis | Alpine acpu | J30 | 04:38:40-04:39:03 | exit 0 |
| analysis commit | hub | 4e56fb2b | 04:39 | |

The pod session ran on one RunPod pod with an NVIDIA H100 80GB HBM3 (driver 580.159.04, GPU-<uuid>),
an Intel Xeon Platinum 8470 host, Linux 6.8.0-136 with glibc 2.39, Python 3.12.3 and gcc 13.3.0, container
gpu-pod. The study lead rented it at 02:29. The DONE marker was written at 03:36:24, and the watchdog had stopped
the pod by 03:44, about 1.2 h of pod time. Its directories mirrored Alpine's:

- `<cluster-scratch>/ising_r17` on the pod: `logs/r17_gpu_pod20261010100258.out`, `out/gpu_pod20261010100258/`
  and `out/preds/` (the T6 read), and `out/gpu_pod20261010094236/` and `out/dry/preds_pod20261010094236/` (the dry
  pass); the venvs were in `<pod-home>/r17venv/` (`venv`, `venv_cudaq`).
- The pod's disk went with the pod; its outputs are on Alpine at the same paths under `$B`, each file checked against
  `$B/logs/pod_pull_<id>.sha256`.
- The small records are in this repository: `run/records/gpu_pod20261010100258/` (the T6 read) and
  `dry/rehearsal/pod_pod20261010094236/` (the dry pass).

`run/records/` also holds the small records of the CPU, sample, BP-OSD, merge and inventory jobs (logs and JSON, no
arrays). Their copies of `cpu_term.json`, `hopper.json`, `reads_ours_pm_ising.json` and the merged BP-OSD sidecars
have the sha256 that the inventory lists.

## 6. Departures from R17_READY.md

- Step 1 ran `pod.sh venvs` and `pod.sh push` at the same time; they write separate paths on the pod.
- The inventory job ran with `--dependency=afterok:J28` (the merge) only. Slurm refused `afterok` on the CPU
  job ("Job dependency problem") because that job had ended an hour earlier and left the controller's memory. The CPU
  job had COMPLETED with exit 0 at 02:56.
- BP-OSD shares 7-13 and 14-19 ran on the same node, 19-31 % slower than share 0-6 (shots per second, by arm). The
  outputs are unaffected; only the wall time grew (the last share ended at 04:37).
- BP-OSD read T6 while the CPU job was still timing. Plan sections 6 and 7.2 (step 7) put the CPU term before any T6
  read. R17_READY.md step 5 submitted the three BP-OSD shares after the manifest push but without a dependency on the
  CPU job (J23), and Slurm started them while that job was still running. The times below come from the job
  records; Alpine's node clocks print Mountain time (UTC-6), one hour ahead of PDT.
  - CPU job, `run/records/cpu_J23/r17_cpu_J23.out`, stamped by `date -Is` on the amem node mem-node-01:
    the last repetition of the q = 8 % term ended at 03:56:27-06:00, and the job's last step at 03:56:51-06:00
    (02:56:51 PDT).
  - BP-OSD share 0-6, job J25 on the A100 node gpu-node-01, stamped by `date -Is` in its log and by the
    `written` field of its sidecars (`r17.py`'s `now()`) on that node's clock: the job started at 03:55:08-06:00. Its
    min-LLR sidecar was written at 04:30:25-06:00 with `wall_s` 2,085 s. That interval is measured by `time.time()` on
    the node; it starts after the share's T6 chunks are loaded and ends when decoding ends. Decoding therefore began at
    about 03:55:40-06:00 (02:55:40 PDT), 71 s before the CPU job ended.
  - BP-OSD shares 7-13 and 14-19, jobs J26 and J27, both on the A100 node gpu-node-02 and on its clock:
    both started at 03:55:08-06:00. The same arithmetic (04:44:13 minus 2,941 s, 04:38:04 minus 2,573 s) puts the start
    of their decoding at about 03:55:12 and 03:55:11-06:00, about 100 s before the CPU job ended.

  The overlap does not touch the CPU term. That term was measured on validation shots (V17b) on an exclusive amem node
  that no BP-OSD share used, and it read no T6 shot. Its decisions all equal the development record's (section 3), and
  the first BP-OSD output was written at 04:30:25-06:00, after the CPU job had ended. The pod's GPU job, which read T6 only
  after its own checks, started at 03:02:58 PDT (its id is that time in UTC, taken from the box's clock by `pod.sh
  start`), also after the CPU job. Nothing was rerun.
- Nothing was repeated: no step failed, and the one-time analysis ran once.

## Summary

1. R17 re-registered study 4's d = 17 row at q = 8 % (tau 4.539173) and read 1,000,000 new T6 shots after the freeze.
2. Every gate passed: the pod's dry pass, the dry analysis, the empty freeze diff and the T6 job's own bitwise checks.
3. Ours at q = 8 % failed 2,708 times in 1,000,000 shots (2,698 at study 4's q = 10 % in the same read).
4. All 24 opponent rows were worse than ours (p < 0.025, intervals below 0), so all eight accuracy cells hold.
5. The closest opponent is correlated PM: 2,943 failures, -235 per million, 97.5 % interval [-299.1, -171.6].
6. Preservation: +10.0 per million, 97.5 % interval [-23.3, +43.3], below delta = 81.7 per million: preserved.
7. Ours takes 135.02 µs per shot = 66.66 (GPU, inherited) + 66.82 (CPU, measured) + 1.54 (handoff, inherited).
8. All eight speed cells hold; the narrowest margin is 26.31 µs against Ising M1 + PM, 25.77 µs with the +0.54 µs sensitivity.
9. Outcome 1 of PLAN_R17.md section 8: the paper's d = 17 rows become R17's, with the disclosures of sections 8-9.
10. Records: `run/` (manifest, inventory, analysis, job records); pod jobs pod20261010094236 and pod20261010100258.
