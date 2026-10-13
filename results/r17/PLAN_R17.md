# R17: re-registering study 4's d = 17 row at a lower routing share (frozen plan)

Branch `ising-r17` of the hub, from master 7fffb623. Written 2026-10-10 and revised the same
day after the stage-gate review (`VERDICT: REVISE`, ten findings, all accepted; section 12 lists what changed).
E3 ended NO-GO, so the engine is FROZEN (`r17.json` engine). The plan is frozen by the freeze commit (section 7.2),
made after the rehearsal on validation shots (section 10.2) and pushed before any T6 shot exists.

## 0. In short

- Study 4 registered d = 17 with the network `d17_b5_r787` and the threshold tau = 5.1220, which sends 10 % of
  calibration shots to correlated PyMatching ("q = 10 %"). On its test set T5 that gave 2,687 failures in 1,000,000
  shots at 151.35 µs per shot. R17 keeps the network, every opponent and every setting, and changes two things: tau
  becomes 4.539173126220703 (q = 8 %), and the test set is new (T6). The GPU engine stays FROZEN: the persistent
  engine eng17 would have replaced it after a GO, and stage E3 ended NO-GO (section 3).
- The new tau is the pre-stated rule C4.2's choice on deep17's calibration set K17d (200,000 shots). It is the rule's
  fallback: no q qualified, so the rule took the q with the fewest calibration failures. Study 4's own tau is the same
  rule's fallback on another calibration set (C17b). The decision to re-register at this tau came after a look at
  the exploratory set E17d; section 1.3 says what the paper must disclose.
- T6 has 1,000,000 new shots with seeds 91,170,000 + j (j = 0..99), a namespace that no branch of the hub, the
  release or the paper repository uses (section 2.2). BP-OSD reads the first 200,000.
- All eight opponents of study 4 are re-run on T6 with study 4's exact variants (24 rows; section 4).
- Accuracy uses study 4's 16 cells at alpha = 0.025 for q = 8 %, plus one pre-stated non-inferiority test of q = 8 %
  against q = 10 % on the same T6 shots. Accuracy counts as preserved if the upper end of the 97.5 % interval of the
  difference in logical error rate lies below 81.7 per million shots, half of the advantage over correlated PM that
  study 4 established. That margin is a tolerance: it allows about 3.04 % relative degradation of study 4's error
  rate, so "preserved" does not mean "unchanged" (section 5).
- Our CPU term is measured again on the new validation mask with study 4's protocol. The GPU term (66.66 µs), the
  handoff (1.54 µs) and every opponent's time are study 4's registered constants (section 6).
- The GPU read (ours, PM and every Ising part) runs in one job on a RunPod H100 80GB HBM3, the registered GPU, after
  the same job has reproduced registered H100 outputs bit for bit. The plan's first choice was an Alpine H200, but the
  rehearsal showed that ours cannot run there: the router library needs glibc 2.38, and Alpine's container has an
  older one (section 7.3). BP-OSD runs on Alpine A100 80GB cards and the CPU term on an exclusive Alpine EPYC 7713
  node (section 7). The run needs about 3 hours of H100 pod time and, as projected from the rehearsal's rates, about
  3.8 A100 80GB card-hours.
- Section 10 lists each script's dry run and the rehearsal on validation shots. Every step that runs on Alpine passed
  (sample, CPU, BP-OSD shares, merge); the GPU step and the analysis on its predictions run on the pod first, and the
  freeze commit follows them (7.2). No T6 shot has been drawn.

## 1. The configuration

### 1.1 Unchanged from study 4 (S4-R3)

| what | pinned as | sha256 or value |
|---|---|---|
| frozen code | the tree of freeze c500ef8e, commit f1c6a0e9; `icmp.t4.frozen_checks` re-hashes its 157 pinned code files in every job | `data/study4/freeze.json` c500ef8e404ca38b... |
| network | `art/d17_b5t_r787_final_ema.pt` (5 blocks x 128 channels) | bc00eb30fb2be3f1... |
| deployed executable | FROZEN (`timing/engine.py`) with 4 pipeline stages, and the router library `lib/libs7router.so` with compiled correlated PM | a939d9c7b25fc5c4... |
| development record | `devrec/d17.npz`: c, routing and decision of all 200,000 V17b validation shots from study 4's development H100 | 8e0ead565bbe3a00... |
| circuit | d = 17 rotated surface code, X memory, 17 rounds, legacy noise p = 0.006 | 5a17d28287e216cc... |
| opponents | plain and correlated PM (PyMatching 2.4.0); Ising M1 (Fast) and M4 (Accurate), each as 8 TensorRT plans and PyTorch, each + plain and + correlated PM; BP-OSD min-LLR and BP 60 (CUDA-Q QEC 0.8.0) | `r17.json` method, parts |
| statistics | `icmp.stats2.paired`: exact two-sided McNemar test, Newcombe method-8 interval at 97.5 %, alpha = 0.025 | freeze route.alpha |
| software | numpy 2.2.6, Stim 1.16.0, PyMatching 2.4.0, scipy 1.18.1, torch 2.11.0+cu126, Triton 3.6.0, TensorRT 10.14.1.48.post1; BP-OSD's environment with numpy 2.5.3 | Alpine's `venv_gpu`, `venv_trt`, `venv_cudaq` |

`r17.json` holds every hash in full, together with S4-R3's records (commit a31d525e on `ising-worker`). `r17.py pins`
checks all of them on each host before anything else runs.

### 1.2 What changes

| what | study 4 | R17 | what it means |
|---|---|---|---|
| tau (q) | 5.121974468231201 (10 %), C4.2 on C17b | 4.539173126220703 (8 %), C4.2 on K17d (`policy_d17b5_r787_q08.json`) | a shot with \|c\| < tau goes to correlated PM; fewer routed shots, less CPU time |
| test set | T5 (seeds from a drand round) | T6 (fixed seeds 91,170,000 + j) | new shots that no decoder has seen |
| GPU engine | FROZEN | FROZEN (E3 ended NO-GO; eng17 would have replaced it after a GO) | unchanged; the decisions and the GPU term are study 4's engine's |
| our CPU term | 83.14 µs (q = 10 % mask) | measured again on the q = 8 % mask | the only timing term that tau changes |

On V17b, the 200,000 validation shots of study 4's timing, the new tau routes 15,759 shots (7.88 %) against 19,957
(9.98 %) at study 4's tau. Both counts come from the development record's c (`r17.py mask`, dry run).

### 1.3 How tau was chosen, and what the paper must say

The rule C4.2 takes the smallest q on the grid (0 and 1-30 %) whose calibration failures satisfy
F_cal(q) <= M - 2 sqrt(2M), where M = round(0.9 x correlated PM's failures on the calibration set). If no q qualifies,
it takes the q with the fewest calibration failures. On K17d, M = 536 and the bar is 470.5 failures. No q met it, and
q = 8 % had the fewest failures (540). Study 4's q = 10 % came out of the same fallback on C17b (bar 487.4), where q =
10 % had 560 failures and q = 8 % had 573, so the two calibration sets rank 8 % and 10 % in opposite order.

The study lead chose "registered artifact + q = 8 %" after seeing E17d (500,000 exploratory shots: 1,364 failures
against 1,356 at the registered tau, and a projected 137.3 µs against 155.8 µs). That choice used exploratory,
test-like data (review finding 3), so R17 tests it on new shots with a pre-stated criterion. The paper says three
things: the threshold came from the pre-stated rule on a calibration set; the decision to re-register at that
threshold followed an exploratory comparison on E17d; and T6 was drawn after that decision was frozen.

## 2. The test set T6

### 2.1 Definition

| item | value |
|---|---|
| shots | 1,000,000 in chunks j = 0..99 of 10,000; BP-OSD reads chunks 0-19 (200,000 shots) |
| seed of chunk j | 91,000,000 + 10,000 x 17 + j = 91,170,000 + j |
| sampler | `circuit.compile_detector_sampler(seed=s).sample(10_000, separate_observables=True)` on the frozen circuit 5a17d282..., with Stim 1.16.0 and numpy 2.2.6 on x86-64 with AVX2 (an Alpine AMD EPYC node; `r17.py sample` refuses other hosts) |
| file | `d17_p0.006_X_test6_{j:03d}.npz`, written by the frozen `icmp.t4.write_npz` (packed detector bits, observables, seed; stored members with fixed timestamps), so a file's sha256 depends only on its shots |
| manifest | `t6_manifest.json`: every chunk's seed and sha256, the circuit, the versions, the host, and the sha256 of `r17.py`, `icmp/t4.py` and `r17.json` |

The seeds follow the project's convention for fixed shot sets: base + 10,000 d + j, as with V17b's 69 M base and
deep17's 95-99 M bases. They are fixed here, before any shot exists. Unlike T2-T5 they do not come from a randomness
beacon. Two things guard against picking them after the fact: this plan fixes them in a commit before any decoder
sees a shot of that range, and the check in 2.2 shows that the range was never used.

### 2.2 The check that the range is new

`ising_compare/r17/seedcheck.sh` (git only) searched every text file on every branch of the hub (27 branches), the
release (8) and the paper repository (1) on 2026-10-10, 38 refs in all. It looks for any integer of the 91 M namespace
(91,000,000-91,999,999), whether written as digits, with underscores or commas, in exponent form, or as a base
("91 M", "91 * 10**6"). R17's own directory is left out.

    repositories: 3; branches searched: 38; hits: 0
    smallest seed of each registered test-set manifest: T2 4.628e18, T3 4.613e18, T4 4.639e18, T5 4.621e18 (2^62 = 4.612e18)

The same search for V17b's base (69 M) finds 1,874 lines, and for deep17's M17 base (95 M) it finds 5,758. Among them
are the places where those bases are defined (`speed2/s19/vsets.py`, `deep17/common.py`, `PLAN_deep17.md`), so the
search does see seeds written the way this project writes them. Training streams start at 10^8 + 10^6 x run
(100,000,000 or more). deep17's namespace survey of 2026-10-08 (its plan, section 5) found the shot bases 20, 30, 40,
50, 60, 62-69, 80, 85, 90 and 96 M in use, and deep17 then took 95, 97, 98 and 99 M; 91 M is in neither list.

### 2.3 When and where it is drawn

After the post-E3 freeze commit is pushed (7.2), `alpine/r17_sample.sbatch` draws T6 on one Alpine CPU node. The
manifest is committed to the hub as `ising_compare/r17/run/t6_manifest.json` and pushed before any GPU or BP-OSD job
starts. Later steps check that T6 was drawn under the same `r17.json`. In the dry run, this sampler on Alpine
reproduced registered shots bit for bit: V17b chunks 0-1 under their own seeds (content sha256 equal to G14's
manifest) and T5 chunks 0-19 (every file equal to S4-R3's manifest), on an AMD EPYC 9534.

## 3. Ours on T6

One read gives both thresholds. The frozen executable (`speed2.s19.t4ours`'s path: `build4`, then `do_chunk` per
chunk) runs at the registered tau exactly as in study 4. For every shot it records the logit c, the routing
|double(c)| < 5.1220 and the decision: c > 0, or the router's compiled correlated PM for a routed shot. A shot routed
at q = 8 % is also routed at q = 10 %, so its correlated-PM decision is already in the record. The q = 8 % decisions
are

    decision_q08 = correlated PM's decision where |double(c)| < 4.539173126220703, else (c > 0)

The analysis checks both thresholds against the frozen routing semantics on every shot (`icmp.s4_analyze.check_ours`,
with the PyMatching wheel's correlated decisions from the `pm_corr` row), and checks that the q = 8 % routed set lies
inside the q = 10 % one. This is the reuse of saved logits for both thresholds that review finding 13 asks for; the
BP-OSD subset uses the first 200,000 of the same decisions.

E3 (the H100 go/no-go of the persistent engine eng17, branch `ising-engine17` at 82e28614) ended NO-GO: eng17 was
exact (0 mismatches) and its sanitizers were clean, but its whole shot took 74.57 µs against FROZEN's 67.20 µs in the
same session, above the bound of 56.66 µs. `r17.json` records the verdict and pins E3's summary
(`e3/E3_SUMMARY.md`, sha256 a48b7588...). The engine is therefore FROZEN, study 4's, and R17 never runs eng17.

For Hopper numerics (review finding 7), the read runs only after the same job has shown on its GPU that c equals the
development record bit for bit on all 200,000 V17b shots, with routing and decisions equal too (the frozen
`speed2.s19.t4ours val --chunks 0-19`, unchanged). The plan's first choice was an Alpine H200 (compute capability 9.0,
like the H100). The rehearsal's GPU job there stopped at this step: the frozen router library `lib/libs7router.so`
needs glibc 2.38 (its `BUILD.json` says so), and Alpine's container `pyfull.sif` has an older glibc. The whole GPU read
therefore runs on an H100 pod with S4-R3's environment (7.3). Never on an A100: `E2_A100.md` documents LayerNorm
rounding differences there.

## 4. The opponents and their variants

| opponent | rows on T6 | variants (exact) | where it runs |
|---|---|---|---|
| plain PM | `pm` | PyMatching 2.4.0 on the circuit's DEM | the GPU job's CPU cores (14 processes) |
| correlated PM | `pm_corr` | the same, `enable_correlations=True` | same |
| Ising M1 + PM | 5 | the 4 registered TensorRT plans `M1_fp16_L3_B1`, `M1_fp16_L5_B1`, `M1_fp8_L3_B1`, `M1_fp8_L5_B1` (S4-R3's session plans, sha256 0f0e051e..., 6ef72bd1..., 75b79f06..., 987f9d31...) and `M1_torch` (NVIDIA's PyTorch module); plain PM on the residual | pre-decoder on the GPU job's H100 after its check; PM on the job's CPU cores |
| Ising M4 + PM | 5 | `M4_fp16_L3_B1` (2904e22c...), `M4_fp16_L5_B1` (4167617f...), `M4_fp8_L3_B1` (050bd9f6...), `M4_fp8_L5_B1` (a2b83609...), `M4_torch` | same |
| Ising M1 + correlated PM | 5 | the same M1 parts, correlated PM on the same residual | same |
| Ising M4 + correlated PM | 5 | the same M4 parts | same |
| min-LLR BP-OSD | `bposd_minllr`, chunks 0-19 | 10 BP iterations, min-LLR initialisation, OSD-CS order 10, bp_batch_size 2048 (`icmp.minllr.NV`) | Alpine A100s |
| BP 60 BP-OSD | `bposd_bp60`, chunks 0-19 | 60 BP iterations, final-LLR initialisation, otherwise the same | Alpine A100s |

These are study 4's 24 rows: every timed batch-1 plan of S4-R3's session (M1 / M4 x fp16 / fp8 x optimizer levels
3 / 5) plus torch. S4-R3 dropped its batch sweep before any Ising read, and R17 reads the same set. An opponent's
accuracy cell holds only if ours beats every one of its rows.

To share work (review finding 13), each Ising pre-decoder runs once per chunk, and its residual is packed (612 bytes a
shot) and decoded by plain and by correlated PM from the same array. Raw detectors go to the PM processes packed as
stored. The PM processes work while the GPU runs the next chunk or part, as in S4-R3's read.

### 4.1 Hopper numerics for the Ising parts

A TensorRT plan is bound to its TensorRT version (10.14.1.48.post1 here and in S4-R3) and to the GPU's compute
capability (9.0 on both H100 and H200), and its kernels were chosen on the H100. Whatever GPU reads T6, the GPU job
first checks there that the parts reproduce S4-R3's H100 outputs (`r17.py hopper`); on the H100 pod (7.3) this checks
the pod's environment, on an H200 it would also check the device:

- every part (8 plans and torch) on T5 chunks 0-19 (200,000 shots), + plain and + correlated PM, against S4-R3's
  decisions for the same row from its H100 read (the 20 row directories copied from the Mac mini, each checked
  against its part digest in `reads_ising.json`): every decision must be equal;
- M1 and M4 torch on the first 10,240 V17b shots, which are the first 20 batches of 512 of S4-R3's val step (that
  step forwarded all of V17b in one call), with the first 10,000 residuals and logical frames compared bit for bit
  with S4-R3's files `res_d17_M*_v17b000.npz`.

If any row or either torch check differs, or ours does (section 3), the job stops before its first T6 read. On an
H200 that sends the whole GPU read to an H100 (7.3); on the H100 pod it stops R17 for a decision.

This departs from the study lead's wording ("bitwise equality with registered H100 outputs on the 200,000
validation shots"), because no registered per-shot H100 outputs of the TensorRT plans exist on validation shots.
S4-R3 kept a validation verdict for each plan from its timing session, ran its val step with the torch numerics, and
kept per-shot plan outputs only from its T5 read. T5's role as a test set ended with study 4's one-time analysis,
and nothing in R17 is chosen from it. The review accepted these checks as sound empirical evidence for H200 use
and found no further H100 reference run necessary (its finding 6).

S4-R3 built the 8 registered plan files on its timing pod (`/workspace/s4/work/s5/work/plans/`). The study lead recovered
them from that pod's archive into `~/work/plans_s4r3_t1` (with `SHA256SUMS`); every file's sha256 equals its pin
in `r17.json`. `deploy.sh plans` checked each one again and installed them in `$B/bundle/plans` on 2026-10-10, and
`r17.py pins` re-hashes them in every job.

## 5. Accuracy: the cells, the preservation test and the analysis

### 5.1 The 16 cells (unchanged from study 4)

Each of the 24 rows is compared with ours at q = 8 % on the same shots (the BP-OSD rows on chunks 0-19), using the
exact two-sided McNemar test and Newcombe's method-8 97.5 % interval of LER(ours) - LER(row). A row counts as
"better" when p < 0.025 and the interval lies below 0. An opponent's accuracy cell holds when all its rows are better.
A speed cell holds when our composite time is strictly below the opponent's. The claim needs all 16 cells (an
intersection-union test, so no multiplicity correction).

### 5.2 Preservation of accuracy (q = 8 % against q = 10 %)

| item | value | what it means |
|---|---|---|
| comparison | ours at q = 8 % against ours at q = 10 %, all 1,000,000 T6 shots, paired, from the same read | the same shots decoded at both thresholds |
| statistic | `icmp.stats2.paired` at alpha = 0.025: diff = LER(q = 8 %) - LER(q = 10 %) and its 97.5 % Newcombe interval [lo, hi] | positive diff means q = 8 % fails more often |
| margin delta | 8.1688 x 10^-5 | a tolerance: up to 81.7 more failures per million shots, about 3.04 % of study 4's LER |
| rule | preserved iff hi < delta; the McNemar p is reported beside it | a one-sided non-inferiority test at 1.25 % |

The margin comes from study 4's narrowest accuracy cell, correlated PM. On T5, LER(ours) - LER(correlated PM) was
-2.20 x 10^-4, with 97.5 % interval [-2.77, -1.63] x 10^-4. The end nearest zero, 1.63 x 10^-4, is the advantage that
study 4 established with its own registered statistic. delta is half of it, so R17 may give up at most half of that
advantage (the 50 % retention rule common in non-inferiority designs). Its source is study 4's registered result
(`r3_summary.json`, pinned); the analysis recomputes delta from that file and stops if it differs. Neither
E17d nor K17d enters it. delta is therefore a tolerance that allows about 3.04 % relative degradation of study 4's
LER (81.69 against 2,687 failures per million on T5). A pass shows that T6 excludes a loss larger than that
tolerance; it does not show that accuracy is unchanged.

Planning numbers, from the development record on V17b before any T6 shot exists: in 200,000 shots, q = 8 % fails 581
times and q = 10 % fails 582 times; 29 shots fail only at q = 8 % and 30 only at q = 10 %. The power figures below are
normal approximations and hold only at that discordance rate (59 discordant shots in 200,000). At 1,000,000 shots it
gives a standard error of about 17.18 per million and an interval half-width of about 38.5 per million, so the test
passes with probability about 0.994 if the true difference is zero, and about 0.554 if q = 8 % is truly worse by half
the margin (40.8 per million). More discordant shots on T6 would widen the interval and lower both.

"Preserved" supports the statement that q = 8 % keeps study 4's accuracy within the tolerance (81.7 per million,
about 3 % relative).
"Not shown" means that T6 cannot exclude a loss of half the advantage, and the paper keeps study 4's row (section 8).

### 5.3 The one-time analysis

`r17.py analyze` reuses study 4's code through a thin driver, as S4-R3 did. It re-hashes the T6 manifest and every
file, loads every row with `icmp.test_analyze.load_part`, runs `check_ours` for both thresholds and
`icmp.stats2.paired`, and computes the composite with r3.py's frozen arithmetic. It refuses any part set other than
the 24 rows plus ours. Order:

1. `WHAT=inventory` writes the sha256 of every input to `final_inputs.sha256`: T6's files, every prediction chunk and
   sidecar, the merged BP-OSD outputs, and the CPU-term, hopper and record files. It is committed to the hub as
   `ising_compare/r17/run/final_inputs.sha256` with the exact command, and pushed.
2. That commit is deployed (`deploy.sh code X`). `WHAT=analyze INV=$B/r17/run/final_inputs.sha256` then checks every
   input against the inventory and runs once into `$B/out/final`, which must not exist yet.
3. Its outputs (`r17_summary.json`, `r17_eval.csv`, the log, the exit code) are committed and pushed whatever they
   say. A run that stops before writing its summary may be repeated once after a fix that does not touch the
   statistics, and the repeat is disclosed.

## 6. Timing

Ours = (a) GPU term + (b) CPU term on the Grace basis + (c) handoff, as registered.

| term | source | value | why |
|---|---|---|---|
| (a) | study 4's phase A (`phase_a.json`, a_counted); E3 ended NO-GO, so the engine is FROZEN | 66.66 µs (66.6595), inherited | same engine, same protocol; E3's same-session FROZEN time, 67.20 µs, is a reported sensitivity (section 8) |
| (b) | the R17 CPU job (below) | measured | tau changes it |
| (c) | study 4's value (E3 ended NO-GO and measured no handoff with study 4's procedure) | 1.543 µs (1.5432), inherited | stated as an estimate (review finding 10) |

(b) uses study 4's protocol (review finding 10). `alpine/r17_cpu.sbatch` runs on one exclusive Alpine amem node with an
AMD EPYC 7713 (`-p amem --qos=mem-normal -N 1 --exclusive --mem=256G --constraint=epyc-7713`), core 8, after the
freeze and before any T6 read. It runs three interleaved repetitions of three things:

- the node's plain and correlated PM per block on V17b chunk 0 (the frozen `speed2.s11.cpu12 ref`, unchanged);
- the term on the q = 8 % mask (`r17.py cputerm`, cpu12's own loop: 50 warm-up decodes, then one correlated decode
  per routed V17b shot in order with perf_counter around each, every decision equal to the development record's);
- the same term on the registered mask, beside it.

Then

    (b) = max(term_q08 over 3 repetitions) x 714.05 / median(the node's correlated PM per block over 3)

where 714.05 µs is the registered Grace-basis time of correlated PM at d = 17. The q = 10 % term timed beside it is
reported and does not enter (b). It measures the saving on one node, free of the cross-session difference to study
4's CPU job.
The mask comes from the development record's c (H100). The GPU job's `ours_val` step shows that the device reading T6
reproduces that c bit for bit, so the CPU term needs no GPU pass of its own.

The opponents' times are study 4's registered constants (review finding 10's second option):

| opponent | µs per shot | source | re-measured? |
|---|---|---|---|
| plain PM | 325.23 | freeze route.speed.grace_numerators_us | no: an extrapolation to Grace; a new node time would change the protocol |
| correlated PM | 714.05 | same | no, same reason |
| Ising M1 + PM | 161.33 | `phase_a.json`: fastest validated plan 62.60 + PM share 98.73 | no: the plan time needs an H100 session, and the H200 is not the registered timing GPU |
| Ising M4 + PM | 165.97 | 84.688 + 81.282 | no, same |
| Ising M1 + correlated PM | 297.51 | 62.60 + 234.91 (`pmshare_r3.json`: Apple M4 core x d = 13 ratio) | no (see below) |
| Ising M4 + correlated PM | 275.79 | 84.688 + 191.11 | no, same |
| min-LLR BP-OSD | 444,502 | `phase_a.json`, fastest of 3 on S4-R3's timing pod | no |
| BP 60 BP-OSD | 619,209 | same | no |

Only the correlated-PM shares after Ising have a cheap same-protocol repeat (the Mac mini, about 10 s). R17 does not
change what they measure: the inputs are the V17b chunk-0 residuals from S4-R3's val step. Both cells have margins
above 120 µs. When S4-R3 measured those shares, the plain-PM shares it timed beside them came within 0.5 % of the
registered ones from another machine (98.91 against 98.73 µs and 80.87 against 81.28 µs).

The results will carry a cross-session caveat: our (a) comes from study 4's GPU session, our (b) from a new Alpine
job, and the opponents' times from study 4's sessions of 2026-10-06 and 2026-10-07. The narrowest speed margin is
Ising M1 + PM at 161.33 µs. The dry runs put (b) at 67-75 µs (section 10), so ours lands near 135-143 µs, a margin of
about 19-26 µs against S4-R3's 9.98 µs. Within S4-R3's GPU session, the five fresh processes of the GPU term spread
over 0.04 µs (66.62-66.66 µs). The only other session that timed FROZEN is E3's: 67.20 µs per shot (median of five
fresh processes, 67.10-67.29), 0.54 µs above the registered value, under E3's own development protocol. R17 keeps the
registered 66.66 µs and reports the +0.54 µs as a sensitivity (section 8); no further timing session is bought.

## 7. Compute placement, order and hours

### 7.1 Where each step runs

| step | where | resources | wall time (estimate) | what it means |
|---|---|---|---|---|
| deploy | box (git) to Alpine | - | minutes | R17's commit, S4-R3's records, the bundle, the plans, S4-R3's T5 decisions |
| CPU term | Alpine amem, one exclusive EPYC 7713 node | 64 cores held, one used | about 10 min | outside QEC's 80-core acpu cap (another partition), as study 4's job was |
| T5 again, then T6 | Alpine acpu | 2 cores (5 allocated for 16 GB) | about 3 min | in the rehearsal, T5 chunks 0-19 took 3 s (already present) and two V17b chunks 4 s |
| checks and T6 reads (the GPU job) | an H100 pod (7.3); Alpine's H200 cannot load ours' router library | 1 H100, 16 processes | about 1 h, plus a 15-min dry pass first | S4-R3 needed 11 min (ours and PM) and 26 min (20 Ising rows) on an H100 |
| BP-OSD | Alpine aa100, A100 80GB cards | three single-card share jobs (chunks 0-6, 7-13, 14-19), each both arms in turn, 4 workers, 80,640 MiB | about 1.4 h wall if the three run at once; about 3.8 card-hours (projections) | see below the table |
| merge, inventory, analysis | Alpine acpu | 2 cores (9 allocated for 32 GB) | about 15 min | |
| Mac mini | not used | | | the PM shares are inherited |
| RunPod | one H100 pod for the GPU job (7.3) | 1 H100 80GB HBM3 | about 3 h of pod time | rented and stopped by the study lead |

The BP-OSD hours are projections (review finding 8). A worker needs about 16.9 GB of host memory at `bp_batch_size` 2048
(the rehearsal's 4-worker share peaked at 67.7 GB), and Alpine grants at most 80,640 MiB per A100 80GB, so a card runs
four workers: with five at 76 GiB the rehearsal lost one to SIGKILL, and the frozen `decode_parallel4` then waits for
it forever (10.2). The rehearsal's 4-worker share decoded 10,000 shots per arm at 32.4 (min-LLR) and 26.3 (BP 60)
shots/s per card. At those rates the 200,000 shots of each arm take 1.71 + 2.12 = 3.83 card-hours, and a share of 7
chunks about 1.4 h. The draft's 5.3 card-hours came from a 2,500-shot pilot (22.7 and 19.4 shots/s with 5 workers,
startup included). The rate of an A100 40GB with 2 workers has not been measured; if it equals the 80 GB card's rate
per worker, a 40 GB card decodes at half the speed of a 4-worker 80 GB card, and the draft's mixed schedule (BP 60 on
six 40 GB cards) would cost several more card-hours (the review estimated 9.6 for it). Both arms are therefore scheduled on
80 GB cards, within the per-user cap of three; a 40 GB card takes a share only if no 80 GB card frees up, and then a
share of at most 3 chunks with `--mem` of at least 34G, so that it ends within the job's 4-hour limit.

QEC's limits on Alpine: acpu has 80 cores shared with the Cascade work, and R17's acpu jobs are small and run one at
a time. On gpu-normal, one user may hold at most 4 H200s, 3 A100 80GB and 6 A100 40GB at once, shared with QEC's
other jobs. An aa100 node has 3 GPUs, 64 cores and 240 GB of memory.

The H200 queue was the schedule risk of the plan's draft: at 01:40 MDT on 2026-10-10 every H200 and MIG slice was
allocated, and at 02:10 Slurm put a new H200 job's start days away. The rehearsal then closed the H200 path for ours
(7.3).

BP-OSD runs in shares (review finding 14): each share is a set of whole chunks with both arms decoded in turn, one job
per share, with study 4's decoder settings and several workers per GPU (4 on an 80 GB card, 2 on a 40 GB card, at
about 13.5 GB of GPU memory each). The frozen `decode_parallel4` pins each worker to its own 1/workers of the job's cores.
`WHAT=merge` joins the shares in chunk order and checks T6's observables.

### 7.2 Order

1. The review; revisions (section 12).
2. E3 ended NO-GO. `r17.json` records it with E3's summary (commit e8a05f10); the GO branch is void.
3. The rehearsal on Alpine (10.2), from commit d4cc2c9a: sample, CPU with three repetitions, BP-OSD in two shares,
   merge. Its H200 GPU job stopped at the first check (glibc), which moves the GPU read to an H100 pod (7.3).
4. On the H100 pod, before any T6 shot exists: the GPU job's dry pass on V17b, T5 chunks 0-19 and T6dry
   (`pod.sh start 1`); then, on Alpine, the dry inventory and the dry analysis of its predictions with the
   rehearsal's CPU and BP-OSD records. This completes the rehearsal.
5. The freeze commit, only after step 4 passed: it adds the pod step's records to `dry/rehearsal/pod_D/`, and
   `r17.py`, `r17.json`, the policy, the Alpine jobs, `pod.sh` and `deploy.sh` stay byte-identical to ef5c042e (the
   code of the BP-OSD shares; it differs from d4cc2c9a only in comments of `alpine/r17_bposd.sbatch`). Pushed. No T6
   shot exists.
6. Deploy the freeze commit to Alpine.
7. The CPU job (any time from here on, before any T6 read).
8. The sample job: T5 chunks 0-19 again, then T6. Commit and push `run/t6_manifest.json`.
9. Only after that push: T6 to the pod and the GPU job there (`pod.sh push`, `pod.sh start 0`), and the three
   BP-OSD shares on Alpine.
10. Merge, then the inventory (committed and pushed), deploy that commit, run the one-time analysis, and commit its
    outputs.
11. The release and the paper (section 8).

### 7.3 The GPU job on an H100 pod

The rule (review finding 3, the study lead's decision): if any H200 bitwise check fails, or no H200 starts within about
3 hours, the whole GPU read goes to an H100. The rehearsal settled it before any T6 shot existed. Its H200 job
(J06, a 3g.71gb slice, which started early by backfill) stopped at its first check, because ours could not load
the frozen router library: `GLIBC_2.38' not found (required by .../libs7router.so)`. The library's `BUILD.json` records
`glibc_needed: GLIBC_2.38`, and Alpine's container `pyfull.sif` has an older glibc. A new Alpine container would be a
new environment, so R17 does not build one. The whole GPU read runs in one job on one H100 pod in S4-R3's
environment: ours, plain and correlated PM and every Ising part, after the same checks (`read` accepts an H100 by
name). BP-OSD, the CPU term, the merge and the analysis stay on Alpine.

The study lead rents one RunPod pod: an NVIDIA H100 80GB HBM3; an Ubuntu 24.04 image with python3 3.12, gcc and
nvidia-smi (glibc 2.39: S4-R3's read pod recorded `Linux-6.8.0-106-generic-x86_64-with-glibc2.39`); at least 60 GB of
container disk. Its paths mirror Alpine's, so `alpine/r17_gpu.sbatch` runs there unchanged (`POD=1`: by bash,
without apptainer). `pod.sh` builds the venvs with the frozen pod recipe and copies the inputs from Alpine and the
outputs back, every file checked by sha256. Its copies were tested against a scratch directory on Alpine standing in
for the pod: 7,314 files pushed and 4 pulled, all equal, and a second pull refused to overwrite. The pod stays up from
the rehearsal's GPU step to the end of the T6 read, about 3 h: the venvs 10-20 min, each push 5-10 min, the dry pass
about 15 min, the T6 job about 40-50 min (S4-R3 needed 11 + 26 min for the same reads on an H100), and the waits for
the freeze, the sample and the manifest commit in between.

### 7.4 Commands

From the hub clone on the box (`~/work/repo`, branch `ising-r17`); $B = <cluster-scratch>/ising_r17;
"on Alpine" means a shell on the cluster in `$B`; POD_SSH is the pod's ssh command.

    # (A) the rehearsal's GPU step, before any T6 shot ($B holds ef5c042e and the rehearsal's outputs)
    bash ising_compare/r17/pod.sh venvs
    bash ising_compare/r17/pod.sh push
    bash ising_compare/r17/pod.sh start 1             # prints the id D
    bash ising_compare/r17/pod.sh status D            # until the log ends with "exit 0" and JOBDONE
    bash ising_compare/r17/pod.sh pull D
    # on Alpine: the dry inventory, then the dry analysis checked against it (job ids I, A)
    sbatch --export=ALL,DRY=1,WHAT=inventory,PREDS=$B/out/dry/preds_D,BPOSD=$B/out/dry/bposd_merged_J20,CPU=$B/out/cpu_J15/cpu_term.json,HOP=$B/out/gpu_D/hopper.json r17/alpine/r17_analyze.sbatch
    sbatch --dependency=afterok:I --export=ALL,DRY=1,WHAT=analyze,PREDS=<as above>,BPOSD=<as above>,CPU=<as above>,HOP=<as above>,INV=$B/out/dry/final_inputs_I.sha256 r17/alpine/r17_analyze.sbatch
    # (B) the freeze commit: the records of D, I and A into dry/rehearsal/pod_D/, commit, push (R17_READY.md step 3)
    # (C) the T6 run, from the freeze commit F
    bash ising_compare/r17/deploy.sh code F
    # on Alpine:
    sbatch -p amem --qos=mem-normal -N 1 --exclusive --mem=256G --constraint=epyc-7713 r17/alpine/r17_cpu.sbatch
    sbatch r17/alpine/r17_sample.sbatch
    # box: $B/shots/t6_manifest.json to ising_compare/r17/run/, its sha256 checked against the sample log; commit, push
    bash ising_compare/r17/pod.sh push                # now with T6
    bash ising_compare/r17/pod.sh start 0             # prints the id G; status G until "exit 0" and JOBDONE; then:
    bash ising_compare/r17/pod.sh pull G
    # on Alpine, after the manifest push (in parallel with the pod):
    sbatch --gres=gpu:a100_80gb:1 -c 20 --mem=80640M --export=ALL,GPUS=1,WORKERS=4,CHUNKS=0-6 r17/alpine/r17_bposd.sbatch
    sbatch --gres=gpu:a100_80gb:1 -c 20 --mem=80640M --export=ALL,GPUS=1,WORKERS=4,CHUNKS=7-13 r17/alpine/r17_bposd.sbatch
    sbatch --gres=gpu:a100_80gb:1 -c 20 --mem=80640M --export=ALL,GPUS=1,WORKERS=4,CHUNKS=14-19 r17/alpine/r17_bposd.sbatch
    sbatch --dependency=afterok:<the three> --export=ALL,WHAT=merge r17/alpine/r17_analyze.sbatch
    sbatch --export=ALL,WHAT=inventory,CPU=$B/out/cpu_<C>/cpu_term.json,HOP=$B/out/gpu_G/hopper.json r17/alpine/r17_analyze.sbatch
    # box: $B/out/final_inputs.sha256 to ising_compare/r17/run/, commit, push; deploy.sh code <that commit>
    sbatch --export=ALL,WHAT=analyze,INV=$B/r17/run/final_inputs.sha256,CPU=<as above>,HOP=<as above> r17/alpine/r17_analyze.sbatch
    # box: $B/out/final/ and the job's log to ising_compare/r17/run/final/, commit, push

`R17_READY.md` gives the same sequence with every command written out.

## 8. What the paper's d = 17 row will say

Pre-stated, by outcome:

1. All 16 cells hold and accuracy is preserved. The d = 17 rows of the accuracy and speed tables become R17's: T6,
   1,000,000 shots (BP-OSD on 200,000), our q = 8 % failures and LER, the routed fraction, every opponent's failures,
   our composite (a) + (b) + (c) with the engine named, and the opponents' registered times. One sentence goes beside
   it: "Re-registered at q = 8 % (R17) after an exploratory study; the threshold is the pre-stated rule's choice on a
   calibration set; on the same T6 shots q = 8 % fails X times against Y at study 4's q = 10 % (difference D per
   million, 97.5 % interval [lo, hi]; preserved within a tolerance of 81.7 per million, about 3 % of study 4's error
   rate)." Study 4 stays in the repository and the release as the earlier registration, with its own numbers.
2. All 16 cells hold, but preservation is not shown. The paper keeps study 4's row (q = 10 %, T5) and reports R17 in
   the release and in a footnote, as a registered re-test whose lower routing share did not show preserved accuracy.
3. A cell fails. The paper keeps study 4's row and reports R17 in full, naming the failed cells.

In every case R17's analysis output is published whole. The poster's study-4 numbers stay correct as they are.

Two statements go with every outcome:

- The preservation margin is a tolerance. A pass means that T6 excludes a loss of more than 81.7 failures per
  million (about 3.04 % of study 4's error rate), not that the accuracy is unchanged; the stated power (0.994 and
  0.554) assumed V17b's discordance rate.
- Speed sensitivity: our composite uses study 4's registered GPU term, 66.66 µs. E3 timed the same engine (FROZEN)
  at 67.20 µs in its own session, under its development protocol. With that value our composite would be 0.54 µs
  higher; the paper gives each speed margin beside this +0.54 µs and says whether any margin is smaller.

## 9. Disclosure and the E3 dependency

- The re-registration was decided after the exploratory deep17 read-out on E17d (1.3). The threshold itself comes
  from K17d, by a rule stated before deep17 ran.
- The GPU engine depended on E3's verdict. E3 ended NO-GO (eng17 exact, but 74.57 µs against FROZEN's 67.20 µs in
  the same session); `r17.json` pins E3's summary, and R17 reads and times study 4's FROZEN.
- Every GPU row comes from a RunPod H100 (7.3), the GPU model of study 4's reads, not from an Alpine H200 as the
  draft planned; the paper names the device of the read.
- The Ising hopper check uses T5's registered H100 decisions instead of validation shots (4.1).
- The opponents' times and our handoff are study 4's constants (section 6).
- The seeds are fixed in this plan rather than drawn from a beacon (2.1).
- BP-OSD runs on Alpine A100s. Study 4 registered A100s (Derecho); S4-R3 used H100 pods.

## 10. Dry runs and the rehearsal (validation and calibration shots only)

### 10.1 The first dry runs (the plan's draft)

The dry runs used `deploy.sh code` from the working tree, then from commits e0497d57 and 3f42d3c3 (the code of this
branch). No T6 shot was drawn or read. T6dry is a stand-in: V17b chunks 0-1 drawn by `r17.py sample --dry` under their
own seeds. The records are in `dry/` (dry/README.md).

| script | job | shots | result |
|---|---|---|---|
| `seedcheck.sh` | git, on the box | none | 0 hits in 38 refs; the positive controls (69 M, 95 M) found 1,874 and 5,758 lines |
| `deploy.sh` | code, bundle, reg, standin | none | every copy checked by sha256; `plans` waits for the plan files |
| `r17.py pins` | in each CPU and sample job | none | frozen code, Stim, numpy and circuit equal; 15 bundle files and 12 S4-R3 records equal; the 8 plans missing, as expected |
| `r17.py t5regen` | J01 | T5 chunks 0-19 (study 4's test set, already analysed) | every file equal to S4-R3's manifest |
| `r17.py sample --dry` | J01 | V17b chunks 0-1 | equal to V17b (G14's content sha256), written as T6dry |
| `r17.py mask` | J08 | V17b (development record) | 15,759 routed (7.88 %); failures 581 (q = 8 %) and 582 (q = 10 %); planning power 0.994 at no difference, 0.554 at half the margin |
| `r17_cpu.sbatch` (`cputerm`, `cpurecord`) | J02, J05, J08 | V17b | see the table below; every decision equal to the development record |
| `r17_bposd.sbatch` (`bposd`) | J04: A100 80GB, 5 workers | T6dry, first 2,500 shots per arm | 22.7 (min-LLR) and 19.4 (BP 60) shots/s |
| `r17_bposd.sbatch` | J07: A100 40GB, 2 workers | T6dry, 1,000 shots per arm | queued |
| `r17_analyze.sbatch` merge (`bposd-merge`) | J09 | J04's output | both arms joined; observables equal to T6dry's |
| `r17_gpu.sbatch` (`pins`, ours on V17b, `hopper`, `read`) | J03: whole H200; J06: H200 MIG 3g.71gb | V17b, T5 chunks 0-19, T6dry | still queued when the draft went to review; kept for the rehearsal (10.2) |
| `r17_analyze.sbatch` inventory and analyze | J10-J13, behind the GPU jobs | T6dry | cancelled on 2026-10-10: their inputs predate the review's fixes (a merged BP-OSD file with 2,500 predictions beside 20,000 observables, a one-repetition CPU record) |

The CPU dry runs (one repetition each, in a 5-core slot of a shared AMD EPYC 9534 node with a load average near 100;
not the registered protocol):

| job | code | node's correlated PM, µs per block | term at q = 8 %, µs (node) | term at q = 10 %, µs (node) | (b) on the Grace basis, µs | q = 8 % / q = 10 % |
|---|---|---|---|---|---|---|
| J02 | working tree | 661.0 | 69.03 | - | 74.6 | - |
| J05 | working tree | 697.3 | 65.69 | 85.14 | 67.3 | 0.77 |
| J08 | e0497d57 | 805.4 | 78.60 | 94.08 | 69.7 | 0.84 |

These are not R17's (b). The routed counts alone would predict a ratio of 0.79 (15,759 / 19,957). The q = 10 % term
beside them converts to 87.2 and 83.4 µs, against study 4's registered 83.14 µs. This spread between jobs is why the
protocol uses an exclusive node and three repetitions.

### 10.2 The rehearsal (after the review)

After the review, the chain ran again on validation shots with the revised code: commit d4cc2c9a for the sample and
CPU jobs and ef5c042e for BP-OSD (the two differ only in comments of `alpine/r17_bposd.sbatch`). No T6 shot was drawn
or read. The registered plans were installed first, and `$B/standin` held copies of them, because the queued GPU dry
jobs' scripts read plans from there. Records: `dry/rehearsal/` (dry/README.md).

| step | job | result |
|---|---|---|
| sample (`pins`, `t5regen`, `sample --dry`) | J14 | every pinned file equal, the 8 registered plans included; T5 chunks 0-19 equal to S4-R3's manifest; T6dry (V17b chunks 0-1) equal to V17b; manifest 758086cc... |
| CPU, three repetitions, dry (`mask`, `cputerm`, `cpurecord`) | J15 | every decision equal to the development record; see below |
| GPU on an H200 (`pins`, `ours_val`, then `hopper`, `read`) | J06 (a 3g.71gb slice of an H200 NVL) | `pins` passed; `ours_val` stopped: `GLIBC_2.38' not found (required by .../libs7router.so)`; `hopper` and `read` never ran (7.3). J03 (a whole H200, same container) was cancelled |
| BP-OSD, 5 workers per A100 80GB, 76 GiB | J17 (chunk 1), J16 (chunk 0) | J17: 30.30 (min-LLR) and 23.62 (BP 60) shots/s; J16 lost a worker to SIGKILL at the memory limit and waited for it, cancelled after 20 min |
| BP-OSD, 4 workers per A100 80GB, 80,640 MiB | J18 (chunk 0), J19 (chunk 1) | J18: 32.43 (min-LLR) and 26.25 (BP 60) shots/s per card, peak 67.7 GB; J19: 32.77 and 26.41 shots/s, peak 65.6 GB. Parent and workers passed the runtime checks; the sidecars record CUDA-Q QEC 0.8.0 (cudaqx 6ac00e5c), numpy 2.5.3, scipy 1.18.1, Stim 1.16.0 and PyMatching 2.4.0, equal to the pins |
| merge | J20 | both arms joined from the two shares: 20,000 predictions beside 20,000 observables (finding 1's fix); 197 (min-LLR) and 305 (BP 60) failures on these V17b shots |
| GPU on the H100 pod (dry pass), dry inventory, dry analysis | the pod session (R17_READY.md steps 1-2) | before the freeze commit; their records go to `dry/rehearsal/pod_<id>/` |

The CPU job ran on a shared AMD EPYC 9534 node with a load average near 100, so its numbers are not R17's (b): term at
q = 8 % 67.97, 81.95 and 72.39 µs on the node; the node's correlated PM 915.8, 727.6 and 751.8 µs per block; dry (b)
= 81.95 x 714.05 / 751.8 = 77.84 µs on the Grace basis; the q = 10 % term beside it converts to 113.46 µs. `b_term()`
accepts the record (three repetitions, one node).


## 11. Files

| file | what |
|---|---|
| `PLAN_R17.md` | this plan |
| `r17.json` | the record: every pin, the configuration, T6, the criteria, the timing sources |
| `policy_d17b5_r787_q08.json` | the q = 8 % routing policy and its provenance |
| `r17.py` | the driver: pins, sample, t5regen, mask, cputerm, cpurecord, hopper, read, bposd, bposd-merge, inventory, analyze |
| `alpine/r17_cpu.sbatch`, `r17_sample.sbatch`, `r17_gpu.sbatch`, `r17_bposd.sbatch`, `r17_analyze.sbatch` | the Alpine jobs (`r17_gpu.sbatch` also runs on the H100 pod with `POD=1`) |
| `deploy.sh` | code, bundle, plans, reg, standin (dry runs) |
| `pod.sh` | the H100 fallback (7.3): venvs, push, start, status, pull |
| `seedcheck.sh` | the seed-range check |
| `e3/E3_SUMMARY.md` | E3's summary (NO-GO), pinned by `r17.json` |
| `R17_READY.md` | the handoff for the T6 run: what changed, the rehearsal, the commands in order |
| `dry/` | the records of the dry runs and the rehearsal |
| `run/` | the run's records as they are committed (T6's manifest, the inventory, the analysis outputs) |

## 12. What changed after the review

the stage-gate review of commit 77c1b924 (`VERDICT: REVISE`) made ten findings; all were accepted.

| finding | what changed |
|---|---|
| 1 (HIGH) the dry analysis could not finish | `bposd-merge` saves the joined shares' observables beside their predictions (it saved T6's whole subset beside 2,500 predictions); dry CPU jobs run three repetitions, as `b_term()` needs; the rehearsal ran the chain end to end (10.2) |
| 2 (HIGH) BP-OSD's runtime pins unenforced | `r17.py bposd` imports what its workers import and every package of `versions_bposd`, runs `s4guard.check_runtime` against `versions_bposd` before any shot and before each output, and passes the freeze to `decode_parallel4`, so every worker runs the frozen worker check; the sidecars record the versions found. `versions_bposd.cudaq_qec` is now the full version string, as the freeze pins it |
| 3 (MED) the selective H100 fallback was not implemented | one rule, the whole GPU read on an H100 (7.3), with `pod.sh` and `r17_gpu.sbatch`'s `POD=1` branch |
| 4 (LOW) `engine.verdict` was null | `r17.json` engine: NO-GO, E3's commit and summary by sha256; (a) 66.6595 µs and (c) 1.5432 µs inherited (section 6) |
| 5 (LOW) wording of the margin and the power | the margin is a tolerance of about 3.04 % relative degradation; the power holds at V17b's discordance rate (5.2, 8) |
| 6 (LOW) the numerical checks | no change |
| 7 (LOW) NO-GO timing | the +0.54 µs sensitivity (E3's same-session FROZEN 67.20 µs) is reported with every outcome (section 8) |
| 8 (MED) the BP-OSD hours | the card-hours are projections, now from the rehearsal's full shares (about 3.8; the draft's 5.3 came from a 2,500-shot pilot); both arms on 80 GB cards, four workers per card because of host memory; the 40 GB rate is unmeasured (7.1) |
| 9 (LOW) the Ising validation pass | `hopper` forwards only the first 10,240 V17b shots, 20 full batches of 512 (4.1) |
| 10 (LOW) small optimisations | `bposd` no longer re-hashes all of T6 before reading; `t6_load` hashes every chunk it reads |
