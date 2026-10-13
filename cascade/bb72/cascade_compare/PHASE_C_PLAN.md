# Phase C of the BB72 Cascade comparison: accuracy curves, routing and speed (plan)

Drafted on 10 October 2026; revised the same day after the phase-C gate review (VERDICT REVISE, 4 HIGH, 3 MED, 3 LOW; every
finding accepted; the first revision), after the study lead's decision that the ConvLines networks are read, screened
and timed on their fused engine (2.4; the second revision), and after an independent check of both (VERDICT REVISE; the
third revision). The study lead's decisions on that check: every GPU step runs on one GPU model in one venv, a full RTX
PRO 6000 (5.1); a routed decoder needs its S screened (6); the screen requires our networks' paths to be bitwise and
judges each batch size alone (2.4); Cascade is timed alone beside the pipelines, in their job (5.3). A revision note,
not part of this release, listed each finding and decision, what changed and the dry run that shows it. Status: no step
of phase C has run; C1 waits for the study lead. The scripts ran end to end on dry-run slices (section 9; records in
`phase_c/dry/`). No shot of the development block (seed kind 10) or of the confirmation blocks (kind 11) has been drawn.
Dry runs 1-10 drew their timing inputs from the speed stream (kind 12); from dry run 12 on every dry run draws kind 13
only. A runbook, not part of this release, listed the commands in order, the hours of each part and what waits for C6'.
Phase C has since run up to the lock (`phase_c/RUN.md`); Amendment 1 (11 October 2026), made before any confirmation
read, changed the selection rule (4.3, step 3).

The work is exploratory, like every stage of `cascade_compare`: fresh shots from new seed kinds only; no registered test
set is read.

## 1. What phase C answers

The authors' four questions and the part of phase C that answers each:

| question | in plain words | phase C part |
|---|---|---|
| 1 | our registered networks r11 and r12, alone and with confidence routing, in Cascade's setting: accuracy and speed | curves of r11 and r12 (section 3); r12 + Relay-BP at the registered operating points, read once (section 4); speed (section 5) |
| 2 | our network retrained in Cascade's setting: accuracy and speed | curves and speed of S55' and S55'-deep |
| 3 | the reproduced large, deep Cascade network | curves and speed of C14' (and C6'), with the paper's own curve and timings beside them as external context |
| 4 | our small network routing its least confident shots to the reproduced Cascade network: accuracy and speed | the routed configurations S -> B and S -> B -> Relay-BP, chosen on the development block by the rule of section 4, read once on a fresh confirmation block, timed end to end |

The primary comparison (stage-gate review, finding 1) is our networks against our Cascade re-implementation on the same
circuits. Phase C makes it in two settings:

| setting | name in the code | what it is | Cascade there |
|---|---|---|---|
| ours | `uniform` | Relay-BP's BB72 X-memory circuits, R = 6 rounds, every noise channel at p | C14 (and C6) |
| Cascade's | `depspam_fm` | the same circuits with the noise variant of the noise-model fit (DEPOLARIZE1(p) after resets and before measurements, a noisy final measurement). It reproduces the paper's Tesseract, BP+OSD and Relay values within error; the noise-model fit describes it as a fit on 17-20 Tesseract failures, not an identification (`CASCADE_CAL.md`) | C14' (and C6') |

The paper's Fig. 2(a) values (digitised) and its timings stay in separate columns as external context and enter no
verdict.

## 2. Fixed inputs

### 2.1 Networks

All files are final EMA weights. The failures are those of each run's last monitoring point (the fixed monitor block,
seed kind 1: 200,000 shots at 0.3 %, 100,000 at 0.55 %). Every phase-C output records each file's full sha256.

| tag | file (Alpine, under `<cluster-scratch>/`) | trained in | network | parameters | failures at 0.3 % / 0.55 % | sha256 (first 16) |
|---|---|---|---|---|---|---|
| C14 | `cascade_w3/ckpt/C14/final_ema.pt` | uniform | Cascade, L 14, H 512 | 19.7M | 500 / 6,321 | 0f44e6141e5a0758 |
| C6 | `cascade_w3/ckpt/C6/final_ema.pt` | uniform | Cascade, L 6, H 512 | 12.3M | 682 / 9,830 | e20101ec1a721799 |
| S55 | `cascade_w3/ckpt/S55/final_ema.pt` | uniform | ConvLines-BB, 3 blocks | 483k | 1,162 / 13,022 | 87886c65fbb6cc49 |
| C14' (`C14p`) | `cascade_w3/ckpt/C14_dfm/final_ema.pt` | depspam_fm | Cascade, L 14, H 512 | 19.7M | 933 / 10,174 | dd0c3b1488eed58d |
| C6' (`C6p`) | `cascade_w4/ckpt/C6_dfm/final_ema.pt` (copied from the Windows workstation when it ends) | depspam_fm | Cascade, L 6, H 512 | 12.3M | still training | - |
| S55' (`S55p`) | `cascade_w3/ckpt/S55_dfm/final_ema.pt` | depspam_fm | ConvLines-BB, 3 blocks | 483k | 1,967 / 19,027 | 22fa0c832444d154 |
| S55'-deep (`S55p_deep`) | `cascade_w3/ckpt/S55_dfm_deep/final_ema.pt` | depspam_fm | ConvLines-BB, 6 blocks | 841k | 1,335 / 13,998 | 051370c37d5aef08 |
| r11 | `bb/ckpt/r11/final_ema.pt` (release `convlines_release`) | uniform, 0.3 % | ConvLines-BB, 3 blocks | 483k | - | 58e60897c3f9302c |
| r12 | `bb/ckpt/r12/final_ema.pt` (release) | uniform, 0.3 % | ConvLines-BB, 3 blocks | 483k | - | f2821c327b4e233a |

"Cascade" in every verdict means C14 in `uniform` and C14' in `depspam_fm` (question 3's large, deep network). C6 and C6'
(the paper's depth rule L ~ d) are a secondary opponent, reported beside it. The baseline is fixed here, before any
phase-C read (finding 12).

### 2.2 Code

Phase C runs the Cascade tree of `ising-cascade-net` at 64b2e25d (`bb.cascade_eval` with its prediction adapter,
`bb.cascade_data`, `bb.cascade_net`, `bb.engine`, `bb.route`) together with this branch's `cascade_compare/phase_c/`:

| file | does |
|---|---|
| `pc_common.py` | the fixed inputs of this plan (networks, seed kinds, p grid, block sizes, candidates, constants, the headline family's size, the one GPU model and its guard `gpu_env`), Eq. (1), Wilson's interval, Holm, the accuracy tolerance of a timed path (`agrees`), the frozen paths; shell helpers for the job files |
| `pc_eval.py` | `bb.cascade_eval` with the ConvLines networks on their fused engine and C6 and C6' under torch.compile (2.4); C14 and C14' exactly as in `bb.cascade_eval` |
| `pc_fused.py` | the fused engine of the ConvLines networks over a batch of shots: `bb.fused`'s kernels with a shot axis on the grid, bitwise the registered batch-1 engine (2.4) |
| `pc_relay.py` | full-DEM Relay-BP on the union of a read's routed sets |
| `pc_route.py` | `select`: the routing rule on the development block and the lock; `confirm`: the one analysis of a confirmation block |
| `pc_speed.py` | `nets`: every network's latencies; `summarize`: the frozen paths and the components the rule uses; `check`: the frozen paths against the evaluated computation at the claim points; `pipeline`: a locked configuration end to end, or Cascade alone beside the pipelines |
| `pc_report.py` | every table of the final report, the one Holm procedure of the headline family and the mechanical verdicts of section 6 |
| `scripts/` | `gpu.sbatch`, `cpu.sbatch` (the job wrappers; the scripts that copied the trees to the cluster and submitted the jobs are not in this release) |
| `jobs/` | the job files, one per step (`curves.sh`, `late.sh`, `relay_dev.sh`, `speed_nets.sh`, `select.sh`, `check.sh`, `confirm.sh`, `confirm_relay.sh`, `speed_pipe.sh`, `report.sh`), and the dry runs `dry*.sh` |

Accuracy reads go through `bb.cascade_eval` by way of `pc_eval.py`, which runs the ConvLines networks on their fused
engine, compiles C6 and C6' (2.4) and changes nothing else. Every network decodes the same rows; the per-shot
decisions, confidences, Cascade's logits or ConvLines' class index and the packed detector rows are kept, with every
checkpoint's sha256, the circuit's sha256 and each chunk's seed and detector digest.

### 2.3 Seeds

`bb.cascade_data.eval_seed(kind, p, chunk, family)` lives in the training stage's namespace (2^55 + ..., with a separate
range per family). A search of every hub branch for seed bases between 2^51 and 2^61 finds only this namespace. Stages
1-2 used kinds 0-7. Phase C uses four kinds nobody has used, in both families:

| kind | use | read when |
|---|---|---|
| 10 | the curves (every network alone) and the routing development block | step C1 |
| 11 | the routing confirmation blocks | once, after `locked_<setting>.json` is committed (C5) |
| 12 | inputs of the speed measurements: the screen of every timed path (C3), the claim-point check (C4b), the pipelines' shot streams (C7) | C3, C4b and C7 |
| 13 | dry runs (section 9), never reported; since the first revision every dry run, its speed inputs included, uses kind 13 only | done |

Relay-BP draws its own random numbers: each decoder keeps a generator (relay-bp's default seed is 0) for the gammas of
its relay legs, so a decoder kept across shots decides a shot according to what it decoded before. Dry runs 2 and 7
split the same unions over 32 and 24 processes and differed by 1 to 4 failures per union. `pc_relay.py` therefore
builds a new decoder for every shot, seeded with the shot's index in its read. A decision then depends only on the read
and the shot. Two runs with 32 and 13 processes and tasks of 10 and 7 shots decided every shot alike (dry run 8), and
so did a run on another CPU model (dry run 9). The timed pipelines (5.3) keep one decoder per process, as a deployed
decoder would; their failure counts are only a check, and the accuracy comes from C5 and C6. The two must not be
confused (gate finding 8): every timing of a decoder with Relay-BP is deployment timing with one retained decoder per
process, while every accuracy is the seeded reconstruction, which builds a new decoder for each routed shot. That build
is in no timing. Dry run 9's note put it at 14 ms; dry run 13 timed it at 7.3 ms on the EPYC 9555 (20 builds once the
detector error model exists), and C4b times it again on the timing node. A deployment that rebuilt the decoder per shot
would pay it on every routed shot: 7 to 280 us per incoming shot at routed fractions of 0.1 to 2 % for builds of 7 to
14 ms. The report prints this cost, from C4b's time, beside every pipeline with Relay-BP.

Relay-BP's draws change many of its decisions on the shots that reach it. On the dry unions at 0.3 %, four seed
families gave failure counts as far apart as 72 and 94 (on 524 shots), and two families disagreed on 46-86 of 524-670
shots. A decoder kept across shots failed as often as new seeded ones (dry run 11: 499.5 against 508.5 failures summed
over 8 unions, inside the spread between families). Phase C counts one draw per shot. Draws are independent between
shots, so each draw is part of its shot's randomness and the paired tests of 4.5 stay valid.

### 2.4 Precision, timed paths and the claim-point check (gate findings 1, 7 and 9; second revision)

Each decoder is evaluated through the path it is timed on, so that its accuracy and its speed describe one computation
(gate finding 1). After dry run 14 the study lead applied this to our networks (second revision).

The ConvLines networks (S55, S55', S55'-deep, r11, r12) run on their fused engine everywhere: the curves, the routing
thresholds, the claim-point check, the confirmation reads and every timing. The engine is `bb.engine`'s registered one,
`bb.fused`'s five Triton kernels (FP16 matmul inputs with FP32 accumulation; the residual stream, LayerNorm, gates, vote
and margin in FP32), and it decodes one shot per call. `pc_fused.py` runs the same kernels over a batch. Each kernel
gets one more grid axis, the shot, and only the base addresses of the shot's buffers move, so every decision and margin
is bitwise that of the batch-1 engine at any batch size. The reads use 4,096 shots per call (`FUSED_READ_B`). Dry run
15x showed the equality on 3,000 shots per network and p (S55, S55'-deep and r12 at 0.3 and 0.55 %, batch sizes 1, 7,
64 and 4,096). Every GPU step runs on one GPU model (5.1), so nothing rests on an equality across models. The screen
(C3) requires the equality at every batch size it times, and the check (C4b) requires it again on the claim points'
reads. On the rows of dry run 12 the fused reads failed as often as the BF16 reads within 8 failures at every point
(S55: 2,665 against 2,664 of 20,000 at 0.55 % in `uniform`), in half the GPU time.

The Cascade networks are read through `bb.cascade_eval --prec bf16`, the module under bf16 autocast in batches of 8,192
rows within each 10,000-shot chunk. That is their training precision: the bipartite convolution and the read-out
scatter run in bf16 Triton kernels, BatchNorm in eval mode. C6 and C6', the secondary opponent, are read through
`pc_eval.py`, which compiles their module with torch.compile (default mode, one graph per batch shape) and changes
nothing else. That saves about 0.36 GPU-hour over the two development blocks (gate finding 9), and C4b checks the
compiled reads against eager reads of the same rows by the tolerance below. C6' is read compiled only if its weights
reach Alpine before C4b; otherwise C1b reads it eagerly (`PC_EAGER=1`), since no check would compare its compiled read.
A Cascade network is timed on its fastest paths that pass the screen against this read. No such path is bitwise the
read: bf16 tiles depend on the batch size (C14's graph path at batch 1 changed 35 of 20,000 decisions at 0.55 % in dry
run 15b) and on the GPU model.

Each network is timed on up to three paths, each with its precision label, which the report prints beside every value:
`graph` (the evaluation backend captured in CUDA graphs, at every batch size; BF16), `compiled` (torch.compile, then
CUDA graphs; batch 1 and 4,096; BF16) and, for the ConvLines networks, `fused` (at every batch size: the registered
engine at batch 1, `pc_fused.py` above; FP16 matmul inputs with FP32 accumulation, never labelled BF16). A ConvLines
network's frozen paths are fused paths. Its graph and compiled paths are timed and screened for description only.

The accuracy tolerance (`pc_common.agrees`). A timed computation and the evaluated one are compared on the same shots
by what differs between them: only_t shots fail (or are routed) only in the timed computation, only_e only in the
evaluated one, and n_e is the evaluated count. They agree when

    |only_t - only_e| <= 0.01 n_e + 3 sqrt(only_t + only_e),

that is, when their counts differ by at most 1 % of the evaluated count beyond three standard deviations of the
symmetric flips that rounding alone produces. Comparing decisions shot by shot is too strict for Cascade: in dry run 1
the evaluation backend itself, run at batch 1 instead of 8,192, changed 34 of 20,000 C14 decisions at 0.55 % (bf16
rounding with batch-dependent Triton tiles) while failing 1,298 times instead of 1,299. Rounding flips shots both ways;
a computation that differs in substance moves only_t away from only_e.

The screen and the frozen paths (step C3). On fresh shots of the network's own setting (kind 12, chunks 0-9; 20,000 for
batch sizes below 64, else 100,000) every path at every batch size is compared with the network's accuracy path (its
read's computation, run chunk by chunk as the reads run it). The rule judges its decisions through their failures at p =
0.55 %, and its confidences through its routing: at each claim point (0.2 % and 0.3 %), the shots it routes below each
quantile of the evaluated confidence that the routing rule can use (0.1, 0.25, 0.5, 1, 2, 5, 10, 20 and 40 %: the grid's
q1, r and q). Beside these counts it records the shots whose decision differs (at 0.55 % and at both claim points) and
those whose confidence differs at all; a path with neither is bitwise the accuracy path. A ConvLines network's path
passes only if it is bitwise, as the check requires of its frozen paths (third revision); a Cascade path passes by the
tolerance. Dry run 13 showed why routing
belongs in the screen: S55''s fused engine passed on failures, but at 0.3 % it routed 9,670 shots to Cascade where the
BF16 margins of its read then routed 10,012 under the same threshold. In dry run 14 every compiled and fused path of S55
and S55' failed against the BF16 reads, on routing at the quantiles of 5 % and above, while their failures agreed: the
margins of the two computations are shifted against each other in both directions. Their frozen paths would then have
been BF16 graph paths, 0.40 ms per single shot where the fused engine takes 0.09 ms. Since the second revision they are
read on the fused engine. In dry run 15b every fused path of S55 and S55' (batch 1, 1,024 and 4,096) and of S55'-deep
and r12 was bitwise its read in every process, and the graph and compiled paths of S55 and S55' failed against the fused
reads as they had failed against the BF16 ones. A path is screened at a batch size when it passes there in every
process; a batch size that fails removes only that batch size's values from the candidates, not the path (third
revision). A network's frozen paths are its fastest screened paths for three uses: single (batch 1: the single-shot
latency and the single-mode pipelines), amortized (the lowest host-to-host time per shot over the screened batch sizes
up to 4,096) and batched (batch 4,096: the batched pipelines). The rule is the same for ours and for Cascade (stage-gate
finding 10), except that a ConvLines network's candidates are its fused paths. `speed_components.json` records the
frozen paths, the lock records its sha256, and the check and the pipelines refuse any other file, as the report does
(6). If a use has no screened path, the report describes the value of the graph path (for a ConvLines network, the fused
path), flagged, and neither the network nor a routed decoder whose S it is enters a headline (6). Nothing falls back
into eligibility.

The claim-point check (step C4b, after the locks and before any confirmation read; gate finding 1). The screen tests
each path alone, before any threshold is locked; the check tests what the headline uses, on shots the screen did not
see. C4b reads 100,000 fresh shots (kind 12, chunks 10-19; the screen read chunks 0-9) at each claim point, 0.2 % and
0.3 %, in each setting through `pc_eval.py` (`curves.sh`: every network, detector rows kept), and `pc_speed.py check`
runs the frozen paths on these rows:
- every network alone that can enter a verdict or stands beside one (our five, Cascade of the setting, and C6 or C6'),
  on its single and amortized paths: its failures against the read's;
- every locked configuration at the claim point as its pipelines run it: single (S and B on their frozen batch-1 paths)
  and batched (on their batch-4,096 paths, B on the routed rows in queue order), the locked thresholds applied to the
  timed confidences in float64 as the pipelines apply them. Against the same configuration routed by `pc_route.masks`
  on the read, it compares the failures, the shots routed to B and the shots routed to R, each by the tolerance above.
  Relay-BP decides every shot that either version routes to it with `pc_relay.py`'s seeded decoder, so a shot both
  route to R gets the same decision, and only the networks' differences remain;
- the compiled reads of C6 (C6') against eager reads of the same rows.
A ConvLines network's read must also be bitwise its frozen paths, every decision and every margin: otherwise the network
alone and every configuration whose S it is fail the check. A decoder enters a headline only if every one of its checks
passes at both claim points, and Cascade's own checks too (section 6). A failed check removes it, and no other path is
tried. The check writes `check_<setting>.json`, which is committed before C5. The check tests the computation, not the
decoders' accuracy, which comes from C1 and C5; its size keeps rounding noise small against the tolerance (at 0.3 % C14
fails about 250 times in 100,000 shots).

FP8 is not used. The paper's torchao FP8 does not reach Cascade's custom convolution and scatter (finding 9), and no
FP8 path has been implemented or validated here. So no FP8 number of ours is claimed, and the paper's FP8 timings stay
in their own column.

The curves of each network alone (section 3) are its development read. A ConvLines network's curves are therefore its
fused engine's, bitwise its timed paths. A Cascade network's curves stay on the evaluation backend, which its timed
paths match within the tolerance but not bitwise (see above), as for its routing and its confirmation. The earlier curve
study's and the training stage's curves of r11 and r12 used fp32; phase C's fused values replace neither, and both are
kept.

## 3. Accuracy curves

Rows. Each setting gets one development block: seed kind 10, chunks of 10,000 shots, at the paper's ten p values:

| p (%) | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 |
|---|---|---|---|---|---|---|---|---|---|---|
| shots | 10,000,000 | 3,000,000 | 1,000,000 | 500,000 | 500,000 | 300,000 | 300,000 | 300,000 | 300,000 | 300,000 |
| expected failures, C14 in `uniform` | ~310 | ~410 | ~400 | ~550 | ~1,250 | ~1,400 | ~2,500 | | | ~19,000 |
| expected failures, C14' in `depspam_fm` | ~580 | ~770 | ~740 | ~1,000 | ~2,300 | ~2,700 | ~4,700 | | | ~30,000 |

That is 16.5 million shots per setting. The expected counts scale each network's 0.3 % monitor rate by the shape of
the paper's Cascade curve. The dry run's 20,000-shot slices agree at 0.3 % (C14: 49 and 62 failures; C14': 80). With
about 300 failures or more, the 95 % interval of the best network's rate stays within about ±11 % of the rate, and
the worse networks have more failures and relatively narrower intervals. If C14 fails less than estimated at 0.10 %,
the block stays as fixed here; it is not extended after looking.

Networks. Every network decodes every row of both settings, including the networks trained in the other setting. This
reads the brief's "every trained network alone, in both settings" literally. Since the rows are shared, it costs only
GPU time, and it measures what a mismatch between training and evaluation setting costs every architecture. Question
1 needs that measurement: r11 and r12 were trained in `uniform` and are judged in `depspam_fm`. C6' joins later through
an add-on read of the same rows (same seeds). The analysis refuses to pair rows whose chunk detector digests differ.

Reported per network and p: block failures / shots; P_block with its 95 % Wilson interval; and P_L per logical qubit
per round by the paper's Eq. (1), P_L = (1 - (2 (1 - P_block)^(1/12) - 1)^(1/6)) / 2, at the estimate and at both ends
of the interval. Eq. (1) assumes p is known and is never applied to a difference.

Paired comparison with Cascade. Each network is compared with C14 (C14') on the same shots at each p: both discordant
directions (shots only it fails, shots only Cascade fails), the exact McNemar test, and Newcombe's matched-pair 95 %
interval for the difference of block failure rates (`bb.route.paired`, which calls `qldpc_neural/bench/paired.py`'s
`newcombe10`). The comparisons of our networks alone (S55, S55', S55'-deep, r11, r12, each at 10 p: 50 per setting)
belong to the headline family of section 6, judged by one Holm procedure at 0.05 over both settings together with
every locked routed configuration. A rejected comparison is "better" or "worse" when Newcombe's interval at its Holm
level lies below or above zero; every other comparison is "level". Comparisons among Cascade networks (C6 against
C14, the cross-trained ones) are descriptive.

## 4. Routing

### 4.1 Configurations

S is a small network, B is Cascade of the setting (C14 or C14'), and R is full-DEM Relay-BP. A shot is routed when its
confidence is strictly below the threshold; a shot exactly at the threshold stays.

| configuration | who decides | thresholds | p |
|---|---|---|---|
| S -> B (`S-B`) | S, except the shots whose S margin < tau1, which B decides | tau1 | all ten |
| S -> B -> R (`S-B-R`) | as S -> B, except that of the shots S routed, those with B confidence < tau2 go to R | tau1, tau2 | 0.2 and 0.3 % |
| S -> R (`S-R`), the known baseline | S, except the shots whose S margin < tau, which R decides | tau | 0.2 and 0.3 % |
| r12 -> R (`S-R` with S = r12) | question 1's accurate mode | tau | 0.2 and 0.3 % |

S is S55 in `uniform`, and S55' or S55'-deep in `depspam_fm` (the rule picks one per p). R is the frozen BB72
configuration (`qnd.teachers.CONFIGS['relay']`, relay-bp 0.2.2) on the full detector error model of the setting's
circuit at p, as the earlier curve study and the training stage ran it. Relay-BP runs only at 0.2 % (the paper's timing
point) and 0.3 % (our registered BB72 point). Neural routing covers the whole sweep (finding 12), and Relay-BP's cost
grows steeply with p (the earlier curve study: 4.6 s per routed shot at 0.55 %).

Confidence scores. For S (ConvLines) the score is the margin, top-1 minus top-2 class score (the registered one). For
B (Cascade) it is either min |logit| or -sum softplus(-|logit|); the second adds up the uncertainty of all twelve
observables (finding 13). The rule picks one on the development block, and both are reported. Thresholds are
quantiles at each p (numpy's default interpolation), so like the earlier curve study's and the registered taus they
assume that p is known.

### 4.2 The development block and the candidates

The development block is the curves block of section 3 (seed kind 10). The networks-alone results make no choice on
it, so choosing routing thresholds there costs them nothing, and the routing claims are made on fresh shots (4.5). On
it:

| candidate grid | values |
|---|---|
| q1: fraction of shots S hands to B; tau1 = the q1-quantile of S's margin | 1, 2, 5, 10, 20, 40 % |
| r: fraction of all shots that reach R in S -> B -> R (r < q1); tau2 = the value below which B's confidence falls on r x N of the S-routed shots | 0.1, 0.25, 0.5, 1 % |
| B's score (S -> B -> R) | min \|logit\|, softplus sum |
| S (`depspam_fm`) | S55', S55'-deep |
| q of S -> R and r12 -> R (a priori, not chosen) | 0.25 % at 0.2 %, 2 % at 0.3 % (the registered BB72-2F and BB72-3F operating points) |

At 0.2 % and 0.3 % Relay-BP decodes the union of every candidate's routed set once (`pc_relay.py --mode dev`), so each
decision serves every candidate that routes that shot (finding 15). Its time per shot is recorded and enters the rule.

### 4.3 The rule (fixed here; applied by `pc_route.py select`)

1. Predicted latency of each candidate, from the speed components of step C3 (section 5, on the one GPU model of 5.1).
   Let t1(n) be network n's mean single-shot latency (host round trip, batch 1) on its frozen single path,
   a(n) its host-to-host time per shot at batch 4,096 on its frozen batched path, and t_R Relay-BP's mean time per
   shot over the candidate's own routed set on the development block. Let f_B and f_R be the fractions of the
   development block that the candidate's thresholds route to B and to R (measured, so shots tied at a threshold count
   as they are routed; gate finding 6). Then
   - single shot: L1 = t1(S) + f_B t1(B) + f_R t_R (for S -> B, f_R = 0; for S -> R, f_B = 0);
   - amortized: A = a(S) + f_B a(B) + f_R t_R / 16 (the batched pipeline runs S and B's queue at 4,096, and Relay-BP
     on the 16 CPU workers of the speed job).
   Cascade's L1 is t1(C14) (t1(C14')), and its A is C14's amortized value (its frozen amortized path: the lowest
   host-to-host time per shot over its screened paths and the batch sizes up to 4,096).
2. A candidate is speed-feasible if L1 <= 0.9 x Cascade's L1 and A <= 0.9 x Cascade's A. The 10 % margin absorbs the
   costs the component model leaves out (finding 11); the verdict uses the end-to-end measurement of 5.3.
3. For S -> B at each p, and for S -> B -> R at 0.2 % and 0.3 %, let F be the fewest development-block failures among
   the speed-feasible candidates (among all candidates if none is feasible). The candidates with at most F + sqrt(F)
   failures (one standard error of the fewest count) are tied, and the rule takes the tied candidate with the lowest
   predicted L1; remaining ties go to fewer failures. If no candidate is feasible, the pick is flagged "predicted
   slower than Cascade". The lock records the rule, F, the band's bound and the tied candidates of every pick. This is
   Amendment 1 (below), made on 11 October 2026 before any confirmation read.
4. S -> R uses the S chosen for S -> B at that p. S -> R and r12 -> R use the a-priori q; they are not chosen.
5. The thresholds, networks, scores and precisions (each network's read: fused for S, BF16 for B; 2.4) are then
   locked. The thresholds of S are quantiles of its fused margins, the computation its timings use. `pc_route.py
   select` writes `locked_<setting>.json` with every candidate's development failures, routed fractions, predicted
   latencies and feasibility; with the locked configurations, their development counts against B, their planned powers
   (4.4) and the non-inferiority margins (4.5); with every network's read precision; and with every network's frozen
   timed paths and the sha256 of the speed components that hold them (2.4). The lock is committed to this branch, and
   its sha256 cited, before the claim-point check and the confirmation read.

#### Amendment 1, 11 October 2026, before any confirmation read

The registered step 3, which the first lock applied (commit 1e8d1c6d, 10 October), took the speed-feasible candidate
with the fewest development-block failures and sent only exact ties to the lower predicted L1. The development curves
are flat above a small routed fraction, so that rule picked large fractions for one or two fewer failures. In `uniform`
at 0.3 %, S -> B (S55 -> C14) had 1,276 failures on the 500,000 development shots at q1 = 5 %, 1,265 at 10 %, 1,264 at
20 % and 1,263 at 40 % (the locked pick); Cascade alone had 1,264. Their predicted L1 were 177, 267, 445 and 802 us,
against Cascade's 1,783 us. The locked configuration was predicted about 2.2 times faster than Cascade, whereas 10 %
would have been about 6.7 times faster with one failure more. The amended rule takes 5 % there (F = 1,263, bound
1,298.5).

The amended step 3 (above) applies to S -> B at every p and to S -> B -> R at 0.2 % and 0.3 %, in both settings. S
still comes out of the S -> B pick, and S -> R keeps the S chosen for S -> B at that p (step 4). The picks come from the
development reads only: `pc_route.py select` ran again on the inputs of the first lock (the reads of C1 and C1b,
Relay-BP's development runs of C2 and the speed components of C3). Only the line that makes the pick changed; in the
first lock's code it was `min(feas or rs, key=lambda r: (r["fail"], r["L1_us"]))`. Nothing else changes: the
candidates, feasibility, the headline family of 132 and its one Holm procedure, the sizing algorithm and its caps, the
non-inferiority margins, the check's tolerance, the timing protocol and the verdicts of section 6.

This amendment is outcome-informed. It was made before any confirmation shot (seed kind 11) was read, but after the
development curves had been seen and, on another branch (`ising-cascade-h100t`), the H100 failure counts of the first
lock's configurations on kind 12, chunks 10-19, the shots of the claim-point check (2.4). The gate review on it (PROCEED WITH
CHANGES) preferred a paired band for S -> B and the accuracy-first rule for S -> B -> R. The authors chose this version
for both, for the largest overall gain, without putting accuracy or speed first. It is frozen before the new claim-point
check, and nothing is tuned afterwards to obtain a pass. The first lock stays in the branch's history as the superseded
one, and its claim-point check (job CJ34), which ran on it, is kept apart as its record. Applied to the first lock's
candidate rows, the amended rule changes 19 of the 24 S -> B and S -> B -> R configurations of the two settings, and in
`depspam_fm` also the S of S -> R at both claim points; `phase_c/RUN.md` lists them.

### 4.4 The size of the confirmation block (finding 8; gate findings 2, 3 and 5)

For each locked configuration D at p, the size follows from the development block's paired counts against B (both
fail, only D, only B, N):
- the meaningful improvement is delta = 10 % of B's block failure rate;
- if the discordant rate d = (only D + only B) / N is below delta, D cannot show that improvement (the two differ on
  too few shots); its comparison is judged all the same (4.5), with no planned power;
- otherwise the alternative puts the discordances at (d - delta) / 2 for D and (d + delta) / 2 for B. The size is the
  smallest n among 10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1,000, 1,500 and 2,000 chunks (between the
  development size and the cap) at which the exact McNemar test and Newcombe's interval, both at alpha = 0.05 / 132,
  detect that improvement with probability at least 0.9 (`bb.route.power`, simulated with its fixed seed). 0.05 / 132
  is one comparison's Bonferroni share of the one alpha of the headline family (section 6: 2 settings x (5 networks
  alone x 10 p + 16 locked configurations)). Holm never judges a comparison at a lower level, so the size holds under
  Holm;
- a comparison that does not reach 0.9 at the cap does not enlarge the block.

The block at p takes the largest size among the comparisons at p that reach 0.9 below the cap, never less than the
development size (so every locked configuration gets a fresh read at least as large as its development read) and never
more than the cap:

| p (%) | 0.10 | 0.15 | 0.20 | 0.25-0.55 |
|---|---|---|---|---|
| cap (shots) | 20,000,000 | 10,000,000 | 5,000,000 | 2,000,000 |

Every locked comparison then carries its planned power: the power under the same alternative and alpha on the block
read at its p (`planned_power` in the lock; none where d < delta). Each claim point is sized for 0.9 on its own. The two
claim points are independent blocks, so a decoder planned at 0.9 at both has a planned probability of about 0.81 of
being "better" at both, which its verdict needs (`joint_power` in the lock is the product of its two planned powers).

Against the earlier alpha of 0.05 / 16 per setting, 0.05 / 132 makes a sized block about 30 % larger ((z(alpha/2) +
z(0.9))^2 grows from 4.24^2 to 4.84^2). The caps stay: at the caps, the worst case, a setting's confirmation holds 49
million shots, about 2.2 hours of GPU time on a full RTX PRO 6000 (section 7).

### 4.5 The one confirmation read

Step C5 reads seed kind 11 through `pc_eval.py` with the networks the lock needs (B, the chosen S, r12; S and r12 on
their fused engine). Step C6 runs Relay-BP on the union of the locked routed sets with the locked thresholds
(`pc_relay.py --mode conf`) and then `pc_route.py confirm` once. That analysis refuses a read whose network sha256 or
read precision differs from the lock. Per locked configuration it reports:
- block failures, P_block with Wilson's interval, and P_L by Eq. (1);
- the fractions routed to B and to R;
- against B alone and against S alone: both discordant directions, the exact McNemar test and Newcombe's 95 %
  interval;
- on the shots S keeps, B's failures that S repairs and S's failures that B would not have made (finding 14);
- on the shots R decides, R's repairs and regressions.

Every locked comparison against B belongs to the headline family (section 6), whether or not its block reached 0.9
power, so a regression at any locked p can veto a win (gate finding 3); each is reported with its planned power.
`pc_report.py` judges the family with one Holm procedure at 0.05 over both settings: "better" or "worse" (rejected,
with Newcombe's interval at the Holm level on that side of zero) or "level". A non-significant result is reported as
"level", not as equivalence.

A secondary statement, in its own family (gate finding 5): D is "not worse than Cascade by more than the margin" when
the upper end of Newcombe's interval for P(D) - P(B), at the one-sided level 0.05 / 32 (Bonferroni over the 32 locked
configurations of both settings), lies below the margin frozen in the lock before the read: 10 % of B's block failure
rate on the development block at that p. It enters no verdict.

### 4.6 Where Relay-BP runs, and its budget (finding 19)

The rule adds Relay-BP's mean time per routed shot, measured in C2, to the GPU times measured in C3, and C7 then times
the pipelines with Relay-BP on the CPU of the timing node. C2 and C6 therefore run on that CPU model, the AMD EPYC 9555
(Zen 5, 128 cores) of every RTX PRO 6000 node, in GPU-partition jobs whose GPU stays idle. acpu's nodes are
mixed and slower. On the same dry unions with the same seeds, Relay-BP took 2.3 times as long per shot on an acpu
EPYC 7713P (dry run 8) as on the EPYC 9555 (dry run 9), and a kept decoder took 1.6-1.8 times as long on an EPYC 7313
(dry run 2) as on the EPYC 9555 (dry run 7). Times measured on acpu would have made every configuration with Relay-BP
look about twice as slow to the rule as it is in C7.

On the EPYC 9555 the mean time per routed shot was 0.08 s (uniform) and 0.09-0.10 s (depspam_fm) at 0.2 %, and
0.20-0.21 s at 0.3 %. A new seeded decoder per shot costs nothing measurable: on the same 150 shots per union in one
process, its decodes took as long as a kept decoder's (within the noise, under 10 % either way; dry run 9), and building
one takes 7-14 ms, which is not timed (2.3). The dry development unions held 1.5-1.7 % of the shots at 0.2 % and 2.9-3.4 % at
0.3 %. On the development block (1,000,000 and 500,000 shots) that comes to about 2.6 core-hours for both settings. The
locked configurations route about 0.5 % (0.2 %) and 3 % (0.3 %) of the shots to Relay-BP, about 8 core-hours for both
settings at the caps. On 32 cores C2 takes about 10 minutes and C6 about 25.

## 5. Speed

### 5.1 One GPU model

Every GPU step of phase C (the curves C1 and C1b, the screen and the timings C3, the claim-point check C4b, the
confirmation reads C5 and the pipelines C7) runs on one GPU model, in one venv: a full RTX PRO 6000 Blackwell Server
Edition (96 GB; partition `artxpro6000`, GRES `gpu:rtx_pro_6000`, never a MIG slice), the type Alpine schedules most
readily, with bb's `venv_cu128` (torch 2.11.0+cu128), in jobs that hold the whole card. bf16 tiles and compiled kernels
depend on the GPU, so a read, a screen, a check and a timing describe one computation only on one model; the fused
engine's reads are bitwise its timed paths on this model, which C3's screen and C4b test. `pc_eval.py` and
`pc_speed.py` refuse any other card (`pc_common.GPU`; a MIG slice adds its profile to the name), every output records
the card, its driver (from nvidia-smi) and torch, and the lock names those of C3. `pc_route.py` refuses a development
or confirmation read made on another card or torch, and the report a check or a pipeline timed on another card. The
paper's H200 timings stay in their own column (5.4). Every timing row also records the SM clock samples (every 100 ms,
`bb.timing.gpu_watch`) and whether another process was on the card. The nodes' CPU, an AMD EPYC 9555, runs Relay-BP in
the pipelines (4.6).

### 5.2 The networks (step C3, `pc_speed.py nets`)

Each network runs in three fresh processes, interleaved across networks. Per path and batch size:

| metric | definition |
|---|---|
| GPU term (batch 1) | SETTINGS 5.1's replay protocol on the captured graph with the input already on the GPU: 200 warm-up replays, then 5 x 100 timed replays with CUDA events; the mean per call |
| single-shot latency (batch 1) | 10,000 host round trips (pinned host -> device copy of one shot, graph replay, device -> pinned host copy of the decision and confidence, synchronise); host wall time per call: mean, median, 99th and 99.9th percentiles, maximum |
| amortized latency (batch B = 1, 16, 64, 256, 1,024, 4,096) | wall time over consecutive batches / (batches x B), per shot and per round (/ 6), as the paper defines it ("total wall-clock time divided by (batch size x syndrome rounds)"); two variants: input resident on the GPU, and host to host (each batch copied in from pinned memory and its outputs copied back) |
| screen | 2.4's tolerance against the network's accuracy path (its read's computation; the fused engine for ConvLines), in every process and per batch size: failures at 0.55 %, routing at 0.2 % and 0.3 %, and for a ConvLines path every decision and confidence equal (bitwise); the decisions and confidences that differ, recorded |

A process's value is its mean, and a network's value is the slowest of all its processes (three are run, REPS = 3; a
rerun adds processes and can only make the value slower). Every network and every pipeline (5.3) gets this statistic
(finding 10 of the stage gate, gate finding 4), and a value from fewer than three processes enters no headline. Single
shot counts the frozen single path, amortized the frozen amortized path (2.4). All processes, paths, precisions and
latency quantiles are reported, with the per-round values (per shot / 6) of every batch size, batch 1 included (gate
finding 7).

### 5.3 The routed pipelines end to end (step C7, `pc_speed.py pipeline`, finding 11)

Each locked configuration at 0.2 % and 0.3 % runs on shots of the speed stream (kind 12 from chunk 0; the screen read
chunks 0-9 at these p, which is harmless since the pipelines select nothing) and makes its own routing decisions, in
three fresh processes per mode, interleaved (rep-major) across configurations; its value is the slowest of all its
processes, as for the networks (5.2; gate finding 4). Each process refuses speed components other than those frozen
in the lock, and compares the confidences with the locked thresholds in float64, as the routing of the reads does:
- single: one shot at a time; S at batch 1; when S routes, B at batch 1 on the detector rows already on the GPU; when
  B routes (or, in S -> R, S), Relay-BP in the same process, single-threaded. Each network runs on its frozen single
  path. The per-shot host latency runs from the shot in pinned memory to the decision on the host,
  over 20,000 shots after 200 untimed ones: mean, median, 99th and 99.9th percentiles, maximum, and the same for the
  shots that reached B and R (the conditional latencies);
- batched: S over batches of 4,096; the routed rows queue on the GPU, and B runs whenever 4,096 are waiting (and once
  at the end, padded); both run on their frozen batched paths (batch 4,096); Relay-BP runs asynchronously in 16 worker
  processes, all of which have built their decoders before the clock starts, in tasks of 4 routed shots. One task per S
  batch kept most workers idle: S -> R at 0.3 % took 1,130-1,420 us per shot in dry run 5 and 280-450 us with tasks of
  4 in dry run 6. Over 500,000 shots, the wall time from the first copy to the last decision divided by N x 6 is the
  amortized latency. Throughput, the B stage's time and Relay-BP's mean time are recorded. The wall time includes the
  last padded B batch and the wait for the last Relay-BP task. At the dry runs' 20,000 shots these ends dominated (S ->
  B at 0.2 %: 43 us per shot measured, 12 predicted); at 500,000 they are 25 times smaller, and they can only make our
  decoders look slower.

Cascade of the setting (C14, C14') runs alone in the same harness, in the same job and the same rep-major loop, at each
claim point in three fresh processes per mode (third revision): single on its frozen single path, batched on its frozen
amortized path (the batch size of its amortized latency). Each routed decoder is compared with this Cascade (6,
condition 2), so that a comparison uses timings from one job: the report requires the pipeline's processes and
Cascade's to come from the same jobs, at least three of each. C3's Cascade values serve the networks alone, which C3
times in one job with Cascade.

The measured routed fractions are reported beside the locked q1, r and q. The Relay fraction is the measured joint
fraction. Relay-BP here is deployed with one retained decoder per process; the report prints, beside every pipeline
with Relay-BP, what the seeded reconstruction's build would add per incoming shot (2.3; gate finding 8).

### 5.4 The paper's timings

The paper's timings go in their own column, matched to what they are. For BB72 the paper reports Cascade's amortized
latency at p = 0.2 % on an H200 with FP8 (Fig. 1(c) and 2(d)); before C8 it is digitised from the paper's SVG by the
method used for Fig. 2(a), with its digitising uncertainty, into `$R/out/paper_timing.json` (source, then per value the
network it stands beside, microseconds per round and a label; committed with the outputs), and `pc_report.py` prints it
in the speed table's last column, "paper (H200, FP8)" (gate finding 7). The text's "single-shot latencies of ~40 us per
cycle on a single NVIDIA H200" refers to the surface codes of Fig. 3(b) and is quoted as such. No paper timing is
combined with a local accuracy or enters a verdict (finding 10).

### 5.5 Not done

FP8 (2.4). CPU timing on the mac-mini: the end-to-end protocol needs the GPU and Relay-BP's CPU on one host, and the
mac-mini has no GPU, so phase C does not use it.

## 6. The final report and the verdicts

`pc_report.py` produces every table. The report, `cascade_compare/PHASE_C_RESULT.md`, states:

| question | what is stated |
|---|---|
| 1 | r11 and r12 alone in both settings (curves, compared with Cascade under Holm); r12 -> R at 0.2 % and 0.3 % on the confirmation block (against Cascade and against r12 alone); their single-shot and amortized latencies, r12 -> R end to end; the paper's values as context |
| 2 | S55' and S55'-deep alone in `depspam_fm` (and S55 in `uniform`): curves against Cascade under Holm, latencies |
| 3 | C14' and C6' against the paper's Cascade curve (ratios with intervals and the digitising caveat), and C14 / C6 in `uniform`; their BF16 latencies on our GPU beside the paper's FP8 timings |
| 4 | the locked S -> B and S -> B -> R in each setting: confirmation accuracy against Cascade and against S alone, the repairs and regressions on S-kept shots, routed fractions, end-to-end latencies |

Whether we "beat Cascade in accuracy and speed" is decided per setting, mechanically (`pc_report.py`). A decoder D of
ours is one of S55, S55', S55'-deep, r11 and r12 alone, or one of the routed decoders S -> B, S -> B -> R, S -> R and
r12 -> R (each with its locked configuration at every p). D beats Cascade when, at p = 0.2 % and at p = 0.3 %:

1. accuracy: D is "better" than Cascade alone (C14 / C14') on the same shots, on the curves block for a network alone
   and on the confirmation block for a routed decoder, in the headline family below;
2. speed: D's mean single-shot latency and its amortized latency are both lower than Cascade's, on the same GPU model
   under the same protocol (networks alone: against Cascade in C3's job, 5.2; routed decoders: against Cascade alone
   timed beside them in C7's job, 5.3, single mode and batched mode; each value the slowest of at least three
   processes);
3. D is not "worse" than Cascade at any other p where it was read;
4. eligibility (gate finding 1): every frozen path behind D's values and Cascade's is screened (for a routed decoder
   also its S's: `pc_report` reads S's `no_path_agrees`), and every claim-point check of D and of Cascade passed at
   both claim points (2.4): for a network alone its single and amortized paths, for a routed decoder its
   configurations in single and batched mode. A decoder that fails is removed from the
   headline and its values are described with the reason; if Cascade's own paths fail, no decoder beats Cascade in
   that setting. No other path replaces a failed one.

The headline family (gate findings 2 and 3). Every comparison with Cascade that can make or veto a headline is judged
in one family by one Holm procedure at alpha = 0.05: our five networks alone at the ten p of the curves and every locked
configuration at every p where it was locked, in both settings, 132 comparisons (2 x (50 + 16)). The study's goal is an
overall win ("beat Cascade in both speed and accuracy"), so the one alpha covers both settings and both kinds of
decoder: the probability that any headline in either setting rests on a false "better" is at most 0.05, however the
report combines the settings. Every comparison stays in the family whatever its planned power (4.4), so a regression at
an untested-looking p still vetoes. The confirmation is sized at 0.05 / 132 (4.4).

The report prints beside every verdict the 99.9th percentile of the single-shot latency. Where D's is higher than
Cascade's, it says "faster on average, slower in the tail". The dry runs already show that a routed decoder has
Cascade's latency or Relay-BP's in its tail. If no decoder satisfies 1-4 in a setting, the report says that we do not
beat Cascade there and lists the partial results (faster but not more accurate, or the reverse, or removed by a check).
Each verdict names the setting (`depspam_fm` is a fit, not the paper's identified setting), the GPU model and the
precision of every path behind it (BF16 for Cascade; FP16 matmul inputs with FP32 accumulation for the fused engine of
our networks, their accuracy and their timings alike; gate finding 7). The speed tables give every value per shot and
per round (per shot / 6), at batch 1 too. The paper's FP8 timings (5.4) and digitised curves are context, never the
opponent.

## 7. Compute and placement

Measured rates (full RTX PRO 6000, each network at its read batch; dry runs 1-1b, 12 and 15a): C14 and C14' about 6,450
shots/s (BF16), C6 33,000 under torch.compile (14,300 eager), S55 / S55' / r11 / r12 about 250,000 on the fused engine
(100,000 in BF16 before the second revision), S55'-deep 167,000 (67,000). Stim sampling does not limit (8 processes give
about 3.5 million shots/s). Single shot on the frozen paths of dry run 16b (C3's protocol: every path at every batch
size, three processes, the slowest counted): C14 and C14' 1.80 ms (compiled; C14''s graph path 2.23 ms), C6 0.80 ms
(compiled), S55, S55', r11 and r12 0.089 ms (fused), S55'-deep 0.138 ms (fused); amortized: C14 and C14' 71.6 us per
shot at 4,096 (compiled; graph 155), C6 31.3 us (compiled), S55, S55', r11 and r12 4.14 us at 1,024 (fused; 4.16 at
4,096), S55'-deep 6.27 us at 4,096. Before the second revision our networks' frozen paths were BF16 graph paths (dry run
14): 0.40 ms per single shot and 9.2 us per shot at 1,024 (S55'-deep 0.61 ms and 13.7 us).

| step | what | where | resources | hours |
|---|---|---|---|---|
| C1 | development block, both settings (every network but C6'; C6 compiled, our networks fused) | Alpine GPU | 2 jobs, one full RTX PRO 6000 each | about 1.85 each, in parallel |
| C1b | C6' add-on, both settings (compiled; eager if its weights arrive after C4b) | Alpine GPU | 1 job, one full RTX PRO 6000 | about 0.4 (0.7 eager) |
| C2 | Relay-BP on the development unions | an Alpine RTX PRO 6000 node's CPUs (EPYC 9555; the GPU idle) | 32 cores | about 0.2 |
| C3 | speed of the networks, three processes each, with the screen at 0.55 % and at the claim points | Alpine GPU | 1 full RTX PRO 6000, 16 cores | about 2.2 (1.9 without C6', dry run 16b) |
| C4 | the rule and the locks | Alpine acpu | 8 cores | about 0.3 |
| C4b | the claim-point check, both settings | Alpine GPU | 1 full RTX PRO 6000, 32 cores | about 0.5 |
| C5 | confirmation reads | Alpine GPU | 2 jobs, one full RTX PRO 6000 each | 0.75-2.2 each, depending on the locked sizes (sized blocks about 30 % larger than before 4.4's change; floor and caps unchanged) |
| C6 | Relay-BP on the confirmation unions, and the analysis | as C2 | 32 cores | about 0.5 |
| C7 | the pipelines end to end, three processes per configuration and mode | Alpine GPU | 1 full RTX PRO 6000, 24 cores | about 2.3, Cascade alone about 0.3 of it (dry run 16c's time per process at the production sizes) |
| C8 | report | Alpine acpu | 8 cores | about 0.2 |

In total about 11-14.5 GPU-hours, all on full RTX PRO 6000 cards (C2 and C6, about 0.7 of them, use only the node's
CPUs): 10-13 before the first revision, whose three pipeline processes (C7) added about 2, the routing screen (C3) about
0.8 and C4b about 0.6 while the compiled C6 and C6' reads saved about 0.7 and the confirmation blocks grew between their
floor and their caps; 12.5-15.5 before the second revision, whose fused reads of our networks save about 0.3 in C1, 0.1
in C4b and 0.1-0.2 in C5; and about 12-15 before the third revision, whose C7 measured in dry run 16c comes out about
0.7 below the first revision's estimate though it now times Cascade alone (about 0.3), and whose C3 measured in dry run
16b about 0.1 below. About 11 core-hours of Relay-BP on the EPYC 9555 and 4 core-hours on acpu; about 6.5 hours from
start to report if the queues cooperate (the longest chain: C3, C4, C4b, C5, C6, C8). Limits respected: at most four
Alpine GPUs at a time (C1's two, C3 and C1b together at the start; later C4b and C7, then C5's two with C7 and C1b);
QEC's acpu cap of 80 cores (only C4 and C8, 8 cores each, never together); the mac-mini is not used (5.5); no RunPod and
nothing on Derecho. The outputs stay on Alpine under `<cluster-scratch>/cascade_w4/out/` (the per-shot files, about 4 GB
per setting). The JSON results, the locks, the checks, the logs and the per-shot files' sha256 are committed under
`cascade_compare/phase_c/`.

## 8. Order of the steps

C1 and C3 start together. C2 follows C1; C4 follows C2 and C3. The locks are committed. Then C4b (the claim-point check)
and C7 (the pipelines) run, on the RTX PRO 6000 like every GPU step; C4b's `check_<setting>.json` is committed before
any confirmation read. Then C5 and C6, and C8 once C6 and C7 have ended. C1b runs whenever C6''s weights reach Alpine
and is needed by C8 only. A runbook, not part of this release, gave the commands.

## 9. Dry runs

All scripts ran end to end on dry-run slices (chunks of 10,000 shots, two chunks per point), never on kind 10 or 11; dry
runs 1-10 timed on kind-12 inputs, dry runs 12-16 on kind 13 only. The code phase C runs is the tree of dry runs 16a-16c
(third revision, c21bd97d); later commits change only documents and records. Every job file ran after its last change,
and every part of the scripts ran after the last change to that part: dry run 16a the card guard of the reads and the
timings (on a MIG slice, which both refused), 16b the networks' speed and the screen as C3 runs them (every path at
every batch size, three processes), 16c everything after them (the rule and the lock, the check with its own reads
through `pc_eval.py`, `confirm.sh`, the confirmation analysis, the pipelines with Cascade alone beside them, the
report). The development reads and Relay-BP's development unions are dry run 15a's: `curves.sh`, `relay_dev.sh` and
`pc_relay.py` have not changed since, and the one change to `pc_eval.py` (the card guard and the driver in the
signature) ran in 16a and 16c. `late.sh` last ran in dry run 13; it reads only Cascade networks through `curves.sh`.
The full record, with the problems the dry runs found and how each was fixed, is `phase_c/dry/DRY_RUNS.md`;
a revision note, not part of this release, mapped each gate finding, the second revision's decision and the independent
check's points to their changes and dry runs.

| dry run | job (tree) | what ran |
|---|---|---|
| 1 | GPU (dcddfd18) | every network on 20,000 shots at 0.2, 0.3 and 0.55 % in both settings (development slice) and on 20,000 other shots at 0.2 and 0.3 % (stand-in confirmation); `pc_speed.py nets`, graph path, every network |
| 1b | GPU (d1c6ba8b) | every path (graph, compiled, fused) for C6 and S55' |
| 2 | acpu (1366c5cb) | Relay-BP on the development unions; the rule and the locks at 0.2 and 0.3 %; Relay-BP on the stand-in confirmation slices and `pc_route.py confirm`; the earlier curve study's figure redrawn |
| 3 | GPU (07d90ea1) | the rule again (after a units fix), the confirmation analysis again, every dry-locked configuration end to end in both modes, `pc_report.py` |
| 4 | GPU (adcc5398) | the chain after the reads again; `confirm.sh` on one more dry chunk; `late.sh` with C6 standing in for C6'; the report with the add-on paired by detector digest |
| 5 | GPU (c8849830) | the pipelines again after a fix (the batched clock starts once every Relay-BP worker is ready) and the report |
| 6 | GPU (8663861f) | the pipelines again with Relay-BP tasks of 4 routed shots, and the report |
| 7 | GPU (ce5ac90b) | `relay_dev.sh` and `speed_nets.sh` (with its final summarize) whole; Relay-BP's failures differed from dry run 2's by 1-4 per union |
| 8 | acpu (ea2d5ff3) | Relay-BP with a seeded decoder per shot, twice (32 and 13 processes, tasks of 10 and 7 shots): the same decision on every shot |
| 9 | GPU node, CPUs only (e01e518d) | `relay_dev.sh` on the EPYC 9555: the same decisions as dry run 8, the time per shot of a kept decoder, acpu 1.6-2.3 times slower |
| 10 | GPU (675639c7) | `confirm_relay.sh`, `speed_pipe.sh` (few shots) and `report.sh` with the changed Relay-BP calls |
| 11 | GPU node, CPUs only (c09edb4c) | Relay-BP's failures on the 8 dry unions under 4 seed families, new decoders per shot and decoders kept over 25 shots: no difference between the two ways beyond the spread between families |
| 12 | GPU (0b8083da) | First revision, part A: the reads again (C6 compiled), `relay_dev.sh`, `speed_nets.sh` in three processes with the new screen (failures), r11 into `speed_late` |
| 13 | GPU (cf62e196) | First revision, part B: the rule and the locks (sizing at 0.05 / 132, planned powers, frozen margins), `check.sh` at full size (100,000 shots per claim point and setting), `confirm.sh`, `confirm_relay.sh`, `late.sh`, the pipelines in three processes (uniform) and one (depspam_fm), the report; the check failed two depspam_fm configurations on routing (DRY_RUNS.md problem 18) |
| 14 | GPU (ded82040) | the screen with routing: `speed_nets.sh`, the rule, `check.sh`, `confirm_relay.sh`, the pipelines and the report again on dry runs 12-13's reads; the compiled and fused paths of S55 and S55' failed the routing screen, so their frozen paths are graph paths, and every check passed in both settings |
| 15x | two GPU MIG slices, an RTX PRO 6000 and an H200 NVL (ded82040 with 4fe03585's `pc_fused.py`) | Second revision: `pc_fused.py`'s batched engine against the registered batch-1 fused engine at batch 1, 7, 64 and 4,096: every decision and margin equal; the batch-1 engine's outputs byte-identical on the two models |
| 15a | GPU (4fe03585) | Second revision: the reads of 12-13 again with our networks on the fused engine (development, stand-in confirmation and check slices); Relay-BP's development unions |
| 15b | GPU (4fe03585) | `speed_nets.sh`, every network screened against its read in 3 processes; every fused path bitwise its read; our networks' frozen paths are fused paths |
| 15c | GPU (4fe03585) | the rule and the locks on 15a-15b, `check.sh` at full size (every read of our networks bitwise its frozen paths; every row passed), `confirm.sh`, `confirm_relay.sh`, the pipelines (3 processes in `uniform`, 1 in `depspam_fm`) and the report |
| 16a | GPU MIG slice: RTX PRO 6000 1g.24gb (c21bd97d) | Third revision: the card guard. A read (`curves.sh`) and a timing (`pc_speed.py nets`) on the slice both stopped with the card's name before drawing a shot |
| 16b | GPU (c21bd97d) | Third revision: `speed_nets.sh` as C3 runs it: every network present (C6' had not reached Alpine), every path at every batch size of BATCHES, 3 processes, 10,000 round trips and 2 s per amortized value; 1 h 54 min; every fused path of our networks bitwise its read at every batch size, none of their BF16 paths within the tolerance; frozen paths: Cascade compiled, ours fused |
| 16c | GPU (c21bd97d) | Third revision: the rule and the locks (naming the card, driver and torch), `check.sh` at full size, `confirm.sh`, `confirm_relay.sh`, the pipelines with Cascade alone beside them (3 processes in both settings), the report, and the branch check (each new rule on a perturbed copy of the run's outputs); every check passed in both settings (694 s and 531 s); Cascade alone took 1,750 us per shot single and 72 us batched beside the pipelines; every decoder eligible, none beating Cascade at these sizes; each new refusal fired on its perturbed copy |
