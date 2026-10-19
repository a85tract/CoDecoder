# ARB at d = 13: arbitration on the routed shots of ConvLines + Cascade on the surface code (exploratory)

19 October 2026. Exploratory: nothing here is registered and nothing is a claim. Fixed before any shot of
the reads below was drawn.

## Why

ARB on BB72 (`~/work/arb`, branch `arb-exp` of the research repository) looked at the routed shots where CoDecoder's and Cascade's decisions
differ. A joint decision there (the network's class score plus w times Cascade's log-likelihood) cut the composition's
failures by 17.9 % and 14.2 % on held-out shots. This plan measures the same thing on the paper's other Cascade
composition, the surface code at d = 13: ConvLines `d13_b5_r775` with Cascade CS13 behind it, the network's least
confident 10 % (p = 0.6 %) and 1 % (p = 0.35 %) routed to Cascade (the comparison's C5, "the pair"). Its results decide
the joint decision's frozen weight for d = 13 in a registered confirmation that follows.

There is one logical bit, so on a routed shot where the two decisions differ exactly one decoder is right. The routed
oracle therefore repairs every routed shot on which only Cascade fails and breaks nothing.

The comparison's re-test (RETEST_RESULTS.md; the paper's Table I, single mode) counted, per 1,000,000 shots at 0.6 % and
30,000,000 at 0.35 %: Cascade alone 4,936 and 2,354, the composition 4,917 and 2,325; only the composition failed on
21 and 0 shots, only Cascade alone on 40 and 29.

## Setting (the d = 13 comparison's, unchanged)

- Release `94aaaa7f` (this repository, cloned to `~/work/arb2/release`, clean). Code: `cascade/d13/surface/`
  (`sv_common.py` for the decoders, the GPU guard, the routing rule and the statistics; `surface_data.py` for the
  shots), `cascade/d13/SURFACE_PLAN.md`, `retest/shim/rt_surface.py` (the model for the seed shim).
- Circuit: `circuits/d13_legacy.stim` through `surface_layout.circuit_at(13, p)`, p = 0.006 and 0.0035.
- The network S: `d13_b5_r775` (sha256 cbd2b84a...) on the release's deployed Triton engine (`timing.engine` FROZEN,
  fp16 matmul inputs, fp32 accumulation) through `sv_common.s_fn`, in batches of 4,096 consecutive shots carried across
  chunk boundaries, the last one zero-padded: the comparison's SB, built as `sv_read` builds it (`sv_common.Queue` over
  `sv_common.Graph`). Logit c (fp32), decision c > 0, confidence |c|.
- Cascade B: `weights/cascade_d13/CS13/final_ema.pt` (the source checkpoint's sha256 5351b27b..., checked by
  `sv_common.load_B`) in eval mode under bf16 autocast, uncompiled: `sv_common.b_fn(model, "graph")` is the module
  itself. Batches of 16 consecutive shots (the comparison's b*), aligned with the chunks (10,000 = 625 x 16). Logit z
  (fp32 of the bf16 output), decision z > 0, confidence |z|.
  - The d = 13 comparison has no separate eager accuracy read: its Cascade computations are torch.compile plus a CUDA
    graph at batch 1 and at batch 16, read and timed as one. "Its eager accuracy path" is taken here to be the
    uncompiled module at the comparison's batch of 16. Direct eager calls at batch 16 may be limited by Python's launch
    overhead, so the read may replay the module in one CUDA graph at batch 16 (`sv_common.Graph`, the comparison's
    screened path graph@16), which launches the same kernels.
  - Rule, fixed here: the dry run runs every dry shot both ways. The full reads use the CUDA graph if all dry logits
    are equal bit for bit and the direct eager calls are more than 10 % slower; otherwise they use direct eager calls.
    With the graph, every part of 1,000,000 shots runs its first 1,024 shots again by direct eager calls, and the read
    ends with an error on any differing bit.
- Routing: the lock's C5 threshold tau1 = 5.242445659637451 (0.6 %) and 8.111059150695802 (0.35 %); a shot goes to
  Cascade iff |c| (float32, compared in float64) lies strictly below tau1 (`sv_common.routed`). The release's copy of
  the lock (`cascade/d13/out/locked_d13.json`, sanitized: sha256 455eb2a6..., while RETEST.md names the committed
  original 57670204...) holds these values; the analysis refuses any other.
- GPU: one full RTX PRO 6000 Blackwell Server Edition (`sv_common.setup_gpu` refuses another card; TF32 off, cuDNN
  autotuning off, cuDNN deterministic), with the comparison's venv pins (`cascade/d13/surface/pod/requirements_cu128.txt`:
  torch 2.11.0+cu128, triton 3.6.0, stim 1.16.0, numpy 2.2.6, scipy 1.18.1) on Python 3.12.14.
- Not part of this experiment: correlated PyMatching, the chain C1, the comparison's compiled paths, timing.

## Steps

1. Fresh shots from new seeds: 4,000,000 at p = 0.6 % (400 chunks of 10,000) and 30,000,000 at p = 0.35 % (3,000
   chunks).
   - TAGs: `convlines arb surface d13 p0.006\n` (set arb-d13-0.006) and `convlines arb surface d13 p0.0035\n`
     (set arb-d13-0.0035), new against `retest/tags.json`, the TAGs RETEST.md section 2 lists and ARB's two BB72 TAGs.
   - Formula and collision rule: RETEST.md section 2's, unchanged. R is the re-test's beacon randomness (drand round
     6,547,616), reused as ARB did; the new TAGs alone make the seeds new. Historical seeds: `retest/historical_seeds.txt`
     (7,034, sha256 f2c7eb44...) and `seeds/hist_extra.txt` (17,534, from `code/hist_scan.sh`: every 19-digit integer in
     [2^62, 2^63) in any seed, beacon or manifest file on every branch of the research repository and in every text file of the
     release; it holds the 7,034, the re-test's 6,115 seeds and ARB's 2,800). No index was dropped.
     `seeds/arb2_seeds.json` (sha256 b869cd7d...; `code/arb2_seeds.py check` recomputes it). First seeds:
     7125939203913419922 (arb-d13-0.006) and 5068027980986011364 (arb-d13-0.0035).
   - Mapping: `code/arb2_shim.py` patches `surface.surface_data.eval_seed` at import, in the main process and in every
     spawned sampling worker (a spawn worker re-runs the shim as `__mp_main__`): kind 11 (the comparison's confirmation
     kind) at d = 13 and p in {0.006, 0.0035}, chunk j gets ARB's seed j. Any other kind-11 request raises, every other
     kind is unchanged, and without the seeds file the shim refuses to start.
   - Checks (`code/arb2_seedcheck.py`, through the shim): every chunk's recorded seed is ARB's and the patched
     function's; the trace shows for each chunk exactly one call by the main process (the seed it records) and one draw
     by a sampling worker; at sampled chunks the saved rows equal a fresh draw from the recorded seed, and their digest
     (`sv_common.digest`) differs from the digests of the same chunk numbers of the comparison's confirmation block
     (kind 11, its arithmetic seed), of the re-test's block (`retest/run/seeds.json`) and of the comparison's
     development block (kind 10); for chunks 0-2 the comparison's own seed reproduces the digest that the re-test's
     replay drew on the cluster (`retest/replay/d13/replay_seeds_off.json`), so the pod's Stim draws the comparison's
     rows from the comparison's seeds; every digest of a read is distinct. If the pod's Stim makes it cheap, every chunk
     digest of the reads is also compared with every chunk of those three blocks at the same p.
   - Dry run: chunks 0-1 at each p through the whole chain. Its digests are checked as above (the small dry run that
     shows the new seeds give new rows), and its rows are later compared bit for bit with the same chunks of the full
     reads. It is never reported as a result.
2. Reads, one per p (`code/arb2_read.py` through the shim). Saved per shot: the observable, the packed detectors
   (`np.packbits`, big bit order, as `sv_common.digest` packs them), the chunk, c (float32), the engine's decision,
   z (float32) and Cascade's decision; per chunk its seed and digest. Parts of 100 chunks, each with its sha256 in the
   read's journal.
3. Halves by chunk parity: fit = even chunks, evaluation = odd chunks. Every parameter is fitted, and the rule to carry
   forward is chosen, on the fit half only. Everything is reported on the evaluation half.
4. Rules, applied only to routed shots where the two decisions differ (everything else as in the composition). The
   objective on the fit half is the composition's failures, i.e. repairs minus breaks; never switching (gain 0) is
   always available.
   - J (joint decision): answer 1 iff c + w z > 0; on a tie, Cascade's answer. On these shots J takes the network's
     answer iff |c| > w |z|, so the fit-half gain is a step function of w with steps at the fit half's ratios |c| / |z|.
     The sweep is exact over all steps; the fitted w is the geometric midpoint of the best step (r1 / 2 on the first
     step [0, r1), 2 rm on the last finite one); ties keep the larger w. The answers are then computed from c + w z
     itself in float64.
   - P (preference): take the network's answer iff |z| < a + b |c|; b in {0} and 10^(k/40), k = -120..120, the best a
     for each b exactly by cumulative counts; ties keep the smaller b, then the smaller a (ARB's A2).
   - T (two thresholds): take the network's answer iff |z| < t_C and |c| > t_S, exact over both thresholds; ties keep
     the fewest switches (ARB's A1).
   - Carried forward: the rule with the largest mean over the two p of its fit-half relative failure reduction (gain
     divided by the composition's fit-half failures); ties go to fewer parameters (J has one), then to the order J, P,
     T; none if the best mean is not positive. P contains J (a = 0, b = 1/w) up to its grid of b, so P's fit-half gain
     will be at least close to J's. J's fitted w is reported either way, since the registered confirmation fixes J's
     weight.
5. Report per p on the evaluation half: failures of Cascade alone, the network alone, the composition and each rule's
   composition; repairs and breaks; the net (repairs - breaks) per 10^4 shots with Newcombe's 95 % interval (method 10,
   `sv_common.newcombe10`, the release's equivalent of `bb.route.paired`); the relative reduction (the net over the
   composition's failures; its interval divides the net's by that count, so it is approximate); the routed oracle and
   the share of its repairs each rule recovers; the exact two-sided McNemar p against the composition
   (`sv_common.mcnemar_p`). Diagnostics: J's gain over the grid of w on both halves, the evaluation half's own best w,
   and one w for both p (the value with the largest mean fit-half relative reduction). As a check of the reads, the
   per-category counts of the whole reads beside the re-test's (Table I) and the comparison's original counts, with
   the z of each rate difference (independent samples; decisions near ties may differ because Cascade's path differs).
6. Exploratory: no registration, no claim. The registered confirmation is the authors' decision.

## Departures from ARB on BB72, all forced by the single logical bit or by the d = 13 code

- A1m and A1s become T: with one logit, min |z| is |z| and the softplus score is monotone in |z|. A2 becomes P with
  dC = |z| and dN = |c|. A3 becomes J: with two classes the network's class-score difference is its logit c, and no
  third answer exists.
- No class-score export: c is the network's whole output, saved by the read itself.
- J's w is fitted exactly over the steps of its gain, instead of on a grid (the grid curve is still reported).
- The routing is the comparison's C5 threshold on |c|, and Cascade runs as described above. The comparison's own
  per-chunk digests are on the cluster, so the seed check recomputes the old chunks with Stim instead of reading their
  records.

## Compute and cost

- One RunPod RTX PRO 6000 (Blackwell Server Edition), requested by the authors; own pod only. Watchdog: stop at
  15 USD, 5 minutes after DONE, or when the heartbeat is 25 minutes old. Every output is copied back to `~/work/arb2/out`
  with sizes and sha256 checked before DONE.
- Time, from the comparison's screen on this card (`cascade/d13/out/speed.json`): Cascade's module at batch 16 in a CUDA
  graph takes 204.7 us per shot and the engine at 4,096 takes 16.2 us, so 34,000,000 shots take about 2.1 h of GPU
  time. With setup, the dry run, the seed checks, the analysis and the copy-back (about 10 GB raw, partly overlapping
  the second read), about 2.5-3 h, roughly 6-8 USD at 2.5 USD/h. If direct eager calls are needed, add up to 1-2 h.
- The analysis runs on the pod. The shared Linux workstation ran only the seed generation, the history scan and tiny tests of
  the analysis on synthetic data.
