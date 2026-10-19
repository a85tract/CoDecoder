# ARB5: a joint decision on the routed shots of the surface-code compositions, scored by matching's complementary gap (exploratory)

This read is exploratory: nothing here was registered and nothing here is a claim, and the result feeds a proposal in
the paper's future work. Every count below comes from the evaluation half (odd chunks) of the paper's registered
surface-code test sets unless a table says otherwise, and every fitted weight comes from the fit half (even chunks).
No new shots were drawn.

## Question

ARB (~/work/arb, branch `arb-exp` of <research-repo>) found on BB72 that on the routed shots where CoDecoder and
its fallback (Cascade) disagree, a joint decision (the network's logit plus w times the fallback's log-likelihood)
removes 18 % and 14 % of the composition's failures. On the surface code the fallback is correlated PyMatching
(correlated PM), which returns no such score. ARB5 measures what the same joint decision gains when the score is a
complementary gap: the weight of the best correction in the other logical class minus the weight of the best
correction, which approximates the log-likelihood ratio of the two classes.

## Summary

- At the registered routed fractions (the paper's rows) the gain is small. With the correlated gap the joint decision
  removes 0.8 % of the composition's failures at d = 7 (132 of 15,922), 2.3 % at d = 9 (one-block network, 138 of
  5,989), 4.0 % at d = 15 (62 of 1,535) and 0.3 % at d = 17 (3 of 1,058, not significant). At d = 11 and 13 the
  registered q is 0, so no shot is routed. ARB's BB72 figure was 14-18 %.
- Plain PM's gap, the existing gap-switching code, gains nothing at any registered q: the fit half picks w = infinity,
  which is the composition itself. Its sign is plain PM's answer, so on every disagreement row where plain PM sides with
  the network the rule must take the network's answer, and on those rows correlated PM is right more often.
- With the joint decision, routing more shots pays off: at q = 5 % the correlated-gap joint decision fails less often
  than the paper's row at every distance (-4.9 % at d = 7, -9.6 % at d = 11, -18.2 % at d = 13, -6.4 % at d = 15 and
  -5.9 % at d = 17), each with its paired 95 % interval below 0. At q = 2 % the numbers are close to these (-9.5 %,
  -18.4 %, -6.1 %, -4.7 % at d = 11-17). Routing 5 % without the joint decision is worse than the paper's row at
  d = 11-17 and within one failure of it at d = 7.
- At d = 7 and q = 2 % the routed oracle would remove 2,723 failures; the joint decision removes a net 132, 5 % as
  many. That share rises with q, to 15-43 % at q = 5 %.
- The fitted w of the correlated gap lies between 0.45 and 2.2, and w = 1 (the plain sum of the two log-odds) keeps
  87-102 % of the fitted w's net gain on the evaluation half, so w barely needs a fit.
- The correlated gap costs one more correlated decode, forced into the other class, on each disagreement row. On one
  core of an AMD EPYC 7713 (the CPU on which the paper times correlated PM) it takes 76 µs per routed shot at d = 7
  and 2,476 µs at d = 17, 1.4-2.4 times correlated PM's own time on the same shots. Run only on the disagreement rows
  (24-39 % of the routed shots), it adds 29-530 µs per routed shot, or about 0.6-5.3 µs per shot averaged over all
  shots at the registered q. Plain PM's gap costs 1.1-1.5 times correlated PM.

## Definitions

| term | plain meaning |
|---|---|
| shot | one simulated memory experiment (rotated surface code, X basis, d rounds, p = 0.6 %) with its detector outcomes and its true observable |
| routed, q, τ | the network's least confident fraction q of shots (\|c\| < τ_q, c the network's logit) goes to correlated PM; the rest keep the network's answer (c > 0 means 1) |
| composition | the paper's decoder: the network on the unrouted shots, correlated PM on the routed ones |
| disagreement rows | routed shots where the network's answer and correlated PM's differ; the joint decision changes nothing elsewhere |
| joint decision | on a disagreement row, answer 1 if c + w s > 0, 0 if it is < 0, correlated PM's answer on a tie |
| s | the signed gap: +gap when the matcher's answer is 1, -gap when it is 0 |
| plain gap | plain PM's complementary gap (`surface_v2/surf_v2/gapsw.py`): the lightest correction in the other class minus plain PM's own, on plain PM's graph; signed by plain PM's answer |
| correlated gap | the weight of a correlated decode forced into the other class minus correlated PM's own weight after its reweighting; signed by correlated PM's answer (details below) |
| routed oracle | the composition that, on every routed shot, takes whichever of the two answers is right; no rule that picks one of the two answers can do better |
| repairs, breaks | disagreement rows where the joint decision fixes, or spoils, the composition's answer |
| relative reduction | (composition - joint) / composition, failures on the evaluation half |
| fit half, evaluation half | the even and the odd 10,000-shot chunks of each test set |

## What was read and computed

- Test sets: the registered `sv2test` (2,000,000 shots at d = 7, 1,000,000 at d = 11-17) and D9-SHALLOW's `d9stest`
  (1,000,000 shots). The networks are the paper's: `d7_b1w64_sv2` (registered q = 2 %), `d9_b1w64_sh` (q = 6 %) and
  `d11/13/15/17_b5_sv2` (q = 0, 0, 1 and 1 %).
- Reused per-shot reads, all registered ones: the networks' logits (Alpine
  `<cluster-scratch>/surf_v2/eval/pod/reads/test/`), plain and correlated PM's answers (`reads/test/`),
  plain PM's gap (`gapsw/reads/test/`); at d = 9 the release's `d9_shallow/records/reads/`.
- τ for each q comes from the q rule's quantile on each network's calibration read, and it equals the policies' τ at
  q = 1, 2 and 4 % bit for bit. The policies' grid has no q = 5 %, so that τ was computed the same way. At d = 9 the
  registered τ = 2.337686538696289 is used.
- Every shot with \|c\| below τ at q = 5 % (q = 6 % at d = 9), between 49,000 and 100,000 per distance, was decoded
  again one at a time: plain PM, its gap, correlated PM with its weight, and the correlated gap. This ran as Alpine job
  AJ47, in the release's container with the deployed SURF-v2 code tree `ce8780e`, with one process per distance.
- Detection events come from the `sv2test` chunk files on Alpine, each checked against the manifest's sha256. The 100
  `d9stest` chunks were regenerated from the manifest's seeds (Stim 1.16.0, numpy 2.2.6), and all 100 match the
  manifest's sha256.
- The correlated gap is a proxy. PyMatching 2.4.0 returns the weight of correlated PM's second matching on the
  reweighted edges (`decode(..., return_weight=True, enable_correlations=True)`), but it does not expose the
  reweighted graph, so the other class's least weight on that same graph is out of reach. Instead the detector error
  model gets one extra node that every observable-flipping boundary component also touches (the split of
  `gapsw.py`, applied to the DEM so that the decomposition and its correlations survive). A correlated decode of
  that model with the node set to the other class gives the other class's correlated weight, with its own first pass
  and its own reweighting. The proxy can be negative: on 6-13 % of the routed shots the other class's correlated
  decode finds a lighter correction than correlated PM's own. On the disagreement rows among them correlated PM is
  wrong 46-68 % of the time, against 25-55 % on the other disagreement rows, and there the joint decision takes the
  network's answer at any finite w.
- w is fitted per distance, per q and per score on the fit half over {0, 10^(k/40) for k = -120..120, infinity},
  ARB's A3 grid, where infinity means the composition; the largest fit-half net gain wins and ties keep the larger w.
- The 95 % intervals are Newcombe's paired intervals (method 10); for the relative reduction the interval of the net
  is divided by the composition's count, as in ARB. The McNemar tests are exact and two-sided.

## Results per distance (evaluation half)

Failures are counts of shots: 1,000,000 evaluation shots at d = 7 and 500,000 at d = 9-17. "Network alone" and
"correlated PM alone" are the two decoders on every shot. At d = 11 and 13 the paper's row is the network alone.

| d | network | q | routed | disagreement rows | correlated PM alone | network alone | composition | joint (plain gap) | joint (corr. gap) | routed oracle | repairs / breaks (plain) | repairs / breaks (corr.) | relative reduction, plain [95 % CI] | relative reduction, corr. [95 % CI] | w plain / corr. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | d7_b1w64_sv2 | 2 % (registered) | 20,120 | 7,720 | 17,592 | 18,196 | 15,922 | 15,922 | 15,790 | 13,199 | 0 / 0 | 1,174 / 1,042 | 0.0 % [-0.0, 0.0] | 0.8 % [0.2, 1.4] | inf / 0.891 |
| 7 | d7_b1w64_sv2 | 5 % | 50,268 | 12,945 | 17,592 | 18,196 | 15,923 | 15,923 | 15,140 | 10,587 | 0 / 0 | 3,136 / 2,353 | 0.0 % [-0.0, 0.0] | 4.9 % [4.0, 5.8] | inf / 0.891 |
| 9 | d9_b1w64_sh | 6 % (registered) | 29,854 | 7,280 | 6,437 | 9,149 | 5,989 | 5,989 | 5,851 | 3,929 | 0 / 0 | 845 / 707 | 0.0 % [-0.1, 0.1] | 2.3 % [1.0, 3.6] | inf / 2.24 |
| 11 | d11_b5_sv2 | 2 % | 10,037 | 2,699 | 4,637 | 3,474 | 3,491 | 3,373 | 3,145 | 2,133 | 1,104 / 986 | 958 / 612 | 3.4 % [0.8, 6.0] | 9.9 % [7.7, 12.1] | 0.251 / 0.447 |
| 11 | d11_b5_sv2 | 5 % | 24,952 | 3,629 | 4,637 | 3,474 | 4,053 | 3,373 | 3,140 | 1,949 | 1,850 / 1,170 | 1,688 / 775 | 16.8 % [14.1, 19.4] | 22.5 % [20.1, 24.9] | 0.251 / 0.447 |
| 13 | d13_b5_sv2 | 2 % | 9,569 | 2,297 | 3,217 | 2,637 | 2,400 | 2,400 | 2,151 | 1,370 | 0 / 0 | 650 / 401 | 0.0 % [-0.2, 0.2] | 10.4 % [7.7, 13.0] | inf / 0.794 |
| 13 | d13_b5_sv2 | 5 % | 24,449 | 2,997 | 3,217 | 2,637 | 2,814 | 2,436 | 2,156 | 1,227 | 1,268 / 890 | 1,156 / 498 | 13.4 % [10.2, 16.7] | 23.4 % [20.6, 26.2] | 0.531 / 0.794 |
| 15 | d15_b5_sv2 | 1 % (registered) | 4,735 | 1,433 | 2,214 | 2,018 | 1,535 | 1,535 | 1,473 | 1,060 | 0 / 0 | 290 / 228 | 0.0 % [-0.2, 0.2] | 4.0 % [1.1, 7.0] | inf / 0.631 |
| 15 | d15_b5_sv2 | 2 % | 9,578 | 1,945 | 2,214 | 2,018 | 1,667 | 1,667 | 1,442 | 870 | 0 / 0 | 579 / 354 | 0.0 % [-0.2, 0.2] | 13.5 % [9.9, 17.1] | inf / 0.562 |
| 15 | d15_b5_sv2 | 5 % | 24,497 | 2,386 | 2,214 | 2,018 | 1,934 | 1,664 | 1,436 | 783 | 883 / 613 | 919 / 421 | 14.0 % [10.0, 17.9] | 25.7 % [22.1, 29.5] | 0.891 / 0.562 |
| 17 | d17_b5_sv2 | 1 % (registered) | 5,033 | 1,318 | 1,459 | 1,630 | 1,058 | 1,058 | 1,055 | 685 | 0 / 0 | 176 / 173 | 0.0 % [-0.4, 0.4] | 0.3 % [-3.2, 3.8] | inf / 0.944 |
| 17 | d17_b5_sv2 | 2 % | 9,897 | 1,669 | 1,459 | 1,630 | 1,091 | 1,091 | 1,008 | 526 | 0 / 0 | 360 / 277 | 0.0 % [-0.4, 0.4] | 7.6 % [3.1, 12.2] | inf / 0.708 |
| 17 | d17_b5_sv2 | 5 % | 25,027 | 1,987 | 1,459 | 1,630 | 1,251 | 1,189 | 996 | 447 | 524 / 462 | 570 / 315 | 5.0 % [0.0, 9.9] | 20.4 % [15.7, 25.1] | 2 / 0.75 |

### Against the paper's row

The registered composition is the paper's row. Every row below compares, shot by shot, the joint decision at q with
that row (negative: fewer failures).

| d | q | the paper's row: registered composition | joint (corr. gap) at q | difference [95 % CI] | exact McNemar p | joint (plain gap) at q | difference [95 % CI] |
|---|---|---|---|---|---|---|---|
| 7 | 2 % (registered) | 15,922 | 15,790 | -0.8 % [-1.4, -0.2] | 0.0054 | 15,922 | -0.0 % [-0.0, +0.0] |
| 7 | 5 % | 15,922 | 15,140 | -4.9 % [-5.7, -4.1] | 7.1e-34 | 15,923 | +0.0 % [-0.9, +0.9] |
| 9 | 6 % (registered) | 5,989 | 5,851 | -2.3 % [-3.6, -1.0] | 0.0005 | 5,989 | -0.0 % [-0.1, +0.1] |
| 11 | 2 % | 3,474 | 3,145 | -9.5 % [-11.4, -7.6] | 8.2e-23 | 3,373 | -2.9 % [-4.3, -1.5] |
| 11 | 5 % | 3,474 | 3,140 | -9.6 % [-11.6, -7.7] | 9.3e-23 | 3,373 | -2.9 % [-4.3, -1.5] |
| 13 | 2 % | 2,637 | 2,151 | -18.4 % [-21.1, -15.8] | 4.6e-44 | 2,400 | -9.0 % [-12.6, -5.4] |
| 13 | 5 % | 2,637 | 2,156 | -18.2 % [-21.0, -15.5] | 4.9e-40 | 2,436 | -7.6 % [-9.8, -5.5] |
| 15 | 1 % (registered) | 1,535 | 1,473 | -4.0 % [-7.0, -1.1] | 0.0073 | 1,535 | -0.0 % [-0.2, +0.2] |
| 15 | 2 % | 1,535 | 1,442 | -6.1 % [-9.4, -2.8] | 0.00037 | 1,667 | +8.6 % [+5.7, +11.5] |
| 15 | 5 % | 1,535 | 1,436 | -6.4 % [-9.9, -3.1] | 0.00021 | 1,664 | +8.4 % [+4.8, +12.0] |
| 17 | 1 % (registered) | 1,058 | 1,055 | -0.3 % [-3.8, +3.2] | 0.91 | 1,058 | -0.0 % [-0.4, +0.4] |
| 17 | 2 % | 1,058 | 1,008 | -4.7 % [-8.9, -0.6] | 0.029 | 1,091 | +3.1 % [-0.4, +6.6] |
| 17 | 5 % | 1,058 | 996 | -5.9 % [-10.2, -1.5] | 0.009 | 1,189 | +12.4 % [+7.4, +17.4] |

### The fit half

| d | q | fit-half disagreement rows | fit-half composition | w (plain) | fit-half net (plain) | w (corr.) | fit-half net (corr.) | eval half: net at the fitted w / at the eval half's own best w / at w = 1 (corr.) |
|---|---|---|---|---|---|---|---|---|
| 7 | 2 % | 7,747 | 15,707 | inf | +0 | 0.891 | +92 | +132 / +143 / +135 |
| 7 | 5 % | 12,979 | 15,787 | inf | +0 | 0.891 | +886 | +783 / +806 / +790 |
| 9 | 6 % | 7,325 | 5,989 | inf | +0 | 2.24 | +112 | +138 / +170 / +138 |
| 11 | 2 % | 2,650 | 3,429 | 0.251 | +172 | 0.447 | +373 | +346 / +357 / +301 |
| 11 | 5 % | 3,619 | 4,066 | 0.251 | +809 | 0.447 | +1,002 | +913 / +926 / +840 |
| 13 | 2 % | 2,310 | 2,337 | inf | +0 | 0.794 | +217 | +249 / +252 / +241 |
| 13 | 5 % | 3,032 | 2,735 | 0.531 | +357 | 0.794 | +608 | +658 / +664 / +641 |
| 15 | 1 % | 1,455 | 1,477 | inf | +0 | 0.631 | +26 | +62 / +64 / +54 |
| 15 | 2 % | 1,982 | 1,590 | inf | +0 | 0.562 | +185 | +225 / +234 / +213 |
| 15 | 5 % | 2,461 | 1,911 | 0.891 | +224 | 0.562 | +508 | +498 / +506 / +482 |
| 17 | 1 % | 1,260 | 1,122 | inf | +0 | 0.944 | +19 | +3 / +11 / +3 |
| 17 | 2 % | 1,618 | 1,158 | inf | +0 | 0.708 | +117 | +83 / +85 / +72 |
| 17 | 5 % | 1,941 | 1,307 | 2 | +77 | 0.75 | +286 | +255 / +259 / +245 |

## The gap's CPU time per routed shot

The times are means over the first 10,000 routed shots of each test set at the registered q (q = 2 % at d = 11 and 13;
d = 15 has only 9,588 routed shots at q = 1 %), in µs per routed shot. Each shot was decoded on its own, after 50
untimed warm-up shots, by one process pinned to one core of an AMD EPYC 7713 node (Alpine job AJ48, partition acpu
with the `epyc-7713` constraint, node cpu-node-25, shared with other jobs), the way the paper's CPU term times
correlated PM. The plain gap is plain PM's decode plus the split-graph decode, because the composition does not run
plain PM otherwise. The correlated gap is the forced-class correlated decode alone, because correlated PM's own weight
comes with the decode the composition already runs. The paper converts EPYC 7713 times to its Grace basis with a factor
of 0.84-0.97; that factor is not applied here. The last column is the registered gap-switching calibration's plain PM
plus split decode (exclusive EPYC 7713 node, core 8), averaged over the network's routed calibration shots at the same
q. Measured there, on other shots, it reads 4-24 % lower than this job's plain gap.

| d | q | routed shots timed | correlated PM (us) | plain gap (us) | correlated gap (us) | plain gap / correlated PM | correlated gap / correlated PM | share of routed shots where the two answers differ | plain gap, only on those (us per routed shot) | correlated gap, only on those (us per routed shot) | plain gap, registered gap-switching calibration (us) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 2 % | 10,000 | 54 | 67 | 76 | 1.24 | 1.40 | 39 % | 26 | 29 | 64 |
| 9 | 6 % | 10,000 | 122 | 151 | 206 | 1.24 | 1.69 | 25 % | 37 | 46 | - |
| 11 | 2 % | 10,000 | 267 | 300 | 395 | 1.12 | 1.48 | 27 % | 80 | 98 | 270 |
| 13 | 2 % | 10,000 | 484 | 574 | 794 | 1.19 | 1.64 | 24 % | 133 | 167 | 481 |
| 15 | 1 % | 9,588 | 899 | 1092 | 1472 | 1.21 | 1.64 | 30 % | 318 | 402 | 831 |
| 17 | 1 % | 10,000 | 1017 | 1490 | 2476 | 1.47 | 2.44 | 26 % | 356 | 530 | 1435 |

## Checks

- On every routed shot read, the recomputed plain and correlated PM answers equal the registered reads (all six
  distances), and the recomputed plain gap equals the gap-switching read bit for bit (d = 7, 11-17).
- The registered compositions recomputed from c, τ and the reads equal the network reads' recorded answers on all
  shots (d = 7, 11-17). At d = 9 the composition gives 11,978 failures, D9-SHALLOW's registered count.
- The joint decision's failures recomputed shot by shot equal the tables' (d = 7 and 11), and the comparison with the
  paper's row is checked against them for every row.
- The forced-class correlated decode returns the class it was set to on every shot. In correlated PM's own class it
  returns correlated PM's weight on 73-77 % of the routed shots; on the rest its first pass, forced into that class,
  reweights differently.
- The timing job recomputed both gaps on every timed shot, on another CPU model, and they equal the first job's bit
  for bit.

## What was not done

- The exact gap on the reweighted graph of correlated PM's second pass needs PyMatching's reweighted edge weights,
  which its API does not expose. The proxy above stands in for it.
- No new shots were drawn and no test was registered. The same evaluation halves serve every q and both scores, and
  w is fitted per distance and q, so none of these tables is a test.
- The deployed networks (r7xx) and the five-block network at d = 9 were not read. No GPU was used.
- The timing is exploratory: one process on a shared node, not an exclusive one, 10,000 routed shots per distance, and
  no conversion to the Grace basis.
- Raising q also raises the composition's time (more correlated PM decodes and a gap on each disagreement row). That
  cost was not projected against the speed bars of the q rule, which admit no routing at d = 11 and 13.

## Files

- `code/arb5_alpine.py`: the per-shot scores and times on Alpine; `code/arb5_job.sbatch` and `code/arb5_time.sbatch`
  the two jobs.
- `code/arb5_analysis.py`: the fit, the tables and `out/arb5_results.json` (every number above).
- `out/arb5_d<d>.json`, `out/t7713/time.json`: the jobs' records (checks, τ, hosts, versions). The per-shot arrays
  (`arb5_d<d>.npz`, `t7713/time_d<d>.npz`, 6-12 MB each) stay out of git; they are kept by the authors (`~/work/arb5/out/` on the
  Linux workstation).
