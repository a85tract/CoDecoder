# Gross (BB144): routing the network's least confident shots to full-DEM Relay-BP, on CAL only

2026-10-07, branch `ising-bb-gross2`. Status: exploratory study, nothing registered. Cell G-4
(`STAGE3_GROSS_PLAN.md`), its freeze and its test are untouched. No test shot was read; every number below comes from
CAL (cal chunks 0-9, 100,000 shots at p = 0.004).

Terms. Sector Relay-BP: relay-bp 0.2.2 `RelayDecoderF32` with the authors' gross parameters on the X detectors only
(`bb.opp relay`, G-4's fallback). Full-DEM Relay-BP: the same decoder and parameters on the full detector error model,
both detector sectors, 1,728 detectors (`bb.opp relay_full`). Hybrid: the frozen network g2_D12k113c10 (sha256
eb380295..., its graph32 CAL read) decides shots with margin >= tau, full-DEM Relay-BP the rest. q: the share of CAL
shots below tau. Composite: time per shot = GPU term 3.406 ms + handoff 0.021 ms (the committed fine frontier's,
3.427 ms in total) + q x full-DEM Relay-BP's mean time on the routed shots. RAND: random CAL shots. Exact McNemar:
the two-sided binomial test on the shots where exactly one of two decoders fails.

## Summary

- Full-DEM Relay-BP is a much better fallback than sector Relay-BP. At q = 2 % the hybrid fails 156 times on CAL. With
  the sector fallback at the same q it fails 282 times (paired on all 100,000 shots: 88 vs 214 shots where only one
  fails, p = 3e-13). G-4's frozen hybrid fails 409 times, the network alone 717, sector Relay-BP alone 490.
- The hybrid also beats full-DEM Relay-BP alone, whose CAL count is estimated at 268 per 100,000 (95 % 199-415) and is
  at least 179. The q = 2 % hybrid wins even in its worst case (full-DEM Relay-BP counted right on every shot it did
  not decode): 156 vs 179, 10 vs 33 discordant shots, p = 0.0006.
- The hybrid is slow. Full-DEM Relay-BP spends 56 s per routed shot at q = 2 % (Derecho, 128 processes sharing the
  node), so the composite is 1.12 s per shot. That is 8.5 % of full-DEM Relay-BP's own mean on the same node (13.1 s),
  and 71 times G-4's 15.8 ms limit: no q meets G-4's registered speed rule (the cheapest, q = 0.25 %, costs 194 ms).
- Recommendation (section 5): a BB72-3F-style cell is worth testing at q = 2 % (tau = 6.276888) with the weaker speed
  claim, on a fresh test split of 150,000 shots (110,000 at the least). Its main cost is full-DEM Relay-BP on every test
  shot, about 550 Derecho core-hours.

## 1. What ran

| step | what | plain meaning |
|---|---|---|
| shot list | `bb.fullroute_gross pick` (commit 435fb675): the 3,000 CAL shots with the smallest margins (margin <= 9.1535, every q of the grid routes a subset of them) and 20,000 CAL shots drawn without replacement (seed 20261007); 602 are in both | `data/bb/fullroute_gross/pick.npz` |
| inputs | CAL shots sha256-checked against the manifest; the network's CAL read is the committed frontier's (engine signature and 717 failures checked); sector Relay-BP's CAL read reproduces the fine frontier (q 0.2 / 0.5 / 1 / 2 / 3 %: 577 / 409 / 260 / 282 / 336) | the study reuses stage 2's reads |
| decode | a Derecho job, develop queue, account <account>, one CPU node (AMD EPYC 7763), 128 single-threaded processes, code 435fb675 as a git archive, 16:59-17:32 PDT. Order: full-DEM Relay-BP on the 3,000-shot tail (21 min), sector Relay-BP on the same tail (1.2 min), then full-DEM Relay-BP on RAND in draw order | each shot decoded once; the hybrid and full-DEM Relay-BP alone share its decision |
| early stop | random shots took 11.4 s each on average, against the 4.3 s that sized the 60 core-hour budget. Finishing RAND would have taken about 113 core-hours, beyond the job's 45-min walltime, so I stopped the job at 32.3 min: 68.9 core-hours charged (128 cores x 32.3 min) | the whole tail and 6,252 of RAND's 20,000 shots are decoded |
| RAND used | its longest fully decoded prefix in draw order: 4,231 shots | a uniform random sample; the shots finished after it lean toward fast (easy) shots, so they enter only the worst-case bound |
| records | per shot: CAL index, decision, failure, wall time (`bb144_p0.004_cal_pick_relay_full.npz`, `..._relay.npz`); report `gross_cal.json`; job log | in `data/bb/fullroute_gross/` |

All times are per-shot wall times recorded inside the decoder while 128 processes shared the node. These loaded-host
numbers do not follow PLAN 2.7's timing protocol. Both sides of the weaker speed comparison come from this one read.

## 2. Full-DEM Relay-BP alone on CAL

| quantity | value | plain meaning |
|---|---|---|
| RAND prefix | 9 failures in 4,231 shots = 0.21 % (exact 95 % interval 0.10-0.40 %) | its block failure rate on random CAL shots |
| CAL estimate | 268 per 100,000 (95 % 199-415): the tail's 173 exactly + 4 / 4,100 RAND shots outside the tail x 97,000 | its expected CAL count, stratified by the tail |
| certain floor | 179 failures among the 8,650 decoded shots | full-DEM Relay-BP fails at least this often on CAL |
| beside, same 4,231 shots | network 34 (0.80 %), sector Relay-BP 17 (0.40 %) | on all of CAL: 717 and 490 per 100,000, so the prefix is typical |
| earlier estimate | 1 / 640 stage-1 dev shots (0.03-0.88 %) | consistent |

## 3. The frontier (CAL, 100,000 shots; tau = the q-quantile of CAL's margins)

Beside it: network alone 717 failures at 3.43 ms; sector Relay-BP alone 490; full-DEM Relay-BP alone about 268
(199-415) at 13.1 s per shot; G-4's frozen hybrid (sector fallback, q 0.5 %) 409 at 15.12 ms.

Accuracy. "Paired on RAND" compares the hybrid with full-DEM Relay-BP alone on the 4,231-shot RAND prefix: both fail /
only the hybrid / only full-DEM, exact McNemar p. "Worst case on CAL" does the same on all 100,000 shots with
full-DEM Relay-BP counted right on every shot it did not decode, which favors full-DEM Relay-BP.

| q | routed | hybrid failures | network fails in the routed set | full-DEM fails there | paired on RAND | worst case on CAL (only hybrid / only full-DEM, p) | sector-fallback hybrid | plain meaning |
|---|---|---|---|---|---|---|---|---|
| 0.25 % | 250 | 523 | 222 | 28 | 3 / 21 / 6, p 0.006 | 429 / 85 | 546 | worse than full-DEM alone |
| 0.5 % | 500 | 355 | 420 | 58 | 3 / 11 / 6, p 0.33 | 257 / 81 | 409 | likely worse (355 vs about 268) |
| 1 % | 1,000 | 164 | 656 | 103 | 3 / 0 / 6, p 0.031 | 55 / 70, p 0.21 | 260 | better on RAND, level in the worst case |
| 1.5 % | 1,500 | 157 | 695 | 135 | 4 / 0 / 5, p 0.062 | 21 / 43, p 0.008 | 263 | better even in the worst case |
| 2 % | 2,000 | 156 | 707 | 146 | 4 / 0 / 5, p 0.062 | 10 / 33, p 0.0006 | 282 | the fewest failures, the strongest worst case |
| 3 % | 3,000 | 179 | 711 | 173 | 5 / 0 / 4, p 0.12 | 6 / 6, p 1 | 336 | the worst case cannot separate them |

The full-DEM fallback beats the sector fallback at every q (paired on all 100,000 shots, p from 3e-3 at q 0.25 % to
3e-16 at q 3 %). The RAND comparison has few discordant shots (at most 27) and the worst case is a bound, so the
CAL-wide gap at q 1-3 % rests on the estimate above: about 268 for full-DEM Relay-BP alone against 156-179.

Time. Full-DEM Relay-BP's times are from the Derecho read. Sector Relay-BP's are Alpine's CAL read, as in the committed
frontier. The yardstick (section 4) puts Alpine at 0.705 x Derecho; the "Alpine-equivalent" column applies that
factor.

| q | full-DEM s per routed shot | composite ms | composite / full-DEM's own mean | Alpine-equivalent composite ms | sector s per routed shot (Alpine) | sector-fallback composite ms (Alpine) | plain meaning |
|---|---|---|---|---|---|---|---|
| 0.25 % | 76.2 | 194 | 1.5 % | 138 | 2.12 | 8.7 | the hardest shots take longest |
| 0.5 % | 74.2 | 374 | 2.9 % | 265 | 2.34 | 15.1 | |
| 1 % | 65.5 | 658 | 5.0 % | 465 | 2.18 | 25.2 | |
| 1.5 % | 60.5 | 911 | 7.0 % | 643 | 1.96 | 32.8 | |
| 2 % | 55.7 | 1,118 | 8.5 % | 789 | 1.78 | 39.0 | 20 x the sector-fallback composite on Alpine's basis |
| 3 % | 48.0 | 1,444 | 11.0 % | 1,019 | 1.53 | 49.3 | |

Every q meets the weaker speed rule (composite <= full-DEM Relay-BP's own mean, 13.1 s); none meets G-4's registered
rule (<= 15.8 ms).

## 4. Time tail of full-DEM Relay-BP (Derecho, 128 processes sharing one node)

| shots | mean | median | 90th pct | 99th pct | 99.9th pct | max | share over 10 s | share over 30 s |
|---|---|---|---|---|---|---|---|---|
| RAND prefix (4,231) | 13.1 s | 5.6 s | 29.5 s | 120 s | 182 s | 259 s | 30 % | 9.8 % |
| tail (3,000 lowest margins) | 48.0 s | 26.4 s | 122 s | 183 s | 282 s | 286 s | 76 % | 47 % |

- The network's least confident shots are also the hardest for Relay-BP: on the tail it often needs many sets before it
  finds the 5 solutions that stop it (at most 80 + 300 x 60 iterations).
- Yardstick: sector Relay-BP on the same 3,000 tail shots took 2.17 s each here and 1.53 s in Alpine's CAL read, a
  factor of 0.705. Scaled to Alpine, full-DEM Relay-BP's mean on random shots is about 9.2 s, above the 4.3 s measured
  on 640 stage-1 dev shots; with a 99th percentile of 120 s, a 640-shot mean is unstable.
- The two sector reads agree on 2,679 of the 3,000 tail shots and fail 325 vs 330 times. Relay-BP draws its per-set
  memory strengths at random (gamma_dist_interval), which is most likely why its decision on a hard shot can differ
  between reads; the failure counts agree within noise.

## 5. Recommendation: is a BB72-3F-style cell worth testing, and at which q?

Yes, at q = 2 %, if the authors want an accuracy claim against full-DEM Relay-BP, the registered reference decoder
(PLAN 3.2). The speed side would be a weak claim.

- q = 2 % (tau = 6.276888, the 2 % quantile of CAL's margins). It has the fewest CAL failures (156; 157 at 1.5 %,
  164 at 1 %, 179 at 3 %; the frontier is flat over 1-2 %, so tau being in-sample matters little) and the strongest
  worst case (p 0.0006, below 0.05 / 3). BB72-3F's rule (the fewest hybrid CAL failures among admissible q) picks it
  too.
- Claim: on the test, the hybrid fails less often than full-DEM Relay-BP alone (exact McNemar p < 0.05 / 3 and the
  Newcombe interval below 0, as in Amendment F). Speed condition: composite <= full-DEM Relay-BP's own quiet-host mean
  time per shot. On CAL's loaded read it holds 12-fold (1.12 s vs 13.1 s). The fastest gating opponent's time is
  reported beside it and not gated.
- Test size: PLAN 3.3's simulation (alpha 0.05 / 3, power 0.90) on the worst-case CAL counts (146 both fail, 10 only
  the hybrid, 33 only full-DEM per 100,000) gives 110,000; RAND's counts give the 100,000 floor. At 150,000 the power
  is 0.97 on the worst-case counts, and 0.99 if the kept-side failures double (tau is in-sample) while full-DEM
  Relay-BP's rate outside the tail sits at its 95 % lower end. Only the bound's extreme (full-DEM Relay-BP never
  failing on the 91,350 shots it did not decode) together with doubled kept-side failures defeats it (power 0.38).
- A fresh test split. G-4's test split has already been read by the network and the opponents.
- Costs:
  - Full-DEM Relay-BP on every test shot for the paired test: at 13.1 s per shot, about 550 Derecho core-hours for
    150,000 shots (400 for 110,000).
  - A graph32 network read of the fresh split on an H100.
  - Quiet-host timing: BB72-3F's samples (1,000 routed and 2,000 random shots, three fresh processes) would take more
    than a day on the Mac mini at these times. The 12-fold slack allows far smaller samples, for example 200 routed and
    500 random shots per process.
  - A new alpha share of 0.05 / 3, which raises the study-wide union bound by 0.017, as each Amendment F cell did.
- Against it: 1.1 s per shot is about 74 times G-4's 15 ms composite and far from real-time decoding. G-4's sector
  hybrid is the speed result; this cell would be an accuracy result only.
- If a tighter estimate of full-DEM Relay-BP's own rate is wanted first, resubmitting the same job resumes RAND where it
  stopped (about 45-50 core-hours more for the rest of the 20,000).

## 6. Reproduce

```
# shot list (Linux workstation; needs the network's CAL read in BB_PREDS and the CAL chunks in BB_SHOTS)
python -m bb.fullroute_gross pick
# reads (Derecho; job script as run, code 435fb675)
qsub -N g6afull bb/scripts/derecho/cpu_fullroute_gross.pbs
# report (Linux workstation, a second): BB_PREDS holds the network's and sector Relay-BP's CAL reads and the two records
python -m bb.fullroute_gross report --out data/bb/fullroute_gross/gross_cal.json
```
The reads ran at 435fb675. This commit adds the worst-case comparison and the test sizes to `report` only; `pick` and
`decode` are unchanged.
