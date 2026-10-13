# Gross (BB144): routing the network's least confident shots to plain BP-OSD instead of sector Relay-BP

2026-10-07, branch `ising-bb-gross2`. Status: exploratory study, nothing registered. Part 1 (sections
1-2) uses CAL only. Part 2 (section 3) is a post hoc look at stage 3's published test outputs. The test was already
analysed once (RESULTS 7d06cd80, outputs 3aa121e8), so part 2 decides nothing and picks nothing for a claim. No test
shot was decoded again.

Terms. CAL: the calibration split (cal chunks 0-9, 100,000 shots at p = 0.004). Margin: the network's confidence in its
decision. tau_q: the q-quantile of the 100,000 CAL margins (`bb.route`'s rule). A shot is routed to the fallback iff its
margin < tau_q, so on CAL exactly the share q is routed. Hybrid: the frozen network g2_D12k113c10 (sha256
eb380295...) decides the kept shots, the fallback decides the routed ones. BP-OSD: belief propagation (BP), an
iterative message-passing decoder, followed by ordered-statistics decoding (OSD), a slower linear-algebra search that
runs only when BP has not converged. Here it is ldpc 2.4.1 with min-sum scale 1.0, 80 iterations, OSD-CS order 60, on
the X sector (`bb.opp bposd`, the registered ms80-CS60). cudaq-qec: NVIDIA's GPU BP-OSD with the same settings.
Sector Relay-BP: G-4's registered fallback (`bb.opp relay`). Composite: time per shot = GPU term 3.4296 ms + handoff
0.0226 ms (stage 3's H100 pod, slowest processes) + routed share x the fallback's mean time per routed shot. Paired
counts are written "both fail / only the hybrid fails / only the other fails"; p is the exact two-sided McNemar test
(a binomial test on the shots where exactly one of the two fails).

## Summary

- BP-OSD is a poor fallback for these shots. On the 1 % least confident CAL shots it decodes 49 % wrong. Sector
  Relay-BP gets 20 % of them wrong, the network itself 66 %.
- So the BP-OSD hybrid beats the network alone by about a quarter at best: 550 CAL failures at q = 1 % against 717
  (-23 %). It never reaches sector Relay-BP alone (490), let alone G-4's hybrid (409 at q = 0.5 %). From q = 2 % on,
  it is worse than the network alone (793 at q = 2 %), because BP-OSD fails many shots that the network gets right.
- It is cheap and steady. A routed shot costs BP-OSD about 0.2 s on one M4 core, against about 1.0 s for sector
  Relay-BP, and its time varies much less. The composite at q = 1 % is 5.5 ms with ldpc on the M4 (scaled from the
  Derecho read) and about 3.7 ms with cudaq-qec on the H100 (an estimate). That speed could only pay for a larger q,
  and a larger q adds failures.
- The post hoc test look agrees at all five q. The best BP-OSD hybrid has 1,550 failures (ldpc) and 1,556
  (cudaq-qec), both at q = 0.5 % of 300,000. Sector Relay-BP alone has 1,456, the registered hybrid 1,177 and the
  network alone 2,037.
- Recommendation: no registered BP-OSD-fallback cell, at any q (section 4). If a new cell is wanted for this network,
  CAL points to sector Relay-BP at q = 1 % instead: 260 CAL failures at an estimated 12.8 ms composite on the stage-3
  hosts, under the 15.65 ms bar.

## 1. What ran

| item | value | plain meaning |
|---|---|---|
| margins | the frozen graph32 engine's H100 CAL read, published in `data/bb/stage3_gross/cal_outputs_G-4.npz` (3aa121e8) | the margins that set G-4's tau |
| thresholds | tau = 0.5629 (q 0.5 %, = the frozen 0.5629201531410217), 2.209 (1 %), 6.277 (2 %), 9.154 (3 %), 14.32 (5 %) | `bb.route`'s quantile rule |
| shots decoded | the 5,000 CAL shots with margin < 14.32 (every shot routed at any q here), plus the first 1,000 CAL shots as a timing anchor | 6,000 decodes, 43 shots in both sets |
| job | Derecho CPU, PBS develop queue (cpudev), account <account>, one node (2 x AMD EPYC 7763), 64 single-threaded processes, 65 s wall, 47 core-minutes; code e396f1c6 | the node was shared with other jobs (load average 41 before, 73 after), so read times are loaded times |
| order | the slices of both sets ran in one pool in a seeded random order | anchor and routed shots saw the same load |
| checks | CAL reproduces the records: network 717, sector Relay-BP 490, G-4's hybrid 409 at q 0.5 %. BP-OSD fails 28 of the 1,000 anchor shots (2.8 %; 2.5 % on the test). The 43 shots decoded twice gave the same decisions | the read is sound |

How the times are turned into composites:

| fallback | time per routed shot | plain meaning |
|---|---|---|
| ldpc, read | the job's per-shot wall time on Derecho | measured on a loaded, shared node, outside the timing protocol |
| ldpc, one M4 core (scaled) | read time x 0.359, stage 3's M4/Derecho ratio on the same shots: BP-OSD took 79.69 ms per shot on the first 1,000 test shots on the idle M4 (session B, slowest of 3 processes) and 221.7 ms in the Derecho test read. This job's own anchor gives 0.352 (77.01 ms on the M4 in session A against 218.7 ms here, same 1,000 CAL shots) | both ratios agree within 2 %; both come from ordinary shots, so the scaled time of the OSD-heavy routed shots is an estimate |
| cudaq-qec, H100 (estimate) | never timed on routed shots. Stage 3 timed 300 batch-1 calls on the first 300 test shots: mean 15.65 ms, median 9.41 ms, 99th percentile 44.3 ms. ldpc's BP did not converge on 37.7 % of those shots. Model: a converged call costs the median, a call where OSD runs costs 26.0 ms (the value that gives the 15.65 ms mean). OSD runs on 92-98 % of routed shots, so a routed shot costs about 25 ms. High bound: every routed call at the 99th percentile | the cudaq-qec composite comes from this model; no routed shot was timed on the GPU |
| sector Relay-BP (beside) | CAL: the Alpine read times x 0.430, the M4/Alpine ratio on G-4's 500 routed CAL shots (1,006 ms per shot on the M4 in session A). Test: the Derecho read times x 0.307, the ratio on the first 1,000 routed test shots (session B) | an estimate except at q 0.5 %, where sessions A and B measured it |

## 2. CAL frontier (the study)

Network alone: 717 failures of 100,000. Sector Relay-BP alone: 490.

Accuracy:

| q | tau | routed shots | failures on the routed shots: network / BP-OSD / sector Relay-BP | BP-OSD hybrid failures | sector Relay-BP hybrid at the same q (beside) | plain meaning |
|---|---|---|---|---|---|---|
| 0.5 % | 0.5629 | 500 | 420 (84 %) / 263 (53 %) / 112 (22 %) | 560 | 409 (G-4's policy) | BP-OSD fixes 188 network failures and breaks 31 shots the network had right |
| 1 % | 2.209 | 1,000 | 656 (66 %) / 489 (49 %) / 199 (20 %) | 550 | 260 | the BP-OSD hybrid at its best |
| 2 % | 6.277 | 2,000 | 707 (35 %) / 783 (39 %) / 272 (14 %) | 793 | 282 | worse than the network alone |
| 3 % | 9.154 | 3,000 | 711 (24 %) / 1,021 (34 %) / 330 (11 %) | 1,027 | 336 | |
| 5 % | 14.32 | 5,000 | 715 (14 %) / 1,357 (27 %) / 382 (8 %) | 1,359 | 384 | |

Paired comparisons (both / only the hybrid / only the other; exact McNemar p):

| q | BP-OSD hybrid vs network alone | BP-OSD hybrid vs sector Relay-BP alone | plain meaning |
|---|---|---|---|
| 0.5 % | 529 / 31 / 188; p 1.3e-28 | 178 / 382 / 312; p 0.009 | better than the network, worse than Relay-BP |
| 1 % | 419 / 131 / 298; p 4.7e-16 | 181 / 369 / 309; p 0.023 | better than the network, worse than Relay-BP |
| 2 % | 392 / 401 / 325; p 0.005 | 222 / 571 / 268; p 5.5e-26 | worse than both |
| 3 % | 390 / 637 / 327; p 1.0e-23 | 252 / 775 / 238; p 6.8e-67 | worse than both |
| 5 % | 388 / 971 / 329; p 8.4e-74 | 285 / 1,074 / 205; p 2.2e-142 | worse than both |

Against sector Relay-BP's hybrid at the same q, the BP-OSD hybrid fails alone on 165-1,074 shots and the Relay-BP
hybrid alone on 14-99 (p <= 7e-34 at every q).

The frontier turns up because BP-OSD fails about half of the least confident shots wherever they sit, while the
network's failures are packed into the lowest margins. In the band between q 0.5 % and 1 % (500 shots) the network fails 236 and
BP-OSD 226. In the band between 1 % and 2 % (1,000 shots) the network fails only 51, BP-OSD 294 and Relay-BP 73.

Time per routed shot (ms):

| q | BP-OSD, Derecho read: mean (median, max) | BP-OSD, one M4 core (scaled) | routed shots where OSD ran | sector Relay-BP, one M4 core (estimate) | plain meaning |
|---|---|---|---|---|---|
| 0.5 % | 568 (536, 1,205) | 204 | 98 % | 1,006 (measured, session A) | BP-OSD about 5x cheaper |
| 1 % | 558 (529, 1,206) | 201 | 97 % | 937 | |
| 2 % | 548 (524, 1,206) | 197 | 96 % | 765 | |
| 3 % | 537 (519, 1,206) | 193 | 94 % | 658 | |
| 5 % | 520 (516, 1,242) | 187 | 92 % | 513 | Relay-BP gets cheaper as easier shots come in; BP-OSD does not, since OSD still runs |

BP-OSD's time per routed shot varies little. On the 500 routed shots at q 0.5 % its read time has a coefficient of variation
(standard deviation / mean) of 0.26 and a maximum 2.1x the mean. Sector Relay-BP's Alpine read on the same shots has
0.82 and 3.5x (at q 5 %: 0.36 and 2.4x against 1.30 and 6.8x).

Composite (ms per shot; G-4's bar is the fastest gating opponent, cudaq-qec at 15.65 ms on the H100):

| q | ldpc, read time | ldpc on the M4, scaled with stage 3's ratio (this job's anchor) | cudaq-qec on the H100, estimate (high bound) | sector Relay-BP hybrid on the M4, estimate | plain meaning |
|---|---|---|---|---|---|
| 0 | 3.45 | 3.45 | 3.45 | 3.45 | the network alone: GPU term + handoff |
| 0.5 % | 6.29 | 4.47 (4.45) | 3.58 (3.67) | 8.5 | stage 3 measured 8.41 ms for G-4 on the test |
| 1 % | 9.04 | 5.46 (5.42) | 3.71 (3.90) | 12.8 | |
| 2 % | 14.41 | 7.39 (7.31) | 3.96 (4.34) | 18.8 | |
| 3 % | 19.56 | 9.24 (9.12) | 4.20 (4.78) | 23.2 | |
| 5 % | 29.47 | 12.80 (12.62) | 4.68 (5.67) | 29.1 | |

Every M4-scaled ldpc composite and every cudaq-qec estimate is under the 15.65 ms bar (the loaded read times do not
count as a timing basis), so the speed condition would allow every q in the grid; accuracy is what rules the
fallback out. A faster GPU engine (GROSS_FAST_ENGINE.md: fusedt, 0.254 ms) would lower every composite here by about
3.2 ms and change no failure count.

## 3. Post hoc look at stage 3's test reads (exploratory; decides nothing)

CAL's thresholds applied to the 300,000 test margins. The fallbacks' decisions are the published stage-3 test reads:
ldpc BP-OSD and sector Relay-BP from Derecho, cudaq-qec from the H100. At q 0.5 %, sector Relay-BP gives G-4's
registered hybrid. Alone: network 2,037, sector Relay-BP 1,456, ldpc BP-OSD 7,494, cudaq-qec 7,460 (= RESULTS).

Accuracy:

| q (CAL) | routed test shots (share) | failures on the routed shots: network / ldpc / cudaq-qec / sector Relay-BP | ldpc hybrid | cudaq-qec hybrid | sector Relay-BP hybrid | plain meaning |
|---|---|---|---|---|---|---|
| 0.5 % | 1,462 (0.487 %) | 1,211 / 724 / 730 / 351 | 1,550 | 1,556 | 1,177 (registered) | BP-OSD hybrids above Relay-BP alone (1,456) |
| 1 % | 2,962 (0.987 %) | 1,825 / 1,424 / 1,384 / 624 | 1,636 | 1,596 | 836 | |
| 2 % | 6,063 (2.021 %) | 1,990 / 2,339 / 2,324 / 867 | 2,386 | 2,371 | 914 | BP-OSD hybrids worse than the network alone |
| 3 % | 8,896 (2.965 %) | 2,014 / 3,007 / 2,962 / 1,011 | 3,030 | 2,985 | 1,034 | |
| 5 % | 14,898 (4.966 %) | 2,029 / 3,872 / 3,847 / 1,164 | 3,880 | 3,855 | 1,172 | |

Paired comparisons (both / only the hybrid / only the other; exact McNemar p):

| q | ldpc hybrid vs network | ldpc hybrid vs sector Relay-BP | cudaq-qec hybrid vs network | cudaq-qec hybrid vs sector Relay-BP | sector Relay-BP hybrid vs sector Relay-BP |
|---|---|---|---|---|---|
| 0.5 % | 1,448 / 102 / 589; 4.0e-84 | 497 / 1,053 / 959; 0.038 | 1,463 / 93 / 574; 1.8e-85 | 506 / 1,050 / 950; 0.027 | 575 / 602 / 881; 4.5e-13 (= RESULTS) |
| 1 % | 1,181 / 455 / 856; 8.4e-29 | 567 / 1,069 / 889; 5.2e-5 | 1,181 / 415 / 856; 1.1e-35 | 570 / 1,026 / 886; 0.001 | 690 / 146 / 766; 4.1e-102 |
| 2 % | 1,100 / 1,286 / 937; 1.4e-13 | 697 / 1,689 / 759; 1.3e-80 | 1,100 / 1,271 / 937; 1.2e-12 | 706 / 1,665 / 750; 4.1e-79 | 897 / 17 / 559; 1.6e-141 |
| 3 % | 1,093 / 1,937 / 944; 8.4e-78 | 793 / 2,237 / 663; 6.3e-198 | 1,092 / 1,893 / 945; 5.0e-72 | 794 / 2,191 / 662; 1.2e-189 | 1,028 / 6 / 428; 4.1e-118 |
| 5 % | 1,088 / 2,792 / 949; 4.3e-208 | 878 / 3,002 / 578; < 1e-300 | 1,088 / 2,767 / 949; 9.3e-204 | 874 / 2,981 / 582; < 1e-300 | 1,171 / 1 / 285; 4.6e-84 |

Both BP-OSD variants behave alike (within 45 failures of each other at every q). At q 0.5-1 % each beats
the network and loses to sector Relay-BP alone; from q 2 % on, each loses to both.

Composite (ms per shot; the stage-3 Derecho read times, scaled as in section 1; within 2 % of CAL's values):

| q | ldpc, read time | ldpc on the M4, scaled | cudaq-qec on the H100, estimate | sector Relay-BP hybrid on the M4, estimate | plain meaning |
|---|---|---|---|---|---|
| 0.5 % | 6.24 | 4.45 | 3.58 | 8.5 | registered and measured: 8.41 |
| 1 % | 9.03 | 5.46 | 3.70 | 12.9 | |
| 2 % | 14.63 | 7.47 | 3.96 | 18.9 | |
| 3 % | 19.49 | 9.22 | 4.20 | 23.1 | |
| 5 % | 29.46 | 12.80 | 4.68 | 29.3 | |

## 4. Conclusions and the question of a registered cell

1. Hard shots for the network are hard for BP-OSD too. On the least confident 0.5-1 % of shots BP-OSD is wrong about
   half the time, against about a fifth for sector Relay-BP. A BP-OSD fallback removes roughly a quarter of the
   network's failures (from 717 to 550-560 on CAL, from 2,037 to 1,550-1,556 on the test) and no more.
2. A larger q does not help. Past q of about 1 %, BP-OSD fails far more of the moderately confident shots than the
   network does (29 % against 5 % in the 1-2 % band), so the hybrid gets worse. From q 2 % on it is worse than the
   network alone, on CAL and on the test.
3. The speed premise holds: about 0.2 s per routed shot on one M4 core, steady, about 5x cheaper than sector Relay-BP
   at q <= 1 %. With cudaq-qec on the H100 the whole hybrid would take an estimated 3.6-3.9 ms per shot at
   q 0.5-1 %, 0.1-0.5 ms more than the network alone. Cheap routing would only let q grow, and a larger q adds
   failures.
4. A registered cell with the BP-OSD fallback is not worth a fresh test set at any q. A cell like G-4 has to beat
   sector Relay-BP, and on CAL the best BP-OSD hybrid (550 at q 1 %) has 12 % more failures than sector Relay-BP
   alone (490). At every q more shots fail only in the hybrid than only in Relay-BP (p 0.009 to 2e-142), and the
   post hoc test look agrees at all five q (1,550-3,880 against 1,456). A weaker claim against BP-OSD and cudaq-qec
   alone needs no fallback: the network alone already beats both 3.7 to 1 (2,037 against 7,494 and 7,460) at
   3.45 ms against 15.65 ms, and G-4 has passed the stronger claim. Outside any claim, BP-OSD could still serve a
   deployment where no CPU can run Relay-BP: there a GPU BP-OSD fallback would cut the network's failures by about
   23 % for 0.1-0.5 ms more per shot.
5. CAL points to a different cell. With the M4 times now known, sector Relay-BP at q 1 % gives 260 CAL failures
   against 409 at G-4's q 0.5 %. Its composite on the stage-3 hosts is an estimated 12.8 ms, under the 15.65 ms bar;
   with fusedt's GPU term it would be about 9.6 ms. Stage 2 rejected q 1 % because its composite then used loaded
   Alpine times (25.2 ms). If a new cell is wanted for this network, this is the candidate CAL supports. Before any
   freeze it needs the M4 timing on the routed CAL shots, since the 12.8 ms is scaled from Alpine reads. The post hoc
   test value (836 at q 1 %) cannot count toward that choice. GROSS_FULLDEM_CAL.md discusses a different option, the
   full-DEM fallback with a weaker speed claim.

## 5. Files and reproduction

- Code: `bb/bposd_fallback.py` (select, decode, test-secs, analyze); Derecho job `bb/scripts/derecho/cpu_bposd_fallback.pbs`.
- Records in `data/bb/bposd_fallback/`: `cal_low5.json` (the 5,000 shots and the taus), `cal_bposd_low5.npz` (per
  shot: packed prediction, wall seconds, BP converged; for the routed set and the anchor; sha256 b40719e5...),
  `cal_bposd_low5.json` (host, versions, job), `derecho_job.log`, `read_secs.npz` (stage 3's per-shot read times for
  the 14,898 test shots below CAL's 5 % tau plus the first 1,000 test shots, 15,844 in all, and CAL's Relay-BP read times for the 5,000; taken
  from the Linux workstation copies after a check that their predictions equal the published ones), `frontier.json` (every number
  above).
- Reproduce the analysis from the repository alone: `PYTHONPATH=.:../qldpc_neural python -m bb.bposd_fallback analyze`
  (from `ising_compare/`, a few seconds). The decode: `qsub -N g6dbposd bb/scripts/derecho/cpu_bposd_fallback.pbs`
  with the code and CAL chunks at the paths in its header (Derecho `<cluster-scratch>/bb/g6d/`).
