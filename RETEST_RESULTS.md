# RETEST_RESULTS: every re-tested number and verdict against its original

This file lists the results of the re-test registered in `RETEST.md` (freeze commit dd332c5) next to the numbers the
paper reports, and says for each whether it repeats. The rules are the frozen ones of `RETEST.md` section 5: "better"
is repeated only by "better" from the frozen analysis on the re-test's shots; non-inferiority is not "better"; a
non-significant result ("level", "inconclusive") is not equality and neither repeats nor refutes a "better"; "worse"
contradicts it. Numbers without a verdict (counts, shares, differences) are listed with their originals and carry no
verdict of their own. Speed is never re-measured: every speed condition is an estimate from the registered timing
sessions (`RETEST.md` section 4).

Seeds come from drand round 6547616 (fetched from two endpoints that agreed), the first round at least 600 s after the
freeze reached GitHub (2026-10-13T02:34:48Z). Each test's records and analysis outputs are in `retest/run/`.

## Summary

| test | original | re-test | repeats |
|---|---|---|---|
| Table I, BB72 (6 cells, 17 gated comparisons) | every comparison "better"; all 6 claims | every comparison "better"; all 6 claims | yes, all 6 (accuracy confirmed, speed estimated) |
| Table I, gross code (2 cells, 7 gated comparisons) | every comparison "better"; both claims | every comparison "better"; both claims | yes, both (accuracy confirmed, speed estimated) |
| Cascade on BB72, D1 (4 comparisons, one Holm family) | all 4 "better"; "more accurate and faster" at 0.2 % and 0.3 % | all 4 "better"; the claim holds at both p on both engines | yes |
| Cascade on the surface code, d = 13 (40 hypotheses, one Holm family) | 31 rejected; pair better than Cascade alone at both p in both modes | 34 rejected; pair better at 0.35 %, only non-inferior at 0.6 %; new "worse" for C1 and C4 against Cascade alone at 0.35 % | partly: the pair's "better" repeats at 0.35 % in both modes but not at 0.6 %; headline chain C5, P1, P2 and the d = 17 decision repeat; C5 is "faster and more accurate" at 0.35 % only |
| Output heads, surface code (14 verdicts) | R1 worse, R2 worse, R3 inconclusive, U1 better, U2 inconclusive, W1 worse, W2 inconclusive, at q = 0 and q = 1 % | the same 14 | yes, 14 of 14 |
| Output heads, BB72 (10 verdicts) | R1 worse, R2 worse, U1 inconclusive, U2 better, W1 worse, at q = 0 and q = 2 % | the same, except U1 at q = 0: worse | 9 of 10; U1 at q = 0 is a new result, so the paper's "a consistent benefit was not established" for the joint class score no longer holds at q = 0 |
| X2 (exploratory, no verdict) | Figure "why" (b): 68 % and 75 % below the threshold, 18 % and 19 % above | 67 % and 75 %, 8 % and 20 % | reported apart |

## 1. Table I

Cells: BB72-3 and BB72-2 are the fast mode (the network alone) at p = 0.3 % (200,000 shots) and 0.2 % (300,000);
BB72-3S and BB72-2S the balanced mode (Amendment S); BB72-3F and BB72-2F the accurate mode; G-4R (balanced) and G-4F
(accurate) the gross code at p = 0.4 % (150,000 shots). "Composite" is the cell's estimated time per shot and "bar" the
fastest gating opponent's time, both in ms; the speed condition is composite below bar. "Routed" counts the shots the
network hands to its fallback decoder.

| cell | failures: original | re-test | routed: original | re-test | composite (ms): original | re-test | bar (ms) | claim: original | re-test |
|---|---|---|---|---|---|---|---|---|---|
| BB72-2 | 254 | 269 | 0 | 0 | 0.1020 | 0.1020 | 0.5009 | True | True |
| BB72-2F | 171 | 204 | 721 | 735 | 0.3238 | 0.3281 | 0.5009 | True | True |
| BB72-3 | 1,341 | 1,376 | 0 | 0 | 0.1019 | 0.1019 | 0.9566 | True | True |
| BB72-3F | 877 | 857 | 3,960 | 3,977 | 3.4306 | 3.4449 | 58.5963 | True | True |
| BB72-2S | 290 | 297 | 721 | 735 | 0.1147 | 0.1149 | 0.5009 | True | True |
| BB72-3S | 1,571 | 1,554 | 3,960 | 3,977 | 0.2602 | 0.2609 | 0.9566 | True | True |
| G-4F | 229 | 234 | 2,983 | 3,020 | 232.1868 | 235.0633 | 2774.7069 | True | True |
| G-4R | 387 | 407 | 1,447 | 1,516 | 9.3584 | 9.7913 | 13.8994 | True | True |

Every gated comparison, failures of the cell and of its opponent on the same shots ("only cell / only opponent" are the
shots where exactly one of the two fails; exact two-sided McNemar test, verdict "better" when p is below 0.0167 and the
paired Newcombe interval lies below zero):

| cell | opponent | failures, cell / opponent: original | re-test | only cell / only opponent: original | re-test | McNemar p: original | re-test | verdict: original | re-test | repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| BB72-2 | BP-OSD | 254 / 903 | 269 / 880 | 110 / 759 | 109 / 720 | 5.4e-120 | 3.6e-111 | better | better | yes |
| BB72-2 | cudaq-qec BP-OSD | 254 / 917 | 269 / 856 | 111 / 774 | 115 / 702 | 4.9e-123 | 1.6e-103 | better | better | yes |
| BB72-2 | sector Relay-BP | 254 / 812 | 269 / 767 | 111 / 669 | 116 / 614 | 6e-98 | 1.1e-82 | better | better | yes |
| BB72-2F | BP-OSD | 171 / 903 | 204 / 880 | 59 / 791 | 67 / 743 | 1.8e-164 | 3.9e-145 | better | better | yes |
| BB72-2F | cudaq-qec BP-OSD | 171 / 917 | 204 / 856 | 54 / 800 | 71 / 723 | 2.8e-171 | 7.7e-137 | better | better | yes |
| BB72-2F | sector Relay-BP | 171 / 812 | 204 / 767 | 64 / 705 | 72 / 635 | 1.9e-137 | 1.9e-113 | better | better | yes |
| BB72-2F | full-DEM Relay-BP | 171 / 274 | 204 / 288 | 13 / 116 | 27 / 111 | 7.8e-22 | 2.7e-13 | better | better | yes |
| BB72-3 | BP-OSD | 1,341 / 3,299 | 1,376 / 3,337 | 483 / 2,441 | 516 / 2,477 | 4e-313 | 1.6e-305 | better | better | yes |
| BB72-3 | cudaq-qec BP-OSD | 1,341 / 3,324 | 1,376 / 3,342 | 482 / 2,465 | 525 / 2,491 | 5.8e-319 | 2e-304 | better | better | yes |
| BB72-3 | sector Relay-BP | 1,341 / 2,704 | 1,376 / 2,769 | 537 / 1,900 | 585 / 1,978 | 1.8e-177 | 1.2e-175 | better | better | yes |
| BB72-3F | full-DEM Relay-BP | 877 / 1,092 | 857 / 1,107 | 73 / 288 | 86 / 336 | 2.5e-31 | 5.5e-36 | better | better | yes |
| BB72-2S | BP-OSD | 290 / 903 | 297 / 880 | 60 / 673 | 57 / 640 | 3.9e-132 | 9.1e-126 | better | better | yes |
| BB72-2S | cudaq-qec BP-OSD | 290 / 917 | 297 / 856 | 65 / 692 | 62 / 621 | 2.9e-133 | 5.5e-117 | better | better | yes |
| BB72-2S | sector Relay-BP | 290 / 812 | 297 / 767 | 10 / 532 | 13 / 483 | 7.9e-143 | 1.5e-124 | better | better | yes |
| BB72-3S | BP-OSD | 1,571 / 3,299 | 1,554 / 3,337 | 287 / 2,015 | 334 / 2,117 | 6.9e-319 | 5.2e-316 | better | better | yes |
| BB72-3S | cudaq-qec BP-OSD | 1,571 / 3,324 | 1,554 / 3,342 | 295 / 2,048 | 309 / 2,097 | 4.2e-322 | 0 (below 1e-300) | better | better | yes |
| BB72-3S | sector Relay-BP | 1,571 / 2,704 | 1,554 / 2,769 | 50 / 1,183 | 45 / 1,260 | 6e-282 | 1.8e-309 | better | better | yes |
| G-4F | BP-OSD | 229 / 3,865 | 234 / 3,646 | 81 / 3,717 | 83 / 3,495 | 0 (below 1e-300) | 0 (below 1e-300) | better | better | yes |
| G-4F | cudaq-qec BP-OSD | 229 / 3,789 | 234 / 3,754 | 84 / 3,644 | 77 / 3,597 | 0 (below 1e-300) | 0 (below 1e-300) | better | better | yes |
| G-4F | sector Relay-BP | 229 / 716 | 234 / 733 | 120 / 607 | 129 / 628 | 3.9e-79 | 1.6e-79 | better | better | yes |
| G-4F | full-DEM Relay-BP | 229 / 484 | 234 / 472 | 7 / 262 | 13 / 251 | 4e-68 | 2.6e-58 | better | better | yes |
| G-4R | BP-OSD | 387 / 3,865 | 407 / 3,646 | 84 / 3,562 | 98 / 3,337 | 0 (below 1e-300) | 0 (below 1e-300) | better | better | yes |
| G-4R | cudaq-qec BP-OSD | 387 / 3,789 | 407 / 3,754 | 95 / 3,497 | 90 / 3,437 | 0 (below 1e-300) | 0 (below 1e-300) | better | better | yes |
| G-4R | sector Relay-BP | 387 / 716 | 407 / 733 | 57 / 386 | 65 / 391 | 4.1e-61 | 8.8e-58 | better | better | yes |

Reported beside, not gated (as frozen):

| item | original | re-test | same outcome |
|---|---|---|---|
| second BB72 network (r12) alone, 0.3 %, 200,000 shots | 1,387 | 1,398 | - |
| second BB72 network alone, 0.2 %, 300,000 shots | 254 | 269 | - |
| BB72-3 against full-DEM Relay-BP, first 200,000 shots | 1,341 vs 1,092, p 4.11e-11, worse | 1,376 vs 1,107, p 2.70e-12, worse | yes |
| BB72-2 against full-DEM Relay-BP, first 200,000 shots | 166 vs 173, p 0.69, level | 182 vs 192, p 0.55, level | yes |
| BB72-2 against full-DEM Relay-BP, all 300,000 shots | 254 vs 274, p 0.30, level | 269 vs 288, p 0.33, level | yes |
| gross network alone | 970 | 1,024 | - |
| G-4 at its old q = 0.5 % | 560 (720 routed) | 595 (719 routed) | - |

## 2. Cascade on BB72 (D1)

The pair (ConvLines-BB S55'-deep, handing the shots with margin below tau1 to Cascade C14') against Cascade alone,
11,000,000 shots at p = 0.2 % and 17,000,000 at 0.3 %, one Holm family at 0.05 over H1e, H2e, H1c, H2c. The compiled
verdict of a point (H1c, H2c) is one joint verdict over the single and batched paths.

| comparison | p | path | Cascade alone / pair failures: original | re-test | only pair / only Cascade: original | re-test | McNemar p: original | re-test | verdict: original | re-test | repeats |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1e | 0.2 % | eager | 7,922 / 7,855 | 8,063 / 7,981 | 19 / 86 | 8 / 90 | 2.3e-11 | 1.09e-18 | better | better | yes |
| H1c | 0.2 % | single | 7,912 / 7,853 | 8,074 / 7,986 | 19 / 78 | 8 / 96 | 1.15e-09 | 2.77e-20 | better | better | yes |
| H1c | 0.2 % | batched | 7,918 / 7,858 | 8,073 / 7,984 | 19 / 79 | 8 / 97 | 7.13e-10 | 1.5e-20 | better | better | yes |
| H2e | 0.3 % | eager | 76,730 / 76,469 | 75,957 / 75,700 | 81 / 342 | 88 / 345 | 3.1e-39 | 4.97e-37 | better | better | yes |
| H2c | 0.3 % | single | 76,676 / 76,409 | 75,979 / 75,740 | 81 / 348 | 90 / 329 | 1.71e-40 | 4.48e-33 | better | better | yes |
| H2c | 0.3 % | batched | 76,711 / 76,445 | 75,973 / 75,735 | 81 / 347 | 90 / 328 | 2.77e-40 | 7.04e-33 | better | better | yes |

| p | routed: original | re-test | kept repairs / breaks (eager): original | re-test | Cascade's failures on routed shots (eager): original | re-test |
|---|---|---|---|---|---|---|
| 0.2 % | 220,058 (2.0005 %) | 219,891 (1.9990 %) | 86 / 19 | 90 / 8 | 7,598 of 7,922 (95.9 %) | 7,719 of 8,063 (95.7 %) |
| 0.3 % | 1,701,544 (10.0091 %) | 1,699,201 (9.9953 %) | 342 / 81 | 345 / 88 | 74,411 of 76,730 (97.0 %) | 73,617 of 75,957 (96.9 %) |

"More accurate and faster" (`claims`): compiled verdict "better", and the pair's single-shot mean and batched time per
shot below Cascade alone's with all S checks zero in the inherited timing session D1-S. Original: holds at both p on
both engines (`base`, `new`). Re-test: holds at both p on both engines. The timing columns and speed-up ratios (5.9-12.7
per single shot, 5.1-16.4 per batch) are D1-S's, inherited and not re-measured.

The paper's other D1 numbers, recomputed from the re-test's `d1_result.json` (no verdict of their own):

| paper's number | definition | original | re-test |
|---|---|---|---|
| "about 0.75 % and 0.35 % less often" | `diff_share_of_C`, single / batched | 0.7457 / 0.7578 % and 0.3482 / 0.3468 % | 1.0899 / 1.1024 % and 0.3146 / 0.3133 % |
| appendix differences per round | `paths.single.stats_95` diff_ler [diff_lo, diff_hi] | 5.36e-06 [3.59, 7.16]e-6 and 1.57e-05 [1.33, 1.81]e-5 | 8.00e-06 [6.17, 9.87]e-6 and 1.41e-05 [1.17, 1.64]e-5 |
| "4.0 % and 3.0 % of Cascade's failures on the kept shots" | `C_fail_kept` / `fail_b`, compiled single / batched | 3.99 / 4.00 % and 3.03 / 3.03 % | 4.33 / 4.35 % and 3.06 / 3.06 % |
| "96-97 %" of Cascade's failures are on routed shots | `paths.eager.C_fail_routed` / `fail_b` | 95.9 % and 97.0 % | 95.7 % and 96.9 % |
| repairs 86 and 342 | `paths.eager.kept_repairs` | 86 and 342 | 90 and 345 |
| breaks 19 and 81 | `paths.eager.kept_breaks` | 19 and 81 | 8 and 88 |
| routed | `paths.eager.routed` | 220,058 (2.0005 %) and 1,701,544 (10.0091 %) | 219,891 (1.9990 %) and 1,699,201 (9.9953 %) |

(The differences in the appendix are written there as positive numbers, Cascade alone minus the pair; the file stores
the pair minus Cascade alone.)

X2, exploratory (Figure "why" panel (b); no say in any verdict). Among the shots where exactly one of S and Cascade
fails, the share where S is the one that fails (X2's `bb_join.py` cells "S only fails" against "C only fails"):

| p | below the threshold (routed): original | re-test | above it (kept): original | re-test |
|---|---|---|---|---|
| 0.2 % | 68 % (4,716 / 2,223) | 67 % (4,710 / 2,293) | 18 % (19 / 86) | 8 % (8 / 90) |
| 0.3 % | 75 % (54,410 / 18,243) | 75 % (54,543 / 18,142) | 19 % (81 / 342) | 20 % (88 / 345) |

The 8 % at 0.2 % rests on 98 shots.

## 3. Cascade on the surface code at d = 13

This section repeats the paper's d = 13 surface-code test on fresh random shots. A shot is one simulated 13-round memory experiment (X memory, legacy noise model); a decoder fails on a shot when its predicted logical outcome is wrong. Five decoders are scored on the same shots. S is ConvLines, B is Cascade, and M is correlated PyMatching 2.4.0. C1 = S -> B -> M (the chain), C2 = B alone (Cascade alone), C3 = M alone (correlated PyMatching), C4 = S -> M (ConvLines with correlated PyMatching) and C5 = S -> B (the pair). In a routed decoder S decodes every shot and hands a shot to the next stage when the absolute value of its confidence is below a frozen threshold; for the pair the thresholds send about 10 % of the shots to Cascade at p = 0.6 % and about 1 % at p = 0.35 %. The two points are p = 0.6 % (1,000,000 shots) and p = 0.35 % (30,000,000 shots), each decoded in single mode (one shot at a time) and in batched mode. Networks, thresholds, matcher settings, shot counts, statistics and analysis code are the frozen ones of the original test (lock commit bddd3f91da1f0284766d06f8aa7a3b6886eee6fe, lock sha256 5767020407ed11f3ec3a2ef985f2c41e7d0ac93a489cb7929e16a8159b5a98f1, the same in both result files). Only the seeds of the confirmation shots are new.

Each pair of decoders is compared on the shots both decoders saw with an exact McNemar test. The test looks only at the shots where exactly one of the two decoders failed ("X only" and "Y only" below) and asks whether the two counts differ by more than a fair coin would allow. Superiority hypotheses (kind S in the files) use the two-sided exact test; the label "better" or "worse" follows from the sign of the difference. Non-inferiority hypotheses (kind N) ask whether X fails at most a fixed margin more often than Y; the margins are 0.0005102 at p = 0.6 % and 7.5e-6 at p = 0.35 % (stored as 7.499999999999999e-06), as absolute differences in failure probability, 10 % of Cascade alone's failure rate on the development shots, and they are not recomputed from the new shots. All 40 hypotheses (2 error rates x 2 modes x (8 superiority + 2 non-inferiority)) form one Holm family at level 0.05. Holm's procedure ranks the p-values from the smallest; the hypothesis at rank r (r = 0, 1, ...) is rejected when its p-value is below 0.05 / (40 - r) and every smaller p-value was rejected; the first failure ends the procedure. The "Holm level" in the table is the threshold the hypothesis was compared with; the files store none for a hypothesis that was not rejected.

The frozen rules for reading the re-test against the paper, fixed before any shot was drawn:

- The unit is one cell (error rate, mode, comparison). The paper's "better" is repeated only by the re-test's label "better" (exact McNemar rejected at its Holm level with fewer failures for the first-named decoder).
- "Non-inferior" is not "better" and does not repeat the claim. "Level" (not significant) is neither equality nor a refutation: it does not repeat the claim and does not refute it. "Worse" contradicts it.
- The sentence "the pair failed less often than Cascade alone at both p in both modes" counts as repeated only when all four cells are "better". Otherwise the cells that repeated and the ones that did not are reported with their labels, counts and p-values. The percentages in the text are recomputed from the new counts and carry no verdict.
- Timing is inherited from the registered timing session and is not re-measured.

Sources. The original analysis is `cascade/d13/out/confirm_d13.json` with `SURFACE_RESULT_draft.md` and its `.json` beside it; the re-test analysis is `retest/run/d13/route/confirm_d13.json` with `SURFACE_RESULT_retest.md` and its `.json`. The public copy of the original `confirm_d13.json` has its cluster paths masked; with the mask undone it hashes to 09869a77ec8e8d1108731a5cdc823ad13f48b55f77440ff369ca63dcedf53b31, the value `RETEST.md` gives.

### The 40 hypotheses

The JSON hypothesis records carry no separate id field. Each record is identified by the fields (p, mode, test, x, y), where test is S (superiority) or N (non-inferiority) and x, y name the decoders compared; the id below writes that key in the style of the keys of the files' `labels` object (`0.006|single|C1-C2`) with the test letter added. Hypotheses were matched between the two files by this key, not by position: all 40 keys are unique in each file and the two key sets are equal (the file order also happens to be identical, but it was not used). "X only / Y only" are the numbers of shots where only the first-named (X) or only the second-named (Y) decoder failed. p-values are rounded to 3 significant figures, Holm levels to 4, as in the files' own report. "Same outcome" means the same reject / not reject decision.

| id | p | mode | comparison | kind | original X only / Y only | original p-value | original Holm level | original rejected | re-test X only / Y only | re-test p-value | re-test Holm level | re-test rejected | same outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.006\|single\|S\|C1-C2 | 0.6 % | single | C1 vs C2 | superiority | 661 / 1,252 | 4.03e-42 | 0.001786 (0.05/28) | yes | 708 / 1,293 | 1.69e-39 | 0.001852 (0.05/27) | yes | yes |
| 0.006\|single\|S\|C5-C2 | 0.6 % | single | C5 vs C2 | superiority | 13 / 40 | 0.000269 | 0.003571 (0.05/14) | yes | 21 / 40 | 0.0204 | none | no | no |
| 0.006\|single\|S\|C4-C2 | 0.6 % | single | C4 vs C2 | superiority | 2,415 / 1,990 | 1.62e-10 | 0.003125 (0.05/16) | yes | 2,418 / 2,013 | 1.25e-09 | 0.002778 (0.05/18) | yes | yes |
| 0.006\|single\|S\|C3-C2 | 0.6 % | single | C3 vs C2 | superiority | 3,886 / 2,529 | 9.09e-65 | 0.001515 (0.05/33) | yes | 3,896 / 2,606 | 6.88e-58 | 0.001471 (0.05/34) | yes | yes |
| 0.006\|single\|S\|C1-C4 | 0.6 % | single | C1 vs C4 | superiority | 1,062 / 2,078 | 1.26e-74 | 0.001429 (0.05/35) | yes | 1,090 / 2,080 | 3.33e-70 | 0.001389 (0.05/36) | yes | yes |
| 0.006\|single\|S\|C5-C4 | 0.6 % | single | C5 vs C4 | superiority | 1,950 / 2,402 | 7.78e-12 | 0.002632 (0.05/19) | yes | 1,973 / 2,397 | 1.52e-10 | 0.0025 (0.05/20) | yes | yes |
| 0.006\|single\|S\|C1-C5 | 0.6 % | single | C1 vs C5 | superiority | 668 / 1,232 | 1.03e-38 | 0.001923 (0.05/26) | yes | 718 / 1,284 | 4.99e-37 | 0.002 (0.05/25) | yes | yes |
| 0.006\|single\|S\|C4-C3 | 0.6 % | single | C4 vs C3 | superiority | 1,345 / 2,277 | 1.44e-54 | 0.001667 (0.05/30) | yes | 1,392 / 2,277 | 1.07e-48 | 0.001667 (0.05/30) | yes | yes |
| 0.006\|single\|N\|C1-C2 | 0.6 % | single | C1 vs C2 | non-inferiority, margin 0.0005102 | 661 / 1,252 | 7.27e-119 | 0.00125 (0.05/40) | yes | 708 / 1,293 | 4.93e-114 | 0.001282 (0.05/39) | yes | yes |
| 0.006\|single\|N\|C5-C2 | 0.6 % | single | C5 vs C2 | non-inferiority, margin 0.0005102 | 13 / 40 | 1.56e-115 | 0.001316 (0.05/38) | yes | 21 / 40 | 1.19e-112 | 0.001351 (0.05/37) | yes | yes |
| 0.006\|batched\|S\|C1-C2 | 0.6 % | batched | C1 vs C2 | superiority | 660 / 1,245 | 1.84e-41 | 0.001852 (0.05/27) | yes | 703 / 1,297 | 1.04e-40 | 0.001786 (0.05/28) | yes | yes |
| 0.006\|batched\|S\|C5-C2 | 0.6 % | batched | C5 vs C2 | superiority | 13 / 40 | 0.000269 | 0.003846 (0.05/13) | yes | 21 / 41 | 0.0151 | none | no | no |
| 0.006\|batched\|S\|C4-C2 | 0.6 % | batched | C4 vs C2 | superiority | 2,413 / 1,978 | 5.57e-11 | 0.002778 (0.05/18) | yes | 2,418 / 2,024 | 3.63e-09 | 0.003571 (0.05/14) | yes | yes |
| 0.006\|batched\|S\|C3-C2 | 0.6 % | batched | C3 vs C2 | superiority | 3,884 / 2,517 | 7.55e-66 | 0.001471 (0.05/34) | yes | 3,896 / 2,617 | 7.78e-57 | 0.001515 (0.05/33) | yes | yes |
| 0.006\|batched\|S\|C1-C4 | 0.6 % | batched | C1 vs C4 | superiority | 1,058 / 2,078 | 2.63e-75 | 0.001389 (0.05/36) | yes | 1,093 / 2,081 | 7.8e-70 | 0.001429 (0.05/35) | yes | yes |
| 0.006\|batched\|S\|C5-C4 | 0.6 % | batched | C5 vs C4 | superiority | 1,938 / 2,400 | 2.46e-12 | 0.0025 (0.05/20) | yes | 1,983 / 2,397 | 4.24e-10 | 0.002632 (0.05/19) | yes | yes |
| 0.006\|batched\|S\|C1-C5 | 0.6 % | batched | C1 vs C5 | superiority | 667 / 1,225 | 4.48e-38 | 0.002 (0.05/25) | yes | 713 / 1,287 | 4.36e-38 | 0.001923 (0.05/26) | yes | yes |
| 0.006\|batched\|S\|C4-C3 | 0.6 % | batched | C4 vs C3 | superiority | 1,345 / 2,277 | 1.44e-54 | 0.001724 (0.05/29) | yes | 1,392 / 2,277 | 1.07e-48 | 0.001724 (0.05/29) | yes | yes |
| 0.006\|batched\|N\|C1-C2 | 0.6 % | batched | C1 vs C2 | non-inferiority, margin 0.0005102 | 660 / 1,245 | 4.46e-118 | 0.001282 (0.05/39) | yes | 703 / 1,297 | 9.72e-116 | 0.00125 (0.05/40) | yes | yes |
| 0.006\|batched\|N\|C5-C2 | 0.6 % | batched | C5 vs C2 | non-inferiority, margin 0.0005102 | 13 / 40 | 1.57e-115 | 0.001351 (0.05/37) | yes | 21 / 41 | 8.45e-113 | 0.001316 (0.05/38) | yes | yes |
| 0.0035\|single\|S\|C1-C2 | 0.35 % | single | C1 vs C2 | superiority | 1,405 / 1,305 | 0.0572 | none | no | 1,483 / 1,205 | 8.93e-08 | 0.004167 (0.05/12) | yes | no |
| 0.0035\|single\|S\|C5-C2 | 0.35 % | single | C5 vs C2 | superiority | 0 / 33 | 2.33e-10 | 0.003333 (0.05/15) | yes | 0 / 29 | 3.73e-09 | 0.003846 (0.05/13) | yes | yes |
| 0.0035\|single\|S\|C4-C2 | 0.35 % | single | C4 vs C2 | superiority | 1,046 / 950 | 0.0334 | none | no | 1,091 / 908 | 4.63e-05 | 0.007143 (0.05/7) | yes | no |
| 0.0035\|single\|S\|C3-C2 | 0.35 % | single | C3 vs C2 | superiority | 1,824 / 1,321 | 2.95e-19 | 0.002273 (0.05/22) | yes | 1,936 / 1,233 | 5.67e-36 | 0.002174 (0.05/23) | yes | yes |
| 0.0035\|single\|S\|C1-C4 | 0.35 % | single | C1 vs C4 | superiority | 828 / 824 | 0.941 | none | no | 905 / 810 | 0.0232 | none | no | yes |
| 0.0035\|single\|S\|C5-C4 | 0.35 % | single | C5 vs C4 | superiority | 917 / 1,046 | 0.00385 | 0.004167 (0.05/12) | yes | 879 / 1,091 | 1.95e-06 | 0.005556 (0.05/9) | yes | yes |
| 0.0035\|single\|S\|C1-C5 | 0.35 % | single | C1 vs C5 | superiority | 1,406 / 1,273 | 0.0108 | none | no | 1,486 / 1,179 | 2.96e-09 | 0.003125 (0.05/16) | yes | no |
| 0.0035\|single\|S\|C4-C3 | 0.35 % | single | C4 vs C3 | superiority | 817 / 1,224 | 1.94e-19 | 0.002083 (0.05/24) | yes | 818 / 1,338 | 2.83e-29 | 0.002273 (0.05/22) | yes | yes |
| 0.0035\|single\|N\|C1-C2 | 0.35 % | single | C1 vs C2 | non-inferiority, margin 7.5e-6 | 1,405 / 1,305 | 0.00826 | none | no | 1,483 / 1,205 | 1 | none | no | yes |
| 0.0035\|single\|N\|C5-C2 | 0.35 % | single | C5 vs C2 | non-inferiority, margin 7.5e-6 | 0 / 33 | 5.84e-56 | 0.001613 (0.05/31) | yes | 0 / 29 | 2.22e-55 | 0.001613 (0.05/31) | yes | yes |
| 0.0035\|batched\|S\|C1-C2 | 0.35 % | batched | C1 vs C2 | superiority | 1,397 / 1,309 | 0.0944 | none | no | 1,484 / 1,208 | 1.13e-07 | 0.004545 (0.05/11) | yes | no |
| 0.0035\|batched\|S\|C5-C2 | 0.35 % | batched | C5 vs C2 | superiority | 0 / 35 | 5.82e-11 | 0.002941 (0.05/17) | yes | 0 / 30 | 1.86e-09 | 0.002941 (0.05/17) | yes | yes |
| 0.0035\|batched\|S\|C4-C2 | 0.35 % | batched | C4 vs C2 | superiority | 1,045 / 954 | 0.0441 | none | no | 1,093 / 908 | 3.85e-05 | 0.00625 (0.05/8) | yes | no |
| 0.0035\|batched\|S\|C3-C2 | 0.35 % | batched | C3 vs C2 | superiority | 1,822 / 1,324 | 6.73e-19 | 0.002381 (0.05/21) | yes | 1,941 / 1,236 | 4.41e-36 | 0.002083 (0.05/24) | yes | yes |
| 0.0035\|batched\|S\|C1-C4 | 0.35 % | batched | C1 vs C4 | superiority | 822 / 825 | 0.961 | none | no | 901 / 810 | 0.0295 | none | no | yes |
| 0.0035\|batched\|S\|C5-C4 | 0.35 % | batched | C5 vs C4 | superiority | 919 / 1,045 | 0.00478 | 0.005 (0.05/10) | yes | 878 / 1,093 | 1.4e-06 | 0.005 (0.05/10) | yes | yes |
| 0.0035\|batched\|S\|C1-C5 | 0.35 % | batched | C1 vs C5 | superiority | 1,398 / 1,275 | 0.0183 | none | no | 1,487 / 1,181 | 3.4e-09 | 0.003333 (0.05/15) | yes | no |
| 0.0035\|batched\|S\|C4-C3 | 0.35 % | batched | C4 vs C3 | superiority | 817 / 1,224 | 1.94e-19 | 0.002174 (0.05/23) | yes | 818 / 1,338 | 2.83e-29 | 0.002381 (0.05/21) | yes | yes |
| 0.0035\|batched\|N\|C1-C2 | 0.35 % | batched | C1 vs C2 | non-inferiority, margin 7.5e-6 | 1,397 / 1,309 | 0.00429 | 0.004545 (0.05/11) | yes | 1,484 / 1,208 | 1 | none | no | no |
| 0.0035\|batched\|N\|C5-C2 | 0.35 % | batched | C5 vs C2 | non-inferiority, margin 7.5e-6 | 0 / 35 | 2.96e-56 | 0.001563 (0.05/32) | yes | 0 / 30 | 1.58e-55 | 0.001563 (0.05/32) | yes | yes |

- Rejected at the Holm level 0.05: 31 of 40 in the original, 34 of 40 in the re-test. The decision is the same for 31 hypotheses and different for 9.
- Hypotheses whose outcome changed (original -> re-test, with the p-values):
  - `0.006|single|S|C5-C2`: rejected (p = 0.000269) -> not rejected (p = 0.0204)
  - `0.006|batched|S|C5-C2`: rejected (p = 0.000269) -> not rejected (p = 0.0151)
  - `0.0035|single|S|C1-C2`: not rejected (p = 0.0572) -> rejected (p = 8.93e-08)
  - `0.0035|single|S|C4-C2`: not rejected (p = 0.0334) -> rejected (p = 4.63e-05)
  - `0.0035|single|S|C1-C5`: not rejected (p = 0.0108) -> rejected (p = 2.96e-09)
  - `0.0035|batched|S|C1-C2`: not rejected (p = 0.0944) -> rejected (p = 1.13e-07)
  - `0.0035|batched|S|C4-C2`: not rejected (p = 0.0441) -> rejected (p = 3.85e-05)
  - `0.0035|batched|S|C1-C5`: not rejected (p = 0.0183) -> rejected (p = 3.4e-09)
  - `0.0035|batched|N|C1-C2`: rejected (p = 0.00429) -> not rejected (p = 1)
- Where Holm stopped: in the original the 32nd smallest p-value (0.008263) was not below 0.05/9 = 0.005556; in the re-test the 35th smallest (0.01513) was not below 0.05/6 = 0.008333. Every smaller p-value was rejected in the same run.
- Not rejected in the original (9): `0.0035|single|S|C1-C2`; `0.0035|single|S|C4-C2`; `0.0035|single|S|C1-C4`; `0.0035|single|S|C1-C5`; `0.0035|single|N|C1-C2`; `0.0035|batched|S|C1-C2`; `0.0035|batched|S|C4-C2`; `0.0035|batched|S|C1-C4`; `0.0035|batched|S|C1-C5`.
- Not rejected in the re-test (6): `0.006|single|S|C5-C2`; `0.006|batched|S|C5-C2`; `0.0035|single|S|C1-C4`; `0.0035|single|N|C1-C2`; `0.0035|batched|S|C1-C4`; `0.0035|batched|N|C1-C2`.
- The two re-test non-inferiority hypotheses for C1 against C2 at p = 0.35 % have p-value 1 because the estimated difference in failure probability (single 9.27e-06, batched 9.2e-06) is already above the margin 7.5e-6.
- 22 superiority hypotheses and 6 non-inferiority hypotheses are rejected in both runs; for all 22 superiority ones the sign of (X only - Y only) is the same in both runs, so no hypothesis was rejected in opposite directions.

### Failures per decoder

Failure counts out of the shots at each error rate (1,000,000 at p = 0.6 %, 30,000,000 at p = 0.35 %). Plain PyMatching (no correlated matching) is context only and does not depend on the mode; the files store one count per error rate.

| p | shots | mode | run | C1 chain | C2 Cascade alone | C3 corr. PM | C4 S -> M | C5 pair | plain PM (context) |
|---|---|---|---|---|---|---|---|---|---|
| 0.6 % | 1,000,000 | single | original | 4,372 | 4,963 | 6,320 | 5,388 | 4,936 | 21,061 |
| 0.6 % | 1,000,000 | single | re-test | 4,351 | 4,936 | 6,226 | 5,341 | 4,917 | 21,250 |
| 0.6 % | 1,000,000 | batched | original | 4,368 | 4,953 | 6,320 | 5,388 | 4,926 | 21,061 |
| 0.6 % | 1,000,000 | batched | re-test | 4,353 | 4,947 | 6,226 | 5,341 | 4,927 | 21,250 |
| 0.35 % | 30,000,000 | single | original | 2,506 | 2,406 | 2,909 | 2,502 | 2,373 | 18,936 |
| 0.35 % | 30,000,000 | single | re-test | 2,632 | 2,354 | 3,057 | 2,537 | 2,325 | 19,015 |
| 0.35 % | 30,000,000 | batched | original | 2,499 | 2,411 | 2,909 | 2,502 | 2,376 | 18,936 |
| 0.35 % | 30,000,000 | batched | re-test | 2,628 | 2,352 | 3,057 | 2,537 | 2,322 | 19,015 |

C3 and C4 give identical counts in single and batched mode in both runs (they have no Cascade stage). Share of shots the pair routed to Cascade: p = 0.6 % 0.0996 original, 0.0998 re-test; p = 0.35 % 0.0100 original, 0.0100 re-test (the paper's table rounds these to 10 % and 1 %).

### The pair against Cascade alone

C5 is the pair, C2 is Cascade alone. "Only pair failed" and "only Cascade alone failed" are the discordant shots of the exact McNemar test (kind S, two-sided). The non-inferiority test (kind N) has margin 0.0005102 at p = 0.6 % and 7.5e-6 at p = 0.35 %. The difference is the pair's failure probability minus Cascade alone's, with the unadjusted Newcombe 95 % interval stored in the files (`lo95`, `hi95`).

| p | mode | run | pair failures | Cascade-alone failures | only pair failed | only Cascade alone failed | exact McNemar p (S) | rejected (S) | non-inferiority p (N) | rejected (N) | difference, pair minus Cascade alone | Newcombe 95 % interval |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.6 % | single | original | 4,936 | 4,963 | 13 | 40 | 0.000269 | yes | 1.56e-115 | yes | -2.7e-05 | (-4.19e-05, -1.23e-05) |
| 0.6 % | single | re-test | 4,917 | 4,936 | 21 | 40 | 0.0204 | no | 1.19e-112 | yes | -1.9e-05 | (-3.48e-05, -3.29e-06) |
| 0.6 % | batched | original | 4,926 | 4,953 | 13 | 40 | 0.000269 | yes | 1.57e-115 | yes | -2.7e-05 | (-4.19e-05, -1.23e-05) |
| 0.6 % | batched | re-test | 4,927 | 4,947 | 21 | 41 | 0.0151 | no | 8.45e-113 | yes | -2e-05 | (-3.6e-05, -4.17e-06) |
| 0.35 % | single | original | 2,373 | 2,406 | 0 | 33 | 2.33e-10 | yes | 5.84e-56 | yes | -1.1e-06 | (-1.5e-06, -7.11e-07) |
| 0.35 % | single | re-test | 2,325 | 2,354 | 0 | 29 | 3.73e-09 | yes | 2.22e-55 | yes | -9.67e-07 | (-1.35e-06, -5.99e-07) |
| 0.35 % | batched | original | 2,376 | 2,411 | 0 | 35 | 5.82e-11 | yes | 2.96e-56 | yes | -1.17e-06 | (-1.58e-06, -7.67e-07) |
| 0.35 % | batched | re-test | 2,322 | 2,352 | 0 | 30 | 1.86e-09 | yes | 1.58e-55 | yes | -1e-06 | (-1.39e-06, -6.27e-07) |

The exact McNemar p-values of the eight pair-against-Cascade-alone cells (S tests) were recomputed here with integer binomial sums and agree with the files. The lock's planned power for this comparison (`planned_power` entry `S:C5-C2` of `locked_d13.json`, also quoted in `RETEST.md`) is 0.34225 at p = 0.6 % and 0.999 at p = 0.35 %.

### Labels and verdicts

Labels for all 32 (error rate, mode, comparison) cells, from the key `labels` object of each file. "Same label" compares the two runs. The last column applies the frozen repeat rule to the cells where the original label was "better"; for the other original labels the rules fix no repeat status, so only the labels are compared.

| id | comparison | p | mode | original label | re-test label | same label | original "better" repeated? |
|---|---|---|---|---|---|---|---|
| 0.006\|single\|C1-C2 | C1 vs C2 | 0.6 % | single | better | better | yes | repeated |
| 0.006\|single\|C5-C2 | C5 vs C2 | 0.6 % | single | better | non-inferior | no | not repeated (non-inferior is not better) |
| 0.006\|single\|C4-C2 | C4 vs C2 | 0.6 % | single | worse | worse | yes | n/a (labels compared only) |
| 0.006\|single\|C3-C2 | C3 vs C2 | 0.6 % | single | worse | worse | yes | n/a (labels compared only) |
| 0.006\|single\|C1-C4 | C1 vs C4 | 0.6 % | single | better | better | yes | repeated |
| 0.006\|single\|C5-C4 | C5 vs C4 | 0.6 % | single | better | better | yes | repeated |
| 0.006\|single\|C1-C5 | C1 vs C5 | 0.6 % | single | better | better | yes | repeated |
| 0.006\|single\|C4-C3 | C4 vs C3 | 0.6 % | single | better | better | yes | repeated |
| 0.006\|batched\|C1-C2 | C1 vs C2 | 0.6 % | batched | better | better | yes | repeated |
| 0.006\|batched\|C5-C2 | C5 vs C2 | 0.6 % | batched | better | non-inferior | no | not repeated (non-inferior is not better) |
| 0.006\|batched\|C4-C2 | C4 vs C2 | 0.6 % | batched | worse | worse | yes | n/a (labels compared only) |
| 0.006\|batched\|C3-C2 | C3 vs C2 | 0.6 % | batched | worse | worse | yes | n/a (labels compared only) |
| 0.006\|batched\|C1-C4 | C1 vs C4 | 0.6 % | batched | better | better | yes | repeated |
| 0.006\|batched\|C5-C4 | C5 vs C4 | 0.6 % | batched | better | better | yes | repeated |
| 0.006\|batched\|C1-C5 | C1 vs C5 | 0.6 % | batched | better | better | yes | repeated |
| 0.006\|batched\|C4-C3 | C4 vs C3 | 0.6 % | batched | better | better | yes | repeated |
| 0.0035\|single\|C1-C2 | C1 vs C2 | 0.35 % | single | level | worse | no | n/a (labels compared only) |
| 0.0035\|single\|C5-C2 | C5 vs C2 | 0.35 % | single | better | better | yes | repeated |
| 0.0035\|single\|C4-C2 | C4 vs C2 | 0.35 % | single | level | worse | no | n/a (labels compared only) |
| 0.0035\|single\|C3-C2 | C3 vs C2 | 0.35 % | single | worse | worse | yes | n/a (labels compared only) |
| 0.0035\|single\|C1-C4 | C1 vs C4 | 0.35 % | single | level | level | yes | n/a (labels compared only) |
| 0.0035\|single\|C5-C4 | C5 vs C4 | 0.35 % | single | better | better | yes | repeated |
| 0.0035\|single\|C1-C5 | C1 vs C5 | 0.35 % | single | level | worse | no | n/a (labels compared only) |
| 0.0035\|single\|C4-C3 | C4 vs C3 | 0.35 % | single | better | better | yes | repeated |
| 0.0035\|batched\|C1-C2 | C1 vs C2 | 0.35 % | batched | non-inferior | worse | no | n/a (labels compared only) |
| 0.0035\|batched\|C5-C2 | C5 vs C2 | 0.35 % | batched | better | better | yes | repeated |
| 0.0035\|batched\|C4-C2 | C4 vs C2 | 0.35 % | batched | level | worse | no | n/a (labels compared only) |
| 0.0035\|batched\|C3-C2 | C3 vs C2 | 0.35 % | batched | worse | worse | yes | n/a (labels compared only) |
| 0.0035\|batched\|C1-C4 | C1 vs C4 | 0.35 % | batched | level | level | yes | n/a (labels compared only) |
| 0.0035\|batched\|C5-C4 | C5 vs C4 | 0.35 % | batched | better | better | yes | repeated |
| 0.0035\|batched\|C1-C5 | C1 vs C5 | 0.35 % | batched | level | worse | no | n/a (labels compared only) |
| 0.0035\|batched\|C4-C3 | C4 vs C3 | 0.35 % | batched | better | better | yes | repeated |

Labels changed in 8 of 32 cells and are the same in 24:
- `0.006|single|C5-C2`: better -> non-inferior
- `0.006|batched|C5-C2`: better -> non-inferior
- `0.0035|single|C1-C2`: level -> worse
- `0.0035|single|C4-C2`: level -> worse
- `0.0035|single|C1-C5`: level -> worse
- `0.0035|batched|C1-C2`: non-inferior -> worse
- `0.0035|batched|C4-C2`: level -> worse
- `0.0035|batched|C1-C5`: level -> worse

C3 against C2 is "worse" in all four cells in both runs; the four cells of C4 against C3 are "better" in both runs.

The paper's claim for the pair, cell by cell:

| cell | paper | original label (pair-only / Cascade-only failures, p-value) | re-test label (pair-only / Cascade-only failures, p-value) | repeats the paper's "better"? |
|---|---|---|---|---|
| 0.6 %, single | better | better (13 / 40, p = 0.000269) | non-inferior (21 / 40, p = 0.0204) | no |
| 0.6 %, batched | better | better (13 / 40, p = 0.000269) | non-inferior (21 / 41, p = 0.0151) | no |
| 0.35 %, single | better | better (0 / 33, p = 2.33e-10) | better (0 / 29, p = 3.73e-09) | yes |
| 0.35 %, batched | better | better (0 / 35, p = 5.82e-11) | better (0 / 30, p = 1.86e-09) | yes |

The verdicts of `sv_route.verdicts` are read from `SURFACE_RESULT_draft.md.json` and `SURFACE_RESULT_retest.md.json`. A chain gets "faster within the predefined accuracy tolerance" at one error rate when, in both modes, its label against C2 is better or non-inferior, it is faster than C2 and the bitwise check passes; it gets "faster and more accurate" when in addition both labels are "better".

| p | chain | original labels (single / batched) | original verdict | re-test labels (single / batched) | re-test verdict | same verdict |
|---|---|---|---|---|---|---|
| 0.6 % | C1 | better / better | faster and more accurate | better / better | faster and more accurate | yes |
| 0.6 % | C5 | better / better | faster and more accurate | non-inferior / non-inferior | faster within the predefined accuracy tolerance | no |
| 0.35 % | C1 | level / non-inferior | no claim | worse / worse | no claim | yes |
| 0.35 % | C5 | better / better | faster and more accurate | better / better | faster and more accurate | yes |

Headline, the conditions for the d = 17 decision, and the per-error-rate verdicts of C5:

| item | original | re-test | same |
|---|---|---|---|
| headline chains (faster within the accuracy tolerance at both p) | [C5] | [C5] | yes |
| headline chains that are also more accurate (`headline_more_accurate`: faster and more accurate at both p) | [C5] | none | no |
| C5 faster and more accurate at p = 0.6 % | true | false | no |
| C5 faster and more accurate at p = 0.35 % | true | true | yes |
| C5 faster within the accuracy tolerance at p = 0.6 % | true | true | yes |
| C5 faster within the accuracy tolerance at p = 0.35 % | true | true | yes |
| P1 (C2 better than C3 at both p in both modes) | true | true | yes |
| P2 chains | [C5] | [C5] | yes |
| d = 17 decision | propose | propose | yes |

The headline chain is still C5 in the re-test, and P1, P2 and the d = 17 decision are unchanged. What changed is the accuracy claim for the pair: at p = 0.35 % both labels are still "better", so C5 is still "faster and more accurate"; at p = 0.6 % both labels are now "non-inferior", so the verdict is only "faster within the predefined accuracy tolerance (up to 10 % more failures than Cascade alone)". `headline_more_accurate` therefore goes from [C5] to none. For C1 the verdict text is "no claim" at p = 0.35 % in both runs, but the labels behind it moved from level / non-inferior to worse / worse. The "faster", same-job and bitwise flags are true in all eight chain-by-error-rate-by-mode rows of both runs, and the speed numbers are identical in the two runs (inherited; see the timing subsection). Both analyses output "propose" for d = 17; `RETEST.md` adds the remark "the original deferred that decision".

### The text's percentages, and which sentences still hold

The percentage in the paper's text is 1 - (pair failures) / (Cascade-alone failures), per cell. Recomputed from the counts of each run, with no verdict attached:

| p | mode | original: pair / Cascade-alone failures | original, fewer pair failures | paper text | re-test: pair / Cascade-alone failures | re-test, fewer pair failures |
|---|---|---|---|---|---|---|
| 0.6 % | single | 4,936 / 4,963 | 0.544 % | 0.54 % | 4,917 / 4,936 | 0.385 % |
| 0.6 % | batched | 4,926 / 4,953 | 0.545 % | 0.54 % | 4,927 / 4,947 | 0.404 % |
| 0.35 % | single | 2,373 / 2,406 | 1.372 % | 1.4 % | 2,325 / 2,354 | 1.232 % |
| 0.35 % | batched | 2,376 / 2,411 | 1.452 % | 1.5 % | 2,322 / 2,352 | 1.276 % |

Which sentences of the text still hold under the frozen rules:

- "The pair failed less often than Cascade alone at both p in both modes" is not repeated. Only two of the four cells are "better" again (p = 0.35 %, single and batched). The two cells at p = 0.6 % are "non-inferior": 21 pair-only against 40 Cascade-only failures in single mode (p = 0.0204) and 21 against 41 in batched mode (p = 0.0151), neither rejected under Holm. In raw counts the pair did have fewer failures than Cascade alone in all four cells (19 fewer at p = 0.6 % single, 20 batched; 29 fewer at p = 0.35 % single, 30 batched), but the frozen rule counts only the label "better" as a repeat.
- The percentages are recomputed above and carry no verdict. The paper's "0.54 %" matches the original single-mode value (0.544 %); the original batched value is 0.5451 %, which would round to 0.55 %. The paper's 1.4 % and 1.5 % match the original 1.372 % and 1.452 %.
- "Faster and more accurate" for the pair holds at p = 0.35 % only. At p = 0.6 % the re-test supports "faster within the predefined accuracy tolerance".
- The speed statements (faster on average, slower in the tail) rest on inherited timings and are not re-measured; the speed conditions of the verdicts hold in every cell because they read the same inherited e7 rows.

### Timing (inherited, not re-measured)

All timing numbers below are inherited from the registered timing session (jobs e7 and e7r, kind-12 speed shots; no timed input is a confirmation shot). They are copied from `RETEST.md` section 3.3 ("What the paper reports") and are the same numbers the paper gives.

| p | Cascade alone, single (us) | pair, single (us) | Cascade alone, batched (us per round) | pair, batched (us per round) |
|---|---|---|---|---|
| 0.6 % | 397.0 | 107.4 | 10.48 | 2.34 |
| 0.35 % | 397.1 | 69.4 | 10.48 | 1.37 |

Text of the paper (inherited, as `RETEST.md` quotes it): the pair was faster on average, 3.70 and 5.72 times per single shot (p = 0.6 % and 0.35 %) and 4.49 and 7.67 times in batches, and slower in the tail, with a 99.9th percentile of about 490 us against 411 and 417 us for Cascade alone (the order matches p = 0.6 % and 0.35 % in the e7 rows below). The 10.48 us per round for Cascade alone in batches is the e7r value (clean repeat of eight flagged timings), a display value for the paper table only that no verdict reads; the verdict timings are e7's and are kept apart.

The two result files carry the verdict timings of the e7 rows (slowest of three processes against Cascade alone from the same job). They are identical in the original and the re-test file (checked field by field: speed, Cascade-alone time, 99.9th percentile, flags), hence inherited. In these files the batched times are microseconds per shot; the paper table gives microseconds per round.

| p | chain | mode | pipeline (us per shot) | Cascade alone, same job (us per shot) | pipeline 99.9th percentile (us) | Cascade alone 99.9th percentile (us) |
|---|---|---|---|---|---|---|
| 0.6 % | C1 | single | 228.99 | 397.03 | 869.99 | 411.44 |
| 0.6 % | C1 | batched | 72.831 | 136.04 | - | - |
| 0.6 % | C5 | single | 107.36 | 397.03 | 490.16 | 411.44 |
| 0.6 % | C5 | batched | 30.370 | 136.04 | - | - |
| 0.35 % | C1 | single | 312.72 | 397.09 | 694.92 | 417.29 |
| 0.35 % | C1 | batched | 100.79 | 136.00 | - | - |
| 0.35 % | C5 | single | 69.424 | 397.09 | 490.28 | 417.29 |
| 0.35 % | C5 | batched | 17.751 | 136.00 | - | - |

For orientation (arithmetic on the table, not a value stored in the files): the batched pipeline times divided by 13 rounds are 2.336 (p = 0.6 %) and 1.365 us per round (p = 0.35 %), in line with the paper table's 2.34 and 1.37.

## 4. Output-head comparison (A1)

This section repeats the paper's comparison of read-out heads (A1) on new random shots. The head is the last part of a small network, the part that turns its internal state into a decoding decision. On the surface code (distance d = 9, physical error rate p = 0.6 %) five arms, a to e, differ in the head (line vote, centre line alone, pooled head) and in whether a training-only line loss is added. On the BB72 code (p = 0.3 %) four arms, f to i, differ in the same way (joint score or pooled independent outputs, with or without a translate loss). Each arm was trained twice from different random starts (seeds 1 and 2). Nothing is retrained: the re-test reads the same 23 frozen networks on 1,000,000 new surface shots and 500,000 new BB72 shots, with the original thresholds and fallbacks. "Alone" means the network decides every shot (q = 0). "Pair" means the network keeps the shots it is confident about and hands the fraction q of its least-confident shots to a fallback decoder: correlated PyMatching on the surface code (registered q = 1 %), Relay-BP on BB72 (registered q = 2 %). Each comparison X vs Y is the difference in failures per million shots, X minus Y, with a 95 % Newcombe interval for each seed. Under the frozen rule a comparison is "better" or "worse" only if the Holm-adjusted p is below 0.05 and both seeds reject in the same direction; otherwise it is "inconclusive". A "better" or "worse" repeats only by the same verdict. "Inconclusive" does not repeat "better" or "worse", and it is not equality: it says the test could not tell, so a repeated "inconclusive" means only that the test again could not tell. A new "better" or "worse" where the original was inconclusive is a new result, reported with the original as an earlier registered test, and a flip from better to worse is reported with both tests' numbers. The verdicts at q = 0 and at the registered q are reported separately. Of the 24 comparison-and-q pairs, 23 repeat (surface 14 of 14, BB72 9 of 10). The one that does not is BB72 U1 (g vs f) at q = 0, inconclusive originally and now a new "worse".

### Verdicts

How to read the tables. X - Y is in failures per million shots; a positive value means X fails more often than Y. The ids follow the paper: R = read-out head changed (the paper's "head"), U = the training-only loss changed (supervision), W = whole design changed (the paper's "both"). Adjusted p is the Holm adjustment of the paper, one family per code (14 surface tests, 10 BB72 tests, q = 0 and the registered q together); "<1e-300" means the p-value is below the smallest number the analysis reports. On BB72 the per-million values are twice the failure counts (500,000 shots).

Surface code (d = 9, p = 0.6 %, 1,000,000 shots, registered q = 1 %):

| id | X vs Y (letters; A1 names) | q (%) | original seed 1, X - Y [95 %] | original seed 2, X - Y [95 %] | re-test seed 1, X - Y [95 %] | re-test seed 2, X - Y [95 %] | adj. p original | adj. p re-test | verdict original | verdict re-test | repeats |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | d vs b (S-P vs S-V) | 0 | +468,208 [+467,209, +469,206] | +2,869 [+2,693, +3,046] | +467,584 [+466,585, +468,582] | +2,862 [+2,686, +3,039] | 3.5e-225 | 3.1e-225 | worse | worse | yes |
| R1 | d vs b (S-P vs S-V) | 1 | +464,169 [+463,172, +465,165] | +1,948 [+1,800, +2,097] | +463,573 [+462,576, +464,569] | +2,012 [+1,863, +2,162] | 1.8e-147 | 7.8e-156 | worse | worse | yes |
| R2 | c vs b (S-L vs S-V) | 0 | +6,450 [+6,235, +6,667] | +4,178 [+3,981, +4,376] | +6,137 [+5,922, +6,354] | +4,266 [+4,069, +4,464] | <1e-300 | <1e-300 | worse | worse | yes |
| R2 | c vs b (S-L vs S-V) | 1 | +4,686 [+4,499, +4,874] | +3,007 [+2,839, +3,176] | +4,549 [+4,362, +4,737] | +3,137 [+2,968, +3,307] | 1.1e-275 | 4.3e-297 | worse | worse | yes |
| R3 | e vs a (S-PA vs S-VA) | 0 | -335 [-532, -138] | +1,744 [+1,548, +1,940] | -94 [-291, +103] | +1,878 [+1,680, +2,077] | 1 | 1 | inconclusive | inconclusive | yes |
| R3 | e vs a (S-PA vs S-VA) | 1 | -258 [-428, -88] | +1,393 [+1,224, +1,562] | -124 [-293, +45] | +1,465 [+1,294, +1,636] | 1 | 1 | inconclusive | inconclusive | yes |
| U1 | b vs a (S-V vs S-VA) | 0 | -4,614 [-4,808, -4,421] | -3,094 [-3,271, -2,918] | -4,357 [-4,551, -4,164] | -3,159 [-3,339, -2,980] | 3.6e-264 | 2.2e-264 | better | better | yes |
| U1 | b vs a (S-V vs S-VA) | 1 | -3,297 [-3,465, -3,130] | -2,148 [-2,301, -1,996] | -3,094 [-3,262, -2,927] | -2,283 [-2,437, -2,130] | 6.7e-171 | 3.5e-190 | better | better | yes |
| U2 | e vs d (S-PA vs S-P) | 0 | -463,929 [-464,935, -462,923] | +1,969 [+1,771, +2,167] | -463,321 [-464,327, -462,315] | +2,175 [+1,976, +2,375] | 1 | 1 | inconclusive | inconclusive | yes |
| U2 | e vs d (S-PA vs S-P) | 1 | -461,130 [-462,132, -460,128] | +1,593 [+1,422, +1,764] | -460,603 [-461,604, -459,601] | +1,736 [+1,564, +1,909] | 1 | 1 | inconclusive | inconclusive | yes |
| W1 | c vs a (S-L vs S-VA) | 0 | +1,836 [+1,611, +2,062] | +1,084 [+883, +1,285] | +1,780 [+1,556, +2,004] | +1,107 [+904, +1,310] | 3.3e-25 | 9.6e-26 | worse | worse | yes |
| W1 | c vs a (S-L vs S-VA) | 1 | +1,389 [+1,194, +1,584] | +859 [+687, +1,032] | +1,455 [+1,260, +1,650] | +854 [+681, +1,027] | 1.2e-21 | 3.0e-21 | worse | worse | yes |
| W2 | d vs a (S-P vs S-VA) | 0 | +463,594 [+462,587, +464,600] | -225 [-414, -36] | +463,227 [+462,220, +464,233] | -297 [-486, -108] | 1 | 1 | inconclusive | inconclusive | yes |
| W2 | d vs a (S-P vs S-VA) | 1 | +460,872 [+459,870, +461,874] | -200 [-361, -39] | +460,479 [+459,477, +461,481] | -271 [-431, -111] | 1 | 1 | inconclusive | inconclusive | yes |

BB72 (p = 0.3 %, 500,000 shots, registered q = 2 %):

| id | X vs Y (letters; A1 names) | q (%) | original seed 1, X - Y [95 %] | original seed 2, X - Y [95 %] | re-test seed 1, X - Y [95 %] | re-test seed 2, X - Y [95 %] | adj. p original | adj. p re-test | verdict original | verdict re-test | repeats |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | h vs g (B-P vs B-J) | 0 | +27,002 [+26,547, +27,463] | +28,250 [+27,787, +28,719] | +26,774 [+26,322, +27,232] | +27,748 [+27,289, +28,213] | <1e-300 | <1e-300 | worse | worse | yes |
| R1 | h vs g (B-P vs B-J) | 2 | +14,214 [+13,874, +14,560] | +15,382 [+15,031, +15,739] | +14,070 [+13,731, +14,415] | +15,162 [+14,813, +15,517] | <1e-300 | <1e-300 | worse | worse | yes |
| R2 | i vs f (B-PA vs B-JT) | 0 | +21,102 [+20,698, +21,511] | +19,352 [+18,965, +19,744] | +21,022 [+20,621, +21,429] | +19,126 [+18,741, +19,517] | <1e-300 | <1e-300 | worse | worse | yes |
| R2 | i vs f (B-PA vs B-JT) | 2 | +8,548 [+8,277, +8,824] | +6,740 [+6,492, +6,993] | +8,258 [+7,991, +8,531] | +6,636 [+6,391, +6,886] | <1e-300 | <1e-300 | worse | worse | yes |
| U1 | g vs f (B-J vs B-JT) | 0 | +2 [-175, +179] | +246 [+71, +421] | +208 [+38, +379] | +240 [+69, +411] | 1 | 0.036 | inconclusive | worse | no: new result |
| U1 | g vs f (B-J vs B-JT) | 2 | +42 [-47, +131] | -44 [-133, +44] | -36 [-120, +48] | -2 [-90, +86] | 1 | 1 | inconclusive | inconclusive | yes |
| U2 | i vs h (B-PA vs B-P) | 0 | -5,902 [-6,306, -5,499] | -9,144 [-9,545, -8,745] | -5,960 [-6,360, -5,561] | -8,862 [-9,260, -8,466] | 1.3e-182 | 5.9e-190 | better | better | yes |
| U2 | i vs h (B-PA vs B-P) | 2 | -5,708 [-6,085, -5,333] | -8,598 [-8,971, -8,228] | -5,776 [-6,148, -5,406] | -8,524 [-8,898, -8,152] | 2.3e-196 | 6.4e-207 | better | better | yes |
| W1 | h vs f (B-P vs B-JT) | 0 | +27,004 [+26,550, +27,464] | +28,496 [+28,032, +28,966] | +26,982 [+26,529, +27,441] | +27,988 [+27,527, +28,455] | <1e-300 | <1e-300 | worse | worse | yes |
| W1 | h vs f (B-P vs B-JT) | 2 | +14,256 [+13,916, +14,602] | +15,338 [+14,987, +15,695] | +14,034 [+13,696, +14,378] | +15,160 [+14,811, +15,515] | <1e-300 | <1e-300 | worse | worse | yes |

Source check. All 24 original rows agree between RETEST.md (section 3.4, "What the paper reports") and the original `comparisons.csv` files: both seeds' differences and intervals, the adjusted p and the verdict. Arm and name pairs (a S-VA, b S-V, c S-L, d S-P, e S-PA; f B-JT, g B-J, h B-P, i B-PA) also agree. No disagreement.

Reading the verdict tables.

- Surface, 14 of 14 repeat: R1 (d vs b) worse, R2 (c vs b) worse, R3 (e vs a) inconclusive, U1 (b vs a) better, U2 (e vs d) inconclusive, W1 (c vs a) worse, W2 (d vs a) inconclusive, each at q = 0 and at q = 1 %. The six inconclusive rows (R3, U2, W2 at both q) are inconclusive for the same reason in both tests: the two seeds point in opposite directions.
- BB72, 9 of 10 repeat: R1 (h vs g) worse, R2 (i vs f) worse, U2 (i vs h) better, W1 (h vs f) worse at both q, and U1 (g vs f) inconclusive at q = 2 %.
- The one non-repeat is BB72 U1 (g vs f, B-J against B-JT, i.e. without and with the translate loss) at q = 0. Original: seed 1 +2 [-175, +179], seed 2 +246 [+71, +421], adjusted p = 1, inconclusive. Re-test: seed 1 +208 [+38, +379], seed 2 +240 [+69, +411], adjusted p = 0.036, worse. Seed 2 barely moved; seed 1 moved from about 0 to +208. The size is 3.1 % and 3.6 % of B-JT's own failures (6,766 and 6,714 per million). The result is close to the threshold: the per-seed unadjusted p-values are 0.018 (seed 1) and 0.0062 (seed 2), and the Holm multiplier for this row is 2 (rank 9 of 10). It is a new result with the original named as an earlier registered test. At q = 2 % the verdict stays inconclusive (re-test seed 1 -36 [-120, +48], seed 2 -2 [-90, +86]), so for this comparison the verdict at q = 0 and at q = 2 % now differ, and a sentence that says "at both q" cannot be kept. The paper's sentence "For the joint class score a consistent benefit was not established, with differences alone of 2 and 246 per million in the two seeds (p-values 1 and 0.006)" (main.tex 1625-1626) does not repeat: on the fresh shots the differences alone are 208 and 240 per million (p-values 0.018 and 0.0062, adjusted 0.036), a consistent benefit of the translate supervision for the joint class score alone; at q = 2 % there is none. The sentence before it repeats: the supervision lowered the pooled head's failures alone by 5,960 and 8,862 per million (original 5,902 and 9,144; better, adjusted p 5.9e-190, original 1.3e-182).
- No row is "not confirmed" and no verdict flips sign.

### Table head

Failures per million test shots of every arm, alone (q = 0) and as a pair at the registered q, per seed, original against re-test. A smaller number is better. † marks arm d seed 1, the network that did not train (see the dagger row below); its numbers are reported as they come.

Surface code (failures per million shots; 1,000,000 shots, so equal to the failure count; pair at q = 1 %):

| arm | A1 name | seed | alone, original | alone, re-test | pair, original | pair, re-test |
|---|---|---|---|---|---|---|
| a | S-VA | 1 | 15,252 | 15,018 | 13,239 | 13,015 |
| a | S-VA | 2 | 13,975 | 13,981 | 12,217 | 12,202 |
| b | S-V | 1 | 10,638 | 10,661 | 9,942 | 9,921 |
| b | S-V | 2 | 10,881 | 10,822 | 10,069 | 9,919 |
| c | S-L | 1 | 17,088 | 16,798 | 14,628 | 14,470 |
| c | S-L | 2 | 15,059 | 15,088 | 13,076 | 13,056 |
| d | S-P | 1 | 478,846 † | 478,245 † | 474,111 | 473,494 |
| d | S-P | 2 | 13,750 | 13,684 | 12,017 | 11,931 |
| e | S-PA | 1 | 14,917 | 14,924 | 12,981 | 12,891 |
| e | S-PA | 2 | 15,719 | 15,859 | 13,610 | 13,667 |

BB72 (failures per million shots; 500,000 shots, so twice the failure count; pair at q = 2 %):

| arm | A1 name | seed | alone, original | alone, re-test | pair, original | pair, re-test |
|---|---|---|---|---|---|---|
| f | B-JT | 1 | 7,044 | 6,766 | 4,268 | 4,246 |
| f | B-JT | 2 | 6,864 | 6,714 | 4,306 | 4,256 |
| g | B-J | 1 | 7,046 | 6,974 | 4,310 | 4,210 |
| g | B-J | 2 | 7,110 | 6,954 | 4,262 | 4,254 |
| h | B-P | 1 | 34,048 | 33,748 | 18,524 | 18,280 |
| h | B-P | 2 | 35,360 | 34,702 | 19,644 | 19,416 |
| i | B-PA | 1 | 28,146 | 27,788 | 12,816 | 12,504 |
| i | B-PA | 2 | 26,216 | 25,840 | 11,046 | 10,892 |

Fallback decoders alone, on the same shots as the corresponding test set:

| fallback alone | original | re-test | plain meaning |
|---|---|---|---|
| correlated PM, surface, 1,000,000 shots | 13,080 failures (13,080 per million) | 12,989 failures (12,989 per million) | the decoder that takes the routed shots on the surface code |
| Relay-BP, BB72, 500,000 shots | 2,691 failures (5,382 per million) | 2,614 failures (5,228 per million) | the decoder that takes the routed shots on BB72 |

Source check. The original values of this table (all 36 alone and pair entries, correlated PM alone 13,080, Relay-BP alone 5,382 per million = 2,691 failures) agree with RETEST.md and the original `standalone.csv`, `pair_curves.csv` and `analysis.json`. No disagreement.

Re-test file checks (read from the files, not recomputed). The thresholds reproduce: `taus_check.json` reports ok with 70 of 70 surface rows and 56 of 56 BB72 rows equal, and the `tau` column of the new `pair_curves.csv` is identical to the original in all 126 rows. The batch-one re-reads of the first 2,000 shots show no differing decision for any of the 18 arm networks (`b1_differ` is 0 in every row). The test shots are new (1,000,000 and 500,000; the set digests differ from the original's). `train_checks.csv` and `sizes.csv` are byte-identical to the original in both codes.

### The paper's derived text numbers, re-checked

These numbers are computed from the tables above and carry no verdict of their own. Formulas are those of RETEST.md (section 3.4, "What the paper reports"). "Holds" means the paper's printed number or range is reproduced after the same rounding.

| text number | formula | original | re-test | holds / changes to |
|---|---|---|---|---|
| "3.8-5.0 times" (abstract, RQ2) | fail_x / fail_y of BB72 R1 (h vs g) and R2 (i vs f) at q = 0, both seeds | 3.82 to 4.97 (R1 4.83, 4.97; R2 4.00, 3.82) | 3.85 to 4.99 (R1 4.84, 4.99; R2 4.11, 3.85); unrounded 3.8487 and 4.9902 | holds: 3.8-5.0 |
| "2.6-4.6 times at q = 2 %" | the same ratios at q = 2 % | 2.57 to 4.61 (R1 4.30, 4.61; R2 3.00, 2.57) | 2.56 to 4.56 (R1 4.34, 4.56; R2 2.94, 2.56); unrounded 2.5592 and 4.5642 | holds: 2.6-4.6 |
| "4,262-4,310" | the BB72 pairs of f and g at q = 2 %, both seeds | 4,262 to 4,310 | 4,210 to 4,256 | changes to 4,210-4,256 |
| "about 20 % below Relay-BP alone 5,382" | 1 - pair / Relay-BP alone, f and g at q = 2 %, both seeds | 19.9 to 20.8 % below 5,382 | 18.6 to 19.5 % below 5,228 | changes to about 19 % below Relay-BP alone 5,228 (18.6-19.5 %) |
| "23-24 %" (less often than correlated PM alone) | 1 - pair(b, q = 1 %) / correlated PM alone, both seeds | 23.99 % and 23.02 % (9,942 and 10,069 against 13,080) | 23.62 % and 23.64 % (9,921 and 9,919 against 12,989) | changes to 24 % (23.6 % in both seeds) |
| r764: 12,293 alone, 10,976 at q = 1 % | `references.csv`, surface | 12,293 and 10,976 | 12,336 and 10,937 | changes to 12,336 and 10,937 |
| r12: 7,126 alone, 4,336 at q = 2 % | `references.csv`, BB72 (per million) | 7,126 and 4,336 | 6,928 and 4,220 | changes to 6,928 and 4,220 |
| ramp reruns 1,170 / 682 / 1,027 / 531 | `ramp.csv` diff_pm of S-P and S-PA against the ramp rerun of S-VA, at q = 0 (S-P, S-PA) and q = 1 % (S-P, S-PA) | 1,170 / 682 / 1,027 / 531 | 1,222 / 597 / 1,097 / 449 | changes to 1,222 / 597 / 1,097 / 449 |
| routed shares within 0.1 pp of q for every network | largest absolute gap between routed_frac and q, `pair_curves.csv` | 0.0987 pp (surface), 0.0788 pp (BB72) | 0.0711 pp (surface), 0.0778 pp (BB72) | holds for the 18 arm networks and the 2 references; ramp reruns: NOT FOUND (below) |
| dagger: arm d seed 1 did not train | `train_checks.csv` ratio of d seed 1 | 1.000016 (1.0000163, trained = False) | same file, same ratio 1.0000163 | holds |

Ratio detail (BB72, R1 = h vs g, R2 = i vs f; counts are failures out of 500,000 shots, so the ratio is the same as per million):

| id (X vs Y) | q (%) | seed | original fail_x / fail_y | original ratio | re-test fail_x / fail_y | re-test ratio |
|---|---|---|---|---|---|---|
| R1 (h vs g) | 0 | 1 | 17,024 / 3,523 | 4.83 | 16,874 / 3,487 | 4.84 |
| R1 (h vs g) | 0 | 2 | 17,680 / 3,555 | 4.97 | 17,351 / 3,477 | 4.99 |
| R1 (h vs g) | 2 | 1 | 9,262 / 2,155 | 4.30 | 9,140 / 2,105 | 4.34 |
| R1 (h vs g) | 2 | 2 | 9,822 / 2,131 | 4.61 | 9,708 / 2,127 | 4.56 |
| R2 (i vs f) | 0 | 1 | 14,073 / 3,522 | 4.00 | 13,894 / 3,383 | 4.11 |
| R2 (i vs f) | 0 | 2 | 13,108 / 3,432 | 3.82 | 12,920 / 3,357 | 3.85 |
| R2 (i vs f) | 2 | 1 | 6,408 / 2,134 | 3.00 | 6,252 / 2,123 | 2.94 |
| R2 (i vs f) | 2 | 2 | 5,523 / 2,153 | 2.57 | 5,446 / 2,128 | 2.56 |

Relay-BP comparison detail. Relay-BP alone is re-measured on the new shots (5,228 per million), so the percentages use the re-test's own value.

| network | seed | pair at q = 2 %, original | below Relay-BP alone 5,382, original (%) | pair at q = 2 %, re-test | below Relay-BP alone 5,228, re-test (%) |
|---|---|---|---|---|---|
| f (B-JT) | 1 | 4,268 | 20.7 | 4,246 | 18.8 |
| f (B-JT) | 2 | 4,306 | 20.0 | 4,256 | 18.6 |
| g (B-J) | 1 | 4,310 | 19.9 | 4,210 | 19.5 |
| g (B-J) | 2 | 4,262 | 20.8 | 4,254 | 18.6 |

Correlated PM comparison detail:

| network | seed | pair at q = 1 %, original | fewer failures than correlated PM alone 13,080, original (%) | pair at q = 1 %, re-test | fewer failures than correlated PM alone 12,989, re-test (%) |
|---|---|---|---|---|---|
| b (S-V) | 1 | 9,942 | 23.99 | 9,921 | 23.62 |
| b (S-V) | 2 | 10,069 | 23.02 | 9,919 | 23.64 |

Ramp reruns (descriptive, seed 1 only, outside the Holm family): X - Y in failures per million, a positive value means the pooled-head ramp rerun fails more often than the vote's ramp rerun. All four re-test intervals exclude zero, so the A1 statement that on seed 1 the vote's ramp rerun beat both pooled ramp reruns holds at both q.

| ramp rerun of X vs ramp rerun of S-VA (seed 1) | q (%) | original X - Y [95 %] | re-test X - Y [95 %] | re-test unadjusted p |
|---|---|---|---|---|
| S-P | 0 | +1,170 [+966, +1,374] | +1,222 [+1,018, +1,426] | 7.7e-32 |
| S-PA | 0 | +682 [+489, +875] | +597 [+404, +790] | 1.4e-09 |
| S-P | 1 | +1,027 [+850, +1,205] | +1,097 [+920, +1,275] | 7.2e-34 |
| S-PA | 1 | +531 [+365, +697] | +449 [+282, +616] | 1.5e-07 |

In the re-test the ramp rerun of S-VA fails 12,989 times at q = 1 %, the same count as correlated PM alone. The two are separate reads: the frozen analysis computes the ramp count from the ramp network's own pair, as in the original, where the two counts were 13,162 and 13,080.

Routed shares. For every network and every q above 0 the routed fraction is within 0.1 pp of q, in the original and in the re-test, so the sentence holds as worded.

| set | largest gap, original | largest gap, re-test | cells above 0.1 pp, re-test |
|---|---|---|---|
| surface, 10 networks x 6 q | 0.0987 pp (S-L seed 1, q = 16 %: 160,987 of 1,000,000 shots routed) | 0.0711 pp (S-P seed 1, q = 16 %: 160,711 of 1,000,000) | 0 of 60 |
| BB72, 8 networks x 6 q | 0.0788 pp (B-J seed 2, q = 5 %: 25,394 of 500,000) | 0.0778 pp (B-PA seed 1, q = 5 %: 24,611 of 500,000, i.e. 4.9222 %) | 0 of 48 |
| r764 (surface reference) | 0.0342 pp (q = 16 %) | 0.0558 pp (q = 4 %) | none |
| r12 (BB72 reference) | 0.0494 pp (q = 2 %) | 0.0260 pp (q = 0.5 %) | none |
| ramp reruns (3 networks) | NOT FOUND: routed counts of the ramp reruns are not in `ramp.csv`, `analysis.json` or `pair_curves.csv` | NOT FOUND (same) | NOT FOUND |

The re-test's BB72 largest gap (0.0778 pp) is close to the 0.1 pp limit but inside it. The sentence "within 0.1 pp of q for every network" can stay; the ramp reruns are not covered by the output files, so the check covers the 18 arm networks and the two references.

Dagger. The re-test reads the same frozen network for arm d seed 1, and `train_checks.csv` (the training-record check, not re-run) is identical to the original: ratio 1.0000163384947602, trained = False, so the dagger holds. The network is as far from working as before: alone 478,245 failures per million (original 478,846), pair at q = 1 % 473,494 (original 474,111). Its seed-1 comparisons are reported as they come: at q = 0, R1 (d vs b) +467,584 [+466,585, +468,582], U2 (e vs d) -463,321 [-464,327, -462,315], W2 (d vs a) +463,227 [+462,220, +464,233]. One more row of the same file is flagged: arm e seed 1 (S-PA, `a1_S-PA_s1`) also has trained = False, ratio 1.0000152, yet it decodes well (alone 14,924 per million, original 14,917). A1_RESULT.md says S-PA seed 1 "failed the not-trained check at 2e7 samples but trained later". The dagger names only d seed 1; this flag on e seed 1 is in the original records as well, and the re-test, which reads the same networks, does not change it.

### Other sentences of the paper's head paragraphs (main.tex 1620-1634)

| sentence | original | re-test | holds |
|---|---|---|---|
| surface: "With the line supervision, seed 1 favours the pooled head over the vote with p = 8.8e-4, and seed 2 favours the vote with p = 2.9e-68" (R3, e vs a, q = 0, per seed) | seed 1 -335 per million, p 8.8e-4; seed 2 +1,744, p 2.9e-68 | seed 1 -94 [-291, +103], p 0.35; seed 2 +1,878, p 3.4e-77 | seed 2 only: seed 1 shows no preference on the fresh shots (R3's verdict, inconclusive, repeats) |
| surface: the vote trained on the final answer (b, S-V) is the best surface-code network and fails less often alone than the deployed d = 9 network (r764) and correlated PM alone | 10,638 and 10,881 against 12,293 and 13,080 | 10,661 and 10,822 against 12,336 and 12,989 | yes |
| BB72: the pooled head with the translate supervision (i, B-PA) gets below Relay-BP alone only at the largest tested q, 5 %, still above the joint class score's pairs at 2 % | 4,736 and 4,742 per million (seeds 1, 2) against 5,382; at q = 3 % 7,488 and 6,378 | 4,652 and 4,608 against 5,228; at q = 3 % 7,282 and 6,250 | yes, with the new numbers |
| BB72: the pooled head without it (h, B-P) stays above Relay-BP at q = 5 % | 6,210 and 6,196 (seeds 1, 2) | 5,992 and 6,116 | yes, with the new numbers |

## 5. How the run went (deviations, platforms)

- Every read and analysis ran once from the frozen code; the D1 and d = 13 analyses were submitted held and released
  only after the tree pins printed `PIN_OK` once the reads had ended (`retest/run/pins.log`).
- Gross CPU reads: the three Derecho jobs had not started after 90 minutes and were deleted by their ids; the Alpine
  fallback's three 64-core jobs did not start either, so they were cancelled by their ids before starting and the same
  script ran as 15 one-chunk jobs of 16 processes (job shape only; decoders, settings, shots and seeds unchanged;
  `RETEST.md` section 3.1 records it).
- The rented H100 that read the gross network and cudaq-qec is a second pod of the same type, set up from the same
  records and checked (bit-equal platform checks) before reading; the first pod was stopped before any read of the run.
- BB72's networks read on Alpine's RTX PRO 6000, so the analysis named that GPU from the reads' engine signatures
  (`--gpu auto`); the gross network read on the frozen H100.
