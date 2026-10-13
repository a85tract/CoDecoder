# Draft amendment (PLAN version 4 proposal, NOT registered): ConvLines + routing to full-DEM Relay-BP

2026-10-06. For the study lead's review. Nothing here is registered until a reviewed PLAN
version is committed. The 4 committed policies, PLAN's three registered cells, their claims, alpha shares, test
replicas and sizes stay as they are. Every number below comes from the BB72 cal splits (bb.fullroute; records
`ising_compare/data/bb/fullroute/bb72_cal.json` and `bb72_cal_sizes.json`, commit 7c324df2). No dev or test number is
used for any choice here.

## Evidence (cal, 200,000 shots per cell, full-DEM Relay-BP decoded on the whole split)
Full-DEM Relay-BP fails 1,117 times at p 0.003 and 169 times at p 0.002 (sector Relay-BP: 2,734 and 523). The
networks alone lose to it at p 0.003 (r11 1,333, r12 1,396; exact McNemar p 1e-8 and 6e-13) and are level with it at
p 0.002 (152 and 146; p 0.26 and 0.12). Every comparison below is paired on the same shots and gives the failures of
only the hybrid / only full-DEM Relay-BP, the exact two-sided McNemar p and the Newcombe 95 % interval of the
difference in failure rate.
- p 0.002, q 0.25 %: hybrid r12 106 (10 / 73, p 6e-13, interval [-4.1e-4, -2.3e-4]) and r11 119 (15 / 65, p 1.4e-8,
  [-3.4e-4, -1.6e-4]) against 169. Provisional composite 0.621 / 0.607 ms against LIMIT_MS 0.63 ms. At q 0.5 % the
  hybrid fails 123 / 125 times and costs 1.01-1.06 ms.
- p 0.003: from q 0.5 % up, the hybrid of either replica beats full-DEM Relay-BP at every q of the grid
  (p <= 0.035). It is best at q 1-3 %: r12 900 at q 2 % (80 / 297, p 2e-30, [-1.28e-3, -0.90e-3]), r11 909 at q 1 %
  (222 / 430, p 3e-16). At
  q 0.25 % it does not (r11 1,123, level, p 0.88; r12 1,200, worse, p 0.019). The winning q cost 4.9-11 ms, and the
  cheapest q (0.25 %, 1.67 ms) is above LIMIT_MS 1.06 ms.
- Provisional CPU basis: full-DEM Relay-BP's per-shot times recorded on an Apple M4 Mac mini (5-6 decoding processes
  sharing it) x 1.87-1.90, the ratio of sector Relay-BP's recorded times on part A (Alpine cal read, 32 processes on a
  shared acpu node) to the same decoder on the same shots on the Mac mini. Part A is the union of both replicas' 5 %
  routed tails (12,592 shots at p 0.003, 13,307 at p 0.002). On this basis full-DEM Relay-BP alone takes 140 ms per
  shot at p 0.003 and 88 ms at p 0.002 (cal means).

## Cell BB72-2F (proposed): [[72,12,6]], p 0.002, under the registered speed rule
- Hybrid: the test replica's network through the frozen fused engine (the committed policy's engine signature).
  Shots with margin < tau go to full-DEM Relay-BP in its registered configuration (`bb.opp relay_full`: relay-bp
  0.2.2 RelayDecoderF32, qnd.teachers 'relay' parameters, all detectors).
- q, chosen on cal. Grid {0.25, 0.5, 1, 2, 3, 5} % (the study lead's, fixed before any full-DEM cal number). A q is
  admissible when its provisional composite (PLAN 2.7's formulas) is at most LIMIT_MS (0.63 ms). The rule picks the
  admissible q with the fewest hybrid cal failures, ties to the smaller q. Result: q 0.25 % for both replicas
  (q 0.5 % costs 1.01-1.06 ms).
- Test replica: the replica with fewer hybrid cal failures at its q; ties go to the smaller provisional composite,
  then to replica 1. Result: replica 2 (r12, 106 < 119). tau = the 0.25 % quantile of r12's cal margins, stored at
  full precision (2.979396343231201).
- Timing to re-measure before registration. r12's composite (0.621 ms) is 1.5 % under the 0.63 ms limit, with a CPU
  term scaled from Mac mini times, a margin too small to rely on. Before registration, full-DEM
  Relay-BP's time on r12's 500 routed cal shots must be re-measured on an exclusive host (a whole node allocated alone,
  as for Q8) and the composite recomputed from it. Above 0.63 ms, BB72-2F has no admissible q and is dropped.
- Claim: on BB72-2's test shots the hybrid fails less often than full-DEM Relay-BP and than each of the three gating
  opponents. Each comparison uses the exact two-sided McNemar test (p < alpha_F) and needs the Newcombe
  (1 - alpha_F) interval entirely below 0; all four must pass (intersection-union). The composite batch-1 time on the
  quiet hosts must also be at most the fastest gating opponent's.
- Multiplicity: alpha_F = 0.05 / 3, the same share as each registered cell. The registered three-cell family keeps
  its 0.05. Over all claims of the study the union bound becomes 0.05 x (3 + k_F) / 3, which is 0.067 for one F cell
  and 0.083 for two. The report must state this.
- Test size: PLAN 3.3's simulation (`bb.fullroute size`, i.e. `bb.route.power`: 20,000 draws, fixed seed), run on the
  test replica's paired cal counts, not dev. Per-comparison power is 1 - 0.10 / 4. Floor 300,000, cap 1,000,000.
  From the cal counts (both fail 96, only the hybrid 10, only full-DEM Relay-BP 73), n = 300,000, the floor (simulated
  power there 1.00). The gating comparisons are not binding: on cal sector Relay-BP fails 523 times. BB72-2F uses
  BB72-2's beacon-seeded test split, sized max(n of BB72-2, n of BB72-2F), and full-DEM Relay-BP decodes all of it
  (registered now: only the first 200,000 test shots).
- Timing (PLAN 2.7, unchanged): GPU term and handoff from r12's rows of the Alpine GPU session (73.3 us, 12.5 us).
  The CPU term is full-DEM Relay-BP on the F policy's routed dev shots (about 750), measured on the Q8 exclusive node
  with three fresh processes, the slowest counted. This is timing only: no dev failure count of the F policy is ever
  computed. The gate is composite <= the fastest gating opponent's quiet-host time. Stage 3 repeats the measurement on
  the routed test shots (3.6). Q8 needs about 10 more minutes.

## Option BB72-3F (p 0.003): only with a weaker speed claim
No q is admissible under LIMIT_MS (1.06 ms); the cheapest costs 1.67 ms. Under the registered speed rule the cell does
not exist. To make the accuracy claim at p 0.003 anyway, the speed condition would change to: composite <= full-DEM
Relay-BP's own quiet-host mean single-shot time (first 10,000 test shots, three fresh processes, the fastest
counted), with the fastest gating opponent's time reported beside it and not gated. With that rule every q of the grid
is admissible, and the fewest hybrid cal failures pick r12 at q 2 % (900; composite 8.35 ms, 6.0 % of full-DEM
Relay-BP's provisional mean time per shot, 140 ms on the same basis). The claim is accuracy against full-DEM Relay-BP
alone at alpha_F = 0.05 / 3. Test size from the cal counts (both fail 820, only the hybrid 80, only full-DEM Relay-BP
297; power 0.90): n = 200,000, the floor (simulated power there 1.00). The study lead decides whether this weaker
speed claim is worth registering.

## Second lever, not proposed for registration
The averaged fp32 scores of the two replicas fail 1,235 times on cal at p 0.003 (r11 1,331, r12 1,397 in fp32) and
133 times at p 0.002 (152, 146), 7-9 % fewer than the better replica. The ensemble adds one engine call, about
+0.074 ms (provisional), and would need a new deployed engine and its own validation. At p 0.003 it still loses to
full-DEM Relay-BP on its own (1,235 against 1,117; 742 / 624, p 0.0015). At p 0.002 it beats full-DEM Relay-BP on cal
without routing (133 against 169; 74 / 110, p 0.0097, interval [-3.2e-4, -0.5e-4]), where the single networks are
only level with it. Routing does not help the ensemble at p 0.002: 114 failures at q 0.25 % (0.71 ms, over the limit)
against r12's 106 (0.62 ms).
