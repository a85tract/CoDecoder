# X2: where ConvLines and its strong decoder fail on different shots (exploratory)

12 October 2026, research branch `ising-rev-x2` (from `ising-revision` 859b9981). This is item X2 of the revision
plan (`REVISION_PLAN.md` section 2.11, a research document not included here), the analysis behind the paper's "why
it works" section. It works from saved results: the recorded per-shot outputs, with the shots rebuilt from their recorded seeds
and re-decoded by PyMatching. It is exploratory and adds no "better / worse" verdict. Every registered comparison
stays as it was, and the routed fractions other than the registered ones in sections 5 and 6 are read after the fact
on the same test shots.

Code: the `*.py` files beside this one. Tables: `data/tables.md` (all of them, generated) and the condensed ones below.
Figures: `figures.py` draws them into `$X2_OUT/fig/`; the paper's two are drawn by `../figures/make_why_fig.py`. Every
number was computed on Alpine CPU nodes, in about 7.0 allocated core-hours (section 9). No GPU was used.

## In short

- On the registered test shots, only the network fails on 5,416 of 1,000,000 shots at d = 13 and on 7,380 of
  1,000,000 at d = 17. Only correlated PyMatching (correlated PM) fails on 3,834 and 1,811, and both fail on 2,715 and
  1,132.
- The two kinds of lone failure look alike by detection-event count, by the density of time-like, space-like and
  space-time event pairs, and by distance to the X boundaries. Both have 4-7 % more events than a typical shot. On both
  kinds, plain matching's two homology classes are nearly tied: the median class gap is 0.9 and 0.7 nats at d = 13
  and 2.7 and 1.1 nats at d = 17, against 10.4 and 11.6 nats on a typical shot.
- Correlated matching resolves the network's lone failures with a clear margin (median about 6 nats at d = 13 and 11
  at d = 17, approximate). On its own lone failures its margin is near zero.
- The network's confidence |c| orders the two kinds. Below the registered threshold, the network is the decoder that
  fails on 72 % (d = 13) and 85 % (d = 17) of the shots where exactly one decoder fails; above it, on 37 % and 35 %.
  The threshold therefore keeps 2,278 shots that only matching fails against 1,356 that only the network fails at
  d = 13 (520 against 285 at d = 17). This difference is the pair's whole gain over correlated PM: 922 and 235
  failures per million shots, the registered numbers.
- The network's failures sit at low confidence (Q2). Its least confident 1, 2 and 5 % of shots hold 46, 67 and 89 %
  of its failures at d = 13 and 43, 65 and 88 % at d = 17. Correlated PM fails on 17.0, 14.8 and 9.5 % of those shots
  at d = 13 and on 5.8, 5.1 and 3.7 % at d = 17. At the registered thresholds (1.94 % and 7.97 % routed), correlated
  PM fails on 2,882 of the 19,405 routed shots (14.9 %) at d = 13 and on 2,229 of 79,652 (2.8 %) at d = 17.
- By the measures used here, no class of physical error separates what the two decoders miss. Correlated PM misses
  shots that stay near-ties under its own weights; on 57 % (d = 13) and 60 % (d = 17) of them plain PM decodes the
  shot, so correlated matching's reweighting tipped it the wrong way. The network misses near-ties that correlated
  matching resolves, and it flags them with low |c|. Their class difference holds a run of three or more time-like
  edges at one site slightly more often (8.2 % against 6.1 % at d = 13, 12.1 % against 9.8 % at d = 17). This fits the
  network's finite reach in time, but the effect is small.
- On BB72 (Q3, the D1 test), our Cascade re-implementation C14' fails on 86 kept shots (of 11,000,000, p = 0.2 %) and
  342 (of 17,000,000, p = 0.3 %) that S55'-deep decodes. On most of them C14' gets a single logical qubit wrong with a
  near-zero logit: 71 of 86 and 251 of 342, median |logit| 0.78 and 0.74. C14''s overall confidence on these shots has
  median percentile 0.07 % and 0.37 %. S55'-deep's joint readout keeps them with a median margin percentile of 17 %
  and 27 %, well above its threshold (2 % and 10 % routed). S55'-deep's own kept failures (19 and 81) sit just above
  that threshold.

## 1. Inputs and checks

| test | shots | network, routing | per-shot records | shots |
|---|---|---|---|---|
| surface code d = 13: study 2's registered test (T2, chunks 0-99) | 1,000,000 | `d13_b5_r775`, q = 2 %, tau = 1.8486117124557495 | study 2's T2 read: `ours_deployed` (the deployed executable's c, routing and decision), `pm_corr`, `pm`. They were on scratch only, so REV-x2 copied them into the archive first (`revision/ARCHIVE_MANIFEST.md`, third copy). | rebuilt from `T2.json`'s seeds |
| surface code d = 17: R17 (T6, chunks 0-99) | 1,000,000 | `d17_b5_r787`, q = 8 %, tau = 4.539173126220703 | R17's read in the archive (`ising_r17/out/preds/d17`, first copy) | rebuilt from T6's seeds (91,170,000 + j) |
| BB72, D1 at p = 0.2 % and 0.3 %, setting `depspam_fm` | 11,000,000 and 17,000,000 | S55'-deep in front of C14', q1 = 2 % and 10 % | D1's eager reads (detector rows, observables, both networks' outputs) and compiled reads, copied into the archive by REV-x2 | stored in the reads |

Checks, all passed:

- Every surface chunk was rebuilt from its seed with Stim 1.16.0 and numpy 2.2.6, on Alpine's AMD EPYC 9534 (d = 13)
  and EPYC 7713P (d = 17) nodes, and the chunk's file bytes have the manifest's sha256: 100 of 100 at each d. Each
  record file has the sha256 its provenance record names, and its `input_sha256` is the chunk's. The records'
  observables equal the rebuilt shots'. The BB72 rows match the reads' per-chunk detector digests (1,100 and 1,700
  chunks).
- PyMatching 2.4.0 on the circuit's DEM reproduces every recorded decision of correlated PM and of plain PM: 0
  differences in 1,000,000 shots at each d.
- The registered counts come out exactly. At d = 13 the routing |c| < tau gives the recorded routing on every shot,
  and the pair fails 5,627 times with 19,405 routed; correlated PM fails 6,549 times. At d = 17, R17's q = 8 % routing
  gives 2,708 failures with 79,652 routed, and correlated PM fails 2,943 times. The d = 17 record carries its own
  routing, made at study 4's tau, and that routing and the record's decisions also match. These are the registered
  numbers (release README). The BB72 reads reproduce D1's table: routed 220,058 and 1,701,544; S55'-deep alone 10,348
  and 112,636; C14' 7,922 and 76,730; kept breaks / repairs / both fail 19 / 86 / 238 and 81 / 342 / 1,977. The
  compiled paths give D1's repairs of 78 / 79 and 348 / 347.
- For the class margins, splitting the X boundary (section 2) sends every X-type boundary edge to row gap 0 or d - 2.
  On every shot the two forced decodes land in different classes. The lower plain weight equals plain PM's own
  solution weight (largest difference 0), and its class is plain PM's decision except on 31 (d = 13) and 8 (d = 17)
  shots, which are ties.

## 2. Definitions

- The network alone decides every shot by c > 0. The four cells cross that decision with correlated PM's: both right,
  only the network fails (NF), only correlated PM fails (MF), both fail (BF). A shot is routed when |c| < tau, as
  registered, and kept otherwise. The pair fails on kept NF, routed MF and every BF shot, so the pair's failures minus
  correlated PM's equal kept NF minus kept MF: the retained-shot excess of the paper's Eq. (2).
- The |c| percentile is the rank of |c| among the 1,000,000 test shots, as a percentage (low = unsure). The routed
  shots are the lowest 1.94 % (d = 13) and 7.97 % (d = 17).
- Detection events are a shot's fired detectors, X and Z type. Time-like, space-like and space-time pairs are the
  pairs of fired X-type detectors joined by a DEM edge of that kind: the same site in consecutive rounds (measurement
  errors), the same round (data errors), or another site in another round (errors during the CNOT sequence). The
  tables give them per 100 X-type events.
- X-type detectors sit between data rows k and k + 1. A Z-error chain from one must cross min(k + 1, d - 1 - k) data
  rows to reach the nearer of the two boundaries where those chains end. The rows come from the release's grid
  (`weights/grid_d*.npz`), and the tables give the mean of this distance over a shot's X-type events as "rows to the
  nearer X boundary".
- For the class gap, the X boundary of the matching graph is split, and its side of logical line 0 becomes an extra
  detector A. Decoding with A = 0 and with A = 1 gives the least weight of a correction in each homology class. The gap
  is W(wrong class) - W(true class), in nats, negative where the least-weight class is wrong. It is exact for plain
  matching. With `enable_correlations` each forced decode reweights after its own first pass, so the correlated gap is
  approximate. Its sign differs from correlated PM's own decision on 1.10 % (d = 13) and 0.72 % (d = 17) of all
  shots, and it is near zero on correlated PM's lone failures.
- The class difference D is the set of edges in which the least-weight plain corrections of the two classes differ.
  D always runs from one X boundary to the other. Its time-like edges, its longest run of time-like edges at one site
  and the rounds it spans show how far the ambiguity runs along the time direction. D was computed on every NF, MF and
  BF shot, and on every 32nd shot (31,300 per d) as the reference.

## 3. The four cells and the routing (Q1)

| | d = 13, T2 | d = 17, T6 |
|---|---|---|
| shots, routed by the registered tau | 1,000,000; 19,405 (1.94 %) | 1,000,000; 79,652 (7.97 %) |
| both right: shots, routed | 988,035; 12,463 (1.3 %) | 989,677; 70,328 (7.1 %) |
| only the network fails (NF): shots, routed | 5,416; 4,060 (75.0 %) | 7,380; 7,095 (96.1 %) |
| only correlated PM fails (MF): shots, routed | 3,834; 1,556 (40.6 %) | 1,811; 1,291 (71.3 %) |
| both fail (BF): shots, routed | 2,715; 1,326 (48.8 %) | 1,132; 938 (82.9 %) |
| failures: network alone / correlated PM / pair | 8,131 / 6,549 / 5,627 | 8,512 / 2,943 / 2,708 |
| pair = kept NF + routed MF + BF | 1,356 + 1,556 + 2,715 | 285 + 1,291 + 1,132 |
| kept NF - kept MF = pair - correlated PM | 1,356 - 2,278 = -922 | 285 - 520 = -235 |
| the network's share of the shots exactly one decoder fails: routed / kept | 4,060 of 5,616 (72.3 %) / 1,356 of 3,634 (37.3 %) | 7,095 of 8,386 (84.6 %) / 285 of 805 (35.4 %) |

The routing sends to matching three quarters (d = 13) and 96 % (d = 17) of the shots that only the network fails. It
also sends 41 % and 71 % of the shots that only matching fails, and about half to four fifths of the shots both fail.
Matching's lone failures sit at higher network confidence: median |c| percentile 2.6 % against 0.9 % for the
network's lone failures at d = 13, and 3.8 % against 1.1 % at d = 17. Figure 2 shows the full distributions.

![Among the shots where exactly one decoder fails, the share that the front end fails, against its confidence](x2/fig/x2_who_fails.png)

*Figure 1 (`fig/x2_who_fails.pdf`).* Among the shots where exactly one decoder fails, the share on which the
front end is the one that fails, against the front end's confidence percentile. Bins hold equal counts on each side of
the registered threshold (vertical lines; 8 bins below and 4 above for the surface code, 8 and 3 for BB72); the bands
are Wilson 95 % intervals. (a) The surface code, the network against correlated PM: d = 13, solid, 9,250 shots;
d = 17, dashed, 9,191 shots. (b) BB72 (D1), S55'-deep against C14': p = 0.2 %, solid, 7,044 shots; 0.3 %, dashed,
73,076 shots. Each curve falls through 50 % near its registered threshold: matching (or C14') is usually the one that
is right below it, and the front end above it.

![ECDF of the network's confidence percentile per cell](x2/fig/x2_confidence.png)

*Figure 2 (`fig/x2_confidence.pdf`).* Cumulative distribution of the network's |c| percentile in each cell: (a)
d = 13 (T2), (b) d = 17 (T6). The cell sizes are in the titles (NF / MF / BF). The vertical line is the registered
threshold; the height of a curve there is the share of the cell routed. Grey dashed: all shots (the diagonal, by
construction).

## 4. What the discordant shots look like, and what each decoder misses (Q1)

Medians [25th, 75th percentiles] unless marked [10th, 90th]. The D rows for "all shots" come from every 32nd shot.

**d = 13 (T2)**

| | all shots | only the network fails | only correlated PM fails | both fail |
|---|---|---|---|---|
| shots | 1,000,000 | 5,416 | 3,834 | 2,715 |
| detection events | 201 [187, 216] | 215 [200, 229] | 215 [201, 229] | 215 [201, 229] |
| time-like pairs per 100 X events | 19.2 [16.7, 21.7] | 19.8 [17.3, 22.2] | 19.5 [17.1, 21.9] | 19.6 [17.2, 22.0] |
| space-like pairs per 100 X events | 34.6 [31.2, 38.0] | 36.1 [32.8, 39.5] | 36.0 [33.0, 39.3] | 36.2 [33.0, 39.6] |
| space-time pairs per 100 X events | 31.5 [27.2, 35.8] | 34.5 [30.2, 38.7] | 34.2 [29.9, 38.5] | 34.3 [30.2, 38.6] |
| rows from X events to the nearer X boundary, mean per shot | 3.50 [3.36, 3.63] | 3.53 [3.40, 3.65] | 3.51 [3.38, 3.64] | 3.51 [3.38, 3.64] |
| \|c\| percentile [10th, 90th] | 50.0 [10.0, 90.0] | 0.9 [0.1, 3.7] | 2.6 [0.4, 13.3] | 2.0 [0.3, 9.8] |
| plain PM fails on | 2.2 % | 40.0 % | 42.6 % | 73.3 % |
| plain class gap, nats [10th, 90th] | 10.4 [4.0, 15.9] | 0.9 [-3.5, 5.5] | 0.7 [-3.7, 5.1] | -2.0 [-6.5, 2.4] |
| correlated class gap (approximate), nats [10th, 90th] | 18.0 [8.2, 26.8] | 5.9 [-1.3, 14.1] | -0.7 [-7.0, 6.1] | -2.2 [-9.6, 4.3] |
| D: time-like edges | 3 [1, 4] | 3 [2, 5] | 3 [2, 5] | 3 [2, 4] |
| D: a run of >= 3 time-like edges at one site | 5.1 % | 8.2 % | 6.1 % | 6.0 % |
| D: rounds spanned | 4 [3, 5] | 4 [3, 6] | 4 [3, 5] | 4 [3, 5] |

**d = 17 (T6)**

| | all shots | only the network fails | only correlated PM fails | both fail |
|---|---|---|---|---|
| shots | 1,000,000 | 7,380 | 1,811 | 1,132 |
| detection events | 467 [445, 489] | 484 [462, 506] | 486 [463, 506] | 485 [464, 508] |
| time-like pairs per 100 X events | 19.3 [17.7, 20.9] | 19.6 [18.0, 21.2] | 19.4 [17.9, 21.1] | 19.5 [18.1, 21.2] |
| space-like pairs per 100 X events | 36.1 [33.9, 38.3] | 37.0 [34.7, 39.2] | 37.0 [34.7, 39.3] | 37.0 [34.6, 39.1] |
| space-time pairs per 100 X events | 33.5 [30.5, 36.3] | 35.1 [32.2, 37.9] | 35.1 [32.1, 37.9] | 35.0 [32.2, 37.9] |
| rows from X events to the nearer X boundary, mean per shot | 4.50 [4.37, 4.62] | 4.52 [4.40, 4.63] | 4.51 [4.40, 4.63] | 4.50 [4.39, 4.62] |
| \|c\| percentile [10th, 90th] | 50.0 [10.0, 90.0] | 1.1 [0.2, 4.6] | 3.8 [0.6, 18.5] | 2.6 [0.4, 11.8] |
| plain PM fails on | 1.6 % | 27.4 % | 39.9 % | 72.1 % |
| plain class gap, nats [10th, 90th] | 11.6 [5.0, 17.3] | 2.7 [-2.9, 8.2] | 1.1 [-4.2, 6.4] | -2.1 [-7.4, 2.9] |
| correlated class gap (approximate), nats [10th, 90th] | 22.7 [11.4, 33.2] | 10.7 [0.7, 22.1] | -0.2 [-8.8, 8.3] | -2.6 [-11.6, 6.1] |
| D: time-like edges | 4 [3, 6] | 5 [3, 7] | 5 [3, 7] | 5 [3, 7] |
| D: a run of >= 3 time-like edges at one site | 9.3 % | 12.1 % | 9.8 % | 12.6 % |
| D: rounds spanned | 6 [4, 9] | 6 [4, 8] | 6 [4, 8] | 6 [4, 8] |

The same rows split into kept and routed shots, and the share of each cell within 2 nats of a tie, are in
`data/tables.md`.

![Matching's class gap per cell](x2/fig/x2_gap.png)

*Figure 3 (`fig/x2_gap.pdf`).* Histograms of matching's class gap W(wrong class) - W(true class), as the share of
each group per 1-nat bin: (a, b) plain matching, exact; (c, d) correlated matching, approximate. Left d = 13, right
d = 17. Grey dashed: every 32nd test shot. The plotted range holds all shots but 0.04 % of BF at d = 13, and 0.35 % of
BF and 0.04 % of the reference at d = 17 (correlated panels).

Reading the tables:

- The two kinds of lone failure look alike. The network's and matching's lone failures carry the same number of
  detection events, the same pair densities and the same boundary distances, at both d. They differ from typical shots
  in the same way: 7 % (d = 13) and 4 % (d = 17) more events, with more space-like and space-time pairs. The boundary
  distance does not separate any cell from the others or from typical shots.
- Both kinds are near-ties for plain matching. The plain class gap is within 2 nats of zero on 42 % of NF and 43 % of
  MF at d = 13, against 4.0 % of typical shots; at d = 17 the shares are 29 % and 36 %, against 2.9 %. Plain PM fails
  on 40 % of NF and 43 % of MF at d = 13, and its gap histograms for NF and MF overlap almost exactly (Figure 3a).
- Correlated matching resolves the network's lone failures with margin and leaves its own near zero. On NF its
  approximate gap is positive on 86 % (d = 13) and 92 % (d = 17) of shots, with median 5.9 and 10.7 nats. On MF it is
  centred at zero (median -0.7 and -0.2 nats, positive on 45 % and 49 %).
- Correlated PM misses shots that stay near-ties after its reweighting. On 2,201 of its 3,834 lone failures at d = 13
  (57 %), and on 1,088 of 1,811 at d = 17 (60 %), plain PM decodes the shot, so the two-pass reweighting turned a
  right decision into a wrong one there. On the rest, both matchers
  choose the wrong least-weight class. The network is right on all of these lone failures and confident enough to keep
  59 % of them at d = 13 and 29 % at d = 17 (Figure 2).
- The network misses near-ties of plain matching that correlated matching resolves with margin; plain PM decodes 60 %
  of them at d = 13 and 73 % at d = 17. The network's |c| is low on them (median percentile 0.9 % and 1.1 %), so the
  registered thresholds route 75 % and 96 %. Their class difference holds a run of three or more time-like edges at
  one site a little more often than matching's lone failures do: 8.2 % against 6.1 % at d = 13 (difference +2.2
  points, 95 % interval [+1.1, +3.2]) and 12.1 % against 9.8 % at d = 17 (+2.2 points, [+0.7, +3.8]). This fits the
  network's finite reach in time (`ising_compare/interp/REPORT.md`, section 1), but the excess covers about one in
  fifty of these shots, too few to set them apart.
- The pair gains on kept shots where matching would fail on a near-tie the network is confident about: 2,278 at
  d = 13 and 520 at d = 17. It loses on kept shots where the network is confidently wrong: 1,356 and 285. Figure 1
  shows the balance: below the threshold most lone failures are the network's, and above it most are matching's.

## 5. How much of the network's failures the routed fraction captures (Q2)

The rows other than "registered tau" route the k least confident test shots for a fraction chosen after the fact. They
describe this test set and carry no verdict. The registered rows reproduce the registered pair failures.

| d | routed: least confident fraction | shots routed | network's failures among them (share of its 8,131 / 8,512) | network's failure rate on them | correlated PM's failures on them (rate) | pair failures | pair - correlated PM |
|---|---|---|---|---|---|---|---|
| 13 | 1 % | 10,000 | 3,715 (45.7 %) | 37.1 % | 1,696 (17.0 %) | 6,112 | -437 |
| 13 | registered tau (1.94 %) | 19,405 | 5,386 (66.2 %) | 27.8 % | 2,882 (14.9 %) | 5,627 | -922 |
| 13 | 2 % | 20,000 | 5,467 (67.2 %) | 27.3 % | 2,954 (14.8 %) | 5,618 | -931 |
| 13 | 5 % | 50,000 | 7,219 (88.8 %) | 14.4 % | 4,753 (9.5 %) | 5,665 | -884 |
| 13 | 10 % | 100,000 | 7,802 (96.0 %) | 7.8 % | 5,726 (5.7 %) | 6,055 | -494 |
| 17 | 1 % | 10,000 | 3,669 (43.1 %) | 36.7 % | 580 (5.8 %) | 5,423 | +2,480 |
| 17 | 2 % | 20,000 | 5,567 (65.4 %) | 27.8 % | 1,024 (5.1 %) | 3,969 | +1,026 |
| 17 | 5 % | 50,000 | 7,516 (88.3 %) | 15.0 % | 1,861 (3.7 %) | 2,857 | -86 |
| 17 | registered tau (7.97 %) | 79,652 | 8,033 (94.4 %) | 10.1 % | 2,229 (2.8 %) | 2,708 | -235 |
| 17 | 10 % | 100,000 | 8,199 (96.3 %) | 8.2 % | 2,386 (2.4 %) | 2,699 | -244 |

The routed shots are hard for both decoders. On them correlated PM fails 14.9 % (d = 13) and 2.8 % (d = 17) of the
time, against 0.65 % and 0.29 % over all shots. The network fails there two (d = 13) to four (d = 17) times as often
as correlated PM, so handing them over removes more failures than it adds. A given routed fraction captures about the
same share of the network's failures at both d (43-46 % at 1 %, 65-67 % at 2 %, 88-89 % at 5 %). At d = 17, though,
the network alone fails 2.9 times as often as correlated PM (8,512 against 2,943), so the routed fraction must be
larger. Routing 1 % or 2 % leaves the pair worse than correlated PM (+2,480 and +1,026 failures); at 2 % the network's
failures on kept shots, 2,945, are already about as many as correlated PM's 2,943 in all. The registered 8 % routes
94 % of them.

![Capture curves](x2/fig/x2_capture.png)

*Figure 4 (`fig/x2_capture.pdf`).* (a) The share of each decoder's own failures that fall among the routed shots,
against the routed fraction: the network's (blue) and correlated PM's (orange); d = 13 solid, d = 17 dashed. (b) The
pair's failures minus correlated PM's, per million shots, against the routed fraction; open circles mark the
registered thresholds. Descriptive: one test set per d, with the fractions other than the registered one read after
the fact.

## 6. BB72 with our Cascade re-implementation (Q3)

S55'-deep is the six-block ConvLines-BB front end. C14' is our re-implementation of Cascade (L 14, H 512, BF16),
not the Cascade paper's network. The setting is D1's: the fitted noise variant `depspam_fm`, eager path. A shot is
routed when S55'-deep's margin is below tau1.

| | p = 0.2 % | p = 0.3 % |
|---|---|---|
| shots; routed | 11,000,000; 220,058 (2.00 %) | 17,000,000; 1,701,544 (10.01 %) |
| failures: S55'-deep alone / C14' / pair | 10,348 / 7,922 / 7,855 | 112,636 / 76,730 / 76,469 |
| only S55'-deep fails: kept (breaks) + routed | 19 + 4,716 | 81 + 54,410 |
| only C14' fails: kept (repairs) + routed | 86 + 2,223 | 342 + 18,243 |
| both fail: kept + routed | 238 + 5,375 | 1,977 + 56,168 |
| S55'-deep's share of the shots exactly one fails: routed / kept | 4,716 of 6,939 (68.0 %) / 19 of 105 (18.1 %) | 54,410 of 72,653 (74.9 %) / 81 of 423 (19.1 %) |
| C14''s failures on kept shots | 324 of 7,922 (4.1 %) | 2,319 of 76,730 (3.0 %) |

Kept shots where exactly one network fails (medians [10th, 90th] for percentiles, [25th, 75th] otherwise):

| | all shots | repairs, 0.2 % | breaks, 0.2 % | repairs, 0.3 % | breaks, 0.3 % |
|---|---|---|---|---|---|
| shots | 11,000,000 / 17,000,000 | 86 | 19 | 342 | 81 |
| detection events | 32 [25, 40] / 47 [39, 56] | 41 [33, 46] | 43 [35, 52] | 56 [47, 64] | 60 [51, 67] |
| S55'-deep margin percentile | - | 16.7 [3.7, 70.7] | 4.1 [2.4, 10.4] | 26.9 [11.8, 66.2] | 12.3 [10.3, 24.2] |
| C14' confidence (min \|logit\| over the 12 logicals) percentile | - | 0.07 [0.03, 0.23] | 0.15 [0.05, 0.24] | 0.37 [0.10, 1.34] | 0.74 [0.35, 4.03] |
| logicals C14' gets wrong: 1 / 2 / >= 3 | | 71 / 10 / 5 | 0 | 251 / 40 / 51 | 0 |
| C14''s \|logit\| on its wrong logical | | 0.78 [0.31, 1.65] | | 0.74 [0.22, 1.83] | |
| time-like pairs per 100 events | 14.0 [10.0, 18.5] / 16.0 [12.5, 20.0] | 14.4 [9.6, 17.7] | 16.2 [14.6, 21.6] | 17.1 [13.5, 20.4] | 16.7 [12.2, 21.7] |

![BB72](x2/fig/x2_bb72.png)

*Figure 5 (`fig/x2_bb72.pdf`).* BB72, D1's shots. (a, b) Cumulative distribution of S55'-deep's margin
percentile for the shots only S55'-deep fails (blue), only C14' fails (orange) and both fail (aqua); the vertical line
is tau1. (c, d) The kept shots where exactly one network fails, placed by S55'-deep's margin percentile and C14''s own
confidence percentile: breaks (blue circles) and repairs (orange squares).

Where the small network beats C14' on the shots it keeps:

- Most repairs are single-logical mistakes made with a near-zero logit. On 71 of the 86 repairs (0.2 %) and 251 of the
  342 (0.3 %), C14' gets exactly one of the 12 logical qubits wrong. The median |logit| of the wrong one is 0.78 and
  0.74; read as log-odds, C14' gave the right value a probability of about 0.3. C14''s overall confidence on these
  shots has median percentile 0.07 % (0.2 %) and 0.37 % (0.3 %), close to that of its routed failures (0.03-0.06 % and
  0.17-0.36 %). S55'-deep reads the 12 logicals as one joint class and keeps these shots: its median margin percentile
  on them is 16.7 % and 26.9 %, well above tau1 (2.00 % and 10.01 %). The repairs carry about 20-30 % more detection
  events than a typical shot, as the other failure cells do, and the same time-like pair density.
- The breaks sit just above tau1. S55'-deep's 19 and 81 kept failures have median margin percentile 4.1 % (tau1 at
  2.00 %) and 12.3 % (tau1 at 10.01 %). C14' is unsure on them too (0.15 % and 0.74 %) but right.
- The ordering is the same as on the surface code. Below tau1, S55'-deep is the one that fails on 68 % and 75 % of the
  shots where exactly one network fails; above it, on 18 % and 19 % (Figure 1b). On BB72 the routed region also holds
  96-97 % of C14''s failures, because hard shots are hard for both networks. Kept shots that both fail (238 and 1,977)
  are mostly multi-logical failures that C14' makes confidently: 202 of 238 and 1,661 of 1,977 have three or more
  logicals wrong, median |logit| 4.8 and 5.0.
- On the routed fraction (Q2 on BB72): S55'-deep's least sure 1, 2 and 5 % of shots hold 94.2, 97.5 and 99.0 % of its
  failures at 0.2 %, and 61.4, 81.1 and 94.5 % at 0.3 %. C14' fails on 6.6, 3.5 and 1.4 % of those shots at 0.2 %,
  and on 24.6, 17.0 and 8.3 % at 0.3 % (`data/tables.md`).

## 7. Draft paragraph for the paper

[Cut: the draft of the paper's paragraph. The paper states the result.]

## 8. Limits

- The analysis is exploratory. It describes one test set per distance and one network per distance, after the
  registered analysis. Routed fractions other than the registered ones were read on the test shots (sections 5 and 6)
  and do not support any choice of threshold.
- The errors themselves are not recorded, because Stim's detector sampler does not keep them. The "structure of the
  errors" is therefore the structure of matching's least-weight corrections: the true-class one and the difference D
  between the classes. D always crosses from one X boundary to the other, so a boundary measure on the defects D
  touches says nothing (its nearest defect is one row from a boundary on 97-98 % of shots in every cell). The boundary
  distance used here is that of the detection events.
- The correlated class gap is approximate and is used only for its broad shape; the exact plain gap carries the main
  statement.
- The difference in time-like runs is small (+2.2 points) and descriptive, with intervals from a normal approximation
  for independent cells.
- The network's c is the deployed executable's, as recorded in the test reads (FP16 inputs, fp32 accumulation). The
  percentiles are among the test shots, not the calibration shots that fixed tau.
- On BB72, C14' is our re-implementation of Cascade, not the Cascade paper's network. The setting is D1's fitted
  noise variant, and the analysis uses the eager path. On the compiled paths the repairs are 78 and 79 (0.2 %) and 348
  and 347 (0.3 %), against 86 and 342 eager, and the breaks are unchanged. There are no "distance to boundary"
  descriptors, because the code has no boundary.

## 9. Jobs, files, and how to rerun

Alpine jobs, in order (`acpu`; allocated core-hours = cores x wall time, 7.0 in all):

| # | what | cores | wall | core-hours |
|---|---|---|---|---|
| 1 | `inspect_inputs.py`: the layout of the inputs | 3 | 34 s | 0.03 |
| 2 | `archive/copy_verify3.sh`: the third archive copy | 4 | 2 min 8 s | 0.14 |
| 3, 4 | `surface_chunk.py --smoke`, chunk 0 at d = 13 and 17 | 4, 4 | 33 s, 1 min 42 s | 0.15 |
| 5 | `surface_chunk.py --d 13 --chunks 0-99 --procs 32` | 32 | 2 min 17 s | 1.22 |
| 6 | `surface_chunk.py --d 17 --chunks 0-99 --procs 32` | 32 | 8 min 41 s | 4.63 |
| 7, 8 | `bb_d1.py layout` | 3, 3 | 6 s, 7 s | 0.01 |
| 9 | `bb_d1.py run` | 11 | 1 min 27 s | 0.27 |
| 10 (failed: deployed before the file existed), 11 | `bb_join.py` | 3, 3 | 4 s, 44 s | 0.04 |
| 12, 13 | `post.py`: joins, tables, figures (the second after the figure and table changes) | 5, 5 | 1 min 17 s, 4 min 35 s | 0.49 |
| 14 | `figures.py` (final figures) | 3 | 15 s | 0.01 |

Most of the time went to the forced correlated decodes at d = 17 (110 s of each chunk's 125 s). They feed only the
approximate correlated gap.

| file | content |
|---|---|
| `x2lib.py` | inputs and their checks, shot rebuilding, the grid geometry, the split-boundary DEM, edge kinds |
| `surface_chunk.py` | one surface chunk: checks, re-decoding, class gaps, syndrome counts, D's structure (checkpointed per chunk) |
| `surface_join.py` | joins the chunks, checks the registered counts, writes the cell descriptors and capture rows |
| `bb_d1.py`, `bb_join.py` | BB72: D1's reads checked against their digests, the detector layout, per-shot features; the tables |
| `tables.py`, `figures.py`, `post.py` | the tables (`data/tables.md`) and the figures; `post.py` runs the two joins, then the tables and the figures |
| `archive/` | the two hash lists that the archive copy checked |
| `data/surface/summary.json`, `data/bb/summary.json` | every number in this file |
| `data/surface/fig_d13.npz`, `data/surface/fig_d17.npz`, `data/bb/fig.npz` | the figures' inputs |

The figures' inputs (5.8 MB) are in `data/`. The per-chunk outputs (63 MB at each d) and the BB72 per-shot features
(181 MB) are derived files that are not included. Rebuilding them needs the per-shot records of the test reads, which
are archived on the cluster and not published. With them, set `X2_ARCHIVE` (and for BB72 `X2_D1_READS` and
`QND_RELAY`; `X2_OUT` is the output directory) and run, from this directory with the repository's Python environment:

    python surface_chunk.py --d 13 --chunks 0-99 --procs 32     # 32 cores, 96 GB
    python surface_chunk.py --d 17 --chunks 0-99 --procs 32
    python bb_d1.py layout && python bb_d1.py run                # 40 GB
    python post.py                                               # the joins, the tables and the figures

The figure colours are chosen to stay distinct under colour-vision deficiency, and line style, markers and legends
also carry identity.
