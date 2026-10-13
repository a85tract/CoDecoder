# Exploratory: a deeper small network routed to Cascade, on phase C's confirmation reads

Everything in this file is exploratory and post hoc, and none of it is a claim. It changes none of phase C's registered
results, locks or verdicts: `RESULTS.md` stands as written, on Amendment 1's lock (629ca1c4). The confirmation block
(seed kind 11) had already been read for the registered analysis, and its registered results were known when this
analysis was specified. Of the configurations below, only the first lock's picks were ever locked, and that lock was
superseded before the confirmation read; the rest were never locked. Fifty-two configurations were evaluated on the
confirmation block, and no p-value is adjusted for multiplicity. No new shot was drawn and no network was run: the
analysis reuses phase C's reads.

## 1. The question

Can a deeper small network routed to Cascade be more accurate than Cascade at p = 0.3 %? The configuration is phase C's
S -> B. S decides, and the shots on which S's margin is below a threshold go to B, which decides them. B is Cascade:
C14' in `depspam_fm` and C14 in `uniform` (Cascade, L 14, H 512, 19.7M parameters). The small networks are S55'
(ConvLines-BB, 3 blocks, 483k parameters) and S55'-deep (6 blocks, 841k) in `depspam_fm`, and S55 (3 blocks) in
`uniform`.

At 0.3 % the first lock (1e8d1c6d) picked S55'-deep -> C14' with q1 = 10 % routed in `depspam_fm` (2,297 development
failures against Cascade's 2,305) and S55 -> C14 with q1 = 40 % in `uniform` (1,263 against 1,264). Amendment 1's lock
replaced both, and phase C never read them on the confirmation block (`RESULTS.md`, section 6). On the development block
the best S55'-deep pair was below Cascade at 9 of the 10 p in `depspam_fm`, by 0.01-2.3 %, each time as the minimum over
6 thresholds (and over 12 candidates when the first lock compared it with S55').

A routed shot gets Cascade's decision and a kept shot gets S's, so the pair and Cascade can only differ on kept shots.
On a kept shot, S repairs Cascade when Cascade fails and S does not, and breaks it when S fails and Cascade does not.
The pair's failures minus Cascade's equal the breaks minus the repairs. Routing more shots can only bring the pair
toward Cascade's level; to beat Cascade, the network must repair more kept shots than it breaks.

## 2. Answer

At 0.3 %, no pair beats Cascade on the confirmation shots. The closest is the first lock's pick in `depspam_fm`,
S55'-deep -> C14' with 10 % routed: 4,452 failures in 1,000,000 shots against Cascade's 4,459, from 4 breaks and 11
repairs. That is 7 fewer (0.16 %), with McNemar p = 0.12 and a 95 % interval for the difference of block failure rates
of -1.6e-05 to +1.4e-06, or -0.35 % to +0.03 % of Cascade's rate. With 20 % and 40 % routed the deep pair is 5 and 1
below Cascade, with 5 % routed 1 above, and with 1-2 % routed clearly worse. S55' -> C14' is never below Cascade at 0.3
%. In `uniform` the closest pair is again the first lock's pick, S55 -> C14 with 40 % routed: 3,715 failures in
1,500,000 shots against 3,716.

At 0.2 % in `depspam_fm` the deep pair is below Cascade at every q1, by 9 to 36 of Cascade's 3,858 failures in 5,000,000
shots. The first lock's pick there, S55'-deep -> C14' with 2 % routed, fails 3,822 times: 7 breaks, 43 repairs, McNemar
p = 2.1e-07, 95 % interval -1.0e-05 to -4.4e-06 (-1.3 % to -0.6 % of Cascade's rate). At q1 = 2, 5, 10 and 20 % the
p-values (1.5e-08 to 3.0e-06) are below 0.05 / 52 = 9.6e-04, a Bonferroni level over everything evaluated here.
Bonferroni covers the number of looks but not the choice of what to look at, which was made after the registered results
were known, so the 0.2 % result is only a lead for a registered test on new shots. In `uniform` at 0.2 %, S55 -> C14 is
level with Cascade (at best 5 fewer, p = 0.12).

The gain at 0.3 % is small because Cascade's failures fall on the shots S55'-deep is least sure of. With 10 % routed,
4,321 of Cascade's 4,459 failures (97 %) are on routed shots, where the pair takes Cascade's decision. If S55'-deep were
right on every kept shot, the pair would still fail 4,321 times, 3.1 % below Cascade. A 10 % gain (446 fewer failures)
is out of reach unless the pair routes 2 % or less, and at 2 % the current network breaks 278 kept shots and repairs 151
(section 5).

## 3. Method

The analysis uses phase C's confirmation reads, `cascade_w4/out/conf/<setting>/p*.{npz,json}` (seed kind 11), and the
development reads that the locks name, `cascade_w4/out/dev/<setting>/` (seed kind 10). Both are under
`<cluster-scratch>/` and were only read. Every `depspam_fm` confirmation read holds C14', S55', S55'-deep and
r12, and every `uniform` one holds C14, S55 and r12. The block sizes are the lock's `conf_chunks`: at 0.2 %, 5,000,000
shots in `depspam_fm` and 1,000,000 in `uniform`; at 0.3 %, 1,000,000 and 1,500,000.

The threshold tau1 follows `pc_route.py`'s rule on the development block: the q1-quantile of S's development margin,
with numpy's default interpolation. The locks store it for every S -> B candidate (`candidates`), with identical values
in both locks, which share one development read. The script takes tau1 from the lock, never fits it on the confirmation
block, and recomputes it from the development margins as a check. A shot goes to Cascade when S's margin is strictly
below tau1 (`pc_route.masks`), so on the confirmation block the routed fraction is close to q1 but not exactly q1.

Three sets of configurations were evaluated: the first lock's S -> B pick at each of the 10 p in both settings (20
configurations); S55'-deep -> C14' and S55' -> C14' in `depspam_fm` at q1 = 1, 2, 5, 10, 20 and 40 %, at 0.2 % and 0.3 %
(24); and S55 -> C14 in `uniform` at the same q1 and p (12). Four first-lock picks are also grid points, which leaves 52
distinct configurations. Amendment 1's 20 S -> B picks were recomputed as a check; 12 of them are not among the 52.

For each configuration and block, the script reports the pair's failures, the failures of Cascade alone and of S alone
on the same shots, the paired counts against Cascade, and the output of `bb.route.paired`, the function
`pc_route.confirm` uses. That function runs the exact two-sided McNemar test on the discordant shots and gives
Newcombe's 95 % interval (method 10) for the difference of block failure rates. On the kept shots the script counts
repairs, breaks and the shots both fail. In every row the shots only the pair fails are the breaks and the shots only
Cascade fails are the repairs, and the script checks this.

To size what a network would need, the script holds each configuration's kept shots fixed and computes three gains on
the confirmation block: the net gain (repairs added plus breaks removed) for fewer failures than Cascade; the repairs to
add, with breaks fixed, or the breaks to remove, with repairs fixed, for an exact McNemar p below 0.05 on a block of the
same size; and the net gain for 10 % fewer failures than Cascade, the alternative that phase C's sizing used
(`pc_common.DELTA_REL`). New repairs can only come from kept shots that both now fail, so the most any network could
gain at that q1 is those shots plus the current breaks. The table marks a 10 % gain beyond that as out of reach.

All 272 checks passed. Every recomputed tau1 equals the lock's. The development failures and routed fractions equal the
locks' records, and the development paired counts of every locked S -> B equal its `vs_B`. Amendment 1's 20 S -> B rows
on the confirmation block equal `confirm_<setting>.json` exactly in failures, Cascade alone, S alone, routed fraction,
kept repairs and regressions, paired counts, McNemar p and interval. In each read the networks have the lock's sha256,
precision, card and torch, and each block has the lock's size.

The analysis ran as one acpu job on Alpine (CJ66; 16 cores on node cpu-node-12; 29 s in all, of which the reads and
the statistics took 2.6 s). It opened each npz once and loaded only the arrays it used (S's margins and failures,
Cascade's failures), spreading the 40 reads over 16 processes. The code is this branch at f41476db, with `bb.route` from
ising-cascade-net 64b2e25d, the tree phase C deployed. It ran in phase C's container and venv and wrote only under
`<cluster-scratch>/cascade_explore/`.

## 4. Results

In the tables, † marks the first lock's pick and ‡ Amendment 1's (registered; recomputed here as a check). "Pair -
Cascade" is the difference in failures on the same shots. The difference column gives the pair's block failure rate
minus Cascade's, with Newcombe's 95 % interval. McNemar p is exact and two-sided, and nothing is adjusted for
multiplicity. "Dev" is the development block, on which the thresholds were set; "conf" is the confirmation block.

### 4.1 The first lock's S -> B at every p

| setting | p | S, q1 | dev: pair / Cascade | routed (conf) | conf: pair / Cascade (shots) | pair - Cascade | only pair / only Cascade (= kept breaks / repairs) | difference [95 %] | McNemar p | Amendment 1's pick: failures |
|---|---|---|---|---|---|---|---|---|---|---|
| `depspam_fm` | 0.1 % | S55', 2 % | 548 / 561 | 2.0 % | 567 / 574 (10,000,000) | -7 (-1.2 %) | 2 / 9 | -7.00e-07 [-1.5e-06, 4.0e-08] | 6.5e-02 | S55', 1 %: 567 |
| `depspam_fm` | 0.15 % | S55', 10 % | 737 / 750 | 10.0 % | 712 / 719 (3,000,000) | -7 (-1.0 %) | 3 / 10 | -2.33e-06 [-5.1e-06, 3.1e-07] | 9.2e-02 | S55', 1 %: 717 |
| `depspam_fm` | 0.2 % | S55'-deep, 2 % | 696 / 701 | 2.0 % | 3,822 / 3,858 (5,000,000) | -36 (-0.9 %) | 7 / 43 | -7.20e-06 [-1.0e-05, -4.4e-06] | 2.1e-07 | S55', 1 %: 3,896 |
| `depspam_fm` | 0.25 % | S55'-deep, 2 % | 936 / 945 | 2.0 % | 1,028 / 1,022 (500,000) | +6 (+0.6 %) | 18 / 12 | 1.20e-05 [-1.1e-05, 3.5e-05] | 0.36 | S55'-deep, 1 %: 1,062 |
| `depspam_fm` | 0.3 % | S55'-deep, 10 % | 2,297 / 2,305 | 10.0 % | 4,452 / 4,459 (1,000,000) | -7 (-0.2 %) | 4 / 11 | -7.00e-06 [-1.6e-05, 1.4e-06] | 0.12 | S55', 5 %: 4,550 |
| `depspam_fm` | 0.35 % | S55'-deep, 20 % | 2,903 / 2,910 | 19.9 % | 2,969 / 2,973 (300,000) | -4 (-0.1 %) | 1 / 5 | -1.33e-05 [-3.4e-05, 6.9e-06] | 0.22 | S55'-deep, 5 %: 3,005 |
| `depspam_fm` | 0.4 % | S55'-deep, 20 % | 5,745 / 5,752 | 20.0 % | 5,779 / 5,789 (300,000) | -10 (-0.2 %) | 12 / 22 | -3.33e-05 [-7.3e-05, 6.6e-06] | 0.12 | S55'-deep, 10 %: 5,836 |
| `depspam_fm` | 0.45 % | S55'-deep, 40 % | 10,667 / 10,677 | 40.0 % | 10,863 / 10,869 (300,000) | -6 (-0.1 %) | 3 / 9 | -2.00e-05 [-4.6e-05, 5.5e-06] | 0.15 | S55'-deep, 20 %: 10,892 |
| `depspam_fm` | 0.5 % | S55'-deep, 40 % | 18,714 / 18,716 | 40.0 % | 18,727 / 18,741 (300,000) | -14 (-0.1 %) | 12 / 26 | -4.67e-05 [-8.9e-05, -4.9e-06] | 3.4e-02 | S55', 40 %: 18,773 |
| `depspam_fm` | 0.55 % | S55'-deep, 40 % | 30,389 / 30,388 | 40.1 % | 30,881 / 30,861 (300,000) | +20 (+0.1 %) | 127 / 107 | 6.67e-05 [-3.4e-05, 1.7e-04] | 0.21 | S55'-deep, 40 %: 30,881 |
| `uniform` | 0.1 % | S55, 2 % | 236 / 238 | 2.0 % | 260 / 260 (10,000,000) | +0 (+0.0 %) | 2 / 2 | 0.0 [-5.5e-07, 5.5e-07] | 1.0 | S55, 1 %: 260 |
| `uniform` | 0.15 % | S55, 1 % | 361 / 364 | 1.0 % | 349 / 348 (3,000,000) | +1 (+0.3 %) | 1 / 0 | 3.33e-07 [-1.1e-06, 1.8e-06] | 1.0 | S55, 1 %: 349 |
| `uniform` | 0.2 % | S55, 2 % | 369 / 372 | 2.0 % | 355 / 360 (1,000,000) | -5 (-1.4 %) | 1 / 6 | -5.00e-06 [-1.2e-05, 1.3e-06] | 0.12 | S55, 1 %: 356 |
| `uniform` | 0.25 % | S55, 20 % | 496 / 496 | 19.9 % | 491 / 492 (500,000) | -1 (-0.2 %) | 0 / 1 | -2.00e-06 [-1.1e-05, 6.5e-06] | 1.0 | S55, 2 %: 506 |
| `uniform` | 0.3 % | S55, 40 % | 1,263 / 1,264 | 39.9 % | 3,715 / 3,716 (1,500,000) | -1 (-0.0 %) | 0 / 1 | -6.67e-07 [-3.5e-06, 2.2e-06] | 1.0 | S55, 5 %: 3,770 |
| `uniform` | 0.35 % | S55, 20 % | 1,611 / 1,615 | 20.0 % | 1,579 / 1,577 (300,000) | +2 (+0.1 %) | 5 / 3 | 6.67e-06 [-1.6e-05, 2.9e-05] | 0.73 | S55, 10 %: 1,579 |
| `uniform` | 0.4 % | S55, 40 % | 3,439 / 3,438 | 39.9 % | 3,326 / 3,328 (300,000) | -2 (-0.1 %) | 0 / 2 | -6.67e-06 [-2.2e-05, 8.8e-06] | 0.50 | S55, 20 %: 3,336 |
| `uniform` | 0.45 % | S55, 40 % | 6,317 / 6,318 | 40.0 % | 6,278 / 6,275 (300,000) | +3 (+0.0 %) | 11 / 8 | 1.00e-05 [-2.1e-05, 4.1e-05] | 0.65 | S55, 20 %: 6,328 |
| `uniform` | 0.5 % | S55, 40 % | 11,579 / 11,593 | 40.0 % | 11,358 / 11,352 (300,000) | +6 (+0.1 %) | 32 / 26 | 2.00e-05 [-3.1e-05, 7.1e-05] | 0.51 | S55, 40 %: 11,358 |
| `uniform` | 0.55 % | S55, 40 % | 18,970 / 18,944 | 40.2 % | 19,151 / 19,124 (300,000) | +27 (+0.1 %) | 82 / 55 | 9.00e-05 [1.3e-05, 1.7e-04] | 2.6e-02 | S55, 40 %: 19,151 |

Under the first lock, S -> B in `depspam_fm` would have used S55' at 0.1 % and 0.15 % and S55'-deep from 0.2 % up. On
the development block those picks were below Cascade at 9 of the 10 p, by 2 to 13 failures. On the confirmation block
they are below Cascade at 8 of the 10 p, by 4 to 36, and above it at 0.25 % (+6) and 0.55 % (+20). McNemar p is below
0.05 at 0.2 % (2.1e-07) and 0.5 % (0.034). In `uniform`, where the first lock routed 20-40 % at most p, the picks stay
within 6 failures of Cascade up to 0.5 % and are 27 above it at 0.55 % (p = 0.026). The last column gives Amendment 1's
pick on the same shots, from `confirm_<setting>.json`. This changes no registered verdict, because Amendment 1's lock is
the lock of record.

### 4.2 `depspam_fm` at 0.2 % and 0.3 %

0.2 % (5,000,000 shots):

| S | q1 | tau1 | dev: pair / Cascade (only pair / only Cascade) | routed (conf) | conf: pair / Cascade | pair - Cascade | only pair / only Cascade (= kept breaks / repairs) | kept shots both fail | difference [95 %] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| S55' alone | - | - | - | 0 % | 6,858 / 3,858 | +3,000 | 4,178 / 1,178 | - | - | - |
| S55'‡ | 1 % | 5.918 | 720 / 701 (32 / 13) | 1.00 % | 3,896 / 3,858 | +38 (+1.0 %) | 115 / 77 | 421 | 7.60e-06 [2.1e-06, 1.3e-05] | 7.4e-03 |
| S55' | 2 % | 7.413 | 711 / 701 (18 / 8) | 2.02 % | 3,850 / 3,858 | -8 (-0.2 %) | 47 / 55 | 260 | -1.60e-06 [-5.6e-06, 2.4e-06] | 0.49 |
| S55' | 5 % | 8.692 | 702 / 701 (8 / 7) | 5.02 % | 3,851 / 3,858 | -7 (-0.2 %) | 31 / 38 | 163 | -1.40e-06 [-4.8e-06, 1.9e-06] | 0.47 |
| S55' | 10 % | 9.507 | 704 / 701 (7 / 4) | 10.01 % | 3,843 / 3,858 | -15 (-0.4 %) | 20 / 35 | 115 | -3.00e-06 [-6.0e-06, -5.5e-09] | 5.8e-02 |
| S55' | 20 % | 10.325 | 703 / 701 (4 / 2) | 20.03 % | 3,839 / 3,858 | -19 (-0.5 %) | 8 / 27 | 62 | -3.80e-06 [-6.3e-06, -1.4e-06] | 1.9e-03 |
| S55' | 40 % | 11.225 | 700 / 701 (1 / 2) | 39.97 % | 3,846 / 3,858 | -12 (-0.3 %) | 6 / 18 | 32 | -2.40e-06 [-4.5e-06, -3.5e-07] | 2.3e-02 |
| S55'-deep alone | - | - | - | 0 % | 4,804 / 3,858 | +946 | 2,079 / 1,133 | - | - | - |
| S55'-deep | 1 % | 5.254 | 701 / 701 (8 / 8) | 1.00 % | 3,830 / 3,858 | -28 (-0.7 %) | 28 / 56 | 280 | -5.60e-06 [-9.3e-06, -1.9e-06] | 3.0e-03 |
| S55'-deep† | 2 % | 7.333 | 696 / 701 (1 / 6) | 2.00 % | 3,822 / 3,858 | -36 (-0.9 %) | 7 / 43 | 138 | -7.20e-06 [-1.0e-05, -4.4e-06] | 2.1e-07 |
| S55'-deep | 5 % | 9.594 | 696 / 701 (0 / 5) | 4.98 % | 3,827 / 3,858 | -31 (-0.8 %) | 2 / 33 | 59 | -6.20e-06 [-8.7e-06, -3.8e-06] | 3.7e-08 |
| S55'-deep | 10 % | 10.844 | 699 / 701 (0 / 2) | 9.96 % | 3,828 / 3,858 | -30 (-0.8 %) | 1 / 31 | 28 | -6.00e-06 [-8.4e-06, -3.7e-06] | 1.5e-08 |
| S55'-deep | 20 % | 11.910 | 699 / 701 (0 / 2) | 19.97 % | 3,836 / 3,858 | -22 (-0.6 %) | 1 / 23 | 14 | -4.40e-06 [-6.5e-06, -2.4e-06] | 3.0e-06 |
| S55'-deep | 40 % | 12.976 | 700 / 701 (0 / 1) | 39.98 % | 3,849 / 3,858 | -9 (-0.2 %) | 1 / 10 | 4 | -1.80e-06 [-3.3e-06, -3.0e-07] | 1.2e-02 |

0.3 % (1,000,000 shots):

| S | q1 | tau1 | dev: pair / Cascade (only pair / only Cascade) | routed (conf) | conf: pair / Cascade | pair - Cascade | only pair / only Cascade (= kept breaks / repairs) | kept shots both fail | difference [95 %] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| S55' alone | - | - | - | 0 % | 9,708 / 4,459 | +5,249 | 6,286 / 1,037 | - | - | - |
| S55' | 1 % | 1.353 | 3,297 / 2,305 (1265 / 273) | 1.02 % | 6,371 / 4,459 | +1,912 (+42.9 %) | 2,424 / 512 | 2,075 | 1.91e-03 [1.8e-03, 2.0e-03] | 8.8e-296 |
| S55' | 2 % | 2.806 | 2,692 / 2,305 (519 / 132) | 2.00 % | 5,183 / 4,459 | +724 (+16.2 %) | 967 / 243 | 1,299 | 7.24e-04 [6.6e-04, 7.9e-04] | 1.6e-102 |
| S55'‡ | 5 % | 5.697 | 2,343 / 2,305 (73 / 35) | 5.00 % | 4,550 / 4,459 | +91 (+2.0 %) | 155 / 64 | 522 | 9.10e-05 [6.2e-05, 1.2e-04] | 6.8e-10 |
| S55' | 10 % | 7.572 | 2,310 / 2,305 (20 / 15) | 10.00 % | 4,486 / 4,459 | +27 (+0.6 %) | 44 / 17 | 257 | 2.70e-05 [1.1e-05, 4.3e-05] | 7.3e-04 |
| S55' | 20 % | 8.975 | 2,302 / 2,305 (5 / 8) | 20.06 % | 4,461 / 4,459 | +2 (+0.0 %) | 14 / 12 | 138 | 2.00e-06 [-8.7e-06, 1.3e-05] | 0.85 |
| S55' | 40 % | 10.291 | 2,301 / 2,305 (2 / 6) | 40.06 % | 4,459 / 4,459 | +0 (+0.0 %) | 6 / 6 | 65 | 0.0 [-7.8e-06, 7.8e-06] | 1.0 |
| S55'-deep alone | - | - | - | 0 % | 6,650 / 4,459 | +2,191 | 3,261 / 1,070 | - | - | - |
| S55'-deep | 1 % | 1.713 | 2,559 / 2,305 (423 / 169) | 1.00 % | 4,948 / 4,459 | +489 (+11.0 %) | 833 / 344 | 1,738 | 4.89e-04 [4.2e-04, 5.6e-04] | 2.9e-47 |
| S55'-deep | 2 % | 3.098 | 2,392 / 2,305 (161 / 74) | 2.00 % | 4,586 / 4,459 | +127 (+2.8 %) | 278 / 151 | 1,017 | 1.27e-04 [8.6e-05, 1.7e-04] | 8.9e-10 |
| S55'-deep | 5 % | 5.506 | 2,298 / 2,305 (15 / 22) | 5.02 % | 4,460 / 4,459 | +1 (+0.0 %) | 30 / 29 | 348 | 1.00e-06 [-1.5e-05, 1.7e-05] | 1.0 |
| S55'-deep† | 10 % | 7.604 | 2,297 / 2,305 (1 / 9) | 10.03 % | 4,452 / 4,459 | -7 (-0.2 %) | 4 / 11 | 127 | -7.00e-06 [-1.6e-05, 1.4e-06] | 0.12 |
| S55'-deep | 20 % | 9.528 | 2,300 / 2,305 (0 / 5) | 20.03 % | 4,454 / 4,459 | -5 (-0.1 %) | 2 / 7 | 46 | -5.00e-06 [-1.2e-05, 2.0e-06] | 0.18 |
| S55'-deep | 40 % | 11.277 | 2,302 / 2,305 (0 / 3) | 39.97 % | 4,458 / 4,459 | -1 (-0.0 %) | 1 / 2 | 14 | -1.00e-06 [-6.1e-06, 4.1e-06] | 1.0 |

At 0.2 % the deep pair is below Cascade at every q1, and S55' -> C14' is below it from q1 = 2 % up, by 7 to 19. As more
is routed, the deep network's breaks drop fast (28 at 1 %, 7 at 2 %, 2 at 5 %, then 1) while its repairs fall from 56 to
10. At 0.3 % both counts are far higher per shot, and below 5 % routed the breaks outnumber the repairs: 833 against 344
at 1 % and 278 against 151 at 2 %. The deep pair reaches parity at 5 % (30 breaks, 29 repairs) and is a few shots ahead
from 10 % on (-7, -5, -1). At q1 = 2-10 %, S55' breaks 3.5 to 11 times as many kept shots as S55'-deep, and its pair
never gets ahead. Alone, S55'-deep fails 31 % less often than S55' at 0.3 % (6,650 against 9,708) and 30 % less at 0.2
%.

The development block shows the same shape at 0.3 %: the deep pair there was 7 below Cascade at 5 %, 8 below at 10 %, 5
below at 20 % and 3 below at 40 %, against 1 above, 7 below, 5 below and 1 below on the confirmation block.

### 4.3 `uniform` at 0.2 % and 0.3 %

0.2 % (1,000,000 shots):

| S | q1 | tau1 | dev: pair / Cascade (only pair / only Cascade) | routed (conf) | conf: pair / Cascade | pair - Cascade | only pair / only Cascade (= kept breaks / repairs) | kept shots both fail | difference [95 %] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| S55 alone | - | - | - | 0 % | 783 / 360 | +423 | 533 / 110 | - | - | - |
| S55‡ | 1 % | 7.053 | 371 / 372 (4 / 5) | 1.01 % | 356 / 360 | -4 (-1.1 %) | 4 / 8 | 38 | -4.00e-06 [-1.2e-05, 3.7e-06] | 0.39 |
| S55† | 2 % | 8.508 | 369 / 372 (1 / 4) | 2.03 % | 355 / 360 | -5 (-1.4 %) | 1 / 6 | 21 | -5.00e-06 [-1.2e-05, 1.3e-06] | 0.12 |
| S55 | 5 % | 9.960 | 369 / 372 (0 / 3) | 5.02 % | 357 / 360 | -3 (-0.8 %) | 1 / 4 | 15 | -3.00e-06 [-8.9e-06, 2.7e-06] | 0.38 |
| S55 | 10 % | 10.952 | 371 / 372 (0 / 1) | 10.01 % | 358 / 360 | -2 (-0.6 %) | 0 / 2 | 5 | -2.00e-06 [-6.8e-06, 2.7e-06] | 0.50 |
| S55 | 20 % | 11.928 | 372 / 372 (0 / 0) | 20.01 % | 359 / 360 | -1 (-0.3 %) | 0 / 1 | 3 | -1.00e-06 [-5.4e-06, 3.3e-06] | 1.0 |
| S55 | 40 % | 12.997 | 372 / 372 (0 / 0) | 40.04 % | 360 / 360 | +0 (+0.0 %) | 0 / 0 | 2 | 0.0 [-3.8e-06, 3.8e-06] | 1.0 |

0.3 % (1,500,000 shots):

| S | q1 | tau1 | dev: pair / Cascade (only pair / only Cascade) | routed (conf) | conf: pair / Cascade | pair - Cascade | only pair / only Cascade (= kept breaks / repairs) | kept shots both fail | difference [95 %] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| S55 alone | - | - | - | 0 % | 8,958 / 3,716 | +5,242 | 6,089 / 847 | - | - | - |
| S55 | 1 % | 2.236 | 1,609 / 1,264 (470 / 125) | 1.02 % | 4,749 / 3,716 | +1,033 (+27.8 %) | 1,356 / 323 | 1,381 | 6.89e-04 [6.4e-04, 7.4e-04] | 2.7e-150 |
| S55 | 2 % | 4.065 | 1,379 / 1,264 (162 / 47) | 2.01 % | 4,026 / 3,716 | +310 (+8.3 %) | 436 / 126 | 745 | 2.07e-04 [1.8e-04, 2.4e-04] | 5.8e-41 |
| S55‡ | 5 % | 6.915 | 1,276 / 1,264 (28 / 16) | 4.98 % | 3,770 / 3,716 | +54 (+1.5 %) | 82 / 28 | 280 | 3.60e-05 [2.2e-05, 5.0e-05] | 2.5e-07 |
| S55 | 10 % | 8.679 | 1,265 / 1,264 (8 / 7) | 9.99 % | 3,725 / 3,716 | +9 (+0.2 %) | 26 / 17 | 139 | 6.00e-06 [-2.9e-06, 1.5e-05] | 0.22 |
| S55 | 20 % | 10.211 | 1,264 / 1,264 (2 / 2) | 19.98 % | 3,721 / 3,716 | +5 (+0.1 %) | 11 / 6 | 65 | 3.33e-06 [-2.6e-06, 9.3e-06] | 0.33 |
| S55† | 40 % | 11.775 | 1,263 / 1,264 (0 / 1) | 39.93 % | 3,715 / 3,716 | -1 (-0.0 %) | 0 / 1 | 22 | -6.67e-07 [-3.5e-06, 2.2e-06] | 1.0 |

At 0.2 %, S55 -> C14 is at or below Cascade at every q1, by up to 5 failures, with p >= 0.12. At 0.3 % it is above
Cascade up to q1 = 20 % (+1,033 at 1 % down to +5 at 20 %) and 1 below at 40 %, where S55 has no breaks and 1 repair.
Phase C has no deeper small network in `uniform`.

## 5. What a deeper or longer-trained network would need at 0.3 %

The table takes each configuration on the confirmation block. Its fifth column counts Cascade's failures on the kept
shots, the only failures the pair can remove, and the sixth gives the pair's best case at that q1, which is Cascade's
failures on the routed shots. The last three columns give the kept-shot gains needed for fewer failures than Cascade,
for an exact McNemar p below 0.05 on a block of this size (one look), and for 10 % fewer failures than Cascade. The 0.2
% figures are in `explore/explore_deep_tables.md`.

| setting | S | q1 | kept breaks / repairs | Cascade's failures on kept shots (share of all) | the pair if S were right on every kept shot | net gain for parity | p < 0.05 on this block: repairs to add / breaks to remove | net gain for 10 % fewer failures than Cascade |
|---|---|---|---|---|---|---|---|---|
| `depspam_fm` | S55' | 1 % | 2,424 / 512 | 2,587 (58.0 %) | 1,872 (-58.0 %) | 1,913 | 2,052 / 1,974 | 2,358 |
| `depspam_fm` | S55' | 2 % | 967 / 243 | 1,542 (34.6 %) | 2,917 (-34.6 %) | 725 | 814 / 767 | 1,170 |
| `depspam_fm` | S55'‡ | 5 % | 155 / 64 | 586 (13.1 %) | 3,873 (-13.1 %) | 92 | 129 / 113 | 537 |
| `depspam_fm` | S55' | 10 % | 44 / 17 | 274 (6.1 %) | 4,185 (-6.1 %) | 28 | 49 / 38 | 473 (out of reach: at most 301) |
| `depspam_fm` | S55' | 20 % | 14 / 12 | 150 (3.4 %) | 4,309 (-3.4 %) | 3 | 16 / 11 | 448 (out of reach: at most 152) |
| `depspam_fm` | S55' | 40 % | 6 / 6 | 71 (1.6 %) | 4,388 (-1.6 %) | 1 | 11 / 6 | 446 (out of reach: at most 71) |
| `depspam_fm` | S55'-deep | 1 % | 833 / 344 | 2,082 (46.7 %) | 2,377 (-46.7 %) | 490 | 572 / 540 | 935 |
| `depspam_fm` | S55'-deep | 2 % | 278 / 151 | 1,168 (26.2 %) | 3,291 (-26.2 %) | 128 | 177 / 161 | 573 |
| `depspam_fm` | S55'-deep | 5 % | 30 / 29 | 377 (8.5 %) | 4,082 (-8.5 %) | 2 | 20 / 15 | 447 (out of reach: at most 378) |
| `depspam_fm` | S55'-deep† | 10 % | 4 / 11 | 138 (3.1 %) | 4,321 (-3.1 %) | 0 | 2 / 2 | 439 (out of reach: at most 131) |
| `depspam_fm` | S55'-deep | 20 % | 2 / 7 | 53 (1.2 %) | 4,406 (-1.2 %) | 0 | 3 / 2 | 441 (out of reach: at most 48) |
| `depspam_fm` | S55'-deep | 40 % | 1 / 2 | 16 (0.4 %) | 4,443 (-0.4 %) | 0 | 6 / - | 445 (out of reach: at most 15) |
| `uniform` | S55 | 1 % | 1,356 / 323 | 1,704 (45.9 %) | 2,012 (-45.9 %) | 1,034 | 1,139 / 1,082 | 1,405 |
| `uniform` | S55 | 2 % | 436 / 126 | 871 (23.4 %) | 2,845 (-23.4 %) | 311 | 371 / 341 | 682 |
| `uniform` | S55‡ | 5 % | 82 / 28 | 308 (8.3 %) | 3,408 (-8.3 %) | 55 | 83 / 68 | 426 (out of reach: at most 362) |
| `uniform` | S55 | 10 % | 26 / 17 | 156 (4.2 %) | 3,560 (-4.2 %) | 10 | 27 / 20 | 381 (out of reach: at most 165) |
| `uniform` | S55 | 20 % | 11 / 6 | 71 (1.9 %) | 3,645 (-1.9 %) | 6 | 18 / 11 | 377 (out of reach: at most 76) |
| `uniform` | S55† | 40 % | 0 / 1 | 23 (0.6 %) | 3,693 (-0.6 %) | 0 | 5 / - | 371 (out of reach: at most 22) |

In `depspam_fm` at 0.3 %, getting ahead of Cascade is cheap at 10-20 % routed, but there is little to gain there.
S55'-deep is already 7 ahead at 10 %, and two more repairs or two fewer breaks would bring McNemar p below 0.05 on
1,000,000 shots. Yet the pair can never be more than 3.1 % (at 10 %) or 1.2 % (at 20 %) below Cascade, because only 138
and 53 of Cascade's failures are on kept shots.

A worthwhile gain requires routing 2 % or less. Ranked by S55'-deep's margin, the kept shots hold 46.7 % of Cascade's
failures at 1 % routed and 26.2 % at 2 %, but only 8.5 % at 5 %, so a 10 % gain (446 fewer failures) is out of reach
from 5 % up. At 2 % it needs a net gain of 573 kept shots, for example no breaks (278 now) and 295 more repairs among
the 1,017 kept shots that both now fail. At 1 % it needs 935, for example no breaks (833 now) and 102 more repairs among
1,738.

The step from S55' to S55'-deep moved the pair at 2 % routed from 724 above Cascade to 127 above, a net gain of 597: it
removed 689 breaks (967 down to 278) and lost 92 repairs (243 down to 151). Another gain of that size would reach the 10
% mark at 2 %, but only 278 breaks are left to remove, so at least 295 of it would have to be new repairs: kept shots on
which Cascade fails and the network is right, with a margin high enough to keep them.

Alone, S55'-deep is right on 1,070 of Cascade's 4,459 failures at 0.3 % (24 %). But 1,059 of these are among the shots
it routes at q1 = 10 %, along with 3,257 shots on which it is wrong and Cascade is right. A deeper or longer-trained
network would need a margin that ranks its correct decisions on Cascade's failures above its own errors. S55'-deep makes
fewer errors of its own than S55', which brings its pair to Cascade's level at 5 % routed instead of 20 %.

In `uniform` at 0.3 %, S55 -> C14 at 2 % routed breaks 436 kept shots and repairs 126. Parity there needs a net gain of
311, and a 10 % gain needs 682 of at most 1,181.

These figures hold the kept shots fixed. A new network would rank the shots its own way, so the figures measure the size
of the change needed and do not predict what any particular network would achieve.

## 6. Conclusion

At 0.3 %, none of the S -> B pairs evaluated here beats Cascade on fresh shots. The deep pair with 10-20 % routed is
level with Cascade, 5 to 7 fewer failures out of 4,459, and its intervals put any advantage below about 0.35 % of
Cascade's failures. At those routed fractions a deeper or longer-trained network could gain at most 1-3 %, because
nearly all of Cascade's failures are routed back to Cascade. A 10 % gain would need routing of 2 % or less, and at 2 %
the network would have to stop breaking the kept shots Cascade gets right (278 now) and repair at least 295 more of the
kept shots on which Cascade fails. The one step in depth seen here, from S55' to S55'-deep, removed breaks on the kept
shots while the repairs there fell. At 0.2 % in `depspam_fm` the deep pair fails about 1 % less often than Cascade on
5,000,000 fresh shots; of the results here, that one would be the candidate for a registered test on new shots.

## 7. Limits

- The confirmation block was read for the registered analysis, and these results come from the same shots, after the
  registered results were known.
- 52 configurations were evaluated, and their p-values are not adjusted; 0.05 / 52 = 9.6e-04 is given for orientation
  only.
- Each first-lock pick was the minimum over its p's development candidates (12 in `depspam_fm`, 6 in `uniform`). That
  selection biases its development counts but not its confirmation counts. The grid rows are all candidates, unselected;
  picking the best grid row now would repeat the selection on the confirmation block.
- Section 5 holds the kept shots fixed (see its last paragraph).
- There are two depths in one setting. S55' and S55'-deep differ in depth and parameter count; nothing here tests longer
  training of either.
- As in phase C, Cascade was read in BF16 and the ConvLines networks on their fused engine. Speed is not considered
  here; the routed fraction sets the pair's latency (`RESULTS.md`, section 4).

## 8. Files

- `phase_c/explore/explore_deep.py`: the analysis. `explore_deep.sbatch` is its acpu job, and `ship.sh` ships the code
  and inputs to Alpine.
- `phase_c/explore/explore_deep.json`: every number above, per configuration (development and confirmation blocks, the
  needs) with the checks. `explore_deep_tables.md` holds the tables as generated, including the 0.2 % needs, and
  `explore_deep_CJ66.out` is the job's log.
- Inputs: the first lock's `locked_<setting>.json` from commit 1e8d1c6d, and Amendment 1's lock and the registered
  `confirm_<setting>.json` from this branch (`phase_c/out/`). The run directory on Alpine is
  `<cluster-scratch>/cascade_explore/run.20261011_133602/`.
