# R17's d = 17 composite on NVIDIA's own Grace values: a correction note

Copy for this release, added 2026-10-11: the hub's correction note (second version) with the rules of `SANITIZE.txt`
applied. The cluster job's id became its label (J23), the hub's paths became this release's (`r17/` for
`ising_compare/r17/`), and the names of an internal plan and its review were removed, with the sentence that tied the
rule to that plan reworded. No number changed. Section 7's line numbers refer to release 8d65844a and paper 53ddc3bd;
the release commit that added this copy made the release changes that section 7 lists.

Second version, 2026-10-11. An earlier version put NVIDIA's d = 17 value into our CPU term only. This version converts
every CPU time of the d = 17 comparison, ours and every opponent's, by one rule at d = 17 (section 2), and keeps the
one-sided substitution only as the most adverse bound (section 5). The registered results stand as registered.
Nothing was re-run: every number below is arithmetic on recorded values (sources in section 8), and margins come from
unrounded terms.

## 1. What was wrong

The release (README, "Where every Grace number comes from", and `timing/grace_basis.csv`) and the paper (`main.tex`
line 484 at 53ddc3bd) say that NVIDIA timed PyMatching (PM) on one Grace core at p = 0.006 only at d = 13 (plain PM and
the PM share after each Ising model) and at d = 9 and 13 (correlated PM). NVIDIA's paper gives more. Its Table VII
lists plain PM and the PM share after models 1, 4 and 5 also at d = 21 and 31, and its Table IX lists correlated PM at
d = 5, 9, 13, 17, 21 and 31. All are batch-1 times per syndrome round on one Grace Neoverse-V2 core; a shot of d
rounds takes the per-round time x d. Per shot, at p = 0.006:

| µs per shot | d = 13 | d = 17 | d = 21 | d = 31 | used by the registered results |
|---|---|---|---|---|---|
| plain PM (Table VII) | 9.97 x 13 = 129.61 | not given | 29.95 x 21 = 628.95 | 91.06 x 31 = 2,822.86 | d = 13 |
| PM share after Ising M1, Ising-Fast (Table VII, model 1) | 3.05 x 13 = 39.65 | not given | 11.30 x 21 = 237.30 | 30.45 x 31 = 943.95 | d = 13 |
| PM share after Ising M4, Ising-Accurate (Table VII, model 4) | 2.55 x 13 = 33.15 | not given | 9.26 x 21 = 194.46 | 22.86 x 31 = 708.66 | d = 13 |
| correlated PM (Table IX) | 21.51 x 13 = 279.63 | 50.63 x 17 = 860.71 | 92.27 x 21 = 1,937.67 | 270.83 x 31 = 8,395.73 | d = 9 (67.59) and d = 13 |

R17's registered comparison converts both sides from the d = 13 anchors, so it is internally consistent. Our CPU term
is scaled by correlated PM's d = 17 Grace value, 714.05 µs, which extrapolates an EPYC 7713 node's 762.77 µs by that
node's d = 13 ratio 0.9361, and the opponents' PM, correlated PM and PM shares take their own d = 13 ratios
(`timing/grace_basis.csv`). On that basis ours takes 135.02 µs against 161.33 µs for Ising M1 + PM, a 26.31 µs margin.
NVIDIA's own d = 17 value of correlated PM, 860.71 µs, is 20.5 % above the extrapolation and above the release's band
for it (712.03-840.31 µs). The paper's Discussion (`main.tex` lines 759-760) says that a Grace value of 995 µs for
correlated PM at d = 17 would erase the 26.3 µs margin; NVIDIA measured 860.71 µs.

## 2. One rule for both sides at d = 17

If Grace runs PM relatively slower at d = 17 than the d = 13 ratios say, that holds for the opponents' PM as much as
for ours, and NVIDIA's d = 13 and 21 values do put the Ising PM shares at d = 17 above their d = 13 conversions. A
correction therefore converts every CPU time of the comparison, ours and every opponent's, by the same rule at the
same distance. We call the result the NVIDIA-distance basis:

1. Where NVIDIA timed the quantity at d = 17, its value: correlated PM, 860.71 µs (Table IX).
2. Otherwise the log-log interpolation in d between NVIDIA's measured d = 13 and d = 21 values, with
   w = ln(17/13) / ln(21/13) = 0.5594:

       plain PM             129.61 x (628.95 / 129.61)^w = 313.59 µs     registered 325.23 µs
       PM share after M1     39.65 x (237.30 /  39.65)^w = 107.87 µs     registered  98.73 µs
       PM share after M4     33.15 x (194.46 /  33.15)^w =  89.18 µs     registered  81.28 µs

3. A CPU time that NVIDIA did not take is converted by the ratio, at d = 17, of the matching quantity's Grace value
   from 1 or 2 to its time on the same CPU:

       (b), ours                        73.038 x 860.71 / 780.54  =  80.54 µs    registered x 714.05 / 780.54 = 66.82 µs
       correlated-PM share after M1    124.944 x 107.87 / 52.513  = 256.66 µs    registered x 1.880 = 234.91 µs
       correlated-PM share after M4    100.110 x  89.18 / 42.579  = 209.68 µs    registered x 1.909 = 191.11 µs

   73.038 µs is R17's term on its node (the largest of 3 repetitions) and 780.54 µs the same node's correlated PM per
   block (median of 3). The correlated-PM shares are study 4's Apple M4 core times. As registered (S4-R3's
   convention), each is converted by the PM-share ratio of the same Ising model, here the interpolated Grace share
   over the M4 core's d = 17 share (52.513 µs after M1, 42.579 µs after M4).

Interpolating each quantity's Grace / CPU ratio geometrically in ln d between NVIDIA's anchors instead gives the same
values when the CPU's times follow a power law in d between them; it would need d = 21 times on our CPUs, which were
not taken.

NVIDIA gives no anchor for three kinds of time:

- The correlated-PM share after Ising M1 or M4, at any d. Table IX's only correlated share follows NVIDIA's
  pre-decoder model 6 (8.68 µs per round at d = 17). Converting these shares by the plain-PM share's ratio stays an
  approximation, as in study 4.
- Our CPU term, since NVIDIA did not run our decoder. It takes correlated PM's ratio, R17's registered form.
- BP-OSD, at any d. Its times were taken at batch 1 on study 4's H100 session host (GPU BP and host OSD) and are not
  converted on either basis. They are about 3,000 and 4,200 times ours, so no conversion could change a cell.

Plain PM and the PM shares also have no NVIDIA value at d = 17 itself; rule 2 bridges d = 13 and 21.

Ours = 66.66 + 80.54 + 1.54 = 148.74 µs (registered 135.02 µs). The Ising plan times, 62.6 µs (M1) and 84.688 µs
(M4), are GPU times and do not change.

| opponent | time, registered (µs) | margin, registered (µs) | time, NVIDIA-distance basis (µs) | margin, NVIDIA-distance basis (µs) | speed cell |
|---|---|---|---|---|---|
| plain PM | 325.23 | 190.21 | 313.59 | 164.85 | holds |
| correlated PM | 714.05 | 579.03 | 860.71 | 711.97 | holds |
| Ising M1 + PM (Ising-Fast) | 161.33 | 26.31 | 170.47 | 21.73 | holds |
| Ising M4 + PM (Ising-Accurate) | 165.97 | 30.95 | 173.87 | 25.13 | holds |
| Ising M1 + correlated PM | 297.51 | 162.49 | 319.26 | 170.52 | holds |
| Ising M4 + correlated PM | 275.79 | 140.78 | 294.37 | 145.63 | holds |
| min-LLR BP-OSD | 444,502.44 | 444,367.43 | 444,502.44 (not converted) | 444,353.70 | holds |
| BP 60 BP-OSD | 619,209.00 | 619,073.98 | 619,209.00 (not converted) | 619,060.25 | holds |

All eight speed cells hold. The narrowest margin is 21.73 µs, over Ising M1 + PM, and 21.19 µs with R17's +0.54 µs
sensitivity on the GPU term (ours 149.28 µs). R17's accuracy result does not change.

## 3. Check: one factor for every CPU time measured on R17's node

As a check, one factor converts every CPU time that R17 measured on its node (AMD EPYC 7713, core 8, CPU job
J23): correlated PM's Grace / node ratio at d = 17, F = 860.71 / 780.54 = 1.1027. The job timed our term, plain
PM and correlated PM there, three repetitions each.

| time on R17's node | node (µs) | x F (µs) | margin to ours (µs) | NVIDIA-distance basis (µs), margin |
|---|---|---|---|---|
| ours, (b) | 73.038 (largest of 71.140, 69.182, 73.038) | 80.54, ours 148.74 | | 148.74 |
| plain PM per block | 298.95 (median of 300.36, 298.95, 297.18) | 329.65 | 180.91 | 313.59, 164.85 |
| correlated PM per block | 780.54 (median of 780.54, 785.61, 763.09) | 860.71 | 711.97 | 860.71, 711.97 |

R17's records hold no node time for the Ising PM shares or the correlated-PM shares, which study 4 timed on an Apple
M4 core, or for BP-OSD, which study 4 timed on its H100 host. F is a Grace / EPYC ratio and cannot convert those, so
the check leaves the four Ising cells and the two BP-OSD cells at their section 2 values. No conclusion differs. The
cells that F reaches hold, plain PM by more than on the NVIDIA-distance basis (180.91 against 164.85 µs), and the
narrowest margin stays 21.73 µs over Ising M1 + PM.

## 4. Study 4's d = 17 result under the same rule

The release still reports study 4's earlier d = 17 result at q = 10 %: composite 151.35 µs, 9.98 µs below Ising M1 +
PM, all 8 speed cells held (README lines 120-129 and 216-220 at 8d65844a). Its CPU term came from its own node session
(91.262 µs, the largest of 3 repetitions, on a node whose correlated PM took 783.78 µs, the median of 3):

    (b)  = 91.262 µs x 860.71 / 783.78 = 100.22 µs       registered: 91.262 x 714.05 / 783.78 = 83.14 µs
    ours = 66.66 + 100.22 + 1.54 = 168.42 µs             registered: 151.35 µs

| opponent | margin, registered (µs) | margin, NVIDIA-distance basis (µs) | margin, one-sided bound (µs) |
|---|---|---|---|
| plain PM | 173.88 | 145.17 | 156.81 |
| correlated PM | 562.70 | 692.29 | 545.63 |
| Ising M1 + PM | 9.98 | 2.05 | -7.09 |
| Ising M4 + PM | 14.62 | 5.45 | -2.45 |
| Ising M1 + correlated PM | 146.16 | 150.84 | 129.09 |
| Ising M4 + correlated PM | 124.45 | 125.95 | 107.37 |
| min-LLR BP-OSD | 444,351.10 | 444,334.02 | 444,334.02 |
| BP 60 BP-OSD | 619,057.65 | 619,040.57 | 619,040.57 |

Study 4's 8 speed cells hold on the registered basis and on the NVIDIA-distance basis, the two Ising + PM cells
narrowly there (2.05 and 5.45 µs). Only the one-sided bound of section 5 would lose those two. R17 replaced study 4's
d = 17 rows in the paper; the release keeps study 4's result and needs this note beside it. On the NVIDIA-distance
basis R17's composite is 19.68 µs below study 4's (registered: 16.33 µs).

## 5. The one-sided bound

Rebasing our side alone, with NVIDIA's 860.71 µs in our CPU term and every opponent left on its d = 13 conversion, is
one-sided. NVIDIA's own d = 13 and 21 values put the Ising PM shares at d = 17 9-10 % above those conversions (107.87
against 98.73 µs after M1, 89.18 against 81.28 µs after M4), so this bound compares a rebased ours with Ising shares
that the d = 13 conversion understates. It is reported only as the most adverse bound, never as the correction.

| opponent | time, registered (µs) | margin to ours at 148.74 µs |
|---|---|---|
| plain PM | 325.23 | 176.49 |
| correlated PM | 714.05 | 565.31 |
| Ising M1 + PM | 161.33 | 12.59 |
| Ising M4 + PM | 165.97 | 17.23 |
| Ising M1 + correlated PM | 297.51 | 148.77 |
| Ising M4 + correlated PM | 275.79 | 127.05 |
| min-LLR BP-OSD | 444,502.44 | 444,353.70 |
| BP 60 BP-OSD | 619,209.00 | 619,060.25 |

R17's 8 cells hold even here, the narrowest by 12.59 µs (12.05 µs with the +0.54 µs). Study 4's composite, 168.42 µs,
would be 7.09 µs slower than Ising M1 + PM and 2.45 µs slower than Ising M4 + PM, so under this bound two of study
4's 8 cells would not hold (section 4).

## 6. Other distances

d = 15: NVIDIA gives no value at d = 15, so the same rule interpolates on both sides. Correlated PM, log-log between
Table IX's d = 13 and 17 values (w = ln(15/13) / ln(17/13) = 0.5334), takes 509.38 µs (registered 464.95). Plain PM
and the shares, between Table VII's d = 13 and 21 values (w = ln(15/13) / ln(21/13) = 0.2984), take 207.65 µs
(registered 208.53), 67.63 µs after M1 (65.50) and 56.20 µs after M4 (54.92). Ours: (b) = 28.642 x 509.38 / 393.30
= 37.10 µs on its node, composite 89.32 µs (registered 86.08). Ising M1 + PM takes 57.348 + 67.63 = 124.97 µs
(registered 122.85) and M4 + PM 74.711 + 56.20 = 130.91 µs (129.63). The narrowest margin is 35.66 µs over Ising M1 +
PM (registered 36.77; 33.53 on the one-sided bound), plain PM's is 118.33 µs and correlated PM's 420.07 µs. No d = 15
cell changes.

d <= 13: the release uses NVIDIA's own values at d = 9 and 13 and converts d = 7 and 11 with d = 13 ratios. Table IX's
d = 5 value (1.78 x 5 = 8.90 µs) would allow interpolating correlated PM at d = 7 and 11; plain PM and the shares have
no NVIDIA value below d = 13. That was not examined here.

## 7. Where the release and the paper take this note

Line numbers refer to release 8d65844a and paper 53ddc3bd.

- release `README.md` lines 120-129 (the speed table's d = 17 row and the study 4 sentence below it): the row is
  registered; R17 on the NVIDIA-distance basis (section 2) and study 4 on it (section 4).
- release `README.md` lines 216-220 (study 4's narrowest speed margin, 9.98 µs): study 4 on the NVIDIA-distance basis
  and under the one-sided bound (sections 4 and 5).
- release `README.md` lines 289-294 (R17's margins): R17 on the NVIDIA-distance basis and the one-sided bound.
- release `README.md` lines 1083-1102 ("Where every Grace number comes from") and lines 1570-1571 ("Only the paper's
  three measurements are Grace numbers"): Tables VII and IX list more distances (section 1), and the NVIDIA-distance
  basis takes its d = 17 values from them (section 2).
- release `README.md` lines 1596-1599 (the narrowest d = 17 margin "depends on these conversions"): the margins on the
  NVIDIA-distance basis and under the one-sided bound.
- release `timing/grace_basis.csv`: the NVIDIA-distance basis's d = 17 values as new rows beside the registered
  extrapolations, under decoder names ending in "(NVIDIA-distance basis)", so that no code that reads the table by
  (d, decoder) picks them up; this note as `timing/R17_GRACE_NOTE.md`, listed in the README's repository layout.
- paper `main.tex` line 484 ("NVIDIA timed PM on one Grace core at d = 13, correlated PM at d = 9 and 13, and the PM
  share after each Ising model at d = 13"): plain PM and the shares were also timed at d = 21 and 31, correlated PM also
  at d = 5, 17, 21 and 31; and one sentence that at d = 17 every CPU time, ours and every opponent's, is also converted
  by one rule at d = 17.
- paper `main.tex` lines 66-67 (abstract), 554 (RQ1), 729 and 743-744 (the composite-time table's d = 17 row and its
  footnote) and 754-760 (Discussion, the 995 µs sentence): say which basis the 135 µs composite and the 26.3 µs margin
  are on, and add R17 on the NVIDIA-distance basis (148.7 against 170.5 µs, 21.7 µs) and NVIDIA's measured 860.71 µs.

## 8. Inputs

| value | source |
|---|---|
| Tables VII and IX | arXiv 2604.12841 v1, pages 21-22, read from the page images; PDF sha256 381378e6...924f80bd, equal to the copy fetched from arxiv.org on 2026-10-11; rechecked on 2026-10-11 against a text extraction of the paper (Table IX's second time column, "Corr PM after PD", is after model 6) |
| R17: (a) 66.6595 µs, term 73.0375 µs (the largest of 3 repetitions), the node's correlated PM 780.54 µs and plain PM 298.95 µs per block (medians of 3), (c) 1.5432 µs, the opponents' registered times | `r17/R17_RESULT.md` §3 and `r17/run/records/cpu_J23/` (`term_r17_q08_rep*.json`, `ref_d17_plain_rep*.json`, `ref_d17_correlated_rep*.json`) |
| study 4: (b) on the node 91.2616 µs, the node's correlated PM 783.7807 µs (median), composite 151.3451 µs; Ising plan times 62.6 µs (M1) and 84.688 µs (M4); the Apple M4 core's d = 17 shares, each its M4-basis speed minus the plan time: PM 52.513 / 42.579 µs and correlated PM 124.944 / 100.110 µs after M1 / M4 | release `results/results_accuracy_speed_d17.json`, 8d65844a |
| d = 15: composite 86.0803 µs, (b) on the node 28.6421 µs, the node's correlated PM 393.2982 µs (median), numerator 464.95 µs; Ising plan times 57.348 / 74.711 µs and PM shares 65.4981 / 54.9205 µs (M1 / M4) | release `results/results_accuracy_speed_d15.json` |
| the d = 13 anchors, the d = 17 extrapolations and their band | release `timing/grace_basis.csv` |
