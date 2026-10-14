# SPEED-RT Decoder Switching stage

Host CPU: AMD Ryzen 9 9950X 16-Core Processor (container quota None, 24 CPUs allowed), timed cores picked per attempt, used: 0, 1, 2, 3, 4, 5, 6, 7, beacon round 6550668, deployed ea1525c739613f5bf256a98c15ac54c7cc66051d, session 1838 s. Host probe reference (plain PM per block, d = 9): 17.323 us per shot.

Statistic: the fastest of the term's three fresh processes (Decoder Switching is the baseline of this comparison; there is no 'our side' in this family; STAGE2_common C6). The slowest and the median of three are sensitivity rows. Every raw Linux workstation time (Ryzen basis) stands beside its Grace-basis value.

## The Grace fit (the stage's, as read from the fit file; this family times no reference points)

Fit file fit_s2.json (sha256 2269d469998dc558c7a9a61983ec6c8da902dc64c71f320ff59f94a1fa1be2e5).

| decoder | d | p | Ryzen time (the fit's: slowest of three) | NVIDIA (Grace) | ratio | residual (ln) |
|---|---|---|---|---|---|---|
| corr | 5 | 0.003 | 3.779 | 5.75 | 1.5216 | 0.0078 |
| corr | 5 | 0.006 | 6.151 | 8.90 | 1.4469 | -0.0425 |
| corr | 9 | 0.003 | 19.070 | 30.15 | 1.5810 | 0.0461 |
| corr | 9 | 0.006 | 44.270 | 67.59 | 1.5268 | 0.0112 |
| corr | 13 | 0.003 | 73.523 | 118.82 | 1.6161 | 0.0681 |
| corr | 13 | 0.006 | 180.864 | 279.63 | 1.5461 | 0.0238 |
| corr | 17 | 0.003 | 193.420 | 410.04 | 2.1199 | 0.3394 |
| corr | 17 | 0.006 | 738.039 | 860.71 | 1.1662 | -0.2582 |
| corr | 21 | 0.003 | 639.686 | 1044.75 | 1.6332 | 0.0786 |
| corr | 21 | 0.006 | 1,510.425 | 1937.67 | 1.2829 | -0.1629 |
| corr | 31 | 0.003 | 2,435.017 | 4132.61 | 1.6972 | 0.1170 |
| corr | 31 | 0.006 | 5,581.913 | 8395.73 | 1.5041 | -0.0038 |
| plain | 13 | 0.003 | 23.026 | 43.94 | 1.9083 | 0.2343 |
| plain | 13 | 0.006 | 61.369 | 129.61 | 2.1120 | 0.3357 |
| plain | 21 | 0.003 | 126.454 | 281.61 | 2.2270 | 0.3887 |
| plain | 21 | 0.006 | 570.458 | 628.95 | 1.1025 | -0.3143 |
| plain | 31 | 0.003 | 991.867 | 892.18 | 0.8995 | -0.5179 |
| plain | 31 | 0.006 | 2,656.207 | 2822.86 | 1.0627 | -0.3511 |

Factors (geometric means): plain 1.4543 (spread 0.4001), correlated 1.5383 (spread 0.1457), common 1.5098; applied: the common factor (plain 1.5098, correlated 1.5098, other 1.5098). Rule: common when |ln f_plain - ln f_corr| <= max(s_plain, s_corr); else plain and corr their own factor and other CPU decoders sqrt(f_plain x f_corr). This family converts the front end with the plain-PM factor and the routed term with the correlated-PM factor (grace_fit.to_grace).

## Gap switching, per distance (us per shot; each term the fastest of three repetitions; Ryzen = raw Linux workstation time)

| d | routed (timing shots) | front end, Ryzen | its plain-PM decode, Ryzen | front end / plain PM | routed PM, Ryzen | raw composite | front end, Grace | routed PM, Grace | composite, Grace | SURF-v2's earlier composite (EPYC 7713 basis) |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 15.62 % | 34.10 | 8.47 | 4.02 | 3.93 | 38.03 | 51.48 | 5.94 | 57.41 | 74.69 |
| 9 | 15.95 % | 83.37 | 18.69 | 4.46 | 8.69 | 92.06 | 125.86 | 13.12 | 138.98 | 146.11 |
| 11 | 16.08 % | 190.52 | 38.85 | 4.90 | 17.03 | 207.54 | 287.63 | 25.70 | 313.34 | 375.87 |
| 13 | 15.97 % | 394.74 | 69.36 | 5.69 | 29.00 | 423.74 | 595.96 | 43.78 | 639.74 | 723.76 |
| 15 | 16.07 % | 752.39 | 113.31 | 6.64 | 53.80 | 806.19 | 1,135.94 | 81.22 | 1,217.16 | 1,359.85 |
| 17 | 30.21 % | 1,439.87 | 172.84 | 8.33 | 151.32 | 1,591.19 | 2,173.85 | 228.46 | 2,402.31 | 2,709.58 |

## Sensitivity rows (deciding nothing; composites, us per shot)

| d | fastest, raw | fastest, Grace | slowest, raw | slowest, Grace | median, raw | median, Grace | fastest, front end plain and routed corr | fastest, front end with the correlated factor |
|---|---|---|---|---|---|---|---|---|
| 7 | 38.03 | 57.41 | 38.20 | 57.68 | 38.06 | 57.46 | 57.41 | 57.41 |
| 9 | 92.06 | 138.98 | 92.56 | 139.75 | 92.48 | 139.63 | 138.98 | 138.98 |
| 11 | 207.54 | 313.34 | 208.72 | 315.11 | 208.46 | 314.72 | 313.34 | 313.34 |
| 13 | 423.74 | 639.74 | 425.25 | 642.03 | 423.98 | 640.10 | 639.74 | 639.74 |
| 15 | 806.19 | 1,217.16 | 820.82 | 1,239.24 | 813.50 | 1,228.19 | 1,217.16 | 1,217.16 |
| 17 | 1,591.19 | 2,402.31 | 1,601.64 | 2,418.10 | 1,597.33 | 2,411.58 | 2,402.31 | 2,402.31 |

## Every repetition (us per shot; raw, then Grace basis)

| d | front end, raw | front end, Grace | routed PM, raw | routed PM, Grace |
|---|---|---|---|---|
| 7 | 34.25, 34.10, 34.12 | 51.71, 51.48, 51.52 | 3.95, 3.93, 3.93 | 5.96, 5.94, 5.94 |
| 9 | 83.85, 83.79, 83.37 | 126.59, 126.50, 125.86 | 8.69, 8.70, 8.71 | 13.12, 13.13, 13.16 |
| 11 | 191.29, 191.44, 190.52 | 288.80, 289.03, 287.63 | 17.28, 17.17, 17.03 | 26.08, 25.93, 25.70 |
| 13 | 394.74, 395.95, 394.93 | 595.96, 597.78, 596.25 | 29.00, 29.05, 29.31 | 43.78, 43.86, 44.25 |
| 15 | 766.21, 759.05, 752.39 | 1,156.79, 1,145.98, 1,135.94 | 54.46, 54.61, 53.80 | 82.21, 82.44, 81.22 |
| 17 | 1,444.13, 1,448.10, 1,439.87 | 2,180.29, 2,186.29, 2,173.85 | 153.54, 153.20, 151.32 | 231.81, 231.29, 228.46 |

Sampling error of the per-shot means (relative standard error, the larger of three repetitions; front end / routed term): d = 7: 0.122 % / 1.109 %; d = 9: 0.130 % / 1.079 %; d = 11: 0.139 % / 1.064 %; d = 13: 0.153 % / 1.057 %; d = 15: 0.170 % / 1.045 %; d = 17: 0.195 % / 0.695 %.

Front end over plain PM at d = 13, on the stage's reference point (the fit's raw time of plain PM, p = 0.6 %, the slowest of three of the reference session; not a process of this family's session): plain PM 61.37 us, front end 394.74 us (fastest of three; ratio 6.43) and 395.95 us (slowest of three; ratio 6.45).

## The rows

- GS-fit: On the host's AMD Ryzen 9 9950X 16-Core Processor, NVIDIA's Grace time over ours for plain PM is 1.4543 (spread 0.4001), for correlated PM 1.5383 (spread 0.1457); one common factor 1.5098 applies (the stage's fit, read from fit_s2.json). Gap switching's front end and routed term are converted as kind 'other' (factors 1.5098 and 1.5098).
- GS-7: On the AMD Ryzen 9 9950X 16-Core Processor, gap switching's front end took 34.10 us per shot at d = 7 (51.48 us on the Grace basis), correlated PM on its routed shots 3.93 us (5.94 us), and the composite 57.41 us (each term the fastest of three processes; SURF-v2's earlier composite on the EPYC 7713 basis: 74.69 us).
- GS-9: On the AMD Ryzen 9 9950X 16-Core Processor, gap switching's front end took 83.37 us per shot at d = 9 (125.86 us on the Grace basis), correlated PM on its routed shots 8.69 us (13.12 us), and the composite 138.98 us (each term the fastest of three processes; SURF-v2's earlier composite on the EPYC 7713 basis: 146.11 us).
- GS-11: On the AMD Ryzen 9 9950X 16-Core Processor, gap switching's front end took 190.52 us per shot at d = 11 (287.63 us on the Grace basis), correlated PM on its routed shots 17.03 us (25.70 us), and the composite 313.34 us (each term the fastest of three processes; SURF-v2's earlier composite on the EPYC 7713 basis: 375.87 us).
- GS-13: On the AMD Ryzen 9 9950X 16-Core Processor, gap switching's front end took 394.74 us per shot at d = 13 (595.96 us on the Grace basis), correlated PM on its routed shots 29.00 us (43.78 us), and the composite 639.74 us (each term the fastest of three processes; SURF-v2's earlier composite on the EPYC 7713 basis: 723.76 us).
- GS-15: On the AMD Ryzen 9 9950X 16-Core Processor, gap switching's front end took 752.39 us per shot at d = 15 (1135.94 us on the Grace basis), correlated PM on its routed shots 53.80 us (81.22 us), and the composite 1217.16 us (each term the fastest of three processes; SURF-v2's earlier composite on the EPYC 7713 basis: 1,359.85 us).
- GS-17: On the AMD Ryzen 9 9950X 16-Core Processor, gap switching's front end took 1439.87 us per shot at d = 17 (2173.85 us on the Grace basis), correlated PM on its routed shots 151.32 us (228.46 us), and the composite 2402.31 us (each term the fastest of three processes; SURF-v2's earlier composite on the EPYC 7713 basis: 2,709.58 us).

Slots not established: none. Attempts beyond the first: none.
