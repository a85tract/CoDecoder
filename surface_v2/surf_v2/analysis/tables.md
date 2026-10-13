# SURF-v2 analysis

Freeze commit f94f4edb843b4a0b4762c556d79782cf2a2167e2; test set sv2test; timing session counted: session 1 of 1 (qualified).

## The surface table (failures on the test set; BP-OSD on its subset)

| d | row network | q | pair | network alone | plain PM | correlated PM | Ising M1 + PM | Ising M4 + PM | Ising M1 + correlated PM | Ising M4 + correlated PM | min-LLR BP-OSD | BP 60 BP-OSD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | d7_b1w64_sv2 | 0.02 | 31,629 | 36,156 | 60,016 | 35,135 | 48,752 | 38,157 | 48,873 | 38,204 | 34,385 | 41,186 |
| 9 | d9_b3_sv2s1 | 0.00 | 10,417 | 10,417 | 27,481 | 12,967 | 22,908 | 17,400 | 22,856 | 17,379 | 16,409 | 19,959 |
| 11 | d11_b5_sv2 | 0.00 | 6,887 | 6,887 | 24,365 | 9,252 | 20,650 | 15,040 | 20,405 | 14,977 | 15,250 | 19,033 |
| 13 | d13_b5_sv2 | 0.00 | 5,318 | 5,318 | 21,086 | 6,344 | 17,491 | 12,440 | 17,101 | 12,298 | 13,405 | 17,791 |
| 15 | d15_b5_sv2 | 0.01 | 3,012 | 4,030 | 18,259 | 4,396 | 15,043 | 9,932 | 14,733 | 9,769 | 2,361 | 3,368 |
| 17 | d17_b5_sv2 | 0.01 | 2,180 | 3,296 | 15,726 | 2,973 | 13,149 | 8,083 | 12,764 | 7,918 | 2,171 | 3,046 |

## Families A and B (per d: 8 accuracy and 8 speed comparisons; the claim needs all 16)

| d | baseline | pair vs baseline failures | p | 97.5 % interval of the LER difference | A | pair us | registered us | B |
|---|---|---|---|---|---|---|---|---|
| 7 | plain PM | 31,629 vs 60,016 | 0 | [-0.0144, -0.0139] | better | 11.79 | 17.16 | holds |
| 7 | correlated PM | 31,629 vs 35,135 | 5.48e-152 | [-0.0019, -0.0016] | better | 11.79 | 35.71 | holds |
| 7 | Ising M1 + PM | 31,629 vs 48,752 | 0 | [-0.00878, -0.00834] | better | 11.79 | 48.76 | holds |
| 7 | Ising M4 + PM | 31,629 vs 38,157 | 2.4e-319 | [-0.00346, -0.00307] | better | 11.79 | 61.41 | holds |
| 7 | Ising M1 + correlated PM | 31,629 vs 48,873 | 0 | [-0.00884, -0.0084] | better | 11.79 | 53.34 | holds |
| 7 | Ising M4 + correlated PM | 31,629 vs 38,204 | 2.77e-322 | [-0.00348, -0.00309] | better | 11.79 | 65.07 | holds |
| 7 | min-LLR BP-OSD | 31,629 vs 34,385 | 5.66e-59 | [-0.00157, -0.00119] | better | 11.79 | 2,864.20 | holds |
| 7 | BP 60 BP-OSD | 31,629 vs 41,186 | 0 | [-0.00499, -0.00457] | better | 11.79 | 5,668.34 | holds |
| 7 | **claim** | | | | holds | | | holds; claim HOLDS |
| 9 | plain PM | 10,417 vs 27,481 | 0 | [-0.0174, -0.0167] | better | 29.05 | 36.99 | holds |
| 9 | correlated PM | 10,417 vs 12,967 | 1.49e-128 | [-0.00279, -0.00231] | better | 29.05 | 67.59 | holds |
| 9 | Ising M1 + PM | 10,417 vs 22,908 | 0 | [-0.0128, -0.0122] | better | 29.05 | 56.40 | holds |
| 9 | Ising M4 + PM | 10,417 vs 17,400 | 0 | [-0.00725, -0.00672] | better | 29.05 | 68.08 | holds |
| 9 | Ising M1 + correlated PM | 10,417 vs 22,856 | 0 | [-0.0128, -0.0121] | better | 29.05 | 68.45 | holds |
| 9 | Ising M4 + correlated PM | 10,417 vs 17,379 | 0 | [-0.00723, -0.0067] | better | 29.05 | 78.23 | holds |
| 9 | min-LLR BP-OSD | 10,417 vs 16,409 | 0 | [-0.00627, -0.00572] | better | 29.05 | 9,353.56 | holds |
| 9 | BP 60 BP-OSD | 10,417 vs 19,959 | 0 | [-0.00985, -0.00924] | better | 29.05 | 18,211.25 | holds |
| 9 | **claim** | | | | holds | | | holds; claim HOLDS |
| 11 | plain PM | 6,887 vs 24,365 | 0 | [-0.0178, -0.0171] | better | 41.64 | 72.02 | holds |
| 11 | correlated PM | 6,887 vs 9,252 | 2.07e-147 | [-0.00257, -0.00216] | better | 41.64 | 155.61 | holds |
| 11 | Ising M1 + PM | 6,887 vs 20,650 | 0 | [-0.0141, -0.0135] | better | 41.64 | 68.63 | holds |
| 11 | Ising M4 + PM | 6,887 vs 15,040 | 0 | [-0.00841, -0.0079] | better | 41.64 | 78.67 | holds |
| 11 | Ising M1 + correlated PM | 6,887 vs 20,405 | 0 | [-0.0138, -0.0132] | better | 41.64 | 97.05 | holds |
| 11 | Ising M4 + correlated PM | 6,887 vs 14,977 | 0 | [-0.00835, -0.00784] | better | 41.64 | 100.97 | holds |
| 11 | min-LLR BP-OSD | 6,887 vs 15,250 | 0 | [-0.00863, -0.00809] | better | 41.64 | 25,202.85 | holds |
| 11 | BP 60 BP-OSD | 6,887 vs 19,033 | 0 | [-0.0124, -0.0118] | better | 41.64 | 46,430.19 | holds |
| 11 | **claim** | | | | holds | | | holds; claim HOLDS |
| 13 | plain PM | 5,318 vs 21,086 | 0 | [-0.0161, -0.0155] | better | 49.91 | 129.61 | holds |
| 13 | correlated PM | 5,318 vs 6,344 | 4.78e-35 | [-0.00121, -0.00084] | better | 49.91 | 279.63 | holds |
| 13 | Ising M1 + PM | 5,318 vs 17,491 | 0 | [-0.0125, -0.0119] | better | 49.91 | 90.26 | holds |
| 13 | Ising M4 + PM | 5,318 vs 12,440 | 0 | [-0.00736, -0.00688] | better | 49.91 | 98.34 | holds |
| 13 | Ising M1 + correlated PM | 5,318 vs 17,101 | 0 | [-0.0121, -0.0115] | better | 49.91 | 144.02 | holds |
| 13 | Ising M4 + correlated PM | 5,318 vs 12,298 | 0 | [-0.00722, -0.00674] | better | 49.91 | 139.84 | holds |
| 13 | min-LLR BP-OSD | 5,318 vs 13,405 | 0 | [-0.00835, -0.00783] | better | 49.91 | 68,177.06 | holds |
| 13 | BP 60 BP-OSD | 5,318 vs 17,791 | 0 | [-0.0128, -0.0122] | better | 49.91 | 115,302.02 | holds |
| 13 | **claim** | | | | holds | | | holds; claim HOLDS |
| 15 | plain PM | 3,012 vs 18,259 | 0 | [-0.0155, -0.015] | better | 56.54 | 208.53 | holds |
| 15 | correlated PM | 3,012 vs 4,396 | 1.9e-172 | [-0.0015, -0.00127] | better | 56.54 | 464.95 | holds |
| 15 | Ising M1 + PM | 3,012 vs 15,043 | 0 | [-0.0123, -0.0118] | better | 56.54 | 122.85 | holds |
| 15 | Ising M4 + PM | 3,012 vs 9,932 | 0 | [-0.00714, -0.0067] | better | 56.54 | 129.63 | holds |
| 15 | Ising M1 + correlated PM | 3,012 vs 14,733 | 0 | [-0.012, -0.0115] | better | 56.54 | 215.90 | holds |
| 15 | Ising M4 + correlated PM | 3,012 vs 9,769 | 0 | [-0.00698, -0.00654] | better | 56.54 | 203.29 | holds |
| 15 | min-LLR BP-OSD | 586 vs 2,361 | 0 | [-0.00942, -0.00834] | better | 56.54 | 197,370.38 | holds |
| 15 | BP 60 BP-OSD | 586 vs 3,368 | 0 | [-0.0146, -0.0133] | better | 56.54 | 294,550.81 | holds |
| 15 | **claim** | | | | holds | | | holds; claim HOLDS |
| 17 | plain PM | 2,180 vs 15,726 | 0 | [-0.0138, -0.0133] | better | 76.77 | 325.23 | holds |
| 17 | correlated PM | 2,180 vs 2,973 | 1.24e-79 | [-0.000889, -0.000698] | better | 76.77 | 714.05 | holds |
| 17 | Ising M1 + PM | 2,180 vs 13,149 | 0 | [-0.0112, -0.0107] | better | 76.77 | 161.33 | holds |
| 17 | Ising M4 + PM | 2,180 vs 8,083 | 0 | [-0.0061, -0.00571] | better | 76.77 | 165.97 | holds |
| 17 | Ising M1 + correlated PM | 2,180 vs 12,764 | 0 | [-0.0108, -0.0103] | better | 76.77 | 297.51 | holds |
| 17 | Ising M4 + correlated PM | 2,180 vs 7,918 | 0 | [-0.00594, -0.00554] | better | 76.77 | 275.79 | holds |
| 17 | min-LLR BP-OSD | 439 vs 2,171 | 0 | [-0.00919, -0.00814] | better | 76.77 | 444,502.44 | holds |
| 17 | BP 60 BP-OSD | 439 vs 3,046 | 0 | [-0.0137, -0.0124] | better | 76.77 | 619,209.00 | holds |
| 17 | **claim** | | | | holds | | | holds; claim HOLDS |

## Family C (new against deployed; McNemar, Holm over all tests at 0.05; 95 % intervals pointwise)

| d | test | new vs deployed failures | p | Holm p | 95 % interval | verdict |
|---|---|---|---|---|---|---|
| 7 | network alone (q = 0) | 36,156 vs 43,317 | 0 | 0 | [-0.00374, -0.00342] | better |
| 7 | pair at its own q | 31,629 vs 33,471 | 3.14e-59 | 1.57e-58 | [-0.00103, -0.00081] | better |
| 9 | network alone (q = 0) | 10,417 vs 12,354 | 1.25e-126 | 9.96e-126 | [-0.0021, -0.00178] | better |
| 9 | pair at its own q | 10,417 vs 10,995 | 6.51e-14 | 1.95e-13 | [-0.000729, -0.000427] | better |
| 11 | network alone (q = 0) | 6,887 vs 8,565 | 6.51e-126 | 4.56e-125 | [-0.00182, -0.00154] | better |
| 11 | pair at its own q | 6,887 vs 7,288 | 2.62e-09 | 5.23e-09 | [-0.000533, -0.000269] | better |
| 13 | network alone (q = 0) | 5,318 vs 8,177 | 3.69e-304 | 3.32e-303 | [-0.00301, -0.00271] | better |
| 13 | pair at its own q | 5,318 vs 5,418 | 0.147 | 0.147 | [-0.000234, 3.4e-05] | inconclusive |
| 15 | network alone (q = 0) | 4,030 vs 8,424 | 0 | 0 | [-0.00456, -0.00423] | better |
| 15 | pair at its own q | 3,012 vs 3,927 | 3.86e-96 | 2.31e-95 | [-0.001, -0.000828] | better |
| 17 | network alone (q = 0) | 3,296 vs 8,661 | 0 | 0 | [-0.00554, -0.00519] | better |
| 17 | pair at its own q | 2,180 vs 2,773 | 5.81e-53 | 2.32e-52 | [-0.00067, -0.000517] | better |

## Composite times (us per shot; GPU term + CPU term on the Grace basis + handoff)

| d | network | role | GPU | CPU (Grace) | handoff | composite | registered |
|---|---|---|---|---|---|---|---|
| 11 | d11_b5_r777 | deployed | 40.547 | 2.216 | 1.100 | 43.863 | 43.925 |
| 11 | d11_b5_sv2 | new | 40.544 | 0.000 | 1.100 | 41.644 |  |
| 13 | d13_b5_r775 | deployed | 48.816 | 7.439 | 1.100 | 57.355 | 57.957 |
| 13 | d13_b5_sv2 | new | 48.810 | 0.000 | 1.100 | 49.910 |  |
| 15 | d15_b5_r782 | deployed | 49.562 | 32.621 | 1.100 | 83.283 | 86.080 |
| 15 | d15_b5_sv2 | new | 49.558 | 5.886 | 1.100 | 56.544 |  |
| 17 | d17_b5_r787_q08 | deployed | 66.708 | 64.906 | 1.100 | 132.714 | 135.019 |
| 17 | d17_b5_sv2 | new | 66.990 | 8.676 | 1.100 | 76.766 |  |
| 7 | d7_b1w64_r740 | deployed | 8.472 | 2.279 | 2.100 | 12.851 | 12.924 |
| 7 | d7_b1w64_sv2 | new | 8.466 | 1.111 | 2.216 | 11.793 |  |
| 9 | d9_b3_r764 | deployed | 27.957 | 1.026 | 1.100 | 30.083 | 30.049 |
| 9 | d9_b3_sv2s1 | new | 27.949 | 0.000 | 1.100 | 29.049 |  |
