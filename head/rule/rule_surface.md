# A1 stage-1 continuation rule v1 (2026-10-12), stage 1: surface -> **RAMP_FIRST**

Validation set val_legacy chunks 0-19 (200000 shots, digest 47fd63955a8e); training stream chain f31c2ad513bae338; margin 3 %, Holm over the 8 decisive tests, alpha 0.05.

| x | y | held | f | failures x | failures y | only x | only y | rel (y - x) / y | p | p Holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-VA | S-PA | line loss on | 0 % | 3031 | 2982 | 1043 | 994 | -1.6 % | 0.288 | 0.458 | tie |
| S-VA | S-PA | line loss on | 1 % | 2144 | 2088 | 768 | 712 | -2.7 % | 0.153 | 0.458 | tie |
| S-VA | S-PA | line loss on | 2 % | 1535 | 1491 | 576 | 532 | -3.0 % | 0.196 | 0.458 | tie |
| S-VA | S-PA | line loss on | 5 % | 668 | 622 | 267 | 221 | -7.4 % | 0.0415 | 0.166 | tie |
| S-V | S-P | no line loss | 0 % | 2151 | 95797 | 1050 | 94696 | +97.8 % | 0 | 0 | win |
| S-V | S-P | no line loss | 1 % | 1337 | 94752 | 664 | 94079 | +98.6 % | 0 | 0 | win |
| S-V | S-P | no line loss | 2 % | 837 | 93754 | 427 | 93344 | +99.1 % | 0 | 0 | win |
| S-V | S-P | no line loss | 5 % | 279 | 90713 | 149 | 90583 | +99.7 % | 0 | 0 | win |
| S-VA | S-V | vote: line loss | 0 % | 3031 | 2151 | 1426 | 546 | -40.9 % | 2.86e-90 |  | descriptive |
| S-VA | S-V | vote: line loss | 1 % | 2144 | 1337 | 1148 | 341 | -60.4 % | 3.7e-102 |  | descriptive |
| S-VA | S-V | vote: line loss | 2 % | 1535 | 837 | 897 | 199 | -83.4 % | 3.03e-106 |  | descriptive |
| S-VA | S-V | vote: line loss | 5 % | 668 | 279 | 450 | 61 | -139.4 % | 2.66e-74 |  | descriptive |
| S-PA | S-P | pooled: line loss | 0 % | 2982 | 95797 | 1447 | 94262 | +96.9 % | 0 |  | descriptive |
| S-PA | S-P | pooled: line loss | 1 % | 2088 | 94752 | 1018 | 93682 | +97.8 % | 0 |  | descriptive |
| S-PA | S-P | pooled: line loss | 2 % | 1491 | 93754 | 752 | 93015 | +98.4 % | 0 |  | descriptive |
| S-PA | S-P | pooled: line loss | 5 % | 622 | 90713 | 341 | 90432 | +99.3 % | 0 |  | descriptive |

- S-VA against S-PA (line loss on): undecided (not trained: S-PA); wins at f = [] %, losses at [] %
- S-V against S-P (no line loss): undecided (not trained: S-P); wins at f = [0.0, 1.0, 2.0, 5.0] %, losses at [] %
