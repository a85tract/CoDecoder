# A1 stage-1 continuation rule v1 (2026-10-12), ramp reruns: surface -> **CONTINUE**

Validation set val_legacy chunks 0-19 (200000 shots, digest 47fd63955a8e); training stream chain c33305dd41bcead8; margin 3 %, Holm over the 4 decisive tests, alpha 0.05.

| x | y | held | f | failures x | failures y | only x | only y | rel (y - x) / y | p | p Holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-VA | S-PA | line loss on | 0 % | 3034 | 3201 | 905 | 1072 | +5.2 % | 0.000187 | 0.000503 | win |
| S-VA | S-PA | line loss on | 1 % | 2160 | 2308 | 690 | 838 | +6.4 % | 0.000168 | 0.000503 | win |
| S-VA | S-PA | line loss on | 2 % | 1535 | 1678 | 493 | 636 | +8.5 % | 2.32e-05 | 9.29e-05 | win |
| S-VA | S-PA | line loss on | 5 % | 655 | 714 | 226 | 285 | +8.3 % | 0.0102 | 0.0102 | win |
| S-PA | S-P | pooled: line loss | 0 % | 3201 | 3223 | 1143 | 1165 | +0.7 % | 0.662 |  | descriptive |
| S-PA | S-P | pooled: line loss | 1 % | 2308 | 2377 | 863 | 932 | +2.9 % | 0.108 |  | descriptive |
| S-PA | S-P | pooled: line loss | 2 % | 1678 | 1724 | 641 | 687 | +2.7 % | 0.217 |  | descriptive |
| S-PA | S-P | pooled: line loss | 5 % | 714 | 748 | 285 | 319 | +4.5 % | 0.179 |  | descriptive |

- S-VA against S-PA (line loss on): favours the line head; wins at f = [0.0, 1.0, 2.0, 5.0] %, losses at [] %
