# A1 stage-1 continuation rule v1 (2026-10-12), stage 1: bb72 -> **CONTINUE**

Validation set dev chunks 0-19 (200000 shots, digest f336bc912b61); training stream chain 43c10675c48735aa; margin 3 %, Holm over the 8 decisive tests, alpha 0.05.

| x | y | held | f | failures x | failures y | only x | only y | rel (y - x) / y | p | p Holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B-JT | B-PA | translate loss on | 0 % | 1392 | 5649 | 72 | 4329 | +75.4 % | 0 | 0 | win |
| B-JT | B-PA | translate loss on | 1 % | 433 | 3801 | 107 | 3475 | +88.6 % | 0 | 0 | win |
| B-JT | B-PA | translate loss on | 2 % | 192 | 2114 | 59 | 1981 | +90.9 % | 0 | 0 | win |
| B-JT | B-PA | translate loss on | 5 % | 51 | 136 | 16 | 101 | +62.5 % | 2.87e-16 | 2.87e-16 | win |
| B-J | B-P | no translate loss | 0 % | 1423 | 6792 | 89 | 5458 | +79.0 % | 0 | 0 | win |
| B-J | B-P | no translate loss | 1 % | 457 | 4971 | 106 | 4620 | +90.8 % | 0 | 0 | win |
| B-J | B-P | no translate loss | 2 % | 190 | 3293 | 50 | 3153 | +94.2 % | 0 | 0 | win |
| B-J | B-P | no translate loss | 5 % | 50 | 512 | 20 | 482 | +90.2 % | 4.61e-116 | 9.22e-116 | win |
| B-JT | B-J | joint: translate loss | 0 % | 1392 | 1423 | 387 | 418 | +2.2 % | 0.29 |  | descriptive |
| B-JT | B-J | joint: translate loss | 1 % | 433 | 457 | 172 | 196 | +5.3 % | 0.23 |  | descriptive |
| B-JT | B-J | joint: translate loss | 2 % | 192 | 190 | 78 | 76 | -1.1 % | 0.936 |  | descriptive |
| B-JT | B-J | joint: translate loss | 5 % | 51 | 50 | 25 | 24 | -2.0 % | 1 |  | descriptive |
| B-PA | B-P | pooled: translate loss | 0 % | 5649 | 6792 | 1513 | 2656 | +16.8 % | 7.05e-71 |  | descriptive |
| B-PA | B-P | pooled: translate loss | 1 % | 3801 | 4971 | 1846 | 3016 | +23.5 % | 1.1e-63 |  | descriptive |
| B-PA | B-P | pooled: translate loss | 2 % | 2114 | 3293 | 1310 | 2489 | +35.8 % | 9.51e-83 |  | descriptive |
| B-PA | B-P | pooled: translate loss | 5 % | 136 | 512 | 55 | 431 | +73.4 % | 2.19e-73 |  | descriptive |

- B-JT against B-PA (translate loss on): favours the line head; wins at f = [0.0, 1.0, 2.0, 5.0] %, losses at [] %
- B-J against B-P (no translate loss): favours the line head; wins at f = [0.0, 1.0, 2.0, 5.0] %, losses at [] %
