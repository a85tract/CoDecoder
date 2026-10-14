# D9-SHALLOW registered test: results (d9_shallow/analysis_d9s.py)

Claim (B1's pair at q = 0.06 against plain PM and correlated PM, alpha = 0.025): **holds**. Paper's d = 9 row: B1's pair (d9_b1w64_sh, q = 0.06).

| criterion | result |
|---|---|
| (1) composite < plain PM's | True |
| (2) fewer failures than plain PM | True |
| (3) fewer failures than correlated PM | True |
| (3) composite < correlated PM's | True |

| timing term | us per shot |
|---|---|
| a (H100, slowest of 5) | 14.205 |
| b raw (Ryzen 9 9950X, slowest of 3) | 3.844 |
| applied factor (corr) | 1.5423 |
| b on the Grace basis | 5.929 |
| c (max(1.1, router estimate)) | 1.100 |
| composite, Grace basis (decides) | 21.234 |
| composite, raw basis (beside) | 19.149 |
| targets: plain PM / correlated PM (Grace) | 26.296678815736907 / 63.916141817786674 |
| d9_b3_sv2s1: a, a + 1.1 (descriptive) | 27.870, 28.970 |

| decoder | failures on 1,000,000 test shots |
|---|---|
| pair | 11978 |
| b1 | 18305 |
| b3 | 10326 |
| pm | 27028 |
| cpm | 12802 |

| comparison | role | only a / only b | exact McNemar p | diff | 97.5 % interval | 95 % interval |
|---|---|---|---|---|---|---|
| pair vs plain PM | claim (better) | 4426 / 19476 | 0 | -0.015050 | (-0.015396, -0.014706) | (-0.015352, -0.014749) |
| pair vs correlated PM | claim (better) | 1600 / 2424 | 9.47e-39 | -0.000824 | (-0.000967, -0.000682) | (-0.000949, -0.000700) |
| pair vs d9_b3_sv2s1 | descriptive | 5430 / 3778 | 1.08e-66 | +0.001652 | (+0.001437, +0.001867) | (+0.001464, +0.001840) |
| B1 alone vs plain PM | descriptive | 9631 / 18354 | 0 | -0.008723 | (-0.009098, -0.008349) | (-0.009051, -0.008396) |
| B1 alone vs correlated PM | descriptive | 12066 / 6563 | 0 | +0.005503 | (+0.005198, +0.005809) | (+0.005236, +0.005771) |
| B1 alone vs d9_b3_sv2s1 | descriptive | 10826 / 2847 | 0 | +0.007979 | (+0.007719, +0.008242) | (+0.007751, +0.008208) |
| d9_b3_sv2s1 vs plain PM | descriptive | 3994 / 20696 | 0 | -0.016702 | (-0.017053, -0.016353) | (-0.017009, -0.016397) |
| d9_b3_sv2s1 vs correlated PM | descriptive | 4449 / 6925 | 4.44e-120 | -0.002476 | (-0.002715, -0.002237) | (-0.002685, -0.002267) |

Missing: nothing
