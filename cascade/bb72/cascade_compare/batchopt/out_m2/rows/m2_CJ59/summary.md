## Pipelines (slowest of the processes; † a process missed the clock rule; speed-up = Cascade alone's / the decoder's, same group of the same job)

| setting, p | decoder | S engine | tail | group | single mean (us) | per round | p99.9 | batched (us/shot) | per round | to B, to R | speed-up single | speed-up batched | the same, against Cascade alone with the tail rule |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| depspam_fm 0.002 | Cascade alone (config -1) | - | - | main | 1,762.7 † | 293.79 † | 1,778.7 † | 72.012 † | 12.002 † | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.002 | Cascade alone (config -1) | - | 512 | main | - | - | - | 71.661 | 11.943 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.002 | S -> B (config 2) | base | - | main | 105.8 | 17.63 | 1,802.0 | 5.324 | 0.887 | 1.02%, 0.00% | 16.67 † | 13.52 † | 13.46 |
| depspam_fm 0.002 | S -> B (config 2) | new | - | main | 83.7 | 13.95 | 1,780.1 | 3.058 | 0.510 | 1.02%, 0.00% | 21.06 † | 23.55 † | 23.43 |
| depspam_fm 0.002 | S -> B (config 2) | new | 512 | main | - | - | - | 2.619 | 0.437 | 1.02%, 0.00% | - | 27.49 † | 27.36 |
| depspam_fm 0.002 | S -> B -> R (config 3) | base | - | main | 256.6 † | 42.76 † | 1,931.8 † | 10.576 † | 1.763 † | 2.03%, 0.11% | 6.87 † | 6.81 † | 6.78 † |
| depspam_fm 0.002 | S -> B -> R (config 3) | new | - | main | 244.6 † | 40.77 † | 1,907.1 † | 9.299 † | 1.550 † | 2.03%, 0.11% | 7.21 † | 7.74 † | 7.71 † |
| depspam_fm 0.002 | S -> B -> R (config 3) | new | 512 | main | - | - | - | 9.382 † | 1.564 † | 2.03%, 0.11% | - | 7.68 † | 7.64 † |
| depspam_fm 0.002 | Cascade alone (config -1) | - | - | rerun | 1,762.9 † | 293.82 † | 1,781.1 † | 72.013 | 12.002 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.002 | Cascade alone (config -1) | - | 512 | rerun | - | - | - | 71.684 | 11.947 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.002 | S -> B (config 2) | base | - | rerun | 105.8 | 17.63 | 1,805.1 | 5.325 | 0.888 | 1.02%, 0.00% | 16.67 † | 13.52 | 13.46 |
| depspam_fm 0.002 | S -> B (config 2) | new | - | rerun | 83.7 | 13.95 | 1,779.5 | 3.059 | 0.510 | 1.02%, 0.00% | 21.06 † | 23.54 | 23.43 |
| depspam_fm 0.002 | S -> B (config 2) | new | 512 | rerun | - | - | - | 2.618 | 0.436 | 1.02%, 0.00% | - | 27.50 | 27.38 |
| depspam_fm 0.002 | S -> B -> R (config 3) | base | - | rerun | 256.7 † | 42.78 † | 1,911.6 † | 10.384 † | 1.731 † | 2.03%, 0.11% | 6.87 † | 6.94 † | 6.90 † |
| depspam_fm 0.002 | S -> B -> R (config 3) | new | - | rerun | 243.7 † | 40.62 † | 1,900.3 † | 9.381 † | 1.563 † | 2.03%, 0.11% | 7.23 † | 7.68 † | 7.64 † |
| depspam_fm 0.002 | S -> B -> R (config 3) | new | 512 | rerun | - | - | - | 9.718 † | 1.620 † | 2.03%, 0.11% | - | 7.41 † | 7.38 † |
| depspam_fm 0.002 | Cascade alone (config -1) | - | - | all | 1,762.9 † | 293.82 † | 1,781.1 † | 72.013 † | 12.002 † | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.002 | Cascade alone (config -1) | - | 512 | all | - | - | - | 71.684 | 11.947 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.002 | S -> B (config 2) | base | - | all | 105.8 | 17.63 | 1,805.1 | 5.325 | 0.888 | 1.02%, 0.00% | 16.67 † | 13.52 † | 13.46 |
| depspam_fm 0.002 | S -> B (config 2) | new | - | all | 83.7 | 13.95 | 1,780.1 | 3.059 | 0.510 | 1.02%, 0.00% | 21.06 † | 23.54 † | 23.43 |
| depspam_fm 0.002 | S -> B (config 2) | new | 512 | all | - | - | - | 2.619 | 0.437 | 1.02%, 0.00% | - | 27.49 † | 27.37 |
| depspam_fm 0.002 | S -> B -> R (config 3) | base | - | all | 256.7 † | 42.78 † | 1,931.8 † | 10.576 † | 1.763 † | 2.03%, 0.11% | 6.87 † | 6.81 † | 6.78 † |
| depspam_fm 0.002 | S -> B -> R (config 3) | new | - | all | 244.6 † | 40.77 † | 1,907.1 † | 9.381 † | 1.563 † | 2.03%, 0.11% | 7.21 † | 7.68 † | 7.64 † |
| depspam_fm 0.002 | S -> B -> R (config 3) | new | 512 | all | - | - | - | 9.718 † | 1.620 † | 2.03%, 0.11% | - | 7.41 † | 7.38 † |
| depspam_fm 0.003 | Cascade alone (config -1) | - | - | main | 1,763.0 † | 293.84 † | 1,780.4 † | 72.054 | 12.009 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.003 | Cascade alone (config -1) | - | 512 | main | - | - | - | 71.726 | 11.954 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.003 | S -> B (config 7) | base | - | main | 177.6 | 29.60 | 1,806.2 | 8.259 | 1.377 | 4.98%, 0.00% | 9.93 † | 8.72 | 8.68 |
| depspam_fm 0.003 | S -> B (config 7) | new | - | main | 155.6 | 25.93 | 1,781.5 | 5.991 | 0.999 | 4.98%, 0.00% | 11.33 † | 12.03 | 11.97 |
| depspam_fm 0.003 | S -> B (config 7) | new | 512 | main | - | - | - | 5.479 | 0.913 | 4.98%, 0.00% | - | 13.15 | 13.09 |
| depspam_fm 0.003 | S -> B -> R (config 8) | base | - | main | 881.2 † | 146.87 † | 127,054.7 † | 64.407 † | 10.734 † | 4.98%, 0.24% | 2.00 † | 1.12 † | 1.11 † |
| depspam_fm 0.003 | S -> B -> R (config 8) | new | - | main | 865.3 † | 144.21 † | 126,318.9 † | 64.968 † | 10.828 † | 4.98%, 0.24% | 2.04 † | 1.11 † | 1.10 † |
| depspam_fm 0.003 | S -> B -> R (config 8) | new | 512 | main | - | - | - | 61.510 † | 10.252 † | 4.98%, 0.24% | - | 1.17 † | 1.17 † |
| depspam_fm 0.003 | Cascade alone (config -1) | - | - | rerun | 1,763.8 † | 293.96 † | 1,778.7 † | 72.051 | 12.008 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.003 | Cascade alone (config -1) | - | 512 | rerun | - | - | - | 71.683 | 11.947 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.003 | S -> B (config 7) | base | - | rerun | 177.7 | 29.62 | 1,811.3 | 8.258 | 1.376 | 4.98%, 0.00% | 9.92 † | 8.73 | 8.68 |
| depspam_fm 0.003 | S -> B (config 7) | new | - | rerun | 155.5 | 25.92 | 1,781.5 | 5.990 | 0.998 | 4.98%, 0.00% | 11.34 † | 12.03 | 11.97 |
| depspam_fm 0.003 | S -> B (config 7) | new | 512 | rerun | - | - | - | 5.479 | 0.913 | 4.98%, 0.00% | - | 13.15 | 13.08 |
| depspam_fm 0.003 | S -> B -> R (config 8) | base | - | rerun | 877.3 † | 146.22 † | 125,605.0 † | 63.451 † | 10.575 † | 4.98%, 0.24% | 2.01 † | 1.14 † | 1.13 † |
| depspam_fm 0.003 | S -> B -> R (config 8) | new | - | rerun | 865.6 † | 144.26 † | 126,509.7 † | 64.003 † | 10.667 † | 4.98%, 0.24% | 2.04 † | 1.13 † | 1.12 † |
| depspam_fm 0.003 | S -> B -> R (config 8) | new | 512 | rerun | - | - | - | 62.929 † | 10.488 † | 4.98%, 0.24% | - | 1.14 † | 1.14 † |
| depspam_fm 0.003 | Cascade alone (config -1) | - | - | all | 1,763.8 † | 293.96 † | 1,780.4 † | 72.054 | 12.009 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.003 | Cascade alone (config -1) | - | 512 | all | - | - | - | 71.726 | 11.954 | 0.00%, 0.00% | - | - | - |
| depspam_fm 0.003 | S -> B (config 7) | base | - | all | 177.7 | 29.62 | 1,811.3 | 8.259 | 1.377 | 4.98%, 0.00% | 9.92 † | 8.72 | 8.68 |
| depspam_fm 0.003 | S -> B (config 7) | new | - | all | 155.6 | 25.93 | 1,781.5 | 5.991 | 0.999 | 4.98%, 0.00% | 11.34 † | 12.03 | 11.97 |
| depspam_fm 0.003 | S -> B (config 7) | new | 512 | all | - | - | - | 5.479 | 0.913 | 4.98%, 0.00% | - | 13.15 | 13.09 |
| depspam_fm 0.003 | S -> B -> R (config 8) | base | - | all | 881.2 † | 146.87 † | 127,054.7 † | 64.407 † | 10.734 † | 4.98%, 0.24% | 2.00 † | 1.12 † | 1.11 † |
| depspam_fm 0.003 | S -> B -> R (config 8) | new | - | all | 865.6 † | 144.26 † | 126,509.7 † | 64.968 † | 10.828 † | 4.98%, 0.24% | 2.04 † | 1.11 † | 1.10 † |
| depspam_fm 0.003 | S -> B -> R (config 8) | new | 512 | all | - | - | - | 62.929 † | 10.488 † | 4.98%, 0.24% | - | 1.14 † | 1.14 † |

Phase C's C7 (job CJ39), slowest of 3, per shot (per round = / 6): depspam_fm 0.002 config -1: single 1,696.5, batched 71.873 †; depspam_fm 0.002 config 2: single 105.5, batched 5.325; depspam_fm 0.002 config 3: single 254.3, batched 10.174; depspam_fm 0.003 config -1: single 1,700.8, batched 71.883; depspam_fm 0.003 config 7: single 178.2, batched 8.251; depspam_fm 0.003 config 8: single 882.4, batched 63.935

