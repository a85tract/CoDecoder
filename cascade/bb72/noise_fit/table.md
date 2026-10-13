| point | model | p (%) | decoder | shots | failures | P_block | P_L (Eq. 1) | P_L 95 % interval | paper P_L | ours / paper | mean s/shot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | uniform | 0.2 | bposd | 12,000 | 235 | 1.96e-02 | 2.75e-04 | 2.42e-04 - 3.12e-04 | 3.14e-04 | 0.88 | 0.451 |
| 0 | uniform | 0.2 | bposd_sector | 1,000 | 3 | 3.00e-03 | 4.17e-05 | 1.42e-05 - 1.23e-04 | 3.14e-04 | 0.13 | 0.003 |
| 0 | uniform | 0.2 | relay_full | 19,000 | 20 | 1.05e-03 | 1.46e-05 | 9.47e-06 - 2.26e-05 | 2.00e-05 | 0.73 | 0.098 |
| 0 | uniform | 0.2 | relay_sector | 30,000 | 73 | 2.43e-03 | 3.38e-05 | 2.69e-05 - 4.25e-05 | 2.00e-05 | 1.69 | 0.003 |
| 0 | uniform | 0.2 | tesseract | 54,000 | 17 | 3.15e-04 | 4.37e-06 | 2.73e-06 - 7.00e-06 | 8.13e-06 | 0.54 | 0.321 |
| 0 | uniform | 0.2 | tesseract_sector | 1,000 | 3 | 3.00e-03 | 4.17e-05 | 1.42e-05 - 1.23e-04 | 8.13e-06 | 5.13 | 0.004 |
| 1 | uniform | 0.3 | bposd | 3,000 | 253 | 8.43e-02 | 1.23e-03 | 1.08e-03 - 1.39e-03 | 1.30e-03 | 0.94 | 0.534 |
| 1 | uniform | 0.3 | bposd_sector | 1,000 | 20 | 2.00e-02 | 2.81e-04 | 1.82e-04 - 4.33e-04 | 1.30e-03 | 0.22 | 0.006 |
| 1 | uniform | 0.3 | relay_full | 11,000 | 57 | 5.18e-03 | 7.22e-05 | 5.57e-05 - 9.35e-05 | 9.50e-05 | 0.76 | 0.169 |
| 1 | uniform | 0.3 | relay_sector | 10,000 | 141 | 1.41e-02 | 1.97e-04 | 1.67e-04 - 2.33e-04 | 9.50e-05 | 2.08 | 0.006 |
| 1 | uniform | 0.3 | tesseract | 10,000 | 20 | 2.00e-03 | 2.78e-05 | 1.80e-05 - 4.29e-05 | 4.73e-05 | 0.59 | 1.362 |
| 1 | uniform | 0.3 | tesseract_bfs | 2,000 | 5 | 2.50e-03 | 3.48e-05 | 1.48e-05 - 8.14e-05 | 4.73e-05 | 0.74 | 1.787 |
| 1 | uniform | 0.3 | tesseract_sector | 1,000 | 20 | 2.00e-02 | 2.81e-04 | 1.82e-04 - 4.33e-04 | 4.73e-05 | 5.94 | 0.012 |
| 2 | uniform_fm | 0.2 | bposd | 4,000 | 83 | 2.08e-02 | 2.91e-04 | 2.35e-04 - 3.61e-04 | 3.14e-04 | 0.93 | 0.831 |
| 3 | uniform_fm | 0.3 | bposd | 1,000 | 84 | 8.40e-02 | 1.22e-03 | 9.85e-04 - 1.51e-03 | 1.30e-03 | 0.94 | 1.007 |
| 6 | spam2_fm | 0.2 | bposd | 4,000 | 116 | 2.90e-02 | 4.09e-04 | 3.41e-04 - 4.91e-04 | 3.14e-04 | 1.30 | 0.762 |
| 6 | spam2_fm | 0.2 | relay_full | 20,000 | 40 | 2.00e-03 | 2.78e-05 | 2.04e-05 - 3.79e-05 | 2.00e-05 | 1.39 | 0.088 |
| 6 | spam2_fm | 0.2 | tesseract | 22,000 | 16 | 7.27e-04 | 1.01e-05 | 6.22e-06 - 1.64e-05 | 8.13e-06 | 1.24 | 0.501 |
| 7 | spam2_fm | 0.3 | bposd | 1,000 | 100 | 1.00e-01 | 1.47e-03 | 1.20e-03 - 1.78e-03 | 1.30e-03 | 1.13 | 1.025 |
| 7 | spam2_fm | 0.3 | relay_full | 5,000 | 46 | 9.20e-03 | 1.28e-04 | 9.63e-05 - 1.71e-04 | 9.50e-05 | 1.35 | 0.164 |
| 7 | spam2_fm | 0.3 | tesseract | 5,000 | 29 | 5.80e-03 | 8.08e-05 | 5.63e-05 - 1.16e-04 | 4.73e-05 | 1.71 | 2.096 |
| 10 | depspam_fm | 0.2 | bposd | 4,000 | 102 | 2.55e-02 | 3.59e-04 | 2.96e-04 - 4.36e-04 | 3.14e-04 | 1.14 | 0.794 |
| 10 | depspam_fm | 0.2 | relay_full | 20,000 | 25 | 1.25e-03 | 1.74e-05 | 1.18e-05 - 2.56e-05 | 2.00e-05 | 0.87 | 0.093 |
| 10 | depspam_fm | 0.2 | tesseract | 34,000 | 20 | 5.88e-04 | 8.17e-06 | 5.29e-06 - 1.26e-05 | 8.13e-06 | 1.01 | 0.432 |
| 11 | depspam_fm | 0.3 | bposd | 1,000 | 90 | 9.00e-02 | 1.31e-03 | 1.07e-03 - 1.61e-03 | 1.30e-03 | 1.01 | 1.115 |
| 11 | depspam_fm | 0.3 | relay_full | 5,000 | 41 | 8.20e-03 | 1.14e-04 | 8.43e-05 - 1.55e-04 | 9.50e-05 | 1.20 | 0.173 |
| 11 | depspam_fm | 0.3 | tesseract | 5,000 | 17 | 3.40e-03 | 4.73e-05 | 2.95e-05 - 7.58e-05 | 4.73e-05 | 1.00 | 2.003 |
paired point 0: bposd vs tesseract: 12 chunks, 12,000 shots, both fail 3, only bposd 232, only tesseract 3
paired point 0: bposd_sector vs relay_full: 1 chunks, 1,000 shots, both fail 2, only bposd_sector 1, only relay_full 1
paired point 0: bposd_sector vs relay_sector: 1 chunks, 1,000 shots, both fail 2, only bposd_sector 1, only relay_sector 0
paired point 0: bposd_sector vs tesseract_sector: 1 chunks, 1,000 shots, both fail 2, only bposd_sector 1, only tesseract_sector 1
paired point 0: relay_full vs relay_sector: 14 chunks, 14,000 shots, both fail 11, only relay_full 7, only relay_sector 16
paired point 0: relay_full vs tesseract: 1 chunks, 1,000 shots, both fail 0, only relay_full 1, only tesseract 0
paired point 0: relay_full vs tesseract_sector: 1 chunks, 1,000 shots, both fail 2, only relay_full 1, only tesseract_sector 1
paired point 0: relay_sector vs tesseract: 1 chunks, 1,000 shots, both fail 0, only relay_sector 1, only tesseract 0
paired point 0: relay_sector vs tesseract_sector: 1 chunks, 1,000 shots, both fail 2, only relay_sector 0, only tesseract_sector 1
paired point 1: bposd vs tesseract: 3 chunks, 3,000 shots, both fail 3, only bposd 250, only tesseract 2
paired point 1: bposd vs tesseract_bfs: 2 chunks, 2,000 shots, both fail 3, only bposd 159, only tesseract_bfs 2
paired point 1: bposd_sector vs relay_full: 1 chunks, 1,000 shots, both fail 5, only bposd_sector 15, only relay_full 2
paired point 1: bposd_sector vs relay_sector: 1 chunks, 1,000 shots, both fail 17, only bposd_sector 3, only relay_sector 3
paired point 1: bposd_sector vs tesseract_sector: 1 chunks, 1,000 shots, both fail 16, only bposd_sector 4, only tesseract_sector 4
paired point 1: relay_full vs relay_sector: 6 chunks, 6,000 shots, both fail 20, only relay_full 11, only relay_sector 61
paired point 1: relay_full vs tesseract_sector: 1 chunks, 1,000 shots, both fail 5, only relay_full 2, only tesseract_sector 15
paired point 1: relay_sector vs tesseract_sector: 1 chunks, 1,000 shots, both fail 18, only relay_sector 2, only tesseract_sector 2
paired point 1: tesseract vs tesseract_bfs: 2 chunks, 2,000 shots, both fail 3, only tesseract 1, only tesseract_bfs 2
paired point 6: bposd vs relay_full: 4 chunks, 4,000 shots, both fail 3, only bposd 113, only relay_full 5
paired point 7: bposd vs relay_full: 1 chunks, 1,000 shots, both fail 4, only bposd 96, only relay_full 2
paired point 10: bposd vs relay_full: 4 chunks, 4,000 shots, both fail 3, only bposd 99, only relay_full 4
paired point 11: bposd vs relay_full: 1 chunks, 1,000 shots, both fail 10, only bposd 80, only relay_full 3
