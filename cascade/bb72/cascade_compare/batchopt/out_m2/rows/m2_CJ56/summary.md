## Lever 2's checks

- depspam_fm (C14p, small graph compiled@512, full compiled@4096; replays 36.5 and 292.4 ms): bounds 32 of 32 pass; queue against the direct computation bitwise; tolerance against the read passes
  - p 0.002, config 2 (S-B): routed to B 1,059, tail [{'rows': 1059, 'graph_batch': 512, 'replays': 3}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 74/71 (only timed 0, only eval 3); routing to_B 1059/1059 (0, 0); pass True
  - p 0.002, config 3 (S-B-R): routed to B 2,080, tail [{'rows': 2080, 'graph_batch': 512, 'replays': 5}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 70/70 (only timed 0, only eval 0); routing to_B 2080/2080 (0, 0); to_R 119/118 (0, 1); pass True
  - p 0.003, config 7 (S-B): routed to B 4,967, tail [{'rows': 871, 'graph_batch': 512, 'replays': 2}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 448/450 (only timed 3, only eval 1); routing to_B 4967/4967 (0, 0); pass True
  - p 0.003, config 8 (S-B-R): routed to B 4,967, tail [{'rows': 871, 'graph_batch': 512, 'replays': 2}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 378/380 (only timed 2, only eval 0); routing to_B 4967/4967 (0, 0); to_R 237/238 (2, 1); pass True
- uniform (C14, small graph compiled@512, full compiled@4096; replays 36.5 and 292.7 ms): bounds 32 of 32 pass; queue against the direct computation bitwise; tolerance against the read passes
  - p 0.002, config 2 (S-B): routed to B 1,005, tail [{'rows': 1005, 'graph_batch': 512, 'replays': 2}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 38/39 (only timed 1, only eval 0); routing to_B 1005/1005 (0, 0); pass True
  - p 0.002, config 3 (S-B-R): routed to B 1,005, tail [{'rows': 1005, 'graph_batch': 512, 'replays': 2}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 58/58 (only timed 0, only eval 0); routing to_B 1005/1005 (0, 0); to_R 114/115 (1, 0); pass True
  - p 0.003, config 7 (S-B): routed to B 4,951, tail [{'rows': 855, 'graph_batch': 512, 'replays': 2}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 230/230 (only timed 0, only eval 0); routing to_B 4951/4951 (0, 0); pass True
  - p 0.003, config 8 (S-B-R): routed to B 4,951, tail [{'rows': 855, 'graph_batch': 512, 'replays': 2}], S bitwise True, queue differ 0, repeat differ 0; failures eval/timed 198/197 (only timed 1, only eval 2); routing to_B 4951/4951 (0, 0); to_R 103/101 (3, 5); pass True

