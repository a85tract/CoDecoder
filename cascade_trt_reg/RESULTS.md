# CASCADE-TRT-REG registered test: results

This file reports the registered speed re-test of the paper's Cascade comparisons with Cascade on the faster of two
engines (rules: `RULES.md`, freeze F 55a3464, commit B 3896ac6). Every comparison was timed on an H100 80GB HBM3 SXM
and on an RTX PRO 6000 Blackwell Server Edition, with Cascade alone on both engines and each pair on the engine with
the lower Cascade-alone time. Abbreviations: compiled = stage 1's registered path (`torch.compile`, BF16, CUDA graph);
TensorRT = a TensorRT 10.16.1.11 engine (BF16, builder level 3) in a CUDA graph; us = microseconds; single = batch 1
with the host round trip; batched = the registered batch sizes (BB72 4,096, d = 13 16).

## Outcome

On both cards, in both modes and at every p, every pair stays faster than Cascade alone on Cascade's faster engine.
The speed-ups are 2.38 to 8.16 on the H100 and 2.05 to 15.57 on the RTX PRO 6000; stage 1, with Cascade compiled,
measured 3.18 to 12.00 and 3.63 to 16.39.

TensorRT made Cascade alone faster in single mode on both cards. BB72's C14p took 478.7 to 482.1 us against 1474.4 to
1479.4 us compiled on the H100, and 371.5 to 374.9 us against 1725.8 to 1732.4 us on the RTX PRO 6000; d = 13's CS13
took 324.8 to 338.3 us against 425.2 to 428.1 us, and 278.6 to 282.1 us against 408.1 to 408.4 us. In batched mode
TensorRT was faster for CS13 (8.0 against 9.0 and 5.8 against 10.5 us per round) and slower for C14p at batch 4,096
(14.6 against 7.2 and 18.9 against 12.0 us per round). The batched BB72 comparisons therefore keep the compiled
Cascade, and their speed-ups stay close to stage 1's, except on the H100 at R2, p = 0.3 % (2.42 against 5.44; note 7).
The speed-ups fall most where TensorRT cut Cascade alone the most, in single mode: R2 at p = 0.2 % goes from 12.00 to
4.33 on the H100 and from 16.39 to 3.58 on the RTX PRO 6000.

The chosen engines are the same on both cards: C14p runs on TensorRT in single mode and on the compiled path in batched
mode, and CS13 runs on TensorRT in both modes.

The decision checks find no difference between the engines' failures beyond chance: at most 47 of 200,000 batched
decisions differ, the failure counts differ by at most 3 per comparison, and every exact McNemar p is at least 0.375.

## Results by card

Times in microseconds (us). Single: mean latency of one shot. Batched: time per round (the per-shot time divided by 6
for BB72 and by 13 for d = 13), as stage 1 reports it. Cascade alone is the fastest of 3 fresh processes and a pair the
slowest of 3. Speed-up = Cascade alone / pair, both on the chosen engine. R1: BB72, D1 harness, deep pairs (S55p_deep
-> C14p), S = the S network's engine (base = BB engine, new = multi-shot engine). R2: BB72, the first registered test's
pairs (S55p -> C14p). R4: d = 13 (ConvLines -> CS13). The stage 1 columns are the published stage-1 values on the same
card model (compiled Cascade). The pairs timed with the other engine decide nothing; they are in `results/tables.md`.

### H100 80GB HBM3 SXM

| Comparison | Cascade alone, compiled (fastest of 3) | Cascade alone, TensorRT (fastest of 3) | Chosen engine | Pair, chosen engine (slowest of 3) | Speed-up | Stage 1 alone | Stage 1 pair | Stage 1 speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 p=0.2 % single S=base (us) | 1476.0 | 479.3 | trt | 170.2 | 2.82 | 1484.6 | 189.4 | 7.84 |
| R1 p=0.2 % single S=new (us) | 1476.0 | 479.3 | trt | 134.3 | 3.57 | 1484.6 | 160.8 | 9.23 |
| R1 p=0.2 % batched S=base (us per round) | 7.2 | 14.6 | compiled | 1.7 | 4.23 | 7.4 | 1.7 | 4.32 |
| R1 p=0.2 % batched S=new (us per round) | 7.2 | 14.6 | compiled | 0.9 | 8.16 | 7.4 | 0.9 | 8.25 |
| R1 p=0.3 % single S=base (us) | 1474.4 | 478.7 | trt | 201.5 | 2.38 | 1484.2 | 316.3 | 4.69 |
| R1 p=0.3 % single S=new (us) | 1474.4 | 478.7 | trt | 173.2 | 2.76 | 1484.2 | 285.7 | 5.20 |
| R1 p=0.3 % batched S=base (us per round) | 7.2 | 14.6 | compiled | 2.3 | 3.15 | 7.4 | 2.3 | 3.18 |
| R1 p=0.3 % batched S=new (us per round) | 7.2 | 14.6 | compiled | 1.4 | 4.98 | 7.4 | 1.5 | 4.96 |
| R2 p=0.2 % single (us) | 1476.5 | 479.0 | trt | 110.6 | 4.33 | 1481.9 | 123.5 | 12.00 |
| R2 p=0.2 % batched (us per round) | 7.2 | 14.6 | compiled | 1.1 | 6.28 | 7.5 | 1.1 | 6.55 |
| R2 p=0.3 % single (us) | 1479.4 | 482.1 | trt | 133.2 | 3.62 | 1483.2 | 189.6 | 7.82 |
| R2 p=0.3 % batched (us per round) | 7.2 | 14.6 | compiled | 3.0 | 2.42 | 7.4 | 1.4 | 5.44 |
| R4 p=0.6 % single (us) | 425.2 | 324.8 | trt | 115.4 | 2.81 | 431.4 | 129.1 | 3.34 |
| R4 p=0.6 % batched (us per round) | 9.0 | 8.0 | trt | 1.9 | 4.10 | 9.2 | 2.1 | 4.38 |
| R4 p=0.35 % single (us) | 428.1 | 338.3 | trt | 83.9 | 4.03 | 433.0 | 83.0 | 5.22 |
| R4 p=0.35 % batched (us per round) | 9.0 | 8.0 | trt | 1.2 | 6.44 | 9.2 | 1.3 | 7.23 |

### RTX PRO 6000 Blackwell Server Edition (a rented pod; departures 1 to 3)

| Comparison | Cascade alone, compiled (fastest of 3) | Cascade alone, TensorRT (fastest of 3) | Chosen engine | Pair, chosen engine (slowest of 3) | Speed-up | Stage 1 alone | Stage 1 pair | Stage 1 speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 p=0.2 % single S=base (us) | 1729.3 | 371.5 | trt | 151.6 | 2.45 | 1723.0 | 168.1 | 10.25 |
| R1 p=0.2 % single S=new (us) | 1729.3 | 371.5 | trt | 120.9 | 3.07 | 1723.0 | 137.5 | 12.53 |
| R1 p=0.2 % batched S=base (us per round) | 12.0 | 18.9 | compiled | 1.4 | 8.63 | 12.0 | 1.4 | 8.86 |
| R1 p=0.2 % batched S=new (us per round) | 12.0 | 18.9 | compiled | 0.8 | 15.57 | 12.0 | 0.7 | 16.38 |
| R1 p=0.3 % single S=base (us) | 1725.8 | 372.5 | trt | 181.4 | 2.05 | 1724.8 | 296.6 | 5.81 |
| R1 p=0.3 % single S=new (us) | 1725.8 | 372.5 | trt | 151.1 | 2.47 | 1724.8 | 276.1 | 6.25 |
| R1 p=0.3 % batched S=base (us per round) | 12.0 | 18.9 | compiled | 2.5 | 4.89 | 12.0 | 2.3 | 5.15 |
| R1 p=0.3 % batched S=new (us per round) | 12.0 | 18.9 | compiled | 1.8 | 6.76 | 12.0 | 1.7 | 7.02 |
| R2 p=0.2 % single (us) | 1732.4 | 374.9 | trt | 104.8 | 3.58 | 1754.6 | 107.1 | 16.39 |
| R2 p=0.2 % batched (us per round) | 12.0 | 18.9 | compiled | 0.9 | 12.99 | 12.0 | 0.9 | 13.52 |
| R2 p=0.3 % single (us) | 1727.8 | 373.2 | trt | 118.7 | 3.15 | 1750.9 | 175.0 | 10.01 |
| R2 p=0.3 % batched (us per round) | 12.0 | 18.9 | compiled | 1.5 | 8.29 | 12.0 | 1.4 | 8.72 |
| R4 p=0.6 % single (us) | 408.4 | 278.6 | trt | 97.9 | 2.85 | 384.8 | 106.0 | 3.63 |
| R4 p=0.6 % batched (us per round) | 10.5 | 5.8 | trt | 1.9 | 3.08 | 10.5 | 2.3 | 4.47 |
| R4 p=0.35 % single (us) | 408.1 | 282.1 | trt | 69.9 | 4.03 | 397.3 | 69.5 | 5.71 |
| R4 p=0.35 % batched (us per round) | 10.5 | 5.8 | trt | 1.3 | 4.38 | 10.5 | 1.4 | 7.67 |

## Decision checks (reported, not a gate)

TensorRT's decisions and failures against the compiled path's on fresh beacon shots (chunks 10 to 19 of each Cascade's
two sets, seed kind 12): the batched engines on all 100,000 shots per p, the batch-1 engines on the first 20,000 of them
(`code/trt_check.py`'s `--n1` default). Exact McNemar p: two-sided binomial test on the shots exactly one engine fails.

| Card | Cascade | p | Path | Shots | Decisions differ | Agreement | Failures, compiled | Failures, TensorRT | Only compiled fails | Only TensorRT fails | Exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX | BB72 (C14p) | 0.2 % | batch 4096 | 100000 | 11 | 0.999890 | 63 | 64 | 1 | 2 | 1 |
| RTX | BB72 (C14p) | 0.3 % | batch 4096 | 100000 | 34 | 0.999660 | 448 | 448 | 4 | 4 | 1 |
| RTX | BB72 (C14p) | total | batch 4096 | 200000 | 45 | 0.999775 | 511 | 512 | 5 | 6 | 1 |
| RTX | BB72 (C14p) | 0.2 % | batch 1 | 20000 | 1 | 0.999950 | 9 | 10 | 0 | 1 | 1 |
| RTX | BB72 (C14p) | 0.3 % | batch 1 | 20000 | 9 | 0.999550 | 87 | 87 | 0 | 0 | 1 |
| RTX | BB72 (C14p) | total | batch 1 | 40000 | 10 | 0.999750 | 96 | 97 | 0 | 1 | 1 |
| RTX | d = 13 (CS13) | 0.35 % | batch 16 | 100000 | 0 | 1.000000 | 8 | 8 | 0 | 0 | 1 |
| RTX | d = 13 (CS13) | 0.6 % | batch 16 | 100000 | 11 | 0.999890 | 481 | 480 | 6 | 5 | 1 |
| RTX | d = 13 (CS13) | total | batch 16 | 200000 | 11 | 0.999945 | 489 | 488 | 6 | 5 | 1 |
| RTX | d = 13 (CS13) | 0.35 % | batch 1 | 20000 | 0 | 1.000000 | 2 | 2 | 0 | 0 | 1 |
| RTX | d = 13 (CS13) | 0.6 % | batch 1 | 20000 | 0 | 1.000000 | 90 | 90 | 0 | 0 | 1 |
| RTX | d = 13 (CS13) | total | batch 1 | 40000 | 0 | 1.000000 | 92 | 92 | 0 | 0 | 1 |
| H100 | BB72 (C14p) | 0.2 % | batch 4096 | 100000 | 11 | 0.999890 | 63 | 62 | 1 | 0 | 1 |
| H100 | BB72 (C14p) | 0.3 % | batch 4096 | 100000 | 36 | 0.999640 | 448 | 446 | 3 | 1 | 0.625 |
| H100 | BB72 (C14p) | total | batch 4096 | 200000 | 47 | 0.999765 | 511 | 508 | 4 | 1 | 0.375 |
| H100 | BB72 (C14p) | 0.2 % | batch 1 | 20000 | 0 | 1.000000 | 9 | 9 | 0 | 0 | 1 |
| H100 | BB72 (C14p) | 0.3 % | batch 1 | 20000 | 8 | 0.999600 | 87 | 87 | 0 | 0 | 1 |
| H100 | BB72 (C14p) | total | batch 1 | 40000 | 8 | 0.999800 | 96 | 96 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | 0.35 % | batch 16 | 100000 | 0 | 1.000000 | 8 | 8 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | 0.6 % | batch 16 | 100000 | 8 | 0.999920 | 481 | 481 | 4 | 4 | 1 |
| H100 | d = 13 (CS13) | total | batch 16 | 200000 | 8 | 0.999960 | 489 | 489 | 4 | 4 | 1 |
| H100 | d = 13 (CS13) | 0.35 % | batch 1 | 20000 | 0 | 1.000000 | 2 | 2 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | 0.6 % | batch 1 | 20000 | 0 | 1.000000 | 90 | 90 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | total | batch 1 | 40000 | 0 | 1.000000 | 92 | 92 | 0 | 0 | 1 |

## Engines

Built on each card before any timing, with the frozen settings: TensorRT 10.16.1.11, BF16, builder optimisation level
3, TensorRT's default workspace, and at d = 13 every convolution held to BF16. The ONNX exports were byte-identical on
the two cards; each card built its own plans.

| Card | Net | Batch | Convolutions held | ONNX sha256 | Plan sha256 |
| --- | --- | --- | --- | --- | --- |
| RTX | bb | 1 | False | f96da0f18f404376bdceb706428012dc4c19b7f7972f190388d2b87fc9920c8e | 784a89955607c7c473fc2ca9ed8bcc33ca86b1860de673724c61ebf2692f6d13 |
| RTX | bb | 4096 | False | a057e9e672762f8ba90e93ac2632a34b27dcde5c6548deadea555e1f2d8ec53a | 310146d9f968ba84390d0d35fabb860d00de188c7d515a2b94eaa2e3ca7cb0aa |
| RTX | d13 | 1 | True | 2ddbe1f3ad3910532b47acaf975b02e4729db0fa0d0a22597f54302bac6d080b | 3219e19264f9c34c790e6973c8f623832a7a927e6aceea2e45b75f36a53a2087 |
| RTX | d13 | 16 | True | 2a592c122a83a1b494a59a0f1b3249b28a9fe83775e64cbcb8c5aa141631c266 | e10b811c3aad9922ab77d4625bbd100008a589922404c2aed3bf2759e41bf73c |
| H100 | bb | 1 | False | f96da0f18f404376bdceb706428012dc4c19b7f7972f190388d2b87fc9920c8e | 7d38d5ad75f1d36fdd19b13d599c00f4e4524ce932fca84dee24cdc13c4e95b4 |
| H100 | bb | 4096 | False | a057e9e672762f8ba90e93ac2632a34b27dcde5c6548deadea555e1f2d8ec53a | 1a774165ba4a9067ffd91603e9887ec5b68e4959b4f718398278a0e02bda1698 |
| H100 | d13 | 1 | True | 2ddbe1f3ad3910532b47acaf975b02e4729db0fa0d0a22597f54302bac6d080b | fa12621a6985480d5e42ed45717736e03898ce4e93415465789c7d9613758683 |
| H100 | d13 | 16 | True | 2a592c122a83a1b494a59a0f1b3249b28a9fe83775e64cbcb8c5aa141631c266 | 4996aad8f5e058ed89f0c962494487f2d030f2276126ed6936f7fc3d4976b793 |

## How it ran (14 October 2026, PDT)

| step | when | what |
|---|---|---|
| freeze F | 10:39 | F 55a3464 pushed; T (GitHub's activity record) 10:39:38 |
| beacon | 10:50 | drand round 6552306 (10:50:00), two or more agreeing endpoints; R = 87e90520...3a43 |
| H100 pods | 10:43-11:19 | the first pod failed stage 1's pre-screen (CPU quota 17 of the 20 needed) and was replaced; the second (Xeon Platinum 8470, driver 580.126.09) passed it at 11:19: 5 of 5 GPU checks at 1980 of 1980 MHz, steal 0 %, quota 22.1 CPUs |
| commit B | 11:25 | B 3896ac6 (beacon record, seed manifest, run code); public 11:25:39 |
| H100 session | 11:25-12:47 | R1 72 slots (27.5 min), R2 48 (22.6 min), R4 48 (27.2 min), every slot established; 3 processes replaced (a clock-sampler gap above 1 s); decision check 12:43-12:47 |
| RTX pod 1 | 11:53-14:03 | rented after the cluster had no free card; session 13:17-13:58, stopped after 16 processes (departure 2) |
| RTX pod 2 | 13:37-15:40 | another host (AMD EPYC 9554, 256 CPUs visible, quota 27.2 CPUs, driver 580.159.04); engines built by 13:53; session 13:53-15:39: R4 48 slots (32.4 min), R1 72 (40.0 min), R2 48 (29.3 min), every slot established at its first attempt; decision check 15:35-15:39 |
| analysis | 15:54 | twice on the sanitized records on a cluster CPU node (Python 3.12.14 in the cluster's container), byte-identical, and once on the raw records with the same output |

## Departures and notes

1. The RTX PRO 6000 part did not run on the cluster's partition `artxpro6000`, which the rules name: the cluster had no
   free whole RTX PRO 6000 that day (the group quota of its normal GPU queue was full, and every whole card was busy
   under the long queue). The authors moved it to rented RunPod pods with an RTX PRO 6000 Blackwell Server Edition
   (secure cloud, driver 580.159.04). The pods ran the H100 pod's scripts with `CARD=rtx` through
   `run/ctrt_pod_rtx.sh`, the one run file written after B: it is `run/ctrt_pod.sh` with `CARD=rtx`, so
   `ctrt_session.sh` takes its RTX branch (the study's d = 13 lock, no H100 session record). As a result the cards
   were not timed at the same time (the RTX session began after the H100 session had ended), and the stage-1 RTX values
   beside the RTX table come from the cluster's RTX PRO 6000 nodes (driver 590.48.01).
2. The first RTX pod was abandoned after 16 timed processes (R4, rep 1, every one established). On its host about 85 %
   of the container's CPU time was spent in the kernel, and its numbers were far from the same card's on the cluster
   (rep 1, p = 0.6 %, compiled path: CS13 alone single 1081.7 us against stage 1's 384.7 us, the pair single 697.0 us
   against 106.0 us, both batched about 1.5 times slower). The authors rented a second pod on
   another host and fixed the rule before its first timed process: keep it if its single-shot numbers are within about
   1.3 times the cluster's. Its first processes gave 111.9 us (the pair) and 414.0 us (CS13 alone), and the first
   pod's session was stopped. The first pod's records are in `records/RTX_p32_abandoned/` and decide nothing.
3. On both RTX pods the pre-screen's first GPU check failed one criterion: the median SM clock under the check's load
   was 2310 (first pod) and 2325 (second pod) of 2430 MHz, 95.1 % and 95.7 %, against a threshold of 97 % that was set
   for the H100, whose clock holds at its maximum under load. Both checks showed no clock-event reason, the default
   power limit (600 W) with at most 429 W drawn, and no other process on the card; the CPU checks passed (steal 0 %,
   quota 24.65 and 27.2 CPUs). The authors decided to time on these pods. The pre-screen stops at its first failed
   check, so the other four checks did not run on the second pod (on the first, four more checks for the record gave
   medians of 2302 to 2317 MHz). Every timed window kept stage 1's window rule (SM clock at least 90 % of the maximum in
   every sample without the SW power cap).
4. The d = 13 harness on the public tree: stage 1's H100 lock derivation (`srt_d13.py h100-lock`) refuses the public
   tree, whose code digest differs from the study tree's. `run/ctrt_common.py` (in B) therefore reduces the d = 13
   identity check to the GPU and torch, and takes the kernel-cache digest from the lock. On the H100 the R4 processes
   ran under stage 1's published session record and derived lock (`speed_rt/cascade/records/H100/R4/`), on the RTX PRO
   6000 under the study's lock (`cascade/d13/out/locked_d13.json`). This departs from section 1.5's per-session lock.
5. No kernel-cache dry pass on the timing pods (the authors' decision at 11:25, after dry runs on the first H100 pod had
   run every code path on the shots of a dry beacon round). Each process compiled its kernels in its untimed warm-up.
6. The beacon record was fetched on the first H100 pod. In the published record the pod's host name in `fetched_by` is
   replaced by `gpu-pod`, and the free-text `push_source` says "run by the authors" in place of the hand-off server's
   wording. Round, signature, randomness and every seed are unchanged. The fetched record (sha256
   006e7c4dd6fd855c6368b5ead2ef583a3b8586477d30871e8a85ddd0eb5554ed) is kept with the private run files.
7. Notes. The RTX session ran R4 first (R4, R1, R2) and the H100 session R1, R2, R4; the rules fix no order. On the
   H100, R2 at p = 0.3 % batched, one of the three processes of the compiled-engine pair measured 17.9 us per shot and
   the other two 8.1 and 8.6, all three accepted by the window rule; the slowest of 3 makes that pair 3.0 us per round
   and its speed-up 2.42 against stage 1's 5.44 (the same pair with the TensorRT Cascade, which decides nothing,
   measured 1.8 us per round). `code/setup_pod.sh` installs `tensorrt-cu12>=10,<11`, which resolved to 10.16.1.11 on
   every pod; the build records name it.

## Costs

RunPod secure pods at 3.99 USD per hour (H100) and 2.49 USD per hour (RTX PRO 6000): the two H100 pods about 0.5 and
1.7 hours (about 8.6 USD), the two RTX PRO 6000 pods about 2.2 and 2.1 hours (about 10.5 USD), about 19 USD in all.
The cluster's CPU jobs cost nothing.

## Files

- `results/`: `numbers.json` and `tables.md` from `run/analysis_ctrt.py` (sha256
  ef92f55ca0b49e45a92f0a8838c199be7ac9b6a384aa06e120fcca5130fd2ecb and
  b477493be02c72be30f893c0a36756c475db8f1de11939593e3466227f3c85e2).
- `records/H100/`, `records/RTX/`: each card's pre-screen, the three rounds (`R1`, `R2`, `R4`: attempts, every
  process's rows, windows, traces and log), the decision check (`check/`), the build records (`build/`: `build.jsonl`,
  `plans.sha256`), the session log, and the setup's `host.txt` and `pip_freeze.txt`. `records/RTX_p32_abandoned/`: the
  first RTX pod's pre-screens, its 16 processes and its session log. `records/RECORDS.sha256` lists every file.
- For publication the pods' host names became `gpu-pod`, the pods' home directory `<pod-home>` and the three GPUs'
  UUIDs `<uuid-4>` to `<uuid-6>`. `records/H100/R4/derived_lock.json` and `session.json` are byte-identical copies of
  stage 1's published files. The analysis gives the same output on the raw records.
