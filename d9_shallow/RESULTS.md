# D9-SHALLOW registered test: results

This file reports the registered test of B1 (`d9_b1w64_sh`: 1 block, width 64, 90,467 parameters) at q = 0.06,
tau = 2.337686538696289, on 1,000,000 fresh test shots at d = 9, p = 0.6 % (plan: `PLAN.md`, freeze F 31300ee, commit
B 445e693). Abbreviations: PM = PyMatching 2.4.0; corr PM = PM with correlations; the pair = B1 with its routed shots
decoded by corr PM; pp = percentage points.

## Verdict

The claim holds. B1's pair fails less often than plain PM and than corr PM, with every 97.5 % interval below 0, and
its composite time on the Grace basis, 21.234 us, is below plain PM's 26.297 us and corr PM's 63.916 us. Under the
publication rule frozen in F, the paper's d = 9 row becomes B1's pair (`d9_b1w64_sh`, q = 0.06); the 3-block row
(`d9_b3_sv2s1`) stays in the provenance.

| criterion | result | holds |
|---|---|---|
| (1) composite below plain PM's 26.296678815736907 us | 21.234 us, 5.063 us below | yes |
| (2) fewer failures than plain PM | 11,978 against 27,028; p below double precision; 97.5 % interval (-1.540, -1.471) pp | yes |
| (3) fewer failures than corr PM | 11,978 against 12,802; p = 9.47e-39; 97.5 % interval (-0.0967, -0.0682) pp | yes |
| (3) composite below corr PM's 63.916141817786674 us | 21.234 us, 42.682 us below | yes |

The analysis (`results/analysis.json`, `results/table.md`, `results/fit_d9s.json`) ran twice on an Alpine CPU node
from commit B's tree; the two outputs were byte-identical. No evidence was missing: the pre-screen passed, all 5 GPU
processes of each network and the handoff were accepted at their first attempt, and the CPU session reached its end with
57 of 57 processes accepted.

## Failures on the 1,000,000 test shots

All four decoders read the same shots (manifest `testsets/d9s_test.json`, set digest b3628270...).

| decoder | failures | failure rate |
|---|---|---|
| B1's pair (q = 0.06; 59,700 test shots routed, 5.970 %) | 11,978 | 1.198 % |
| plain PM | 27,028 | 2.703 % |
| corr PM | 12,802 | 1.280 % |
| B1 alone (q = 0), descriptive | 18,305 | 1.831 % |
| `d9_b3_sv2s1` (q = 0), descriptive | 10,326 | 1.033 % |

Exact two-sided McNemar tests and Newcombe (method 8) intervals of the failure-rate difference (row minus opponent).
The 97.5 % intervals decide; the 95 % intervals are description only. "Only a / only b" counts the shots that only the
row's decoder, or only the opponent, gets wrong.

| comparison | role | only a / only b | p | difference (pp) | 97.5 % interval (pp) | 95 % interval (pp) |
|---|---|---|---|---|---|---|
| pair vs plain PM | claim: better | 4,426 / 19,476 | < 1e-300 | -1.5050 | (-1.5396, -1.4706) | (-1.5352, -1.4749) |
| pair vs corr PM | claim: better | 1,600 / 2,424 | 9.47e-39 | -0.0824 | (-0.0967, -0.0682) | (-0.0949, -0.0700) |
| pair vs `d9_b3_sv2s1` | descriptive | 5,430 / 3,778 | 1.08e-66 | +0.1652 | (+0.1437, +0.1867) | (+0.1464, +0.1840) |
| B1 alone vs plain PM | descriptive | 9,631 / 18,354 | < 1e-300 | -0.8723 | (-0.9098, -0.8349) | (-0.9051, -0.8396) |
| B1 alone vs corr PM | descriptive | 12,066 / 6,563 | < 1e-300 | +0.5503 | (+0.5198, +0.5809) | (+0.5236, +0.5771) |
| B1 alone vs `d9_b3_sv2s1` | descriptive | 10,826 / 2,847 | < 1e-300 | +0.7979 | (+0.7719, +0.8242) | (+0.7751, +0.8208) |
| `d9_b3_sv2s1` vs plain PM | descriptive | 3,994 / 20,696 | < 1e-300 | -1.6702 | (-1.7053, -1.6353) | (-1.7009, -1.6397) |
| `d9_b3_sv2s1` vs corr PM | descriptive | 4,449 / 6,925 | 4.44e-120 | -0.2476 | (-0.2715, -0.2237) | (-0.2685, -0.2267) |

("< 1e-300": below double precision; the analysis reports p = 0.) The pair gives up accuracy against the 3-block
network: it fails 16.0 % more often (11,978 against 10,326). This test makes no ranking claim of B1 over the exploratory
run's other arm B2.

## Timing and the composite

| term | value (us per shot) | how |
|---|---|---|
| a | 14.205 | slowest of B1's 5 accepted GPU processes: 14.199, 14.195, 14.205, 14.202, 14.199 (H100 80GB HBM3, `timing.gpu_term`, 200,000 timing shots) |
| b raw | 3.844 | slowest of 3 CPU processes: 3.8345, 3.8443, 3.8399 (corr PM on the 12,136 routed timing shots of the counted mask, 6.068 %, divided by all 200,000; AMD Ryzen 9 9950X) |
| Grace factor | 1.5423 | this session's fit: common factor (plain 1.4699, corr 1.5798; spreads 0.4189 and 0.1603; the two agree) |
| b, Grace basis | 5.929 | b raw x 1.5423 |
| c | 1.100 | max(1.1 us, the router's estimate 0.572 us) |
| composite, Grace basis | 21.234 | a + b + c; decides |
| composite, raw basis | 19.149 | a + b raw + c; beside, decides nothing |

- On the Grace basis the pair takes 1.23 times less time than plain PM (26.297 / 21.234, rounded down) and 3.01 times
  less than corr PM (63.916 / 21.234).
- On the raw basis (the workstation's own CPU time, no conversion) the composite, 19.149 us, lies above the re-test's
  raw plain PM, 17.418 us, and below its raw corr PM, 42.335 us. The speed claim therefore rests on the Grace basis, an
  estimate with no Grace hardware used; no deployment-latency claim is made.
- The composite sums terms from two machines: the GPU term and the handoff from the H100 pod (input already on the
  GPU), the CPU term from the Linux workstation's Ryzen 9 9950X converted to the Grace basis. Plain PM's and corr PM's
  times are the speed re-test's registered values (the fastest of three on the same workstation, converted with stage
  2's factor 1.5098) and were not timed again. This session's factor, 1.5423, is 2.2 % larger than stage 2's; a larger
  factor lengthens our composite.
- A routed shot cost 63.35 us of corr PM on the Ryzen (3.844 us x 200,000 / 12,136), 97.7 us on the Grace basis.
- Descriptive: `d9_b3_sv2s1`'s GPU term in the same session was 27.870 us (slowest of 5: 27.867, 27.866, 27.869,
  27.870, 27.865); at its registered q = 0 its composite is 28.970 us (a + the 1.1 us allowance).

### The Grace fit of this session (18 reference points, slowest of three each)

| decoder, d, p | ours (us per shot) | NVIDIA / ours | residual (ln) |
|---|---|---|---|
| corr, 5, 0.3 % | 3.824 | 1.504 | -0.025 |
| corr, 5, 0.6 % | 6.174 | 1.442 | -0.068 |
| corr, 9, 0.3 % | 17.232 | 1.750 | +0.126 |
| corr, 9, 0.6 % | 44.206 | 1.529 | -0.009 |
| corr, 13, 0.3 % | 69.894 | 1.700 | +0.097 |
| corr, 13, 0.6 % | 171.424 | 1.631 | +0.056 |
| corr, 17, 0.3 % | 182.264 | 2.250 | +0.378 |
| corr, 17, 0.6 % | 720.129 | 1.195 | -0.255 |
| corr, 21, 0.3 % | 621.527 | 1.681 | +0.086 |
| corr, 21, 0.6 % | 1,501.064 | 1.291 | -0.178 |
| corr, 31, 0.3 % | 2,437.946 | 1.695 | +0.095 |
| corr, 31, 0.6 % | 5,523.476 | 1.520 | -0.015 |
| plain, 13, 0.3 % | 22.912 | 1.918 | +0.218 |
| plain, 13, 0.6 % | 60.552 | 2.141 | +0.328 |
| plain, 21, 0.3 % | 119.073 | 2.365 | +0.428 |
| plain, 21, 0.6 % | 608.314 | 1.034 | -0.400 |
| plain, 31, 0.3 % | 989.105 | 0.902 | -0.536 |
| plain, 31, 0.6 % | 2,533.680 | 1.114 | -0.325 |

## How it ran (14 October 2026, PDT)

| step | when | what |
|---|---|---|
| freeze F | 03:17 | F 31300ee pushed; T (GitHub's activity record) 03:17:57 |
| beacon | 03:28 | drand round 6551422 (03:28:00), two or more agreeing endpoints; R = 49356145...fff2 |
| draw | 03:28 | on the pod's CPU (Xeon Platinum 8470, AVX2); SURF-v2's dry chunk drawn again equal to its manifest; test set digest b3628270..., timing set digest 65d37a53... |
| commit B | 03:29 | B 445e693 (beacon record, both manifests); public before any read or timing |
| pre-screen | 03:30 | passed: 5 of 5 GPU checks at 1980 of 1980 MHz, steal and quota passed; timed core 143 |
| GPU session | 03:31-03:33 | masks, 5 rounds of B1 and `d9_b3_sv2s1` interleaved, the handoff: 11 slots, every one accepted at its first attempt |
| reads | 03:33-03:37 | B1 and `d9_b3_sv2s1` (deployed engine, batch one), plain and corr PM (the pod's CPUs); everything pulled with sha256 and the pod released |
| CPU session | 04:06-04:10 | session reference 16.986 us per block (gate 19.04); 9 windows of up to 8 processes, every window passed at its first attempt; largest segment spread 1.0202 (limit 1.06), every reference value at most 1.025 times the session reference (limit 1.10); no block read by our processes |
| analysis | 04:11 | twice on an Alpine CPU node, byte-identical |

One pod (RunPod secure H100 80GB HBM3), about 18 minutes from its start to its release, about 1-2 USD.

## Departures and notes

1. The beacon record's free-text `push_source`, copied from the hand-off server's note, named an internal role that the
   public repository's residue scan refuses. The same round was fetched again, before commit B, with the text "run by the
   authors" (as the speed re-test's record words it). Round, signature, previous signature, randomness, the activity
   list and every seed are identical; only the fetch times and that text differ. The draw ran from the first record,
   whose seeds equal the published record's (the manifests name the round and the randomness only). The first record
   (sha256 4f3fb300653fd2c643c1db4ed14c8437638d542dbedcc8a695da0b5c48097e1b) is kept with the private run files. This
   departs from `args/pod.args`, which passes the server's text unchanged.
2. On the pod the draw (03:28) ran before the pre-screen (03:30), while `args/pod.args` lists the pre-screen first. Both
   ran before commit B was public and before anything was timed.
3. The pod's image had CUDA 12.4 only; nvcc 12.1.105 was installed from NVIDIA's apt repository before the setup, which
   uses it by default. The compiled kernel's sha256 equals the studies' (8f4180d7...). The beacon fetch ran with the
   pod's system Python 3.11.10 (standard library and SURF-v2's beacon module only).
4. For publication, as the re-test did: the logged-in account names in `records/cpu/cpu_session.json` are replaced by
   `<account>`, and the host name in `records/logs/box_D9.log` by a label. Their analysed sha256 are in
   `records/RECORDS.sha256` and in the analysis' input list. The raw window records of the CPU session, which list other
   accounts' processes, are kept with the private run files; `records/logs/box_D9.log` holds each window's verdict,
   reference values, spreads, disk reads and clocks. The shot chunks are not committed: `draw_d9s.py` regenerates them
   from the beacon record, and their sha256 are in the manifests and `records/RECORDS.sha256`.
5. The exploratory run's numbers are not pooled with these. For scale only, the exploratory run
   counted 12,087 failures for the pair on other shots, at an estimated 21.454 us that used stand-ins for b and c.

## Files

- `results/`: `analysis.json`, `fit_d9s.json`, `table.md` (sha256 8196e9e2..., bc99e3f4..., 9f757e0c...).
- `records/`: the GPU session (`pod/`: pre-screen, attempts, every process's record and watch window, masks, the
  hand-off), the reads (`reads/`), the CPU session's export (`cpu/`) and its plan (`cpu_plan.json`), the logs (`logs/`),
  the beacon fetch log (`beacon/`) and `RECORDS.sha256` (every analysed file, shot chunks included).
