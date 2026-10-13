# SURF-v2: the secondary comparison with gap switching (Decoder Switching)

This file registers, with the freeze commit, one more comparison of the SURF-v2 test: the ConvLines pair of each
distance against gap switching, Decoder Switching's construction (Toshio et al., arXiv:2510.25222). This gap switching
is built from this evaluation's decoders, plain PM as the front end and correlated PM as the fallback, and not from the
published system's decoders (its fallback is belief-matching). It is secondary and changes nothing in families A, B and
C of `REGISTRATION.md`, in a distance's claim, or in the paper's rows.

## Gap switching

| part | what it is |
|---|---|
| front end | plain PyMatching, the evaluation's plain PM (`convlines.circuits`: PyMatching 2.4.0 on the verdict circuit's DEM, `decompose_errors=True`, `approximate_disjoint_errors=True`); it decides every shot |
| reliability | the complementary gap: the least weight of a correction in the other logical class minus the weight of plain PM's correction, on plain PM's own matching graph and weights |
| fallback | correlated PyMatching, the evaluation's correlated PM and the ConvLines pairs' fallback; a shot whose gap is below τ takes its answer |

Only the front end differs from a ConvLines pair. The gap comes from a second decode, as the paper computes it: plain
PM's graph with the boundary split by the logical class (the boundary edges that flip the observable go to one extra
node), decoded with that node set so that the correction must lie in the other class (`surf_v2/gapsw.py`). The
front end therefore runs two matchings per shot. The code was checked against an exhaustive search at d = 3, 5 and 7
(`surf_v2/gapsw_validate.py`), and on every calibration shot the split graph's least weight in plain PM's class equals
plain PM's weight bit for bit.

## Its routed fraction, fixed before the freeze

The rule is C4.2 (`surf_v2/qrule.py`) with the gap in place of |c| and plain PM's answer in place of c > 0, on the
calibration set `sv2cal` (200,000 shots per d). It uses the same grid, targets M, limit M - 2√(2M) and fallback as the
networks' rule (`config.json` qrule). For every q of the grid, τ_q is the q-quantile of the gap (τ_0 = 0), and a shot is
routed when its gap is below τ_q. The projection uses the front end's own times, measured one shot at a time on core 8
of an exclusive EPYC 7713 node and converted to the Grace basis as the CPU term converts them. projected(0) is plain
PM alone, and projected(q) is the front end plus the routed fraction times correlated PM's mean Grace-basis time on the
routed calibration shots (`surf_v2/calcost.py`).

As registered, with the speed bar, the rule admits no routed pair at any d. The bars lie below plain PM's registered
time at every d (15.16 against 17.16 µs at d = 7). The front end alone measured 3.9 to 9.4 times plain PM's time, so
every q above 0 projects to 4 to 9 times the bar. The registered outcome is plain PM alone: at q = 0 where plain PM's
measured time meets the bar (d = 9, 15 and 17), and "no admissible q" elsewhere. A comparison of plain PM with the
ConvLines pair is already family A's. The comparison therefore uses the same rule without the bar, so that every q of
the grid is admissible. No q reaches the limit, so the rule takes the q with the fewest calibration failures (ties to
the smaller q). That is the best choice on the registered grid. Each policy holds both outcomes. The
largest admissible q, the fallback of the deployed d = 7 policy, gives the same q at every d, with the bar and without
it. The analysis recomputes both choices from each policy's table under `config.json`'s rule at that distance and
refuses a policy whose choices differ.

| d | q | τ | routed of 200,000 | calibration failures | correlated PM alone | the registered rule (with the bar) | policy sha256 |
|---|---|---|---|---|---|---|---|
| 7 | 16 % | 4.1978906897222 | 31,996 | 3,501 | 3,497 | no admissible q | `5219dc5aff5f80f6ee4655b789c86d55e2c3bc3c5be95941f731d67affb820d6` |
| 9 | 16 % | 4.610442494875592 | 32,000 | 2,625 | 2,562 | q = 0, plain PM alone | `cb3c8ff70c8ad0865f4e454f15fe0a7efb7b5c8b7f5f2d6e38c0944a3c3a9f47` |
| 11 | 16 % | 5.0630521115234615 | 31,999 | 1,835 | 1,792 | no admissible q | `f9e1a874acfe6cb8c6b0b6ae3abfb2f011ecb109ce47913809b0e7f73ababff5` |
| 13 | 16 % | 5.522469286402384 | 31,999 | 1,376 | 1,325 | no admissible q | `f340ce5250090a1a64abce811bc090f0e733695dcea916883abf728aacc39571` |
| 15 | 16 % | 6.002344052429958 | 32,000 | 961 | 903 | q = 0, plain PM alone | `c14591d9e5fa387b64354f3904180400848f3ee9d6086e4e947bfcbb6ee2d27e` |
| 17 | 30 % | 9.002478955597553 | 59,996 | 619 | 619 | q = 0, plain PM alone | `922dd497ea5e5bbb7486f041b2db095ea1d4c0d829bc42b522e2009c85063d08` |

The policies are `surf_v2/gapsw/policy_d<d>.json`. Each holds the whole table (τ_q, routed shots, calibration failures
and projected time at every q) and the sha256 of its inputs. The calibration records are `surf_v2/gapsw/cal/`, and the
validation's summary is `surf_v2/gapsw/validate.json`. Every one of them names the `surf_v2/gapsw.py` that made it
(sha256 `7c652b0c966af77911b20acb61a940d00780e8d5eb329e01e9f658aa6e3b1f94`), and the analysis refuses any other. The
largest q of the grid is chosen at every d. At d = 7-15 gap switching's calibration failures still fall at the grid's
top, and at d = 17 they reach correlated PM's. Correlated PM alone, shown beside, is its limit as q grows.

## On the test set

| step | when | argument file | what it gives |
|---|---|---|---|
| reads | after the test draw and commit B, beside the PM reads | `alpine_gapsw_test.txt` | plain PM's answer, its weight and the gap on every test shot, in `$E/gapsw/reads/test` |
| times | after the freeze, on one exclusive EPYC 7713 node, from a deployment of F or B | `alpine_gapsw_cpu.txt` | the validation shots routed at the frozen τ, then three interleaved fresh-process repetitions of the ref, the front end's term on the 200,000 validation shots and correlated PM on the routed ones, in a fresh `$E/gapsw/cpu`, with the job's record `job.json` |
| analysis | after `alpine_analysis.txt` | `alpine_gapsw_analysis.txt` | `surf_v2/gapsw_analysis.py` twice, byte-identical, in `$E/gapsw/analysis` |

Gap switching's composite is the slowest repetition of the front end's term plus the slowest repetition of its CPU
term, each times G(d) over the median of the same job's three refs. No handoff is added, because the whole pair runs on
the CPU.

The timing job writes into a fresh directory, and every file once. Its last step writes `job.json`: the Slurm job, the
deployed public commit, and every file the job wrote with its sha256. The analysis requires that record, exactly those
files, refs and CPU terms of that job, front-end terms of that commit, the frozen read path, and a commit that is the
freeze commit F or B. A timing job that fails runs again into a fresh directory, and the first job that completes
counts: no timing job starts once one has written its `job.json`.

## The comparison (family G)

- One test per distance: the exact two-sided McNemar test of the gap-switching pair at its frozen τ against the
  ConvLines pair, which is `surf_v2.analysis`'s row network at its frozen τ (under a training-failure exception, the
  deployed network).
- The Holm step-down runs over the distances at 0.05, as a family of its own.
- A rejected test is reported as "ConvLines pair fewer failures" or "gap switching fewer failures", according to the
  discordant counts. Any other test is "inconclusive", never "no difference". The Newcombe 95 % interval of the
  difference is shown beside it, pointwise.
- Described beside the tests:
  - the failures of gap switching, the ConvLines pair, correlated PM and plain PM, with Wilson 95 % intervals;
  - the composite times (gap switching's, the ConvLines pair's from `analysis.json`, and the registered times of
    correlated and plain PM);
  - gap switching's failures at every q of the grid (τ_q from the calibration shots), with the ConvLines pair's own q
    marked.
- The analysis refuses a front-end read, policy or timing record made by another `surf_v2/gapsw.py`, a front-end read
  whose answers are not plain PM's test read's, a timing directory that is not one completed job's from a deployment of
  F or B, and every read `surf_v2.analysis` refuses (the same checks, imported).
