# SURF-v2: the registered test of the retrained surface-code networks

This directory registers the test of the surface-code networks retrained without line supervision (loss BCE(c, y_0)
alone, at the deployed sizes, d = 7-17), and of the deployed networks read again beside them. Everything a result
depends on is fixed in the freeze commit, which adds this tree as `surface_v2/` to the paper's public artifact
repository (GitHub `a85tract/Convlines_Public`, branch `surf-v2`, on top of the repository's `main`) and is
pushed to GitHub before any test shot exists. The test seeds come from a public randomness beacon that was released
after that push, so nobody could know the test shots when the networks, thresholds and analysis were fixed.

`surface_v2/` holds the release's evaluation files in the release's layout, and every path and command below is
relative to it.

## What the freeze commit holds

| item | where | what it means |
|---|---|---|
| the networks | `weights/*.safetensors`, `weights/configs_sv2.json`, `freeze.json` networks | the six new networks (`d7_b1w64_sv2` ... `d17_b5_sv2`; at d = 9 and 11 the continuation stages N9b and N11b, trained to the deployed networks' cumulative 4e8 and 8e8 shots) and the six deployed networks, each with its sha256 |
| q and τ of each network | `freeze.json`, `weights/configs_sv2.json`, `policies/` | the routed fraction q and the threshold τ (float32, written with `repr`), chosen by the registered rule on fresh calibration shots; the deployed networks keep their published τ (R17's 4.539173126220703 at d = 17) |
| the evaluation code | `convlines/`, `timing/`, `surf_v2/` | the reads (`convlines.evaluate --net --tau --save`), the sets, the beacon draw, the q rule, the timing session and the analysis; `freeze.json` holds the read path's code digest, and every read and timing record must carry it |
| the analysis and its configuration | `surf_v2/analysis.py`, `surf_v2/config.json` | families A, B and C, the registered baseline times, the timing rules and the rule for the paper's rows; `freeze.json` holds the config's sha256 |
| the read and timing commands | `surf_v2/args/` | argument files: the pod reads and timing (`pod_e1_*.txt`), the Alpine jobs (`alpine_*.txt`) |
| the test seeds | `convlines/beacon.py`, `freeze.json` test | the formula below and one tag per distance |
| any training-failure exception | `freeze.json` exceptions | a distance whose new network did not train even with the ramp keeps its deployed network, labelled here before any test shot exists |

Nothing on this list changes after the freeze. A defect found later is fixed in a new commit, and the analysis reports
the departure.

## The sets

| set | seeds (chunk j of distance d, 10,000 shots each) | chunks | used for |
|---|---|---|---|
| `sv2cal` | 81,000,000 + 10,000 d + j | 0-19 at every d (200,000 shots) | fitting τ and choosing q |
| `sv2dry` | 89,000,000 + 10,000 d + j | 0-3 (40,000 shots) | dry runs only, never in a result |
| `sv2test` | the beacon formula below | 0-199 at d = 7, 0-99 at d = 9-17 | every reported count and comparison |

The circuits are the release's verdict circuits (`circuits/`, rotated surface code, X basis, d rounds, p = 0.6 %). A
chunk file holds Stim's detector sample of its seed; each set's manifest in `testsets/` lists every chunk with its seed
and sha256. Sets are drawn once on an x86-64 CPU with AVX2 (Stim 1.16.0, numpy 2.2.6).

## The test seeds

1. The freeze commit is pushed to GitHub. T is the time of the earliest record in GitHub's activity list of the
   repository whose `after` is the freeze commit (GitHub's own record of the push), so a later push of the same commit
   never moves T. `beacon.json` keeps that record with its id and the whole list as GitHub returned it, and every check
   of the record derives T from them again.
2. The beacon is the first drand mainnet round (chain `8990e7a9...51b2ce`, a round every 30 s) whose time is at least
   600 s after T. `python -m convlines.beacon fetch` waits for the round, asks four public endpoints, requires two
   agreeing answers, and writes `surf_v2/beacon.json` (randomness R = SHA-256 of the round's signature).
3. seed(d, j) = 2^62 + (the first 8 bytes, big-endian, of SHA-256(R ‖ TAG(d) ‖ j as 2 bytes big-endian)) mod 2^62,
   with TAG(d) = `convlines surf-v2 test d<d>` followed by a newline. This is the formula of the BB72 and gross tests.
4. The draw checks the beacon record again, samples every chunk, and writes the manifest. `beacon.json` and
   `testsets/sv2test.json` are committed on top of the freeze commit and pushed to GitHub after the draw and before any
   read.

## The routed fraction

Each new network's q comes from the studies' registered rule C4.2, run as the deployed policies ran it
(`surf_v2/qrule.py`; `ising_compare/arch/policy.py` of the research repository). On the network's calibration read
(200,000 shots, c from its deployed engine), for every q of the grid, τ_q is the q-quantile of |c| (τ_0 = 0) and a shot
is routed to correlated PyMatching when |c| < τ_q. The projected time of a pair is the locked measured path at d = 7, and
at d = 9-17 the network's registered GPU time plus, for q > 0, the router's 1.1 µs and the realized routed fraction
times correlated PyMatching's mean Grace-basis time on the routed calibration shots. A q is admissible when its
projection is at or below the speed bar. The rule takes the smallest admissible q whose calibration failures are at most
M - 2√(2M). When no admissible q is, it takes the fallback of that distance's registered policy: at d = 7 the largest
admissible q (the deployed d = 7 policy was fitted with this rule before the studies changed the fallback, and it was
kept), at d = 9-17 the admissible q with the fewest calibration failures (ties to the smaller q). The targets M, the
bars, the grids and the fallbacks are the studies' (`config.json` qrule). Run on the studies' own calibration records,
the code gives their registered choices: q = 4, 1, 1, 2, 6 and 10 % at d = 7-17, with the same τ to the last bit, and
each registered policy record names the fallback the code uses at its distance.

## The comparisons

At each d, on the same test shots:

| family | comparisons | test | verdict |
|---|---|---|---|
| A | the new pair against plain PM, correlated PM, Ising M1 / M4 + PM, Ising M1 / M4 + correlated PM, min-LLR BP-OSD and BP 60 BP-OSD: accuracy | exact two-sided McNemar test and Newcombe's method-8 interval at 97.5 % | "better" when p < 0.025 and the whole interval lies below 0 |
| B | the same 8: speed | the pair's composite time strictly below the baseline's registered time on the Grace basis | holds or not; "not established" without a qualified timing session or without a CPU term that matches it |
| C | new against deployed: each network alone, and each pair at its own q | exact two-sided McNemar test, Holm over the tests of all six distances at 0.05; Newcombe 95 % interval beside it (pointwise) | better, worse or inconclusive |

A distance's claim against the baselines holds when all 16 of its comparisons in A and B hold. BP-OSD reads every test
shot at d = 7-13 and the first 200,000 at d = 15 and 17. Ising's accuracy is read with NVIDIA's official PyTorch
numerics only; its speed is the studies' historical TensorRT composite. The baselines' times are the registered ones;
only the new and the deployed networks are timed again, in one session.

A secondary comparison with gap switching (Decoder Switching) is registered in `GAPSW_REGISTRATION.md`; it changes nothing above.

## Timing

The composite time per shot at batch one is the GPU term (the slowest of five fresh processes on one H100, every process
accepted only with full clocks, the GPU to itself and outputs equal to its engine's bit for bit) plus the CPU term (the
routed validation shots decoded by correlated PyMatching on one pinned core of an exclusive AMD EPYC 7713 node, the
slowest of three repetitions, converted to the Grace basis by the registered ratio) plus the handoff (the d = 7 kernel's
own, the largest of five; else max(1.1 µs, the router's estimate)). The CPU term is one Slurm job, run right after the
session's handoffs; its records are written once and carry the job and the code digest, and at d = 9-17 each network's
routed validation shots must be as many as its handoff routed in the session. A failed CPU-term job runs again into a
fresh directory, and the first job that completes counts. A session qualifies when its GPU check passes at the start and
at the end and every network has its accepted processes within two replacements. A second session is started only after
a technical failure, on a new host; the first qualifying session counts; there are at most two. Before anything is read
or timed on it, every timing pod is pre-screened: the same GPU check runs five times on its idle GPU
(`surf_v2/args/pod_prescreen.txt`), and a pod where any of the five fails is replaced by a new one. A screened-out pod
holds no session and is not a technical failure. The test reads on the session's pod start only once the session's
verdict is known, after its check at the end.

## The paper's rows

The paper's surface rows become the new networks' pairs at every distance, whatever their results. The one exception is
a distance under the training-failure exception, whose row is the deployed network's pair read on the fresh test set,
labelled as such. The deployed networks' registered results stay in the provenance table as the earlier tests.

## The analysis

`python -m surf_v2.analysis` runs once after the last read. It first checks the beacon record and re-derives every test
seed from it, then refuses any read whose network sha256, τ, set manifest, code digest or observables do not match the
freeze, or that was not made by the deployed engine at batch one; any calibration read other than the one the freeze
names; any timing record made with a limit or on another GPU; and CPU-term records from more than one job. It finds the
timing sessions itself and orders them by their GPU checks' times. With no timing session, with more than two session
directories or two sessions on one GPU (then no session counts), or with a CPU term that does not match the counted
session (a network without its three repetitions, or another number of routed validation shots than the session's
handoff), it still reports families A and C in full, and family B as not established, with the reason. A second run on
the same inputs must give byte-identical outputs: `analysis.json` (every number, verdict and input digest), `tables.md`,
the CSV files of the families, the composite table, the pair curves, the frontiers and the paper's rows, and the figures
`pair_curves` and `frontiers`. Anything computed afterwards is labelled exploratory.