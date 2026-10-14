# D9-SHALLOW: the registered test of the shallow d = 9 network B1

Registered on 14 October 2026 (PDT) by this directory's commit F. Abbreviations: PM = PyMatching 2.4.0 (minimum-weight
matching); corr PM = PM with `enable_correlations=True`; q = the share of shots routed to corr PM; tau = the routing
threshold on the network's |logit| c; the pair = a network with its routed shots decoded by corr PM.

An exploratory run (D9-SHALLOW, 13-14 October; not registered, its analysis not frozen before its data) trained two
shallower versions of SURF-v2's 3-block d = 9 network on that network's own training shots and chose, by a rule fixed
before its results, B1 = `d9_b1w64_sh`: 1 block, width 64, bottleneck 16, 90,467 parameters, at q = 0.06 and tau =
2.337686538696289. On its 1,000,000 test shots B1's pair failed 12,087 times against plain PM's 27,207 and corr PM's
12,945, at an estimated composite of 21.45 us (that composite used stand-ins for the CPU term and the handoff). This
test measures the same pair again on fresh shots with every term timed, under the speed re-test's rules (SPEED-RT stage 2,
`speed_rt/STAGE2_common.md` C1-C8, `speed_rt/STAGE2_surface.md`) and SURF-v2's beacon (`surface_v2/surf_v2/REGISTRATION.md`).

## 1. The claim

At d = 9, p = 0.6 %, on 1,000,000 fresh test shots, B1's pair at its frozen q = 0.06 and tau = 2.337686538696289
meets all three criteria:

1. its composite time a + b + c on the Grace basis is strictly below plain PM's 26.296678815736907 us;
2. it fails less often than plain PM;
3. it fails less often than corr PM, and its composite is strictly below corr PM's 63.916141817786674 us.

"Fails less often" means the exact two-sided McNemar test on the discordant shots gives p < alpha = 0.025 and the whole
97.5 % Newcombe (method 8) interval of the failure-rate difference lies below 0 (`surface_v2/convlines/stats.py`
`paired`, alpha 0.025). The claim joins its parts as an intersection-union test: it holds only when every part holds, so
no further correction applies. 95 % intervals appear beside the verdict intervals as labelled description only.

The targets are the speed re-test's registered values for plain PM and corr PM at d = 9
(`speed_rt/s2/surface/results/rows.csv`; `table.md` prints them as 26.297 and 63.916 us): the fastest of three processes
on the Linux workstation, converted to the Grace basis by that stage's fit (1.5097605583565314). They are not timed again.
Our composite must lie strictly below the unrounded values.

B1, q = 0.06 and tau are frozen unconditionally. Nothing measured in this test can switch to another network, q or tau,
and in particular not to the exploratory run's other arm B2 (`d9_b2_sh`): this test makes no ranking claim of B1 over
B2. Descriptive rows, with no verdict: B1 alone (q = 0) and SURF-v2's 3-block network `d9_b3_sv2s1` (its registered
q = 0) against plain PM, corr PM and each other, and the pair against `d9_b3_sv2s1`, including the accuracy B1's pair
gives up against the 3-block network.

## 2. What the freeze holds

Everything below is bound by sha256: the new files by `d9_shallow/SHA256SUMS`, the public files this test runs
unchanged by `d9_shallow/PUBLIC.sha256` (each at its path in the parent commit 7204673; every listed file is
byte-identical to the speed re-test's freeze F_2 7988e04 and analysis commit e0a9377, `srt-ref.json` to its commit B_2),
the constants by `d9_shallow/freeze_d9s.json`.

| item | content |
|---|---|
| B1's weights | `weights/d9_b1w64_sh.safetensors`, sha256 28d17a86c169662a1377e9d14cabc184dbab6d706533c554f59da7de14fb2105, exported (`surf_v2.export --check-arch`) from stage B's `final_ema.pt` 27a3632acdbd77c4e0f6554aaa46774d5958390349015ab21a1982cd934b0607; its export record `weights/d9_b1w64_sh.export.json`; its config entry `weights/d9_b1w64_sh.config.json` (arch, parameters, sha256, policy q 0.06, tau 2.337686538696289), which the pod overlays on its run tree's `surface_v2/weights/configs_sv2.json` |
| the 3-block network | `surface_v2/weights/d9_b3_sv2s1.safetensors`, sha256 7dd879cf5eeb6f8b39ddd5a5d1bdaac34ffffe17e3ed5dafdcebedab5fb63aa0, unchanged, q = 0 |
| q and tau | SURF-v2's frozen rule (`surf_v2/qrule.py fit_rule`, `config.json` qrule at d = 9: grid {0, 1, 2, 4, 6, 8, 10, 12, 14, 16} %, M = 2500, bar 33.99 us, fallback argmin) on the exploratory calibration read (200,000 shots). No q met the limit L = 2,358.6 (the best, 2,508 failures, at q = 6 %), so the argmin fallback chose q = 0.06; tau is the float32 quantile, written with `repr`. A shot is routed when abs(double(float32 c)) < tau. Not refitted. |
| the circuit | `surface_v2/circuits/d9_legacy.stim` (sha256 19390af8855ca9d885dfb12ff5d852c76f69eacf9600bebb26c6ff3286b77351): rotated surface code, X basis, 9 rounds, p = 0.6 %, SURF-v2's verdict circuit. PM and corr PM decode its DEM with `decompose_errors=True, approximate_disjoint_errors=True` (`convlines/circuits.py`). |
| the read path | `convlines/`, `timing/` (the deployed Triton engine, FROZEN config, batch one): code digest 61b3f4f96edb85a82c76339ef6d5707ed1a2ca756183951394f569558a980f21 (`convlines.provenance.code_digest`), the one `speed_rt/freeze_surf.json` holds; every read and timing record must carry it |
| new code | `beacon_d9s.py` (the beacon record and seeds), `draw_d9s.py` (the draw and manifests), `pod_run.sh` and `pod_d9s.py` (the pod: setup, pre-screen, masks, GPU session, handoff, hand-off to the workstation), `reads_d9s.py` (the reads), `box_d9s.sh` and `plan_d9s.py` (the CPU session), `collect_d9s.sh` (the analysis inputs), `analysis_d9s.py` (the analysis) |
| unchanged public code it runs | `timing.gpu_term`, `timing.deployed`, `timing.cpu_term`, `speed_rt.watch` (rule throttle), `speed_rt.gpu_prescreen`, `speed_rt.gref`, `speed_rt.grace_fit`, `speed_rt.analysis_s` (`qualifies_cpu`, `cpu_reader`, `fit_s2`), the CPU harness `speed_rt/cpu_ws/` (sha256 list 2fd2afa5df2be2cd, the one stage 2 deployed), `speed_rt/circuits/` and `speed_rt/testsets/srt-ref.json` (the Grace reference circuits and shots) |
| environment | the pod: `env/pod.txt` (CPython 3.12.3, torch 2.11.0+cu126 with Triton 3.6.0, numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, safetensors 0.8.0; nvcc 12.1 for the router); the workstation: stage 2's tmpfs environment, `env/box.txt` (CPython 3.12.3, numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, ...); the analysis: `env/analysis.txt` |
| argument files | `args/pod.args`, `args/box.args`, `args/analysis.args`: every command line of the test |

After F, only mechanical corrections are allowed (a path, a typo that stops a script), each in its own commit and each
reported as a departure by the results. A change to the claim, the network, q, tau, a rule, a statistic or a verdict
would need a new registration and fresh shots.

## 3. The shots

1. The freeze commit F is pushed to the public repository and to GitHub. T is the time of the earliest record in
   GitHub's activity list of the repository whose `after` is F.
2. The beacon is the first drand mainnet round at least 600 s after T, fetched from SURF-v2's four endpoints with at
   least two agreeing answers; R = SHA-256 of the round's signature (32 bytes). `beacon_d9s.py` uses SURF-v2's fetch and
   checks unchanged and writes the record `d9_shallow/testsets/beacon_d9s.json` with the whole activity list.
3. seed(set, j) = `convlines.beacon.seed_tag(R, tag, j)` = 2^62 + (the first 8 bytes, big-endian, of SHA-256(R || tag
   || j as 2 bytes big-endian)) mod 2^62, with two new tags:
   - test: `convlines d9-shallow test d9` followed by a newline, j = 0..99: 100 chunks of 10,000, 1,000,000 test shots;
   - timing: `convlines d9-shallow timing d9` followed by a newline, j = 0..19: 20 chunks, 200,000 timing shots.
4. Both sets are drawn once, on the GPU pod's CPUs (x86-64 with AVX2; Stim 1.16.0, numpy 2.2.6), in the chunk format of
   `convlines.testset`, by `draw_d9s.py`, which re-derives every seed from the record and, as a check of the sampler on
   that host, draws SURF-v2's dry chunk 0 of d = 9 again and compares it with its manifest. The manifests
   (`d9_shallow/testsets/d9s_test.json`, `d9s_timing.json`) hold every chunk's index, seed and sha256 and the digest of
   each whole set.
5. Commit B, a direct child of F, adds the beacon record and both manifests and is pushed. Nothing is read or timed
   before B is public. The timing shots are these fresh beacon shots, never the validation chunks.

## 4. The pod: GPU session, handoff and reads

One secure RunPod H100 80GB HBM3 pod (`pod_run.sh`). A pod is pre-screened before anything is timed on it
(`speed_rt.gpu_prescreen`, stage 1's rule: five GPU checks, CPU steal at most 1 % over 5 s, a quota of at least 20 CPUs).
A pod that fails is released and replaced; at most five are screened (STAGE2_common C1). After B is public, in order:

1. Masks (untimed): `timing.deployed` on the 200,000 timing shots for B1 (at its frozen tau) and `d9_b3_sv2s1`.
2. The GPU session: 5 rounds; in each, one fresh process of B1, then one of `d9_b3_sv2s1`, each `timing.gpu_term` under
   `speed_rt.watch run --rule throttle`, pinned to the pre-screen's timed core (2,000 blocks of 100 CUDA-graph replays over
   the 200,000 timing shots, input on the GPU). A process counts when it exits 0, its watch window is clean (clocks, no
   other process on the GPU) and gpu_term accepts it (outputs bit-equal to the mask's logits, clocks, the GPU alone). A
   process that does not count is replaced at once by a fresh one with the same arguments, at most twice. a = the slowest
   of B1's 5 counted processes.
3. The handoff: `timing.gpu_term --handoff` for B1 at its tau, one slot, at most 3 attempts: the compiled router
   (`timing/d7/build.sh`) drives the engine as the deployed decoder over all 200,000 timing shots. c = max(1.1 us, the
   router's estimate). (`d9_b3_sv2s1` routes nothing at q = 0; its descriptive composite takes the 1.1 us allowance.)
4. The hand-off to the Linux workstation: the counted session's mask of B1 (`val_d9_b1w64_sh.npz`: c, routed, tau, the
   timing set's digest) and the 20 timing chunks, with their sha256. The workstation never routes again.
5. The reads (untimed, after the session): the logits c of B1 and `d9_b3_sv2s1` on all 1,000,000 test shots from the
   deployed engine at batch one (`timing.deployed.Logits`, SURF-v2's test-read path), and plain PM's and corr PM's answers
   on the pod's CPUs. All four decoders read the same 1,000,000 shots. Every chunk is checked against B's manifest first.

Everything is pulled with sha256. The session counts only when it reaches its end; a session lost before its end is a
technical failure, and the role runs once more as a whole on a new pre-screened pod (STAGE2_surface 1.6); a second loss,
or a slot without a counted process after 3 attempts, leaves the GPU terms not established.

## 5. The CPU session on the Linux workstation

The CPU term b runs in one session on the Linux workstation (AMD Ryzen 9 9950X) with the speed re-test's CPU harness and
rules, unchanged (`speed_rt/cpu_ws/`, `cfg_box.json`, STAGE2_common C1): the authors' settings (governor performance,
energy-performance preference performance, boost on, CPUs 16-23 offline, every other process on CCD1), the tmpfs inputs,
8 slots on CCD0 with fillers, the per-window reference segments and clock probes, the disk-read check on the harness's
own read counter, the frequency records, and the window rule: every reference value within 1.10 times the session
reference and each segment's largest over smallest at most 1.06, every process exit 0. The session starts only when its
session reference reads within 1.10 times 17.306 us per block. A process of a rejected window or a failed process runs
again in a later window, at most 3 attempts a slot (C5).

`box_d9s.sh` imports the hand-off unit (SHA256SUMS checked), builds the timing view, and runs `plan_d9s.py`'s plan: three
fresh processes (repetitions) of each of
- the 18 Grace reference points of STAGE2_common C3 (`speed_rt.gref time` on the sets `srt-ref-d<d>-p<3|6>`, the stage-2
  reference chunks and manifest `speed_rt/testsets/srt-ref.json`), and
- B1's CPU term (`timing.cpu_term term` with the counted mask on the 200,000 timing shots: corr PM on the routed shots),

packed by the re-test's `plans.pack` (longest first, 8 a window, the sides alternating), repetition by repetition. b raw
= the slowest of the 3 processes' `term_us`, which is corr PM's time over the routed timing shots divided by all 200,000
timing shots. This session's Grace fit is C3's frozen fit (`speed_rt.analysis_s.fit_s2`, `speed_rt/grace_fit.py`): per
point the slowest of three, one factor per decoder, the common factor when the two agree. b on the Grace basis = b raw x
the factor applied to corr PM. If the fit is not established by C3's rule (a point not established, or the session not
at its end), the converted b is "not established".

## 6. The analysis

`analysis_d9s.py`, from commit B's tree, run twice on an Alpine CPU node; the two outputs must be byte-identical. It
refuses incomplete evidence: a failed pre-screen, fewer than 5 counted GPU processes of B1, a missing or invalid watch
record, a handoff not established, a CPU slot (reference point or term) without 3 accepted processes, a CPU session that
did not reach its end. Missing evidence makes the timing "not established", with no fallback value and no earlier value
standing in. It checks:
- the beacon record (T, the round, R, the agreeing fetches) and every seed re-derived from it;
- every shot file against the pushed manifests (sha256), exact chunk coverage (j = 0..99 and 0..19, nothing else), each
  set's digest, and observables aligned across the shot files and every read;
- the weights' sha256 and the code digest in every read and timing record, the deployed engine at batch one on the H100;
- the mask against the timing set's digest and tau; the CPU term's records against the counted session's mask (its
  sha256), the timing set's digest and the routed count.

The composite: a + b + c with a = the slowest of B1's 5 counted GPU processes, b = b raw x the fit's corr-PM factor,
c = max(1.1 us, the router's estimate). No proxy routed cost, no scaled a and no test-shot routed share enter it. Reported
separately: a, b raw, the factor, b on the Grace basis, c, the composite on the Grace basis (which decides) and on the raw
basis (the workstation's own CPU time, beside it; the re-test's raw-basis values of plain PM and corr PM are 17.418 and
42.335 us). The composite is a sum of terms from two machines (the H100 pod, the Ryzen 9 9950X); the Grace basis is an
estimate, with no Grace hardware used, and no deployment-latency claim is made.

## 7. Publication rule

The test and its outcome are published in this repository whatever the result. If the claim holds, the paper's d = 9 row
becomes B1's pair. If it fails, or any timing part is not established, the paper keeps the 3-block row (`d9_b3_sv2s1`).
The 3-block row stays in the provenance in either case.

## 8. Scope and departures

- This is a new d = 9 test: nothing is pooled with the exploratory run's counts or with SURF-v2's test counts.
- From the speed re-test: only the parts this claim needs are timed (B1 and `d9_b3_sv2s1` on the GPU, B1's handoff, B1's
  CPU term and the 18 reference points on the workstation); plain PM and corr PM keep their registered times. The
  reference points read stage 2's reference chunks (B_2's manifest), as the re-test did.
- From SURF-v2: the shots are drawn on the GPU pod's CPUs, not on Alpine; both are x86-64 with AVX2, and the draw checks
  the sampler against SURF-v2's dry chunk.
- One training run per arm: the claim concerns this checkpoint, not the architecture in general.
- Correction to the exploratory report (`D9B2_RESULTS.md`, section 1, "The only change is the network's size"): the two
  arms also changed the stage-B hardware (B1 and B2 trained both stages on H100 pods; `sv2_N9b`'s stage B ran on an
  RTX PRO 6000 on Alpine) and the sampler worker counts (B2 4 and B1 8 workers, against A1's 2 and N9b's 4; any count
  that divides 8 gives the same shot stream, as the stream chains confirm).
