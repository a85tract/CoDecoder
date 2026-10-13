# ConvLines-BB: ConvLines with confidence routing on bivariate bicycle codes (stage 1 design)

Written on branch `ising-bb` from master eea057f0, opened 2026-10-05 ~22:55 PDT
by the study lead at the authors' request to try ConvLines plus confidence routing on BB72 and BB144. Stage 1 = design,
port and smoke runs; no training beyond smoke runs. The plan for stages 2 and 3 is `bb/PLAN.md`. Every number below
comes from a job on the cluster named beside it; times are PDT unless marked MDT (Alpine, Derecho).

Both files are versioned registration documents (PLAN Q7). Version 1 is 6f7ff034 (end of stage 1); version 2
(2026-10-06) answers the independent stage-2 gate review; version 3 (2026-10-06) answers its re-check of c2a01ed4 and
records the study lead's decisions Q8 and Q9; sections 10 and 11 list what changed.

## Terms

| term | plain meaning |
|---|---|
| BB72, gross | the bivariate bicycle codes [[72,12,6]] (l = m = 6) and [[144,12,12]] (l = 12, m = 6): 72 / 144 data qubits, 12 logical qubits, distance 6 / 12; A = x^3 + y + y^2, B = y^3 + x + x^2 |
| torus, site | the code lives on an l x m grid with wrap-around edges; site (a, b) holds one X check, one Z check, one "L" and one "R" data qubit |
| monomial, translate | x^a y^b = the shift by (a, b) on the torus; the translate of an operator is the same operator shifted |
| detector | a parity of two consecutive measurements of one check (1 = "fired"); in an X memory the X checks give detectors in every round, the Z checks from the second round on |
| DEM | detector error model: every elementary fault of the circuit with its probability and the detectors and observables it flips |
| co-firing offset | the shift (dt, da, db) between two detectors flipped by the same elementary fault |
| Tanner offset | the shift between two checks that share a data qubit (one step of the code's own neighbourhood) |
| logical observable, block failure | the 12 X-type logical operators the circuit measures; a block failure = any of the 12 predicted wrongly |
| translate query | one read-out query per (logical k, shift g): predicts the flip of the shifted logical |
| joint vote | the decision over all 2^12 = 4,096 logical classes from all translate logits (section 4.4) |
| margin | the network's confidence ranking: best class score minus second-best (not a calibrated probability) |
| routing, fallback | a shot whose margin is below tau goes to a classical decoder (the fallback) instead of the network's decision |
| Relay-BP | IBM's "relay" belief-propagation decoder (relay-bp 0.2.2); "sector" = on the X detectors only (IBM's standard), "full DEM" = on all detectors |
| BP-OSD | belief propagation + ordered-statistics decoding (ldpc 2.4.1 on CPU; NVIDIA cudaq-qec 0.8.0 on GPU) |
| MAC | one multiply-add |
| A100-h, GPU-h | one A100 (any GPU) for one hour |
| deployed engine | the one fixed computation (batch 1, fixed precision, one CUDA graph) through which a network is read and timed (`bb.engine`, section 5.3) |
| provisional estimate | a speed number from a loaded or shared host, frozen in source and rule, used only to screen policies (PLAN 2.7) |
| quiet host | the registered timing hosts: a full card of the registered GPU model allocated alone (an RTX PRO 6000 Blackwell; A100-40GB the fallback, PLAN 2.8), and an exclusive Alpine CPU node running one decoder process at a time |
| registered GPU model | the one GPU model of a code's deployed reads, engine validation and GPU-side timing (PLAN 2.8, Q9) |
| commit guard | `bb.route`'s refusal of any dev or test read unless every policy of the code is a committed file of the clean deployed commit |

## 1. Codes, circuits and the detector grid (`bb/code.py`; Alpine, `data/bb/design_{bb72,bb144}.json`)

The circuits are the closed qLDPC route's pinned Relay-BP reference circuits (github.com/trmue/relay at d185194,
`memory_X`, uniform circuit noise, Bravyi et al.'s depth-8 cycle; sha256 checked against
`qldpc_neural/data/circuit_manifest.csv` at every load), 6 noisy rounds for BB72 and 12 for gross. Relay-BP orders the
qubits L data, R data, X checks, Z checks, each block by site index a m + b. `bb.code.Layout` reads every detector's
measurement records off the circuit and places it at (t, type, a, b): t = the round of the latest ancilla measurement
it compares (X: 0 ... R - 1, then R for the final data-measurement layer; Z: 1 ... R - 1), type X or Z, (a, b) its
check's site.

| fact | BB72 | gross | plain meaning |
|---|---|---|---|
| grid (T, l, m), channels | (7, 6, 6), 2 | (13, 12, 6), 2 | time layers x torus; an X and a Z detector channel per site |
| detectors | 432 = 7 x 36 X + 5 x 36 Z | 1,728 = 13 x 72 X + 11 x 72 Z | no Z detector at t = 0 (random first Z round of an X memory) or t = T - 1 (data layer) |
| X stabilisers read off the final layer | equal the polynomial construction | equal | the layout and the code agree |
| DEM faults (undecomposed, merged) | 16,200 | 67,824 | elementary faults |
| DEM invariant under x and y | 0 faults missing, prior difference 0 | 0, 0 | the circuit is exactly translation-invariant, so weights can be shared across sites |
| fired detectors per shot (10,000 shots) | 43.6 at p = 0.003, 30.0 at 0.002 | 181 / 232 / 280 / 324 at p = 0.003 / 4 / 5 / 6 | about 10 % of the detectors |

## 2. Receptive field: what one convolution must see (`bb.design`)

### 2.1 Co-firing offsets
For every fault and every ordered pair of detectors it flips, the offset (dt, da, db) mod the torus.
They span dt in {-1, 0, 1} and 96 offsets on BB72 (of the 108 a 3 x 6 x 6 box holds) and 131 on gross (of 216). Their
spatial projection is the whole 6 x 6 torus on BB72 and 65 of 72 sites on gross: circuit-level faults (CNOT faults,
hook errors) spread far beyond one data qubit's three checks.

### 2.2 The code's own neighbourhood
Checks sharing one data qubit sit at the Tanner offsets X-X and Z-Z: A_i - A_j and
B_i - B_j (13 offsets with 0), X -> Z: A_i + B_j and Z -> X: -(A_i + B_j) (9 each); 26 distinct offsets on BB72, 29 on
gross. The non-local monomials x^3 and y^3 put them up to 3 sites away in both directions (e.g. (3, -1), (-1, 3), (3, 3)).
All of them lie inside the co-firing set (`taps.tanner.*_inside` true for both codes).

### 2.3 Choice of taps
The stem's taps are exactly the co-firing offsets, so one stem step sees every elementary fault that touches
a site's detectors, the non-local monomials included (the analogue of ConvLines' 3 x 5 x 5 stem). Each block's taps are
the 26 / 29 Tanner offsets x dt in {-1, 0, 1} (78 / 87 taps; the analogue of ConvLines' 3 x 3 x 3 block, "one step of
the code's neighbourhood"). The dense alternative (every co-firing spatial offset x 3: 108 / 195 taps) is kept as an
option (`taps_from_design(blk="dense")`); it costs 1.4x / 2.2x the block MACs and would make each block a global spatial
mixer.

The audit below expands the receptive field with the taps the network uses (the stem's co-firing taps, then the
Tanner block taps), jointly in time and space (`bb.design.receptive_field`; Alpine, the same result
from the full design run and from `--rf-update` of the committed JSONs).

| receptive field of one site (BB72 / gross) | stem | + 1 block | + 2 blocks | + 3 blocks (b3) | plain meaning |
|---|---|---|---|---|---|
| time reach (layers) | +-1 | +-2 | +-3 | +-4 | one layer per stage, as in ConvLines |
| share of the torus reached at dt = 0 | 100 % / 79.2 % | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % | same time layer |
| share of the torus at the edge of the reach | 83.3 % / 51.4 % | 100 % / 100 % | 100 % / 100 % | 100 % / 100 % | the least covered layer |
| full window: every site reached at every abs(dt) up to | 0 / none | 2 / 2 | 3 / 3 | 4 / 4 | layers |

So "does the receptive field reach every check's support" has a short answer on these codes: every check's six data
qubits, and every check sharing one of them, lie inside the stem's taps; after the stem and one block a site's features
depend on every detector within +-2 layers. In space, extra depth adds reasoning steps without adding reach. In time,
b3's trunk sees +-4 layers; the read-out then pools all sites and all layers (T = 7 / 13), so every logit depends on
the whole syndrome, but detectors more than 4 layers apart (up to 6 on BB72, 12 on gross) interact only through that
pooled sum.

## 3. The logicals under translation (`bb.code.translates`, `bb.design`)

Every translate T_g ell_k of a logical is again an X logical, T_g ell_k = sum_j M_g[k, j] ell_j + (a product of X
checks c_{k,g}), found by GF(2) elimination against [ell; H_X]. Its flip is exact from the shot:
y_{k,g} = (y M_g^T)_k XOR (XOR over the checks in c_{k,g} of all their X detectors), because one check's X detectors
XOR to its final data-derived value. On 2,000 fresh shots of the circuit with one extra observable per translate, every
extra observable equals this formula (BB72 420 of 420 translates, gross 852 of 852), and the extra observables leave the
sampled detectors and the 12 observables unchanged.

| fact | BB72 | gross | plain meaning |
|---|---|---|---|
| translations acting trivially on all 12 logicals | 1 only | 1 and x^6 | gross: the known fact, confirmed; BB72: no non-trivial translation leaves every logical alone |
| translates equivalent to each logical (its own stabiliser) | 1 for 10 logicals; 4 for logicals 8, 9 ({1, x^3, y^3, x^3 y^3}) | 2 for 10 logicals; 8 for logicals 8, 9 (x^{3i} y^{3j}) | ConvLines' "equivalent lines" per logical |
| logical weights | 6 x 6, 14, 12, 12, 12, 8, 12 | 12 x 6, 26, 30, 28, 28, 20, 24 | the circuit's observables |
| distinct functionals among the k G translates | 153 of 432 | 153 of 864 | each translate's flip is one of 153 linear functions of the 12 logical flips; together they have rank 12 |

The strict analogue of ConvLines' lines (a vote over the translates equivalent to one logical) therefore has a single
line for 10 of the 12 BB72 logicals and two on gross: too thin to vote. The design uses every translate (section 4.4).

## 4. The network (`bb/net.py`)

### 4.1 Input and stem
Detector bits on the (T, l, m) grid, two channels (X, Z). Stem: one convolution with the co-firing taps, circular in
(a, b) and zero-padded in t, 2 -> C channels, plus a learned bias per time layer (it tells the network where the Z
channel is structurally empty and where the boundaries are). Circular padding + shared weights make every layer exactly
translation-equivariant.

### 4.2 Blocks
As ConvLines (arch.net): L pre-norm bottleneck residual blocks, LayerNorm -> 1 x 1 x 1 C -> Cb -> SiLU -> tap convolution
Cb -> Cb (the Tanner taps x dt in {-1, 0, 1}) -> SiLU -> Cb -> C (zero-initialised), added to the stream. Both check types
share a site, so no type-specific kernels are needed (ConvLines needed them because X and Z sites alternated on a
chessboard). Training implements a tap convolution as a dense conv3d on the circularly padded grid with a fixed 0/1
kernel mask (cuDNN); an inference engine would gather only the taps.

### 4.3 Read-out: translate queries
One query per (logical k, translate g), k G queries (432 / 864). Each pools the site features with
gate = sigmoid(q0 . key(site) + bc) x sigmoid(b[k, h, t, site - g]) / 64, summed over all sites (sums keep counts, as in
ConvLines, since a logical's flip is a parity). The first factor is a content gate shared by all queries (ConvLines'
shared query q0); the second is a per-logical position gate indexed by the site's offset from the translate (ConvLines'
bias by the signed distance to the line, here a full table over the torus and the time layers), initialised to prefer
the X checks that touch the logical's support. Then out-projection, residual with the logical's query q0 + e_k,
LayerNorm + MLP (C -> 2C -> 1) -> logit z[k, g]. Translating the syndrome by d maps z[k, g] to z[k, g - d] exactly
(`bb.checks`: largest difference 3.3e-16 on BB72 and 5.0e-16 on gross, float64, random weights).

### 4.4 Decision: the joint vote over the 4,096 logical classes
For a candidate class y (12 bits), translate (k, g) predicts y_{k,g} = phi_{k,g}(y) XOR g_{k,g}(detectors). If the
translate logits were independent, the log-likelihood of y would be, up to a y-independent constant,

    score(y) = sum_{k,g} (-1)^{phi_{k,g}(y)} s_{k,g} + sum_j (y_j - 1/2) beta_j,   s_{k,g} = -(-1)^{g_{k,g}} w_k z_{k,g} / 2,

a (4,096 x k G) sign matrix times a vector: 1.8 M MACs per shot on BB72, 3.5 M on gross. Decision = argmax_y score(y);
the margin between the best and the second-best score ranks the shots by confidence. With every translate equivalent
to its logical this is exactly ConvLines' vote c = sum_j w_j (-1)^{g_j} z_j + b0, one logical at a time; here it also
uses the translates that are equivalent to combinations of logicals. The vote with "perfect" logits (+-20 on the true
translate flips) returns the true class on every shot of both codes (`bb.checks`). The per-logical strict vote is
logged as a diagnostic.

The translates of one syndrome share their evidence, so their logits are far from independent. Enumerating the 4,096
classes maximises the learned score exactly, but neither the score nor softmax(score) is an exact physical posterior
or a calibrated confidence. The margin is used only as a ranking statistic: tau is its q-quantile on the calibration
split, and the calibration frontier measures what routing each fraction of the least confident shots buys (section
5.1).

The loss is the cross-entropy of softmax(score) against the true class (a classifier's objective; softmax(score) is
the distribution the independence model would give and is not calibrated as a posterior) + the mean over translates of
BCE(z_{k,g}, y_{k,g}) (every translate supervised, as every line in ConvLines). Labels come free from (detectors,
observables) on the GPU.

### 4.5 Sizes

| configuration | parameters | MACs per shot | of which read-out + vote | plain meaning |
|---|---|---|---|---|
| BB72, width 128, bottleneck 32, 3 blocks (the smoke model) | 483,357 | 118.3 M | 45.5 M | about ConvLines d = 9 b3's 100.7 M |
| gross, width 128, bottleneck 32, 3 blocks (the smoke model) | 821,085 | 486.6 M | 182.0 M | stem 31.4 M, blocks 273.2 M, read-out pooling 103.5 M; about 5 times ConvLines d = 9 b3 |

## 5. Routing and the composite time

### 5.1 Rule
As ConvLines' policy (arch README 15.5 / 16.1): the margin m of each shot, read through the deployed engine (5.3);
tau_q = the q-quantile of m on the calibration split; shots with m < tau_q go to the fallback. q from a grid {0, 0.5,
1, 2, 3, 4, 6, 8, 10, 12, 15, 20 %} (BB72 errors are dense, so the grid reaches higher than the surface code's 16 %).
Only admissible q are candidates: q is admissible when its projected composite, from PLAN 2.7's frozen provisional
estimates (the deployed engine's GPU term, the fallback's recorded time on the routed calibration shots, the measured
handoff, at least 1.1 us), is at most the cell's provisional limit (the fastest gating opponent's stage-1 batch-1
time). The policy = the smallest admissible q whose calibration failures are at most M - 2 sqrt(2 M), M = round(0.90 x
sector Relay-BP's calibration failures), else the admissible q with the fewest calibration failures (ties to the
smaller q). If no q is admissible there is no policy and the cell stops (PLAN 2.5). The routed shots are the slow ones
for Relay-BP, so the bound decides q on gross. tau is stored at full precision; nothing is refitted after a dev read.
`bb.route frontier / apply` implement it; the frontier refuses any split but cal and any read not made by a deployed
engine.

### 5.2 Fallback
Sector Relay-BP with the closed route's (the authors' tuned) parameters: the strongest *fast* opponent
(BB72 dev: 1.35 % against BP-OSD's 1.65 %, section 6), at a mean of 2.3-3.8 ms per BB72 shot on one loaded core and
17 ms per routed shot (section 8.4). It plays the role correlated PyMatching played on the surface code: the opponent
itself decides the shots the network is unsure of, so the hybrid beats it only where the network is better than it on
the confident shots. Full-DEM Relay-BP is the more accurate alternative fallback (0.555 % on BB72 dev, section 6) at
27 times the mean time per shot (105 ms); even q = 0.5 % would put its CPU term near cudaq-qec's batch-1 time, so the
admissibility bound of 5.1 rules it out. BP-OSD is the accuracy reference only: here it is less accurate than Relay-BP
and not faster.
The composite takes the fallback's decisions from the fallback's stored read of the split (every shot decoded in chunk
order). Relay-BP draws its random memory strengths from one stream per decoder process, so a shot's decision depends on
its place in the stream: decoding 150 cal shots in reverse order changed some decisions (stage-2 smoke test). A decode
of the routed shots alone would draw differently, with the same distribution. On the routed shots the stored read pairs
the composite with sector Relay-BP on the same draws; neither arm's failure probability changes, so the paired tests
stay valid. Network reads are deterministic for a fixed engine signature, and ldpc's BP-OSD has no random draws.
cudaq-qec's GPU BP-OSD varies between processes (three reads of BB72-3 cal chunk 0: 161, 164 and 168 failures in 10,000
shots), so each of its reads, like each Relay-BP read, is one draw of the decoder, made once and stored. Its reads and
its timing are separate commands (`bb.gpu_bposd read` and `time`; timing writes no predictions); a read stores its id
in every chunk and replaces the stored read only once complete, and `bb.opp.load_preds` refuses chunks of different
reads, so a rerun always replaces the whole read.

### 5.3 Composite time per shot (batch 1), the BB form of SETTINGS 5 / 0.8
The deployed engine (`bb.engine`) is one fixed computation per network, used for every calibration, dev and test read
and for the timing, so the margins that set tau are the deployed decoder's own: batch 1 (no padding), one captured
CUDA graph, fixed precision (rung `graph`: bf16 autocast for the trunk and read-out, fp32 vote and margin; `inductor`:
the same compiled by torch.compile; a fused Triton engine if PLAN 2.4's port succeeds; `graph32`: the graph in fp32,
for a network whose bf16 rungs fail validation), cuDNN deterministic, TF32 off, on the registered GPU model (PLAN 2.8:
a full RTX PRO 6000 Blackwell, A100-40GB the fallback). Its signature, which names the GPU model, is stored with every
read and every policy; `bb.route` refuses to mix signatures.
(a) GPU term: the deployed engine's device-only call (detector bits on the GPU -> decision and margin on the GPU; 200
warm-up replays, then 5 x 100 timed replays; a process's value = the mean of the 500), the slowest of five fresh
processes (`bb.latency`).
(b) CPU term: routed fraction x the mean single-core fallback time of the record's routed shots (routed shots are the
hard ones, so this exceeds the fallback's mean shot), one pinned core, three fresh processes, the slowest counted
(`bb.timing routed`).
(c) Handoff: the larger of the frozen allowance of 1.1 us per shot and the measured host round trip of one shot's
transfers around the graph (pinned host -> device copy of the detector bits, device -> host copy of the decision and
margin, synchronised), as S14.6. Composite = (a) + (b) + (c).
The registered numbers come from the quiet hosts (PLAN 2.7): (a) and (c) on a full card of the registered model
allocated alone, measured in one watched window per process with SETTINGS' acceptance (no other process on the GPU,
every SM clock sample at least 97 % of the maximum); (b) on an exclusive CPU node, one process at a time. Every timing
row names its session (the Slurm job), its process and what it is bound to (the engine signature, or the policy file
for (b)), and `bb.timing gate` counts only the named sessions' rows, one per process. The provisional estimates
that screen policies apply the same formulas to numbers from shared hosts (PLAN 2.7).

### 5.4 CPU basis
The surface-code studies converted CPU times to the Grace basis because the opponent's (NVIDIA's) PM
share came from the paper's Grace CPU. Here no published Grace time exists for Relay-BP or BP-OSD on these codes, and
Alpine's GH200 nodes are closed to this account. Every CPU decoder, opponents and fallback alike, is
timed single-threaded on one pinned core type of one exclusive node in one session, so the basis is common and no
conversion is made; the core model is named in every row.

### 5.5 Opponents' batch-1 times and the speed gate Relay-BP and BP-OSD: the mean single-shot decode on that core over
the first 10,000 (BB72) or 1,000 (gross) dev shots, three fresh processes, the fastest counted (`bb.timing opponent`;
the fastest is the stricter bar for us). cudaq-qec GPU BP-OSD: its batch-1 call on the GPU model of (a), host to host
(its API takes host arrays; the device-only part cannot be isolated, which favours us and is stated with the result),
300 calls, three fresh processes, the fastest counted, its best-batch throughput beside. Speed gate: the composite is
at most the
fastest of the three (`bb.timing gate`). Development numbers from stage 1 are in section 8.

## 6. Opponents and their dev accuracy (fresh dev shots; `bb.opp`, `bb.gpu_bposd`)

Each decoder read the same stored shots once. The CPU decoders ran in three Alpine jobs
on an AMD EPYC 9534 node, 32 single-threaded decoder processes at a time; the GPU decoder ran in
two Derecho jobs (cudaq-qec 0.8.0, A100-SXM4-40GB). A failure is a block failure (any of the 12
logicals wrong). The CPU times are per shot inside one of the 32 concurrent processes. They are development numbers
from a loaded node and probably slower than the quiet single-core times that the registered timing of section 5
measures.

| cell (dev shots) | decoder | dev failures (rate) | CPU ms per shot: mean / median / 99th pct | plain meaning |
|---|---|---|---|---|
| BB72, p = 0.003 (20,000) | sector Relay-BP | 269 (1.345 %) | 3.82 / 2.66 / 25.8 | the strongest fast opponent; also the fallback |
| | ldpc BP-OSD ms80-CS60 | 330 (1.650 %) | 2.36 / 0.55 / 32.4 | the closed route's accuracy reference |
| | cudaq-qec GPU BP-OSD ms80-CS60 | 327 (1.635 %) | GPU (table below) | NVIDIA's BP-OSD; same count as ldpc's within noise |
| | full-DEM Relay-BP | 111 (0.555 %) | 104.8 / 70.9 / 613.5 | 2.4 times fewer failures than sector Relay-BP at 27 times its mean time |
| BB72, p = 0.002 (30,000) | sector Relay-BP | 72 (0.240 %) | 2.29 / 1.88 / 13.9 | |
| | ldpc BP-OSD | 87 (0.290 %) | 0.86 / 0.39 / 26.3 | |
| | cudaq-qec GPU BP-OSD | 87 (0.290 %) | GPU | |
| | full-DEM Relay-BP | 27 (0.090 %) | 62.8 / 49.1 / 270.9 | 2.7 times fewer failures than sector Relay-BP |
| gross, p = 0.004 (20,000) | sector Relay-BP | 103 (0.515 %) | 143.2 / 35.6 / 3,123 | 5 times fewer failures than BP-OSD |
| | ldpc BP-OSD | 523 (2.615 %) | 130.5 / 12.3 / 434.9 | |
| | cudaq-qec GPU BP-OSD | 515 (2.575 %) | GPU | |
| | full-DEM Relay-BP | 1 of the first 640 (0.16 %) | 4,310 / 1,913 / 41,858 | 4.3 s per shot, so read on 640 shots only |

Calibration splits (20,000 shots each; they only fix tau): sector Relay-BP fails 262 (BB72, p = 0.003), 59 (BB72,
p = 0.002) and 99 (gross, p = 0.004).

Gross noise scan (2,000 smoke shots per p, bb_opp1; sector Relay-BP / BP-OSD failures, then their mean ms per shot):
p = 0.003: 1 / 8 (40.9 / 52.7 ms); 0.004: 12 / 52 (156.9 / 131.0); 0.005: 100 / 290 (657 / 248); 0.006: 403 / 768
(1,895 / 354). cudaq-qec fails 273 of 2,000 at p = 0.005. The gross cell therefore sits at p = 0.004 (PLAN.md
section 1).

cudaq-qec GPU BP-OSD timing (A100-SXM4-40GB, Derecho gpu-node-10; host-to-host call with numpy input):

| code, p | batch 1: ms per call, mean / median / 99th pct | best batch: us per shot (shots per s) | plain meaning |
|---|---|---|---|
| BB72, 0.003 | 1.30 / 0.64 / 6.84 (300 calls); rerun 1.06 / 0.58 / 6.16 (50 calls) | 50.3 (19,893) at batch 4,096 | the batch-1 bar the BB72 composite must beat |
| BB72, 0.002 | 0.63 / 0.46 / 2.99 (300 calls) | 40.5 (24,681) at 4,096 | |
| gross, 0.004 | 15.8 / 11.1 / 32.1 (100 calls) | 832 (1,202) at 256 | the batch-1 bar for gross |
| gross, 0.005 | 23.4 mean (100 calls) | | |

Relay-BP's and BP-OSD's batch-1 times come from stage 2's quiet-host session (section 5.5, PLAN.md 2.7).

## 7. Training (stage-1 recipe; PLAN.md sizes stage 2)

arch.train's recipe: AdamW (0.9, 0.99), weight decay 0.01 (not on biases, norms, the position table), 500 warm-up steps
to a peak learning rate of 1e-3, cosine to 2 % of the peak, gradient clip 1, bf16 autocast, EMA 0.999 (the read-out),
batch 1,024, fresh true-label Stim shots. Labels: the 12 observables and the exact flips of all translates (section
3). Monitor: block failures on 10,000 stored monitor shots (or fresh monitor-split shots where none is stored), current
and EMA weights. Optional noise curriculum `--p-ramp P0:F`: the training noise rises from P0 to the cell's p over the
first fraction F of the samples, snapped down to the Relay circuit grid (0.001, 0.0015, ..., 0.006); added after the
gross smoke (section 8).

The stream (version 2): global chunk c of 8,192 shots has its own seed (section 9), so the run id and the noise
schedule fix the shot order whatever the number of sampler processes, and a run can resume at any chunk. `bb.train`
checkpoints at chunk boundaries (every monitor point and every pause) with the weights, the EMA, the optimizer, the
step (which fixes the learning-rate schedule and the stream position) and the log; `--resume` continues exactly and
refuses changed settings. Every monitor row records the chunk index and a digest of its shots, so a resumed run can be
checked against an uninterrupted one. Ends: completed (final_ema.pt), early_stop (`--early-stop S:F`: at the first
chunk boundary at or after S samples the EMA network fails more than a fraction F of the monitor shots), nonfinite
(the loss; the checkpoint is marked terminal) and paused (`--max-minutes`, resumable). Only a completed run's network
is read by `bb.route` (smoke tests excepted). In the stage-2 smoke test a run paused after 2 minutes and resumed
matched the uninterrupted run at all four evaluations (chunk digests, losses, EMA failures).

## 8. Smoke results (stage 1)

### 8.1 Checks
`bb.checks` ran in float64 on both codes at the start of two Derecho jobs. The network is
translation-equivariant to rounding (largest deviation 3.3e-16 on BB72, 5.0e-16 on gross), and with perfect logits
the joint and the strict votes return the true class.

### 8.2 Training smoke
Derecho A100-SXM4-40GB, one GPU and 8 cores; runs r1-r5 in three Derecho jobs. The monitor is the
10,000 stored BB72 monitor shots at p = 0.003, or 5,000 fresh monitor-split gross shots at the run's p.

| run | code, model, training noise | samples, wall time, rate | monitor block failures (current / EMA weights) | plain meaning |
|---|---|---|---|---|
| smoke72_r1 | BB72, b3 w128 (483,357 parameters, 118.3 M MACs per shot), p = 0.003, eager | 10.67 M, 540 s, 19.7 k/s | per 10,000: 782 / 4,766 at 2 M; 341 / 463 at 4 M; 239 / 233 at 6 M; 223 / 182 at 8 M; 171 / 150 at 10.67 M | after 9 minutes the network alone fails 1.50 %, near sector Relay-BP's 1.35 % on dev |
| smoke72c_r3 | the same with torch.compile | 4.52 M, 180 s, 25.0 k/s | 309 / 304 at 4.5 M | compiling trains 27 % faster |
| smoke144_r2 | gross, b3 w128 (821,085 parameters, 486.6 M MACs), p = 0.005 | 2.20 M, 421 s, 5.2 k/s | 5,000 / 4,998 of 5,000 | did not learn in 2.2 M samples (7 minutes) from scratch at p = 0.005: the loss stayed at chance, ln 4,096 + ln 2 = 9.011 |
| smoke144c_r4 | gross, ramp 0.001 -> 0.004 over the first 70 % | 3.00 M, 568 s, 5.3 k/s | 4,952 / 4,998 of 5,000 at p = 0.004 | the loss fell below chance while p <= 0.003, but at 3 M samples the network still fails nearly every p = 0.004 shot |
| smoke144l_r5 | gross, p = 0.001 | 3.00 M, 567 s, 5.3 k/s | current: 3,328 (0.5 M), 1,066 (1 M), 484 (1.5 M), 121 (2 M), 60 (2.5 M), 45 (3 M) of 5,000; EMA 402 at 3 M | gross learns at low noise (evidence of learning only: 0.9 % at p = 0.001 is far from useful) |

Most of the accuracy comes from the joint vote. At 10.67 M samples smoke72_r1's strict vote (each logical from its own
equivalent translates only) fails 1,433 / 1,281 per 10,000 and the 12 base logits alone 1,526 / 1,370, against the
joint vote's 171 / 150.

### 8.3 GPU latency of the PyTorch module
`bb.latency` in a Derecho job (A100-SXM4-40GB): one CUDA graph in bf16; median of 5 x 100 replays in one process,
the slowest repetition in brackets.

| code (b3 w128) | batch 1, us per call | batch 64, us per shot | batch 1,024, us per shot | plain meaning |
|---|---|---|---|---|
| BB72 | 541 (683) | 23.2 | 17.5 | batch 1 is launch-bound (many small kernels); a fused engine should remove much of it (PLAN 2.4's early check measures it) |
| gross | 721 (772) | 79.9 | 70.9 | |

### 8.4 Routing smoke
smoke72_r1's stopped EMA network with sector Relay-BP as the fallback (two Alpine
jobs). The calibration frontier below uses the 20,000 cal shots at p = 0.003, where sector Relay-BP alone fails
262 times and the network alone 315. The projected composite takes 541 us for the GPU term, 1.1 us for the handoff and
the fallback's contended times on the routed shots.

| routed fraction q | 0 | 0.5 % | 1 % | 2 % | 3 % | 4 % | 6 % | 8 % | 10 % | 12 % | 15 % | 20 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid cal failures | 315 | 273 | 235 | 206 | 202 | 209 | 216 | 230 | 235 | 242 | 247 | 252 |
| Relay-BP ms per routed shot | - | 20.6 | 21.9 | 19.1 | 17.2 | 16.0 | 13.5 | 12.1 | 11.0 | 10.3 | 9.5 | 8.2 |
| projected composite, ms per shot | 0.542 | 0.645 | 0.761 | 0.923 | 1.058 | 1.182 | 1.353 | 1.508 | 1.643 | 1.779 | 1.960 | 2.191 |

Sending the least confident 3 % of shots to Relay-BP removes 36 % of the network's failures and leaves 23 % fewer
failures than Relay-BP alone. The routed shots are the hard ones: Relay-BP spends 17 ms on each, against its 3.8 ms
mean. The smoke's policy is the q with the fewest failures, q = 3 % (tau = 3.2995777130126953). Under the stage-2 rule
(target M = round(0.90 x 262) = 236, admissibility limit at cudaq-qec's 1.06 ms) no q reaches
M - 2 sqrt(2 M) = 192.6, and the same q is chosen, admissible by 2 us.

One dev read (20,000 shots, bb_route1) routed 2.91 % of the shots and gave 189 failures (0.945 %); the network alone
failed 314 times. The hybrid beat sector Relay-BP's 269 (exact McNemar p = 1.6e-14; Newcombe 95 % interval of the rate
difference [-0.511 %, -0.297 %]) and BP-OSD's 330 (p = 1.5e-21), and lost to full-DEM Relay-BP's 111 (p = 1.0e-8).

With the PyTorch graph the development composite, about 1.06 ms, is level with cudaq-qec's batch-1 time of 1.06-1.30
ms. Half of it is the GPU term, so an engine whose batch-1 call takes tens of microseconds would leave the CPU term's
0.52 ms as the bulk. These are development numbers: 541 us is one process's median, while the registered GPU term is
the slowest of five fresh processes through the deployed engine (section 5.3); and this network trained for 9 minutes,
where stage 2 trains about 20 times longer, so a better network may route fewer shots and shrink the CPU term. The
smoke decides nothing about speed.

The gross evidence is narrower than "gross cannot learn at p >= 0.004": one 7-minute run from scratch at p = 0.005 and
one 3 M-sample curriculum ending at p = 0.004 failed; nothing longer was tried. PLAN 2.3's probe tests the longer
curriculum with an enforced stop.

## 9. Seeds and stored shots (`bb/shots.py`, `data/bb/shot_manifest.csv`)

Fresh namespace seed = 2^41 + 10^8 split + 10^7 code + 10^6 p_idx + j (split: dev 0, cal 1, smoke 2, monitor 3; code
bb72 0, bb144 1; p_idx as qldpc_neural's table; one chunk = 10,000 shots); training chunk c of run run_id at training
noise p: 2^42 + 10^9 (10 run_id + p_idx) + c (c < 10^9, run_id < 5,000; at most 5.5 x 10^13; the stage-1
smoke streams used 2^41 + 10^9 + 10^4 run_id + v + 10^5 round(10^6 p)). Every registered ising_compare namespace is
below 2^33, qldpc_neural's below 3.1 x 10^8, interp's at 2^40 + 10,000 d + j, T2-T4 at or above 2^62, so none
overlaps. Stage-2 run ids: 11, 12 (BB72), 13, 14 (BB72-2 fine-tunes), 21, 22 (gross); smoke runs use 90 and up.
Chunks are sampled once on Alpine (Stim's output depends on the host), stored bit-packed with their sha256 in the
manifest, and read back only through `bb.shots.load`, which refuses a chunk whose hash differs or that the manifest
lacks. Training streams are regenerated, not stored, so a run records its sampler environment (Stim, NumPy and Python
versions, the venv and the hash of its package list, and a fingerprint of Stim's output at the run's p) and
`bb.train --resume` refuses any other (PLAN 2.8). No test split exists here; stage 3's seeds come from a public beacon.

## 10. Changes in version 2 (2026-10-06, after the stage-2 gate review of 6f7ff034)

| review finding | change | where |
|---|---|---|
| HIGH: read order | calibration reads, policies and screens first; all policies of a code committed in one commit; only then dev reads. Opponents read dev blind; the network's dev read and the paired evaluation pass the commit guard | PLAN 2.0, 2.5; `bb.route` guard; `bb.opp`, `bb.gpu_bposd` |
| HIGH: calibrate and evaluate through the deployed engine | every read and timing of a network through `bb.engine` (batch 1, fixed precision, one CUDA graph, deterministic settings, A100-40GB); signatures stored and compared; order train -> engine -> calibrate -> commit -> dev; a fp32 rung (`graph32`) for a network whose bf16 rungs fail validation | 5.3; PLAN 2.0, 2.4 |
| HIGH: the gross probe's stop | `bb.train --early-stop 5e7:0.20`; replica 2 only after replica 1's screen (calibration accuracy and a non-empty feasible set) | 7; PLAN 2.3 |
| HIGH: pre-specify the test | test replica by calibration rank; claim per cell over the three gating opponents at alpha 0.05 / 3; n across all three comparisons; above the cap = not tested | PLAN 3.1-3.3; `bb.route size` |
| HIGH: timing | provisional estimates frozen (sources and rule; limits in `bb.route.LIMIT_MS`); quiet-host protocol before any test freeze, failure = stop; CPU term from the routed shots; an empty feasible set = no policy, stop; opponents counted at their fastest (version 1: the slowest) | 5.1, 5.3-5.5; PLAN 2.7; `bb.timing`, `bb.latency` |
| HIGH: resume | per-chunk training seeds; checkpoints with optimizer, EMA, step and log; `--resume` (version 1's 8 virtual streams could not resume, and its `--init` restored weights only) | 7, 9; PLAN 2.8 |
| HIGH: maintenance | 05:30 PDT start; 04:30 drain buffer, never a restart time; resume after restoration; realistic walltimes with `--time-min`; reservations checked (pm-10.7-*, every partition, to 10-08 05:30 PDT) | PLAN 2.8; `gpu.sbatch` |
| MED: receptive-field audit | recomputed from the taps the network uses (stem: co-firing offsets; blocks: Tanner offsets x dt), jointly in time and space; version 1 expanded every block with the dense spatial set and reported only the spatial projection | 2.3; `bb.design` |
| MED: joint vote and margin | worded as a learned ranking and a ranking statistic, not a posterior | 4.4; `bb.net` |
| MED: fine-tuning, retries | the BB72-2 fine-tune is mandatory when triggered (run ids 13, 14, fixed streams, replacement by calibration); infrastructure retry rules frozen; Relay-BP and cudaq-qec reads are not reproducible, so a rerun replaces the whole read | 5.2; PLAN 2.2, 2.8 |
| MED: budget, GPU cap, queues | full budget including the engine, opponents and timing; 3 GPUs and 64 cores; no Derecho | PLAN 0, 4, 5 |
| MED: gross wording | narrowed to the two failed stage-1 runs | 8.2, 8.4; PLAN 2.3, 6 |
| Q5 note | 541 us is a one-process median; no speed verdict from the smoke | 8.4 |

## 11. Changes in version 3 (2026-10-06, after the re-check of c2a01ed4: REVISE, no critical, 4 HIGH, 1 MED)

| re-check finding | change | where |
|---|---|---|
| HIGH: the guard trusted DEPLOYED and file existence | `deploy.sh` writes DEPLOYED_POLICIES: the deployed commit and the sha256 of every policy file in it, taken from git; the guard reads each policy's bytes and refuses any whose hash is not that commit's, so a policy created or edited after the deployment never passes; the dirty check covers every shipped path | `bb.route` guard; `bb/scripts/deploy.sh` |
| HIGH: `apply --split test` at alpha 0.05 | test applies use ALPHA_TEST = 0.05 / 3 (McNemar and Newcombe), recorded in the output | `bb.route apply`; PLAN 3.2 |
| HIGH: timing records not bound | rows carry session, process and binding (engine signature digest; policy file sha256 for the CPU term); the gate takes the two sessions, counts one row per process, needs the handoff from the GPU-term processes and fails the session at a third rejection; the handoff is measured inside the watched window | 5.3; `bb.timing`, `bb.latency`; PLAN 2.7 |
| HIGH: cudaq-qec timing overwrote predictions | separate `read` and `time` commands; a read is replaced only whole (read id in every chunk, checked at load) | 5.2; `bb.gpu_bposd`, `bb.opp`; PLAN 2.8 |
| MED: sampler environment of resumed training | the run records Stim, NumPy, Python, the venv, its package hash and a Stim output fingerprint; resume refuses any other | 9; `bb.train`; PLAN 2.8 |
| Q8 (study lead) | quiet-host CPU side on one exclusive 64-core node, announced a day ahead | PLAN 0, 2.7 |
| Q9 (study lead) | registered GPU model: a full RTX PRO 6000 Blackwell (torch built for CUDA 12.8), A100-40GB the fallback by availability only, fixed per code before its first cal read | 5.3; PLAN 0, 2.4, 2.7, 2.8; `gpu.sbatch` |

Smoke of version 3 (Alpine, args/s2smoke4.txt, an RTX PRO 6000 Blackwell MIG 1g.24gb slice, tree c2a01ed4
plus these changes): a paused run resumed and completed, and a resume whose stored Stim fingerprint was altered was
refused; on Blackwell, ising's venv_gpu (CUDA 12.6) failed a one-line tensor computation while `$R/venv_cu128` ran
every rung (graph bitwise equal across two processes; decisions equal to the fp32 module on 99.53 % of BB72-3 cal
chunk 0 for graph, 99.57 % for inductor and 100 % for graph32, on the barely trained version-2 smoke network);
cudaq-qec read cal chunks 0-1 (166 and 151 failures), its two timing processes left the stored read unchanged, a rerun
of chunk 0 alone made `load_preds` refuse the pair until the whole read was made again; `apply` used alpha 0.05 on cal
and 0.05 / 3 on the stand-in test split; the gate passed a complete made-up session and refused a process counted
twice, another session, another engine signature, handoffs from other processes and the CPU term of another policy
file, and a third rejection failed the session; the guard allowed four committed policies and refused a missing
DEPLOYED_POLICIES, another commit's, a policy created or edited after the deployment, a stopped replica and a dirty
tree.

## 12. Changes after the second re-check (of de3e17ad: REVISE, no critical, 1 HIGH, 1 MED; 2026-10-06, before any cal read)

| re-check finding | change | where |
|---|---|---|
| HIGH: the CPU term was bound to the policy file only, so the dev gate counted a CPU term measured on cal (or on part of dev) | a `routed` row records its population: split, chunks, shots, routed count, a digest of the routed indices and the number timed (by default the first 5,000 on BB72, 1,000 on gross); `gate --split dev` (or test) recomputes that record from every chunk of the split in the shot manifest and the policy network's read at tau, and counts only CPU-term rows that carry exactly it | 5.3 (b); `bb.timing` population, routed, gate; PLAN 2.7 |
| MED: cudaq-qec's configuration and device unchecked by the gate | cudaq-qec rows count only if bound to ms80_cs60 (`bb.timing.CUDAQ`) and recorded on the policy's GPU (the device in its engine signature) | 5.5; `bb.timing gate`; PLAN 2.7 |

Test (`python -m bb.test_timing`, CPU only: made-up reads of BB72-3's 20 dev and 20 cal chunks, 2,000 shots each, the
CPU rows from bb.timing's own commands, the GPU-side rows as bb.latency and bb.gpu_bposd write them; on Alpine).
In de3e17ad's tree 1 of the 10 cases came out as registered: the gate counted CPU terms measured on cal (0.717 ms
against dev's 0.733 ms), on dev chunks 0-9, on the first 50 or on all 6,112 routed shots and on a dev read made again
later, counted cal rows beside complete dev rows (6 processes), gated on --split cal, and counted cudaq-qec rows of
another configuration or another GPU. With the fix all 10 did: a complete dev session passes with the CPU term
0.733 ms (the dev routed fraction x the slowest process's mean), the cal rows beside it are not counted, and every
other case is refused.
