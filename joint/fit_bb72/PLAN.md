# ARB: arbitration on the routed shots of CoDecoder + Cascade (BB72), exploratory

## Why
In the Cascade follow-up (D1) on BB72 with the fitted noise model, the composition (six-block CoDecoder network
S55'-deep, our Cascade C14' as the fallback, routing the network's least confident 2 % at p = 0.2 % and 10 % at
p = 0.3 %) beat Cascade alone by only 0.85 % and 0.34 % (D1's original reads; per-shot categories in
~/work/rev_x2/qec/revision/x2/data/bb/fig.npz on the Linux workstation; the paper's appendix "Analysis of the Discordant Shots" reports the
re-test). Counts on all 11,000,000 / 17,000,000 shots:

| p | Cascade fails | network fails | only network | only Cascade | both |
|---|---|---|---|---|---|
| 0.2 % | 7,922 | 10,348 | 4,735 | 2,309 | 5,613 |
| 0.3 % | 76,730 | 112,636 | 54,491 | 18,585 | 58,145 |

96 % and 98 % of the "only Cascade fails" shots are routed: the network is right there but unsure (median network-margin
percentile 0.075 % and 0.6 %; thresholds 2 % and 10 %). On routed shots both answers already exist, but the composition
keeps only Cascade's. An arbiter that chose perfectly between the two answers on the routed shots would leave the "both"
failures plus the kept "only network" failures, 5,632 and 58,226 (29 % and 24 % below Cascade alone), i.e. it could add
at most 2,223 and 18,243 repairs. Question: how much of that a cheap arbiter, using only the two decoders' outputs,
recovers on fresh shots, net of the breaks it adds. A negative answer limits what these outputs and rules recover; it
does not by itself rule out training a network to complement Cascade.

## Setting (D1's, unchanged)
- BB72, fitted circuit-level noise depspam_fm, p = 0.2 % and 0.3 %; S55'-deep on the fused engine with D1's frozen
  thresholds tau1 (phase_d/d1_locked_depspam_fm.json) and margins; C14' on its eager accuracy path in bf16 with D1's
  batches (8,192 and the 1,808 left of each 10,000-shot chunk); on a full RTX PRO 6000 (Blackwell Server Edition), the
  GPU model of D1's reads, so pc_eval's GPU guard holds as is.
- Code: the release (this repository): phase_c/pc_eval.py (bb.cascade_eval), pc_fused.py,
  pc_common.py, phase_d/d1_common.py; weights/cascade_bb72/S55p_deep.pt and C14p.pt.

## Steps
1. Fresh shots: 11,000,000 at p = 0.2 % and 17,000,000 at 0.3 % (D1's sizes), from new seeds. pc_eval draws through
   the arithmetic eval_seed inside its spawned sampling workers, so new TAG strings alone change nothing: map the seeds
   in the workers as retest/shim/rt_d1.py does, with new TAGs checked by RETEST.md's collision rule (section 2).
2. One read per p. Save per shot, as D1's reads do: obs, det (packed), chunk, each network's decision, S55'-deep's
   margin, and C14''s 12 logits. For the routed shots on which the two decisions differ (about 141 k at D1's rates), also
   save S55'-deep's 4,096 class scores, taken from the fused engine itself (pc_fused overwrites its buffers per
   sub-batch, so export them there, or rerun the same fused engine with a score export on those shots only; another
   inference path would change D1's arithmetic).
3. Split each p's shots by chunk parity: fit half (even chunks), evaluation half (odd). Choose every threshold, weight
   and the single arbiter to carry forward on the fit half only; report all arbiters on the evaluation half.
4. Arbiters, applied only to routed shots where the two decisions differ (everything else as in the composition). The
   fitting objective is the composition's failures on the fit half (repairs minus breaks; "always Cascade" is the
   baseline; shots both decoders fail count zero):
   - A1, two thresholds: take the network's answer if Cascade's confidence < t_C and the network's margin > t_S, with
     Cascade's confidence either min|logit| (A1m) or -sum softplus(-|logit|) (A1s, pc_common.confidence).
   - A2, preference: take the network's answer if dC < a + b * dN, with dC = log P_C(Cascade's answer) -
     log P_C(network's answer) = sum over the differing bits of |z|, and dN = the network's score of its answer minus its
     score of Cascade's answer.
   - A3, joint score over all 4,096 classes: argmax_y [ s_N(y) + w * (y . z) ] (log P_C(y) = y . z + const). A3 may pick
     a third answer; count such repairs and breaks apart from the two-answer ones.
5. Report per p on the evaluation half: failures of Cascade alone, the composition and each arbiter's composition;
   repairs, breaks and third-answer changes; the net improvement (repairs - breaks)/N_eval with a 95 % interval, the
   relative failure reduction and the recovered share of the routed oracle; exact McNemar against the composition; when
   an arbiter shows no break, the one-sided 95 % upper bound 3/N. Also D1's per-category counts for comparison (the
   rates should agree; near-tie decisions may differ between reads).
6. Exploratory: no registration and no claim. A registered test on fresh shots follows only if an arbiter pays off.

## Compute and cost
- One RunPod RTX PRO 6000 (Blackwell Server Edition, 1.69 USD/h, secure cloud), the PI's account, own pod only.
  Watchdog: stop at 15 USD, 5 min after DONE, or when the heartbeat is 25 min old. Every output is copied back to
  the Linux workstation before DONE.
- Time: D1's plan (section 2.5) budgets about 76 min for both eager reads on this card; with setup, sampling (bounded
  prefetch on the pod's CPUs), the score export on ~141 k rows and the analysis, about 2 h, about 3.5 USD.
- Analysis on the pod: features computed once on the disagreement rows (a few hundred thousand), threshold sweeps by
  cumulative counts, detectors kept packed. No computation on the shared Linux workstation.

## Review dispositions (review of 2026-10-16, verdict REVISE; cited by finding number)
Accepted: 1-13, 15, 16, 18 (folded in above; 11 by renting an RTX PRO 6000). 17 accepted in part (features once,
cumulative counts); the analysis stays on the pod (minutes, about 0.1 USD; the Linux workstation is not for computation). Rejected:
14, reusing D1's or the re-test's per-shot outputs as the fit half: they are on Alpine, which is unreachable now.
