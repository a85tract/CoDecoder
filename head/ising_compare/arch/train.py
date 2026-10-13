"""Stage G1 (arch/README.md): training of a ConvLines network (arch.net) with every logical line supervised.

icmp.train's recipe and data order, with the line labels added:
  --phase truth    fresh shots of arch.stream.LineStream (the ordered virtual streams of icmp.train --ordered with the
                   same run id: the same shots in the same order); targets y_j = the sampled line flips
  --phase distill  stored teacher-labelled shots (icmp.gen_labels files, as icmp.train); targets y_0 = the teacher's label
                   and y_j = y_0 XOR g_j, g_j = y_j XOR y_0 the exact GF(2) function of the detectors (icmp.lines.gmap)
Loss = BCE(c, y_0) + mean_j BCE(z_j, y_j) (c the standard-observable logit, z_j the line logits). Batches as icmp.train
--ordered: a pool of 32 batches of 1,024 shots, stably sorted by fired count, cut, shuffled by a generator seeded with
the run id. AdamW (0.9, 0.99), weight decay 0.01, warm-up then cosine to 2 % of the peak, gradient clip 1, bf16 autocast,
EMA of the weights (--ema), evaluation on val_legacy chunk 0 every --eval-every samples: failures of c > 0 (the
decision), of z_0 > 0 and of the d aligned line logits' mean, for the ordinary and the averaged weights. Checkpoints
under ICMP_CKPT/<run>/: best.pt / best_ema.pt (chunk-0 monitoring only), last.pt, final_ema.pt (the primary read-out,
SETTINGS 0.6), each {'state', 'cfg', 'hp', 'row'} with cfg = the ConvLines configuration; row records the samples done
and whether the run completed its schedule. A run stopped early (--max-hours) writes stopped_ema.pt instead of
final_ema.pt and is never labelled final. --init loads weights only (a new run: fresh optimizer, schedule, EMA started
at the loaded weights, its own stream), not a resume. Every evaluation also logs the read-out's gate saturation and the
scale of the pooled line features on the first 2,000 chunk-0 shots (fp32).

Stage G3 (arch/README.md section 9), --g3-arm Ap / B / C with --phase truth: one continuous run whose labels switch at
pool boundaries (pools of --pool batches, the ordered pools above): pools [0, prefix) true labels (Ap) or correlated PM
(B, C; computed by the stream's samplers); window pools [prefix, prefix + window): --g3-per-pool slots of each pool
(a fixed random subset per seed and pool) carry Tesseract-pool training shots in a fixed order (passes over the pool,
each a fixed permutation per seed), labelled truth (Ap), correlated PM (B) or Tesseract (C), the other slots stream
shots labelled truth (Ap) or correlated PM (B, C); true labels afterwards. A teacher's line targets are y_0 XOR g_j.
--save-pools / --stop-pool write the complete state (weights, EMA, optimizer, schedule, batch generator, pool count:
state_p<pools>.pt) at pool boundaries; --resume continues from one in a new process (the stream skips the consumed
chunks), so B and C share one prefix run. --trace logs every step's learning rate, loss, batch hash and state sums.
Segmented runs (a queue's walltime shorter than the run): --max-hours with --max-hours-at-pool stops at the end of the
pool in which the time ran out, after saving the complete state there; the next segment resumes from it.
    python -m arch.train --run g1_smoke --run-id 700 --d 9 --phase truth --samples 3e7 --lr 1e-3 --ema 0.999 --compile

Readout ablation A1 (revision/REVISION_PLAN.md 2.1; every default leaves a run as above): --head vote (the deployed
read-out), line (the line --line-index alone, default the centre; BCE(c, y_0) only) or pooled (arch.net.PooledQueries;
with --line-weight > 0 a training-only line read-out carries the line loss); --line-weight 0 drops the line loss.
--seed s takes the torch seed (the trunk), the head seed s + 1,000, the batch shuffle and the stream from a1.seeds
instead of --run-id, so every arm of seed s reads the same shots in the same order; --init of a trunk file (a1.trunk)
loads the trunk alone. --p-ramp P0:F reads arch.stream.RampLineStream (p from P0 to 0.6 % over the first F of the
samples). --train-check S compares, at S samples, the decision loss BCE(c, y_0) averaged over the last 2,000 steps with
its constant-guess value, the binary entropy of y_0's rate over the same steps: within 1 % = not trained
(train_check.json in the run's directory; the run goes on). The check's sums over its window are kept in the complete
state, so a resume inside the window completes the same 2,000 steps. Every checkpoint and train_check.json is written
to a temporary name and renamed, so an interruption never leaves a truncated newest file.
Stream record (A1 on rented hosts, gate finding 4): stream_chain = sha256(previous chain + the pool's packed detectors +
packed line flips, as the stream returned them) over every pool in order, from 64 zeros; kept in the complete state (a
resume continues it), in every csv row and the summary, and printed after the first 4 pools = 16 chunks of 8,192 shots
(STREAM line) for a1/pod/fingerprint.py's value. Two runs read the same shots exactly when their chains agree."""
import argparse
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from a1 import seeds as a1seeds
from a1 import trunk as a1trunk
from arch import net
from icmp import circuits, shots
from icmp.train import CKPT, LOGS, OUT, Stored, unpack

RUNS = os.path.join(OUT, "runs_arch.csv")


def save_atomic(obj, path):
    """torch.save to path + '.tmp', then rename: the file at path is always complete."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def chain_next(chain, *arrays):
    """The stream record's next link: sha256(previous chain's bytes + the arrays' bytes), hex."""
    h = hashlib.sha256(bytes.fromhex(chain))
    for x in arrays:
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


@torch.no_grad()
def readout_stats(model, x):
    """Gate saturation and pooled-feature scale of the decision read-out on shots x (fp32, no autocast): mean gate, the
    share of gates above 0.99 / below 0.01, the mean gate sum per (query, head) and the RMS of the pooled values."""
    with torch.autocast("cuda", enabled=False):
        g, a = model.head.gates(model.features(x).float())
    return {"gate_mean": float(g.mean()), "gate_hi": float((g > 0.99).float().mean()),
            "gate_lo": float((g < 0.01).float().mean()), "gate_sum": float(g.sum(-1).mean()),
            "pool_rms": float(a.pow(2).mean().sqrt())}


@torch.no_grad()
def evaluate(model, Xv, ov, m, bs=2000):
    """Val failures and loss of the decision c; for the vote head also line 0's and the aligned lines' mean."""
    model.eval()
    vote = model.kind == "vote"
    cs, zs = [], []
    for i in range(0, len(Xv), bs):
        x = unpack(Xv[i:i + bs], m)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c, z = model(x)
        cs.append(c.float())
        if vote:
            zs.append((z.float() * model.gsign(x)))
    c = torch.cat(cs)
    y = torch.as_tensor(ov, device=c.device, dtype=torch.float32)
    st = readout_stats(model, unpack(Xv[:2000], m))
    model.train()
    out = {"val_fail": int(((c > 0).float() != y).sum()), "val_loss": F.binary_cross_entropy_with_logits(c, y).item()}
    if vote:
        za = torch.cat(zs)
        out.update(val_fail_line0=int(((za[:, 0] > 0).float() != y).sum()),
                   val_fail_linemean=int(((za.mean(1) > 0).float() != y).sum()))
    return {**out, **st}


def _main(state):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-id", type=int, required=True, help="stream seed index (unique per run)")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--phase", choices=["distill", "truth"], required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bott", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--stem-k", default="3,5,5")
    ap.add_argument("--block-k", default="3,3,3")
    ap.add_argument("--line-weight", type=float, default=1.0, help="weight of the mean line loss beside BCE(c, y_0)")
    ap.add_argument("--init", default="")
    ap.add_argument("--init-transfer", default="", help="a trained smaller-distance checkpoint, mapped by arch.transfer")
    ap.add_argument("--init-sha", default="", help="the init file's registered sha256 (checked)")
    ap.add_argument("--samples", type=float, required=True)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=float, default=2e7)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--profile-steps", type=int, default=0, help="stop after this many steps and print throughput")
    ap.add_argument("--g3-arm", choices=["", "Ap", "B", "C"], default="")
    ap.add_argument("--g3-pool", default="", help="arch.tess_pool's training part (tess_pool_d9_train.npz)")
    ap.add_argument("--g3-prefix", type=int, default=580, help="pools before the window")
    ap.add_argument("--g3-window", type=int, default=32, help="window pools")
    ap.add_argument("--g3-per-pool", type=int, default=10_000, help="Tesseract-pool shots per window pool")
    ap.add_argument("--save-pools", default="", help="comma list of pool counts with a complete-state save")
    ap.add_argument("--stop-pool", type=int, default=0, help="stop (complete state saved, not final) after this pool")
    ap.add_argument("--resume", default="", help="a complete state state_p*.pt to continue from")
    ap.add_argument("--trace", default="", help="per-step trace csv")
    ap.add_argument("--max-hours-at-pool", action="store_true",
                    help="a --max-hours stop waits for the end of the pool and saves the complete state there")
    ap.add_argument("--save-every", type=int, default=0, help="also save the complete state every this many pools")
    ap.add_argument("--plateau-kill", type=float, default=0.0,
                    help="stop at the first evaluation if the training loss is not finite or still above this value")
    ap.add_argument("--collapse-kill", type=float, default=0.0,
                    help="stop at any evaluation if the training loss is above this value (a collapse)")
    ap.add_argument("--no-val", action="store_true",
                    help="read no val shot at all: evaluations log the training metrics only, no best.pt")
    ap.add_argument("--head", choices=net.HEADS, default="vote",
                    help="A1: the decision read-out: vote (the deployed), line (one line), pooled")
    ap.add_argument("--line-index", type=int, default=-1, help="A1, --head line: the line read (-1: the centre)")
    ap.add_argument("--seed", type=int, default=None,
                    help="A1: torch seed, head seed, batch shuffle and stream from seed s (a1.seeds), not from --run-id")
    ap.add_argument("--p-ramp", default="", help="A1: P0:F = noise ramp from P0 to the training p over the first F of "
                    "the samples (arch.stream.RampLineStream; a --seed run)")
    ap.add_argument("--train-check", type=float, default=0.0, help="A1: at this many samples, the decision loss of "
                    "the last 2,000 steps against its constant-guess value (train_check.json; the run goes on)")
    a = ap.parse_args()
    RESUME_FREE = {"run", "g3_arm", "resume", "save_pools", "stop_pool", "trace", "workers", "max_hours", "compile",
                   "eval_every", "profile_steps", "init", "max_hours_at_pool", "save_every", "no_val", "init_sha",
                   "plateau_kill", "collapse_kill"}
    t_start = time.time()
    torch.manual_seed(a.run_id if a.seed is None else a.seed)
    dev = torch.device("cuda")
    assert a.head != "line" or a.line_weight == 0, "the single-line head has no line loss: its loss is BCE(c, y_0)"
    assert not a.p_ramp or (a.seed is not None and a.phase == "truth" and not a.g3_arm), "a noise ramp is an A1 run"
    cfg = dict(width=a.width, bott=a.bott, blocks=a.blocks, heads=a.heads,
               stem_k=tuple(int(v) for v in a.stem_k.split(",")), block_k=tuple(int(v) for v in a.block_k.split(",")))
    model = net.ConvLines(a.d, **cfg, head=a.head, line_index=None if a.line_index < 0 else a.line_index,
                          aux_lines=a.head == "pooled" and a.line_weight > 0,
                          head_seed=None if a.seed is None else a1seeds.head_seed(a.seed)).to(dev)
    m, d = model.m, model.d
    init_sha, init_kind = "", ""
    if a.init:                                          # weights only: a new run, not a resume
        init_sha = hashlib.sha256(open(a.init, "rb").read()).hexdigest()
        assert not a.init_sha or init_sha == a.init_sha, f"init file sha256 {init_sha} is not the registered {a.init_sha}"
        ck = torch.load(a.init, map_location=dev, weights_only=False)     # a trusted file, its sha256 checked above
        init_kind = ck.get("kind", "network")
        if init_kind == "trunk":                        # A1: seed s's trunk alone; the heads keep their head-seed draw
            a1trunk.load_trunk(model, ck, a.seed, net.trunk_keys)
        else:
            assert {k: ck["cfg"][k] for k in ("width", "bott", "blocks", "heads")} == \
                {k: model.cfg[k] for k in ("width", "bott", "blocks", "heads")}, "init configuration differs"
            model.load_state_dict(ck["state"])
    if a.init_transfer:                                 # weights of a smaller distance, mapped (arch.transfer)
        from arch import transfer
        init_sha = hashlib.sha256(open(a.init_transfer, "rb").read()).hexdigest()
        assert not a.init_sha or init_sha == a.init_sha, f"init file sha256 {init_sha} is not the registered {a.init_sha}"
        ck = torch.load(a.init_transfer, map_location=dev)
        print("init-transfer:", transfer.from_smaller(model, ck["state"]), flush=True)
    g3 = bool(a.g3_arm)
    assert not g3 or (a.phase == "truth" and not a.init), "G3 arms are truth-phase runs from scratch or a resume"
    per_pool_shots = a.batch * a.pool
    cpp = per_pool_shots // 8192                        # stream chunks per pool
    assert not (g3 or a.resume) or per_pool_shots % 8192 == 0
    pool_sha, P = "", None
    if g3:
        pool_sha = hashlib.sha256(open(a.g3_pool, "rb").read()).hexdigest()
        P = dict(np.load(a.g3_pool))
        n_pool = len(P["obs"])
        need = a.g3_window * a.g3_per_pool
        prs = np.random.default_rng([a.run_id, 9])        # the pool order: passes, each a permutation (same for all arms)
        S = np.concatenate([prs.permutation(n_pool) for _ in range(-(-need // n_pool))])[:need]
        P["label"] = {"Ap": P["obs"], "B": P["pm"], "C": P["tess"]}[a.g3_arm]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
    steps = int(a.samples // a.batch)
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    ema = [p.detach().clone() for p in params] if a.ema else None
    warm = min(a.warmup, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = np.random.default_rng(a.run_id if a.seed is None else a1seeds.shuffle_seed(a.seed))  # the batch shuffle
    start, pools_built, st = 0, 0, None
    if a.resume:                                         # the complete state of an earlier run (G3: B's prefix for C)
        st = torch.load(a.resume, map_location=dev, weights_only=False)
        diff = {k for k in set(st["args"]) | set(vars(a)) if k not in RESUME_FREE     # a later flag: its default
                and st["args"].get(k, ap.get_default(k)) != vars(a).get(k)}
        assert not diff and st["pool_sha256"] == pool_sha, f"resume: settings differ {sorted(diff)}"
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        for e, v in zip(ema, st["ema"]):
            e.copy_(v)
        rng.bit_generator.state = st["rng"]
        start, pools_built = st["step"], st["pools"]
        assert start == pools_built * a.pool, "resume at a pool boundary"
    fwd = torch.compile(model.forward_train) if a.compile else model.forward_train     # (c, line logits or None)
    stream_id = None
    if a.phase == "truth":
        from arch.stream import LineStream, RampLineStream
        upto = (a.g3_prefix + a.g3_window) * cpp if a.g3_arm in ("B", "C") else 0
        if a.p_ramp:
            assert per_pool_shots % 8192 == 0
            stream_id = a1seeds.surface_ramp_stream(a.seed)
            src = RampLineStream(a.d, stream_id, a.workers, steps * a.batch, circuits.P_VERDICT,
                                 tuple(float(v) for v in a.p_ramp.split(":")), skip_chunks=pools_built * cpp)
        else:
            stream_id = a.run_id if a.seed is None else a1seeds.surface_stream(a.seed)
            src = LineStream(a.d, stream_id, a.workers, skip_chunks=pools_built * cpp, teacher_upto=upto)
    else:
        src = Stored(a.labels, a.run_id)
    state["src"] = src
    Xv = ov = None
    if not a.no_val:                                     # --no-val: the run reads no val shot at all
        Xv, ov = shots.load(shots.VAL, a.d, [0], packed=True)
        Xv = torch.as_tensor(Xv, device=dev)
    out = os.path.join(CKPT, a.run)
    os.makedirs(out, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    mcfg = dict(model.cfg)
    hp = {**vars(a), "cfg": mcfg, "steps": steps, "n_params": net.n_params(model), "torch": str(torch.__version__),
          "init_sha256": init_sha, "init_is_resume": False, "resume_from": a.resume,
          "resume_sha256": hashlib.sha256(open(a.resume, "rb").read()).hexdigest() if a.resume else "",
          "resume_step": start, "g3_pool_sha256": pool_sha,
          "circuit_sha256": circuits.sha(circuits.circuit(a.d, noise=circuits.NOISE_VERDICT)),
          "stream_seeds": src.seeds if a.phase == "truth" else None, "stream_id": stream_id, "init_kind": init_kind,
          "a1": None if a.seed is None else {"seed": a.seed, "head_seed": a1seeds.head_seed(a.seed),
                                             "shuffle_seed": a1seeds.shuffle_seed(a.seed), "head_macs":
                                             net.head_macs(mcfg, m), "trunk_keys": len(net.trunk_keys(model))},
          "label_files": None if a.phase == "truth" else [os.path.basename(f) for f in src.files]}
    print(json.dumps(hp, default=str), flush=True)
    csv_path = os.path.join(LOGS, a.run + ".csv")       # a resume appends to its run's csv (rows before it kept)
    csv_keep = bool(a.resume) and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    logf = open(csv_path, "a" if csv_keep else "w", newline="")
    save_at = {int(v) for v in a.save_pools.split(",") if v} | ({a.stop_pool} if a.stop_pool else set())
    trf = open(a.trace, "w") if a.trace else None
    if trf:
        trf.write("step,lr,loss,x_sha1,y_sha1,pool_label_errors,param_sum,ema_sum,adam_v_sum\n")
    pool_err = ""

    def save_state(pools):
        save_atomic({"step": step, "pools": pools, "model": model.state_dict(), "ema": [e.detach().clone() for e in ema],
                     "names": names, "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                     "args": vars(a), "pool_sha256": pool_sha, "best": best, "best_ema": best_ema,
                     "stream_chain": chain, "chk": {"acc": chk_acc.tolist(), "n": chk_n}},
                    os.path.join(out, f"state_p{pools}.pt"))
    lw = None
    eval_every = max(1, int(a.eval_every // a.batch))
    best = best_ema = (10**9, 1e9)
    if st is not None:
        best, best_ema = tuple(st["best"]), tuple(st["best_ema"])
    chk = int(math.ceil(a.train_check / a.batch)) if a.train_check else 0      # A1's not-trained check step
    chk_acc = torch.zeros(2, device=dev, dtype=torch.float64)                   # decision loss, y_0 rate (summed)
    chk_n = 0
    chain = "0" * 64                                                            # the stream record (module doc)
    if st is not None:                       # a resume: the chain so far (None: a state from before the record) and
        chain = st.get("stream_chain")       # the check's sums if it stopped inside the check's window
        if st.get("chk") and not os.path.exists(os.path.join(out, "train_check.json")):
            chk_acc += torch.tensor(st["chk"]["acc"], device=dev, dtype=torch.float64)
            chk_n = st["chk"]["n"]
    batches = []
    run_loss, run_n, step, t_data = 0.0, 0, 0, 0.0
    model.train()
    t_loop = time.time()
    step = start
    for step in range(start + 1, steps + 1):
        if not batches:
            t0 = time.time()
            X, Y, *T = src.next(per_pool_shots)
            t_data += time.time() - t0
            if chain is not None:
                chain = chain_next(chain, X, Y)
                if pools_built == 3:                    # this is the 4th pool: 16 chunks of 8,192 shots
                    print("STREAM", json.dumps({"pools": 4, "chunks": 16, "chain": chain}), flush=True)
            if g3:                                       # this pool's labels (and window shots) by the arm's plan
                pidx = pools_built
                y0 = np.unpackbits(Y, axis=1)[:, 0].copy()
                if a.g3_arm in ("B", "C") and pidx < a.g3_prefix + a.g3_window:
                    assert T[0] is not None, "teacher labels missing"
                    y0 = T[0].copy()
                if a.g3_prefix <= pidx < a.g3_prefix + a.g3_window:
                    w = pidx - a.g3_prefix
                    idx = S[w * a.g3_per_pool:(w + 1) * a.g3_per_pool]
                    slots = np.random.default_rng([a.run_id, 10, w]).choice(per_pool_shots, a.g3_per_pool, replace=False)
                    X = X.copy()
                    X[slots] = P["det"][idx]
                    y0[slots] = P["label"][idx]
                    truth0 = np.unpackbits(Y, axis=1)[:, 0].copy()
                    truth0[slots] = P["obs"][idx]
                else:
                    truth0 = np.unpackbits(Y, axis=1)[:, 0]
                pool_err = int((y0 != truth0).sum())           # labels that differ from the true observable
            pools_built += 1
            Xg = torch.from_numpy(X).pin_memory().to(dev, non_blocking=True)
            x_all = unpack(Xg, m)
            if g3:
                g = (1.0 - model.gsign(x_all)) * 0.5
                Yl = (torch.from_numpy(y0).to(dev).float()[:, None] + g) % 2.0         # y_0 XOR g_j on every line
            elif a.phase == "truth":
                Yl = unpack(torch.from_numpy(Y).to(dev), d).float()                # (N, d) line flips
            else:
                y0 = torch.from_numpy(Y).to(dev).float()
                g = (1.0 - model.gsign(x_all)) * 0.5
                Yl = (y0[:, None] + g) % 2.0                                        # teacher's label on every line
            order = torch.argsort(x_all.sum(1, dtype=torch.int32), stable=True)
            chunks = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
            perm = rng.permutation(len(chunks))
            batches = [(x_all[chunks[j]], Yl[chunks[j]]) for j in perm]
        x, yl = batches.pop()
        lr_now = sched.get_last_lr()[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c, zl = fwd(x)
        loss_dec = F.binary_cross_entropy_with_logits(c.float(), yl[:, 0])
        loss = loss_dec if a.line_weight == 0 else \
            loss_dec + a.line_weight * F.binary_cross_entropy_with_logits(zl.float(), yl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_mul_(ema, a.ema)
                torch._foreach_add_(ema, [p.detach() for p in params], alpha=1 - a.ema)
        if chk and chk - 2000 < step <= chk:
            with torch.no_grad():
                chk_acc += torch.stack([loss_dec.detach().double(), yl[:, 0].double().mean()])
            chk_n += 1
            if step == chk:
                lv, r = (chk_acc / chk_n).tolist()
                hb = -(r * math.log(r) + (1 - r) * math.log(1 - r)) if 0 < r < 1 else 0.0
                rec = {"run": a.run, "head": a.head, "line_weight": a.line_weight, "seed": a.seed, "p_ramp": a.p_ramp,
                       "samples": step * a.batch, "window_steps": chk_n, "decision_loss": lv, "y0_rate": r,
                       "constant_guess": hb, "ratio": lv / hb if hb else None,
                       "trained": bool(hb) and lv < 0.99 * hb}
                with open(os.path.join(out, "train_check.json.tmp"), "w") as fh:
                    json.dump(rec, fh, indent=1)
                os.replace(os.path.join(out, "train_check.json.tmp"), os.path.join(out, "train_check.json"))
                print("TRAIN CHECK", json.dumps(rec), flush=True)
        if step % 50 == 0:
            run_loss += loss.item()
            run_n += 1
        if trf:
            with torch.no_grad():
                hx = hashlib.sha1(x.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                hy = hashlib.sha1(yl.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                vs = sum(float(opt.state[p]["exp_avg_sq"].double().sum()) for p in params)
                trf.write(f"{step},{lr_now!r},{loss.item()!r},{hx},{hy},{pool_err},"
                          f"{float(sum(p.double().sum() for p in params))!r},"
                          f"{float(sum(e.double().sum() for e in ema))!r},{vs!r}\n")
                pool_err = ""
        if a.profile_steps and step == a.profile_steps // 2:            # after compilation and warm-up
            torch.cuda.synchronize()
            t_half, d_half = time.time(), t_data
        if a.profile_steps and step == a.profile_steps:
            torch.cuda.synchronize()
            dt = time.time() - t_half
            print(f"PROFILE steps={step} samples/s={(step - a.profile_steps // 2) * a.batch / dt:.0f} (second half) "
                  f"data_wait={(t_data - d_half) / dt:.3f} max_mem_GB={torch.cuda.max_memory_allocated() / 2**30:.1f}",
                  flush=True)
            return
        done = step == steps
        over = (time.time() - t_start) / 3600 > a.max_hours
        if over and a.max_hours_at_pool and batches:            # run on to the end of this pool, then save and stop
            over = False
        pool_end = not batches and (pools_built in save_at or (over and a.max_hours_at_pool)
                                    or (a.save_every and pools_built % a.save_every == 0))
        last = done or over or (pool_end and pools_built == a.stop_pool)
        if pool_end:
            save_state(pools_built)
        if step % eval_every == 0 or last:
            ev = {} if a.no_val else evaluate(model, Xv, ov, m)
            row = {"step": step, "samples": step * a.batch, "completed": done, "lr": sched.get_last_lr()[0],
                   "train_loss": run_loss / max(1, run_n), **ev, "wall_s": time.time() - t_start,
                   "samples_per_s": (step - start) * a.batch / (time.time() - t_loop), "data_wait": t_data / (time.time() - t_loop),
                   "pools": pools_built, "stream_chain": chain}
            if ema is not None:
                with torch.no_grad():
                    keep = [p.detach().clone() for p in params]
                    torch._foreach_copy_(params, ema)
                    eve = {} if a.no_val else evaluate(model, Xv, ov, m)
                    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
                    torch._foreach_copy_(params, keep)
                    del keep
                row.update({k + "_ema": v for k, v in eve.items()})
            run_loss, run_n = 0.0, 0
            if lw is None:
                lw = csv.DictWriter(logf, fieldnames=list(row))
                if not csv_keep:
                    lw.writeheader()
            lw.writerow(row)
            logf.flush()
            print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if a.plateau_kill and (not math.isfinite(row["train_loss"]) or
                                   (start == 0 and step <= eval_every and row["train_loss"] > a.plateau_kill)):
                print(f"PLATEAU KILL: training loss {row['train_loss']} at step {step} (limit {a.plateau_kill})", flush=True)
                raise SystemExit(3)
            if a.collapse_kill and (not math.isfinite(row["train_loss"]) or row["train_loss"] > a.collapse_kill):
                print(f"COLLAPSE KILL: training loss {row['train_loss']} at step {step} (limit {a.collapse_kill})", flush=True)
                raise SystemExit(3)
            ck = {"state": {k: v for k, v in model.state_dict().items()}, "cfg": mcfg, "hp": hp, "row": row}
            if ema is not None:
                ck["ema"] = {n: e.detach().clone() for n, e in zip(names, ema)}
                if not a.no_val and (eve["val_fail"], eve["val_loss"]) < best_ema:
                    best_ema = (eve["val_fail"], eve["val_loss"])
                    save_atomic({"state": ema_state, "cfg": mcfg, "hp": hp, "row": row}, os.path.join(out, "best_ema.pt"))
            if not a.no_val and (ev["val_fail"], ev["val_loss"]) < best:
                best = (ev["val_fail"], ev["val_loss"])
                save_atomic(ck, os.path.join(out, "best.pt"))
            save_atomic(ck, os.path.join(out, "last.pt"))
            if last and ema is not None:              # final only if the schedule ran to its end
                save_atomic({"state": ema_state, "cfg": mcfg, "hp": hp, "row": row},
                            os.path.join(out, "final_ema.pt" if done else "stopped_ema.pt"))
        if last:
            break
    logf.close()
    if trf:
        trf.close()
    summ = {"run": a.run, "d": a.d, "phase": a.phase, "cfg": json.dumps(mcfg), "samples_done": step * a.batch,
            "head": a.head, "seed": a.seed, "line_weight": a.line_weight, "p_ramp": a.p_ramp,
            "samples_planned": steps * a.batch, "completed": step == steps,
            "best_val_fail_chunk0": best[0], "best_ema_val_fail_chunk0": best_ema[0],
            "wall_h": (time.time() - t_start) / 3600, "stream_chain": chain}
    new = not os.path.exists(RUNS)
    with open(RUNS, "a", newline="") as fs:
        w = csv.DictWriter(fs, fieldnames=list(summ))
        if new:
            w.writeheader()
        w.writerow(summ)
    print("SUMMARY", json.dumps(summ), flush=True)


def main():
    """_main with the sampler pool closed on every exit path (icmp.train.main's scheme)."""
    import signal
    import sys
    state = {}

    def _stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _stop)
    code = 0
    try:
        _main(state)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        src = state.get("src")
        if hasattr(src, "close"):
            src.close()
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
