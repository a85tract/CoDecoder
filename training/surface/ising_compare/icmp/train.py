"""Training (SETTINGS §7, ASSUMPTIONS I13): nemu DefectTransformer on the verdict circuits (legacy, p = 0.006).

Two phases, one script:
  --phase distill : stored teacher-labelled shots (icmp.gen_labels files: packed detectors, sampled observable, the
                    teacher's label, hard 0/1 or soft P(flip)); target = the teacher's label (BCE on the logit).
  --phase truth   : fresh shots streamed from Stim by CPU worker processes inside the job (seed 10^8 + 10^6 run_id +
                    worker, disjoint from val / test / label seeds); target = the sampled observable.
Batches are bucketed by fired-detector count (a pool of `--pool` batches sorted by count, cut, shuffled). AdamW, warm-up
then cosine to 2 % of the peak, gradient clip 1, bf16 autocast. Every `--eval-every` samples: failures against the
sampled observable on val_legacy chunk 0 (10,000 shots); the best checkpoint (fewest failures, then lower val loss) and
the last are saved under ICMP_CKPT/<run>/. Stops at --samples or --max-hours (then saves last.pt), whichever first.
    python -m icmp.train --run d7_distill --d 7 --phase distill --labels 'label_pm_corr_d7_*.npz' --samples 3e8
    python -m icmp.train --run d7_truth --d 7 --phase truth --init CKPT --samples 3e8 --workers 12"""
import argparse
import csv
import glob
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from icmp import circuits, shots
from icmp.streamer import sampler as _sampler

CKPT = os.environ.get("ICMP_CKPT", os.path.expanduser("~/work/ising/ckpt"))
LABELS = os.environ.get("ICMP_LABELS", os.path.expanduser("~/work/ising/labels"))
OUT = os.environ.get("ICMP_OUT", os.path.join(os.path.dirname(__file__), "..", "data"))   # outside the code dir on the Windows workstation
LOGS = os.path.join(OUT, "train_logs")
RUNS = os.path.join(OUT, "runs.csv")


def unpack(packed, m):
    sh = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> sh) & 1).reshape(packed.shape[0], -1)[:, :m]


class Stream:
    """Fresh shots from `workers` Stim processes; next(k) returns k shots (packed, target). The workers are started
    without re-importing the main module (spawn would otherwise import icmp.train, i.e. torch, ~0.5 GB per worker) and
    are terminated and joined by close(), which main() calls on every exit path."""
    V = 8                                                # virtual streams of the ordered mode

    def __init__(self, d, run_id, workers, chunk=8192, ordered=False):
        """ordered: the RRStream design (branch ising-aux, review of A1): V = 8 virtual streams with Stim seeds
        10^8 + 10^6 run_id + v, worker w owns the streams w, w + workers, ... and cycles them in order, one queue per
        worker read round-robin, so the shot sequence does not depend on worker timing or count (a divisor of V)."""
        import sys
        ctx = mp.get_context("spawn")
        text = str(circuits.circuit(d, noise=circuits.NOISE_VERDICT))
        self.ordered = ordered
        if ordered:
            from icmp.streamer import sampler_rr
            assert self.V % workers == 0, f"--ordered: workers must divide {self.V}"
            self.seeds = [10**8 + 10**6 * run_id + v for v in range(self.V)]
            self.qs = [ctx.Queue(maxsize=4) for _ in range(workers)]
            self.q = self.qs[0]
            self.ps = [ctx.Process(target=sampler_rr, args=(text, self.seeds[w::workers], chunk, q), daemon=True)
                       for w, q in enumerate(self.qs)]
            self.turn = 0
        else:
            self.q = ctx.Queue(maxsize=4 * workers)
            self.ps = [ctx.Process(target=_sampler, args=(text, 10**8 + 10**6 * run_id + w, chunk, self.q), daemon=True)
                       for w in range(workers)]
        main_mod = sys.modules["__main__"]
        saved = main_mod.__dict__.get("__spec__"), main_mod.__dict__.get("__file__")
        main_mod.__spec__ = None                         # spawn: no init_main_from_name / _from_path in the children
        main_mod.__dict__.pop("__file__", None)
        try:
            for p in self.ps:
                p.start()
        finally:
            main_mod.__spec__ = saved[0]
            if saved[1] is not None:
                main_mod.__file__ = saved[1]
        self.buf_x, self.buf_y = [], []
        self.have = 0

    def next(self, k):
        while self.have < k:
            if self.ordered:
                x, y = self.qs[self.turn].get()
                self.turn = (self.turn + 1) % len(self.qs)
            else:
                x, y = self.q.get()
            self.buf_x.append(x)
            self.buf_y.append(y)
            self.have += len(x)
        X, Y = np.concatenate(self.buf_x), np.concatenate(self.buf_y)
        self.buf_x, self.buf_y, self.have = [X[k:]], [Y[k:]], len(X) - k
        return X[:k], Y[:k]

    def close(self):
        for q in getattr(self, "qs", [self.q]):
            q.cancel_join_thread()
        for p in self.ps:
            if p.is_alive():
                p.terminate()
        for p in self.ps:
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join(5)


class Stored:
    """Teacher-labelled shots from icmp.gen_labels files, cycled in random order (epochs)."""
    def __init__(self, pattern, seed):
        files = sorted(glob.glob(os.path.join(LABELS, pattern)))
        assert files, f"no label files {pattern}"
        X, Y = [], []
        for f in files:
            z = np.load(f)
            X.append(z["det"])
            Y.append(z["label"].astype(np.float32))
        self.X, self.Y = np.concatenate(X), np.concatenate(Y)
        self.files, self.rng, self.pos = files, np.random.default_rng(seed), 0
        self.perm = self.rng.permutation(len(self.X))
        self.epochs = 0

    def next(self, k):
        if self.pos + k > len(self.perm):
            self.perm, self.pos = self.rng.permutation(len(self.X)), 0
            self.epochs += 1
        i = np.sort(self.perm[self.pos:self.pos + k])
        self.pos += k
        return self.X[i], self.Y[i]


def build_model(d, cfg):
    from nemu import nets
    from icmp.dt_cached import CachedDT                    # cached geometry dims (review item 10; same outputs)
    ann = circuits.annotated(circuits.memory_circuit(d, noise=circuits.NOISE_VERDICT))
    model = CachedDT(ann, width=cfg["width"], layers=cfg["layers"], heads=cfg["heads"])
    f = nets.Geometry(ann).features()
    model.near_clip = int(max(f[:, 1].max(), f[:, 3].max(), f[:, 5].max()))
    cfg["near_clip"] = model.near_clip
    return model


@torch.no_grad()
def evaluate(model, Xv, ov, m, bs=2048):
    model.eval()
    lg = []
    for i in range(0, len(Xv), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg.append(model(unpack(Xv[i:i + bs], m)).float())
    lg = torch.cat(lg)
    model.train()
    y = torch.as_tensor(ov, device=lg.device, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(lg, y).item()
    fails = int(((lg > 0).float() != y).sum())
    return fails, loss


def _main(state):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-id", type=int, required=True, help="stream seed index (unique per run)")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--phase", choices=["distill", "truth"], required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--init", default="")
    ap.add_argument("--samples", type=float, required=True)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=float, default=2e7)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: checkpoint and stop at the first step after it")
    ap.add_argument("--stop-file", default="", help="checkpoint and stop when this file exists (checked every 100 steps)")
    ap.add_argument("--mem-frac", type=float, default=0, help="cap this process's share of GPU memory")
    ap.add_argument("--init-expand", action="store_true",
                    help="init from a shallower checkpoint of the same width: its layers first, the extra layers as identity")
    ap.add_argument("--reserve-gib", type=int, default=0,
                    help="allocate and free this many GiB before compiling (the Windows workstation's WSL pinned-memory workaround)")
    ap.add_argument("--ordered", action="store_true",
                    help="reproducible data order (controlled comparisons): ordered virtual streams, seeded stable sort "
                         "and batch shuffle (the RRStream design, ising-aux); identical arguments give identical batches")
    ap.add_argument("--ema", type=float, default=0,
                    help="decay of an exponential moving average of the weights per optimizer update (0: off); both "
                         "weight sets are evaluated at every evaluation, the average is saved in each checkpoint, its "
                         "chunk-0 best as best_ema.pt and its final value as final_ema.pt (normal checkpoints of the "
                         "averaged weights; final_ema.pt is the primary read-out, SETTINGS §0.6)")
    a = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(a.run_id)
    dev = torch.device("cuda")
    cfg = {"arch": "dt", "width": a.width, "layers": a.layers, "heads": a.heads}
    model = build_model(a.d, cfg).to(dev)
    m = model.m
    if a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        if a.init_expand:
            L0 = ck["cfg"]["layers"]
            assert ck["cfg"]["width"] == cfg["width"] and ck["cfg"]["heads"] == cfg["heads"] and L0 < cfg["layers"]
            st = model.state_dict()
            for k, v in ck["state"].items():
                if k in ("b_rel", "b_type"):
                    st[k][..., :v.shape[-1]] = v              # bias columns of the first L0 layers; new ones stay 0
                else:
                    st[k] = v
            model.load_state_dict(st)
            with torch.no_grad():                            # extra layers: residual branches output 0 (identity)
                for layer in model.layers[L0:]:
                    for lin in (layer.proj, layer.ff[2]):
                        lin.weight.zero_()
                        lin.bias.zero_()
        else:
            model.load_state_dict(ck["state"])
            for k in ("width", "layers", "heads"):
                assert ck["cfg"][k] == cfg[k], f"init {k} differs"
    if a.reserve_gib:
        torch.empty(a.reserve_gib << 30, dtype=torch.uint8, device=dev)
        torch.cuda.empty_cache()
    if a.compile:
        torch._dynamo.config.recompile_limit = 64
        model.pad_tokens = 32
        model.body = torch.compile(model.body, dynamic=False)
    src = Stored(a.labels, a.run_id) if a.phase == "distill" else Stream(a.d, a.run_id, a.workers, ordered=a.ordered)
    rng = np.random.default_rng(a.run_id) if a.ordered else np.random      # the batch shuffle (seeded if --ordered)
    state["src"] = src
    Xv, ov = shots.load(shots.VAL, a.d, [0], packed=True)
    Xv = torch.as_tensor(Xv, device=dev)
    steps = int(a.samples // a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    ema = [p.detach().clone() for p in params] if a.ema else None   # initialised to the starting weights
    warm = min(a.warmup, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    out = os.path.join(CKPT, a.run)
    os.makedirs(out, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    hp = {**vars(a), "cfg": cfg, "steps": steps, "n_params": sum(p.numel() for p in model.parameters()),
          "circuit_sha256": circuits.sha(circuits.circuit(a.d, noise=circuits.NOISE_VERDICT)),
          "stream_seeds": None if a.phase == "distill" else (src.seeds if a.ordered else
                                                             [10**8 + 10**6 * a.run_id, a.workers]),
          "ordered": a.ordered,
          "label_files": None if a.phase == "truth" else [os.path.basename(f) for f in src.files]}
    print(json.dumps(hp, default=str), flush=True)
    logf = open(os.path.join(LOGS, a.run + ".csv"), "w", newline="")
    lw = None
    eval_every = max(1, int(a.eval_every // a.batch))
    best = best_ema = (10**9, 1e9)
    best_row = best_ema_row = None
    batches = []
    run_loss, run_n, step = 0.0, 0, 0
    model.train()
    for step in range(1, steps + 1):
        if not batches:
            X, Y = src.next(a.batch * a.pool)
            Xg = torch.from_numpy(X).pin_memory().to(dev, non_blocking=True)
            Yg = torch.from_numpy(Y).to(dev, non_blocking=True)
            x_all = unpack(Xg, m)
            order = torch.argsort(x_all.sum(1, dtype=torch.int32), stable=a.ordered)
            chunks = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
            perm = rng.permutation(len(chunks))
            batches = [(x_all[chunks[j]], Yg[chunks[j]]) for j in perm]
        x, y = batches.pop()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        loss = F.binary_cross_entropy_with_logits(lg.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_mul_(ema, a.ema)
                torch._foreach_add_(ema, [p.detach() for p in params], alpha=1 - a.ema)
        if step % 50 == 0:
            run_loss += loss.item()
            run_n += 1
        last = step == steps or (time.time() - t_start) / 3600 > a.max_hours or (a.deadline and time.time() > a.deadline) \
            or (a.stop_file and step % 100 == 0 and os.path.exists(a.stop_file))
        if step % eval_every == 0 or last:
            fails, vloss = evaluate(model, Xv, ov, m)
            row = {"step": step, "samples": step * a.batch, "lr": sched.get_last_lr()[0],
                   "train_loss": run_loss / max(1, run_n), "val_fail": fails, "val_loss": vloss,
                   "wall_s": time.time() - t_start,
                   "epochs": getattr(src, "epochs", 0) + getattr(src, "pos", 0) / max(1, len(getattr(src, "X", [1])))}
            if ema is not None:                          # the averaged weights, same shots, same compiled graph
                with torch.no_grad():
                    keep = [p.detach().clone() for p in params]
                    torch._foreach_copy_(params, ema)
                    fails_e, vloss_e = evaluate(model, Xv, ov, m)
                    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
                    torch._foreach_copy_(params, keep)
                    del keep
                row.update(val_fail_ema=fails_e, val_loss_ema=vloss_e)
            run_loss, run_n = 0.0, 0
            if lw is None:
                lw = csv.DictWriter(logf, fieldnames=list(row))
                lw.writeheader()
            lw.writerow(row)
            logf.flush()
            print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            state = {"state": {k: v for k, v in model.state_dict().items()}, "cfg": cfg, "hp": hp, "row": row}
            if ema is not None:
                state["ema"] = {n: e.detach().clone() for n, e in zip(names, ema)}
                if (fails_e, vloss_e) < best_ema:
                    best_ema, best_ema_row = (fails_e, vloss_e), row
                    torch.save({"state": ema_state, "cfg": cfg, "hp": hp, "row": {**row, "val_fail": fails_e,
                                "val_loss": vloss_e, "weights": f"EMA {a.ema}"}}, os.path.join(out, "best_ema.pt"))
            if (fails, vloss) < best:
                best, best_row = (fails, vloss), row
                torch.save(state, os.path.join(out, "best.pt"))
            torch.save(state, os.path.join(out, "last.pt"))
            if last and ema is not None:                 # the primary read-out (SETTINGS §0.6): the final average
                torch.save({"state": ema_state, "cfg": cfg, "hp": hp, "row": {**row, "val_fail": fails_e,
                            "val_loss": vloss_e, "weights": f"final EMA {a.ema}"}}, os.path.join(out, "final_ema.pt"))
        if last:
            break
    logf.close()
    summ = {"run": a.run, "d": a.d, "phase": a.phase, "width": a.width, "layers": a.layers, "heads": a.heads,
            "init": a.init, "samples_done": step * a.batch, "best_samples": best_row["samples"],
            "best_val_fail_chunk0": best_row["val_fail"], "best_val_loss": best_row["val_loss"],
            "wall_h": (time.time() - t_start) / 3600, "ckpt": os.path.join(out, "best.pt"),
            **({"best_ema_samples": best_ema_row["samples"], "best_ema_val_fail_chunk0": best_ema[0],
                "ckpt_ema": os.path.join(out, "best_ema.pt")} if ema is not None and best_ema_row else {})}
    new = not os.path.exists(RUNS)
    head = list(summ) if new else next(csv.reader(open(RUNS)))   # keep the file's columns (EMA fields: SUMMARY line)
    with open(RUNS, "a", newline="") as fs:
        w = csv.DictWriter(fs, fieldnames=head, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(summ)
    print("SUMMARY", json.dumps(summ), flush=True)


def main():
    """_main with the sampler pool closed on every exit path (normal end, exception, SIGTERM / SIGINT from a stop)."""
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
    os._exit(code)                                  # workers already joined; skip waiting on queue feeder threads


if __name__ == "__main__":
    main()
