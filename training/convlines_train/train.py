"""Train a front end from a Stim circuit: one command from your circuit and noise model to a trained ConvLines network,
with the defaults of our recipes (training/LINEAGE.md lists the recipes of the released networks).

    python -m convlines_train.train --circuit my.stim --out runs/my                            # surface code
    python -m convlines_train.train --circuit my_bb.stim --bb 6,6,3,1,2,3,1,2 --out runs/bb     # a BB code

The circuit is a memory experiment of the code with your noise model:
  surface  a rotated surface code with one observable and QUBIT_COORDS (convlines_train.surface says what it needs);
           the network is convlines/net.py's ConvLines, the code of the released surface-code networks.
  bb       a bivariate bicycle code given by --bb l,m,a1,a2,a3,b1,b2,b3 (A = x^a1 + y^a2 + y^a3, B = y^b1 + x^b2 +
           x^b3), in Relay-BP's conventions (convlines_train.bbcode); the network is ConvLinesBB, the code of the
           BB72 and gross networks (training/bb).
Defaults (our recipes; every one can be changed):
  network    surface: 1 block of width 64 (bottleneck 16) up to d = 7, 3 blocks of width 128 at d = 9, 5 above
             (bottleneck 32, 4 heads, stem 3x5x5, blocks 3x3x3); bb: 3 blocks of width 128, bottleneck 32, 4 heads,
             the stem's taps every co-firing offset of the DEM, the blocks' the Tanner offsets at dt = -1, 0, 1.
  loss       surface: BCE(c, y_obs) + mean over the d lines of BCE(z_j, y_j); bb: CE(scores, class) + mean BCE over
             the translates.
  optimizer  AdamW (0.9, 0.99), weight decay 0.01 (bb: matrices only), batch 1,024, peak LR 1e-3 after --warmup steps
             (1,000 surface, 500 bb; at most a tenth of the run), cosine to 2 % of the peak at the last step, gradient
             clip 1, bf16 autocast on a GPU (fp32 on a CPU), EMA 0.999 of the weights: the EMA is the front end.
  data       --samples fresh shots (2e8 in our final stages) in pools of 32 batches; each pool sorted by its shots'
             number of fired detectors, cut into batches, the batches shuffled (the studies' order). Pool k is Stim's
             sample with seed SeedSequence([--seed, k]). --curriculum S:H:R multiplies every noise probability of the
             circuit by S for the first fraction H of the samples, then by a factor rising linearly to 1 over the next
             R (bb default 0.36:0.01:0.01, the S55 networks' 0.2 % -> 0.55 %; surface default none).
  growth     --init-from a trained front end: surface, a smaller distance of the same width (training/surface's
             arch.transfer: stem, blocks and the read-out's projections copied, line embeddings and vote weights
             interpolated, new blocks the identity); bb, the same code with fewer blocks (training/bb's bb.train warm
             start, new blocks the identity, checked at step 0).
Monitoring: every --eval-every samples, the failures of the EMA network alone (and of the current weights) on
--eval-shots fixed shots of the circuit (seed SeedSequence([--seed, 2^32])) go to log.jsonl.
Output in --out: final_ema.pt (the front end; convlines_train.frontend loads it, convlines_train.tune tunes its
threshold), last.pt (the full state at the last pool boundary), log.jsonl, run.json (the settings and the result).
--max-minutes stops at a pool boundary after saving last.pt (exit status 3); --resume continues from last.pt."""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from convlines_train import frontend

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root
FREE = {"resume", "max_minutes", "workers", "compile"}   # settings a resumed run may change (none changes the run)
EVAL_SEED_INDEX = 2 ** 32


def default_arch(kind, d=None):
    if kind == "bb":
        return dict(width=128, bott=32, blocks=3, heads=4)
    if d <= 7:
        return dict(width=64, bott=16, blocks=1, heads=4)
    return dict(width=128, bott=32, blocks=3 if d <= 9 else 5, heads=4)


def unpack(packed, n, dev):
    """Stim's little-endian bit-packed rows (B, ceil(n / 8)) uint8 -> (B, n) uint8 on dev."""
    t = torch.from_numpy(np.ascontiguousarray(packed)).to(dev)
    sh = torch.arange(8, device=dev, dtype=torch.uint8)
    return ((t.unsqueeze(-1) >> sh) & 1).reshape(t.shape[0], -1)[:, :n]


class Surface:
    kind, warmup = "surface", 1000

    def __init__(self, circuit, out):
        from convlines_train import surface
        self.spec = surface.spec(circuit)
        self.train_circuit, col = surface.label_circuit(self.spec)
        self.col = [int(c) for c in col]                  # observable index of each line
        self.n_obs = 1                                     # the circuit's own observable is the training circuit's 0
        self.d = self.spec["d"]

    def build(self, arch):
        return frontend.build("surface", self.spec, arch)

    def param_groups(self, model, wd):
        return [{"params": list(model.parameters()), "weight_decay": wd}]

    def loss(self, model, fwd, det, obs):
        y = obs[:, self.col].float()                       # (B, d) line flips in line order
        c, z = fwd(det)
        return F.binary_cross_entropy_with_logits(c.float(), y[:, self.spec["obs_line"]]) + \
            F.binary_cross_entropy_with_logits(z.float(), y)

    def decide(self, model, det):
        c = model(det)[0].float()
        return (c > 0).to(torch.uint8)[:, None]

    def init_from(self, model, path):
        sys.path.insert(0, os.path.join(ROOT, "training", "surface", "ising_compare"))
        from arch import transfer
        ck = torch.load(path, map_location="cpu", weights_only=True)
        old = {k: ck["arch"][k] for k in ("width", "bott", "heads")}
        assert old == {k: model.cfg[k] for k in old}, f"--init-from: width, bottleneck or heads differ: {old}"
        return {"mapped": transfer.from_smaller(model, ck["state"])}


class BB:
    kind, warmup = "bb", 500

    def __init__(self, circuit, out, params):
        from convlines_train import bbcode
        self.name = bbcode.register(params, circuit)
        path = os.path.join(out, "design.json")
        rep = bbcode.design(self.name, path)
        self.spec = dict(kind="bb", params=[params[0], params[1], list(params[2]), list(params[3])],
                         circuit=str(circuit), design=path, name=self.name)
        self.design = rep
        self.train_circuit = circuit
        self.n_obs = circuit.num_observables
        self.d = None

    def build(self, arch):
        return frontend.build("bb", self.spec, arch)

    def param_groups(self, model, wd):                     # bb.train: decay on matrices, not on head.b*
        dec = [q for n, q in model.named_parameters() if q.ndim >= 2 and not n.startswith("head.b")]
        oth = [q for n, q in model.named_parameters() if not (q.ndim >= 2 and not n.startswith("head.b"))]
        return [{"params": dec, "weight_decay": wd}, {"params": oth, "weight_decay": 0.0}]

    def loss(self, model, fwd, det, obs):
        s, z, g = fwd(det)
        y, cls = model.labels(det, obs, g)
        return F.cross_entropy(s.float(), cls) + F.binary_cross_entropy_with_logits(z.float().reshape(len(z), -1), y)

    def decide(self, model, det):
        return model.decide(model(det)[0].float())[1]

    def init_from(self, model, path):
        from convlines_train import bbcode
        bbcode.frozen()
        from bb.train import step0_check                  # training/bb's warm-start check
        ck = torch.load(path, map_location="cpu", weights_only=True)
        sp = frontend.from_saved(ck["spec"])
        assert sp["params"] == self.spec["params"], f"--init-from: another code {sp['params']}"
        ref = frontend.build("bb", sp, ck["arch"], ck["cfg"])          # its own circuit (the same code)
        ref.load_state_dict(ck["state"])
        diff = {k: (model.cfg.get(k), v) for k, v in ck["cfg"].items()
                if k not in ("code", "p", "blocks", "tblocks", "rbins") and model.cfg.get(k) != v}
        assert not diff, f"--init-from: another base network {diff}"
        res = model.load_state_dict(ck["state"], strict=False)
        new = set(model.state_dict()) - set(ref.state_dict())
        assert not res.unexpected_keys and set(res.missing_keys) == new and \
            all(k.startswith("blocks.") for k in new), f"--init-from: keys {res}"
        dev = next(model.parameters()).device
        det, _ = self.train_circuit.compile_detector_sampler(seed=0).sample(2048, separate_observables=True)
        chk = step0_check(model, ref.to(dev).eval(), det.astype(np.uint8), dev)
        assert chk["equal"], f"--init-from: the step-0 decisions differ from the init network's {chk}"
        return {"new_keys": sorted(new), "step0": chk}


def curriculum_scale(spec, samples, pool_shots):
    if not spec:
        return lambda k: 1.0
    S, H, R = (float(x) for x in spec.split(":"))

    def scale(k):
        f = k * pool_shots / samples
        s = S if f < H else (S + (1 - S) * (f - H) / R if f < H + R else 1.0)
        return round(s, 4)
    return scale


@torch.no_grad()
def failures(task, model, det, obs, amp, dev, bs=2048):
    model.eval()
    n = 0
    for i in range(0, len(det), bs):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            p = task.decide(model, det[i:i + bs])
        n += int((p.to(obs.dtype) != obs[i:i + bs]).any(1).sum())
    return n


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", required=True, help="a Stim circuit file")
    ap.add_argument("--out", required=True, help="the run directory")
    ap.add_argument("--bb", default="", help="a BB code: l,m,a1,a2,a3,b1,b2,b3 (without: a surface code)")
    ap.add_argument("--samples", type=float, default=2e8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--bott", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--pool", type=int, default=32, help="batches per pool")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=None, help="warm-up steps (default 1,000 surface, 500 bb)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--curriculum", default=None, help="S:H:R noise scale S for the first H of the samples, then "
                    "linear to 1 over R ('' for none; default none surface, 0.36:0.01:0.01 bb)")
    ap.add_argument("--init-from", default="", help="a trained front end to start from (growth)")
    ap.add_argument("--eval-shots", type=int, default=100_000)
    ap.add_argument("--eval-every", type=float, default=1e7)
    ap.add_argument("--workers", type=int, default=4, help="Stim sampler processes")
    ap.add_argument("--device", default="")
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast on a GPU")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap


def main():
    a = parser().parse_args()
    import stim
    from convlines_train import bbcode, stream
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    circuit = stim.Circuit.from_file(a.circuit)
    task = BB(circuit, a.out, bbcode.parse(a.bb)) if a.bb else Surface(circuit, a.out)
    arch = default_arch(task.kind, task.d)
    arch.update({k: getattr(a, k) for k in ("width", "bott", "blocks", "heads") if getattr(a, k) is not None})
    if a.curriculum is None:
        a.curriculum = "0.36:0.01:0.01" if task.kind == "bb" else ""
    a.warmup = task.warmup if a.warmup is None else a.warmup
    steps = int(a.samples // a.batch)
    pool_shots = a.batch * a.pool
    last_path = os.path.join(a.out, "last.pt")
    st = None
    if a.resume:
        st = torch.load(last_path, map_location="cpu", weights_only=True)
        diff = {k for k, v in vars(a).items() if k not in FREE and st["args"].get(k) != v}
        assert not diff, f"--resume: settings differ from the run's: {sorted(diff)}"
    scale = curriculum_scale(a.curriculum, a.samples, pool_shots)
    pools = stream.Pools(task.train_circuit, a.seed, pool_shots, scale, start=st["pool"] if st else 0,
                         workers=a.workers)                # forked before any GPU work
    try:
        run(a, task, arch, steps, pools, st, last_path, t_start)
    finally:
        pools.close()


def run(a, task, arch, steps, pools, st, last_path, t_start):
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda" and not a.fp32
    if task.kind == "bb" and dev.type == "cuda":           # as bb.train
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    model = task.build(arch)
    init = {}
    if a.init_from and st is None:
        init = dict(task.init_from(model.to(dev), a.init_from), path=a.init_from,
                    sha256=frontend.sha256(a.init_from))
    model.to(dev).train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(task.param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.99))
    warm = min(a.warmup, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    step, k, minutes_before = 0, 0, 0.0
    if st is not None:
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, k, minutes_before, init = st["step"], st["pool"], st["minutes"], st["init"]
    fwd = torch.compile(model) if a.compile else model
    pe, pm = list(ema.parameters()), list(model.parameters())
    m, n_tr = task.train_circuit.num_detectors, task.train_circuit.num_observables
    from convlines_train.stream import pool_seed
    dv, ov = task.train_circuit.compile_detector_sampler(seed=pool_seed(a.seed, EVAL_SEED_INDEX)).sample(
        a.eval_shots, separate_observables=True)
    dv = torch.as_tensor(dv, dtype=torch.uint8, device=dev)
    ov = torch.as_tensor(ov[:, :task.n_obs], dtype=torch.uint8, device=dev)
    logf = open(os.path.join(a.out, "log.jsonl"), "a")
    eval_steps = max(1, int(a.eval_every // a.batch))
    t0, s0, loss_sum, loss_n = time.time(), step, 0.0, 0
    t1, s1 = t0, step
    stopped = False

    def record(final=False):
        tb, sb = (t1, s1) if final else (t0, s0)       # final: the whole process (evaluations included)
        rate = (step - sb) * a.batch / max(1e-9, time.time() - tb)
        row = dict(step=step, samples=step * a.batch, pool=k, lr=sched.get_last_lr()[0], noise_scale=pools.scale(k),
                   train_loss=loss_sum / loss_n if loss_n else None, ema_failures=failures(task, ema, dv, ov, amp, dev),
                   failures=failures(task, model, dv, ov, amp, dev), eval_shots=a.eval_shots,
                   minutes=minutes_before + (time.time() - t_start) / 60, samples_per_s=rate, final=final)
        model.train()
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        print(json.dumps(row), flush=True)
        return row

    def save_state():
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        step=step, pool=k, minutes=minutes_before + (time.time() - t_start) / 60, init=init,
                        args=vars(a)), last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

    it = iter(pools)
    while step < steps:
        kk, dp, op = next(it)
        assert kk == k, (kk, k)
        det, obs = unpack(dp, m, dev), unpack(op, n_tr, dev)
        order = torch.argsort(det.sum(1, dtype=torch.int32), stable=True)
        chunks = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
        for j in np.random.default_rng([a.seed, k]).permutation(len(chunks)):
            if step >= steps:
                break
            x, o = det[chunks[j]], obs[chunks[j]]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                loss = task.loss(model, fwd, x, o)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                torch._foreach_lerp_(pe, pm, 1.0 - a.ema)
            step += 1
            if step % 10 == 0:
                loss_sum += loss.item()
                loss_n += 1
            if step % eval_steps == 0 and step < steps:
                record()
                t0, s0, loss_sum, loss_n = time.time(), step, 0.0, 0
        k += 1
        if step < steps and a.max_minutes and (time.time() - t_start) / 60 >= a.max_minutes:
            stopped = True
            break
    save_state()
    if stopped:
        print(f"stopped at step {step} of {steps} (pool {k}); continue with --resume", flush=True)
        raise SystemExit(3)
    row = record(final=True)
    info = dict(args=vars(a), steps=steps, samples=steps * a.batch, init=init, final=row,
                torch=str(torch.__version__), stim=str(__import__("stim").__version__), device=str(dev),
                circuit_sha256=frontend.text_sha256(str(task.train_circuit)) if task.kind == "bb" else
                frontend.text_sha256(task.spec["circuit"]))
    path = os.path.join(a.out, "final_ema.pt")
    torch.save(dict(format=frontend.FORMAT, kind=task.kind, spec=frontend.to_saved(
        {k_: v for k_, v in task.spec.items() if k_ != "design"}), arch=arch, cfg=getattr(ema, "cfg", None),
        state={k_: v.detach().cpu() for k_, v in ema.state_dict().items()}, train=info), path + ".tmp")
    os.replace(path + ".tmp", path)
    json.dump(dict(info, front_end=path, front_end_sha256=frontend.sha256(path), arch=arch,
                   design=getattr(task, "design", None)), open(os.path.join(a.out, "run.json"), "w"), indent=1,
              default=str)
    print(f"front end -> {path}: EMA alone {row['ema_failures']} failures of {a.eval_shots} monitor shots", flush=True)


if __name__ == "__main__":
    main()
