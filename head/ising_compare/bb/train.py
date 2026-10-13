"""Training of ConvLines-BB (bb/DESIGN.md section 7) on fresh true-label Stim shots, every translate supervised.
arch.train's recipe: AdamW (0.9, 0.99), weight decay 0.01, linear warm-up then cosine to 2 % of the peak, gradient
clip 1, bf16 autocast, EMA of the weights (the read-out), batch 1,024. The stream: global chunk c (8,192 shots) has the
seed bb.shots.chunk_seed(run_id, p_c, c), p_c = the training noise at that point of the run; W sampler processes take
the chunks c0 + w, c0 + w + W, ... and their queues are read round-robin, so the run id and the noise schedule fix the
shot order whatever W is, and a run can resume at any chunk. Optional noise curriculum --p-ramp P0:F: p rises from P0
to --p over the first F of the samples, on the Relay circuit grid (gross did not learn at its target p in the stage-1
smoke runs: bb/DESIGN.md section 8). Labels: the 12 observables and the exact flips of every translate
(ConvLinesBB.labels). Monitor: block failures of the decision on the monitor chunks (stored chunks if the manifest
lists them, else fresh monitor-split shots sampled here, logged as such), current and EMA weights, at the first chunk
boundary after every --eval-every samples; each monitor row carries the chunk index and a digest of its shots.

Ends (bb/PLAN.md 2.3, 2.8). Checkpoints are taken at chunk boundaries only: run_dir/last.pt holds the weights, the EMA, the
optimizer, the step (which fixes the schedule and the stream position), the sampler environment (sampler_env) and the
log, and --resume continues it exactly, in the same sampler environment only.
  completed   every sample done: final_ema.pt (completed true), a row in runs_bb.csv; exit 0
  early_stop  --early-stop S:F: at the first chunk boundary at or after S samples the EMA network fails more than a
              fraction F of the monitor shots: stopped_ema.pt (completed false), a row; exit 0 (PLAN 2.3's probe stop)
  nonfinite   the loss is not finite (checked every 100 steps and before every checkpoint): a row, no new checkpoint;
              exit 4 (the replica fails, PLAN 2.8)
  paused      --max-minutes reached (the job's end or the drain before maintenance): last.pt only; exit 3
  python -m bb.train --run smoke72 --run-id 91 --code bb72 --p 0.003 --design design_bb72.json --samples 3e6
  python -m bb.train <the same arguments> --resume       (every registered setting must equal the checkpoint's)
gross2 (exploratory, bb/PLAN_gross2.md; every default leaves the run as above): --decay-start F holds the peak LR
from the warm-up's end to F of the steps, then runs the cosine (0 = the schedule above, 1 = flat); --tblocks and
--rbins add bb.net's time blocks and read-out branch R; --init warm-starts (fresh optimizer, schedule and EMA) from a
network that may lack only those modules and extra trunk blocks, all starting at zero, and stops unless the step-0
decisions equal the init network's on the monitor shots. Measurement (no effect on the run): every eval row carries
eval_s (its two evaluations), every step line ends with the rate and the data-wait share of the last 100 steps, and
--profile S:K times K steps from step S with a device sync between phases (data, h2d, labels, fwd, bwd, ema) and
reports the samplers' steady capacity. Uniform speed options, off by default (bit-equal inputs and labels): --lean
takes the labels' detector parity from the forward and updates the EMA in one foreach call; --chunk-h2d moves each
chunk to the device once (pinned). The final files are written to a temporary name and renamed; --resume of a
checkpoint that already ended (completed or early_stop: a crash between it and the export) only exports again.
--stop-at S (gross2 stage d) pauses like --max-minutes at the first chunk boundary at or after S samples and writes the
EMA network there as ema_<samples>.pt; a resumed segment may set another S.
Readout ablation A1 (revision/REVISION_PLAN.md 2.1; every default leaves a run as above): --head joint (the deployed
translate read-out and joint vote) or pooled (bb.net.PooledQueriesBB, 12 logical logits, loss mean_k BCE(z_k, y_k));
--translate-weight scales the translate loss (0: none; with --head pooled > 0 adds a training-only translate read-out).
--seed s takes the torch seed (the trunk), the head seed s + 1,000 and the stream id (bb.shots.chunk_seed) from
a1.seeds instead of --run-id; --init of a trunk file (a1.trunk) loads the trunk alone (--init-sha checks it).
--train-check S: at the first chunk boundary at or after S samples, the decision loss averaged over the last 2,000
steps against its constant-guess value over the same steps (joint: the entropy of the class distribution; pooled: the
mean of the 12 bits' binary entropies); within 1 % = not trained (train_check in log.json; the run goes on). The
check's sums over its window are kept in last.pt, so a resume inside the window completes the same 2,000 steps.
Stream record (A1 on rented hosts, gate finding 4): stream_chain = sha256(previous chain + the chunk's packed detectors
+ packed observables) over every training chunk in order, from 64 zeros; kept in last.pt (a resume continues it), in
every eval row and log.json, and printed after the first 16 chunks (STREAM line) for a1/pod/fingerprint.py's value.
Two runs read the same shots exactly when their chains agree."""
import argparse
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from a1 import seeds as a1seeds
from a1 import trunk as a1trunk
from bb import net, shots

OUT = os.path.expanduser(os.environ.get("BB_OUT", "~/work/out"))
CKPT = os.path.expanduser(os.environ.get("BB_CKPT", "~/work/ckpt"))
CHUNK = 8192
FREE = {"resume", "max_minutes", "workers", "profile", "stop_at"}   # a resumed segment may change these (not the run)


GRID = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006]   # Relay-BP circuit files


def p_of(frac, p, ramp):
    """Training noise at a fraction of the run: ramp = (p0, f) raises p linearly from p0 to p over the first f of the
    samples, snapped down to the circuit grid; ramp None = p throughout."""
    if not ramp or frac >= ramp[1]:
        return p
    target = ramp[0] + (p - ramp[0]) * frac / ramp[1]
    return max(g for g in GRID if g <= target + 1e-12)


def lr_factor(s, steps, warmup, decay_start=0.0):
    """LR multiplier at step s: linear warm-up, the peak until step max(warmup, decay_start x steps), then a cosine to
    2 % at the last step (decay_start 0: the cosine starts right after the warm-up; 1: flat)."""
    s0 = max(warmup, int(decay_start * steps))
    if s < warmup:
        return (s + 1) / warmup
    if s < s0:
        return 1.0
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - s0) / max(1, steps - s0)))


def warm_start(model, path, dev):
    """--init: the network at path into model. Its base (code, widths, heads, taps) must be model's, every one of its
    keys must load, and model's keys it lacks must be exactly those of the modules it does not have (trunk blocks beyond
    its count, time blocks, the read-out branch R; zero-initialised outputs). -> (the init network in eval mode, for the
    step-0 check; the keys left at their initialisation)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    diff = {k: (model.cfg.get(k), v) for k, v in ck["cfg"].items()
            if k not in ("p", "blocks", "tblocks", "rbins") and model.cfg.get(k) != v}
    if diff:
        raise SystemExit(f"--init: another base network (now, init): {diff}")
    ref = net.build(ck["cfg"]).to(dev).eval()
    ref.load_state_dict(ck["state"])
    res = model.load_state_dict(ck["state"], strict=False)
    new = set(model.state_dict()) - set(ref.state_dict())
    if res.unexpected_keys or set(res.missing_keys) != new or \
            not all(k.startswith(("blocks.", "tblocks.", "head.rb.")) for k in new):
        raise SystemExit(f"--init: keys that do not load {res.unexpected_keys}, missing {res.missing_keys}, new {new}")
    return ref, sorted(new)


@torch.no_grad()
def step0_check(model, ref, det, dev, bs=1024):
    """--init's step-0 check: the warm start (its additions at zero) and the init network on the same shots, fp32
    without autocast. equal = no decision differs and every score agrees within 1e-4."""
    model.eval()
    worst, differ = 0.0, 0
    for i in range(0, len(det), bs):
        d = torch.as_tensor(det[i:i + bs], device=dev)
        s, r = model(d)[0], ref(d)[0]
        worst = max(worst, float((s - r).abs().max()))
        differ += int((model.decide(s)[0] != ref.decide(r)[0]).sum())
    model.train()
    return {"shots": len(det), "decisions_differ": differ, "max_abs_score_diff": worst,
            "equal": differ == 0 and worst <= 1e-4}


def sampler_env(code, p):
    """The environment the run's shot stream depends on (Stim regenerates it; PLAN 2.8): the Stim, NumPy and Python
    versions, the venv (path and the sha256 of its sorted name==version list) and a fingerprint of Stim's output (1,024
    shots of this code at p from a fixed seed, which also catches a host whose SIMD width changes the samples).
    Recorded with the run; --resume refuses a different one."""
    import importlib.metadata as md
    import stim
    from qnd import circuits
    pk = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    det, obs = circuits.load(code, "X", p).compile_detector_sampler(seed=20261006).sample(
        1024, separate_observables=True, bit_packed=True)
    return {"stim": stim.__version__, "numpy": np.__version__, "python": platform.python_version(), "venv": sys.prefix,
            "packages_sha256": hashlib.sha256("\n".join(pk).encode()).hexdigest()[:16],
            "stim_fingerprint": hashlib.sha256(det.tobytes() + obs.tobytes()).hexdigest()[:16]}


def sampler(code, p, run_id, q, c0, step, total, ramp):
    """Global chunks c0, c0 + step, ...: CHUNK shots at noise p_c = p_of(c CHUNK / total), seed chunk_seed(run_id, p_c, c)."""
    from qnd import circuits
    circ = {}
    c = c0
    while True:
        t = time.time()
        pc = p_of(c * CHUNK / total, p, ramp)
        if pc not in circ:
            circ[pc] = circuits.load(code, "X", pc)
        det, obs = circ[pc].compile_detector_sampler(seed=shots.chunk_seed(run_id, pc, c)).sample(
            CHUNK, separate_observables=True, bit_packed=True)
        q.put((c, det, obs, time.time() - t))              # + the chunk's sampling time (--profile)
        c += step


class Stream:
    def __init__(self, code, p, run_id, workers, total, ramp, c0=0):
        ctx = mp.get_context("spawn")
        self.qs = [ctx.Queue(maxsize=6) for _ in range(workers)]
        self.ps = [ctx.Process(target=sampler, args=(code, p, run_id, q, c0 + w, workers, total, ramp), daemon=True)
                   for w, q in enumerate(self.qs)]
        for pr in self.ps:
            pr.start()
        self.c0 = self.c = c0
        self.t_sample = []                                 # each chunk's sampling time in its worker

    def next(self):
        """The next chunk in order: (c, det, obs), bit-packed."""
        c, d, o, t = self.qs[(self.c - self.c0) % len(self.qs)].get()
        assert c == self.c, (c, self.c)
        self.c += 1
        self.t_sample.append(t)
        return c, d, o

    def capacity(self):
        """Steady samples/s of all workers (each worker's first chunk, which loads its circuit, left out)."""
        t = self.t_sample[len(self.qs):]
        return len(self.qs) * CHUNK / (sum(t) / len(t)) if t else None

    def close(self):
        for pr in self.ps:
            pr.terminate()


def unpack_le(x, n, dev):
    """Stim's little-endian bit-packed rows (B, ceil(n/8)) uint8 (numpy, or a tensor already on dev) -> (B, n) uint8."""
    t = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
    bits = (t[:, :, None] >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
    return bits.reshape(len(t), -1)[:, :n]


@torch.no_grad()
def evaluate(model, det, obs, dev, bs=4096):
    """Monitor failures of the decision (any logical wrong) and its median margin; for the joint head also the strict
    vote's and the base translate logits' failures."""
    model.eval()
    joint = model.kind == "joint"
    fail = fail_strict = fail_base = 0
    margins = []
    for i in range(0, len(det), bs):
        d = torch.as_tensor(det[i:i + bs], device=dev)
        o = torch.as_tensor(obs[i:i + bs], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            s, z, g = model(d)
        _, bits, mg = model.decide(s)
        fail += int((bits != o).any(1).sum())
        if joint:
            fail_strict += int((model.strict(z, g) != o).any(1).sum())
            fail_base += int(((z[:, :, 0].float() > 0).to(torch.uint8) != o).any(1).sum())
        margins.append(mg.float().cpu())
    model.train()
    m = torch.cat(margins)
    out = {"shots": len(det), "fail": fail, "margin_median": float(m.median())}
    if joint:
        out.update(fail_strict_vote=fail_strict, fail_base_logits=fail_base)
    return out


def entropy_nats(counts):
    """Plug-in entropy (nats) of a histogram."""
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--design", required=True, help="bb.design JSON of this code (taps)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bott", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--samples", type=float, required=True)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=float, default=1e6)
    ap.add_argument("--monitor", default="0", help="monitor chunks (stored if listed in the manifest, else fresh)")
    ap.add_argument("--monitor-shots", type=int, default=10000)
    ap.add_argument("--max-minutes", type=float, default=0, help="pause (checkpoint, exit 3) after this wall time")
    ap.add_argument("--stop-at", type=float, default=0, help="pause (checkpoint, exit 3) at the first chunk boundary at "
                    "or after S samples and write the EMA network as ema_<samples>.pt (gross2 stage d)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--init", default="", help="start from these weights (fine-tuning: fresh optimizer and schedule)")
    ap.add_argument("--resume", action="store_true", help="continue run_dir/last.pt exactly")
    ap.add_argument("--early-stop", default="", help="S:F = stop at the first chunk boundary at or after S samples if "
                    "the EMA network fails more than a fraction F of the monitor shots")
    ap.add_argument("--p-ramp", default="", help="P0:F = noise curriculum: p rises from P0 to --p over the first F of "
                    "the samples (circuit grid); the monitor uses --p")
    ap.add_argument("--decay-start", type=float, default=0.0, help="F = hold the peak LR until F of the steps, then "
                    "the cosine (0: right after the warm-up; 1: flat)")
    ap.add_argument("--tblocks", default="", help="dilations of time-only blocks after the trunk, e.g. 4,8 (bb.net)")
    ap.add_argument("--rbins", default="", help="time-bin edges of the read-out branch R, e.g. 0,4,9,13 (bb.net)")
    ap.add_argument("--profile", default="", help="S:K = time K steps from step S phase by phase (device syncs)")
    ap.add_argument("--lean", action="store_true", help="labels from the forward's detector parity, foreach EMA")
    ap.add_argument("--chunk-h2d", action="store_true", help="one (pinned) host-to-device copy per chunk")
    ap.add_argument("--head", choices=net.HEADS, default="joint", help="A1: the decision read-out (joint: the deployed)")
    ap.add_argument("--translate-weight", type=float, default=1.0, help="A1: weight of the translate loss (0: none)")
    ap.add_argument("--seed", type=int, default=None,
                    help="A1: torch seed, head seed and stream id from seed s (a1.seeds), not from --run-id")
    ap.add_argument("--init-sha", default="", help="the --init file's registered sha256 (checked)")
    ap.add_argument("--train-check", type=float, default=0, help="A1: at this many samples, the decision loss of the "
                    "last 2,000 steps against its constant-guess value (log.json's train_check; the run goes on)")
    return ap


def main():
    ap = parser()
    a = ap.parse_args()
    assert CHUNK % a.batch == 0, "a batch must divide the chunk"
    bpc = CHUNK // a.batch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(CKPT, a.run)
    last_pt = os.path.join(run_dir, "last.pt")
    if os.path.exists(last_pt) != a.resume:
        raise SystemExit(f"{last_pt} {'exists: use --resume or a new --run' if not a.resume else 'is missing'}")
    if not a.resume and os.path.exists(os.path.join(run_dir, "log.json")):
        raise SystemExit(f"run {a.run} has ended (log.json exists); a replica is never restarted (PLAN 2.8)")
    torch.manual_seed(a.run_id if a.seed is None else a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    stem, blk = net.taps_from_design(a.design)
    model = net.ConvLinesBB(a.code, a.p, stem, blk, a.width, a.bott, a.blocks, a.heads,
                            tblocks=[int(x) for x in a.tblocks.split(",") if x],
                            rbins=[int(x) for x in a.rbins.split(",") if x], head=a.head,
                            aux_translates=a.head == "pooled" and a.translate_weight > 0,
                            head_seed=None if a.seed is None else a1seeds.head_seed(a.seed)).to(dev)
    trunk_init = None
    if a.init and not a.resume:
        isha = hashlib.sha256(open(a.init, "rb").read()).hexdigest()
        if a.init_sha and isha != a.init_sha:
            raise SystemExit(f"--init file sha256 {isha} is not the registered {a.init_sha}")
        ick = torch.load(a.init, map_location=dev, weights_only=False)
        if ick.get("kind") == "trunk":                   # A1: seed s's trunk alone; the heads keep their head-seed draw
            a1trunk.load_trunk(model, ick, a.seed, net.trunk_keys)
            trunk_init = {"path": os.path.abspath(a.init), "sha256": isha, "seed": ick["seed"], "kind": "trunk"}
        else:
            ref, new_keys = warm_start(model, a.init, dev)
        del ick
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    nodecay = ("head.b", "aux.b")                         # the read-outs' gate tables (A1's aux like the deployed head)
    decay = [q for n, q in model.named_parameters() if q.ndim >= 2 and not n.startswith(nodecay)]
    other = [q for n, q in model.named_parameters() if not (q.ndim >= 2 and not n.startswith(nodecay))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.99))
    env = sampler_env(a.code, a.p)
    print("sampler env", json.dumps(env), flush=True)
    st = {"step": 0, "seconds": 0.0, "t_data": 0.0, "next_eval": a.eval_every, "log": [], "early_stop": None,
          "segments": [], "status": None, "env": env, "stream_chain": "0" * 64}
    if a.resume:
        ck = torch.load(last_pt, map_location=dev, weights_only=False)
        old = {k: ck["hp"].get(k, ap.get_default(k)) for k in vars(a)}     # a setting added later: its default
        diff = {k: (v, old[k]) for k, v in vars(a).items() if k not in FREE and old[k] != v}
        if diff:
            raise SystemExit(f"--resume with changed settings (now, checkpoint): {diff}")
        if ck["run"]["status"] not in (None, "paused", "completed", "early_stop"):   # None: a periodic checkpoint
            raise SystemExit(f"nothing to resume: the run ended with status {ck['run']['status']}")
        old = ck["run"].get("env") or {}
        diff = {k: (v, old.get(k)) for k, v in env.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"--resume in a different sampler environment (now, checkpoint): {diff}")
        model.load_state_dict(ck["state"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        st = ck["run"]
        if st["status"] in ("completed", "early_stop"):    # ended, but cut off before the export: export only
            print(f"export again: the run ended ({st['status']}) at step {st['step']}", flush=True)
        else:
            st["status"] = None
    info = {"params": net.n_params(model), "macs": net.macs(model), "readout_macs": net.readout_macs(model),
            "cfg": {k: v for k, v in model.cfg.items() if "taps" not in k},
            "stem_taps": len(model.cfg["stem_taps"]), "blk_taps": len(model.cfg["blk_taps"]),
            "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "torch": torch.__version__}
    print("model", json.dumps(info), flush=True)
    # monitor shots
    mon_det, mon_obs, src = [], [], []
    for j in shots.parse_range(a.monitor):
        try:
            d, o = shots.load(a.code, a.p, "monitor", j)
            src.append(f"stored {j}")
        except FileNotFoundError:
            d, o = shots.sample(a.code, a.p, "monitor", j, a.monitor_shots)
            src.append(f"fresh {j}")
        mon_det.append(d[:a.monitor_shots])
        mon_obs.append(o[:a.monitor_shots])
    mon_det, mon_obs = np.concatenate(mon_det), np.concatenate(mon_obs)
    print("monitor", src, len(mon_det), "shots; any-flip rate", float(mon_obs.any(1).mean()), flush=True)
    if trunk_init:                                       # kept in the run's state: a resumed segment logs it too
        st["init"] = trunk_init
        print("init", json.dumps(st["init"]), flush=True)
    elif a.init and not a.resume:
        st["init"] = dict(step0_check(model, ref, mon_det, dev), path=os.path.abspath(a.init),
                          sha256=hashlib.sha256(open(a.init, "rb").read()).hexdigest(), new_keys=new_keys)
        print("init", json.dumps(st["init"]), flush=True)
        if not st["init"]["equal"]:
            raise SystemExit("--init: the step-0 decisions differ from the init network's")
        del ref
    fwd = torch.compile(model.forward_train) if a.compile else model.forward_train   # (decision, translates, g)
    stream_id = a.run_id if a.seed is None else a1seeds.bb_stream(a.seed)
    st.setdefault("stream_id", stream_id)
    print("stream", json.dumps({"stream_id": stream_id, "seed": a.seed, "head_seed": None if a.seed is None else
                                a1seeds.head_seed(a.seed)}), flush=True)
    steps = int(a.samples // a.batch)
    sched = lambda s: lr_factor(s, steps, a.warmup, a.decay_start)
    ramp = tuple(float(x) for x in a.p_ramp.split(":")) if a.p_ramp else None
    es = tuple(float(x) for x in a.early_stop.split(":")) if a.early_stop else None
    step0 = st["step"]
    assert step0 % bpc == 0 or st["status"]
    print(f"{'resume' if a.resume else 'start'} at step {step0} of {steps} (chunk {step0 // bpc})", flush=True)
    stream = None if st["status"] else Stream(a.code, a.p, stream_id, a.workers, a.samples, ramp, c0=step0 // bpc)
    D, K = model.D, model.K
    pr0, prk = (int(float(x)) for x in a.profile.split(":")) if a.profile else (-1, 0)
    prof, lap = {}, [0.0]

    def tick(k, on):                                       # --profile: the time since the last tick, synced, to k
        if on:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            prof[k] = prof.get(k, 0.0) + t - lap[0]
            lap[0] = t
    pe_, pm_ = list(ema.parameters()), list(model.parameters())
    chk = int(math.ceil(a.train_check / a.batch)) - 1 if a.train_check else -1   # A1's check: the step reaching S
    chk_acc = torch.zeros(1 + K, device=dev, dtype=torch.float64)                 # decision loss, the K bits' rates
    chk_cls = torch.zeros(2 ** K, device=dev, dtype=torch.float64)                # the class histogram
    chk_n = 0
    if st.get("chk_state") and "train_check" not in st:  # a resume inside the check's window: its sums so far
        chk_acc += torch.tensor(st["chk_state"]["acc"], device=dev, dtype=torch.float64)
        chk_cls += torch.tensor(st["chk_state"]["cls"], device=dev, dtype=torch.float64)
        chk_n = st["chk_state"]["n"]
    win = [time.time(), st["t_data"], step0 * a.batch]     # the step line's window: time, data-wait, samples
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0, sec0 = time.time(), st["seconds"]
    seg = {"job": os.environ.get("SLURM_JOB_ID", ""), "host": platform.node(), "device": info["device"], "step0": step0}

    def save(step_next):
        st["step"], st["seconds"] = step_next, sec0 + time.time() - t0
        if chk_n and "train_check" not in st:            # the check's sums so far (a resume continues them)
            st["chk_state"] = {"acc": chk_acc.tolist(), "cls": chk_cls.tolist(), "n": chk_n}
        torch.save({"state": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "cfg": model.cfg,
                    "hp": vars(a), "run": st}, last_pt + ".tmp")
        os.replace(last_pt + ".tmp", last_pt)

    for step in range(step0, steps) if stream else ():
        on = pr0 <= step < pr0 + prk
        tick("other" if step > pr0 else "start", on)         # other: logging and checks between steps
        if step % bpc == 0:                                # data-wait = time blocked on the samplers' queues
            td = time.time()
            c, dch, och = stream.next()
            st["t_data"] += time.time() - td
            if st.get("stream_chain") is not None:      # the stream record (module doc; None: a run from before it)
                st["stream_chain"] = hashlib.sha256(bytes.fromhex(st["stream_chain"]) + dch.tobytes()
                                                    + och.tobytes()).hexdigest()
                if c == 15:
                    print("STREAM", json.dumps({"chunks": 16, "chain": st["stream_chain"]}), flush=True)
            tick("data", on)
            if a.chunk_h2d:                                # the chunk on the device once; batches are slices of it
                gd, go = (torch.from_numpy(x).pin_memory().to(dev, non_blocking=True) if dev.type == "cuda"
                          else torch.from_numpy(x) for x in (dch, och))
        i = (step % bpc) * a.batch
        det = unpack_le((gd if a.chunk_h2d else dch)[i:i + a.batch], D, dev)
        obs = unpack_le((go if a.chunk_h2d else och)[i:i + a.batch], K, dev)
        tick("h2d", on)
        for gr in opt.param_groups:
            gr["lr"] = a.lr * sched(step)
        if not a.lean:
            y, cls = model.labels(det, obs)
            tick("labels", on)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            s, z, g = fwd(det)
        tick("fwd", on)
        if a.lean:                                         # the same labels: g is the parity of det @ Gdet
            y, cls = model.labels(det, obs, g)
            tick("labels", on)
        if a.head == "pooled":                             # mean over the K logicals of BCE(z_k, y_k)
            loss_c = F.binary_cross_entropy_with_logits(s.float(), obs.float())
        else:
            loss_c = F.cross_entropy(s.float(), cls)
        if a.translate_weight:
            loss_t = F.binary_cross_entropy_with_logits(z.float().reshape(len(z), -1), y)
            loss = loss_c + a.translate_weight * loss_t
        else:
            loss_t, loss = torch.zeros((), device=dev), loss_c
        if 0 <= chk - step < 2000:                         # A1's not-trained check window (no host sync)
            with torch.no_grad():
                chk_acc += torch.cat([loss_c.detach().double()[None], obs.double().mean(0)])
                chk_cls += torch.bincount(cls, minlength=2 ** K).double()
            chk_n += 1
            if step == chk:
                lv, *rates = (chk_acc / chk_n).tolist()
                hb = [-(r * math.log(r) + (1 - r) * math.log(1 - r)) if 0 < r < 1 else 0.0 for r in rates]
                guess = float(np.mean(hb)) if a.head == "pooled" else entropy_nats(chk_cls.cpu().numpy())
                st["train_check"] = {"run": a.run, "head": a.head, "translate_weight": a.translate_weight,
                                     "seed": a.seed, "p_ramp": a.p_ramp, "samples": (step + 1) * a.batch,
                                     "window_steps": chk_n, "decision_loss": lv, "constant_guess": guess,
                                     "guess_kind": "mean bit entropy" if a.head == "pooled" else "class entropy",
                                     "ratio": lv / guess if guess else None, "trained": bool(guess) and lv < 0.99 * guess}
                print("TRAIN CHECK", json.dumps(st["train_check"]), flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tick("bwd", on)
        with torch.no_grad():
            if a.lean:
                torch._foreach_lerp_(pe_, pm_, 1.0 - a.ema)
            else:
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - a.ema)
        tick("ema", on)
        done = (step + 1) * a.batch
        if on and step == pr0 + prk - 1:
            ms = {k: 1e3 * v / prk for k, v in prof.items() if k != "start"}
            cap = stream.capacity()
            print("profile", json.dumps({"steps": [pr0, prk], "ms_per_step": ms, "step_ms": sum(ms.values()),
                                         "gpu_bound_samples_per_s": a.batch / (sum(ms.values()) - ms.get("data", 0))
                                         * 1e3, "workers": a.workers, "sampler_capacity_samples_per_s": cap,
                                         "chunks_timed": len(stream.t_sample)}), flush=True)
        if step % 100 == 0:
            el = time.time() - t0
            lv = loss.item()
            now = time.time()
            wr, wf = (done - win[2]) / (now - win[0]), (st["t_data"] - win[1]) / (now - win[0])
            win[:] = [now, st["t_data"], done]
            print(f"step {step} samples {done:.3g} p {p_of(c * CHUNK / a.samples, a.p, ramp)} loss {lv:.4f} ({a.head} "
                  f"{loss_c.item():.4f} translates {loss_t.item():.4f}) gnorm {float(gn):.3f} lr {a.lr * sched(step):.2e} "
                  f"{(done - step0 * a.batch) / el:.0f} samples/s data-wait {st['t_data'] / (sec0 + el):.0%} "
                  f"window {wr:.0f} samples/s wait {wf:.3f}", flush=True)
            if not math.isfinite(lv):
                st["status"] = "nonfinite"
                break
        if (step + 1) % bpc and step != steps - 1:
            continue                                       # not a chunk boundary
        last = step == steps - 1
        paused = bool(a.max_minutes) and time.time() - t0 > 60 * a.max_minutes or 0 < a.stop_at <= done
        check = es is not None and st["early_stop"] is None and done >= es[0]
        if not (done >= st["next_eval"] or last or paused or check):
            continue
        if not math.isfinite(loss.item()):
            st["status"] = "nonfinite"
            break
        while st["next_eval"] <= done:
            st["next_eval"] += a.eval_every
        te = time.time()
        r = {"samples": done, "seconds": sec0 + te - t0, "loss": loss.item(), "chunk": c,
             "chunk_sha": hashlib.sha256(dch.tobytes()).hexdigest()[:16], "stream_chain": st.get("stream_chain"),
             "cur": evaluate(model, mon_det, mon_obs, dev), "ema": evaluate(ema, mon_det, mon_obs, dev)}
        r["eval_s"] = time.time() - te
        st["log"].append(r)
        print("eval", json.dumps(r), flush=True)
        if check:
            rate = r["ema"]["fail"] / r["ema"]["shots"]
            st["early_stop"] = {"samples": done, "ema_fail_rate": rate, "threshold": es[1], "fired": rate > es[1]}
            print("early-stop check", json.dumps(st["early_stop"]), flush=True)
            if rate > es[1]:
                st["status"] = "early_stop"
        if last and not st["status"]:
            st["status"] = "completed"
        if paused and not st["status"]:
            st["status"] = "paused"
        seg.update(step1=step + 1, seconds=time.time() - t0)
        if st["status"]:
            st["segments"].append(seg)
        save(step + 1)
        if st["status"]:
            break
    if stream:
        stream.close()
    status = st["status"] or "paused"
    print("status", status, flush=True)
    if dev.type == "cuda":                                # measurement only
        print("memory", json.dumps({"max_allocated_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 3),
                                    "max_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 1e9, 3)}), flush=True)
    if status == "paused":
        done = st["step"] * a.batch
        if 0 < a.stop_at <= done:                         # the trunk's network at the pause, for a cooldown's --init
            fp = os.path.join(run_dir, f"ema_{done}.pt")
            torch.save({"state": ema.state_dict(), "cfg": model.cfg, "hp": vars(a), "samples": done,
                        "completed": False, "status": "paused"}, fp + ".tmp")
            os.replace(fp + ".tmp", fp)
            print("snapshot", json.dumps({"path": fp, "samples": done}), flush=True)
        sys.exit(3)
    if status == "nonfinite":                             # mark the last finite checkpoint as terminal (no resume)
        st["segments"].append(dict(seg, step1=step + 1, seconds=time.time() - t0))
        if os.path.exists(last_pt):
            ck = torch.load(last_pt, map_location="cpu", weights_only=False)
            ck["run"]["status"] = "nonfinite"
            torch.save(ck, last_pt + ".tmp")
            os.replace(last_pt + ".tmp", last_pt)
    done = st["step"] * a.batch
    final = {"completed": "final_ema.pt", "early_stop": "stopped_ema.pt"}.get(status)
    if final:                                             # written whole, then renamed (the log last)
        fp = os.path.join(run_dir, final)
        torch.save({"state": ema.state_dict(), "cfg": model.cfg, "hp": vars(a), "samples": done,
                    "completed": status == "completed", "status": status, "early_stop": st["early_stop"]}, fp + ".tmp")
        os.replace(fp + ".tmp", fp)
    lg = st["log"][-1] if st["log"] else {"cur": {}, "ema": {}}
    row = {"run": a.run, "run_id": a.run_id, "code": a.code, "p": a.p, "status": status, "samples": done,
           "seconds": st["seconds"], "samples_per_s": done / max(st["seconds"], 1e-9), "params": info["params"],
           "macs": info["macs"]["total"], "monitor_shots": len(mon_det), "fail_ema": lg["ema"].get("fail"),
           "fail_cur": lg["cur"].get("fail"), "device": info["device"], "segments": len(st["segments"]),
           "checkpoint": final or ""}
    path = os.path.join(OUT, "runs_bb.csv")
    new = not os.path.exists(path)
    if new or not any((r["run"], r["status"], r["samples"]) == (a.run, status, str(done))   # once (export again)
                      for r in csv.DictReader(open(path))):
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    lj = os.path.join(run_dir, "log.json")
    with open(lj + ".tmp", "w") as fh:
        json.dump({"info": info, "env": st["env"], "log": st["log"], "segments": st["segments"],
                   "early_stop": st["early_stop"], "row": row, **({"init": st["init"]} if "init" in st else {}),
                   **{k: st[k] for k in ("train_check", "stream_id", "stream_chain") if k in st}}, fh, indent=1)
    os.replace(lj + ".tmp", lj)
    print("done", json.dumps(row), flush=True)
    sys.exit(4 if status == "nonfinite" else 0)


if __name__ == "__main__":
    main()
