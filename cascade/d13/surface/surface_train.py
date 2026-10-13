"""Train Cascade-surface (surface.surface_net) on the release's rotated surface-code memory with shots sampled on the fly
(surface.surface_data). bb/cascade_train.py of ising-cascade-net (64b2e25d) with the surface network, circuits and
seeds; the loop, checkpoints, resume rules, monitor, stopping rule and benchmark are unchanged.

Defaults are the paper's (arXiv:2604.08358, Methods "Training") in the literal form the BB pilot chose
(ising_compare/bb/cascade/stage2/PILOT.md: --res skip, --mup-base 0): batch 3,328, 80,000 steps, BCE on the one
logical observable, the three-stage curriculum to p2 = 0.7 % (the surface codes' target; first two stages 2 % of the
steps, p1 = 0.2 %: not stated), Muon (peak LR 3e-3) for the blocks' convolutions and Lion (peak LR 2e-4) for the rest,
1,000 warm-up steps then a cosine to 1/10 of the peak at step 50,000, weight decay 3e-3, EMA 0.9998, bfloat16 autocast,
gradient-norm clipping (at 1.0: not stated). R = d rounds (the release's circuits).

Shots: Stim's circuit sampler in --workers spawned processes, one chunk of B / world shots per step and rank (seed
surface_data.train_seed(d, run, rank, step)).

Ends (status): completed at --max-steps (final_ema.pt, exit 0); no_improvement, the stopping rule below (final_ema.pt,
exit 0); paused at --max-minutes or --pause-step (last.pt, exit 3: the next segment resumes with --resume); nonfinite,
a non-finite loss or weights (exit 4; last.pt keeps the last finite state). At every end rank 0 writes status.json in
the run directory (status, start and end step, job, host, time), which the job scripts read (jobs/cs13.sh): torchrun
turns every non-zero exit into 1. last.pt (model, EMA, both optimisers, the step, the arguments, the world size, the
log and every segment's host and Stim fingerprint) is also written every --ckpt-minutes, and never from a non-finite
state: the loss accumulated since the last log line, the weights and the EMA are checked before every write. A resumed
run continues the same shot stream (seeds by step), schedule and EMA; the arguments that define the run must not
change (FREE lists those that may), nor the world size.

Fixed budget and the stopping rule: --steps is the schedule's length (the warm-up, cosine and curriculum are fractions
of it) and the paper's fixed budget; --max-steps (> --steps) lets the run go on past it at the schedule's floor LR and
p2, with budget_ema.pt (the EMA at --steps) kept. --monitor names a fixed block of fresh shots (seed kind
--monitor-kind), evaluated with the EMA at the first step at or after every multiple of --monitor-every samples, at
--steps and at the end, and with the raw weights too at the points below --raw-until samples (the early-learning
diagnostic, while the EMA still holds much of the initial weights); --monitor-pm decodes the monitor shots once with
correlated PyMatching (the reference row "pm"). --no-improvement W:TOL: at every such multiple past --steps, the run
stops (status no_improvement, final_ema.pt, exit 0) when at every monitor p the EMA failures fell by less than the
fraction TOL since the multiple W samples earlier. A rise counts as well, so the rule reports no observed improvement,
not convergence. Otherwise the run ends at --max-steps. --bench N: time N steps after --bench-warmup steps (samples/s
with sampling included, data-wait share, peak memory), print one JSON line, and exit without a checkpoint. --fault
(checks of the job chain only): nan:S makes the loss of step S non-finite, crash:S raises on rank 0 at step S.

Multi-GPU: torchrun --nproc_per_node G ... (DDP, NCCL; BN statistics per GPU; each rank its own shot stream)."""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from surface import cascade_optim as copt
from surface import surface_data as sdata
from surface import surface_layout as sl
from surface import surface_net as snet

ROOT = os.path.expanduser(os.environ.get("CASCADE_ROOT", "~/work"))
FREE = {"resume", "max_minutes", "workers", "ckpt_minutes", "bench", "bench_warmup", "log_every", "eval_every",
        "monitor_shots", "monitor_p", "monitor_pm", "pm_procs", "compile", "compile_mode", "act_ckpt", "profile",
        "max_steps", "pause_step", "raw_until", "fault"}


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name: $CASCADE_ROOT/ckpt/<run>")
    ap.add_argument("--run-id", type=int, required=True, help="seed namespace of the run's shots (< 1000)")
    ap.add_argument("--d", type=int, default=13)
    ap.add_argument("--L", type=int, default=13)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--res", default="skip", choices=["branch", "skip"])
    ap.add_argument("--batch", type=int, default=3328, help="global batch (all GPUs)")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--p1", type=float, default=0.002)
    ap.add_argument("--p2", type=float, default=0.007)
    ap.add_argument("--f1", type=float, default=0.01, help="share of the steps at p1")
    ap.add_argument("--f2", type=float, default=0.01, help="share of the steps annealing p1 -> p2")
    ap.add_argument("--lr-muon", type=float, default=3e-3)
    ap.add_argument("--lr-lion", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--cosine-steps", type=int, default=50000)
    ap.add_argument("--lr-floor", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=3e-3)
    ap.add_argument("--ema", type=float, default=0.9998)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mup-base", type=int, default=0, help="muP base width of the read-out's Lion LR (0 = off)")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="Stim processes per GPU")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="default", help="torch.compile mode (e.g. max-autotune-no-cudagraphs)")
    ap.add_argument("--resid-bf16", action="store_true", help="keep the residual stream in bfloat16 under autocast")
    ap.add_argument("--micro", type=int, default=1, help="micro-batches per step (gradient accumulation; BN statistics "
                    "per micro-batch, so a resumed run keeps it)")
    ap.add_argument("--act-ckpt", action="store_true", help="recompute each block in the backward")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000, help="the small monitor's interval in steps")
    ap.add_argument("--monitor-shots", type=int, default=20000, help="small monitor: fresh shots per p (0 = none)")
    ap.add_argument("--monitor-p", default="0.006,0.007", help="the small monitor's p values")
    ap.add_argument("--monitor-kind", type=int, default=1, help="seed kind of the monitor shots (surface_data)")
    ap.add_argument("--monitor", default="", help="fixed monitoring block p:c0:c1[,p:c0:c1]: chunks c0 .. c1-1 of "
                    "10,000 fresh shots at each p; replaces the small monitor")
    ap.add_argument("--monitor-every", type=float, default=1e7, help="--monitor: evaluate at the first step at or after "
                    "every multiple of this many samples")
    ap.add_argument("--monitor-pm", action="store_true", help="decode the monitor shots once with correlated PyMatching")
    ap.add_argument("--pm-procs", type=int, default=8, help="processes for --monitor-pm")
    ap.add_argument("--raw-until", type=float, default=0, help="evaluate the raw weights as well at the monitor points "
                    "below this many samples (0 = at every point); the EMA at every point")
    ap.add_argument("--max-steps", type=int, default=0, help="stopping limit (0 = --steps); past --steps the LR stays "
                    "at its floor and p at p2; a resumed segment may change it")
    ap.add_argument("--no-improvement", default="", help="W:TOL: past --steps, stop (status no_improvement) when the "
                    "EMA failures on the --monitor block fell by less than TOL at every p since W samples earlier "
                    "(a rise included)")
    ap.add_argument("--pause-step", type=int, default=0, help="pause (checkpoint, exit 3) at this step (checks, pilots)")
    ap.add_argument("--fault", default="", help="checks of the job chain: nan:S (a non-finite loss at step S) or "
                    "crash:S (an exception on rank 0 at step S)")
    ap.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms (checks; set "
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8 before the start)")
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--max-minutes", type=float, default=0, help="pause (checkpoint, exit 3) after this wall time")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--bench-warmup", type=int, default=30)
    ap.add_argument("--profile", action="store_true", help="with --bench: torch.profiler over the timed steps")
    return ap


def p_block_to_pl(pb, k=1, R=13):
    """Eq. (1) of the paper: the block failure rate -> P_L per logical qubit per round (k = 1 for the surface code)."""
    x = 2 * (1 - pb) ** (1 / k) - 1
    return (1 - x ** (1 / R)) / 2 if x > 0 else float("nan")


def wilson(f, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    ph = f / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def rates(fail, n, R):
    lo, hi = wilson(fail, n)
    return {"shots": n, "fail": fail, "p_block": fail / n, "pl": p_block_to_pl(fail / n, R=R),
            "pl_lo": p_block_to_pl(lo, R=R), "pl_hi": p_block_to_pl(hi, R=R)}


@torch.no_grad()
def evaluate(model, det, obs, dev, bs=1024):
    """-> dict(shots, fail, p_block, pl, pl_lo, pl_hi, bce) of a model in eval mode on (det, obs) numpy uint8."""
    model.eval()
    fail, bce = 0, 0.0
    for i in range(0, len(det), bs):
        d = torch.as_tensor(det[i:i + bs], device=dev)
        o = torch.as_tensor(obs[i:i + bs], device=dev).float()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z = model(d).float()
        fail += int(((z > 0).float() != o).any(1).sum())
        bce += float(F.binary_cross_entropy_with_logits(z, o, reduction="sum")) / o.shape[1]
    return dict(rates(fail, len(det), model.R), bce=bce / len(det))


def unpack_le(x, n, dev):
    """Stim's little-endian bit-packed rows (B, ceil(n/8)) uint8 -> (B, n) uint8 on dev."""
    t = torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
    bits = (t[:, :, None] >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
    return bits.reshape(len(t), -1)[:, :n]


def parse_monitor(spec):
    """'p:c0:c1[,p:c0:c1]' -> [(p, c0, c1)]."""
    out = []
    for part in spec.split(","):
        pm, c0, c1 = part.split(":")
        out.append((float(pm), int(c0), int(c1)))
    return out


def improvement_check(log, rows, k, back, tol, key="ema"):
    """The stopping rule at the monitor point of multiple k. rows: {p: this point's row}; log: the earlier rows, each
    with its multiple "k" and failures row[key]["fail"]. -> None if the point of multiple k - back is missing at some p,
    else {"improvement": {p: 1 - fail_now / fail_then}, "stop": every improvement < tol}. A rise (improvement < 0)
    stops as well: "stop" means no observed improvement, not convergence."""
    prev = {r["p"]: r for r in log if r.get("k") == k - back and key in r}
    if any(pm not in prev for pm in rows):
        return None
    imp = {str(pm): 1 - r[key]["fail"] / max(1, prev[pm][key]["fail"]) for pm, r in rows.items()}
    return {"improvement": imp, "stop": all(v < tol for v in imp.values())}


def finite(loss_acc, *models):
    """True if the accumulated loss and every floating-point parameter and buffer of the models are finite."""
    ts = [loss_acc.reshape(1)] + [v for m in models for v in m.state_dict().values() if v.dtype.is_floating_point]
    return bool(torch.stack([torch.isfinite(t).all() for t in ts]).all())


def all_ranks(flags, dev, world):
    """Each boolean OR-ed over the ranks, so that every rank takes the same decision (the flags as they are if world
    is 1)."""
    if world == 1:
        return flags
    import torch.distributed as dist
    fl = torch.tensor([float(x) for x in flags], device=dev)
    dist.all_reduce(fl, op=dist.ReduceOp.MAX)
    return [bool(x) for x in fl.tolist()]


def main():
    a = parser().parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not a.deterministic   # fixed shapes: cuDNN picks its 3D-convolution kernels once
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    if a.batch % (world * a.micro):
        raise SystemExit("--batch must be a multiple of world size x --micro")
    B = a.batch // world                                 # this rank's shots per step
    limit = max(a.steps, a.max_steps)                    # the stopping limit; a.steps fixes the schedule
    E = a.monitor_every if a.monitor else 0              # the fixed block's interval in samples
    rule = None
    if a.no_improvement:
        if not (a.monitor and E):
            raise SystemExit("--no-improvement needs --monitor")
        w_, tol = (float(x) for x in a.no_improvement.split(":"))
        rule = (max(1, round(w_ / E)), tol)              # (multiples back, tolerance)
    fault = None
    if a.fault:
        fk, fs = a.fault.split(":")
        if fk not in ("nan", "crash"):
            raise SystemExit(f"--fault {a.fault}: nan:S or crash:S")
        fault = (fk, int(fs))
    run_dir = os.path.join(ROOT, "ckpt", a.run)
    last_pt = os.path.join(run_dir, "last.pt")
    if not a.bench and os.path.exists(last_pt) != a.resume:
        raise SystemExit(f"{last_pt} {'exists: use --resume or a new --run' if not a.resume else 'is missing'}")
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    layout = sl.Layout(a.d)
    cfg = {"d": a.d, "R": layout.R, "H": a.H, "L": a.L, "res": a.res, "act_ckpt": a.act_ckpt,
           "resid_bf16": a.resid_bf16}
    torch.manual_seed(a.init_seed)
    model = snet.build(cfg, layout).to(dev)
    k = model.k
    n_par = sum(p.numel() for p in model.parameters())
    mg, lg = copt.param_groups(model, snet.muon_params(model), a.lr_muon, a.lr_lion, a.wd, a.mup_base)
    muon, lion = copt.Muon(mg), copt.Lion(lg)
    ema = copt.EMA(model, a.ema)
    start, log, segments = 0, [], []
    env = sdata.env_record(a.d)
    if a.resume:
        ck = torch.load(last_pt, map_location=dev, weights_only=False)
        ap = parser()                                    # an argument added later: its default
        changed = {kk: (ck["args"].get(kk, ap.get_default(kk)), v) for kk, v in vars(a).items()
                   if kk not in FREE and ck["args"].get(kk, ap.get_default(kk)) != v}
        if changed:
            raise SystemExit(f"--resume with changed arguments (saved, now): {changed}")
        if ck.get("world", 1) != world:
            raise SystemExit(f"--resume with another world size (saved {ck.get('world', 1)}, now {world})")
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        muon.load_state_dict(ck["muon"])
        lion.load_state_dict(ck["lion"])
        start, log, segments = ck["step"], ck["log"], ck.get("segments", [])
        if ck["env"]["stim_fingerprint"] != env["stim_fingerprint"] and rank == 0:
            print(f"note: Stim fingerprint differs from the run's start ({ck['env']}, now {env}): the stream continues "
                  "with other shots of the same distribution", flush=True)
    seg = {"job": os.environ.get("SLURM_JOB_ID", ""), "host": platform.node(), "start": start,
           "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
           "stim_fingerprint": env["stim_fingerprint"]}
    net = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local], broadcast_buffers=False)
    if a.compile and a.act_ckpt:                        # Dynamo cannot trace checkpoint's context_fn: compile each
        for blk in model.blocks:                         # block, checkpoint around it in eager mode
            blk.compile(mode=a.compile_mode)
        fwd = net
    else:
        fwd = torch.compile(net, mode=a.compile_mode) if a.compile else net

    sched = (a.steps, a.p1, a.p2, a.f1, a.f2)
    stream = sdata.StimStream(a.d, a.run_id, rank, sched, B, a.workers, start=start)
    mon = {}
    if rank == 0 and a.monitor and not a.bench:          # the fixed block: chunks c0 .. c1-1 at each p
        for pm, c0, c1 in parse_monitor(a.monitor):
            mon[pm] = sdata.eval_shots(a.d, pm, range(c0, c1), n=10000, kind=a.monitor_kind, procs=8)
    elif rank == 0 and a.monitor_shots and not a.bench:
        n = a.monitor_shots
        for pm in (float(x) for x in a.monitor_p.split(",")):
            dm, om = sdata.eval_shots(a.d, pm, range((n + 9999) // 10000), n=10000, kind=a.monitor_kind, procs=4)
            mon[pm] = (dm[:n], om[:n])
    if rank == 0:
        print(json.dumps({"run": a.run, "params": n_par, "macs_per_shot": snet.macs_per_shot(a.H, a.L, a.d),
                          "world": world, "device": seg["device"], "start": start, "limit": limit,
                          "monitor_shots": {str(pm): len(sh[0]) for pm, sh in mon.items()}, "env": env,
                          "layout": layout.describe(), "args": vars(a)}), flush=True)
        if a.monitor_pm and mon and not any(r.get("pm") for r in log):
            for pm, (dm, om) in mon.items():
                t = time.time()
                pred = sdata.pm_predict(a.d, pm, dm, True, procs=a.pm_procs)
                r = {"pm": "correlated", "p": pm, **rates(int((pred != om[:, 0]).sum()), len(dm), layout.R),
                     "secs": round(time.time() - t, 1)}
                log.append(r)
                print(json.dumps(r), flush=True)

    def save(path, final=False):
        st = {"cfg": cfg, "args": vars(a), "step": step, "log": log, "env": env, "world": world, "status": status,
              "segments": segments + [dict(seg, end=step)], "ema": ema.model.state_dict()}
        if not final:
            st.update(model=model.state_dict(), muon=muon.state_dict(), lion=lion.state_dict())
        torch.save(st, path + ".tmp")
        os.replace(path + ".tmp", path)

    t0 = time.time()
    t_ck = t0
    t_log, wait = time.time(), 0.0
    loss_acc = torch.zeros((), device=dev)
    bench_t = None
    status = "running"                                   # until the loop ends
    step = start
    mb = B // a.micro
    k_done = int(start * a.batch // E) if E else 0       # every multiple of E up to the start has been monitored
    model.train()
    while step < limit:
        if fault == ("crash", step + 1) and rank == 0:
            raise RuntimeError(f"--fault crash:{step + 1}")
        tw = time.time()
        p, d, o = stream.next()
        det, obs = unpack_le(d, layout.D, dev), unpack_le(o, k, dev).float()
        wait += time.time() - tw
        f = copt.lr_factor(step, a.warmup, a.cosine_steps, a.lr_floor)
        for opt in (muon, lion):
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
        for i in range(a.micro):
            sl_ = slice(i * mb, (i + 1) * mb)
            sync = world == 1 or i == a.micro - 1
            ctx = net.no_sync() if not sync else _null()
            with ctx:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    z = fwd(det[sl_])
                loss = F.binary_cross_entropy_with_logits(z.float(), obs[sl_])
                if fault == ("nan", step + 1):
                    loss = loss * float("nan")
                (loss / a.micro).backward()
            loss_acc += loss.detach() / a.micro
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        muon.step()
        lion.step()
        muon.zero_grad(set_to_none=True)
        lion.zero_grad(set_to_none=True)
        ema.update(model)
        step += 1
        if a.bench:
            if step - start == a.bench_warmup:
                torch.cuda.synchronize()
                bench_t, wait = time.time(), 0.0
                torch.cuda.reset_peak_memory_stats()
                if a.profile:
                    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                    prof.__enter__()
            if step - start == a.bench_warmup + a.bench:
                torch.cuda.synchronize()
                dt = time.time() - bench_t
                if a.profile:
                    prof.__exit__(None, None, None)
                    if rank == 0:
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))
                if rank == 0:
                    cap = stream.capacity(B)
                    print("BENCH " + json.dumps({
                        "d": a.d, "L": a.L, "H": a.H, "batch": a.batch, "world": world, "micro": a.micro,
                        "act_ckpt": a.act_ckpt, "compile": a.compile and a.compile_mode, "resid_bf16": a.resid_bf16,
                        "workers": a.workers, "p": p, "device": torch.cuda.get_device_name(dev), "params": n_par,
                        "macs_per_shot": snet.macs_per_shot(a.H, a.L, a.d), "steps": a.bench, "secs": round(dt, 3),
                        "samples_per_s": round(a.bench * a.batch / dt, 1), "step_ms": round(1000 * dt / a.bench, 2),
                        "data_wait": round(wait / dt, 4), "sampler_capacity_per_s": cap and round(cap * world, 1),
                        "max_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2),
                        "max_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 2),
                        "loss": round(float(loss.detach()), 5)}), flush=True)
                break
            continue
        bad = noimp = False
        if step % a.log_every == 0 or step in (a.steps, limit):
            la = float(loss_acc) / max(1, (step - 1) % a.log_every + 1)
            bad = not math.isfinite(la)
            dt = time.time() - t_log
            row = {"step": step, "samples": step * a.batch, "p": p, "lr_f": round(f, 5), "loss": round(la, 5),
                   "sps": round(a.log_every * a.batch / dt, 1), "wait": round(wait / dt, 3),
                   "mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2) if dev.type == "cuda" else 0,
                   "min": round((time.time() - t0) / 60, 2)}
            log.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
            t_log, wait = time.time(), 0.0
            loss_acc.zero_()
        kk = None                                        # the multiple of E reached at this step, if any
        if E and step * a.batch >= (k_done + 1) * E:
            kk = k_done = int(step * a.batch // E)
        if mon and (kk is not None or (not E and step % a.eval_every == 0) or step in (a.steps, limit)):
            rows = {}
            for pm, sh in mon.items():
                r = {"eval": step, "samples": step * a.batch, "p": pm, "k": kk, "ema": evaluate(ema.model, *sh, dev)}
                if not a.raw_until or step * a.batch < a.raw_until:     # the early-learning diagnostic
                    r["raw"] = evaluate(model, *sh, dev)
                log.append(r)
                rows[pm] = r
                print(json.dumps(r), flush=True)
            model.train()
            if rule and kk is not None and step > a.steps:
                c = improvement_check(log[:-len(rows)], rows, kk, *rule)
                if c is not None:
                    c = {"rule": step, "k": kk, "vs_k": kk - rule[0], "tol": rule[1], **c}
                    log.append(c)
                    print(json.dumps(c), flush=True)
                    noimp = c["stop"]
        stop = bool(a.max_minutes) and time.time() - t0 > 60 * a.max_minutes or 0 < a.pause_step <= step
        due = rank == 0 and time.time() - t_ck > 60 * a.ckpt_minutes      # rank 0's checkpoint timer
        stop, bad, noimp, due = all_ranks([stop, bad, noimp, due], dev, world)
        budget = step == a.steps and limit > a.steps
        if (stop or due or budget) and not bad:          # a checkpoint follows: never from a non-finite state
            bad, = all_ranks([not finite(loss_acc, model, ema.model)], dev, world)
        if bad:
            status = "nonfinite"
            break
        if budget and rank == 0:
            save(os.path.join(run_dir, "budget_ema.pt"), final=True)    # the fixed budget's EMA network
        if noimp:
            status = "no_improvement"
            break
        if rank == 0 and (stop or due):
            save(last_pt)
            t_ck = time.time()
        if stop:
            status = "paused"
            break
    if status == "running":                              # the loop reached the limit
        status = "completed"
    if status in ("completed", "no_improvement") and not a.bench:     # the end's checkpoints follow
        bad, = all_ranks([not finite(loss_acc, model, ema.model)], dev, world)
        if bad:
            status = "nonfinite"
    stream.close()
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    if a.bench:
        return 0
    if rank == 0:
        if status != "nonfinite":                        # keep the last good checkpoint
            save(last_pt)
        if status in ("completed", "no_improvement"):
            save(os.path.join(run_dir, "final_ema.pt"), final=True)
        with open(os.path.join(run_dir, "log.json"), "w") as fh:
            json.dump(log, fh)
        st = {"status": status, "step": step, "start": start, "samples": step * a.batch,
              "minutes": round((time.time() - t0) / 60, 2), "job": seg["job"], "host": seg["host"],
              "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(os.path.join(run_dir, "status.json.tmp"), "w") as fh:     # read by the job scripts
            json.dump(st, fh)
        os.replace(os.path.join(run_dir, "status.json.tmp"), os.path.join(run_dir, "status.json"))
        print(json.dumps(st), flush=True)
    return {"completed": 0, "no_improvement": 0, "paused": 3, "nonfinite": 4}[status]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


if __name__ == "__main__":
    sys.exit(main())
