"""Train Cascade (bb.cascade_net) on a BB code memory with shots sampled on the fly (bb.cascade_data).

Defaults are the paper's (arXiv:2604.08358, Methods "Training"): batch 3,328, 80,000 steps, BCE per observable averaged
over the observables, the three-stage curriculum to p2 = 0.55 % (first two stages 2 % of the steps), Muon (peak LR 3e-3)
for the backbone's matrices and Lion (peak LR 2e-4) for the rest, 1,000 warm-up steps then a cosine to 1/10 of the
peak at step 50,000, weight decay 3e-3, EMA 0.9998, bfloat16 autocast, gradient-norm clipping (at 1.0: not stated).

Shots: --sampler stim (default) = Stim's circuit sampler in --workers spawned processes, one chunk of B / world shots
per step and rank (seed cascade_data.train_seed(run, rank, step)); --sampler dem = the exact GPU sampler of Stim's
detector error model (cascade_data.DemSampler), same seeds, for hosts whose CPUs cannot keep up. --family depspam_fm
draws the training and monitoring shots from the noise-model fit's variant with that family's own seeds (cascade_data);
the default, uniform, leaves every shot as before.

Ends: completed (final_ema.pt, exit 0); paused at --max-minutes or --pause-step (last.pt, exit 3: the job script
resubmits with --resume); nonfinite loss (exit 4). last.pt (model, EMA, both optimisers, the step, the arguments, the
world size, the log and every segment's host and Stim fingerprint) is also written every --ckpt-minutes. A resumed run
continues the same shot stream (seeds by step), schedule and EMA; the arguments that define the run must not change
(FREE lists those that may), nor the world size. A segment on a host whose Stim fingerprint differs continues with
other shots of the same distribution (a distribution-preserving continuation, not the same shots).

Fixed budget and convergence (stage 2; the re-implementation's notes, section 7): --steps is the schedule's length (the
warm-up, cosine and curriculum are fractions of it) and the paper's fixed budget; --max-steps (> --steps) lets the run go
on past it at the schedule's floor LR and p2, with budget_ema.pt (the EMA at --steps) kept. --monitor names a fixed
block of fresh shots (seed kind 1), evaluated (EMA and raw weights) at the first step at or after every multiple of
--monitor-every samples, at --steps and at the end. --converge W:TOL: at every such multiple past --steps, the run stops
(status converged, final_ema.pt, exit 0) when at every monitor p the EMA failures fell by less than the fraction TOL
since the multiple W samples earlier; otherwise it ends at --max-steps. A resumed segment may set another --max-steps.
--bench N: time N steps after --bench-warmup steps (samples/s with sampling included, data-wait share, peak memory),
print one JSON line, and exit without a checkpoint.

Multi-GPU: torchrun --nproc_per_node G ... (DDP, NCCL; BN statistics per GPU; each rank its own shot stream). Stage 2
trains on one GPU; DDP is a fallback that has not been exercised with --resume."""
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

from bb import cascade_data as cdata
from bb import cascade_net as cnet
from bb import cascade_optim as copt

ROOT = os.path.expanduser(os.environ.get("CASCADE_ROOT", "~/work"))
FREE = {"resume", "max_minutes", "workers", "ckpt_minutes", "bench", "bench_warmup", "log_every", "eval_every",
        "monitor_shots", "monitor_p", "compile", "compile_mode", "act_ckpt", "profile", "conv_impl", "max_steps",
        "pause_step"}


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name: $CASCADE_ROOT/ckpt/<run>")
    ap.add_argument("--run-id", type=int, required=True, help="seed namespace of the run's shots (< 1000)")
    ap.add_argument("--code", default="bb72")
    ap.add_argument("--family", default="uniform", choices=list(cdata.FAMILIES),
                    help="circuit family of the training and monitoring shots (cascade_data; their own seeds)")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--res", default="branch", choices=["branch", "skip"])
    ap.add_argument("--batch", type=int, default=3328, help="global batch (all GPUs)")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--p1", type=float, default=0.002)
    ap.add_argument("--p2", type=float, default=0.0055)
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
    ap.add_argument("--mup-base", type=int, default=256, help="muP base width of the read-out's Lion LR (0 = off)")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--sampler", default="stim", choices=["stim", "dem"])
    ap.add_argument("--workers", type=int, default=8, help="Stim processes per GPU")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="default", help="torch.compile mode (e.g. max-autotune-no-cudagraphs)")
    ap.add_argument("--resid-bf16", action="store_true", help="keep the residual stream in bfloat16 under autocast")
    ap.add_argument("--conv-impl", default="triton", choices=["torch", "triton"], help="bipartite convolution kernels")
    ap.add_argument("--micro", type=int, default=1, help="micro-batches per step (gradient accumulation; BN statistics "
                    "per micro-batch, so a resumed run keeps it)")
    ap.add_argument("--act-ckpt", action="store_true", help="recompute each block in the backward")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000, help="the small monitor's interval in steps")
    ap.add_argument("--monitor-shots", type=int, default=20000, help="small monitor: fresh shots per p (0 = none)")
    ap.add_argument("--monitor-p", default="0.0055,0.003", help="the small monitor's p values")
    ap.add_argument("--monitor", default="", help="fixed monitoring block p:c0:c1[,p:c0:c1]: chunks c0 .. c1-1 of "
                    "10,000 fresh shots (seed kind 1) at each p; replaces the small monitor")
    ap.add_argument("--monitor-every", type=float, default=1e7, help="--monitor: evaluate at the first step at or after "
                    "every multiple of this many samples")
    ap.add_argument("--max-steps", type=int, default=0, help="stopping limit (0 = --steps); past --steps the LR stays "
                    "at its floor and p at p2; a resumed segment may change it")
    ap.add_argument("--converge", default="", help="W:TOL: past --steps, stop when the EMA failures on the --monitor "
                    "block fell by less than TOL at every p since W samples earlier")
    ap.add_argument("--pause-step", type=int, default=0, help="pause (checkpoint, exit 3) at this step (checks, pilots)")
    ap.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms (checks; set "
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8 before the start)")
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--max-minutes", type=float, default=0, help="pause (checkpoint, exit 3) after this wall time")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--bench-warmup", type=int, default=30)
    ap.add_argument("--profile", action="store_true", help="with --bench: torch.profiler over the timed steps")
    return ap


def p_block_to_pl(pb, k=12, R=6):
    """Eq. (1) of the paper: the block failure rate -> P_L per logical qubit per round."""
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


@torch.no_grad()
def evaluate(model, det, obs, dev, bs=4096):
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
    n = len(det)
    lo, hi = wilson(fail, n)
    return {"shots": n, "fail": fail, "p_block": fail / n, "pl": p_block_to_pl(fail / n), "pl_lo": p_block_to_pl(lo),
            "pl_hi": p_block_to_pl(hi), "bce": bce / n}


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


def converge_check(log, rows, k, back, tol, key="ema"):
    """The stopping rule at the monitor point of multiple k. rows: {p: this point's row}; log: the earlier rows, each
    with its multiple "k" and failures row[key]["fail"]. -> None if the point of multiple k - back is missing at some p,
    else {"improvement": {p: 1 - fail_now / fail_then}, "stop": every improvement < tol}."""
    prev = {r["p"]: r for r in log if r.get("k") == k - back and key in r}
    if any(pm not in prev for pm in rows):
        return None
    imp = {str(pm): 1 - r[key]["fail"] / max(1, prev[pm][key]["fail"]) for pm, r in rows.items()}
    return {"improvement": imp, "stop": all(v < tol for v in imp.values())}


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
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    if a.batch % (world * a.micro):
        raise SystemExit("--batch must be a multiple of world size x --micro")
    B = a.batch // world                                 # this rank's shots per step
    limit = max(a.steps, a.max_steps)                    # the stopping limit; a.steps fixes the schedule
    E = a.monitor_every if a.monitor else 0              # the fixed block's interval in samples
    rule = None
    if a.converge:
        if not (a.monitor and E):
            raise SystemExit("--converge needs --monitor")
        w_, tol = (float(x) for x in a.converge.split(":"))
        rule = (max(1, round(w_ / E)), tol)              # (multiples back, tolerance)
    run_dir = os.path.join(ROOT, "ckpt", a.run)
    last_pt = os.path.join(run_dir, "last.pt")
    if not a.bench and os.path.exists(last_pt) != a.resume:
        raise SystemExit(f"{last_pt} {'exists: use --resume or a new --run' if not a.resume else 'is missing'}")
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    from bb import code as bbcode
    layout = bbcode.Layout(a.code)
    cfg = {"code": a.code, "H": a.H, "L": a.L, "res": a.res, "act_ckpt": a.act_ckpt, "resid_bf16": a.resid_bf16,
           "conv_impl": a.conv_impl}
    torch.manual_seed(a.init_seed)
    model = cnet.build(cfg, layout).to(dev)
    k = model.k
    n_par = sum(p.numel() for p in model.parameters())
    mg, lg = copt.param_groups(model, cnet.muon_params(model), a.lr_muon, a.lr_lion, a.wd, a.mup_base)
    muon, lion = copt.Muon(mg), copt.Lion(lg)
    ema = copt.EMA(model, a.ema)
    start, log, segments = 0, [], []
    env = cdata.env_record(a.code)
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
    if a.sampler == "stim":
        stream = cdata.StimStream(a.code, a.run_id, rank, sched, B, a.workers, start=start, family=a.family)
    else:
        stream = cdata.DemSampler(a.code, a.run_id, rank, sched, B, dev, start=start, family=a.family)
    mon = {}
    if rank == 0 and a.monitor and not a.bench:          # the fixed block: chunks c0 .. c1-1 at each p
        for pm, c0, c1 in parse_monitor(a.monitor):
            mon[pm] = cdata.eval_shots(a.code, pm, range(c0, c1), n=10000, kind=1, procs=8, family=a.family)
    elif rank == 0 and a.monitor_shots and not a.bench:
        n = a.monitor_shots
        for pm in (float(x) for x in a.monitor_p.split(",")):
            d, o = cdata.eval_shots(a.code, pm, range((n + 9999) // 10000), n=10000, kind=1, procs=4,
                                    family=a.family)
            mon[pm] = (d[:n], o[:n])
    if rank == 0:
        print(json.dumps({"run": a.run, "params": n_par, "macs_per_shot": cnet.macs_per_shot(a.H, a.L),
                          "world": world, "device": seg["device"], "start": start, "limit": limit,
                          "monitor_shots": {str(pm): len(sh[0]) for pm, sh in mon.items()}, "env": env,
                          "args": vars(a)}), flush=True)

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
        tw = time.time()
        if a.sampler == "stim":
            p, d, o = stream.next()
            det, obs = unpack_le(d, layout.D, dev), unpack_le(o, k, dev).float()
        else:
            p, det, obs = stream.next_unpacked(k)
            obs = obs.float()
        wait += time.time() - tw
        f = copt.lr_factor(step, a.warmup, a.cosine_steps, a.lr_floor)
        for opt in (muon, lion):
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
        for i in range(a.micro):
            sl = slice(i * mb, (i + 1) * mb)
            sync = world == 1 or i == a.micro - 1
            ctx = net.no_sync() if not sync else _null()
            with ctx:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    z = fwd(det[sl])
                loss = F.binary_cross_entropy_with_logits(z.float(), obs[sl])
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
                    cap = stream.capacity(B) if a.sampler == "stim" else None
                    print("BENCH " + json.dumps({
                        "L": a.L, "H": a.H, "batch": a.batch, "world": world, "micro": a.micro, "act_ckpt": a.act_ckpt,
                        "compile": a.compile and a.compile_mode, "resid_bf16": a.resid_bf16, "conv": a.conv_impl, "sampler": a.sampler, "workers": a.workers, "p": p,
                        "device": torch.cuda.get_device_name(dev), "params": n_par,
                        "macs_per_shot": cnet.macs_per_shot(a.H, a.L), "steps": a.bench, "secs": round(dt, 3),
                        "samples_per_s": round(a.bench * a.batch / dt, 1), "step_ms": round(1000 * dt / a.bench, 2),
                        "data_wait": round(wait / dt, 4), "sampler_capacity_per_s": cap and round(cap * world, 1),
                        "max_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2),
                        "max_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 2),
                        "loss": round(float(loss.detach()), 5)}), flush=True)
                break
            continue
        bad = conv = False
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
                r = {"eval": step, "samples": step * a.batch, "p": pm, "k": kk,
                     "ema": evaluate(ema.model, *sh, dev), "raw": evaluate(model, *sh, dev)}
                log.append(r)
                rows[pm] = r
                print(json.dumps(r), flush=True)
            model.train()
            if rule and kk is not None and step > a.steps:
                c = converge_check(log[:-len(rows)], rows, kk, *rule)
                if c is not None:
                    c = {"rule": step, "k": kk, "vs_k": kk - rule[0], "tol": rule[1], **c}
                    log.append(c)
                    print(json.dumps(c), flush=True)
                    conv = c["stop"]
        if step == a.steps and limit > a.steps and rank == 0:
            save(os.path.join(run_dir, "budget_ema.pt"), final=True)    # the fixed budget's EMA network
        stop = bool(a.max_minutes) and time.time() - t0 > 60 * a.max_minutes or 0 < a.pause_step <= step
        if world > 1:                                    # every rank takes the same decision
            import torch.distributed as dist
            fl = torch.tensor([float(stop), float(bad), float(conv)], device=dev)
            dist.all_reduce(fl, op=dist.ReduceOp.MAX)
            stop, bad, conv = (bool(x) for x in fl.tolist())
        if bad:
            status = "nonfinite"
            break
        if conv:
            status = "converged"
            break
        if rank == 0 and (stop or time.time() - t_ck > 60 * a.ckpt_minutes):
            save(last_pt)
            t_ck = time.time()
        if stop:
            status = "paused"
            break
    if status == "running":                              # the loop reached the limit
        status = "completed"
    if a.sampler == "stim":
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
        if status in ("completed", "converged"):
            save(os.path.join(run_dir, "final_ema.pt"), final=True)
        with open(os.path.join(run_dir, "log.json"), "w") as fh:
            json.dump(log, fh)
        print(json.dumps({"status": status, "step": step, "minutes": round((time.time() - t0) / 60, 2)}), flush=True)
    return {"completed": 0, "converged": 0, "paused": 3, "nonfinite": 4}[status]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


if __name__ == "__main__":
    sys.exit(main())
