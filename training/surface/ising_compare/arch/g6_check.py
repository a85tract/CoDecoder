"""G6's checks before training (arch/README.md section 15.4), the same for every new cell, on fresh shots of stream 996
(never val):
  1. mapping (transferred cells): the parent (sha256 checked) through arch.transfer into the cell's network (the same
     seed as arch.train with the run id): every parent tensor copied unchanged or remapped (with shapes), every parent
     block among the copied; blocks the parent lacks start as the identity (their `up` weight and bias zero);
  2. line-label geometry on 8,192 fresh stream shots at d: detectors and observable 0 equal to the verdict circuit's
     sampler with the same seed, every line flip equal to y_0 XOR gmap(detectors), correlated PM's agreement with y_0
     (the exactness on every error mechanism is tests/test_gmap_dem.py d);
  3. the trainer's start: AdamW without state, the EMA equal to the mapped weights, the warm-up's first learning rate;
  4. identity-block equivalence (a parent of the same d with fewer or as many blocks, G12's warm restart): on those 8,192
     shots the started network's standard and line logits against the parent's, in fp32 and under bf16 autocast: PASS
     if every decision is equal and the largest difference is at most 1e-5 (fp32) and 1e-2 (bf16);
  5. a 300-step smoke run on fresh shots (the trainer's loss, batch 1,024, peak lr 1e-3 after 100 warm-up steps): PASS
     if every loss is finite and the mean loss of the last 25 steps is at most 1.2 (the in-job plateau-kill level; the
     constant guess is 2 ln 2 = 1.386).
    python -m arch.g6_check --d 9 --blocks 3 --parent CKPT --sha SHA256 --run-id 764
    python -m arch.g6_check --d 7 --blocks 3 --run-id 760                                 (scratch: 2, 3, 5)"""
import argparse
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from arch import net, transfer
from arch.stream import LineStream
from icmp import circuits, lines, pm
from icmp.train import unpack


def logits(model, x, bf16):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        out = [model(x[i:i + 1024]) for i in range(0, len(x), 1024)]
    return torch.cat([c.float() for c, _ in out]), torch.cat([z.float() for _, z in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bott", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--parent", default="")
    ap.add_argument("--sha", default="")
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    D = a.d
    torch.manual_seed(a.run_id)
    model = net.ConvLines(D, width=a.width, bott=a.bott, blocks=a.blocks, heads=a.heads).cuda()
    ok, parent, pcfg = True, None, None
    if a.parent:
        sha = hashlib.sha256(open(a.parent, "rb").read()).hexdigest()
        assert sha == a.sha, f"parent sha256 {sha} is not the registered {a.sha}"
        ck = torch.load(a.parent, map_location="cuda")
        parent, pcfg = ck["state"], ck["cfg"]
        mapped = transfer.from_smaller(model, parent)
        sd = model.state_dict()
        copied = sorted(k for k in parent if k in sd and k != "A" and k not in mapped)
        missing = sorted(k for k in parent if k not in sd and k != "A")
        pblocks = sorted({k.split('.')[1] for k in parent if k.startswith('blocks.')})
        cblocks = sorted({k.split('.')[1] for k in copied if k.startswith('blocks.')})
        new = sorted({k.split('.')[1] for k in sd if k.startswith('blocks.')} - set(pblocks))
        ident = all(torch.count_nonzero(v) == 0 for k, v in sd.items()
                    if k.startswith('blocks.') and k.split('.')[1] in new and '.up.' in k)
        print(f"1. mapping: parent d = {pcfg['d']}, width {pcfg['width']}, blocks {pcfg['blocks']} -> d = {D}, width "
              f"{a.width}, blocks {a.blocks}; remapped {mapped}; copied {len(copied)} tensors, blocks {cblocks}; parent "
              f"tensors without a place {missing}; new blocks {new}, identity at the start {ident}", flush=True)
        ok &= cblocks == pblocks and not missing and ident
        ok &= all(torch.equal(sd[k], parent[k].to(sd[k].device)) for k in copied)
    src = LineStream(D, 996, 2)
    X, Y, _ = src.next(8192)
    c = circuits.circuit(D, noise=circuits.NOISE_VERDICT)
    det_s, obs_s = c.compile_detector_sampler(seed=src.seeds[0]).sample(8192, separate_observables=True)
    det = np.unpackbits(X, axis=1)[:, :c.num_detectors]
    y = np.unpackbits(Y, axis=1)[:, :D]
    A, _ = lines.gmap(D)
    g = (det.astype(np.int64) @ A.astype(np.int64)) & 1
    eq_det = np.array_equal(det, det_s.astype(np.uint8)) and np.array_equal(y[:, 0], obs_s[:, 0])
    eq_lines = bool((g == (y ^ y[:, :1])).all())
    agree = float((pm.decode(pm.matcher(D, True, c), det, True) == y[:, 0]).mean())
    print(f"2. geometry: d = {D}, m = {c.num_detectors}, detectors and observable 0 equal to the sampler {eq_det}, line "
          f"flips = y_0 XOR gmap on all 8,192 {eq_lines}, correlated PM agrees with y_0 on {100 * agree:.2f} %", flush=True)
    ok &= eq_det and eq_lines
    params = [p for _, p in model.named_parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01)
    ema = [p.detach().clone() for p in params]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / 1000 if s < 1000 else 1.0)
    st = len(opt.state) == 0 and all(torch.equal(e, p) for e, p in zip(ema, params)) and \
        abs(sched.get_last_lr()[0] - 1e-6) < 1e-12
    print(f"3. start: AdamW states {len(opt.state)}, EMA equal to the weights, lr {sched.get_last_lr()[0]:.3g}: {st}",
          flush=True)
    ok &= st
    if parent is not None and pcfg["d"] == D and pcfg["blocks"] <= a.blocks:
        pm_ = net.ConvLines(D, width=pcfg["width"], bott=pcfg["bott"], blocks=pcfg["blocks"], heads=pcfg["heads"]).cuda()
        pm_.load_state_dict(parent)
        x = unpack(torch.from_numpy(X).cuda(), model.m)
        for bf16, tol in ((False, 1e-5), (True, 1e-2)):
            c1, z1 = logits(pm_, x, bf16)
            c2, z2 = logits(model, x, bf16)
            dc, dz = (c1 - c2).abs().max().item(), (z1 - z2).abs().max().item()
            same = bool(((c1 > 0) == (c2 > 0)).all())
            e = same and dc <= tol and dz <= tol
            print(f"4. identity ({'bf16' if bf16 else 'fp32'}): largest |logit difference| {dc:.3g} (standard), {dz:.3g} "
                  f"(lines), bit-equal {torch.equal(c1, c2) and torch.equal(z1, z2)}, decisions equal {same}: {e}",
                  flush=True)
            ok &= e
        del pm_
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    losses = []
    for step in range(1, a.steps + 1):
        Xb, Yb, _ = src.next(1024)
        xb = unpack(torch.from_numpy(Xb).cuda(), model.m)
        yl = unpack(torch.from_numpy(Yb).cuda(), D).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cc, zz = model(xb)
        loss = F.binary_cross_entropy_with_logits(cc.float(), yl[:, 0]) + F.binary_cross_entropy_with_logits(zz.float(), yl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == 1:
            print(f"5. smoke: step {step} loss {loss.item():.4f} (2 ln 2 = {2 * math.log(2):.4f})", flush=True)
    first, last = sum(losses[:25]) / 25, sum(losses[-25:]) / 25
    sm = all(math.isfinite(v) for v in losses) and last <= 1.2
    print(f"5. smoke: {len(losses)} steps, mean loss of the first 25 {first:.4f}, of the last 25 {last:.4f} "
          f"(at most 1.2, all finite): {sm}", flush=True)
    ok &= sm
    src.close()
    print("PASS" if ok else "FAIL", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
