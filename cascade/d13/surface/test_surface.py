"""Unit checks of Cascade-surface (python -m surface.test_surface [--d 13] [--gpu]); each prints one JSON line and the
run ends with ALL PASS or the failures. CPU unless --gpu (then also the bf16 channels-last forward on the GPU)."""
import argparse
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from surface import cascade_optim as copt
from surface import surface_data as sdata
from surface import surface_layout as sl
from surface import surface_net as snet

RESULTS = []


def check(name, ok, **info):
    RESULTS.append((name, bool(ok)))
    print(json.dumps({"check": name, "pass": bool(ok), **info}), flush=True)


def test_circuits(d):
    sha = sl.check_circuit_at(d)
    check("circuit_at_reproduces_pinned", True, sha256=sha)
    c3 = sl.circuit_at(d, 0.003)
    args = sorted({(i.name, i.gate_args_copy()[0]) for i in c3.flattened()
                   if i.name in sl.DEP or i.name in sl.FLIP})
    ok = all(abs(a - (0.003 if n in sl.DEP else 0.002)) < 1e-15 for n, a in args)
    c6 = sl.pinned_circuit(d)
    ok &= c3.num_detectors == c6.num_detectors and c3.num_observables == c6.num_observables == 1
    det, obs = sdata.sample(d, 0.003, 2000, sdata.eval_seed(d, 2, 0.003, 0))
    det6, obs6 = sdata.sample(d, 0.006, 2000, sdata.eval_seed(d, 2, 0.006, 0))
    check("circuit_at_other_p", ok and det.mean() < det6.mean(), noise=args, det_rate_p003=round(float(det.mean()), 5),
          det_rate_p006=round(float(det6.mean()), 5))


def test_layout(d):
    lay = sl.Layout(d)
    h = (d * d - 1) // 2
    want = [(h, 0)] + [(h, h)] * (lay.R - 1) + [(h, 0)]
    check("layout_layers", lay.R == d and lay.per_layer == want, R=lay.R, per_layer=lay.per_layer[:3])
    sup = lay.support
    line_r = len(np.unique(sup[:, 0])) == 1
    sl.release_root()
    from convlines import circuits as rc
    g = np.load(f"{rc.WEIGHTS}/grid_d{d}.npz")                # the same grid file that sl.Layout reads
    off = g["line_off"][:, 0].astype(int)                # the release's signed distance to logical line 0
    axis = lay.i if line_r else lay.j
    k0 = int(sup[0, 0] if line_r else sup[0, 1])
    near = sorted(set(axis[np.abs(off) == 1].tolist()))
    check("support_matches_release_line0", near == [k0, k0 + 1] and len(sup) == d,
          support=sup.tolist(), rows_at_half_line=near, line_is_row=line_r)
    return lay


def small_model(lay, H=16, L=2, res="skip"):
    torch.manual_seed(0)
    return snet.build({"d": lay.d, "R": lay.R, "H": H, "L": L, "res": res}, lay)


def test_net(lay):
    m = small_model(lay).double()
    det, obs = sdata.sample(lay.d, 0.006, 64, sdata.eval_seed(lay.d, 2, 0.006, 1))
    x = torch.as_tensor(det)
    tok = m.tokens(x)
    check("tokens", int((tok == snet.ABSENT).sum()) == len(det) * (m.N - m.D) and
          bool((tok[:, torch.as_tensor(np.flatnonzero(m.gidx.numpy() < m.D))] < 4).all()),
          absent_per_shot=m.N - m.D, nodes=m.N)
    with torch.no_grad():
        nn_ = m.mlp2
        torch.nn.init.normal_(nn_.weight, std=0.1)       # zero at init: give the read-out something to show
        m.eval()
        h = m.backbone(tok)
        z1, z2 = m.readout(h), m.readout_ref(h)
    err = float((z1 - z2).abs().max())
    check("readout_equals_scatter_then_pool", err < 1e-9, max_abs_diff=err, shape=list(z1.shape))
    m.train()
    z = m(x)
    loss = F.binary_cross_entropy_with_logits(z, torch.as_tensor(obs).double())
    loss.backward()
    nograd = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
              or float(p.grad.abs().sum()) == 0]
    check("gradients_reach_every_parameter", not nograd, missing=nograd[:5])


def test_optim(lay):
    m = small_model(lay)
    mu = snet.muon_params(m)
    mg, lg = copt.param_groups(m, mu, 3e-3, 2e-4, 3e-3, 0)
    ids = [id(p) for g in mg + lg for p in g["params"]]
    allp = [id(p) for p in m.parameters()]
    conv5 = all(p.ndim == 5 for p in mu) and len(mu) == 3 * m.L
    check("param_groups_cover_once", sorted(ids) == sorted(allp) and len(set(ids)) == len(ids) and conv5,
          muon=len(mu), lion=sum(len(g["params"]) for g in lg))
    muon, lion = copt.Muon(mg), copt.Lion(lg)
    det, obs = sdata.sample(lay.d, 0.006, 32, sdata.eval_seed(lay.d, 2, 0.006, 2))
    torch.nn.init.normal_(m.mlp2.weight, std=0.1)
    before = [p.detach().clone() for p in m.parameters()]
    loss = F.binary_cross_entropy_with_logits(m(torch.as_tensor(det)), torch.as_tensor(obs).float())
    loss.backward()
    muon.step()
    lion.step()
    moved = sum(int(not torch.equal(b, p)) for b, p in zip(before, m.parameters()))
    check("optimisers_step", math.isfinite(float(loss)) and moved == len(before), moved=moved, params=len(before))


def test_seeds():
    ok = True
    for d in (13, 17):
        tr = (sdata.train_seed(d, 0, 0, 0), sdata.train_seed(d, 999, 99, 10 ** 10 - 1))
        ev = (sdata.eval_seed(d, 0, 0.000001, 0), sdata.eval_seed(d, 99, 0.009999, 10 ** 7 - 1))
        ok &= tr[1] < ev[0] or ev[1] < tr[0]
        ok &= 2 ** 56 < min(tr[0], ev[0]) and max(tr[1], ev[1]) < 2 ** 57
        ok &= sdata.train_seed(d, 999, 99, 10 ** 10 - 1) < sdata.train_seed(d + 1, 0, 0, 0)
        ok &= sdata.eval_seed(d, 99, 0.009999, 10 ** 7 - 1) < sdata.eval_seed(d + 1, 0, 0.000001, 0)
    check("seed_ranges_disjoint", ok, train13=[sdata.train_seed(13, 0, 0, 0)], eval13=[sdata.eval_seed(13, 0, 1e-6, 0)])


def test_stream(d):
    sched = (80000, 0.002, 0.007, 0.01, 0.01)
    st = sdata.StimStream(d, 999, 0, sched, 64, 2, start=799)
    try:
        rows = [st.next() for _ in range(3)]
    finally:
        st.close()
    ok = [r[0] for r in rows] == [copt.p_schedule(c, *sched) for c in (799, 800, 801)]
    import stim
    c = sl.circuit_at(d, rows[1][0])
    det, obs = c.compile_detector_sampler(seed=sdata.train_seed(d, 999, 0, 800)).sample(64, separate_observables=True,
                                                                                        bit_packed=True)
    ok &= np.array_equal(det, rows[1][1]) and np.array_equal(obs, rows[1][2])
    check("stream_seeded_by_step", ok, p=[r[0] for r in rows], stim=stim.__version__)


def test_gpu(lay):
    if not torch.cuda.is_available():
        check("gpu_forward", False, error="no CUDA device")
        return
    dev = torch.device("cuda")
    m = small_model(lay, H=64, L=3).to(dev)
    det, _ = sdata.sample(lay.d, 0.006, 256, sdata.eval_seed(lay.d, 2, 0.006, 3))
    x = torch.as_tensor(det, device=dev)
    torch.nn.init.normal_(m.mlp2.weight, std=0.1)
    m.eval()
    with torch.no_grad():
        z32 = m(x).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z16 = m(x).float()
    err = float((z32 - z16).abs().max() / z32.abs().max().clamp_min(1e-6))
    cl = m.blocks[0].conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
    check("gpu_forward_bf16_vs_fp32", err < 0.05 and cl, rel_err=err, channels_last=cl,
          device=torch.cuda.get_device_name(dev))


def test_rules(lay):
    """The trainer's stopping rule reports no observed improvement (a rise stops as well), its finiteness check before
    every checkpoint, and the options of the job chain (review findings 9-11)."""
    from surface import surface_train as st
    log = [{"p": 0.006, "k": 3, "ema": {"fail": 100}}, {"p": 0.007, "k": 3, "ema": {"fail": 200}}]

    def rows(f6, f7):
        return {0.006: {"p": 0.006, "ema": {"fail": f6}}, 0.007: {"p": 0.007, "ema": {"fail": f7}}}
    fall = st.improvement_check(log, rows(90, 190), 6, 3, 0.02)       # 10 % and 5 % fewer failures: go on
    flat = st.improvement_check(log, rows(99, 199), 6, 3, 0.02)       # 1 % and 0.5 %: stop
    rise = st.improvement_check(log, rows(120, 210), 6, 3, 0.02)      # more failures: stop as well
    mixed = st.improvement_check(log, rows(90, 210), 6, 3, 0.02)      # one p still improves: go on
    missing = st.improvement_check(log, rows(90, 190), 7, 3, 0.02)    # no point at k = 4: no decision
    check("rule_no_observed_improvement", not fall["stop"] and flat["stop"] and rise["stop"] and not mixed["stop"]
          and missing is None, rise=rise["improvement"])
    m = small_model(lay)
    zero = torch.zeros(())
    clean = st.finite(zero, m, m) and not st.finite(torch.tensor(float("nan")), m)
    with torch.no_grad():
        next(m.parameters()).view(-1)[0] = float("inf")
    check("finite_before_checkpoint", clean and not st.finite(zero, m))
    a = st.parser().parse_args(["--run", "x", "--run-id", "903", "--no-improvement", "3e7:0.02", "--raw-until", "8e7",
                                "--fault", "nan:65"])
    check("chain_options", a.no_improvement == "3e7:0.02" and a.raw_until == 8e7 and a.fault == "nan:65"
          and {"raw_until", "fault"} <= st.FREE and "no_improvement" not in st.FREE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=13)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    test_circuits(a.d)
    lay = test_layout(a.d)
    print(json.dumps({"layout": lay.describe()}), flush=True)
    test_net(lay)
    test_optim(lay)
    test_seeds()
    test_stream(a.d)
    test_rules(lay)
    if a.gpu:
        test_gpu(lay)
    bad = [n for n, ok in RESULTS if not ok]
    print("ALL PASS" if not bad else f"FAILED: {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
