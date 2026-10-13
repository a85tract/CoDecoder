"""Cascade's optimisation (arXiv:2604.08358, Methods "Training"): Muon for the matrix-valued parameters, Lion for the
rest, a warm-up + cosine learning-rate schedule, an EMA of the weights. The paper's numbers are the defaults of
bb.cascade_train; the choices it leaves open are in bb/cascade/CASCADE_IMPL.md section 2.

Muon follows K. Jordan's reference implementation (github.com/KellerJordan/Muon, muon.py, 2025): momentum 0.95 with
Nesterov, the quintic Newton-Schulz iteration (5 steps, coefficients 3.4445, -4.7750, 2.0315) in bfloat16 on the
(out, in) matrix, the update scaled by max(1, out / in)^0.5, decoupled weight decay p <- p (1 - lr wd). Our relation
weights are stored as (out, 12 in), i.e. as a convolution filter flattened the way the reference flattens conv filters.
Lion follows Chen et al. (2023), Algorithm 2: u = sign(b1 m + (1 - b1) g), p <- p (1 - lr wd) - lr u,
m <- b2 m + (1 - b2) g, betas (0.9, 0.99)."""
import math

import torch


def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if tall:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon (see the module docstring). Matrices of one shape are orthogonalised together (a batched Newton-Schulz:
    the same arithmetic per matrix, fewer kernel launches)."""

    def __init__(self, params, lr=3e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            by_shape = {}
            for p in gr["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = st["momentum_buffer"]
                buf.lerp_(p.grad, 1 - gr["momentum"])
                u = p.grad.lerp(buf, gr["momentum"]) if gr["nesterov"] else buf
                u = u.reshape(len(u), -1)
                by_shape.setdefault(tuple(u.shape), []).append((p, u))
            for (r, c), items in by_shape.items():
                U = newton_schulz(torch.stack([u for _, u in items]), gr["ns_steps"]) * max(1.0, r / c) ** 0.5
                ps = [p for p, _ in items]
                torch._foreach_mul_(ps, 1 - gr["lr"] * gr["weight_decay"])
                torch._foreach_add_(ps, [U[i].view_as(p).to(p.dtype) for i, p in enumerate(ps)], alpha=-gr["lr"])


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            b1, b2 = gr["betas"]
            ps = [p for p in gr["params"] if p.grad is not None]
            if not ps:
                continue
            for p in ps:
                if "exp_avg" not in self.state[p]:
                    self.state[p]["exp_avg"] = torch.zeros_like(p)
            ms = [self.state[p]["exp_avg"] for p in ps]
            gs = [p.grad for p in ps]
            u = torch._foreach_mul(ms, b1)
            torch._foreach_add_(u, gs, alpha=1 - b1)
            torch._foreach_sign_(u)
            torch._foreach_mul_(ps, 1 - gr["lr"] * gr["weight_decay"])
            torch._foreach_add_(ps, u, alpha=-gr["lr"])
            torch._foreach_lerp_(ms, gs, 1 - b2)


def param_groups(model, muon_list, lr_muon, lr_lion, wd, mup_base=0):
    """-> (Muon groups, Lion groups). Lion gets every parameter Muon does not; weight decay on matrices only (not on
    biases or BN's affine parameters). mup_base > 0: the read-out's matrices (scatter, MLP) train at
    lr_lion mup_base / H, muP's 1/width rule for hidden and output weights under an Adam-like (sign) update; the
    embedding and the vectors keep lr_lion; Muon's shape factor already makes its step width-independent."""
    mu = {id(p) for p in muon_list}
    H = model.H
    ro_scale = mup_base / H if mup_base else 1.0
    lion = {"decay": [], "nodecay": [], "readout": []}
    for n, p in model.named_parameters():
        if id(p) in mu:
            continue
        if n in ("scatter", "mlp1.weight", "mlp2.weight"):
            lion["readout"].append(p)
        elif p.ndim >= 2:
            lion["decay"].append(p)
        else:
            lion["nodecay"].append(p)
    muon_groups = [dict(params=list(muon_list), lr=lr_muon, base_lr=lr_muon, weight_decay=wd)]
    lion_groups = [dict(params=lion["decay"], lr=lr_lion, base_lr=lr_lion, weight_decay=wd),
                   dict(params=lion["nodecay"], lr=lr_lion, base_lr=lr_lion, weight_decay=0.0),
                   dict(params=lion["readout"], lr=lr_lion * ro_scale, base_lr=lr_lion * ro_scale, weight_decay=wd)]
    return muon_groups, [g for g in lion_groups if g["params"]]


def lr_factor(step, warmup=1000, cosine_steps=50000, floor=0.1):
    """Linear warm-up over `warmup` steps, then a cosine from the peak to floor x peak at step cosine_steps, then
    floor x peak (Methods: "a cosine schedule over 50000 steps (decaying to 1/10 of peak) with linear warmup over
    1000 steps", and 80,000 steps in all)."""
    if step < warmup:
        return (step + 1) / warmup
    if step >= cosine_steps:
        return floor
    x = (step - warmup) / max(1, cosine_steps - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * x))


def p_schedule(step, steps, p1, p2, f1=0.01, f2=0.01, quantum=1e-5):
    """The three-stage curriculum (Methods): p1 for the first f1 of the steps, a linear anneal p1 -> p2 over the next
    f2, then p2 (f1 + f2 <= 0.02 in the paper). Rounded to `quantum` so the samplers compile few circuits."""
    s1, s2 = int(f1 * steps), int(f2 * steps)
    if step < s1:
        p = p1
    elif step < s1 + s2:
        p = p1 + (p2 - p1) * (step - s1) / max(1, s2)
    else:
        p = p2
    return round(round(p / quantum) * quantum, 10)


class EMA:
    """EMA of every floating-point parameter and buffer (BN running statistics included, as timm's ModelEmaV2 does);
    integer buffers are copied."""

    def __init__(self, model, decay):
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        fe, fs, ie, is_ = [], [], [], []
        for k, v in self.model.state_dict().items():
            (fe if v.dtype.is_floating_point else ie).append(v)
            (fs if v.dtype.is_floating_point else is_).append(src[k])
        torch._foreach_lerp_(fe, fs, 1 - self.decay)
        for e, m in zip(ie, is_):
            e.copy_(m)


@torch.no_grad()
def recalibrate_bn(model, batches):
    """Recompute every BN layer's running statistics as the plain average over `batches` (an iterable of detector
    tensors) in training mode, without touching the weights. -> the model in eval mode."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for det in batches:
        model(det)
    for m, mo in zip(bns, saved):
        m.momentum = mo
    return model.eval()
