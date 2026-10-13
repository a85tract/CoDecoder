"""A ConvLines network of distance D initialised from a trained one of a smaller distance d (arch/README.md section
11.3; the DT route's "d = 13 from d = 11" idea for this architecture). The stem, the blocks and the line read-out's
projections, layer norms and MLP do not depend on the distance and are copied; the distance-dependent parameters are
mapped: the per-line embeddings head.e (d -> D rows) and the vote weights w by linear interpolation over the line's
relative position j / (d - 1) (w rescaled by d / D, so the vote's total weight stays), the time bias head.b_t (T ->
T' layers, both ends kept) by interpolation over t / (T - 1), the distance-bin bias head.b_line centred, its outer
bins padded with the edge values. Geometry buffers (A, the masks, the offsets) are the new distance's own."""
import torch


def _interp(v, n, dim):
    """Linear interpolation of v along dim to n points, both ends kept."""
    v = v.movedim(dim, -1)
    k = v.shape[-1]
    pos = torch.linspace(0, k - 1, n, dtype=torch.float64, device=v.device)
    lo = pos.floor().long().clamp(max=k - 1)
    hi = (lo + 1).clamp(max=k - 1)
    f = (pos - lo).to(v.dtype)
    out = v[..., lo] * (1 - f) + v[..., hi] * f
    return out.movedim(-1, dim)


def from_smaller(model, state):
    """Load state (a trained ConvLines state dict of a smaller distance) into model; returns the mapped keys."""
    sd = model.state_dict()
    mapped = {}
    d_old, d_new = state["w"].shape[0], sd["w"].shape[0]
    for k, v in state.items():
        if k not in sd or k == "A":
            continue
        t = sd[k]
        if v.shape == t.shape:
            sd[k] = v.clone()
            continue
        if k == "head.e":
            sd[k] = _interp(v, t.shape[0], 0)
        elif k == "w":
            sd[k] = _interp(v, t.shape[0], 0) * (d_old / d_new)
        elif k == "head.b_t":
            sd[k] = _interp(v, t.shape[-1], 2)
        elif k == "head.b_line":
            r_old, r_new = v.shape[-1] // 2, t.shape[-1] // 2
            idx = (torch.arange(-r_new, r_new + 1, device=v.device).clamp(-r_old, r_old) + r_old)
            sd[k] = v[..., idx]
        else:
            raise ValueError(f"{k}: no mapping for {tuple(v.shape)} -> {tuple(t.shape)}")
        mapped[k] = (tuple(v.shape), tuple(t.shape))
    model.load_state_dict(sd)
    return mapped
