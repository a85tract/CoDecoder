"""Logical-line labels and the line-query read-out (Stage A3 arm Q, aux_a/README.md).

In the pinned NVIDIA MemoryCircuit (code_rotation 'XV'), observable 0 is the final X-basis measurement of data qubits
0 ... d-1; the parallel lines L_j = {j d, ..., j d + d - 1} differ from L_0 by X stabilisers, so every line flip y_j
equals y_0 XOR g_j(syndrome). Labels are made WITHOUT changing a quantum operation: `line_circuit` appends
OBSERVABLE_INCLUDE(j) over the final data-measurement records rec[q - d^2], q in L_j, for j = 1 ... d-1 (checked: the
original observable 0 uses exactly rec[q - d^2], q in L_0, and the last d^2 records measure data qubits 0 ... d^2 - 1 in
order). The final data measurements are never network inputs; only detectors are.
  - `gmap`: g = A D mod 2, the GF(2)-linear map from the detectors to g_j = y_j XOR y_0, solved on sampled shots and
    verified exactly on fresh ones (tests/test_lines.py), used to align line logits as (-1)^g_j z_j;
  - `line_offsets`: per detector and line, the signed distance across the lines in half-line units (the position the
    line queries use);
  - `LineHead`: one query per line, a learned relative-position bias by that distance (4 heads; initialised to prefer
    tokens near the line), one cross-attention block over the final token states, an MLP to one logit per line."""
import numpy as np
import stim
import torch
import torch.nn.functional as F
from torch import nn

from icmp import circuits


def line_circuit(d, noise=circuits.NOISE_VERDICT, p=circuits.P_VERDICT):
    """(original circuit, augmented circuit with observables 1 ... d-1 for the parallel lines); p the noise level (a
    noise ramp's lower levels, arch.stream.RampLineStream)."""
    c = circuits.circuit(d, p=p, noise=noise)
    n2 = d * d
    flat = c.flattened()
    meas = [t.value for inst in flat if stim.gate_data(inst.name).produces_measurements for t in inst.targets_copy()]
    assert meas[-n2:] == list(range(n2)), "the last d^2 records are not data qubits 0 ... d^2 - 1 in order"
    obs0 = [inst for inst in flat if inst.name == "OBSERVABLE_INCLUDE"]
    assert len(obs0) == 1 and int(obs0[0].gate_args_copy()[0]) == 0
    look = sorted(t.value for t in obs0[0].targets_copy())
    assert look == [q - n2 for q in range(d)], f"observable 0 targets {look[:5]}... are not rec[q - d^2], q < d"
    aug = c.copy()
    for j in range(1, d):
        aug.append(stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_rec(q - n2) for q in range(j * d, j * d + d)],
                                           [j]))
    return c, aug


def line_offsets(d, noise=circuits.NOISE_VERDICT):
    """(m, d) int64: per detector and line j, round(2 (p_det - p_j) / s), p the coordinate across the lines (the axis on
    which a line's data qubits share one value), s the spacing between neighbouring lines."""
    mc = circuits.memory_circuit(d, noise=noise)
    data = mc.code.code_dict["data"]
    co = np.array([data[q]["coord"] for q in sorted(data)], float)            # (d^2, 2)
    lines = co.reshape(d, d, 2)
    axis = 0 if np.allclose(lines[:, :, 0].std(1), 0) else 1
    assert np.allclose(lines[:, :, axis].std(1), 0), "a line's data qubits do not share one coordinate"
    pj = lines[:, 0, axis]
    s = float(np.median(np.abs(np.diff(pj))))
    g = circuits.geometry(mc)                                                  # (m, 5): x, y, t, is_x, boundary
    pd = g[:, axis].astype(float)
    return np.rint(2 * (pd[:, None] - pj[None, :]) / s).astype(np.int64)


def _gf2_solve(Dm, G):
    """A solution X (m, k) of Dm X = G over GF(2) (Dm (N, m), G (N, k), uint8), free variables 0; rows bit-packed."""
    N, m = Dm.shape
    M = np.packbits(np.concatenate([Dm, G], 1).astype(np.uint8), axis=1)
    bit = lambda c: (M[:, c >> 3] >> (7 - (c & 7))) & 1
    piv, r = [], 0
    for col in range(m):
        cb = bit(col)
        rows = np.flatnonzero(cb[r:]) + r
        if len(rows) == 0:
            continue
        p = rows[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
            cb[[r, p]] = cb[[p, r]]
        hit = np.flatnonzero(cb)
        hit = hit[hit != r]
        M[hit] ^= M[r]
        piv.append(col)
        r += 1
        if r == N:
            break
    full = np.unpackbits(M, axis=1)[:, :m + G.shape[1]]
    assert not full[r:, m:].any(), "inconsistent system: line flips are not a linear function of the detectors"
    X = np.zeros((m, G.shape[1]), np.uint8)
    for i, col in enumerate(piv):
        X[col] = full[i, m:]
    return X, len(piv)


def gmap(d, n_fit=6000, seed=777, noise=circuits.NOISE_VERDICT):
    """A (m, d) uint8 with g = D A mod 2 (column 0 zero): fitted on n_fit shots of the augmented circuit."""
    _, aug = line_circuit(d, noise)
    det, obs = aug.compile_detector_sampler(seed=seed).sample(n_fit, separate_observables=True)
    G = (obs[:, 1:] ^ obs[:, :1]).astype(np.uint8)
    X, rank = _gf2_solve(det.astype(np.uint8), G)
    return np.concatenate([np.zeros((X.shape[0], 1), np.uint8), X], 1), rank


class LineHead(nn.Module):
    """One query per logical line over a shot's final token states -> one logit per line."""

    def __init__(self, w, offsets, heads=4, R=None):
        super().__init__()
        m, n = offsets.shape
        R = R or int(np.abs(offsets).max())
        self.h, self.n = heads, n
        self.q = nn.Parameter(torch.randn(n, w) * 0.02)
        self.kv = nn.Linear(w, 2 * w)
        self.qp = nn.Linear(w, w)
        self.o = nn.Linear(w, w)
        self.mlp = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1))
        bins = torch.arange(-R, R + 1, dtype=torch.float32)
        self.bias = nn.Parameter(-0.25 * bins.abs().repeat(heads, 1))         # (H, 2R + 1): prefer tokens near the line
        self.register_buffer("off", torch.as_tensor(np.clip(offsets, -R, R) + R, dtype=torch.long), persistent=False)

    def forward(self, x, mask, idx):                                           # (B, T, W), (B, T), (B, T) -> (B, n)
        B, T, W = x.shape
        H, n = self.h, self.n
        k, v = self.kv(x.float()).view(B, T, 2, H, W // H).permute(2, 0, 3, 1, 4)        # (B, H, T, dh)
        q = self.qp(self.q).view(n, H, W // H).permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)   # (B, H, n, dh)
        b = self.bias[:, self.off[idx]]                                        # (H, B, T, n)
        b = b.permute(1, 0, 3, 2)                                              # (B, H, n, T)
        mk = mask.clone()
        mk[~mk.any(1), 0] = True
        b = b.masked_fill(~mk[:, None, None, :], float("-inf"))
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=b.to(q.dtype))
        a = att.permute(0, 2, 1, 3).reshape(B, n, W)
        return self.mlp(self.q + self.o(a)).squeeze(-1)
