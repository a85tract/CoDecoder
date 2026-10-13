"""BB codes on the l x m torus (bb/DESIGN.md section 1): Relay-BP's reference circuits, the site layout of their
detectors and the translation action on the 12 logical observables.

Circuits: qldpc_neural's pinned Relay-BP files (qnd.circuits.load, sha256-checked against
qldpc_neural/data/circuit_manifest.csv; QND_RELAY = the relay clone at d185194), X-basis memory, uniform circuit noise.
Relay-BP's qubit order: L data, R data, X checks, Z checks (n2 = l m each); inside a block the index of site (a, b)
is a m + b, the monomial x^a y^b. The circuit is invariant under the translations x^a y^b (Bravyi et al.'s schedule is
the same on every site), so every detector has a place (t, type, a, b):
  t = the 0-based round of the latest ancilla measurement it compares (X: t = 0 ... R - 1, then t = R for the final
      data-measurement layer; Z: t = 1 ... R - 1, the first Z round is random in an X memory), type 0 = X, 1 = Z.
The grid is (T = R + 1, l, m) with two detector channels per site (X, Z).

Logicals: the circuit's 12 observables are X-type operators ell_k on the data (L sites, then R sites). Every translate
T_g ell_k is again an X logical, T_g ell_k = sum_j M_g[k, j] ell_j + (product of X stabilisers c_{k,g}), so its flip is
  y_{k,g} = sum_j M_g[k, j] y_j  XOR  sum_{s in c_{k,g}} (XOR over t of the X detectors of check s)   (exact),
because the XOR of one check's X detectors over all layers is that check's final data-derived value."""
import os
from functools import lru_cache

import numpy as np

from qnd import bb as qbb
from qnd import circuits

PARAMS = qbb.PARAMS                                    # code -> (l, m, a, b); A = x^a1 + y^a2 + y^a3, B = y^b1 + x^b2 + x^b3
ROUNDS = {"bb72": 6, "bb144": 12}
MEAS = {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}


def torus(code):
    l, m = PARAMS[code][:2]
    return l, m


def shift_sites(code, g):
    """Site permutation of the translation by g = (ga, gb): site (a, b) -> (a + ga, b + gb) (index a m + b)."""
    l, m = torus(code)
    a, b = np.divmod(np.arange(l * m), m)
    return ((a + g[0]) % l) * m + (b + g[1]) % m


class Layout:
    """Detector layout, logicals and stabilisers of one circuit (any p: the layout does not depend on p)."""

    def __init__(self, code, p=0.003):
        self.code, self.p = code, p
        self.l, self.m = torus(code)
        self.n2 = n2 = self.l * self.m
        self.R = ROUNDS[code]
        self.T = self.R + 1
        c = circuits.load(code, "X", p)
        self.circuit = c
        meas, dets, obs = [], [], {}
        for inst in c.flattened():
            name = inst.name
            tg = inst.targets_copy()
            if name in MEAS:
                meas += [t.value for t in tg if t.is_qubit_target]
            elif name == "DETECTOR":
                dets.append([len(meas) + t.value for t in tg])
            elif name == "OBSERVABLE_INCLUDE":
                obs.setdefault(int(inst.gate_args_copy()[0]), []).extend(len(meas) + t.value for t in tg)
        meas = np.array(meas)
        role, site = np.divmod(meas, n2)                   # role 0 L, 1 R, 2 X check, 3 Z check
        occ = np.zeros(len(meas), np.int64)                # 0-based occurrence of each record's qubit
        seen = {}
        for r, q in enumerate(meas):
            occ[r] = seen.get(q, 0)
            seen[q] = occ[r] + 1
        D = len(dets)
        self.D = D
        self.t = np.zeros(D, np.int64)
        self.typ = np.zeros(D, np.int64)
        self.site = np.zeros(D, np.int64)
        hx = np.zeros((n2, 2 * n2), np.uint8)
        for d, recs in enumerate(dets):
            recs = np.array(recs)
            anc = recs[role[recs] >= 2]
            data = recs[role[recs] < 2]
            q = set(meas[anc])
            if len(q) != 1:
                raise ValueError(f"detector {d}: ancillas {sorted(q)}")
            self.typ[d] = role[anc[0]] - 2
            self.site[d] = site[anc[0]]
            if len(data):
                if self.typ[d] != 0:
                    raise ValueError(f"detector {d}: a Z detector with data records")
                self.t[d] = self.R
                hx[self.site[d], role[data] * n2 + site[data]] ^= 1
            else:
                self.t[d] = occ[anc].max()
        self.pos = {(int(t), int(y), int(s)): d for d, (t, y, s) in enumerate(zip(self.t, self.typ, self.site))}
        if len(self.pos) != D:
            raise ValueError("two detectors share a grid place")
        self.hx = hx                                       # X stabilisers read off the final detectors
        self.k = len(obs)
        self.logicals = np.zeros((self.k, 2 * n2), np.uint8)
        for k, recs in obs.items():
            recs = np.array(recs)
            if (role[recs] >= 2).any():
                raise ValueError(f"observable {k} includes an ancilla record")
            np.add.at(self.logicals[k], role[recs] * n2 + site[recs], 1)
        self.logicals %= 2
        self.flat = (self.t * 2 + self.typ) * n2 + self.site   # index into a (T, 2, n2) grid

    def detector_perm(self, g):
        """perm[d] = the detector at d's place translated by g (raises if the place holds no detector)."""
        s = shift_sites(self.code, g)
        return np.array([self.pos[(int(t), int(y), int(s[x]))] for t, y, x in zip(self.t, self.typ, self.site)])

    def data_perm(self, g):
        s = shift_sites(self.code, g)
        return np.concatenate([s, self.n2 + s])


# ---------------------------------------------------------------------------------------------- GF(2) linear algebra
def _row(v):
    return int("".join("1" if x else "0" for x in v[::-1]) or "0", 2)


class GF2Solver:
    """x B = v over GF(2) for a fixed generator matrix B (rows), any consistent v; returns one solution x."""

    def __init__(self, B):
        B = np.asarray(B, np.uint8) % 2
        self.nr = len(B)
        self.piv = []                                      # (pivot bit, row bits, combination bits)
        for i, r in enumerate(B):
            v, comb = _row(r), 1 << i
            for pb, pr, pc in self.piv:
                if v >> pb & 1:
                    v ^= pr
                    comb ^= pc
            if v:
                pb = v.bit_length() - 1
                for j, (qb, qr, qc) in enumerate(self.piv):  # keep the basis reduced
                    if qr >> pb & 1:
                        self.piv[j] = (qb, qr ^ v, qc ^ comb)
                self.piv.append((pb, v, comb))
        self.rank = len(self.piv)

    def solve(self, v):
        v, comb = _row(v), 0
        for pb, pr, pc in self.piv:
            if v >> pb & 1:
                v ^= pr
                comb ^= pc
        if v:
            return None
        return np.array([comb >> i & 1 for i in range(self.nr)], np.uint8)


@lru_cache(None)
def translates(code, p=0.003):
    """The translation action on the logicals (bb/DESIGN.md section 3).
    Returns dict: M (G, k, k) with T_g ell_k = sum_j M[g, k, j] ell_j + stabilisers, C (G, k, n2) the stabiliser
    combination c_{k,g}, Mcol (k, k G) and Gdet (D, k G) uint8 so that the flips of all translates (column k G + g,
    g = the translation's site index) are (y Mcol + det Gdet) mod 2."""
    lay = Layout(code, p)
    n2, k = lay.n2, lay.k
    G = n2
    gens = np.vstack([lay.logicals, lay.hx])
    sol = GF2Solver(gens)
    if sol.rank != k + qbb.gf2_rank(lay.hx):
        raise ValueError("logicals not independent modulo the X stabilisers")
    M = np.zeros((G, k, k), np.uint8)
    C = np.zeros((G, k, n2), np.uint8)
    hz = qbb.matrices(code)[1]
    for g in range(G):
        perm = lay.data_perm(divmod(g, lay.m))
        for kk in range(k):
            v = np.zeros(2 * n2, np.uint8)
            v[perm] = lay.logicals[kk]
            if ((hz.astype(np.int64) @ v) % 2).any():
                raise ValueError("a translate anticommutes with a Z check")
            x = sol.solve(v)
            if x is None:
                raise ValueError(f"translate g={g} of logical {kk} is outside the span")
            M[g, kk] = x[:k]
            C[g, kk] = x[k:]
    Mcol = np.zeros((k, k * G), np.uint8)
    Gdet = np.zeros((lay.D, k * G), np.uint8)
    isx = lay.typ == 0
    for kk in range(k):
        for g in range(G):
            col = kk * G + g
            Mcol[:, col] = M[g, kk]
            Gdet[isx, col] = C[g, kk][lay.site[isx]]
    return {"M": M, "C": C, "Mcol": Mcol, "Gdet": Gdet, "layout": lay}


def translate_circuit(code, p):
    """The reference circuit with one extra observable per translate (k, g), g != 0: the original 12 keep indices
    0 ... k - 1 and the extras follow k-major (returned list `extra` of (k, g)). No quantum operation changes, so the
    seeded detector sampler gives the original detectors (checked in bb.design)."""
    import stim
    lay = Layout(code, p)
    c = lay.circuit
    n2 = lay.n2
    nm = c.num_measurements
    # record index of the final data measurement of each data qubit (L sites then R sites)
    meas = []
    for inst in c.flattened():
        if inst.name in MEAS:
            meas += [t.value for t in inst.targets_copy() if t.is_qubit_target]
    last = {}
    for r, q in enumerate(meas):
        if q < 2 * n2:
            last[q] = r
    text = []
    extra = []
    for kk in range(lay.k):
        for g in range(1, n2):
            perm = lay.data_perm(divmod(g, lay.m))
            v = np.zeros(2 * n2, np.uint8)
            v[perm] = lay.logicals[kk]
            idx = lay.k + len(extra)
            extra.append((kk, g))
            recs = " ".join(f"rec[{last[q] - nm}]" for q in np.nonzero(v)[0])
            text.append(f"OBSERVABLE_INCLUDE({idx}) {recs}")
    return c + stim.Circuit("\n".join(text)), extra


def relay_dir():
    return os.path.expanduser(os.environ.get("QND_RELAY", "~/work/relay"))
