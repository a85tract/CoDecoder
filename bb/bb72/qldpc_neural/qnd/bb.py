"""Bivariate bicycle codes and Bravyi et al.'s depth-8 syndrome cycle (github.com/sbravyi/BivariateBicycleCodes,
decoder_setup.py, commit fa77e33), re-implemented here only to cross-check the reference circuits."""
import numpy as np

# code name -> (ell, m, (a1, a2, a3), (b1, b2, b3)) with A = x^a1 + y^a2 + y^a3, B = y^b1 + x^b2 + x^b3
PARAMS = {"bb72": (6, 6, (3, 1, 2), (3, 1, 2)), "bb144": (12, 6, (3, 1, 2), (3, 1, 2))}
# Bravyi et al.: order in which X / Z checks couple to their six neighbours (0-2 left block, 3-5 right block)
SX = ["idle", 1, 4, 3, 5, 0, 2]
SZ = [3, 5, 0, 1, 2, 4, "idle"]


def gf2_rank(M) -> int:
    M = (np.asarray(M) % 2).astype(np.uint8).copy()
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]
        others = others[others != r]
        M[others] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def matrices(code: str):
    ell, m, a, b = PARAMS[code]
    Il, Im = np.eye(ell, dtype=np.uint8), np.eye(m, dtype=np.uint8)
    x = lambda i: np.kron(np.roll(Il, i, axis=1), Im)
    y = lambda i: np.kron(Il, np.roll(Im, i, axis=1))
    terms_A = [x(a[0]), y(a[1]), y(a[2])]
    terms_B = [y(b[0]), x(b[1]), x(b[2])]
    A = sum(terms_A) % 2
    B = sum(terms_B) % 2
    hx = np.hstack([A, B]).astype(np.uint8)
    hz = np.hstack([B.T, A.T]).astype(np.uint8)
    return hx, hz, terms_A, terms_B


def bravyi_qubits(code: str):
    """Bravyi's linear order: X checks, left data, right data, Z checks (n/2 each)."""
    ell, m, _, _ = PARAMS[code]
    n2 = ell * m
    return {"X": list(range(n2)), "L": list(range(n2, 2 * n2)), "R": list(range(2 * n2, 3 * n2)),
            "Z": list(range(3 * n2, 4 * n2))}


def bravyi_cnot_layers(code: str):
    """The seven CNOT layers of one cycle as sets of (control, target) in Bravyi's qubit order."""
    ell, m, _, _ = PARAMS[code]
    n2 = ell * m
    _, _, (A1, A2, A3), (B1, B2, B3) = matrices(code)
    Q = bravyi_qubits(code)
    nbx = {}
    nbz = {}
    for i in range(n2):
        nbx[i] = [Q["L"][np.nonzero(M[i])[0][0]] for M in (A1, A2, A3)] + \
                 [Q["R"][np.nonzero(M[i])[0][0]] for M in (B1, B2, B3)]
        nbz[i] = [Q["L"][np.nonzero(M[:, i])[0][0]] for M in (B1, B2, B3)] + \
                 [Q["R"][np.nonzero(M[:, i])[0][0]] for M in (A1, A2, A3)]
    layers = []
    for t in range(7):
        s = set()
        if SX[t] != "idle":
            s |= {(Q["X"][i], nbx[i][SX[t]]) for i in range(n2)}
        if SZ[t] != "idle":
            s |= {(nbz[i][SZ[t]], Q["Z"][i]) for i in range(n2)}
        layers.append(s)
    return layers
