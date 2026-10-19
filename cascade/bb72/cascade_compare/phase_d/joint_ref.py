"""The joint decision's CPU references (numpy only, no torch): joint_np is phase_d/joint.py's GPU arithmetic bit for bit
(numpy's float32 elementwise operations are IEEE operations rounded to nearest, never fused), a3_f64 ARB's float64 rule
(arb_analysis.a3_answers). joint.py's docstring fixes the arithmetic."""
import numpy as np

K, NY = 12, 4096
YB = ((np.arange(NY)[:, None] >> np.arange(K)) & 1).astype(bool)            # (4,096, 12): bit j of class y


def w32_of(w):
    """w as the kernels use it: float32, rounded to nearest; and its bit pattern."""
    x = np.float32(w)
    return float(x), f"0x{int(np.array([x]).view(np.uint32)[0]):08x}"


def joint_np(sc, z, w, block=2048, gap=False):
    """The joint answer in numpy, bit for bit joint._k_jtop's: sc (n, 4,096) float32 class scores, z (n, 12) float32
    logits -> classes (n,) int64 (the lowest index among the maxima of t), and with gap=True also the float32 gap
    between the largest t and the largest t of the other classes (0 at a tie)."""
    w32 = np.float32(w)
    zero = np.float32(0.0)
    n = len(z)
    out = np.empty(n, np.int64)
    gp = np.empty(n, np.float32) if gap else None
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        zz = np.asarray(z[lo:hi], np.float32)
        e = np.zeros((hi - lo, NY), np.float32)
        for j in range(K):
            e = e + np.where(YB[None, :, j], zz[:, j:j + 1], zero)
        t = np.asarray(sc[lo:hi], np.float32) + w32 * e
        a = t.argmax(1)
        out[lo:hi] = a
        if gap:
            r = np.arange(hi - lo)
            top = t[r, a]
            t[r, a] = -np.inf
            gp[lo:hi] = top - t.max(1)
    return (out, gp) if gap else out


def a3_f64(sc, z, w, block=2048):
    """ARB's rule A3 (arb_analysis.a3_answers) in float64: argmax_y [s(y) + w (z . y)], the dot product by matmul."""
    Y = YB.astype(np.float64)
    n = len(z)
    out = np.empty(n, np.int64)
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        S = np.asarray(sc[lo:hi]).astype(np.float64)
        out[lo:hi] = np.argmax(S + w * (np.asarray(z[lo:hi]).astype(np.float64) @ Y.T), 1)
    return out


def cls_of(bits):
    """Decision bits (n, 12) -> class indices (bit j = observable j)."""
    return (np.asarray(bits).astype(np.int64) << np.arange(K)).sum(1)


def bits_of_cls(cls):
    return ((np.asarray(cls)[:, None] >> np.arange(K)) & 1).astype(np.uint8)
