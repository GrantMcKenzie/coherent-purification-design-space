"""Vectorized float engine for large n=4 scans (exact engine reserved for minima)."""
import numpy as np
from general_n import n_labels, symp, y_parity, check_label, inv_S

n = 4
N = n_labels(n)
d = 2 * n
BITS = np.array([[(v >> i) & 1 for i in range(d)] for v in range(N)], dtype=np.uint8)
SYMP = np.zeros((N, N), dtype=np.uint8)
for p in range(N):
    for v in range(N):
        SYMP[p, v] = symp(p, v, n)
YPAR = np.array([y_parity(p, n) for p in range(N)], dtype=np.uint8)
TGT = np.array([(v & 3) == 0 for v in range(N)])
XORS = np.array([[v ^ p for v in range(N)] for p in range(N)], dtype=np.int64)

ACC1 = {}
for k in range(1, n):
    for c in 'XZY':
        e = check_label(c, k)
        ACC1[(c, k)] = SYMP[e] == 0
def acc_mask(checks):
    m = ACC1[(checks[0], 1)] & ACC1[(checks[1], 2)] & ACC1[(checks[2], 3)]
    return m

def weights_from_S(S):
    U = (BITS @ inv_S(S).T) & 1
    w = np.zeros(N, dtype=np.int64)
    for k in range(n):
        w += (U[:, 2 * k] | U[:, 2 * k + 1]).astype(np.int64)
    return w

def fast_stats(w, checks, F):
    """(p0, Fout, lambda_max) in float for one row."""
    q = (1.0 - F) / 3.0
    rho = F ** (n - w) * q ** w
    acc = acc_mask(checks)
    p0 = rho[acc].sum()
    N0 = rho[acc & TGT].sum()
    Fout = N0 / p0
    tgt_term = TGT.astype(float) - Fout
    lmax = -np.inf
    for p in range(1, N):
        sel = acc & (SYMP[p] == YPAR[p])
        if not sel.any():
            continue
        diff = rho[XORS[p][sel]] - rho[sel]
        val = -4.0 / p0 * np.dot(diff, tgt_term[sel])
        if val > lmax:
            lmax = val
    return p0, Fout, lmax
