"""
general_n.py -- master-formula engine for bilateral-Clifford purification at any n.

Conventions (match the manuscript):
  * Bell labels v in F_2^{2n}, encoded as ints; bit 2k = x-bit of pair k,
    bit 2k+1 = z-bit of pair k.  Pair 0 is kept; pairs 1..n-1 sacrificial.
  * Pauli of a pair-label (x,z): (0,0)=I (1,0)=X (0,1)=Z (1,1)=Y.
  * Symplectic form <u,v> = sum_k u_x[k] v_z[k] + u_z[k] v_x[k]  (mod 2).
  * Y-parity y(p) = #{k : p_x[k]=p_z[k]=1} mod 2.
  * Werner populations in the input frame: rho_u = F^{n-w(u)} q^{w(u)}, q=(1-F)/3.
  * A circuit is a symplectic S acting on labels: |b_u> -> |b_{Su}| (up to sign);
    post-circuit population at label v is rho'_v = rho_{S^{-1} v}.
  * Check c_k in {X,Z,Y} on sacrificial pair k; accepted sector
    Acc = {v : <e_{c_k}, v> = 0 for all k}.  Target T = {v : v|pair0 = 0}.
  * Master formula:
      M_gg = -(4/p0) * sum_{v in Acc, <p,v>=y(p)} (rho'_{v+p}-rho'_v)(1_T(v)-Fout)
"""

from fractions import Fraction
import numpy as np
import random

# ---------------------------------------------------------------- label helpers

def n_labels(n):
    return 1 << (2 * n)

def pair_bits(v, k):
    return (v >> (2 * k)) & 1, (v >> (2 * k + 1)) & 1

def weight(v, n):
    """excitation weight: number of pairs with a non-identity Pauli."""
    w = 0
    for k in range(n):
        if (v >> (2 * k)) & 3:
            w += 1
    return w

def symp(u, v, n):
    """symplectic form <u,v> mod 2."""
    s = 0
    for k in range(n):
        ux, uz = pair_bits(u, k)
        vx, vz = pair_bits(v, k)
        s ^= (ux & vz) ^ (uz & vx)
    return s

def y_parity(p, n):
    y = 0
    for k in range(n):
        px, pz = pair_bits(p, k)
        y ^= px & pz
    return y

def check_label(pauli, k):
    """label of Pauli check `pauli` in {'X','Z','Y'} on pair k."""
    x = 1 if pauli in ('X', 'Y') else 0
    z = 1 if pauli in ('Z', 'Y') else 0
    return (x << (2 * k)) | (z << (2 * k + 1))

# ------------------------------------------------------------- symplectic group

def identity_S(n):
    return np.eye(2 * n, dtype=np.uint8)

def apply_S(S, v, n):
    """apply symplectic matrix S (2n x 2n over F2) to label v (int)."""
    bits = np.array([(v >> i) & 1 for i in range(2 * n)], dtype=np.uint8)
    out = S.dot(bits) & 1
    r = 0
    for i, b in enumerate(out):
        if b:
            r |= (1 << i)
    return r

def transvection_matrix(a, n):
    """T_a(x) = x + <x,a> a  as a matrix over F2."""
    d = 2 * n
    S = np.eye(d, dtype=np.uint8)
    for i in range(d):
        e = 1 << i
        if symp(e, a, n):
            for j in range(d):
                if (a >> j) & 1:
                    S[j, i] ^= 1
    return S

def random_symplectic(n, rng, n_transvections=None):
    if n_transvections is None:
        n_transvections = 6 * n * n
    d = 2 * n
    S = np.eye(d, dtype=np.uint8)
    for _ in range(n_transvections):
        a = rng.randrange(1, 1 << d)
        S = (transvection_matrix(a, n).dot(S)) & 1
    return S

def cnot_bilateral_S(control_pair, target_pair, n):
    """Symplectic action of bilateral CNOT (control->target) on Bell labels:
       x_target += x_control ; z_control += z_target."""
    d = 2 * n
    S = np.eye(d, dtype=np.uint8)
    cx, cz = 2 * control_pair, 2 * control_pair + 1
    tx, tz = 2 * target_pair, 2 * target_pair + 1
    S[tx, cx] = 1   # new x_t gets x_c
    S[cz, tz] = 1   # new z_c gets z_t
    return S

def inv_S(S):
    """invert over F2 by Gaussian elimination."""
    d = S.shape[0]
    A = np.concatenate([S.copy() & 1, np.eye(d, dtype=np.uint8)], axis=1)
    r = 0
    for c in range(d):
        piv = None
        for i in range(r, d):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            raise ValueError("singular")
        A[[r, piv]] = A[[piv, r]]
        for i in range(d):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
    return A[:, d:]

# --------------------------------------------------------------- protocol rows

class Row:
    """One (circuit S, check assignment) row of the design space."""

    def __init__(self, S, checks, n):
        """checks: tuple of Paulis for sacrificial pairs 1..n-1."""
        self.n = n
        self.S = S
        self.Sinv = inv_S(S)
        self.checks = checks
        N = n_labels(n)
        # circuit-frame membership and input-frame weights
        e = [check_label(c, k + 1) for k, c in enumerate(checks)]
        self.acc = np.zeros(N, dtype=bool)
        self.tgt = np.zeros(N, dtype=bool)
        self.w_of_v = np.zeros(N, dtype=np.int64)   # weight of S^{-1} v
        for v in range(N):
            self.acc[v] = all(symp(ek, v, n) == 0 for ek in e)
            self.tgt[v] = (v & 3) == 0
            self.w_of_v[v] = weight(apply_S(self.Sinv, v, n), n)

    # ---- weight distributions -> exact fidelity map
    def weight_dists(self):
        n, N = self.n, n_labels(self.n)
        A = [0] * (n + 1)   # over Acc
        B = [0] * (n + 1)   # over Acc & T
        for v in range(N):
            if self.acc[v]:
                A[int(self.w_of_v[v])] += 1
                if self.tgt[v]:
                    B[int(self.w_of_v[v])] += 1
        return tuple(A), tuple(B)

    def p0_N0(self, F):
        """exact (Fraction) acceptance and target weight at input fidelity F."""
        n = self.n
        q = (1 - F) / 3
        A, B = self.weight_dists()
        p0 = sum(a * F ** (n - w) * q ** w for w, a in enumerate(A))
        N0 = sum(b * F ** (n - w) * q ** w for w, b in enumerate(B))
        return p0, N0

    def fout(self, F):
        p0, N0 = self.p0_N0(F)
        return N0 / p0

    # ---- master formula
    def rho_vec(self, F):
        n = self.n
        q = (1 - F) / 3
        return np.array([F ** (n - int(w)) * q ** int(w) for w in self.w_of_v], dtype=object)

    def M_diag(self, F):
        """diagonal susceptibility over all 4^n-1 common-mode generators (exact)."""
        n, N = self.n, n_labels(self.n)
        rho = self.rho_vec(F)
        p0, N0 = self.p0_N0(F)
        Fout = N0 / p0
        out = {}
        acc_idx = np.nonzero(self.acc)[0]
        for p in range(1, N):
            y = y_parity(p, n)
            tot = Fraction(0)
            for v in acc_idx:
                if symp(p, int(v), n) == y:
                    tot += (rho[v ^ p] - rho[v]) * ((1 if self.tgt[v] else 0) - Fout)
            out[p] = Fraction(-4, 1) * tot / p0
        return out

    def lambda_max(self, F):
        return max(self.M_diag(F).values())


def label_name(p, n):
    P = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}
    return ''.join(P[pair_bits(p, k)] for k in range(n))
