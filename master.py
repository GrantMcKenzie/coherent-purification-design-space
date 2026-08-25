"""Direct implementation of the master formula, at any number of pairs.

Labels: v in F_2^{2n}, two bits per pair in (x, z) order, so that
    (0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=Y.
This matches the (x, z) per-pair convention of general_n.py; the two engines
are cross-checked in tests/test_engine_agreement.py.

Symplectic form  <u,v> = sum_k (u_x^k v_z^k + u_z^k v_x^k)  mod 2.
Y-parity         y(p) = #{k : p_k = (1,1)} mod 2.
Accepted sector  Acc  = {v : <e_c, v> = 0} for each check label e_c.
Target sector    T    = {v : v_0 = 0}  (kept pair is Phi+).

Master formula:
    M_gg = -(4/p0) * sum_{v in Acc, <p,v> = y(p)} (rho_{v+p} - rho_v)(1_T(v) - Fout)
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np


def labels(n):
    return list(itertools.product((0, 1), repeat=2 * n))


def symp(u, v):
    s = 0
    for k in range(0, len(u), 2):
        s += u[k] * v[k + 1] + u[k + 1] * v[k]
    return s % 2


def yparity(p):
    return sum(1 for k in range(0, len(p), 2) if p[k] == 1 and p[k + 1] == 1) % 2


def add(u, v):
    return tuple((a + b) % 2 for a, b in zip(u, v))


def weight(v):
    return sum(1 for k in range(0, len(v), 2) if (v[k], v[k + 1]) != (0, 0))


def werner_pops(n, F0, S_inv=None):
    """Bell populations of rho_C, exact if F0 is a Fraction.

    S_inv is the inverse symplectic map as an n-tuple-of-rows matrix over F_2;
    if None, the identity (no Clifford).
    """
    q = (1 - F0) / 3
    pops = {}
    for v in labels(n):
        w = v if S_inv is None else apply_mat(S_inv, v)
        val = 1
        for k in range(0, len(w), 2):
            val = val * (F0 if (w[k], w[k + 1]) == (0, 0) else q)
        pops[v] = val
    return pops


def apply_mat(M, v):
    """M is a list of rows over F_2; returns M v."""
    return tuple(sum(M[i][j] * v[j] for j in range(len(v))) % 2 for i in range(len(M)))


def accepted(n, check_labels):
    """check_labels: list of e_c labels. Acc = all v orthogonal to every e_c."""
    return [v for v in labels(n) if all(symp(e, v) == 0 for e in check_labels)]


def in_target(v):
    return v[0] == 0 and v[1] == 0


def branch(pops, acc):
    p0 = sum(pops[v] for v in acc)
    n0 = sum(pops[v] for v in acc if in_target(v))
    return p0, n0 / p0


def M_gg(pops, acc, p0, Fout, p):
    """Master formula for one generator label p."""
    yp = yparity(p)
    tot = 0
    for v in acc:
        if symp(p, v) != yp:
            continue
        tot += (pops[add(v, p)] - pops[v]) * ((1 if in_target(v) else 0) - Fout)
    return -4 * tot / p0


def diagonal(pops, acc, check_labels, n):
    """All 4^n - 1 diagonal entries."""
    p0, Fout = branch(pops, acc)
    out = {}
    for p in labels(n):
        if not any(p):
            continue
        out[p] = M_gg(pops, acc, p0, Fout, p)
    return out, p0, Fout


# ---------------------------------------------------------------- names

NAME = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
BITS = {v: k for k, v in NAME.items()}


def to_name(v):
    return "".join(NAME[(v[k], v[k + 1])] for k in range(0, len(v), 2))


def from_name(s):
    out = []
    for ch in s:
        out.extend(BITS[ch])
    return tuple(out)
