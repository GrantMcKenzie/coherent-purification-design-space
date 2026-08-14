"""Three-pair design space: enumerate pair frames and evaluate the master formula.

An atom is an unordered orthogonal decomposition of F_2^{2n} into n
non-degenerate two-dimensional symplectic subspaces ("pair frame").  The
populations of rho_C follow directly: writing v = w_1 + ... + w_n with
w_i in W_i, the excitation weight is #{i : w_i != 0}.
"""

from __future__ import annotations

import itertools
from fractions import Fraction as Fr

from master import add, labels, symp, yparity, in_target, from_name, to_name


def nondeg_2subspaces(n):
    """All 2-dimensional non-degenerate symplectic subspaces of F_2^{2n}."""
    L = [v for v in labels(n) if any(v)]
    seen = {}
    for u in L:
        for v in L:
            if u >= v or symp(u, v) != 1:
                continue
            W = frozenset({u, v, add(u, v)})
            seen[W] = W
    return list(seen.values())


def frames(n):
    """Unordered orthogonal decompositions into n non-degenerate 2-subspaces."""
    subs = nondeg_2subspaces(n)
    out = []

    def orth(W1, W2):
        return all(symp(a, b) == 0 for a in W1 for b in W2)

    def rec(chosen, used):
        if len(chosen) == n:
            out.append(tuple(sorted(chosen, key=lambda W: sorted(W))))
            return
        start = 0 if not chosen else subs.index(chosen[-1]) + 1
        for i in range(start, len(subs)):
            W = subs[i]
            if W & used:
                continue
            if all(orth(W, C) for C in chosen):
                rec(chosen + [W], used | W)

    rec([], frozenset())
    return out


def pops_from_frame(frame, F0):
    """Bell populations: F0^{n-k} q^k with k the number of non-zero components."""
    n = len(frame)
    q = (1 - F0) / 3
    basis = []
    for W in frame:
        basis.append(sorted(W))
    pops = {}
    for v in labels(n):
        k = 0
        # decompose v uniquely across the orthogonal frame subspaces
        for W in frame:
            # component in W is the unique w in W with <w, x> = <v, x> for all x in W
            comp = None
            for w in list(W) + [tuple([0] * (2 * n))]:
                if all(symp(w, x) == symp(v, x) for x in W):
                    comp = w
                    break
            if comp is not None and any(comp):
                k += 1
        pops[v] = F0 ** (n - k) * q ** k
    return pops


def evaluate(frame, check_labels, F0, gens=None):
    n = len(frame)
    pops = pops_from_frame(frame, F0)
    acc = [v for v in labels(n) if all(symp(e, v) == 0 for e in check_labels)]
    p0 = sum(pops[v] for v in acc)
    Fout = sum(pops[v] for v in acc if in_target(v)) / p0
    if gens is None:
        gens = [p for p in labels(n) if any(p)]
    diag = {}
    for p in gens:
        yp = yparity(p)
        tot = 0
        for v in acc:
            if symp(p, v) != yp:
                continue
            tot += (pops[add(v, p)] - pops[v]) * ((1 if in_target(v) else 0) - Fout)
        diag[p] = -4 * tot / p0
    return p0, Fout, diag
