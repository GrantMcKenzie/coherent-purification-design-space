"""The coherent susceptibility Hessian ``M``.

With ``F_out(theta) = F_out(F_0, 0) - theta^T M theta + O(|theta|^3)``
(Eq. (1)), and the linear term vanishing by Lemma 1, the entries are

    M_ab = -(1 / 2 p_0) (N_ab - F_out(F_0, 0) p_ab)          Eq. (9)

where ``N_ab`` and ``p_ab`` come from the symmetrised double-commutator
superoperator

    L_ab(rho) = H_a rho H_b + H_b rho H_a
                - (1/2) {H_a H_b + H_b H_a, rho}.            Eq. (8)

``hessian_exact`` evaluates this directly.  ``hessian_findiff`` recomputes it
by central differences on the fully simulated accepted fidelity, and is used
throughout the test suite as an independent check.
"""

from __future__ import annotations

import numpy as np

from .circuits import (
    acceptance_projector,
    accepted_branch,
    accepted_fidelity,
    generator,
    rho_after_circuit,
)
from .paulis import TWO_QUBIT_GENERATORS, phi_plus_projector_pair0


def _L(ha: np.ndarray, hb: np.ndarray, rho: np.ndarray) -> np.ndarray:
    anti = ha @ hb + hb @ ha
    return ha @ rho @ hb + hb @ rho @ ha - 0.5 * (anti @ rho + rho @ anti)


def hessian_from_rho(rho_c: np.ndarray, check: str,
                     labels: tuple[str, ...] = TWO_QUBIT_GENERATORS) -> np.ndarray:
    """Exact Hessian from a post-circuit state, however that state was built.

    Taking ``rho_C`` as the argument rather than a Clifford is what lets the
    group-theoretic engine (which constructs ``rho_C`` from the quadratic form,
    never touching a unitary) share this code path with the density-matrix
    engine.  Agreement between the two is asserted in ``verify_all.py``.
    """
    pacc = acceptance_projector(check)
    target = phi_plus_projector_pair0()
    p0, fout = accepted_branch(rho_c, check)

    gens = [generator(g) for g in labels]
    n = len(gens)
    m = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            lab = _L(gens[a], gens[b], rho_c)
            acc = pacc @ lab @ pacc
            n_ab = float(np.real(np.trace(target @ acc)))
            p_ab = float(np.real(np.trace(acc)))
            val = -(n_ab - fout * p_ab) / (2.0 * p0)
            m[a, b] = m[b, a] = val
    return m


def hessian_exact(c: np.ndarray, f0: float, check: str,
                  labels: tuple[str, ...] = TWO_QUBIT_GENERATORS) -> np.ndarray:
    """Exact susceptibility Hessian over ``labels`` (default: all fifteen)."""
    return hessian_from_rho(rho_after_circuit(c, f0), check, labels)


def hessian_findiff(c: np.ndarray, f0: float, check: str,
                    labels: tuple[str, ...] = TWO_QUBIT_GENERATORS,
                    h: float = 1e-3) -> np.ndarray:
    """Central-difference Hessian of ``-F_out`` about zero angle.

    Independent of :func:`hessian_exact`: it calls the full density-matrix
    simulator with an actual matrix exponential rather than expanding by hand.
    """
    n = len(labels)
    m = np.zeros((n, n))

    def f(angles: dict[str, float]) -> float:
        return accepted_fidelity(c, f0, check, angles)

    f00 = f({})
    for a in range(n):
        ga = labels[a]
        fp = f({ga: h})
        fm = f({ga: -h})
        m[a, a] = -0.5 * (fp - 2 * f00 + fm) / h**2
    for a in range(n):
        for b in range(a + 1, n):
            ga, gb = labels[a], labels[b]
            fpp = f({ga: h, gb: h})
            fpm = f({ga: h, gb: -h})
            fmp = f({ga: -h, gb: h})
            fmm = f({ga: -h, gb: -h})
            mixed = (fpp - fpm - fmp + fmm) / (4 * h**2)
            m[a, b] = m[b, a] = -0.5 * mixed
    return m


def spectral_radius(m: np.ndarray) -> float:
    """Worst-direction susceptibility ``rho(M)`` = largest eigenvalue magnitude."""
    return float(np.max(np.abs(np.linalg.eigvalsh(m))))


def syndrome(label: str, check: str) -> int:
    """Syndrome ``s(H_a)`` of a generator against a single check.

    ``0`` if the generator's sacrificial-pair factor commutes with the check
    (logical), ``1`` if it anticommutes (detected).
    """
    p1 = label[1]
    if p1 == "I" or p1 == check:
        return 0
    return 1
