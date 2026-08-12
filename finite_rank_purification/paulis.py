"""Pauli algebra, the two-pair Bell basis, and the symplectic labelling.

Conventions follow the manuscript:

*   Qubit order is ``[A0, B0, A1, B1]``: pair ``k`` occupies qubits
    ``(2k, 2k+1)``.  Pair 0 is kept, pair 1 is sacrificial.
*   Alice holds ``(A0, A1) = (0, 2)``; Bob holds ``(B0, B1) = (1, 3)``.
*   Bell states are ``|Phi^pm> = (|00> +- |11>)/sqrt(2)`` and
    ``|Psi^pm> = (|01> +- |10>)/sqrt(2)``, with target ``|Phi^+>``.
*   A single-pair Bell label is ``a = (a_x, a_z) in F_2^2``, defined by
    ``|beta_a> = (X^{a_x} Z^{a_z} tensor I) |Phi^+>``.  A two-pair label is
    ``v = (a, b) in F_2^4`` with ``a`` the pair-0 bits and ``b`` the pair-1 bits.
*   Generator labels ``(P0, P1)`` are always *invariant* labels: ``P0`` is the
    Pauli on the kept pair, ``P1`` the Pauli on the sacrificial pair.  This is
    the convention of Sec. 2.4, not the (control, target) convention used in
    the fixed-circuit figures.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------------
# single-qubit Paulis
# --------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
PAULI_LABELS = ("I", "X", "Y", "Z")

#: the fifteen non-identity two-qubit Pauli labels, in invariant (P0, P1) order
TWO_QUBIT_GENERATORS = tuple(
    p0 + p1
    for p0, p1 in itertools.product(PAULI_LABELS, repeat=2)
    if p0 + p1 != "II"
)
assert len(TWO_QUBIT_GENERATORS) == 15


def kron(*mats: np.ndarray) -> np.ndarray:
    """Kronecker product of any number of matrices, left to right."""
    out = np.array([[1.0 + 0.0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def pauli_string(label: str) -> np.ndarray:
    """Dense matrix of a Pauli string such as ``'XZ'`` or ``'IY'``."""
    return kron(*(PAULI[c] for c in label))


# --------------------------------------------------------------------------
# Bell basis
# --------------------------------------------------------------------------

_PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def bell_label_to_pauli(a: tuple[int, int]) -> np.ndarray:
    """``X^{a_x} Z^{a_z}`` as a single-qubit matrix."""
    ax, az = a
    m = np.eye(2, dtype=complex)
    if az:
        m = m @ Z
    if ax:
        m = X @ m
    return m


@lru_cache(maxsize=None)
def bell_basis() -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Return ``(vectors, labels)`` for the four single-pair Bell states.

    ``vectors[i]`` is the state with label ``labels[i] = (a_x, a_z)``.
    """
    labels = tuple(itertools.product((0, 1), repeat=2))
    vecs = np.empty((4, 4), dtype=complex)
    for i, a in enumerate(labels):
        op = kron(bell_label_to_pauli(a), I2)
        vecs[i] = op @ _PHI_PLUS
    return vecs, labels


def phi_plus_projector_pair0() -> np.ndarray:
    """``|Phi^+><Phi^+|`` on the kept pair (A0, B0), identity elsewhere."""
    proj = np.outer(_PHI_PLUS, _PHI_PLUS.conj())
    return kron(proj, np.eye(4, dtype=complex))


@lru_cache(maxsize=None)
def two_pair_bell_basis() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """The sixteen two-pair Bell states in the ``[A0, B0, A1, B1]`` ordering.

    Returns ``(vectors, labels)`` where each label is
    ``v = (a_x, a_z, b_x, b_z)``.
    """
    single, single_labels = bell_basis()
    vecs, labels = [], []
    for i, a in enumerate(single_labels):
        for j, b in enumerate(single_labels):
            vecs.append(np.kron(single[i], single[j]))
            labels.append(tuple(a) + tuple(b))
    return np.array(vecs), tuple(labels)


def symplectic_form(u: tuple[int, ...], v: tuple[int, ...]) -> int:
    """Symplectic form on ``F_2^{2n}`` recording (anti)commutation.

    Labels are ordered ``(x_1, z_1, x_2, z_2, ...)``; the form is
    ``sum_i (x_i z'_i + z_i x'_i) mod 2``.
    """
    if len(u) != len(v) or len(u) % 2:
        raise ValueError("labels must have equal, even length")
    total = 0
    for i in range(0, len(u), 2):
        total += u[i] * v[i + 1] + u[i + 1] * v[i]
    return total % 2


def werner(f0: float) -> np.ndarray:
    """Symmetric Bell-diagonal (Werner) single-pair state, Eq. (2)."""
    q = (1.0 - f0) / 3.0
    vecs, _ = bell_basis()
    rho = np.zeros((4, 4), dtype=complex)
    for i, vec in enumerate(vecs):
        weight = f0 if i == 0 else q
        rho += weight * np.outer(vec, vec.conj())
    return rho


def two_pair_input(f0: float) -> np.ndarray:
    """``rho_0 tensor rho_0`` in the ``[A0, B0, A1, B1]`` ordering."""
    single = werner(f0)
    return np.kron(single, single)


def D(f0):
    """``D(F_0) = 8 F_0^2 - 4 F_0 + 5``; strictly positive (discriminant -144)."""
    return 8 * f0**2 - 4 * f0 + 5
