"""The group-theoretic engine: Arf-zero quadratic forms and the ten atoms.

This module never builds a circuit unitary.  It works entirely with the
symplectic image ``S`` of a Clifford, the transported quadratic form
``qtilde = q . S^{-1}``, and the population vector Eq. (28).  The state it
produces is fed to the same Hessian code path as the density-matrix engine,
so agreement between the two is a genuine cross-validation of two
independently derived routes to ``rho_C`` (Sec. 11).
"""

from __future__ import annotations

import itertools

import numpy as np

from .clifford import symplectic_image
from .paulis import bell_basis, two_pair_bell_basis

#: label bits -> Pauli name on a single pair
BITS_TO_NAME = {(1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
NAME_TO_BITS = {v: k for k, v in BITS_TO_NAME.items()}

_NONZERO = ((1, 0), (0, 1), (1, 1))
_ALL_V = tuple(itertools.product((0, 1), repeat=4))


def base_form(v: tuple[int, ...]) -> int:
    """``q(a, b) = [a != 0] + [b != 0] mod 2``, Eq. (27)."""
    a, b = v[:2], v[2:]
    return (int(any(a)) + int(any(b))) % 2


def arf(form) -> int:
    """Arf invariant of a quadratic form on ``F_2^{2n}``.

    ``Arf = 0`` iff the form takes value 0 on ``2^{2n-1} + 2^{n-1}`` points.
    """
    n_vars = 4
    zeros = sum(1 for v in _ALL_V if form(v) == 0)
    half = 2 ** (n_vars - 1)
    root = 2 ** (n_vars // 2 - 1)
    if zeros == half + root:
        return 0
    if zeros == half - root:
        return 1
    raise ValueError(f"{zeros} zeros is not a valid Arf count")


def transported_form(s: np.ndarray):
    """``qtilde = q . S^{-1}`` as a callable on labels."""
    s_inv = _inverse_mod2(s)

    def qt(v: tuple[int, ...]) -> int:
        w = tuple(int(x) % 2 for x in (s_inv @ np.array(v, dtype=np.int8)))
        return base_form(w)

    return qt


def _inverse_mod2(m: np.ndarray) -> np.ndarray:
    """Inverse of an invertible matrix over ``F_2`` by Gauss-Jordan."""
    n = m.shape[0]
    a = np.concatenate([m.astype(np.int8) % 2, np.eye(n, dtype=np.int8)], axis=1)
    row = 0
    for col in range(n):
        piv = next((r for r in range(row, n) if a[r, col]), None)
        if piv is None:
            raise ValueError("matrix is singular over F_2")
        a[[row, piv]] = a[[piv, row]]
        for r in range(n):
            if r != row and a[r, col]:
                a[r] = (a[r] + a[row]) % 2
        row += 1
    return a[:, n:]


def atom_label(s: np.ndarray) -> tuple[frozenset, frozenset]:
    """``(T_0, T_1)``: the non-identity labels of value 1 on each pair factor."""
    qt = transported_form(s)
    t0 = frozenset(
        BITS_TO_NAME[a] for a in _NONZERO if qt(a + (0, 0)) == 1
    )
    t1 = frozenset(
        BITS_TO_NAME[b] for b in _NONZERO if qt((0, 0) + b) == 1
    )
    return t0, t1


def atom_name(label: tuple[frozenset, frozenset]) -> str:
    """Human-readable atom name, e.g. ``'q[X|Z]'`` or ``'q[XYZ|XYZ]'``."""
    def fmt(t):
        return "".join(sorted(t, key="XYZ".index))
    return f"q[{fmt(label[0])}|{fmt(label[1])}]"


def populations(s: np.ndarray, f0: float) -> dict[tuple[int, ...], float]:
    """Bell populations of ``rho_C`` from the transported form, Eq. (28)."""
    q = (1.0 - f0) / 3.0
    qt = transported_form(s)
    out = {}
    for v in _ALL_V:
        if not any(v):
            out[v] = f0**2
        elif qt(v) == 1:
            out[v] = f0 * q
        else:
            out[v] = q**2
    return out


def rho_from_form(s: np.ndarray, f0: float) -> np.ndarray:
    """Build ``rho_C`` directly from the quadratic form, with no unitary."""
    vecs, labels = two_pair_bell_basis()
    pops = populations(s, f0)
    rho = np.zeros((16, 16), dtype=complex)
    for vec, lab in zip(vecs, labels):
        rho += pops[lab] * np.outer(vec, vec.conj())
    return rho


def is_purifying(label: tuple[frozenset, frozenset], check: str) -> bool:
    """Theorem 8: the atom is fixed-point iff ``c in T_1``, purifying otherwise."""
    return check not in label[1]


def enumerate_atoms(cliffords) -> dict[str, dict]:
    """Group a list of Cliffords by atom.

    Returns a mapping from atom name to a dict with the symplectic image of a
    representative, the ``(T_0, T_1)`` label, and the multiplicity.
    """
    atoms: dict[str, dict] = {}
    for c in cliffords:
        s = symplectic_image(c)
        label = atom_label(s)
        name = atom_name(label)
        entry = atoms.setdefault(
            name,
            {"label": label, "S": s, "representative": c, "multiplicity": 0},
        )
        entry["multiplicity"] += 1
    return atoms
