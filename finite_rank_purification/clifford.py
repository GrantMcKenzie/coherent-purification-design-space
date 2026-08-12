"""Enumeration of the two-qubit Clifford group modulo phases.

The group ``C_2 / phases`` has order 11520.  Its image in ``Sp(4, 2)`` has
order 720, the kernel being the Pauli group modulo phases (order 16), and
``11520 = 720 * 16``.  Because a Pauli factor does not change the Bell-diagonal
state ``rho_C`` produced by ``C tensor conj(C)``, the classification of
Sec. 9 is really a statement about the 720 symplectic images; the multiplicity
1152 per atom is ``720/10 * 16``.

Elements are represented as ``4 x 4`` complex unitaries, canonicalised modulo
global phase so that they can be used as dictionary keys.
"""

from __future__ import annotations

import numpy as np

from .paulis import I2, X, Y, Z, kron

_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

#: CNOT with qubit 0 as control, qubit 1 as target
CNOT01 = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
#: CNOT with qubit 1 as control, qubit 0 as target
CNOT10 = np.array(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
)

GENERATORS = (
    kron(_H, I2),
    kron(I2, _H),
    kron(_S, I2),
    kron(I2, _S),
    CNOT01,
)

_ROUND = 9


def canonical(u: np.ndarray) -> bytes:
    """Canonical key for a unitary modulo global phase.

    The first entry of magnitude above tolerance is rotated to be real and
    positive, then the matrix is rounded.  Rounding at 9 decimals is safe here
    because every entry of a two-qubit Clifford lies in
    ``(1/2) Z[i]`` scaled by powers of ``1/sqrt(2)``, so distinct elements are
    separated by far more than the rounding step.
    """
    flat = u.reshape(-1)
    idx = int(np.argmax(np.abs(flat) > 1e-9))
    phase = flat[idx] / abs(flat[idx])
    v = (u / phase).round(_ROUND)
    v = v + 0.0  # normalise -0.0 to 0.0
    return v.tobytes()


def enumerate_clifford2() -> list[np.ndarray]:
    """Breadth-first enumeration of all 11520 two-qubit Cliffords mod phase."""
    identity = np.eye(4, dtype=complex)
    seen = {canonical(identity): identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for u in frontier:
            for g in GENERATORS:
                w = g @ u
                key = canonical(w)
                if key not in seen:
                    seen[key] = w
                    nxt.append(w)
        frontier = nxt
    return list(seen.values())


# --------------------------------------------------------------------------
# symplectic image
# --------------------------------------------------------------------------

_LABEL_PAULIS = {
    (0, 0): I2,
    (1, 0): X,
    (0, 1): Z,
    (1, 1): X @ Z,
}


def _two_qubit_from_label(v: tuple[int, int, int, int]) -> np.ndarray:
    return kron(_LABEL_PAULIS[(v[0], v[1])], _LABEL_PAULIS[(v[2], v[3])])


_BASIS_LABELS = (
    (1, 0, 0, 0),  # X on qubit 0
    (0, 1, 0, 0),  # Z on qubit 0
    (0, 0, 1, 0),  # X on qubit 1
    (0, 0, 0, 1),  # Z on qubit 1
)

_ALL_LABELS = [
    (a, b, c, d)
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
    for d in (0, 1)
]
_LABEL_MATRICES = {v: _two_qubit_from_label(v) for v in _ALL_LABELS}


def symplectic_image(c: np.ndarray) -> np.ndarray:
    """The ``4 x 4`` matrix over ``F_2`` induced by conjugation ``P -> C P C^dag``.

    Columns are the images of ``X_0, Z_0, X_1, Z_1`` in the label basis
    ``(x_0, z_0, x_1, z_1)``.
    """
    cols = []
    for label in _BASIS_LABELS:
        conj = c @ _LABEL_MATRICES[label] @ c.conj().T
        for target, mat in _LABEL_MATRICES.items():
            if target == (0, 0, 0, 0):
                continue
            overlap = np.trace(conj @ mat.conj().T) / 4.0
            if abs(abs(overlap) - 1.0) < 1e-8:
                cols.append(target)
                break
        else:  # pragma: no cover - would mean C is not Clifford
            raise RuntimeError("conjugation left the Pauli group")
    return np.array(cols, dtype=np.int8).T % 2


def symplectic_key(c: np.ndarray) -> bytes:
    """Hashable key for the symplectic image of ``c``."""
    return symplectic_image(c).tobytes()
