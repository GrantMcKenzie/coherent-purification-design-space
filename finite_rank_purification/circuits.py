"""Bilateral-Clifford circuits, Pauli checks, and the accepted branch.

The circuit is ``U_C = C_(A0,A1) tensor conj(C)_(B0,B1)`` followed by a single
Pauli check on the sacrificial pair,
``P_acc = (1/2)(I +- c_{A1} c_{B1})`` with ``+`` for the ``Z`` and ``X``
coincidence checks and ``-`` for the ``Y`` anti-coincidence check.

The conjugation on Bob's side is what makes ``U_C`` preserve
``|Phi^+>^{tensor 2}``, and it is the origin of the ``Y`` asymmetry in the
blind cores.
"""

from __future__ import annotations

import numpy as np

from .clifford import CNOT01, CNOT10
from .paulis import (
    PAULI,
    kron,
    phi_plus_projector_pair0,
    two_pair_input,
)

CHECKS = ("Z", "X", "Y")

#: sign of the check: ``+1`` coincidence, ``-1`` anti-coincidence
CHECK_SIGN = {"Z": +1, "X": +1, "Y": -1}


def _reorder_alice_bob(op: np.ndarray) -> np.ndarray:
    """Map an operator from ``[A0, A1, B0, B1]`` to ``[A0, B0, A1, B1]``."""
    t = op.reshape([2] * 8)
    # source axes: (A0, A1, B0, B1 | A0', A1', B0', B1')
    # target axes: (A0, B0, A1, B1 | A0', B0', A1', B1')
    t = t.transpose(0, 2, 1, 3, 4, 6, 5, 7)
    return t.reshape(16, 16)


def bilateral(c: np.ndarray) -> np.ndarray:
    """``U_C = C_(A0,A1) tensor conj(C)_(B0,B1)`` in the ``[A0,B0,A1,B1]`` order."""
    return _reorder_alice_bob(np.kron(c, c.conj()))


def acceptance_projector(check: str) -> np.ndarray:
    """``P_acc`` acting on the sacrificial pair ``(A1, B1) = (2, 3)``."""
    if check not in CHECK_SIGN:
        raise ValueError(f"unknown check {check!r}")
    p = PAULI[check]
    corr = kron(np.eye(4, dtype=complex), p, p)
    return 0.5 * (np.eye(16, dtype=complex) + CHECK_SIGN[check] * corr)


def generator(label: str) -> np.ndarray:
    """Common-mode generator ``H_g = P_(A0,A1) + P_(B0,B1)``, Eq. (5).

    ``label`` is an invariant two-character label ``(P0, P1)``: ``P0`` acts on
    the kept pair, ``P1`` on the sacrificial pair.
    """
    p0, p1 = PAULI[label[0]], PAULI[label[1]]
    eye = np.eye(2, dtype=complex)
    alice = kron(p0, eye, p1, eye)  # qubits A0=0, A1=2
    bob = kron(eye, p0, eye, p1)  # qubits B0=1, B1=3
    return alice + bob


def rho_after_circuit(c: np.ndarray, f0: float) -> np.ndarray:
    """``rho_C = U_C rho_in U_C^dag``; Bell diagonal for every ``C``."""
    u = bilateral(c)
    return u @ two_pair_input(f0) @ u.conj().T


def accepted_branch(rho_c: np.ndarray, check: str) -> tuple[float, float]:
    """Return ``(p_0, F_out)`` for the accepted branch with no coherent error."""
    pacc = acceptance_projector(check)
    target = phi_plus_projector_pair0()
    accepted = pacc @ rho_c @ pacc
    p0 = float(np.real(np.trace(accepted)))
    n0 = float(np.real(np.trace(target @ accepted)))
    return p0, n0 / p0


def accepted_fidelity(c: np.ndarray, f0: float, check: str,
                      angles: dict[str, float] | None = None) -> float:
    """Accepted fidelity, optionally with coherent over-rotation angles.

    ``angles`` maps invariant generator labels to angles; the coherent unitary
    is ``exp(-i sum_g theta_g H_g)`` applied after the circuit Clifford.
    """
    rho = rho_after_circuit(c, f0)
    if angles:
        from scipy.linalg import expm

        h = sum(theta * generator(g) for g, theta in angles.items())
        u = expm(-1j * h)
        rho = u @ rho @ u.conj().T
    _, fout = accepted_branch(rho, check)
    return fout


# --------------------------------------------------------------------------
# the three fixed-family circuits of Secs. 3-8
# --------------------------------------------------------------------------

#: forward orientation: the kept pair controls the sacrificial pair
C_FORWARD = CNOT01
#: reverse orientation: the sacrificial pair controls the kept pair
C_REVERSE = CNOT10

FIXED_FAMILY = {
    "forward-Z": (C_FORWARD, "Z"),
    "reverse-X": (C_REVERSE, "X"),
    "reverse-Y": (C_REVERSE, "Y"),
}
