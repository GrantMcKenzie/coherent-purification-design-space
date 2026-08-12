"""Coherent-error robustness of entanglement purification.

Companion package to the manuscript *Coherent-error robustness of entanglement
purification is a finite design problem*.

Two independent engines are provided and cross-validated against each other:

*   a **density-matrix engine** (:mod:`circuits`, :mod:`susceptibility`) that
    builds the 16 x 16 circuit unitary explicitly and simulates the accepted
    branch, and
*   a **group-theoretic engine** (:mod:`clifford`, :mod:`classification`) that
    never touches a unitary, working instead with the symplectic image of the
    Clifford and the transported Arf-zero quadratic form.

``verify_all.py`` asserts that they agree, and reproduces every stated result.
"""

__version__ = "1.0.0"

from . import (  # noqa: F401
    circuits,
    classification,
    clifford,
    closed_forms,
    paulis,
    susceptibility,
)

__all__ = [
    "paulis",
    "clifford",
    "circuits",
    "susceptibility",
    "classification",
    "closed_forms",
    "__version__",
]
