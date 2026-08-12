"""Closed forms from the manuscript, as exact sympy expressions.

Every function here returns a sympy expression in the raw fidelity ``F0``, so
that identities can be checked symbolically rather than numerically.  The
numeric engines in :mod:`susceptibility` and :mod:`classification` are checked
against these in ``verify_all.py``.
"""

from __future__ import annotations

import sympy as sp

F0 = sp.Symbol("F0", positive=True)

#: ``D(F_0) = 8 F_0^2 - 4 F_0 + 5``; discriminant -144, so strictly positive
D = 8 * F0**2 - 4 * F0 + 5

#: accepted fidelity of a purifying atom, Eq. (14)
FOUT_PURIFYING = (10 * F0**2 - 2 * F0 + 1) / D
#: acceptance probability of a purifying atom, Eq. (12)
P0_PURIFYING = D / 9
#: acceptance probability of a fixed-point atom, Thm. 8
P0_FIXED = (2 * F0 + 1) / 3

# --------------------------------------------------------------------------
# the five-letter alphabet on purifying atoms, Eq. (30)
# --------------------------------------------------------------------------

LAMBDA = {
    "A": 4 * (2 * F0 + 1) * (4 * F0 - 1) / D,
    "B": 4 * (4 * F0 - 1) ** 2 / D,
    "C": 4 * (1 - F0) * (2 * F0 + 1) * (4 * F0 - 1) * (8 * F0 + 1) / D**2,
    "D": 8 * (1 - F0) * (F0 + 2) * (2 * F0 + 1) * (4 * F0 - 1) / D**2,
    "E": 12 * (1 - F0) * (2 * F0 - 1) * (2 * F0 + 1) * (4 * F0 - 1) / D**2,
}

#: the three letters on fixed-point atoms, Eq. (31)
MU = {
    "A": 4 * (4 * F0 - 1) / 3,
    "B": 4 * (1 - F0) * (4 * F0 - 1) / 3,
    "C": -4 * (1 - F0) * (2 * F0 - 1) * (4 * F0 - 1) / (3 * (2 * F0 + 1)),
}

#: letters carrying the threshold factor ``(2 F_0 - 1)``
THRESHOLD_LETTERS = ("E",)

#: the recurrence fixed-point polynomial, whose roots are 1/4, 1/2, 1
RECURRENCE_POLY = 8 * F0**3 - 14 * F0**2 + 7 * F0 - 1

#: the three fixed-circuit sensitivities of Eqs. (15)-(22)
M_FWD_Z = {"XI": LAMBDA["A"], "XZ": LAMBDA["A"], "ZZ": LAMBDA["B"]}


def alphabet_numeric(f0: float) -> dict[str, float]:
    """The five letters evaluated at a numeric fidelity."""
    return {k: float(v.subs(F0, f0)) for k, v in LAMBDA.items()}


def mu_numeric(f0: float) -> dict[str, float]:
    """The three fixed-point letters evaluated at a numeric fidelity."""
    return {k: float(v.subs(F0, f0)) for k, v in MU.items()}


def identify_letter(value: float, f0: float, tol: float = 1e-9) -> str | None:
    """Match a numeric eigenvalue to a letter of the alphabet.

    Returns the letter name, ``'0'`` for a null direction, or ``None`` if the
    value is not in the alphabet -- which is the condition ``verify_all.py``
    asserts never occurs on a purifying atom.
    """
    if abs(value) < tol:
        return "0"
    for name, expr in {**{f"lambda_{k}": v for k, v in LAMBDA.items()},
                       **{f"mu_{k}": v for k, v in MU.items()}}.items():
        if abs(value - float(expr.subs(F0, f0))) < tol:
            return name
    return None


def worst_case_no_go() -> sp.Expr:
    """Thm. 9: ``lambda_max`` on every purifying atom of every check."""
    return LAMBDA["A"]


def no_go_differences() -> dict[str, tuple[sp.Expr, sp.Expr | None]]:
    """The four differences in the Thm. 9 proof, with their discriminants.

    ``lambda_A - lambda_B`` is manifestly positive on ``(1/4, 1)`` and has no
    quadratic cofactor.  The other three factor as a positive prefactor times a
    quadratic in ``F_0`` with discriminants -135, -36 and -279 respectively,
    hence are strictly positive.
    """
    out: dict[str, tuple[sp.Expr, sp.Expr | None]] = {}
    prefactor = 4 * (2 * F0 + 1) * (4 * F0 - 1) / D**2
    for other in ("B", "C", "D", "E"):
        diff = sp.factor(sp.simplify(LAMBDA["A"] - LAMBDA[other]))
        if other == "B":
            out[other] = (diff, None)
            continue
        quad = sp.simplify(sp.cancel(diff / prefactor))
        poly = sp.Poly(quad, F0)
        disc = sp.discriminant(poly, F0) if poly.degree() == 2 else None
        out[other] = (diff, disc)
    return out
