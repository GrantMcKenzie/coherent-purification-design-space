#!/usr/bin/env python3
"""Reproduce and assert every stated result of the manuscript.

Run with ``python verify_all.py``.  Exits nonzero if any claim fails.

Each check prints the claim, the reproduced value, and PASS/FAIL.  Claims are
tagged with the label they carry in the paper's status ledger:

    [THM]  proved in the paper; here reproduced numerically
    [VER]  established by this computation
    [FIX]  a correction to an earlier draft, retained as a regression guard

The [FIX] entries exist because three claims in an earlier draft were wrong.
They are asserted here so the errors cannot reappear silently.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter

import numpy as np
import sympy as sp

from finite_rank_purification import closed_forms as cf
from finite_rank_purification.circuits import (
    FIXED_FAMILY,
    accepted_branch,
    accepted_fidelity,
    rho_after_circuit,
)
from finite_rank_purification.classification import (
    arf,
    atom_label,
    atom_name,
    base_form,
    enumerate_atoms,
    is_purifying,
    rho_from_form,
    transported_form,
)
from finite_rank_purification.clifford import enumerate_clifford2, symplectic_image
from finite_rank_purification.paulis import D, TWO_QUBIT_GENERATORS
from finite_rank_purification.susceptibility import (
    hessian_exact,
    hessian_findiff,
    hessian_from_rho,
    spectral_radius,
    syndrome,
)

F0 = cf.F0
FAILURES: list[str] = []
COUNT = 0


def check(tag: str, claim: str, condition: bool, detail: str = "") -> None:
    global COUNT
    COUNT += 1
    status = "PASS" if condition else "FAIL"
    line = f"  [{tag}] {claim:<62s} {status}"
    if detail:
        line += f"   {detail}"
    print(line)
    if not condition:
        FAILURES.append(claim)


def close(a, b, tol=1e-9) -> bool:
    return bool(abs(float(a) - float(b)) < tol)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ==========================================================================
print(__doc__.splitlines()[0])
print("=" * 78)

F = 0.9  # working fidelity for numeric checks

section("Setup: group enumeration")
cliffords = enumerate_clifford2()
check("THM", "|C_2 / phases| = 11520", len(cliffords) == 11520, f"got {len(cliffords)}")
sympl = {symplectic_image(c).tobytes() for c in cliffords}
check("THM", "|Sp(4,2)| = 720 distinct symplectic images", len(sympl) == 720,
      f"got {len(sympl)}")
check("THM", "11520 = 720 x 16 (Pauli fibre)", len(cliffords) == len(sympl) * 16)

atoms = enumerate_atoms(cliffords)

# --------------------------------------------------------------------------
section("Sec. 3.2  Lemma 1: the linear term vanishes")
for name, (c, chk) in FIXED_FAMILY.items():
    grads = []
    for g in TWO_QUBIT_GENERATORS:
        h = 1e-4
        fp = accepted_fidelity(c, F, chk, {g: h})
        fm = accepted_fidelity(c, F, chk, {g: -h})
        grads.append(abs(fp - fm) / (2 * h))
    check("THM", f"{name}: max |dF/dtheta| = 0 over 15 generators",
          max(grads) < 1e-8, f"max {max(grads):.2e}")

# --------------------------------------------------------------------------
section("Sec. 3.3-3.4  Exact one-parameter forms and the baseline")
c, chk = FIXED_FAMILY["forward-Z"]
worst = 0.0
for g, coeff in (("XI", (2 * F + 1) * (4 * F - 1) / D(F)),
                 ("XZ", (2 * F + 1) * (4 * F - 1) / D(F)),
                 ("ZZ", (4 * F - 1) ** 2 / D(F))):
    f_out0 = accepted_fidelity(c, F, chk)
    for theta in np.linspace(-0.5, 0.5, 41):
        exact = accepted_fidelity(c, F, chk, {g: float(theta)})
        closed = f_out0 - coeff * np.sin(2 * theta) ** 2
        worst = max(worst, abs(exact - closed))
check("THM", "F_g(theta) = F(0) - c sin^2(2 theta) exactly, Eqs. (10)-(11)",
      worst < 1e-14, f"max dev {worst:.1e}")

p0, fout = accepted_branch(rho_after_circuit(c, F), chk)
check("THM", "p_0 = D(F_0)/9, Eq. (12)", close(p0, D(F) / 9))
check("THM", "F_out = (10F^2-2F+1)/D, Eq. (14)",
      close(fout, (10 * F**2 - 2 * F + 1) / D(F)))

# --------------------------------------------------------------------------
section("Sec. 4  Closed-form sensitivities of the fixed family")
sub = ("XI", "XZ", "ZZ")
expected = {
    "forward-Z": (3.6954314720812, 3.6954314720812, 3.4314720812183),
    "reverse-Y": (0.2719987631735, 0.2719987631735, 0.3845499755211),
}
for name, (c, chk) in FIXED_FAMILY.items():
    m = hessian_exact(c, F, chk, sub)
    off = np.abs(m - np.diag(np.diag(m))).max()
    check("THM", f"{name}: sub-family M is exactly diagonal", off < 1e-12,
          f"max off-diag {off:.1e}")
    mf = hessian_findiff(c, F, chk, sub)
    check("VER", f"{name}: exact vs finite-difference Hessian agree",
          np.abs(m - mf).max() < 1e-5, f"max diff {np.abs(m - mf).max():.1e}")

# note: in invariant labels the reverse-X null is IX, not XI (Sec. 2.4)
c_rev, _ = FIXED_FAMILY["reverse-X"]
m_rev = hessian_exact(c_rev, F, "X", ("IX",))
check("THM", "reverse-X: M_{IX,IX} = 0 exactly (invariant label)",
      abs(m_rev[0, 0]) < 1e-12, f"got {m_rev[0,0]:.1e}")

for f0 in (0.3, 0.5, 0.7, 0.9, 0.99):
    m = hessian_exact(FIXED_FAMILY["forward-Z"][0], f0, "Z", sub)
    ok = (close(m[0, 0], 4 * (2 * f0 + 1) * (4 * f0 - 1) / D(f0))
          and close(m[2, 2], 4 * (4 * f0 - 1) ** 2 / D(f0)))
    check("THM", f"forward-Z closed forms (15)-(17) at F_0={f0}", ok)

m_quarter = hessian_exact(FIXED_FAMILY["forward-Z"][0], 0.25, "Z")
check("THM", "all sensitivity vanishes at F_0 = 1/4 (maximally mixed)",
      np.abs(m_quarter).max() < 1e-12, f"max {np.abs(m_quarter).max():.1e}")

# --------------------------------------------------------------------------
section("Sec. 5  Theorem 2: syndrome-block decomposition")
for name, (c, chk) in FIXED_FAMILY.items():
    m = hessian_exact(c, F, chk)
    worst_off = 0.0
    for a, ga in enumerate(TWO_QUBIT_GENERATORS):
        for b, gb in enumerate(TWO_QUBIT_GENERATORS):
            if syndrome(ga, chk) != syndrome(gb, chk):
                worst_off = max(worst_off, abs(m[a, b]))
    check("THM", f"{name}: M_ab = 0 across syndrome sectors",
          worst_off < 1e-12, f"max off-block {worst_off:.1e}")

# --------------------------------------------------------------------------
section("Sec. 7  Theorem 4: threshold inertia (forward-Z)")
c, chk = FIXED_FAMILY["forward-Z"]
for f0, want_rank in ((0.25, 0), (0.5, 8), (1.0, 4), (0.9, 10), (0.4, 10)):
    ev = np.linalg.eigvalsh(hessian_exact(c, f0, chk))
    rank = int(np.sum(np.abs(ev) > 1e-9))
    check("THM", f"rank M = {want_rank} at F_0 = {f0}", rank == want_rank,
          f"got {rank}")

for f0, psd in ((0.6, True), (0.9, True), (1.0, True), (0.25, True),
                (0.3, False), (0.45, False), (0.1, False)):
    ev = np.linalg.eigvalsh(hessian_exact(c, f0, chk))
    is_psd = bool(ev.min() > -1e-9)
    check("THM", f"M PSD is {psd} at F_0 = {f0}", is_psd == psd,
          f"lambda_min {ev.min():+.4f}")

# the proof needs a strictly positive eigenvalue on (0, 1/4), not only a
# negative one -- this is the clause added to the Thm. 4 proof
ev_low = np.linalg.eigvalsh(hessian_exact(c, 0.15, chk))
check("FIX", "M indefinite on (0,1/4): both signs present",
      ev_low.min() < -1e-9 and ev_low.max() > 1e-9,
      f"[{ev_low.min():+.4f}, {ev_low.max():+.4f}]")

check("THM", "8F^3-14F^2+7F-1 = (4F-1)(2F-1)(F-1)",
      sp.simplify(cf.RECURRENCE_POLY - (4 * F0 - 1) * (2 * F0 - 1) * (F0 - 1)) == 0)

# --------------------------------------------------------------------------
section("Sec. 9.1-9.3  Theorem 7: the ten-atom classification")
check("THM", "exactly ten atoms", len(atoms) == 10, f"got {len(atoms)}")
mults = {a["multiplicity"] for a in atoms.values()}
check("THM", "every atom has multiplicity 1152", mults == {1152}, f"got {mults}")
check("THM", "Arf(q) = 0 for the Werner-squared input", arf(base_form) == 0)
for name, a in atoms.items():
    if arf(transported_form(a["S"])) != 0:
        check("THM", f"Arf(qtilde) = 0 for {name}", False)
        break
else:
    check("THM", "Arf(qtilde) = 0 on all ten atoms (Sp-invariance)", True)

labels = {atom_name(a["label"]): a["label"] for a in atoms.values()}
shapes = {(len(t0), len(t1)) for t0, t1 in labels.values()}
check("THM", "labels are both-singleton or both-full", shapes <= {(1, 1), (3, 3)},
      f"got {sorted(shapes)}")

# engine cross-validation: unitary route vs quadratic-form route
worst_eng = 0.0
for a in atoms.values():
    r1 = rho_after_circuit(a["representative"], F)
    r2 = rho_from_form(a["S"], F)
    worst_eng = max(worst_eng, float(np.abs(r1 - r2).max()))
check("VER", "density-matrix and group-theoretic engines agree on rho_C",
      worst_eng < 1e-12, f"max diff {worst_eng:.1e}")

# --------------------------------------------------------------------------
section("Sec. 9.4  Theorem 8: purifying / fixed-point dichotomy")
for chk in "ZXY":
    npur = sum(1 for a in atoms.values() if is_purifying(a["label"], chk))
    check("THM", f"check {chk}: 6 purifying, 4 fixed-point atoms", npur == 6,
          f"got {npur}")

for a in atoms.values():
    for chk in "ZXY":
        p0, fout = accepted_branch(rho_from_form(a["S"], F), chk)
        if is_purifying(a["label"], chk):
            ok = close(p0, D(F) / 9) and close(fout, (10 * F**2 - 2 * F + 1) / D(F))
        else:
            ok = close(p0, (2 * F + 1) / 3) and close(fout, F)
        if not ok:
            check("THM", f"dichotomy table fails at {atom_name(a['label'])}/{chk}", False)
            break
    else:
        continue
    break
else:
    check("THM", "exactly two (p_0, F_out) pairs over all 30 rows", True)

fwd_atom = atom_name(atom_label(symplectic_image(FIXED_FAMILY["forward-Z"][0])))
rev_atom = atom_name(atom_label(symplectic_image(FIXED_FAMILY["reverse-X"][0])))
check("VER", "forward-CNOT lands in q[Z|X]", fwd_atom == "q[Z|X]", fwd_atom)
check("VER", "reverse-CNOT lands in q[X|Z] (Sec. 11)", rev_atom == "q[X|Z]", rev_atom)
check("VER", "reverse-CNOT is fixed-point under a Z check",
      not is_purifying(labels[rev_atom], "Z"))

# --------------------------------------------------------------------------
section("Sec. 9.5  Universal eigenbasis and the five-letter alphabet")
lam = cf.alphabet_numeric(F)
mu = cf.mu_numeric(F)

table = {}
worst_off = 0.0
for name, a in atoms.items():
    for chk in "ZXY":
        m = hessian_from_rho(rho_from_form(a["S"], F), chk)
        worst_off = max(worst_off, float(np.abs(m - np.diag(np.diag(m))).max()))
        diag = np.diag(m)
        ref = lam if is_purifying(a["label"], chk) else mu
        pattern = Counter()
        unknown = []
        for val in diag:
            if abs(val) < 1e-9:
                pattern["0"] += 1
                continue
            for k, v in ref.items():
                if abs(val - v) < 1e-9:
                    pattern[k] += 1
                    break
            else:
                unknown.append(val)
        table[(name, chk)] = (pattern, unknown, float(diag.sum()),
                              int(np.sum(np.abs(diag) > 1e-9)))

check("VER", "M is diagonal in the Pauli basis on all 30 (atom, check) pairs",
      worst_off == 0.0, f"max off-diagonal {worst_off:.1e} (exactly zero)")
check("VER", "no eigenvalue outside the alphabet on any of the 30 pairs",
      all(not u for _, u, _, _ in table.values()))

pur = {k: v for k, v in table.items() if is_purifying(labels[k[0]], k[1])}
check("THM", "18 purifying and 12 fixed-point (atom, check) pairs",
      len(pur) == 18 and len(table) - len(pur) == 12)

# [FIX] the Y check has no exceptional atom
ypat = {tuple(sorted(p.items())) for (n, c), (p, _, _, _) in pur.items() if c == "Y"}
check("FIX", "Y check: all six purifying atoms share one pattern",
      len(ypat) == 1, f"{len(ypat)} distinct pattern(s)")
yranks = {r for (n, c), (_, _, _, r) in pur.items() if c == "Y"}
check("FIX", "Y check: uniform rank 10", yranks == {10}, f"got {yranks}")

zx_exc = {n for (n, c), (p, _, _, r) in pur.items() if c in "ZX" and r != 10}
check("VER", "Z/X checks: exceptional atoms are exactly those with T_0 = {Y}",
      zx_exc == {"q[Y|X]", "q[Y|Y]", "q[Y|Z]"}, str(sorted(zx_exc)))

n_lamE = sum(1 for _, (p, _, _, _) in pur.items() if p.get("E", 0) > 0)
check("THM", "16 of the 18 purifying pairs carry lambda_E", n_lamE == 16,
      f"got {n_lamE}")
n_muC = sum(1 for k, (p, _, _, _) in table.items()
            if not is_purifying(labels[k[0]], k[1]) and p.get("C", 0) > 0)
check("THM", "11 of the 12 fixed-point pairs carry mu_C", n_muC == 11, f"got {n_muC}")

qyy = table[("q[Y|Y]", "Z")][0]
check("THM", "q[Y|Y] carries no threshold letter (escapes the threshold)",
      qyy.get("E", 0) == 0)

# --------------------------------------------------------------------------
section("Sec. 10  Trace of M and the covariance-weighted payoff")
traces = Counter(round(v[2], 4) for k, v in pur.items())
check("FIX", "Z/X purifying pairs have THREE distinct traces, not one",
      sorted(t for t in traces if t != 12.9045) == [15.792, 15.8697, 16.7701],
      str(dict(traces)))
ytr = {round(v[2], 4) for k, v in pur.items() if k[1] == "Y"}
check("THM", "Y check: Tr M = 12.9045, six-fold degenerate", ytr == {12.9045},
      str(ytr))
tr_closed = -4 * (4 * F - 1) * (8 * F**3 + 12 * F**2 - 84 * F - 17) / D(F) ** 2
check("THM", "Eq. (36) reproduces the Y-check trace", close(tr_closed, 12.90453245, 1e-6))
sym_pattern = (2 * cf.LAMBDA["A"] + cf.LAMBDA["B"] + 3 * cf.LAMBDA["C"]
               + 3 * cf.LAMBDA["D"] + cf.LAMBDA["E"])
tr_expr = -4 * (4 * F0 - 1) * (8 * F0**3 + 12 * F0**2 - 84 * F0 - 17) / cf.D**2
check("VER", "Eq. (36) == pattern (2,1,3,3,1) identically in F_0",
      sp.simplify(sym_pattern - tr_expr) == 0)
check("THM", "Y check is the isotropic optimum", min(traces) == 12.9045)

# --------------------------------------------------------------------------
section("Sec. 9.6  Theorem 9: the worst-case no-go")
radii = {round(spectral_radius(
    hessian_from_rho(rho_from_form(atoms[n]["S"], F), c)), 10)
    for (n, c) in pur}
check("THM", "rho(M) identical on every purifying (atom, check) pair",
      len(radii) == 1, f"{radii}")
check("THM", "rho(M) = lambda_A = 4(2F+1)(4F-1)/D", close(list(radii)[0], lam["A"]))

for other, (diff, disc) in cf.no_go_differences().items():
    if other == "B":
        continue
    check("THM", f"lambda_A - lambda_{other}: discriminant < 0",
          disc is not None and float(disc) < 0, f"disc = {disc}")

# --------------------------------------------------------------------------
section("Sec. 9.7  Irreducible core and blind cores")
nullable = set()
for (n, chk) in pur:
    m = np.diag(hessian_from_rho(rho_from_form(atoms[n]["S"], F), chk))
    for g, val in zip(TWO_QUBIT_GENERATORS, m):
        if abs(val) < 1e-9:
            nullable.add(g)
irreducible = set(TWO_QUBIT_GENERATORS) - nullable
check("VER", "8 nullable, 7 irreducible generators",
      len(nullable) == 8 and len(irreducible) == 7,
      f"{len(nullable)}/{len(irreducible)}")
check("VER", "irreducible core = {XI,XX,XZ,ZI,ZX,ZZ,YY}",
      irreducible == {"XI", "XX", "XZ", "ZI", "ZX", "ZZ", "YY"},
      str(sorted(irreducible)))

predicted = {g for g in TWO_QUBIT_GENERATORS
             if g[0] == "I" or (g[0] == "Y") != (g[1] == "Y")}
check("THM", "criterion (33) predicts the nullable set exactly",
      predicted == nullable)

blind = {}
for chk in "ZXY":
    common = None
    for n in {n for (n, c) in pur if c == chk}:
        m = np.diag(hessian_from_rho(rho_from_form(atoms[n]["S"], F), chk))
        nulls = {g for g, v in zip(TWO_QUBIT_GENERATORS, m) if abs(v) < 1e-9}
        common = nulls if common is None else common & nulls
    blind[chk] = common
check("VER", "blind core (Z) = {IZ, YI, YZ}", blind["Z"] == {"IZ", "YI", "YZ"},
      str(sorted(blind["Z"])))
check("VER", "blind core (X) = {IX, YI, YX}", blind["X"] == {"IX", "YI", "YX"},
      str(sorted(blind["X"])))
check("VER", "blind core (Y) = {IY, XY, YI, ZY} (larger, Y asymmetry)",
      blind["Y"] == {"IY", "XY", "YI", "ZY"}, str(sorted(blind["Y"])))

# [FIX] the irreducible-core minimum crosses at the threshold
diff_CD = sp.factor(sp.simplify(cf.LAMBDA["C"] - cf.LAMBDA["D"]))
check("FIX", "lambda_C - lambda_D carries the threshold factor (2F-1)",
      sp.simplify(diff_CD / (2 * F0 - 1)).is_rational_function(F0)
      and sp.simplify((cf.LAMBDA["C"] - cf.LAMBDA["D"]).subs(F0, sp.Rational(1, 2))) == 0)
for f0, expect in ((0.4, "C"), (0.7, "D")):
    lo = cf.alphabet_numeric(f0)
    smaller = "C" if lo["C"] < lo["D"] else "D"
    check("FIX", f"minimum letter for mixed irreducible gens at F_0={f0} is lambda_{expect}",
          smaller == expect, f"got lambda_{smaller}")

# --------------------------------------------------------------------------
section("Sec. 9.8  Theorem 10: two-sided inertia")
for f0, want in ((0.4, False), (0.7, True)):
    a = atoms["q[Z|X]"]
    ev = np.linalg.eigvalsh(hessian_from_rho(rho_from_form(a["S"], f0), "Z"))
    check("THM", f"purifying atom carrying lambda_E: PSD is {want} at F_0={f0}",
          bool(ev.min() > -1e-9) == want, f"lambda_min {ev.min():+.5f}")
for f0, want in ((0.4, True), (0.7, False)):
    a = atoms["q[X|Z]"]
    ev = np.linalg.eigvalsh(hessian_from_rho(rho_from_form(a["S"], f0), "Z"))
    check("THM", f"fixed-point atom carrying mu_C: PSD is {want} at F_0={f0}",
          bool(ev.min() > -1e-9) == want, f"lambda_min {ev.min():+.5f}")
for f0 in (0.3, 0.45, 0.6, 0.9):
    a = atoms["q[Y|Y]"]
    ev = np.linalg.eigvalsh(hessian_from_rho(rho_from_form(a["S"], f0), "Z"))
    check("THM", f"q[Y|Y] is PSD at F_0={f0} (escapes the threshold)",
          ev.min() > -1e-9, f"lambda_min {ev.min():+.5f}")

# --------------------------------------------------------------------------
section("Sec. 13  Regime of validity of the quadratic prediction")
rng = np.random.default_rng(20260805)
for name, (c, chk) in FIXED_FAMILY.items():
    m = hessian_exact(c, F, chk, sub)
    f_out0 = accepted_fidelity(c, F, chk)
    pred, act = [], []
    for _ in range(60):
        th = rng.uniform(-0.05, 0.05, size=3)
        angles = dict(zip(sub, th.tolist()))
        pred.append(float(th @ m @ th))
        act.append(f_out0 - accepted_fidelity(c, F, chk, angles))
    pred, act = np.array(pred), np.array(act)
    r2 = 1 - np.sum((act - pred) ** 2) / np.sum((act - act.mean()) ** 2)
    check("VER", f"{name}: R^2 > 0.999 for |theta| < 0.05", r2 > 0.999,
          f"R^2 = {r2:.6f}")

# ==========================================================================
print("\n" + "=" * 78)
if FAILURES:
    print(f"FAILED: {len(FAILURES)} of {COUNT} checks")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"All {COUNT} checks passed.")
