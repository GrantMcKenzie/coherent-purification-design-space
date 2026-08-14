#!/usr/bin/env python3
"""Assertions for the master formula and the three-pair design space.

Complements verify_all.py, which covers the two-pair density-matrix and
group-theoretic engines. This script exercises the third engine — direct
evaluation of the master formula — and every three-pair claim in the paper.

Run with ``python verify_threepair.py``. Exits nonzero if any claim fails.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction as Fr

import numpy as np

from finite_rank_purification.circuits import FIXED_FAMILY
from finite_rank_purification.classification import _inverse_mod2
from finite_rank_purification.clifford import symplectic_image
from finite_rank_purification.susceptibility import hessian_exact
from master import accepted, diagonal, from_name, to_name, werner_pops
from threepair import evaluate, frames, nondeg_2subspaces, pops_from_frame

FAILURES: list[str] = []
COUNT = 0


def check(tag, claim, ok, detail=""):
    global COUNT
    COUNT += 1
    line = f"  [{tag}] {claim:<58s} {'PASS' if ok else 'FAIL'}"
    if detail:
        line += f"   {detail}"
    print(line)
    if not ok:
        FAILURES.append(claim)


def section(t):
    print(f"\n{t}\n{'-' * len(t)}")


print("Master formula and three-pair verification")
print("=" * 74)

# ----------------------------------------------------------------------
section("Sec. 3  Master formula against the density-matrix engine")
F = Fr(9, 10)
worst = 0.0
for name, (c, chk) in FIXED_FAMILY.items():
    Sinv = _inverse_mod2(symplectic_image(c))
    pops = werner_pops(2, F, [list(r) for r in Sinv])
    acc = accepted(2, [from_name("I" + chk)])
    d, p0, Fo = diagonal(pops, acc, None, 2)
    got = {to_name(k): float(v) for k, v in d.items()}
    labels = ("XI", "XZ", "ZZ", "IX", "YY", "ZI", "ZX")
    ref = np.diag(hessian_exact(c, 0.9, chk, labels))
    err = max(abs(got[g] - r) for g, r in zip(labels, ref))
    worst = max(worst, err)
    check("THM", f"{name}: master formula = density-matrix engine", err < 1e-12,
          f"max diff {err:.1e}")
check("THM", "third engine agrees to better than 1e-12", worst < 1e-12,
      f"{worst:.1e}")

# ----------------------------------------------------------------------
section("Sec. 5  Fixed family in invariant labels")
EXPECT = {
    "forward-Z": ("A", "A", "B", "C"),
    "reverse-X": ("B", "D", "C", "0"),
    "reverse-Y": ("B", "D", "C", "D"),
}
f0 = 0.9
D = 8 * f0**2 - 4 * f0 + 5
LET = {
    "A": 4 * (2 * f0 + 1) * (4 * f0 - 1) / D,
    "B": 4 * (4 * f0 - 1) ** 2 / D,
    "C": 4 * (1 - f0) * (2 * f0 + 1) * (4 * f0 - 1) * (8 * f0 + 1) / D**2,
    "D": 8 * (1 - f0) * (f0 + 2) * (2 * f0 + 1) * (4 * f0 - 1) / D**2,
    "0": 0.0,
}
for name, (c, chk) in FIXED_FAMILY.items():
    d = np.diag(hessian_exact(c, f0, chk, ("XI", "XZ", "ZZ", "IX")))
    want = [LET[k] for k in EXPECT[name]]
    ok = all(abs(a - b) < 1e-9 for a, b in zip(d, want))
    check("THM", f"{name}: (XI,XZ,ZZ,IX) = {EXPECT[name]}", ok,
          "(" + ", ".join(f"{x:.4f}" for x in d) + ")")

# every circuit attains lambda_A somewhere over all fifteen
from finite_rank_purification.paulis import TWO_QUBIT_GENERATORS
for name, (c, chk) in FIXED_FAMILY.items():
    d = np.diag(hessian_exact(c, f0, chk))
    at = {g for g, v in zip(TWO_QUBIT_GENERATORS, d) if abs(v - LET["A"]) < 1e-9}
    check("THM", f"{name}: attains lambda_A over all 15 generators", len(at) == 2,
          str(sorted(at)))

# ----------------------------------------------------------------------
section("Sec. 10  Pair-frame counts")
for n, nsub, nfr in ((2, 20, 10), (3, 336, 1120)):
    subs = nondeg_2subspaces(n)
    fr = frames(n)
    check("THM", f"n={n}: {nsub} non-degenerate 2-subspaces", len(subs) == nsub,
          f"got {len(subs)}")
    check("THM", f"n={n}: N_atoms = {nfr}", len(fr) == nfr, f"got {len(fr)}")

FR3 = frames(3)
CHECKS = [(a, b) for a in "XYZ" for b in "XYZ"]

# ----------------------------------------------------------------------
section("Sec. 11  Three-pair fidelity maps and check-independence")
t = time.time()
D3 = 16 * F**2 - 14 * F + 7
Dq = 8 * F**2 - 4 * F + 5
MAP_I = ((2 * F + 1) * Dq / 27, (10 * F**2 - 2 * F + 1) / Dq)
MAP_II = ((2 * F + 1) * D3 / 27, (14 * F**2 - 7 * F + 2) / D3)

maps = Counter()
percheck = Counter()
byatom = defaultdict(int)
for i, f in enumerate(FR3):
    pops = pops_from_frame(f, F)
    for (c1, c2) in CHECKS:
        E = [from_name("I" + c1 + "I"), from_name("II" + c2)]
        acc = [v for v in pops if all(__import__("master").symp(e, v) == 0 for e in E)]
        p0 = sum(pops[v] for v in acc)
        Fo = sum(pops[v] for v in acc if v[0] == 0 and v[1] == 0) / p0
        maps[(p0, Fo)] += 1
        if Fo != F:
            percheck[(c1, c2)] += 1
            byatom[i] += 1
pur = {k: v for k, v in maps.items() if k[1] != F}
check("THM", "exactly two purifying (p0, Fout) maps", len(pur) == 2, str(len(pur)))
check("THM", "map I matches (2F+1)D/27, (10F^2-2F+1)/D", MAP_I in pur,
      f"{pur.get(MAP_I)} rows")
check("THM", "map II matches (2F+1)D3/27, (14F^2-7F+2)/D3", MAP_II in pur,
      f"{pur.get(MAP_II)} rows")
check("THM", "7776 purifying rows of 10080", sum(pur.values()) == 7776,
      str(sum(pur.values())))
check("VER", "every check pair purifies exactly 864 atoms",
      set(percheck.values()) == {864}, str(sorted(set(percheck.values()))))
dist = Counter(byatom.get(i, 0) for i in range(len(FR3)))
check("VER", "only 228 atoms purify under all nine checks", dist[9] == 228,
      str(dist[9]))
check("VER", "10 atoms purify under no check", dist[0] == 10, str(dist[0]))
check("VER", "remainder split 135/216/72/459 over 4/5/6/8 checks",
      (dist[4], dist[5], dist[6], dist[8]) == (135, 216, 72, 459),
      str((dist[4], dist[5], dist[6], dist[8])))
print(f"    ({time.time() - t:.0f}s)")

# ----------------------------------------------------------------------
section("Sec. 11  Theorem: the two-pair bound is beaten below F0*")
from master import add, in_target, labels, symp, yparity

GENS3 = [p for p in labels(3) if any(p)]
ALL3 = labels(3)


def lam_max_by_map(Fv):
    d2 = 8 * Fv**2 - 4 * Fv + 5
    d3 = 16 * Fv**2 - 14 * Fv + 7
    mI = ((2 * Fv + 1) * d2 / 27, (10 * Fv**2 - 2 * Fv + 1) / d2)
    mII = ((2 * Fv + 1) * d3 / 27, (14 * Fv**2 - 7 * Fv + 2) / d3)
    out = defaultdict(set)
    for f in FR3:
        pops = pops_from_frame(f, Fv)
        for (c1, c2) in CHECKS:
            E = [from_name("I" + c1 + "I"), from_name("II" + c2)]
            acc = [v for v in ALL3 if all(symp(e, v) == 0 for e in E)]
            p0 = sum(pops[v] for v in acc)
            Fo = sum(pops[v] for v in acc if in_target(v)) / p0
            if Fo == Fv:
                continue
            mx = None
            for p in GENS3:
                yp = yparity(p)
                tot = 0
                for v in acc:
                    if symp(p, v) != yp:
                        continue
                    tot += (pops[add(v, p)] - pops[v]) * ((1 if in_target(v) else 0) - Fo)
                val = -4 * tot / p0
                if mx is None or val > mx:
                    mx = val
            key = "I" if (p0, Fo) == mI else ("II" if (p0, Fo) == mII else "?")
            out[key].add(mx)
    return out


for Fv, lower in ((Fr(52, 100), "II"), (Fr(55, 100), "II"), (Fr(9, 10), "I")):
    t = time.time()
    out = lam_max_by_map(Fv)
    d2 = 8 * Fv**2 - 4 * Fv + 5
    d3 = 16 * Fv**2 - 14 * Fv + 7
    lI = 4 * (2 * Fv + 1) * (4 * Fv - 1) / d2
    lII = 12 * Fv * (4 * Fv - 1) / d3
    check("THM", f"F0={float(Fv)}: lambda_max single-valued on each map",
          len(out["I"]) == 1 and len(out["II"]) == 1)
    check("THM", f"F0={float(Fv)}: closed forms match enumeration",
          out["I"] == {lI} and out["II"] == {lII},
          f"I={float(lI):.4f} II={float(lII):.4f}")
    got = "II" if lII < lI else "I"
    check("THM", f"F0={float(Fv)}: map {lower} is lower", got == lower,
          f"({time.time() - t:.0f}s)")

star = (3 * np.sqrt(2) - 2) / 4
check("THM", "F0* = (3*sqrt(2)-2)/4", abs(star - 0.5606601717798213) < 1e-12,
      f"{star:.12f}")
check("THM", "8F^2+8F-7 vanishes at F0*", abs(8 * star**2 + 8 * star - 7) < 1e-12)

# ----------------------------------------------------------------------
section("Sec. 14  Matched three-pair protocols")
YYZ = from_name("YYZ")
rows = []
for i, f in enumerate(FR3):
    pops = pops_from_frame(f, F)
    for (c1, c2) in CHECKS:
        E = [from_name("I" + c1 + "I"), from_name("II" + c2)]
        acc = [v for v in ALL3 if all(symp(e, v) == 0 for e in E)]
        p0 = sum(pops[v] for v in acc)
        Fo = sum(pops[v] for v in acc if in_target(v)) / p0
        if (p0, Fo) != MAP_II:
            continue
        tr = 0
        yyz = None
        for p in GENS3:
            yp = yparity(p)
            tot = 0
            for v in acc:
                if symp(p, v) != yp:
                    continue
                tot += (pops[add(v, p)] - pops[v]) * ((1 if in_target(v) else 0) - Fo)
            val = -4 * tot / p0
            tr += val
            if p == YYZ:
                yyz = val
        rows.append((i, (c1, c2), float(tr), float(yyz)))

check("THM", "map II: p0 = 0.763259, Fout = 0.956522",
      abs(float(MAP_II[0]) - 0.763259) < 1e-6 and abs(float(MAP_II[1]) - 0.956522) < 1e-6)
A = [r for r in rows if abs(r[2] - 29.744) < 0.01 and abs(r[3] - 3.8152) < 0.01]
B = [r for r in rows if abs(r[2] - 37.485) < 0.01 and abs(r[3] - 0.0307) < 0.01]
check("VER", "protocol A exists (Tr M 29.744, M_YYZ 3.8152)", len(A) > 0, f"{len(A)} rows")
check("VER", "protocol B exists (Tr M 37.485, M_YYZ 0.0307)", len(B) > 0, f"{len(B)} rows")
if A and B:
    check("VER", "isotropic prefers A, YYZ-biased prefers B",
          A[0][2] < B[0][2] and B[0][3] < A[0][3])
    check("VER", "YYZ ratio is about 124x",
          abs(A[0][3] / B[0][3] - 124) < 3, f"{A[0][3] / B[0][3]:.1f}x")

# ----------------------------------------------------------------------
print("\n" + "=" * 74)
if FAILURES:
    print(f"FAILED: {len(FAILURES)} of {COUNT} checks")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print(f"All {COUNT} checks passed.")
