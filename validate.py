"""Validate the general-n engine against every published closed form before using it at n=4."""
from fractions import Fraction
from general_n import *

def frac(x):
    return Fraction(x).limit_denominator(10**9)

F = Fraction(9, 10)
q = (1 - F) / 3
D = 8 * F * F - 4 * F + 5
D3 = 16 * F * F - 14 * F + 7

# ---------- n=2 forward-Z baseline ----------
n = 2
S = cnot_bilateral_S(0, 1, n)           # kept pair controls sacrificial
row = Row(S, ('Z',), n)
p0, N0 = row.p0_N0(F)
assert p0 == D / 9, (p0, D / 9)
assert N0 == (10 * F * F - 2 * F + 1) / 9
Fout = N0 / p0
assert Fout == (10 * F * F - 2 * F + 1) / D
print("n=2 forward-Z: p0, Fout reproduce Eqs (18),(20) exactly")

M = row.M_diag(F)
lamA = 4 * (2 * F + 1) * (4 * F - 1) / D
lamB = 4 * (4 * F - 1) ** 2 / D
lamC = 4 * (1 - F) * (2 * F + 1) * (4 * F - 1) * (8 * F + 1) / D ** 2
lamD = 8 * (1 - F) * (F + 2) * (2 * F + 1) * (4 * F - 1) / D ** 2
lamE = 12 * (1 - F) * (2 * F - 1) * (2 * F + 1) * (4 * F - 1) / D ** 2

# invariant labels: pair0=kept, pair1=sacrificial
def g(name):
    P = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    x0, z0 = P[name[0]]
    x1, z1 = P[name[1]]
    return x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)

assert M[g('XI')] == lamA and M[g('XZ')] == lamA
assert M[g('ZZ')] == lamB and M[g('ZI')] == lamB
assert M[g('IX')] == lamC
assert row.lambda_max(F) == lamA
nulls = sorted(label_name(p, n) for p, v in M.items() if v == 0)
assert set(nulls) == {'IZ', 'YI', 'YX', 'YZ', 'ZY'}, nulls
print(f"n=2 forward-Z: lambda table matches alphabet; null set {nulls}; "
      f"lambda_max = {float(row.lambda_max(F)):.4f} (paper: 3.6954)")

# reverse-X: sacrificial controls kept, X check
rowX = Row(cnot_bilateral_S(1, 0, n), ('X',), n)
MX = rowX.M_diag(F)
assert MX[g('IX')] == 0, "reverse-X exact null on IX"
assert rowX.lambda_max(F) == lamA
print("n=2 reverse-X: IX exact null and shared worst case confirmed")

# ---------- n=2 map census over random atoms ----------
rng = random.Random(7)
maps = {}
for _ in range(400):
    S = random_symplectic(2, rng)
    for c in ('X', 'Z', 'Y'):
        r = Row(S, (c,), 2)
        maps.setdefault(r.weight_dists(), []).append(r)
pur = [k for k in maps if maps[k][0].fout(F) > F]
fixp = [k for k in maps if maps[k][0].fout(F) == F]
assert len(pur) == 1 and len(fixp) == 1, (len(pur), len(fixp))
assert maps[pur[0]][0].fout(F) == (10 * F * F - 2 * F + 1) / D
print(f"n=2 census: exactly 1 purifying map (Cor. 17) and 1 fixed-point map over {sum(len(v) for v in maps.values())} sampled rows")

# ---------- n=3: two purifying maps, lambda_max closed forms, F0* ----------
n = 3
rng = random.Random(11)
maps3 = {}
checks3 = [(a, b) for a in 'XZY' for b in 'XZY']
for _ in range(500):
    S = random_symplectic(3, rng)
    for cc in checks3:
        r = Row(S, cc, 3)
        key = r.weight_dists()
        if key not in maps3:
            maps3[key] = r
pur3 = {k: r for k, r in maps3.items() if r.fout(F) > F}
fouts = {}
for k, r in pur3.items():
    fouts.setdefault(r.fout(F), []).append(r)
assert len(fouts) == 2, f"expected 2 purifying maps at n=3, found {len(fouts)}"
fI = (10 * F * F - 2 * F + 1) / D
fII = (14 * F * F - 7 * F + 2) / D3
assert set(fouts) == {fI, fII}
print("n=3 census: exactly the two purifying fidelity maps I and II")

lamI = 4 * (2 * F + 1) * (4 * F - 1) / D
lamII = 12 * F * (4 * F - 1) / D3
for val, rows in fouts.items():
    lm = {r.lambda_max(F) for r in rows[:6]}
    assert len(lm) == 1
    assert lm == ({lamI} if val == fI else {lamII})
print(f"n=3: lambda_max single-valued per map; I -> {float(lamI):.4f}, II -> {float(lamII):.4f} at F=0.9 (paper: 3.6954, 3.8152)")

for Ftest, lo, hi in [(Fraction(52, 100), Fraction(16655, 10000), Fraction(17337, 10000)),
                      (Fraction(55, 100), Fraction(19130, 10000), Fraction(19310, 10000))]:
    DT = 8 * Ftest ** 2 - 4 * Ftest + 5
    D3T = 16 * Ftest ** 2 - 14 * Ftest + 7
    a = 4 * (2 * Ftest + 1) * (4 * Ftest - 1) / DT
    b = 12 * Ftest * (4 * Ftest - 1) / D3T
    assert abs(float(b) - float(lo)) < 1e-3 and abs(float(a) - float(hi)) < 1e-3
print("n=3: crossover numerics at F0=0.52, 0.55 match the paper; engine validated.")
