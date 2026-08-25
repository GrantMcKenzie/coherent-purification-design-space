"""Exact analysis of the nested (two-round, untwirled) n=4 recurrence map:
closed-form lambda_max, where it sits against lambda_A (two-pair floor) and
lambda_II (three-pair map II), fixed points, and the corrected n=4 frontier."""
from fractions import Fraction
import pickle
import numpy as np
import sympy as sp
from general_n import Row, cnot_bilateral_S, label_name
from structured_n4 import compose, fout_sym

n = 4
F = sp.symbols('F', positive=True)

S = compose(cnot_bilateral_S(0, 1, n), cnot_bilateral_S(2, 3, n), cnot_bilateral_S(0, 2, n))
row = Row(S, ('Z', 'Z', 'Z'), n)

# ---- purifying range / fixed points
p0f, fo = fout_sym(row)
fp = sp.factor(sp.numer(sp.cancel(fo - F)))
print("Fout - F numerator factors:", fp)
print("real roots in (0,1):", [sp.nsimplify(r) for r in sp.solve(sp.Eq(fp, 0), F) if r.is_real and 0 < r < 1])

# ---- identify argmax generator across the purifying range (exact Fractions)
grid = [Fraction(x, 100) for x in (51, 55, 60, 70, 80, 90, 95, 99)]
argmax = {}
for Fq in grid:
    M = row.M_diag(Fq)
    m = max(M.values())
    winners = tuple(sorted(label_name(p, n) for p, v in M.items() if v == m))
    argmax[Fq] = (m, winners)
    print(f"F={float(Fq):.2f}  lambda_max={float(m):.6f}  attained by {winners}")

# ---- closed form of the winning entry (symbolic master formula on one generator)
def M_entry_symbolic(row, p):
    from general_n import symp, y_parity, n_labels
    q = (1 - F) / 3
    rho = [F ** (n - int(w)) * q ** int(w) for w in row.w_of_v]
    A, B = row.weight_dists()
    p0 = sum(a * F ** (n - w) * q ** w for w, a in enumerate(A))
    N0 = sum(b * F ** (n - w) * q ** w for w, b in enumerate(B))
    Fout = N0 / p0
    y = y_parity(p, n)
    tot = 0
    for v in range(n_labels(n)):
        if row.acc[v] and symp(p, v, n) == y:
            tot += (rho[v ^ p] - rho[v]) * ((1 if row.tgt[v] else 0) - Fout)
    return sp.cancel(-4 * tot / p0)

# pick one winner at F=0.9 and one at F=0.55 (they may differ)
from general_n import label_name as ln
def label_from_name(name):
    P = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    v = 0
    for k, ch in enumerate(name):
        x, z = P[ch]
        v |= (x << (2 * k)) | (z << (2 * k + 1))
    return v

for Fq in (Fraction(55, 100), Fraction(90, 100)):
    name = argmax[Fq][1][0]
    expr = sp.factor(M_entry_symbolic(row, label_from_name(name)))
    print(f"\nclosed form of M[{name}] (a lambda_max winner at F={float(Fq):.2f}):")
    sp.pprint(expr)

# ---- compare against lambda_A and lambda_II over (1/2, 1)
D = 8 * F ** 2 - 4 * F + 5
D3 = 16 * F ** 2 - 14 * F + 7
lamA = 4 * (2 * F + 1) * (4 * F - 1) / D
lamII = 12 * F * (4 * F - 1) / D3

name9 = argmax[Fraction(90, 100)][1][0]
lam_nested = sp.factor(M_entry_symbolic(row, label_from_name(name9)))
diffA = sp.factor(sp.cancel(lamA - lam_nested))
print("\nlambda_A - lambda_nested =")
sp.pprint(diffA)
# robust numeric root check on (1/2, 1)
import numpy as _np
fn = sp.lambdify(F, diffA, 'numpy')
xs = _np.linspace(0.5001, 0.9999, 4001)
vals = fn(xs)
sign_changes = _np.where(_np.diff(_np.sign(vals)) != 0)[0]
print(f"sign of (lambda_A - lambda_nested) on (1/2,1): min={vals.min():.4g}, max={vals.max():.4g}, "
      f"sign changes: {len(sign_changes)}")

diffII = sp.factor(sp.cancel(lamII - lam_nested))
fn2 = sp.lambdify(F, diffII, 'numpy')
vals2 = fn2(xs)
sc2 = _np.where(_np.diff(_np.sign(vals2)) != 0)[0]
print(f"sign of (lambda_II - lambda_nested) on (1/2,1): min={vals2.min():.4g}, max={vals2.max():.4g}, "
      f"sign changes: {len(sc2)}")
if len(sc2):
    for i in sc2:
        print(f"   crossing near F = {xs[i]:.6f}")

# ---- corrected frontier including nested + sampled maps
with open('n4_results.pkl', 'rb') as fh:
    saved = pickle.load(fh)
print("\ncorrected n=4 frontier (sampled maps + nested):")
Fgrid = [Fraction(51, 100), Fraction(52, 100), Fraction(55, 100), Fraction(58, 100),
         Fraction(60, 100), Fraction(65, 100), Fraction(3, 4), Fraction(9, 10)]
lamAq = lambda Fq: 4 * (2 * Fq + 1) * (4 * Fq - 1) / (8 * Fq * Fq - 4 * Fq + 5)
lamIIq = lambda Fq: 12 * Fq * (4 * Fq - 1) / (16 * Fq * Fq - 14 * Fq + 7)
nested_lam = {Fq: row.lambda_max(Fq) for Fq in Fgrid}
# sampled minima recomputed on this grid from stored reps (first rep per map)
sampled_rows = [Row(r['reps'][0][0], r['reps'][0][1], n) for r in saved['results']]
print(f"{'F0':>6} {'lam_A':>8} {'lam_II':>8} {'min sampled':>12} {'nested':>8} {'overall min':>12}")
for Fq in Fgrid:
    ms = min(r.lambda_max(Fq) for r in sampled_rows)
    overall = min(ms, nested_lam[Fq])
    print(f"{float(Fq):6.2f} {float(lamAq(Fq)):8.4f} {float(lamIIq(Fq)):8.4f} "
          f"{float(ms):12.4f} {float(nested_lam[Fq]):8.4f} {float(overall):12.4f}")
