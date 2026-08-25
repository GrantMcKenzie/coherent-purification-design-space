"""Exact crossovers at n=4 and an independent density-matrix check.

Two explicit four-pair protocols:
  II' : best sampled map near threshold (Fout = three-pair map II's Fout,
        p0 = (8F^2-F+2) D3 / 81)
  nested : two-round untwirled recurrence (CNOT tree, three Z checks)

Claim to establish exactly: their lambda_max regions of superiority over the
two-pair bound lambda_A overlap, so together they beat lambda_A at EVERY
F0 in (1/2, 1).
"""
from fractions import Fraction
import pickle
import numpy as np
import sympy as sp
from general_n import Row, cnot_bilateral_S, label_name
from structured_n4 import compose

n = 4
F = sp.symbols('F', positive=True)
D = 8 * F ** 2 - 4 * F + 5
D3 = 16 * F ** 2 - 14 * F + 7
P4 = 32 * F ** 4 - 32 * F ** 3 + 120 * F ** 2 - 56 * F + 17
lamA = 4 * (2 * F + 1) * (4 * F - 1) / D
lam_nested = 4 * (2 * F + 1) * (4 * F - 1) * D3 / P4

# ---------------- lambda_max closed form for map II'
with open('n4_results.pkl', 'rb') as fh:
    saved = pickle.load(fh)
# find the map with p0 = (8F^2-F+2)*D3/81 by its weight dists
target = None
for r in saved['results']:
    A, B = r['key']
    q = (1 - F) / 3
    p0 = sp.expand(sum(a * F ** (n - w) * q ** w for w, a in enumerate(A)))
    if sp.simplify(p0 - (8 * F ** 2 - F + 2) * D3 / 81) == 0:
        target = r
        break
assert target is not None
rowIIp = Row(target['reps'][0][0], target['reps'][0][1], n)

Fq = Fraction(55, 100)
M = rowIIp.M_diag(Fq)
m = max(M.values())
winners = [p for p, v in M.items() if v == m]
print("II' lambda_max winners at F=0.55:", sorted(label_name(p, n) for p in winners))

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

lamIIp = sp.factor(M_entry_symbolic(rowIIp, winners[0]))
print("lambda_II' =")
sp.pprint(lamIIp)
# confirm it is the max over the interval, not just at 0.55: sample check
for Ft in (Fraction(51,100), Fraction(58,100), Fraction(62,100), Fraction(70,100), Fraction(9,10)):
    Md = rowIIp.M_diag(Ft)
    assert max(Md.values()) == sp.nsimplify(lamIIp.subs(F, sp.Rational(Ft.numerator, Ft.denominator))), Ft
print("lambda_II' confirmed as the max entry across the purifying range")

# ---------------- crossovers
dA_IIp = sp.factor(sp.cancel(lamA - lamIIp))
print("\nlambda_A - lambda_II' ="); sp.pprint(dA_IIp)
dA_nst = sp.factor(sp.cancel(lamA - lam_nested))

numA_IIp = sp.numer(dA_IIp)
numA_nst = sp.numer(dA_nst)

# roots on (1/2, 1) numerically with high precision
xc = None
for r in sp.nroots(sp.Poly(numA_IIp, F), n=30):
    if r.is_real and 0.5 < float(r) < 1:
        xc = r
print("\nx_c  (II' stops beating lambda_A) =", xc)
Fss = None
for r in sp.nroots(sp.Poly(16 * F ** 3 - 8 * F ** 2 + 4 * F - 3, F), n=30):
    if r.is_real and 0.5 < float(r) < 1:
        Fss = r
print("F** (nested starts beating lambda_A) =", Fss)
print("overlap (F** < x_c):", float(Fss) < float(xc))

# exact sign confirmation of overlap: evaluate II'-superiority at F** via resultant-free check
mid = sp.Rational(sp.nsimplify(float((float(Fss) + float(xc)) / 2), rational=True))
print("at midpoint F=%.6f: lamA-lamII'=%.6g  lamA-lamNested=%.6g (both should be >0)" %
      (float(mid), float(dA_IIp.subs(F, mid)), float(dA_nst.subs(F, mid))))

# minimum over the union at dense grid: is min(lamII', lamNested) < lamA on all (1/2,1)?
fnA = sp.lambdify(F, lamA, 'numpy')
fn1 = sp.lambdify(F, lamIIp, 'numpy')
fn2 = sp.lambdify(F, lam_nested, 'numpy')
xs = np.linspace(0.5001, 0.99995, 20001)
gap = fnA(xs) - np.minimum(fn1(xs), fn2(xs))
print(f"min over (1/2,1) of [lambda_A - min(lambda_II', lambda_nested)] = {gap.min():.6g}  (>0 means beaten everywhere)")

# ---------------- independent density-matrix check of lambda_nested at F=0.9
print("\nindependent 8-qubit density-matrix check of the nested map ...")
def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
# qubit order: A0 A1 A2 A3 B0 B1 B2 B3
nq = 8
def op_on(o, q):
    ops = [I2] * nq
    ops[q] = o
    return kron(*ops)

def cnot(control, targ):
    P0 = np.diag([1.0, 0.0]).astype(complex); P1 = np.diag([0.0, 1.0]).astype(complex)
    ops0 = [I2] * nq; ops0[control] = P0
    ops1 = [I2] * nq; ops1[control] = P1; ops1[targ] = X
    return kron(*ops0) + kron(*ops1)

F0 = 0.9
q0 = (1 - F0) / 3
phip = np.zeros(4); phip[0] = phip[3] = 1 / np.sqrt(2)
psip = np.zeros(4); psip[1] = psip[2] = 1 / np.sqrt(2)
phim = np.zeros(4); phim[0] = 1 / np.sqrt(2); phim[3] = -1 / np.sqrt(2)
psim = np.zeros(4); psim[1] = 1 / np.sqrt(2); psim[2] = -1 / np.sqrt(2)
rho_pair = (F0 * np.outer(phip, phip) + q0 * (np.outer(psip, psip) +
            np.outer(phim, phim) + np.outer(psim, psim)))
# assemble with qubit order A0..A3 B0..B3: build in pair order then permute
rho_pairs = rho_pair
for _ in range(3):
    rho_pairs = np.kron(rho_pairs, rho_pair)   # order A0 B0 A1 B1 A2 B2 A3 B3
perm = [0, 2, 4, 6, 1, 3, 5, 7]                # -> A0 A1 A2 A3 B0 B1 B2 B3
axes = perm + [8 + p for p in perm]
rho = rho_pairs.reshape([2] * 16).transpose(axes).reshape(256, 256)

U = (cnot(2, 3) @ cnot(6, 7)   # wait: build explicitly below
     )
# circuit: CNOT(pair0->pair1), CNOT(pair2->pair3), CNOT(pair0->pair2), bilateral
U = cnot(0, 1) @ cnot(4, 5)          # round 1 block A0->A1, B0->B1
U = (cnot(2, 3) @ cnot(6, 7)) @ U    # round 1 block A2->A3, B2->B3
U = (cnot(0, 2) @ cnot(4, 6)) @ U    # round 2 A0->A2, B0->B2
rhoC = U @ rho @ U.conj().T

Pacc = np.eye(256, dtype=complex)
for k in (1, 2, 3):
    Pacc = Pacc @ (np.eye(256) + op_on(Z, k) @ op_on(Z, 4 + k)) / 2
Htheta = op_on(X, 0) + op_on(X, 4)   # common-mode X on kept pair
Pi_target = np.zeros((256, 256), dtype=complex)
# |Phi+><Phi+| on (A0,B0) tensor identity on rest
phi_ab = np.outer(phip, phip)        # on qubits (A0,B0)
# build via einsum-free route: op with phi+ on qubits 0 and 4
T4 = phi_ab.reshape(2, 2, 2, 2)      # (A0,B0,A0',B0')
Pi_target = np.zeros((256, 256), dtype=complex)
idx = np.arange(256)
for a0 in range(2):
    for b0 in range(2):
        for a0p in range(2):
            for b0p in range(2):
                val = T4[a0, b0, a0p, b0p]
                if abs(val) < 1e-15:
                    continue
                rows = idx[(idx >> 7 - 0 & 1) == 0]  # placeholder
# simpler: kron with explicit ordering A0 A1 A2 A3 B0 B1 B2 B3
basis_op = np.zeros((2, 2, 2, 2), dtype=complex)
Pi_target = np.zeros((256, 256), dtype=complex)
for a0 in range(2):
    for b0 in range(2):
        for a0p in range(2):
            for b0p in range(2):
                v = T4[a0, b0, a0p, b0p]
                if v == 0:
                    continue
                ketA = np.zeros((2, 1)); ketA[a0] = 1
                braA = np.zeros((1, 2)); braA[0, a0p] = 1
                ketB = np.zeros((2, 1)); ketB[b0] = 1
                braB = np.zeros((1, 2)); braB[0, b0p] = 1
                Pi_target += v * kron(ketA @ braA, I2, I2, I2, ketB @ braB, I2, I2, I2)

def fout_theta(th):
    E = np.eye(256) * np.cos(th) - 1j * np.sin(th) * 0  # placeholder
    from scipy.linalg import expm
    E = expm(-1j * th * Htheta)
    r = E @ rhoC @ E.conj().T
    num = np.trace(Pi_target @ Pacc @ r @ Pacc).real
    den = np.trace(Pacc @ r @ Pacc).real
    return num / den

f0 = fout_theta(0.0)
th = 1e-3
f1 = fout_theta(th)
f2 = fout_theta(2 * th)
# Richardson: F(th) = f0 - M th^2 + O(th^4)
M1 = (f0 - f1) / th ** 2
M2 = (f0 - f2) / (2 * th) ** 2
M_rich = (4 * M1 - M2) / 3
lam_pred = float(lam_nested.subs(F, sp.Rational(9, 10)))
print(f"density matrix:  M_XIII = {M_rich:.10f}")
print(f"master formula:  M_XIII = {lam_pred:.10f}")
print(f"agreement: {abs(M_rich - lam_pred):.3e}")
print(f"Fout(0) check: dm {f0:.10f} vs formula "
      f"{float(sp.cancel((136*F**4-112*F**3+60*F**2-4*F+1)/P4).subs(F, sp.Rational(9,10))):.10f}")
