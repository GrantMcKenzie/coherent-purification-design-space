"""Structured n=4 protocols the random sampler may miss: nested (iterated)
recurrence, CNOT chains, stars -- exact fidelity maps and lambda_max."""
from fractions import Fraction
import numpy as np
import sympy as sp
from general_n import Row, cnot_bilateral_S, inv_S, label_name

n = 4
Fsym = sp.symbols('F', positive=True)

def compose(*mats):
    """apply left-to-right in time order: compose(A, B) = B @ A."""
    out = np.eye(2 * n, dtype=np.uint8)
    for M in mats:
        out = (M @ out) & 1
    return out

def fout_sym(row):
    A, B = row.weight_dists()
    q = (1 - Fsym) / 3
    p0 = sum(a * Fsym ** (n - w) * q ** w for w, a in enumerate(A))
    N0 = sum(b * Fsym ** (n - w) * q ** w for w, b in enumerate(B))
    return sp.factor(sp.expand(p0)), sp.cancel(N0 / p0)

R = (10 * Fsym ** 2 - 2 * Fsym + 1) / (8 * Fsym ** 2 - 4 * Fsym + 5)
RR = sp.cancel(R.subs(Fsym, R))

circuits = {
    # two-round iterated recurrence, deferred: round1 (0->1),(2->3); round2 (0->2)
    'nested (R o R)': (compose(cnot_bilateral_S(0, 1, n), cnot_bilateral_S(2, 3, n),
                               cnot_bilateral_S(0, 2, n)), ('Z', 'Z', 'Z')),
    'star 0->1,0->2,0->3': (compose(cnot_bilateral_S(0, 1, n), cnot_bilateral_S(0, 2, n),
                                    cnot_bilateral_S(0, 3, n)), ('Z', 'Z', 'Z')),
    'chain 0->1->2->3': (compose(cnot_bilateral_S(0, 1, n), cnot_bilateral_S(1, 2, n),
                                 cnot_bilateral_S(2, 3, n)), ('Z', 'Z', 'Z')),
}

if __name__ == '__main__':
    F9 = Fraction(9, 10)
    for name, (S, checks) in circuits.items():
        row = Row(S, checks, n)
        p0f, fo = fout_sym(row)
        is_RR = sp.simplify(fo - RR) == 0
        lm = row.lambda_max(F9)
        print(f"{name}:")
        print(f"   p0   = {p0f}")
        print(f"   Fout = {fo}")
        print(f"   equals R(R(F)): {is_RR};  lambda_max(0.9) = {float(lm):.4f}")
        print(f"   weight dists A,B = {row.weight_dists()}")
        print()
