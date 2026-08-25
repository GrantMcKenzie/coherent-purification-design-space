"""
explore_n4.py -- what happens to the design space at n=4.

Questions (the manuscript's open questions, one level up):
  Q1  How many purifying fidelity maps? (exactly 1 at n=2, exactly 2 at n=3)
  Q2  Is lambda_max still single-valued per map?
  Q3  Does any n=4 map beat the n=3 improvement -- lower worst case, wider window?
  Q4  Is the two-pair value lambda_A still the floor at large F (open question 2)?
  Q5  Is iterated recurrence R(R(F)) among the n=4 maps?

Method: random sampling over Sp(8,2) (transvection products), all 27 check
triples per sample, exact weight-distribution fingerprints, master formula for
susceptibilities.  Sampling gives a lower bound on the map census and
counterexample-grade answers to Q2-Q5 (any map found is exactly computed).
"""
import random, pickle
from fractions import Fraction
import numpy as np
from general_n import (n_labels, weight, symp, y_parity, check_label,
                       random_symplectic, inv_S, Row, label_name,
                       cnot_bilateral_S)
import sympy as sp

n = 4
N = n_labels(n)          # 256
d = 2 * n

# ---------------------------------------------------------------- fast tables
BITS = np.array([[(v >> i) & 1 for i in range(d)] for v in range(N)], dtype=np.uint8)

def weights_of_matrix(Minv):
    """w(Minv v) for all v, vectorized."""
    U = (BITS @ Minv.T) & 1          # rows are bit-vectors of S^{-1} v
    w = np.zeros(N, dtype=np.int64)
    for k in range(n):
        w += (U[:, 2 * k] | U[:, 2 * k + 1]).astype(np.int64)
    return w

# accept masks for each (pauli, sacrificial pair)
ACC1 = {}
for k in range(1, n):
    for c in 'XZY':
        e = check_label(c, k)
        ACC1[(c, k)] = np.array([symp(e, v, n) == 0 for v in range(N)])
TGT = np.array([(v & 3) == 0 for v in range(N)])
CHECK_TRIPLES = [(a, b, c) for a in 'XZY' for b in 'XZY' for c in 'XZY']
ACC3 = {cc: ACC1[(cc[0], 1)] & ACC1[(cc[1], 2)] & ACC1[(cc[2], 3)] for cc in CHECK_TRIPLES}

# ---------------------------------------------------------------- 1. census
def census(n_samples=4000, seed=1, keep_reps=6):
    rng = random.Random(seed)
    maps = {}     # (A,B) -> dict(count, reps=[(S, checks)])
    for i in range(n_samples):
        S = random_symplectic(n, rng)
        Sinv = inv_S(S)
        w = weights_of_matrix(Sinv)
        for cc in CHECK_TRIPLES:
            acc = ACC3[cc]
            A = tuple(np.bincount(w[acc], minlength=n + 1).tolist())
            B = tuple(np.bincount(w[acc & TGT], minlength=n + 1).tolist())
            rec = maps.setdefault((A, B), {'count': 0, 'reps': []})
            rec['count'] += 1
            if len(rec['reps']) < keep_reps:
                rec['reps'].append((S.copy(), cc))
    return maps

def polys(A, B):
    F = sp.symbols('F', positive=True)
    q = (1 - F) / 3
    p0 = sp.expand(sum(a * F ** (n - w) * q ** w for w, a in enumerate(A)))
    N0 = sp.expand(sum(b * F ** (n - w) * q ** w for w, b in enumerate(B)))
    return sp.factor(p0), sp.cancel(N0 / p0)

def fout_frac(A, B, F):
    q = (1 - F) / 3
    p0 = sum(a * F ** (n - w) * q ** w for w, a in enumerate(A))
    N0 = sum(b * F ** (n - w) * q ** w for w, b in enumerate(B))
    return N0 / p0

if __name__ == '__main__':
    print("sampling Sp(8,2) ...")
    maps = census()
    total_rows = sum(m['count'] for m in maps.values())
    print(f"{total_rows} rows sampled -> {len(maps)} distinct (p0,Fout) fingerprints")

    Fs = [Fraction(3, 5), Fraction(3, 4), Fraction(9, 10)]
    purifying, fixed, other = {}, {}, {}
    for key, rec in maps.items():
        A, B = key
        vals = [fout_frac(A, B, F) for F in Fs]
        if all(v > F for v, F in zip(vals, Fs)):
            purifying[key] = rec
        elif all(v == F for v, F in zip(vals, Fs)):
            fixed[key] = rec
        else:
            other[key] = rec
    print(f"purifying maps: {len(purifying)}   fixed-point: {len(fixed)}   other: {len(other)}")

    # ---- Q5: iterated recurrence present?
    Fsym = sp.symbols('F', positive=True)
    R = (10 * Fsym ** 2 - 2 * Fsym + 1) / (8 * Fsym ** 2 - 4 * Fsym + 5)
    RR = sp.cancel(R.subs(Fsym, R))
    found_RR = None
    for key in purifying:
        _, fo = polys(*key)
        if sp.simplify(fo - RR) == 0:
            found_RR = key
            break
    print(f"iterated recurrence R(R(F)) present among sampled maps: {found_RR is not None}")

    # ---- Q2/Q3/Q4: lambda_max per purifying map
    lamI = lambda F: 4 * (2 * F + 1) * (4 * F - 1) / (8 * F * F - 4 * F + 5)
    lamII = lambda F: 12 * F * (4 * F - 1) / (16 * F * F - 14 * F + 7)
    Fgrid = [Fraction(51, 100), Fraction(52, 100), Fraction(55, 100),
             Fraction(56, 100), Fraction(58, 100), Fraction(60, 100),
             Fraction(65, 100), Fraction(3, 4), Fraction(9, 10)]
    results = []
    print("computing lambda_max per purifying map (representatives) ...")
    for j, (key, rec) in enumerate(purifying.items()):
        reps = [Row(S, cc, n) for S, cc in rec['reps'][:3]]
        per_rep = []
        for r in reps:
            per_rep.append([r.lambda_max(F) for F in Fgrid])
        single = all(all(per_rep[0][i] == pr[i] for i in range(len(Fgrid))) for pr in per_rep[1:])
        results.append({'key': key, 'count': rec['count'], 'single': single,
                        'lmax': per_rep[0], 'reps': rec['reps']})
    n_single = sum(r['single'] for r in results)
    print(f"lambda_max single-valued per map: {n_single}/{len(results)} maps "
          f"(on {min(3, 6)} sampled representatives each)")

    # frontier: min over maps at each F, vs lambda_I and lambda_II
    print(f"\n{'F0':>6} {'lam_I(2pair)':>13} {'lam_II(3pair)':>14} {'min n=4':>10}  beats both?")
    frontier = []
    for i, F in enumerate(Fgrid):
        best = min(r['lmax'][i] for r in results)
        beats = best < min(lamI(F), lamII(F))
        frontier.append((F, best, beats))
        print(f"{float(F):6.2f} {float(lamI(F)):13.4f} {float(lamII(F)):14.4f} "
              f"{float(best):10.4f}  {beats}")

    # which maps achieve the minimum near threshold, exact closed forms
    i055 = Fgrid.index(Fraction(55, 100))
    best_val = min(r['lmax'][i055] for r in results)
    winners = [r for r in results if r['lmax'][i055] == best_val]
    print(f"\nbest map(s) at F0=0.55: {len(winners)} map(s)")
    for r in winners[:3]:
        p0f, fo = polys(*r['key'])
        print("  p0  =", p0f)
        print("  Fout=", fo)
    with open('n4_results.pkl', 'wb') as fh:
        pickle.dump({'results': results, 'frontier': frontier,
                     'purifying': list(purifying), 'found_RR': found_RR}, fh)
    print("\nsaved n4_results.pkl")
