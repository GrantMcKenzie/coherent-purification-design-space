# Coherent-error robustness of entanglement purification is a finite design problem

[![verify](https://github.com/GrantMcKenzie/coherent-purification-design-space/actions/workflows/verify.yml/badge.svg)](https://github.com/GrantMcKenzie/coherent-purification-design-space/actions/workflows/verify.yml)
[![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Companion code for G. McKenzie, *Coherent-error robustness of entanglement purification
is a finite design problem* (arXiv:XXXX.XXXXX).

The paper gives a complete leading-order theory of the coherent response of two-pair,
single-selection Clifford entanglement purification, and classifies the entire design
space into ten equivalence classes. This repository reproduces every number and figure
in the paper and checks each stated result with a machine assertion.

```
$ python verify_all.py        # two-pair engines
All 91 checks passed.
$ python verify_threepair.py  # master formula and three pairs
All 38 checks passed.
```

## Quick start

```bash
git clone https://github.com/GrantMcKenzie/coherent-purification-design-space.git
cd coherent-purification-design-space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-frozen.txt   # exact pinned environment
python -m pytest tests/ -q               # 22 unit tests, ~10 s
python verify_all.py                     # 91 assertions, ~15 min
python verify_threepair.py               # 38 assertions, ~3 min
python make_figures.py                   # regenerates the paper's 7 figures
```

`verify_all.py` exits nonzero if any stated result fails to reproduce. It is the same
suite CI runs on every push.

## Three independent engines

The central design choice is that the susceptibility is computed by routes that share
no code, so agreement between them is evidence rather than repetition:

| engine | route | modules |
|---|---|---|
| density-matrix | build the 16x16 unitary `C (x) conj(C)`, apply it to `rho_0 (x) rho_0`, project onto the accepted branch | `circuits`, `susceptibility` |
| group-theoretic | take the symplectic image `S` of `C`, transport the form, read the Bell populations off the level sets | `clifford`, `classification` |
| master formula | evaluate the closed-form signed sum over Bell labels directly; never builds a unitary or a density matrix, and works at any `n` | `master.py`, `threepair.py` |

The second never constructs a unitary; the third never constructs either. Their pairwise
agreement (`2.5e-16` and `1.3e-15`) is asserted in `verify_all.py` and
`verify_threepair.py`. The third engine is what makes the three-pair section checkable at
all: it evaluates all 635040 susceptibilities of the 1120-atom design space.

## What is where

| path | contents |
|---|---|
| `finite_rank_purification/paulis.py` | Pauli algebra, Bell bases, symplectic form, Werner input, `D(F_0)` |
| `finite_rank_purification/clifford.py` | BFS enumeration of all 11520 two-qubit Cliffords mod phase; symplectic images |
| `finite_rank_purification/circuits.py` | bilateral circuits, the three Pauli checks, accepted branch |
| `finite_rank_purification/susceptibility.py` | the Hessian `M`: exact second-order and finite-difference engines |
| `finite_rank_purification/classification.py` | Arf invariant, ten atoms, dichotomy, populations from the form |
| `finite_rank_purification/closed_forms.py` | the five-letter alphabet and fixed-point letters as sympy expressions |
| `master.py` | the master formula at any `n`; Bell labels, symplectic form, `Y`-parity |
| `threepair.py` | pair-frame enumeration and three-pair evaluation |
| `verify_all.py` | 91 assertions covering the two-pair claims |
| `verify_threepair.py` | 38 assertions covering the master formula and three pairs |
| `make_figures.py` | regenerates the paper's seven figures at `pdf.fonttype=42` |
| `tests/` | 22 unit tests |

## Selected reproduced results

| result | paper | reproduced |
|---|---|---|
| `\|C_2 / phases\|` | 11520 | 11520 |
| distinct atoms | 10, multiplicity 1152 each | 10, all 1152 |
| `M` diagonal in the Pauli basis | [VER], all 30 (atom, check) pairs | max off-diagonal **exactly 0.0** |
| worst-direction susceptibility | `4(2F+1)(4F-1)/D` on every purifying pair | 3.6954314721, single value over all 18 |
| `lambda_A - lambda_{C,D,E}` discriminants | -135, -36, -279 | -135, -36, -279 |
| irreducible core | `{XI,XX,XZ,ZI,ZX,ZZ,YY}` | identical; the nullability criterion predicts it exactly |
| blind cores | Z,X of size 3; Y of size 4 | identical |
| exact one-parameter form | `F(0) - c sin^2(2 theta)` | max deviation `3.3e-16` |
| pair-frame atom counts | 10 at `n=2`, 1120 at `n=3` | 10 and 1120, by direct enumeration |
| three-pair fidelity maps | exactly two | two (2592 and 5184 of 7776 purifying rows) |
| `F0*` crossover | `(3*sqrt(2)-2)/4` | 0.560660171780 |
| three-pair `lambda_max` | one value per map | single-valued, matches both closed forms |

Three entries in `verify_all.py` are tagged `[FIX]`. They guard corrections to an earlier
draft and exist so those errors cannot reappear silently:

1. The Z and X checks do **not** share one `Tr M`. Their twelve purifying pairs split
   into 15.792 (x8), 15.870 (x2) and 16.770 (x2) at `F_0 = 0.9`.
2. The Y check has **no** exceptional atom: all six of its purifying atoms share the
   multiplicity pattern `(2,1,3,3,1)` at uniform rank 10. This is why the isotropic
   optimum is six-fold degenerate, and `2 lambda_A + lambda_B + 3 lambda_C + 3 lambda_D
   + lambda_E` equals Eq. (36) identically in `F_0`.
3. The irreducible-core minimum is `lambda_D` only above threshold:
   `lambda_C - lambda_D` carries the factor `(2F_0 - 1)` and the two letters cross at
   `F_0 = 1/2`.
4. At three pairs, each check pair purifies exactly 864 of the 1120 atoms, but **not the
   same 864**: only 228 atoms purify under all nine. The count is check-independent; the
   set is not.

## Conventions

Generator labels are **invariant** `(P_0, P_1)` labels throughout the code, the figures
and the paper: `P_0` acts on the kept pair, `P_1` on the sacrificial pair. The
fixed-circuit (control, target) convention differs by a transposition under reverse
orientation, which is why the reverse-`X` exact null is `IX` here and appears as `XI` in
older tables written in that convention.

`master.py` uses `(x, z)` bit order per pair internally: `(0,0)=I`, `(1,0)=X`, `(0,1)=Z`,
`(1,1)=Y`. This matches the density-matrix engine's Bell labelling, and the agreement
assertion in `verify_threepair.py` is what pins it down.

## Environment

Verified on Python 3.12.3 (x86_64 Linux) and Python 3.11 (macOS arm64), with
numpy 2.4.4 / scipy 1.17.1 / sympy 1.14.0 / matplotlib 3.10.8 on both. `requirements.txt` gives ranged dependencies;
`requirements-frozen.txt` gives the exact versions used for the submitted figures;
`environment.yml` is the conda equivalent. For the state that produced the paper, check
out the tag `v1.0.0`.

CI runs the full suite on both Python 3.11 and 3.12. Figures are written with
`pdf.fonttype = 42`; CI fails the build if any figure embeds a Type 3 font, which arXiv
rejects.

## Citing

See `CITATION.cff`, or the "Cite this repository" button above. Please cite both the
paper and the archived software release.

## License

MIT. See `LICENSE`.
