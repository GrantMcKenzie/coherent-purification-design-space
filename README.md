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
$ python verify_all.py
...
All 91 checks passed.
```

## Quick start

```bash
git clone https://github.com/GrantMcKenzie/coherent-purification-design-space.git
cd coherent-purification-design-space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-frozen.txt   # exact pinned environment
python -m pytest tests/ -q               # 22 unit tests, ~7 s
python verify_all.py                     # 91 assertions, ~15 min
python make_figures.py                   # regenerates figures/
```

`verify_all.py` exits nonzero if any stated result fails to reproduce. It is the same
suite CI runs on every push.

## Two independent engines

The central design choice is that `rho_C`, the state after the bilateral Clifford, is
computed twice by routes that share no code:

| engine | route | modules |
|---|---|---|
| density-matrix | build the 16x16 unitary `C (x) conj(C)`, apply it to `rho_0 (x) rho_0`, project onto the accepted branch | `circuits`, `susceptibility` |
| group-theoretic | take the symplectic image `S` of `C`, transport the Arf-zero form `qtilde = q . S^-1`, read the Bell populations off Eq. (28) | `clifford`, `classification` |

The second never constructs a unitary. Their agreement (to `2.5e-16` over all ten atoms)
is asserted in `verify_all.py`, and is what §11 of the paper reports as a cross-validation
of two separately written engines.

## What is where

| path | contents |
|---|---|
| `finite_rank_purification/paulis.py` | Pauli algebra, Bell bases, symplectic form, Werner input, `D(F_0)` |
| `finite_rank_purification/clifford.py` | BFS enumeration of all 11520 two-qubit Cliffords mod phase; symplectic images |
| `finite_rank_purification/circuits.py` | bilateral circuits, the three Pauli checks, accepted branch |
| `finite_rank_purification/susceptibility.py` | the Hessian `M`: exact second-order and finite-difference engines |
| `finite_rank_purification/classification.py` | Arf invariant, ten atoms, dichotomy, populations from the form |
| `finite_rank_purification/closed_forms.py` | the five-letter alphabet and fixed-point letters as sympy expressions |
| `verify_all.py` | 91 assertions covering every claim |
| `make_figures.py` | regenerates figures at `pdf.fonttype=42` |
| `tests/` | 22 unit tests |

## Selected reproduced results

| result | paper | reproduced |
|---|---|---|
| `\|C_2 / phases\|` | 11520 | 11520 |
| distinct atoms | 10, multiplicity 1152 each | 10, all 1152 |
| `M` diagonal in the Pauli basis | [VER], all 30 (atom, check) pairs | max off-diagonal **exactly 0.0** |
| worst-direction susceptibility | `4(2F+1)(4F-1)/D` on every purifying pair | 3.6954314721, single value over all 18 |
| `lambda_A - lambda_{C,D,E}` discriminants | -135, -36, -279 | -135, -36, -279 |
| irreducible core | `{XI,XX,XZ,ZI,ZX,ZZ,YY}` | identical; criterion (33) predicts it exactly |
| blind cores | Z,X of size 3; Y of size 4 | identical |
| exact one-parameter form | `F(0) - c sin^2(2 theta)` | max deviation `2.8e-16` |

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

## Conventions

Generator labels are **invariant** `(P_0, P_1)` labels throughout the code and the
regenerated figures: `P_0` acts on the kept pair, `P_1` on the sacrificial pair
(paper, Sec. 2.4). The fixed-circuit (control, target) convention differs by a
transposition under reverse orientation, which is why the reverse-`X` exact null appears
as `IX` here and as `XI` in the fixed-circuit tables.

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
