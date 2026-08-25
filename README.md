# Coherent-error robustness of entanglement purification is a finite design problem

[![verify](https://github.com/GrantMcKenzie/coherent-purification-design-space/actions/workflows/verify.yml/badge.svg)](https://github.com/GrantMcKenzie/coherent-purification-design-space/actions/workflows/verify.yml)
[![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Companion code for G. McKenzie, *Coherent-error robustness of entanglement purification
is a finite design problem* (arXiv:XXXX.XXXXX).

The paper gives a complete leading-order theory of the coherent response of
single-selection Clifford entanglement purification, classifies the two-pair design
space into ten equivalence classes, and follows the resulting worst-case no-go up the
resource ladder to three and four pairs. This repository reproduces every number and
figure in the paper and checks each stated result with a machine assertion.

```
$ python verify_all.py        # two-pair engines
All 91 checks passed.
$ python verify_threepair.py  # master formula and three pairs
All 38 checks passed.
$ python validate.py          # general-n engine vs. every published n<=3 number
n=3: crossover numerics ... engine validated.
```

## Quick start

```bash
git clone https://github.com/GrantMcKenzie/coherent-purification-design-space.git
cd coherent-purification-design-space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-frozen.txt   # exact pinned environment
python -m pytest tests/ -q               # unit tests, ~10 s
python verify_all.py                     # 91 assertions, ~15 min
python verify_threepair.py               # 38 assertions, ~3 min
python validate.py                       # general-n engine self-check, ~1 min
python make_figures.py                   # regenerates the paper's figures 1-7
python make_fig8.py                      # regenerates figure 8 (four-pair window)
```

`verify_all.py` exits nonzero if any stated result fails to reproduce. It is the same
suite CI runs on every push. `validate.py` is the gate for the general-`n` engine: it
must reproduce every published two- and three-pair number before any four-pair claim
built on that engine is trusted.

## Three independent engines

The central design choice is that the susceptibility is computed by routes that share
no code, so agreement between them is evidence rather than repetition:

| engine | route | modules |
|---|---|---|
| density-matrix | build the 16x16 unitary `C (x) conj(C)`, apply it to `rho_0 (x) rho_0`, project onto the accepted branch | `circuits`, `susceptibility` |
| group-theoretic | take the symplectic image `S` of `C`, transport the form, read the Bell populations off the level sets | `clifford`, `classification` |
| master formula | evaluate the closed-form signed sum over Bell labels directly; never builds a unitary or a density matrix, and works at any `n` | `master.py`, `threepair.py`, `general_n.py` |

The second never constructs a unitary; the third never constructs either. Their pairwise
agreement (`2.5e-16` and `1.3e-15` at two and three pairs) is asserted in `verify_all.py`
and `verify_threepair.py`. The master-formula engine is what makes the higher-`n`
sections checkable at all: it evaluates all 635040 susceptibilities of the 1120-atom
three-pair design space, and scales to the four-pair analysis below.

These are the same three routes the paper refers to in Sec. 3.3 and Appendix C: the
density-matrix engine is the literal `4^n x 4^n` implementation (available in both exact
second-order and central-finite-difference form), the group-theoretic engine is the
Bell-basis reduction whose closed forms are checked symbolically in Appendix A, and the
master formula is the direct evaluation of Eq. (13).

## The general-`n` engine and the four-pair results

`general_n.py` is the master formula packaged for arbitrary pair count: integer Bell
labels (bit `2k` = x-bit of pair `k`, bit `2k+1` = z-bit), the symplectic form, `Y`-parity,
Werner populations, and exact-`Fraction` susceptibilities for any symplectic circuit and
Pauli check set. `validate.py` pins it against the published record before it is used
anywhere else; every four-pair number in the paper is produced by this engine and
cross-checked as noted below.

| script | what it does |
|---|---|
| `general_n.py` | master-formula engine at any `n`: labels, symplectic group, `Row` class, exact susceptibilities |
| `validate.py` | reproduces every published `n=2` and `n=3` number from `general_n.py` (must pass first) |
| `fastrow.py` | vectorized float engine for large four-pair scans (exact engine reserved for confirmed minima) |
| `explore_n4.py` | sampled census of Sp(8,2): fidelity maps, `lambda_max` frontier, open-question tests |
| `structured_n4.py` | structured four-pair circuits (nested trees, chains, stars) the sampler can miss |
| `nested_analysis.py` | exact analysis of the two-round (untwirled) recurrence map |
| `exact_crossovers.py` | exact four-pair crossovers plus an independent 256-dim density-matrix check |
| `depolarizing_expansion.py` | small-`p` depolarizing expansion `G(F_0)` and the additive combined model |
| `qutip_validation.py` | leading-order prediction against a realistic T1/T2 background in QuTiP |
| `make_fig8.py` | regenerates `figures/fig8_fourpair.pdf` (the widened four-pair window) |

**Headline four-pair results (all exact unless marked):**

- A four-pair map `II'` shares map II's accepted fidelity but improves its worst case at
  *every* fidelity: `lambda_II' - lambda_II = -4(4F-1)^2(F-1)^2 / ((8F^2-F+2) D3) < 0`.
- The improvement window over the two-pair bound widens from `(1/2, (3*sqrt(2)-2)/4)`
  ~= `(1/2, 0.5607)` at three pairs to `(1/2, F**)` with `F**` the real root of
  `16F^3 - 8F^2 + 4F - 3` ~= `0.602047` at four pairs.
- Above the window the two-pair value persists as the floor: over 99008 genuinely
  purifying sampled rows plus structured trees at `F_0 = 0.9`, the minimum worst-direction
  susceptibility is exactly `lambda_A`, and `lambda_max` stays single-valued per fidelity
  map. This is sampled evidence, not exhaustive proof (the four-pair design space has
  1523200 atoms); the paper states it as such.
- One four-pair susceptibility is checked against an independent 256-dimensional
  density-matrix simulation, agreeing to `4.7e-11`.

## Small-rate depolarizing noise

`depolarizing_expansion.py` derives the leading stochastic response and shows it is
degenerate across purifying circuits:

- `Fout(F_0; p) = Fout(F_0, 0) - p G(F_0) + O(p^2)` with
  `G(F_0) = 2(4F-1)(8F+1)(4F^2-2F+7) / (3 D^2)`, **identical** across all 18 purifying
  (atom, check) pairs -- verified exhaustively over all 720 elements of Sp(4,2).
- Because the depolarized state stays Bell-diagonal, there is no `O(p theta)` cross term
  and the combined response is additive at leading order,
  `Fout = Fout(F_0,0) - p G - theta^T M theta + O(p^2, p|theta|^2, |theta|^4)`,
  checked numerically against density-matrix simulation.

The operational point: acceptance rate, accepted fidelity, and first-order depolarizing
sensitivity are all blind to which purifying circuit was built. Through first order in
the noise rate, stochastic Pauli benchmarking cannot separate purifying two-pair
circuits -- the coherent response is the only separator.

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
| `general_n.py` | general-`n` master-formula engine used for the four-pair analysis |
| `validate.py` | general-`n` engine self-check against every published `n<=3` number |
| `fastrow.py`, `explore_n4.py`, `structured_n4.py`, `nested_analysis.py`, `exact_crossovers.py` | four-pair census, structured circuits, crossovers, independent check |
| `depolarizing_expansion.py` | small-`p` depolarizing expansion and the combined additive model |
| `qutip_validation.py` | leading-order prediction against a realistic T1/T2 background |
| `verify_all.py` | 91 assertions covering the two-pair claims |
| `verify_threepair.py` | 38 assertions covering the master formula and three pairs |
| `make_figures.py`, `make_fig8.py` | regenerate the paper's figures at `pdf.fonttype=42` |
| `figures/` | generated PDF figures |
| `tests/` | unit tests (engine internals + master/general_n agreement) |

## Selected reproduced results

| result | paper | reproduced |
|---|---|---|
| `\|C_2 / phases\|` | 11520 | 11520 |
| distinct atoms | 10, multiplicity 1152 each | 10, all 1152 |
| `M` diagonal in the Pauli basis | Corollary 5, all 30 (atom, check) pairs | max off-diagonal **exactly 0.0** |
| worst-direction susceptibility | `4(2F+1)(4F-1)/D` on every purifying pair | 3.6954314721, single value over all 18 |
| `lambda_A - lambda_{C,D,E}` discriminants | -135, -36, -279 | -135, -36, -279 |
| irreducible core | `{XI,XX,XZ,ZI,ZX,ZZ,YY}` | identical; the nullability criterion predicts it exactly |
| blind cores | Z,X of size 3; Y of size 4 | identical |
| exact one-parameter form | `F(0) - c sin^2(2 theta)` | max deviation `3.3e-16` |
| pair-frame atom counts | 10 at `n=2`, 1120 at `n=3` | 10 and 1120, by direct enumeration |
| three-pair fidelity maps | exactly two | two (2592 and 5184 of 7776 purifying rows) |
| `F0*` crossover (three pairs) | `(3*sqrt(2)-2)/4` | 0.560660171780 |
| three-pair `lambda_max` | one value per map | single-valued, matches both closed forms |
| four-pair window boundary | root of `16F^3-8F^2+4F-3` | 0.602047318427 |
| four-pair gain `lambda_II' - lambda_II` | `-4(4F-1)^2(F-1)^2/((8F^2-F+2)D3)` | perfect square, negative on `(1/4,1)` |
| depolarizing `G(F_0)` | one function over all purifying pairs | identical over all 18, `G(0.9)=1.931906` |

Three corrections to an earlier draft are retained in `verify_all.py` as `[FIX]`
regression guards (six `check("FIX", ...)` assertions in total) so those errors
cannot reappear silently:

1. The Z and X checks do **not** share one `Tr M`. Their twelve purifying pairs split
   into 15.792 (x8), 15.870 (x2) and 16.770 (x2) at `F_0 = 0.9`.
2. The Y check has **no** exceptional atom: all six of its purifying atoms share the
   multiplicity pattern `(2,1,3,3,1)` at uniform rank 10. This is why the isotropic
   optimum is six-fold degenerate, and why `2 lA + lB + 3 lC + 3 lD + lE` is a single
   rational function of `F_0`, equal to the `Tr M = 12.90` quoted in the design-payoff section (isotropic-`Sigma` optimum).
3. The irreducible-core minimum is `lambda_D` only above threshold:
   `lambda_C - lambda_D` carries the factor `(2F_0 - 1)` and the two letters cross at
   `F_0 = 1/2`.

`verify_threepair.py` separately records the three-pair check-independence result
(as `[VER]` checks, not `[FIX]` guards): each check pair purifies exactly 864 of the
1120 atoms, but **not the same 864** -- only 228 atoms purify under all nine, so the
count is check-independent while the set is not.

A further correction is recorded in the four-pair scripts: the plain nested all-`Z` CNOT
tree (two rounds, no interleaved rotations) does **not** purify -- its fixed-point
polynomial factors as `(F-1)(2F-1)(4F-1)(4F^2-14F+1)` and `Fout < F` on all of `(1/2,1)`.
The textbook iterated map `R(R(F))` presumes an inter-round twirl, a stochastic operation
outside this architecture. `explore_n4.py` and `corrected` floor tests filter on a
purification margin so fixed-point rows (`mu_A = 52/15`) cannot leak through a float
`Fout > F` comparison.

## Conventions

Generator labels are **invariant** `(P_0, P_1)` labels throughout the code, the figures
and the paper: `P_0` acts on the kept pair, `P_1` on the sacrificial pair. The
fixed-circuit (control, target) convention differs by a transposition under reverse
orientation, which is why the reverse-`X` exact null is `IX` here and appears as `XI` in
older tables written in that convention.

`master.py` and `general_n.py` use `(x, z)` bit order per pair internally: `(0,0)=I`,
`(1,0)=X`, `(0,1)=Z`, `(1,1)=Y`. This matches the density-matrix engine's Bell labelling,
and the agreement assertion in `verify_threepair.py` (plus `validate.py` for the
general-`n` engine) is what pins it down.

## Environment

Verified on Python 3.12.3 (x86_64 Linux) and Python 3.11 (macOS arm64), with
numpy 2.4.4 / scipy 1.17.1 / sympy 1.14.0 / matplotlib 3.10.8 on both. The four-pair
independent check and `qutip_validation.py` additionally require qutip 5.3.1.
`requirements.txt` gives ranged dependencies; `requirements-frozen.txt` gives the exact
versions used for the submitted figures; `environment.yml` is the conda equivalent. For
the state that produced the paper, check out the tag `v1.0.0`.

CI runs the full suite on both Python 3.11 and 3.12. Figures are written with
`pdf.fonttype = 42`; CI fails the build if any figure embeds a Type 3 font, which arXiv
rejects.

## Citing

See `CITATION.cff`, or the "Cite this repository" button above. Please cite both the
paper and the archived software release. The arXiv identifier and the Zenodo DOI are
placeholders until posting, at which point the release is archived and both are filled
in here and in `CITATION.cff`.

## License

MIT. See `LICENSE`.
