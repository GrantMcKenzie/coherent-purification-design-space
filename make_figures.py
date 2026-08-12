#!/usr/bin/env python3
"""Regenerate the manuscript figures.

All output is written to ``figures/`` with ``pdf.fonttype = 42``, so no Type 3
fonts are embedded (arXiv rejects those).  ``verify_all.py`` and the CI
workflow both check this.

Generator axes are labelled in the **invariant** ``(P0, P1)`` convention of
Sec. 2.4 -- the kept-pair Pauli first -- so that figures and prose agree
without a transposition step.  An earlier draft drew these in the
(control, target) convention while quoting invariant labels in the text, which
made the figure look inconsistent with Sec. 4.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9

import matplotlib.pyplot as plt
import numpy as np

from finite_rank_purification.circuits import FIXED_FAMILY, accepted_fidelity
from finite_rank_purification.classification import (
    enumerate_atoms,
    is_purifying,
    rho_from_form,
)
from finite_rank_purification.clifford import enumerate_clifford2
from finite_rank_purification.paulis import D, TWO_QUBIT_GENERATORS
from finite_rank_purification.susceptibility import (
    hessian_exact,
    hessian_from_rho,
    spectral_radius,
    syndrome,
)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
SUB = ("XI", "XZ", "ZZ")
COLOURS = {"forward-Z": "#c44e52", "reverse-X": "#dd8452", "reverse-Y": "#55a868"}


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig1_exact_forms(f0=0.9):
    c, chk = FIXED_FAMILY["forward-Z"]
    thetas = np.linspace(-0.5, 0.5, 201)
    f_out0 = accepted_fidelity(c, f0, chk)
    coeffs = {
        "XI": (2 * f0 + 1) * (4 * f0 - 1) / D(f0),
        "XZ": (2 * f0 + 1) * (4 * f0 - 1) / D(f0),
        "ZZ": (4 * f0 - 1) ** 2 / D(f0),
    }
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    worst = 0.0
    for g, colour in zip(SUB, ("#4c72b0", "#dd8452", "#55a868")):
        closed = f_out0 - coeffs[g] * np.sin(2 * thetas) ** 2
        exact = [accepted_fidelity(c, f0, chk, {g: float(t)}) for t in thetas[::10]]
        worst = max(worst, np.abs(np.array(exact) - closed[::10]).max())
        ax.plot(thetas, closed, color=colour, lw=1.4, label=f"{g} closed form")
        ax.plot(thetas[::10], exact, "o", ms=3.5, mfc="none", color=colour,
                label=f"{g} exact sim")
    ax.set_xlabel(r"coherent angle $\theta$")
    ax.set_ylabel(r"accepted fidelity $F_{\mathrm{out}}$")
    ax.set_title(rf"Exact one-parameter form, forward-$Z$, $F_0={f0}$", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    save(fig, "fig1_exact.pdf")
    print(f"    max deviation closed form vs exact simulation: {worst:.1e}")


def fig2_sensitivities():
    f0s = np.linspace(0.26, 1.0, 60)
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6), sharey=False)
    for ax, (name, (c, chk)) in zip(axes, FIXED_FAMILY.items()):
        for g, colour in zip(SUB, ("#4c72b0", "#dd8452", "#55a868")):
            vals = [hessian_exact(c, f, chk, (g,))[0, 0] for f in f0s]
            ax.plot(f0s, vals, color=colour, lw=1.4, label=g)
        ax.axvline(0.25, ls=":", color="0.5", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(r"$F_0$")
    axes[0].set_ylabel(r"coherent sensitivity $M_{gg}$")
    axes[0].legend(fontsize=7, title="generator", title_fontsize=7)
    save(fig, "fig2_sensitivity.pdf")


def fig3_worst_case():
    f0s = np.linspace(0.26, 1.0, 60)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for name, (c, chk) in FIXED_FAMILY.items():
        vals = [spectral_radius(hessian_exact(c, f, chk, SUB)) for f in f0s]
        ax.plot(f0s, vals, color=COLOURS[name], lw=1.6,
                ls="--" if name == "forward-Z" else "-", label=name)
    ax.set_xlabel(r"$F_0$")
    ax.set_ylabel(r"spectral radius $\rho(M)$")
    ax.set_title("Worst-direction coherent susceptibility", fontsize=9)
    ax.annotate("forward-$Z$ and reverse-$X$ coincide", xy=(0.55, 2.0),
                fontsize=6.5, color="0.35")
    ax.legend(fontsize=7)
    save(fig, "fig3_worstcase.pdf")


def fig4_all_generators(f0=0.9):
    """All fifteen generators, in INVARIANT labels."""
    data = np.array([
        np.diag(hessian_exact(c, f0, chk))
        for c, chk in FIXED_FAMILY.values()
    ])
    fig, ax = plt.subplots(figsize=(8.0, 2.1))
    im = ax.imshow(data, aspect="auto", cmap="magma_r", vmin=0)
    ax.set_xticks(range(15), TWO_QUBIT_GENERATORS, fontsize=7)
    ax.set_yticks(range(3), list(FIXED_FAMILY), fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, "0" if abs(v) < 1e-9 else f"{v:.1f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if v > data.max() * 0.6 else "black")
    ax.set_title(
        rf"Diagonal coherent sensitivity, all 15 generators "
        rf"($F_0={f0}$, invariant $(P_0,P_1)$ labels)", fontsize=9)
    fig.colorbar(im, ax=ax, label=r"$M_{gg}$", fraction=0.02)
    save(fig, "fig4_allgen.pdf")


def fig5_syndrome(f0=0.9):
    c, chk = FIXED_FAMILY["forward-Z"]
    order = sorted(range(15), key=lambda i: syndrome(TWO_QUBIT_GENERATORS[i], chk))
    labels = [TWO_QUBIT_GENERATORS[i] for i in order]
    m = hessian_exact(c, f0, chk)[np.ix_(order, order)]
    split = sum(1 for g in labels if syndrome(g, chk) == 0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    lim = np.abs(m).max()
    ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.axhline(split - 0.5, color="#2b3a67", lw=1.2)
    ax.axvline(split - 0.5, color="#2b3a67", lw=1.2)
    ax.set_xticks(range(15), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(15), labels, fontsize=6)
    ax.set_title("(a) $M$ reordered by syndrome", fontsize=9)

    names, ins, outs = [], [], []
    for name, (cc, kk) in FIXED_FAMILY.items():
        mm = hessian_exact(cc, f0, kk)
        inb = outb = 0.0
        for a, ga in enumerate(TWO_QUBIT_GENERATORS):
            for b, gb in enumerate(TWO_QUBIT_GENERATORS):
                if syndrome(ga, kk) == syndrome(gb, kk):
                    inb = max(inb, abs(mm[a, b]))
                else:
                    outb = max(outb, abs(mm[a, b]))
        names.append(name)
        ins.append(inb)
        outs.append(max(outb, 1e-18))
    x = np.arange(3)
    ax2.bar(x - 0.2, ins, 0.4, label="max in-block", color="#4c72b0")
    ax2.bar(x + 0.2, outs, 0.4, label="max off-block", color="#dd8452")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-19, 1e2)
    ax2.axhline(2.2e-16, ls=":", color="0.5", lw=0.8)
    ax2.text(2.35, 3e-16, "machine precision", fontsize=6, color="0.4", ha="right")
    ax2.set_xticks(x, names, fontsize=7)
    ax2.set_title("(b) off-block vanishes identically", fontsize=9)
    ax2.legend(fontsize=7)
    save(fig, "fig5_syndrome.pdf")
    print(f"    largest off-block entry across all three checks: {max(outs):.1e}")


def fig7_inertia():
    c, chk = FIXED_FAMILY["forward-Z"]
    f0s = np.linspace(0.02, 1.0, 90)
    lo, hi, rank = [], [], []
    for f in f0s:
        ev = np.linalg.eigvalsh(hessian_exact(c, f, chk))
        lo.append(ev.min())
        hi.append(ev.max())
        rank.append(int(np.sum(np.abs(ev) > 1e-9)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 2.8))
    a1.plot(f0s, lo, color="#dd8452", label=r"$\lambda_{\min}(M)$")
    a1.plot(f0s, hi, color="#4c72b0", label=r"$\lambda_{\max}(M)$")
    a1.axvspan(0.25, 0.5, color="#dd8452", alpha=0.12)
    a1.text(0.375, 3.0, "indefinite", fontsize=6.5, ha="center", color="#a2543a")
    for v in (0.25, 0.5):
        a1.axvline(v, ls="--", color="0.4", lw=0.8)
    a1.axhline(0, color="0.7", lw=0.6)
    a1.set_xlabel(r"input fidelity $F_0$")
    a1.set_ylabel(r"eigenvalue of $M$")
    a1.set_title(r"(a) PSD iff $F_0 \geq 1/2$", fontsize=9)
    a1.legend(fontsize=7)

    a2.plot(f0s, rank, color="#4c72b0", lw=1.4)
    for v, r in ((0.25, 0), (0.5, 8), (1.0, 4)):
        ev = np.linalg.eigvalsh(hessian_exact(c, v, chk))
        a2.plot([v], [int(np.sum(np.abs(ev) > 1e-9))], "o", color="#dd8452", ms=6)
    a2.set_yticks([0, 4, 8, 10])
    a2.set_xlabel(r"input fidelity $F_0$")
    a2.set_ylabel(r"$\mathrm{rank}\,M$")
    a2.set_title("(b) rank drops at the recurrence fixed points", fontsize=9)
    save(fig, "fig7_inertia.pdf")


def fig11_design_space(f0=0.9):
    """New: the full thirty-row design table as a heatmap of Tr M."""
    cliffords = enumerate_clifford2()
    atoms = enumerate_atoms(cliffords)
    names = sorted(atoms)
    grid = np.zeros((len(names), 3))
    mark = np.empty((len(names), 3), dtype=object)
    for i, n in enumerate(names):
        for j, chk in enumerate("ZXY"):
            m = hessian_from_rho(rho_from_form(atoms[n]["S"], f0), chk)
            grid[i, j] = np.trace(m)
            mark[i, j] = "P" if is_purifying(atoms[n]["label"], chk) else "F"
    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(3), [f"${c}$" for c in "ZXY"])
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(3):
            ax.text(j, i, f"{mark[i,j]}\n{grid[i,j]:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if grid[i, j] > grid.mean() else "black")
    ax.set_title(rf"Design space: $\mathrm{{Tr}}\,M$ at $F_0={f0}$" "\n"
                 "P = purifying, F = fixed point", fontsize=9)
    fig.colorbar(im, ax=ax, label=r"$\mathrm{Tr}\,M$", fraction=0.05)
    save(fig, "fig11_designspace.pdf")


def main():
    print("Regenerating figures (pdf.fonttype = 42)")
    fig1_exact_forms()
    fig2_sensitivities()
    fig3_worst_case()
    fig4_all_generators()
    fig5_syndrome()
    fig7_inertia()
    fig11_design_space()
    print("done")


if __name__ == "__main__":
    main()
