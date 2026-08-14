#!/usr/bin/env python3
"""Regenerate every figure in the manuscript.

Layout rules enforced throughout:
  * pdf.fonttype = 42 (no Type 3 bitmaps; arXiv flags those)
  * legends outside the data area
  * no annotation drawn across a plotted curve
  * constrained_layout for all multi-panel figures

Generator axes use the INVARIANT (kept, sacrificial) labels of Sec. 2.6,
matching the text everywhere.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.titlesize"] = 9
matplotlib.rcParams["legend.fontsize"] = 7.5

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finite_rank_purification.circuits import FIXED_FAMILY, accepted_fidelity
from finite_rank_purification.paulis import D, TWO_QUBIT_GENERATORS
from finite_rank_purification.susceptibility import hessian_exact, syndrome

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

C = {"forward-Z": "#3b6ea5", "reverse-X": "#c1663d", "reverse-Y": "#2e8b6f"}
GC = ["#3b6ea5", "#c1663d", "#2e8b6f", "#8a6bbe"]
SUB = ("XI", "XZ", "ZZ", "IX")


def lam(F0):
    d = D(F0)
    return {
        "A": 4 * (2 * F0 + 1) * (4 * F0 - 1) / d,
        "B": 4 * (4 * F0 - 1) ** 2 / d,
        "C": 4 * (1 - F0) * (2 * F0 + 1) * (4 * F0 - 1) * (8 * F0 + 1) / d**2,
        "D": 8 * (1 - F0) * (F0 + 2) * (2 * F0 + 1) * (4 * F0 - 1) / d**2,
        "E": 12 * (1 - F0) * (2 * F0 - 1) * (2 * F0 + 1) * (4 * F0 - 1) / d**2,
    }


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p)


# ---------------------------------------------------------------- fig 1
def fig1(f0=0.9):
    c, chk = FIXED_FAMILY["forward-Z"]
    th = np.linspace(-0.55, 0.55, 241)
    F00 = accepted_fidelity(c, f0, chk)
    coef = {"XI": (2 * f0 + 1) * (4 * f0 - 1) / D(f0),
            "XZ": (2 * f0 + 1) * (4 * f0 - 1) / D(f0),
            "ZZ": (4 * f0 - 1) ** 2 / D(f0)}
    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    worst = 0.0
    for g, col in zip(("XI", "XZ", "ZZ"), GC):
        closed = F00 - coef[g] * np.sin(2 * th) ** 2
        pts = th[::12]
        sim = np.array([accepted_fidelity(c, f0, chk, {g: float(t)}) for t in pts])
        worst = max(worst, np.abs(sim - (F00 - coef[g] * np.sin(2 * pts) ** 2)).max())
        ax.plot(th, closed, color=col, lw=1.5, label=f"${g}$ closed form", zorder=2)
        ax.plot(pts, sim, "o", ms=3.2, mfc="none", mew=0.9, color=col,
                label=f"${g}$ exact sim", zorder=3)
    ax.set_xlabel(r"coherent angle $\theta$")
    ax.set_ylabel(r"accepted fidelity $F_{\mathrm{out}}$")
    ax.set_title(rf"forward-$Z$, $F_0={f0}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              frameon=False, handletextpad=0.4, columnspacing=1.0)
    save(fig, "fig1_exact_forms.pdf")
    print(f"    max |closed form - exact sim| = {worst:.1e}")


# ---------------------------------------------------------------- fig 2
def fig2():
    f0s = np.linspace(0.26, 1.0, 70)
    pts = np.linspace(0.30, 0.98, 12)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharey=True,
                             constrained_layout=True)
    for ax, (name, (c, chk)) in zip(axes, FIXED_FAMILY.items()):
        for g, col in zip(SUB, GC):
            curve = [hessian_exact(c, f, chk, (g,))[0, 0] for f in f0s]
            dots = [hessian_exact(c, f, chk, (g,))[0, 0] for f in pts]
            ax.plot(f0s, curve, color=col, lw=1.5, label=f"${g}$", zorder=2)
            ax.plot(pts, dots, "o", ms=3.0, mfc="none", mew=0.9, color=col, zorder=3)
        ax.axvline(0.25, ls=":", color="0.55", lw=0.9)
        ax.set_title(name)
        ax.set_xlabel(r"$F_0$")
        ax.set_xlim(0.24, 1.0)
    axes[0].set_ylabel(r"$M_{gg}$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
               frameon=False)
    save(fig, "fig2_sensitivities.pdf")


# ---------------------------------------------------------------- fig 3
def fig3(f0=0.9):
    data = np.array([np.diag(hessian_exact(c, f0, chk))
                     for c, chk in FIXED_FAMILY.values()])
    fig, ax = plt.subplots(figsize=(7.1, 1.85), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_xticks(range(15), [f"${g}$" for g in TWO_QUBIT_GENERATORS], fontsize=7.5)
    ax.set_yticks(range(3), list(FIXED_FAMILY), fontsize=8)
    hi = data.max()
    for i in range(3):
        for j in range(15):
            v = data[i, j]
            ax.text(j, i, "0" if abs(v) < 1e-9 else f"{v:.1f}", ha="center",
                    va="center", fontsize=6.2,
                    color="white" if v > 0.55 * hi else "0.15")
    ax.set_title(rf"diagonal coherent sensitivity, all 15 generators "
                 rf"(invariant labels, $F_0={f0}$)")
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, label=r"$M_{gg}$")
    save(fig, "fig3_allgen.pdf")


# ---------------------------------------------------------------- fig 4
def fig4(f0=0.9):
    c, chk = FIXED_FAMILY["forward-Z"]
    order = sorted(range(15), key=lambda i: syndrome(TWO_QUBIT_GENERATORS[i], chk))
    labs = [TWO_QUBIT_GENERATORS[i] for i in order]
    m = hessian_exact(c, f0, chk)[np.ix_(order, order)]
    split = sum(1 for g in labs if syndrome(g, chk) == 0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 3.0),
                                 gridspec_kw={"width_ratios": [1.05, 1]},
                                 constrained_layout=True)
    lim = np.abs(m).max()
    im = a1.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    a1.axhline(split - 0.5, color="0.2", lw=1.1)
    a1.axvline(split - 0.5, color="0.2", lw=1.1)
    a1.set_xticks(range(15), [f"${g}$" for g in labs], rotation=90, fontsize=6)
    a1.set_yticks(range(15), [f"${g}$" for g in labs], fontsize=6)
    a1.set_title("(a) ordered by check syndrome")
    fig.colorbar(im, ax=a1, fraction=0.045, pad=0.02)

    names, ins, outs = [], [], []
    for name, (cc, kk) in FIXED_FAMILY.items():
        mm = hessian_exact(cc, f0, kk)
        i_b = o_b = 0.0
        for a, ga in enumerate(TWO_QUBIT_GENERATORS):
            for b, gb in enumerate(TWO_QUBIT_GENERATORS):
                if syndrome(ga, kk) == syndrome(gb, kk):
                    i_b = max(i_b, abs(mm[a, b]))
                else:
                    o_b = max(o_b, abs(mm[a, b]))
        names.append(name); ins.append(i_b); outs.append(o_b)
    x = np.arange(3)
    a2.bar(x - 0.19, ins, 0.38, color="#3b6ea5", label="largest within-syndrome")
    a2.bar(x + 0.19, outs, 0.38, color="#c1663d", label="largest cross-syndrome")
    for xi, o in zip(x, outs):
        a2.plot([xi + 0.19], [0], marker="_", ms=14, color="#c1663d")
        a2.annotate("exactly 0", xy=(xi + 0.19, 0), xytext=(xi + 0.19, 0.28),
                    ha="center", fontsize=6.5, color="#8a4425")
    a2.set_xticks(x, names, fontsize=8)
    a2.set_ylim(0, max(ins) * 1.28)
    a2.set_ylabel(r"$|M_{ab}|$")
    a2.set_title("(b) cross-syndrome entries vanish")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False)
    save(fig, "fig4_syndrome.pdf")
    print(f"    largest cross-syndrome entry over all three checks: {max(outs):.1e}")


# ---------------------------------------------------------------- fig 5
def fig5():
    f0s = np.linspace(0.02, 1.0, 200)
    L = lam(f0s)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    for k, col in zip("ABCDE", GC + ["#b23b5e"]):
        a1.plot(f0s, L[k], color=col, lw=2.0 if k == "E" else 1.3,
                label=rf"$\lambda_{k}$", zorder=3 if k == "E" else 2)
    a1.axhline(0, color="0.7", lw=0.7)
    for v in (0.25, 0.5):
        a1.axvline(v, ls="--", color="0.45", lw=0.8)
    a1.axvspan(0.25, 0.5, color="#c1663d", alpha=0.10)
    a1.set_xlabel(r"input fidelity $F_0$")
    a1.set_ylabel("susceptibility entry")
    a1.set_title(r"(a) only $\lambda_E$ changes sign, at $F_0=1/2$")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False)

    labels_ = ["isotropic", "$YYZ$-biased"]
    A = [29.744, 3.8152]
    B = [37.485, 0.0307]
    x = np.arange(2)
    a2.bar(x - 0.19, A, 0.38, color="#3b6ea5", label="protocol A")
    a2.bar(x + 0.19, B, 0.38, color="#c1663d", label="protocol B")
    a2.set_yscale("log")
    a2.set_ylim(0.01, 400)
    for xi, (a, b) in zip(x, zip(A, B)):
        a2.text(xi - 0.19, a * 1.25, f"{a:.2f}", ha="center", fontsize=6.5)
        a2.text(xi + 0.19, b * 1.25, f"{b:.2f}", ha="center", fontsize=6.5)
        lo = "A" if a < b else "B"
        a2.text(xi, 150, f"{lo} lower", ha="center", fontsize=7, color="0.25")
    a2.set_xticks(x, labels_, fontsize=8)
    a2.set_ylabel(r"$\mathrm{Tr}(M\Sigma)$  (lower is better)")
    a2.set_title("(b) identical $p_0$ and $F_{\\mathrm{out}}$,\nopposite winners")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    save(fig, "fig5_threshold_design.pdf")


# ---------------------------------------------------------------- fig 6
def fig6():
    f0s = np.linspace(0.26, 1.0, 300)
    D2 = 8 * f0s**2 - 4 * f0s + 5
    D3 = 16 * f0s**2 - 14 * f0s + 7
    lI = 4 * (2 * f0s + 1) * (4 * f0s - 1) / D2
    lII = 12 * f0s * (4 * f0s - 1) / D3
    star = (3 * np.sqrt(2) - 2) / 4
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    a1.plot(f0s, lI, color="#3b6ea5", lw=1.6, label="best two-pair map")
    a1.plot(f0s, lII, color="#c1663d", lw=1.6, label="three-pair map II")
    a1.axvspan(0.5, star, color="#2e8b6f", alpha=0.16)
    a1.set_xticks([0.3, 0.5, star, 0.7, 0.9],
                  ["0.3", "0.5", r"$F_0^\star$", "0.7", "0.9"])
    a1.set_xlabel(r"input fidelity $F_0$")
    a1.set_ylabel(r"$\lambda_{\max}(M)$")
    a1.set_title("(a) worst-direction susceptibility")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)

    a2.plot(f0s, lII - lI, color="#c1663d", lw=1.6)
    a2.axhline(0, color="0.6", lw=0.8)
    a2.axvspan(0.5, star, color="#2e8b6f", alpha=0.16)
    a2.set_xticks([0.3, 0.5, star, 0.7, 0.9],
                  ["0.3", "0.5", r"$F_0^\star$", "0.7", "0.9"])
    a2.set_xlabel(r"input fidelity $F_0$")
    a2.set_ylabel(r"$\lambda^{(\mathrm{II})}_{\max}-\lambda^{(\mathrm{I})}_{\max}$")
    a2.set_title(rf"(b) crossover at $F_0^\star={star:.6f}$")
    a2.annotate("three-pair map lower", xy=(0.53, (lII - lI).min() * 0.55),
                fontsize=6.8, color="#1f6b53", ha="left")
    save(fig, "fig6_threepair.pdf")
    print(f"    F0* = {star:.9f}")


# ---------------------------------------------------------------- fig 7
def fig7(f0=0.9):
    rng = np.random.default_rng(20260812)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    for name, (c, chk) in FIXED_FAMILY.items():
        m = hessian_exact(c, f0, chk, SUB)
        F00 = accepted_fidelity(c, f0, chk)
        pred, act = [], []
        for _ in range(80):
            th = rng.uniform(-0.15, 0.15, size=len(SUB))
            pred.append(float(th @ m @ th))
            act.append(F00 - accepted_fidelity(c, f0, chk, dict(zip(SUB, th.tolist()))))
        a1.plot(pred, act, "o", ms=2.6, alpha=0.75, color=C[name], label=name)
    lims = [0, max(a1.get_xlim()[1], a1.get_ylim()[1])]
    a1.plot(lims, lims, "--", color="0.4", lw=0.9)
    a1.set_xlabel(r"predicted $\bm{\theta}^{\top}M\bm{\theta}$".replace(r"\bm", ""))
    a1.set_ylabel("exact excess risk")
    a1.set_title("(a) closed form predicts excess risk")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

    scales = np.linspace(0.05, 0.35, 13)
    for name, (c, chk) in FIXED_FAMILY.items():
        m = hessian_exact(c, f0, chk, SUB)
        F00 = accepted_fidelity(c, f0, chk)
        r2s = []
        for s in scales:
            pred, act = [], []
            for _ in range(60):
                th = rng.uniform(-s, s, size=len(SUB))
                pred.append(float(th @ m @ th))
                act.append(F00 - accepted_fidelity(c, f0, chk,
                                                   dict(zip(SUB, th.tolist()))))
            pred, act = np.array(pred), np.array(act)
            r2s.append(1 - np.sum((act - pred) ** 2) / np.sum((act - act.mean()) ** 2))
        a2.plot(scales, r2s, "o-", ms=3, lw=1.3, color=C[name], label=name)
    a2.set_xlabel(r"coherent angle scale $|\theta|_{\max}$")
    a2.set_ylabel(r"$R^2$")
    a2.set_ylim(-0.35, 1.06)
    a2.axhline(1.0, color="0.75", lw=0.7, ls=":")
    a2.set_title("(b) regime of validity")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    save(fig, "fig7_validity.pdf")


def main():
    print("Regenerating figures (pdf.fonttype = 42, invariant labels)")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("done")


if __name__ == "__main__":
    main()
