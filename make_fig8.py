"""fig8_fourpair.pdf -- the improvement window widens at four pairs.
Layout rules: legend below axes, no annotation over data, constrained_layout,
pdf.fonttype=42, crossovers as x-axis ticks."""
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

F = np.linspace(0.5, 1.0, 800)
D = 8 * F**2 - 4 * F + 5
D3 = 16 * F**2 - 14 * F + 7
lamA = 4 * (2 * F + 1) * (4 * F - 1) / D
lamII = 12 * F * (4 * F - 1) / D3
lamIIp = 4 * (2 * F + 1) * (4 * F - 1) * (10 * F**2 - 2 * F + 1) / ((8 * F**2 - F + 2) * D3)

Fstar3 = (3 * np.sqrt(2) - 2) / 4          # 0.560660...
Fstar4 = np.roots([16, -8, 4, -3])
Fstar4 = float([r.real for r in Fstar4 if abs(r.imag) < 1e-9][0])   # 0.602047...

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)

ax.plot(F, lamA, color='#1f4e79', lw=1.8, label=r'$\lambda_A$ (two-pair bound)')
ax.plot(F, lamII, color='#b45f06', lw=1.6, ls='--', label=r'$\lambda^{(\mathrm{II})}_{\max}$ (three pairs)')
ax.plot(F, lamIIp, color='#38761d', lw=1.8, ls='-.', label=r"$\lambda^{(\mathrm{II}')}_{\max}$ (four pairs)")
ax.axvspan(0.5, Fstar4, color='#38761d', alpha=0.10)
ax.axvspan(0.5, Fstar3, color='#b45f06', alpha=0.12)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(1.4, 4.1)
ax.set_xticks([0.5, Fstar3, Fstar4, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(['0.5', r'$F_0^{\star}$', r'$F_0^{\star\star}$', '0.7', '0.8', '0.9', '1.0'])
ax.set_xlabel(r'input fidelity $F_0$')
ax.set_ylabel(r'$\lambda_{\max}(M)$')
ax.set_title('(a) worst-direction susceptibility', fontsize=10)

d3 = lamII - lamA
d4 = lamIIp - lamA
ax2.axhline(0, color='0.6', lw=0.8)
ax2.plot(F, d3, color='#b45f06', lw=1.6, ls='--', label=r'$\lambda^{(\mathrm{II})}_{\max}-\lambda_A$')
ax2.plot(F, d4, color='#38761d', lw=1.8, ls='-.', label=r"$\lambda^{(\mathrm{II}')}_{\max}-\lambda_A$")
ax2.set_xlim(0.5, 1.0)
ax2.set_xticks([0.5, Fstar3, Fstar4, 0.7, 0.8, 0.9, 1.0])
ax2.set_xticklabels(['0.5', r'$F_0^{\star}$', r'$F_0^{\star\star}$', '0.7', '0.8', '0.9', '1.0'])
ax2.set_xlabel(r'input fidelity $F_0$')
ax2.set_ylabel(r'difference from $\lambda_A$')
ax2.set_title('(b) the improvement window widens', fontsize=10)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, frameon=False, fontsize=8.5)

fig.savefig('/home/claude/work/figures/fig8_fourpair.pdf', bbox_inches='tight')
print("saved fig8_fourpair.pdf")
print(f"F0* (3-pair) = {Fstar3:.9f}   F0** (4-pair) = {Fstar4:.9f}")
