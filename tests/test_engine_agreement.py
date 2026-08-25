"""
Cross-check the two independent master-formula implementations against each other.

master.py       -- tuple labels, the evaluator wired into verify_threepair.py
general_n.py    -- int labels, the construction+evaluation engine used at n=4

They share no code. This test asserts they produce identical susceptibilities on
the forward-Z baseline, so the "independent engines" claim in the README is
machine-enforced rather than asserted by hand.
"""
from fractions import Fraction

import master as M
import general_n as G


def test_engines_agree_forward_z():
    n = 2
    F = Fraction(9, 10)

    S = G.cnot_bilateral_S(0, 1, n)
    row = G.Row(S, ('Z',), n)
    g_named = {G.label_name(p, n): v for p, v in row.M_diag(F).items()}

    pops = M.werner_pops(n, F, S_inv=row.Sinv.tolist())
    check = [M.from_name('IZ')]
    m_diag, p0, fout = M.diagonal(pops, M.accepted(n, check), check, n)
    m_named = {M.to_name(p): v for p, v in m_diag.items()}

    assert row.fout(F) == fout
    assert set(g_named) == set(m_named)
    for k in g_named:
        assert g_named[k] == m_named[k], (k, g_named[k], m_named[k])


if __name__ == "__main__":
    test_engines_agree_forward_z()
    print("master.py and general_n.py agree exactly on all 15 generators.")
