"""Unit tests for the core algebra and both engines."""

import numpy as np
import pytest

from finite_rank_purification.circuits import (
    FIXED_FAMILY, acceptance_projector, accepted_branch, bilateral,
    generator, rho_after_circuit,
)
from finite_rank_purification.classification import (
    arf, atom_label, atom_name, base_form, enumerate_atoms, is_purifying,
    rho_from_form,
)
from finite_rank_purification.clifford import (
    CNOT01, enumerate_clifford2, symplectic_image,
)
from finite_rank_purification.paulis import (
    D, TWO_QUBIT_GENERATORS, bell_basis, two_pair_bell_basis, werner,
)
from finite_rank_purification.susceptibility import (
    hessian_exact, hessian_findiff, hessian_from_rho, syndrome,
)

F0 = 0.9


@pytest.fixture(scope="session")
def cliffords():
    return enumerate_clifford2()


@pytest.fixture(scope="session")
def atoms(cliffords):
    return enumerate_atoms(cliffords)


def test_bell_basis_orthonormal():
    vecs, _ = bell_basis()
    assert np.allclose(vecs @ vecs.conj().T, np.eye(4))


def test_two_pair_bell_basis_orthonormal():
    vecs, labels = two_pair_bell_basis()
    assert len(labels) == 16
    assert np.allclose(vecs @ vecs.conj().T, np.eye(16))


def test_werner_is_a_state():
    rho = werner(F0)
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho).real, 1.0)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_werner_fidelity():
    vecs, _ = bell_basis()
    phi = vecs[0]
    assert np.isclose((phi.conj() @ werner(F0) @ phi).real, F0)


def test_generator_is_hermitian():
    for g in TWO_QUBIT_GENERATORS:
        h = generator(g)
        assert np.allclose(h, h.conj().T)


def test_common_mode_generator_squares_correctly():
    # H_g^2 = 2I + 2 P_A P_B != I  (Sec. 6)
    h = generator("XZ")
    assert not np.allclose(h @ h, np.eye(16))


def test_acceptance_projectors_are_projectors():
    for chk in "ZXY":
        p = acceptance_projector(chk)
        assert np.allclose(p @ p, p)
        assert np.isclose(np.trace(p).real, 8.0)


def test_bilateral_preserves_double_phi_plus():
    vecs, labels = two_pair_bell_basis()
    idx = labels.index((0, 0, 0, 0))
    for c in (CNOT01,):
        out = bilateral(c) @ vecs[idx]
        assert np.isclose(abs(np.vdot(vecs[idx], out)), 1.0)


def test_clifford_group_order(cliffords):
    assert len(cliffords) == 11520


def test_symplectic_images_number_720(cliffords):
    assert len({symplectic_image(c).tobytes() for c in cliffords}) == 720


def test_rho_c_is_bell_diagonal():
    vecs, _ = two_pair_bell_basis()
    rho = rho_after_circuit(CNOT01, F0)
    in_bell = vecs.conj() @ rho @ vecs.T
    off = in_bell - np.diag(np.diag(in_bell))
    assert np.abs(off).max() < 1e-12


def test_baseline_closed_forms():
    p0, fout = accepted_branch(rho_after_circuit(CNOT01, F0), "Z")
    assert np.isclose(p0, D(F0) / 9)
    assert np.isclose(fout, (10 * F0**2 - 2 * F0 + 1) / D(F0))


def test_ten_atoms_multiplicity_1152(atoms):
    assert len(atoms) == 10
    assert {a["multiplicity"] for a in atoms.values()} == {1152}


def test_input_form_is_arf_zero():
    assert arf(base_form) == 0


def test_engines_agree_on_rho(atoms):
    for a in atoms.values():
        r1 = rho_after_circuit(a["representative"], F0)
        r2 = rho_from_form(a["S"], F0)
        assert np.abs(r1 - r2).max() < 1e-12


def test_hessian_exact_matches_finite_difference():
    sub = ("XI", "XZ", "ZZ")
    for c, chk in FIXED_FAMILY.values():
        a = hessian_exact(c, F0, chk, sub)
        b = hessian_findiff(c, F0, chk, sub)
        assert np.abs(a - b).max() < 1e-5


def test_hessian_is_diagonal_everywhere(atoms):
    for a in atoms.values():
        for chk in "ZXY":
            m = hessian_from_rho(rho_from_form(a["S"], F0), chk)
            assert np.abs(m - np.diag(np.diag(m))).max() == 0.0


def test_syndrome_block_structure():
    for c, chk in FIXED_FAMILY.values():
        m = hessian_exact(c, F0, chk)
        for i, gi in enumerate(TWO_QUBIT_GENERATORS):
            for j, gj in enumerate(TWO_QUBIT_GENERATORS):
                if syndrome(gi, chk) != syndrome(gj, chk):
                    assert abs(m[i, j]) < 1e-12


def test_sensitivity_vanishes_at_quarter():
    assert np.abs(hessian_exact(CNOT01, 0.25, "Z")).max() < 1e-12


def test_forward_and_reverse_atoms():
    from finite_rank_purification.circuits import C_FORWARD, C_REVERSE
    assert atom_name(atom_label(symplectic_image(C_FORWARD))) == "q[Z|X]"
    assert atom_name(atom_label(symplectic_image(C_REVERSE))) == "q[X|Z]"


def test_reverse_cnot_is_fixed_point_under_z():
    from finite_rank_purification.circuits import C_REVERSE
    label = atom_label(symplectic_image(C_REVERSE))
    assert not is_purifying(label, "Z")
    assert is_purifying(label, "X")
    assert is_purifying(label, "Y")


def test_dichotomy_counts(atoms):
    for chk in "ZXY":
        assert sum(1 for a in atoms.values() if is_purifying(a["label"], chk)) == 6
