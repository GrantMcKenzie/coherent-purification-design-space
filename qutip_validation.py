"""QuTiP validation under a realistic noise model.

Question: the manuscript's leading-order prediction R_excess ~ theta^T M theta
was validated against exact unitary simulation.  Does it survive a realistic
background -- amplitude damping (T1) and pure dephasing (T2) on every qubit --
as an experiment would have?

Model (forward-Z, two pairs, qubits A0 A1 B0 B1):
  1. Werner input at F0.
  2. Bilateral CNOT.
  3. Coherent over-rotation exp(-i theta_g H_g) (common-mode).
  4. Per-qubit amplitude damping (gamma) and dephasing (gamma_phi) Kraus maps.
  5. Postselect on Z_A1 Z_B1 coincidence; read out kept-pair fidelity.

Test: excess risk over the *noisy* baseline, F(0,gamma)-F(theta,gamma),
against theta^T M theta with the closed-form M of the paper -- i.e. is the
coherent susceptibility still the right compass when T1/T2 noise is present?
"""
import numpy as np
import qutip as qt

F0 = 0.9
q = (1 - F0) / 3
D = 8 * F0 ** 2 - 4 * F0 + 5
mA = 4 * (2 * F0 + 1) * (4 * F0 - 1) / D      # M_XI = M_XZ
mB = 4 * (4 * F0 - 1) ** 2 / D                # M_ZZ

# Bell states on one pair
b00 = qt.bell_state('00'); b01 = qt.bell_state('01')
b10 = qt.bell_state('10'); b11 = qt.bell_state('11')
rho_pair = (F0 * b00.proj() + q * (b01.proj() + b10.proj() + b11.proj()))

# qubit order: A0 A1 B0 B1  (pair k = (A_k, B_k))
rho0 = qt.tensor(rho_pair, rho_pair)               # (A0 B0) (A1 B1)
rho0 = rho0.permute([0, 2, 1, 3])                  # -> A0 A1 B0 B1

from qutip import gates
CN2 = gates.cnot()                                  # 2-qubit CNOT (control 0, target 1)
CN_A = qt.expand_operator(CN2, dims=[2, 2, 2, 2], targets=[0, 1])
CN_B = qt.expand_operator(CN2, dims=[2, 2, 2, 2], targets=[2, 3])
U = CN_A * CN_B
rhoC = U * rho0 * U.dag()

sx, sz, I = qt.sigmax(), qt.sigmaz(), qt.qeye(2)
def onq(op, i):
    ops = [I] * 4
    ops[i] = op
    return qt.tensor(ops)

Pacc = (qt.tensor([I] * 4) + onq(sz, 1) * onq(sz, 3)) / 2
Pi_t = qt.tensor(b00.proj().permute([0, 1]), I, I)   # placeholder; build below
# target = |Phi+><Phi+| on (A0,B0) x id on (A1,B1): build with permutation
Pi_pair = b00.proj()
Pi_t = qt.tensor(Pi_pair, qt.qeye([2, 2]))           # (A0 B0)(A1 B1)
Pi_t = Pi_t.permute([0, 2, 1, 3])                    # -> A0 A1 B0 B1

def kraus_amp(gamma):
    K0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = qt.Qobj([[0, np.sqrt(gamma)], [0, 0]])
    return [K0, K1]

def kraus_deph(gphi):
    K0 = np.sqrt(1 - gphi / 2) * qt.qeye(2)
    K1 = np.sqrt(gphi / 2) * qt.sigmaz()
    return [K0, K1]

def apply_1q_kraus(rho, kraus, i):
    return sum(onq(K, i) * rho * onq(K, i).dag() for K in kraus)

def fout(thetas, gamma, gphi):
    """thetas: dict generator-name -> angle, generators XI, XZ, ZZ common-mode."""
    H = 0
    gens = {'XI': (sx, None), 'XZ': (sx, sz), 'ZZ': (sz, sz)}
    for gname, th in thetas.items():
        P0, P1 = gens[gname]
        # common-mode: same Pauli pattern on Alice side (A0,A1) and Bob side (B0,B1)
        opA = onq(P0, 0) * (onq(P1, 1) if P1 is not None else qt.tensor([I]*4))
        opB = onq(P0, 2) * (onq(P1, 3) if P1 is not None else qt.tensor([I]*4))
        H = H + th * (opA + opB)
    r = rhoC if H == 0 else ((-1j * H).expm() * rhoC * (1j * H).expm())
    for i in range(4):
        if gamma:
            r = apply_1q_kraus(r, kraus_amp(gamma), i)
        if gphi:
            r = apply_1q_kraus(r, kraus_deph(gphi), i)
    num = (Pi_t * Pacc * r * Pacc).tr().real
    den = (Pacc * r * Pacc).tr().real
    return num / den

print(f"noiseless baseline check: Fout = {fout({}, 0, 0):.10f} "
      f"(closed form {(10*F0**2-2*F0+1)/D:.10f})")

print(f"\n{'gamma':>6} {'gphi':>6} {'theta_XI':>9} {'theta_ZZ':>9} "
      f"{'measured excess':>16} {'theta^T M theta':>16} {'ratio':>7}")
for gamma, gphi in [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01), (0.02, 0.02)]:
    base = fout({}, gamma, gphi)
    for thXI, thZZ in [(0.05, 0.0), (0.0, 0.05), (0.04, 0.03)]:
        f = fout({'XI': thXI, 'ZZ': thZZ}, gamma, gphi)
        excess = base - f
        pred = mA * thXI ** 2 + mB * thZZ ** 2
        print(f"{gamma:6.2f} {gphi:6.2f} {thXI:9.2f} {thZZ:9.2f} "
              f"{excess:16.6f} {pred:16.6f} {excess/pred:7.3f}")
