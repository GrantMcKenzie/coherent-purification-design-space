"""Small-p depolarizing expansion for the fixed two-pair family.

Model: after the circuit Clifford, apply single-qubit depolarizing with
probability p to each of the four physical qubits (both sides, both pairs) --
the same insertion point as the coherent error. Each one-sided Pauli acts on
Bell labels as a translation, so the channel is a label-convolution and the
state stays Bell diagonal. Hence (Cor. 6 applies verbatim) there is no O(p*theta)
cross term and the combined leading-order response is additive:

  Fout(F0; theta, p) = Fout(F0,0) - p G(F0) - theta^T M(F0) theta
                        + O(p^2, p theta^2, ||theta||^4).

This script derives G(F0) in closed form for forward-Z, reverse-X, reverse-Y,
and verifies the additivity claim numerically.
"""
import sympy as sp
from fractions import Fraction
import numpy as np

F, p = sp.symbols('F p', nonnegative=True)
q = (1 - F) / 3
n = 2
N = 16

def pair_bits(v, k):
    return (v >> (2 * k)) & 1, (v >> (2 * k + 1)) & 1

def weight(v):
    return sum(1 for k in range(n) if (v >> (2 * k)) & 3)

def symp(u, v):
    s = 0
    for k in range(n):
        ux, uz = pair_bits(u, k); vx, vz = pair_bits(v, k)
        s ^= (ux & vz) ^ (uz & vx)
    return s

# --- circuit action on labels: bilateral CNOT control pair c -> target pair t
def S_cnot(c, t):
    def act(v):
        xbit = lambda k: (v >> (2 * k)) & 1
        zbit = lambda k: (v >> (2 * k + 1)) & 1
        x = [xbit(0), xbit(1)]; z = [zbit(0), zbit(1)]
        x[t] ^= x[c]
        z[c] ^= z[t]
        return x[0] | (z[0] << 1) | (x[1] << 2) | (z[1] << 3)
    return act

def label(pauli, k):
    x = 1 if pauli in 'XY' else 0
    z = 1 if pauli in 'ZY' else 0
    return (x << (2 * k)) | (z << (2 * k + 1))

def run(orientation, check):
    """exact Fout(F,p) for one fixed-family circuit with per-qubit depol p on all 4 qubits."""
    act = S_cnot(0, 1) if orientation == 'fwd' else S_cnot(1, 0)
    rho = [sp.S(0)] * N
    for v in range(N):
        rho[act(v)] += F ** (n - weight(v)) * q ** weight(v)
    # per-qubit depolarizing = label convolution, once per qubit (A and B of each pair)
    for k in range(2):
        for _ in range(2):          # A side and B side act identically on labels
            new = [sp.S(0)] * N
            for v in range(N):
                new[v] = (1 - p) * rho[v] + (p / 3) * sum(
                    rho[v ^ label(P, k)] for P in 'XYZ')
            rho = new
    e = label(check, 1)
    p0 = sum(rho[v] for v in range(N) if symp(e, v) == 0)
    N0 = sum(rho[v] for v in range(N) if symp(e, v) == 0 and (v & 3) == 0)
    return sp.cancel(N0 / p0)

D = 8 * F ** 2 - 4 * F + 5
results = {}
for name, ori, chk in [('forward-Z', 'fwd', 'Z'), ('reverse-X', 'rev', 'X'), ('reverse-Y', 'rev', 'Y')]:
    Fo = run(ori, chk)
    F0term = sp.cancel(Fo.subs(p, 0))
    G = sp.factor(-sp.diff(Fo, p).subs(p, 0))
    results[name] = (F0term, G)
    print(f"{name}:  Fout(F,0) = {F0term}")
    print(f"           G(F) = {G}")
    print(f"           G(0.9) = {float(G.subs(F, sp.Rational(9,10))):.6f}\n")

same = sp.simplify(results['forward-Z'][1] - results['reverse-X'][1]) == 0 and \
       sp.simplify(results['forward-Z'][1] - results['reverse-Y'][1]) == 0
print("G identical across the fixed family:", same)

# --- numerical additivity check: F(theta,p) vs Fout - p G - theta^2 M for forward-Z, XI generator
import numpy.linalg as la
def dm_check(F0=0.9, theta=0.05, prate=0.01):
    I2 = np.eye(2); X = np.array([[0,1],[1,0]], dtype=complex)
    Z = np.diag([1.,-1.]).astype(complex); Y = 1j*X@Z
    def kron(*ops):
        out = np.array([[1.+0j]])
        for o in ops: out = np.kron(out, o)
        return out
    # qubits A0 A1 B0 B1
    phip = np.zeros(4); phip[0]=phip[3]=1/np.sqrt(2)
    psip = np.zeros(4); psip[1]=psip[2]=1/np.sqrt(2)
    phim = np.zeros(4); phim[0]=1/np.sqrt(2); phim[3]=-1/np.sqrt(2)
    psim = np.zeros(4); psim[1]=1/np.sqrt(2); psim[2]=-1/np.sqrt(2)
    q0=(1-F0)/3
    rp = F0*np.outer(phip,phip)+q0*(np.outer(psip,psip)+np.outer(phim,phim)+np.outer(psim,psim))
    rho = np.kron(rp, rp)  # order A0 B0 A1 B1
    perm=[0,2,1,3]; axes=perm+[4+i for i in perm]
    rho = rho.reshape([2]*8).transpose(axes).reshape(16,16)  # A0 A1 B0 B1
    P0=np.diag([1.,0.]).astype(complex); P1=np.diag([0.,1.]).astype(complex)
    CN = kron(P0,I2,I2,I2)+kron(P1,X,I2,I2)      # A0->A1
    CNB= kron(I2,I2,P0,I2)+kron(I2,I2,P1,X)      # B0->B1
    U = CNB@CN
    rhoC = U@rho@U.conj().T
    # depolarizing Kraus per qubit
    def depol(rho, qb, pr):
        ops=[I2]*4
        out=(1-pr)*rho
        for P in (X,Y,Z):
            o=[I2]*4; o[qb]=P
            K=kron(*o)
            out = out + (pr/3)*K@rho@K.conj().T
        return out
    for qb in range(4):
        rhoC = depol(rhoC, qb, prate)
    from scipy.linalg import expm
    H = kron(X,I2,I2,I2)+kron(I2,I2,X,I2)        # common-mode XI
    E = expm(-1j*theta*H)
    r = E@rhoC@E.conj().T
    Pacc = (np.eye(16)+kron(I2,Z,I2,Z))/2
    Pi = kron(np.outer(phip[:2],[0,0]),I2,I2,I2)  # placeholder, build properly:
    # target on (A0,B0)
    T4 = np.outer(phip,phip).reshape(2,2,2,2)
    Pi = np.zeros((16,16),dtype=complex)
    for a in range(2):
        for b in range(2):
            for ap in range(2):
                for bp in range(2):
                    v=T4[a,b,ap,bp]
                    if v==0: continue
                    ka=np.zeros((2,1)); ka[a]=1; ba=np.zeros((1,2)); ba[0,ap]=1
                    kb=np.zeros((2,1)); kb[b]=1; bb=np.zeros((1,2)); bb[0,bp]=1
                    Pi += v*kron(ka@ba, I2, kb@bb, I2)
    num = np.trace(Pi@Pacc@r@Pacc).real
    den = np.trace(Pacc@r@Pacc).real
    Fsim = num/den
    Fo, G = results['forward-Z']
    mA = 4*(2*F0+1)*(4*F0-1)/(8*F0**2-4*F0+5)
    Fpred = float(Fo.subs(F, sp.Float(F0))) - prate*float(G.subs(F, sp.Float(F0))) - mA*theta**2
    return Fsim, Fpred

Fs, Fp = dm_check()
print(f"\nadditivity check (F0=0.9, theta=0.05, p=0.01): sim={Fs:.6f}  pred={Fp:.6f}  |diff|={abs(Fs-Fp):.2e}")
Fs2, Fp2 = dm_check(theta=0.02, prate=0.004)
print(f"additivity check (theta=0.02, p=0.004):        sim={Fs2:.6f}  pred={Fp2:.6f}  |diff|={abs(Fs2-Fp2):.2e}")
