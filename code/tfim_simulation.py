# TFIM Trotter simulation (Listings 1-4 of the report, Section 6).
# Builds the 3-qubit TFIM, computes exact evolution, and runs the Trotter
# circuit on an ideal simulator and on the FakeManilaV2 noise model.
#
# Requires: numpy, scipy, matplotlib, qiskit, qiskit-aer, qiskit-ibm-runtime

# ---------------- Listing 1. Hamiltonian setup (J = 1.0, h = 1.0, N = 3) ----------------
import numpy as np
from scipy.linalg import expm

J, h, N = 1.0, 1.0, 3
times = np.linspace(0, 2.0, 15)        # 15 time points
num_trotter_steps = 5

I2 = np.eye(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

def kron_list(ops):
    res = ops[0]
    for op in ops[1:]: res = np.kron(res, op)
    return res

# H = -J (Z0Z1 + Z1Z2) - h (X0 + X1 + X2)
H_ZZ = -J * (kron_list([Z, Z, I2]) + kron_list([I2, Z, Z]))
H_X = -h * (kron_list([X, I2, I2]) + kron_list([I2, X, I2])
            + kron_list([I2, I2, X]))
H = H_ZZ + H_X

# ---------------- Listing 2. Exact evolution via matrix exponentiation ----------------
initial_state = np.zeros(2**N); initial_state[0] = 1.0
Z_ops = [kron_list([Z if k == i else I2 for k in range(N)])
         for i in range(N)]

def exact_mag(t):
    U   = expm(-1j * H * t)
    psi = U @ initial_state
    return sum(np.real(psi.conj() @ Zi @ psi) for Zi in Z_ops) / N

# ---------------- Listing 3. Parameterised Trotter circuit ----------------
from qiskit import QuantumCircuit

def build_trotter_circuit(t, steps):
    """Build a QuantumCircuit with `steps` Trotter steps for time t."""
    dt = t / steps
    qc = QuantumCircuit(N)
    # |000> is the hardware default; no state-preparation gates needed.

    for _ in range(steps):
        # ZZ layer: CNOT - Rz(2*theta) - CNOT, where theta = -J*dt
        for i in range(N - 1):
            qc.cx(i, i + 1)
            qc.rz(-2 * J * dt, i + 1)
            qc.cx(i, i + 1)
        # X-field layer: Rx(-2*h*dt) per qubit
        for i in range(N):
            qc.rx(-2 * h * dt, i)

    qc.measure_all()
    return qc

# ---------------- Listing 4. Ideal and noisy simulation (Qiskit Runtime V2 Sampler) ----------------
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager)

ideal_backend = AerSimulator()
noisy_backend = AerSimulator.from_backend(FakeManilaV2())
SHOTS = 2048

def get_mag(counts):
    """Convert bitstring counts to average <M_Z>."""
    avg_z = 0
    for state, count in counts.items():
        z_val = state.count('0') - state.count('1')
        avg_z += z_val * (count / SHOTS)
    return avg_z / N

ideal_results, hardware_results = [], []

for t in times:
    if t == 0:
        ideal_results.append(1.0)
        hardware_results.append(1.0)
        continue

    qc = build_trotter_circuit(t, num_trotter_steps)

    # Ideal (noiseless) simulation
    pm_i = generate_preset_pass_manager(
        backend=ideal_backend, optimization_level=1)
    res_i = Sampler(mode=ideal_backend).run(
        [pm_i.run(qc)], shots=SHOTS).result()
    ideal_results.append(
        get_mag(res_i[0].data.meas.get_counts()))

    # Noisy simulation (FakeManilaV2 calibrated noise model)
    pm_n = generate_preset_pass_manager(
        backend=noisy_backend, optimization_level=1)
    res_n = Sampler(mode=noisy_backend).run(
        [pm_n.run(qc)], shots=SHOTS).result()
    hardware_results.append(
        get_mag(res_n[0].data.meas.get_counts()))

# ---------------- Plot: exact vs ideal Trotter vs noisy (Figure 1 style) ----------------
import matplotlib.pyplot as plt

exact_results = [exact_mag(t) for t in times]

plt.figure(figsize=(10, 5))
plt.plot(times, exact_results, 'k-', label='Exact evolution')
plt.plot(times, ideal_results, 'bo--', label='Ideal Trotter circuit')
plt.plot(times, hardware_results, 'rd--', label='Noisy simulation (FakeManilaV2)')
plt.xlabel('Time (t)')
plt.ylabel(r'Average magnetisation $\langle M_Z \rangle$')
plt.title(f'3-Qubit TFIM: Trotter (n = {num_trotter_steps}) vs Exact')
plt.grid(True, ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('fig_main.png', dpi=200)
print('Saved fig_main.png')
