# gen_plots.py: Trotter step-count study (Figure 2 of the report).
# Pure NumPy/SciPy: exact evolution via scipy.linalg.expm, ideal Trotter
# evolution by multiplying the layer unitaries (no Qiskit, no shot noise).
# Panel (a): magnetisation for n = 1, 2, 5, 10. Panel (b): max error vs n.
# Requires: numpy, scipy, matplotlib
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

J, h, N = 1.0, 1.0, 3
I2 = np.eye(2)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)

def kron_list(ops):
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return res

H_ZZ = -J * sum(kron_list([Z if k in (i, i + 1) else I2 for k in range(N)])
                for i in range(N - 1))
H_X = -h * sum(kron_list([X if k == i else I2 for k in range(N)])
               for i in range(N))
H = H_ZZ + H_X

psi0 = np.zeros(2**N, dtype=complex); psi0[0] = 1.0           # |000>
M_Z = sum(kron_list([Z if k == i else I2 for k in range(N)])
          for i in range(N)) / N

def mag(psi):
    return float(np.real(psi.conj() @ M_Z @ psi))

def exact_mag(t):
    return mag(expm(-1j * H * t) @ psi0)

def trotter_mag(t, n):
    dt = t / n
    step = expm(-1j * H_X * dt) @ expm(-1j * H_ZZ * dt)   # ZZ layer first, then X
    return mag(np.linalg.matrix_power(step, n) @ psi0)

# ---- Figure 2 ----
t_fine = np.linspace(0, 2.0, 200)
t_pts = np.linspace(0, 2.0, 20)
exact_fine = [exact_mag(t) for t in t_fine]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)

ax1.plot(t_fine, exact_fine, 'k-', lw=2.5, label='Exact')
styles = {1: ('s', '#e41a1c'), 2: ('^', '#ff7f00'), 5: ('o', '#377eb8'), 10: ('D', '#4daf4a')}
for n, (mk, col) in styles.items():
    ax1.plot(t_pts, [trotter_mag(t, n) for t in t_pts], '--', marker=mk,
             color=col, ms=5, label=f'Trotter $n={n}$')
ax1.set_xlim(0, 2.0); ax1.set_ylim(-0.25, 1.05)
ax1.set_xlabel('Time $t$'); ax1.set_ylabel(r'$\langle M_Z(t)\rangle$')
ax1.set_title('(a) Ideal Trotter: Effect of Step Count $n$')
ax1.grid(True, ls='--', alpha=0.5); ax1.legend()

ns = np.arange(1, 21)
t_err = np.linspace(0.1, 2.0, 40)
exact_err = np.array([exact_mag(t) for t in t_err])
errs = [np.max(np.abs(np.array([trotter_mag(t, n) for t in t_err]) - exact_err))
        for n in ns]
ax2.loglog(ns, errs, 'k-o', lw=2,
           label=r'Trotter error $\max_t|\langle M_Z\rangle_{\rm Trotter}-\langle M_Z\rangle_{\rm exact}|$')
ax2.loglog(ns, 0.3 / ns, 'r--', label=r'$\propto 1/n$ reference')
ax2.set_xlim(1, 20)
ax2.set_xlabel('Number of Trotter steps $n$'); ax2.set_ylabel('Maximum magnetisation error')
ax2.set_title('(b) Trotter Error Scaling with $n$')
ax2.grid(True, which='both', ls='--', alpha=0.5); ax2.legend()

plt.tight_layout()
plt.savefig('fig_trotter_steps.png', bbox_inches='tight')
print('n, max error:', [(int(n), round(e, 4)) for n, e in zip(ns, errs)][:6])
