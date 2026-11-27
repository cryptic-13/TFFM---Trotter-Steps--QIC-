# Trotter Simulation of the Transverse-Field Ising Model

This project simulates the time evolution of a 3-qubit, one-dimensional Transverse-Field Ising Model (TFIM) using a first-order Suzuki–Trotter decomposition. The quantum circuits are written in Qiskit and run on an ideal simulator and on IBM's calibrated `FakeManilaV2` noise model. Results are checked against exact evolution computed by matrix exponentiation.

**Authors:** Arnav Ajit Godbole, Arindam Jakhwal, Arushi Kumar, Aryan Tripathi
**Course:** BITS F386 Quantum Information and Computation, BITS Pilani K K Birla Goa Campus

## The model

$$H = -J\sum_{i=0}^{N-2} \sigma_z^{(i)}\sigma_z^{(i+1)} \;-\; h\sum_{i=0}^{N-1} \sigma_x^{(i)}, \qquad N = 3,\; J = h = 1$$

This uses open boundary conditions and the initial state $|000\rangle$. With $J = h$ the system sits at the quantum critical point $h/J = 1$. The observable is the average magnetisation $\langle M_Z(t)\rangle = \frac{1}{N}\sum_i \langle \sigma_z^{(i)} \rangle$.

One Trotter step $e^{-iH_{ZZ}\Delta t}\,e^{-iH_X\Delta t}$ is built from:
- a **ZZ layer**: for each bond, `CNOT – Rz(-2JΔt) – CNOT`
- an **X layer**: `Rx(-2hΔt)` on every qubit

That is 4 CNOTs, 2 Rz and 3 Rx gates per step. The first-order Trotter error scales as $O(t^2/n)$.

## Repository layout

```
report/   TFIM_Trotter_Report.pdf    full report (theory, derivations, results, error analysis)
code/     tfim_simulation.py         Hamiltonian, exact evolution, Trotter circuit, ideal + noisy runs -> fig_main.png
          gen_plots.py               Trotter step-count study (n = 1, 2, 5, 10; error vs n) -> fig_trotter_steps.png
figures/  fig_main.png               Figure 1 of the report
          fig_trotter_steps.png      Figure 2 of the report
```

## Running the code

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd code
python tfim_simulation.py          # takes a minute or two; writes fig_main.png
python gen_plots.py                # a few seconds; writes fig_trotter_steps.png
```

`tfim_simulation.py` samples 2048 shots per circuit, so the ideal and noisy points change a little from run to run. `gen_plots.py` is deterministic, because it multiplies the Trotter unitaries directly, and it reproduces Figure 2 exactly.

## Results in brief

- With n = 5 Trotter steps, the ideal circuit tracks the exact curve closely. The noisy FakeManilaV2 run decays faster because of gate errors, decoherence and readout errors.
- n = 1 fails by about t = 0.5 and n = 2 still deviates a lot. n = 5 and n = 10 match the exact evolution well.
- The maximum magnetisation error falls at least as fast as 1/n.
- On real hardware, adding steps increases gate noise. The report discusses this trade-off and why n = 5 is a practical choice.
