"""
Asymmetric Quantum Well Simulation

Author: Dusan Zdravkovic
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxwab

# Constants
a = 10 * 1.6022e-19
h_bar = 1.05457e-34
L = 5e-10
M = 9.1094e-31


# Hamiltonian Matrix Entry Function
def H_entry(m, n):
    if m != n and m % 2 == n % 2:
        return 0
    elif m != n and m % 2 != n % 2:
        return -(8 * a * m * n) / (np.pi**2 * (m**2 - n**2) ** 2)
    else:
        return a / 2 + (np.pi * h_bar * n) ** 2 / (2 * M * L**2)


# Matrix Sizes
sizes = [10, 100]
eigenvalues = {}

for mmax in sizes:
    nmax = mmax
    H = np.zeros((mmax, nmax))
    for m in range(1, mmax + 1):
        for n in range(1, nmax + 1):
            H[m - 1, n - 1] = H_entry(m, n)
    E, V = np.linalg.eigh(H)
    eigenvalues[mmax] = E / 1.6022e-19

# Print Eigenvalues
for size, values in eigenvalues.items():
    print(f"The first 10 eigenvalues for a matrix of size {size}x{size} in [eV]:")
    print(values[:10] if size == 100 else values)
    print()


# Wavefunction Functions
def psi(x, V_psi):
    return sum(V_psi[i] * np.sin(np.pi * (i + 1) * x / L) for i in range(len(V_psi)))


def psi_norm(x, V_psi):
    x_pts, w = gaussxwab(100, 0, L)
    A = sum(w * abs(psi(x_pts, V_psi)) ** 2)
    return psi(x, V_psi) / np.sqrt(A)


# Plotting
x_arr = np.linspace(0, L, 1000)
pdfs = [abs(psi_norm(x_arr, V[:, i])) ** 2 for i in range(3)]

plt.figure(figsize=(8, 6), dpi=200)
plt.title("Probability Density in a Quantum Well")
for i, pdf in enumerate(pdfs):
    plt.plot(x_arr, pdf, label=f"State {i}")
plt.xlabel("Position x [m]")
plt.ylabel("Probability Density $|\psi(x)|^2$")
plt.legend()
plt.tight_layout()
plt.savefig("Quantum_Well_Probability_Density.png")
plt.show()
