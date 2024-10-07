"""
Calculating Quantum Mechanical Observables 

Author: Dusan Zdravkovic
Purpose: Analysis of quantum mechanical observables using Hermite polynomials and Gaussian quadrature.
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxwab  # Importing Gaussian quadrature


# Define the function H(n,x) recursively as per the Hermite polynomials
def H(n: int, x):
    if n == 0:
        return 1 + 0 * x
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * H(n - 1, x) - 2 * (n - 1) * H(n - 2, x)


# Define the function psi(n, x)
def psi(n, x):
    n_fac = np.math.factorial(n)
    factor0 = 1 / (np.sqrt(float(2**n) * float(n_fac) * np.sqrt(np.pi)))
    factor1 = np.exp(-(x**2) / 2)
    factor2 = H(n, x)
    return factor0 * factor1 * factor2


# Plotting Wave Function
x = np.arange(-4, 4, 0.05)
plt.figure(dpi=200)
plt.title("The wave function $\psi_n(x)$ vs. $x$")
plt.xlabel("$x$")
plt.ylabel("$\psi_n(x)$")

for ns in range(4):
    plt.plot(x, psi(ns, x), label=f"$n = {ns}$")

plt.legend()
plt.grid()
plt.savefig("wave_function_psi_n_x.png")

# Plotting Wave Function
x = np.arange(-10, 10, 0.05)
plt.figure(dpi=200)
plt.title("The wave function $\psi_{30}(x)$ vs. $x$")
plt.xlabel("$x$")
plt.ylabel("$\psi_{30}(x)$")

plt.plot(x, psi(30, x), label="$n = 30$")
plt.legend()
plt.grid()
plt.savefig("wave_function_psi_30_x.png")


# Analysis
# Define the derivative of psi
def ddx_psi(n, x):
    n_fac = np.math.factorial(n)
    factor0 = 1 / (np.sqrt(float(2**n) * float(n_fac) * np.sqrt(np.pi)))
    factor1 = np.exp(-(x**2) / 2)
    if n == 0:
        factor2 = -x * H(n, x)
    else:
        factor2 = -x * H(n, x) + 2 * n * H(n - 1, x)
    return factor0 * factor1 * factor2


# Define the mean-squared x function
def ms_x(n):
    x_pts, w = gaussxwab(100, -np.pi / 2, np.pi / 2)
    u = np.tan(x_pts)
    integrand = u**2 * abs(psi(n, u)) ** 2 / np.cos(x_pts) ** 2
    return sum(w * integrand)


# Define the momentum uncertainty
def ms_p(n):
    x_pts, w = gaussxwab(100, -np.pi / 2, np.pi / 2)
    u = np.tan(x_pts)
    integrand = abs(ddx_psi(n, u)) ** 2 / np.cos(x_pts) ** 2
    return sum(w * integrand)


# Define the energy function
def E(msx, msp):
    return 1 / 2 * (msx + msp)


print("For part 3(c):")
for ns in range(16):
    msx = ms_x(ns)
    msp = ms_p(ns)
    energy = E(msx, msp)
    print(f"For n={ns}:")
    print(f"<x^2> = {msx}, sqrt(<x^2>) = {np.sqrt(msx)}")
    print(f"<p^2> = {msp}, sqrt(<p^2>) = {np.sqrt(msp)}")
    print(f"E = {energy}\n")
