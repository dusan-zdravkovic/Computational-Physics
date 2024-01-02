"""
Using the Shooting Method to Find Energies of a Hydrogen Atom and Plot Corresponding Wavefunctions

Author: Dusan Zdravkovic

Adapted from squarwell.py by Mark Newman
"""

# Imports
import matplotlib.pyplot as plt
from numpy import array, arange, pi
import scipy.constants as pc

# Constants
a = pc.physical_constants["Bohr radius"][0]
m = pc.m_e  # electron mass
hbar = pc.hbar
e = pc.e  # elementary charge
epsilon_0 = pc.epsilon_0


# Potential function
def V(x):
    return -(e**2) / (4 * pi * epsilon_0 * x)


# RHS function for numerical integration of ODE's
def f(r, x, E, l):
    R, S = r
    fR = S
    fS = (2 * m / hbar**2) * (V(x) - E) * R + l * (l + 1) * R / (x**2)
    return array([fR, fS], float)


# Calculate the wavefunction for a particular energy
def solve(E, l, h, r_inf):
    R, S = 0.0, 1.0
    r = array([R, S], float)
    for x in arange(h, r_inf, h):
        k1 = h * f(r, x, E, l)
        k2 = h * f(r + 0.5 * k1, x + 0.5 * h, E, l)
        k3 = h * f(r + 0.5 * k2, x + 0.5 * h, E, l)
        k4 = h * f(r + k3, x + h, E, l)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return r[0]


# Generate the wavefunction for a particular energy for plotting
def wavefunction(E, l, h, r_inf):
    R, S = 0.0, 1.0
    r = array([R, S], float)
    xpoints = arange(h, r_inf, h)
    Rpoints = []
    for x in xpoints:
        k1 = h * f(r, x, E, l)
        k2 = h * f(r + 0.5 * k1, x + 0.5 * h, E, l)
        k3 = h * f(r + 0.5 * k2, x + 0.5 * h, E, l)
        k4 = h * f(r + k3, x + h, E, l)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        Rpoints.append(r[0])
    return xpoints, Rpoints


# Main program to find the energy using the secant method
def E(n, l, r_inf, h):
    E1 = -15 * e / n**2
    E2 = -13 * e / n**2
    R2 = solve(E1, l, h, r_inf)
    target = e / 2000
    while abs(E1 - E2) > target:
        R1, R2 = R2, solve(E2, l, h, r_inf)
        E1, E2 = E2, E2 - R2 * (E2 - E1) / (R2 - R1)
    print(f"For n = {n}, and l = {l}, the eigen energy is {E2/e} eV")

    rpoints, Rpoints = wavefunction(E2, l, h, r_inf)
    N = len(Rpoints)
    integral = Rpoints[0] + Rpoints[-1]
    for k in range(1, N, 2):
        integral += 4 * Rpoints[k]
    for k in range(2, N, 2):
        integral += 2 * Rpoints[k]
    integral *= h / 3
    Rpoints /= integral

    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(rpoints / a, Rpoints, label=f"$n = {n}, \ell = {l}$")
    plt.title(f"Normalized Radial Wavefunction $R(r)$ vs. $r$ for n = {n}, l = {l}")
    plt.xlabel("$r$ ($a_0$)")
    plt.ylabel("Normalized $R(r)$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"wavefunction_n{n}_l{l}.png")
    plt.show()


# Setting parameters for each simulation
# Simulation 1
r_inf = 20 * a  # very big reference r
h = 0.0005 * a  # step size
E1 = E(1, 0, r_inf, h)

# Simulation 2
r_inf = 40 * a
h = 0.001 * a
E20 = E(2, 0, r_inf, h)

# Simulation 3
r_inf = 30 * a
h = 0.002 * a
E21 = E(2, 1, r_inf, h)
