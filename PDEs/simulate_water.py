"""
Simulate Shallow Water Waves by FTCS

Author: Dusan Zdravkovic
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1  # Length of water system in meters
J = 50  # Number of divisions in grid
dx = L / J  # Grid spacing
g = 9.81  # Gravitational constant [m/s^2]
eta_b = 0  # Bottom topography [m]
H = 0.01  # Average water height [m]
u_bound = 0  # u at the boundary (fixed walls)
h = 0.01  # Time-step
epsilon = h / 1000

# Time intervals for plots
t1 = 0.0
t2 = 1.0
t3 = 4.0
tend = t3 + epsilon

# Initialize arrays for u and eta
u = np.zeros(J + 1, float)  # Initial u (all 0)
u[0] = u[J] = u_bound

# Define the initial wave profile
A = 0.002
mu = 0.5
sigma = 0.05
x = np.linspace(0.0, L, J + 1)
bump = A * np.exp(-((x - mu) ** 2) / sigma**2)
eta = H + bump - np.mean(bump)  # To preserve average depth

# Create new arrays for the next time step
u_new = np.copy(u)
eta_new = np.copy(eta)

# Plotting setup
plt.figure(dpi=200)

# Main loop for the simulation
t = 0.0
while t < tend:
    # Update u and eta
    for j in range(1, J):
        term1 = 1 / 2 * (u[j + 1] ** 2) + g * eta[j + 1]
        term2 = 1 / 2 * (u[j - 1] ** 2) + g * eta[j - 1]
        u_new[j] = u[j] - h / (2 * dx) * (term1 - term2)
        term1 = u[j + 1] * (eta[j + 1] - eta_b)
        term2 = u[j - 1] * (eta[j - 1] - eta_b)
        eta_new[j] = eta[j] - h / (2 * dx) * (term1 - term2)

    # Boundary conditions for eta
    dfdx1 = u_new[1] * (eta[1] - eta_b) - u_new[0] * (eta[0] - eta_b)
    eta_new[0] = eta[0] - 2 * (h / (2 * dx)) * dfdx1
    dfdx2 = u_new[J] * (eta[J] - eta_b) - u_new[J - 1] * (eta[J - 1] - eta_b)
    eta_new[J] = eta[J] - 2 * (h / (2 * dx)) * dfdx2

    # Update eta and u
    eta = np.copy(eta_new)
    u = np.copy(u_new)
    t += h

    # Make plots at specified times
    if abs(t - h - t1) < epsilon or abs(t - t2) < epsilon or abs(t - t3) < epsilon:
        plt.plot(x, eta, label=f"$t={round(t, 1)}$ s")

# Finalize plot
plt.legend()
plt.xlabel("x [meter]")
plt.ylabel("$\eta$ [meter]")
plt.title("Shallow Water Wave")
plt.savefig("shallow_water_wave_simulation.png")
plt.show()
