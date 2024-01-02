"""
Simulate an Electric Potential Using the Gauss-Seidel Method

Author: Dusan Zdravkovic
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time

M = 100  # Grid squares on a side
target = 1e-6  # Precision

# Initialize the potential grid
phi = np.zeros([M + 1, M + 1], float)
cm2 = round(M * 2 / 10)  # Index of 2 cm
cm8 = round(M * 8 / 10)  # Index of 8 cm

# Set initial conditions for the metal plates
phi[cm2 : cm8 + 1, cm2] = 1.0
phi[cm2 : cm8 + 1, cm8] = -1.0

# Gauss-Seidel without overrelaxation
start = time()
delta = 1.0
while delta > target:
    errors = []
    for i in range(1, M):
        for j in range(1, M):
            if not ((j == cm2 or j == cm8) and cm2 <= i <= cm8):
                phi_old_ij = phi[i, j]
                phi[i, j] = (
                    phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]
                ) / 4
                errors.append(abs(phi_old_ij - phi[i, j]))
    delta = max(errors)
end = time()
print("Simulation time for no overrelaxation: {:.3f} s".format(end - start))

# Plotting potential
x = np.linspace(0, 10, M + 1)
y = np.linspace(0, 10, M + 1)
plt.figure(figsize=(6, 4), dpi=200)
plt.contourf(x, y, phi, levels=10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Potential $\phi$ [V]")
plt.colorbar()
plt.grid()
plt.savefig("potential_no_overrelaxation.png")
plt.show()

# Plotting electric field lines
X, Y = np.meshgrid(x, y)
Ey, Ex = np.gradient(-phi, y, x)
plt.figure(dpi=200)
strm = plt.streamplot(X, Y, Ex, Ey, color=phi, linewidth=2, cmap="autumn")
plt.colorbar(strm.lines).set_label("Potential $V$")
plt.title("Electric Field Lines")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid()
plt.axis("equal")
plt.tight_layout()
plt.savefig("electric_field_lines_no_overrelaxation.png")
plt.show()

# Gauss-Seidel with overrelaxation
phi = np.zeros([M + 1, M + 1], float)  # Reset potential grid
phi[cm2 : cm8 + 1, cm2] = 1.0
phi[cm2 : cm8 + 1, cm8] = -1.0
omega = 0.1  # Overrelaxation parameter

start = time()
delta = 1.0
while delta > target:
    errors = []
    for i in range(1, M):
        for j in range(1, M):
            if not ((j == cm2 or j == cm8) and cm2 <= i <= cm8):
                phi_old_ij = phi[i, j]
                phi[i, j] = (1 + omega) / 4 * (
                    phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]
                ) - omega * phi[i, j]
                errors.append(abs(phi_old_ij - phi[i, j]))
    delta = max(errors)
end = time()
print("Simulation time for omega = {}: {:.3f} s".format(omega, end - start))

# Plotting potential with overrelaxation
plt.figure(figsize=(6, 4), dpi=200)
plt.contourf(x, y, phi, levels=10)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Potential $\phi$ [V] with overrelaxation ($\omega$ = {omega})")
plt.colorbar()
plt.grid()
plt.savefig("potential_with_overrelaxation.png")
plt.show()

# Plotting electric field lines with overrelaxation
Ey, Ex = np.gradient(-phi, y, x)
plt.figure(dpi=200)
strm = plt.streamplot(X, Y, Ex, Ey, color=phi, linewidth=2, cmap="autumn")
plt.colorbar(strm.lines).set_label("Potential $V$")
plt.title(f"Electric Field Lines, with overrelaxation ($\omega$ = {omega})")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid()
plt.axis("equal")
plt.tight_layout()
plt.savefig("electric_field_lines_with_overrelaxation.png")
plt.show()
