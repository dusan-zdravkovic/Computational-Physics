"""
Particle Simulation under Lennard-Jones Potential

Author: Dusan Zdravkovic

Purpose: Simulates particles under Lennard-Jones Potential using the Verlet method.
"""

import numpy as np
import matplotlib.pyplot as plt


def f(v_this, v_that):
    """Calculates acceleration on the particle due to another particle under Lennard-Jones Potential."""
    d = v_that - v_this
    r = np.sqrt(d[0] ** 2 + d[1] ** 2)
    return (24 * (r ** (-13)) - 12 * (r ** (-7))) * (-d / r)


# Setup for simulation
N_dt = 100
dt = 0.01
h = 2 * dt
t_arr = np.arange(0, N_dt * dt, dt)

# Helper for initial positions
N = 16
Lx, Ly = 4.0, 4.0
dx, dy = Lx / np.sqrt(N), Ly / np.sqrt(N)
x_grid, y_grid = np.arange(dx / 2, Lx, dx), np.arange(dy / 2, Ly, dy)
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
x_initial, y_initial = xx_grid.flatten(), yy_grid.flatten()

r = np.empty((N, N_dt, 2))
v = np.empty((N, N_dt, 2))

# Initial conditions
r[:, 0, 0], r[:, 0, 1] = x_initial, y_initial
v[:, 0] = np.zeros((N, 2))

KE, PE = np.zeros(N_dt), np.zeros(N_dt)


# Function for potential on 1 particle
def V_2(v_this, v_that):
    d = v_that - v_this
    r = np.sqrt(d[0] ** 2 + d[1] ** 2)
    return 2 * (r ** (-12) - r ** (-6))


def V(r, i, t):
    V_tot = sum(V_2(r[i, t], r[j, t]) for j in range(N) if i != j)
    return V_tot


def F(r, i, t):
    F_net = sum(f(r[i, t], r[j, t]) for j in range(N) if i != j)
    return F_net


# Verlet simulation
t = 0
for i in range(N):
    v[i, t + 1] = v[i, t] + h / 2 * F(r, i, t)

while t < N_dt - 2:
    for i in range(N):
        r[i, t + 2] = r[i, t] + h * v[i, t + 1]
    k = np.array([h * F(r, i, t + 2) for i in range(N)])
    for i in range(N):
        v[i, t + 2] = v[i, t + 1] + 0.5 * k[i]
        v[i, t + 3] = v[i, t + 1] + k[i]
    KE[t] = sum((v[i, t, 0] ** 2 + v[i, t, 1] ** 2) / 2 for i in range(N))
    PE[t] = sum(V(r, i, t) / 2 for i in range(N))
    t += 2

# Plot trajectories
plt.figure(figsize=(6, 6), dpi=200)
plt.title("Trajectory in (x,y) with 16 particles in a grid")
for i in range(N):
    plt.plot(r[i, ::2, 0], r[i, ::2, 1], ".")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("trajectory_xy.png")
plt.show()

# Plot energy
plt.figure(dpi=200)
plt.title("Energy vs. time")
plt.plot(t_arr[::2], KE[::2], label="Kinetic Energy")
plt.plot(t_arr[::2], PE[::2], label="Potential Energy")
plt.plot(t_arr[::2], KE[::2] + PE[::2], label="Total Energy")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.savefig("energy_vs_time.png")
plt.show()
