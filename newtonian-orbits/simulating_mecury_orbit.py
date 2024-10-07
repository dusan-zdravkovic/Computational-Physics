"""
Simulation of Mercury's Orbit

Author: Dusan Zdravkovic
Purpose: Simulating Mercury's orbit using the Euler-Cromer method and accounting for general relativity effects.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
Ms = 2.0e30  # Sun's mass in kg
AU = 1.496e11  # Astronomical Unit in m
G = 6.67e-11  # Gravitational constant in m^3 kg^-1 s^-2
E_year = 365 * 24 * 60 * 60  # Earth year in seconds


# Function F(x, r) for Newtonian gravity
def F(x, r):
    """Compute the force based on the x-direction acceleration."""
    return -(G * Ms * x) / (r**3)


# Simulation parameters
dt = 0.0001 * E_year
time_array = np.arange(0, E_year, dt)
number_steps = len(time_array)

# Arrays for positions and velocities
x = np.empty(number_steps)
y = np.empty(number_steps)
v_x = np.empty(number_steps)
v_y = np.empty(number_steps)

# Initial conditions
x[0] = 0.47 * AU
y[0] = 0.0 * AU
v_x[0] = 0.0 * AU / E_year
v_y[0] = 8.17 * AU / E_year

# Euler-Cromer method for orbital simulation
for i in range(number_steps - 1):
    r = np.sqrt(x[i] ** 2 + y[i] ** 2)
    v_x[i + 1] = v_x[i] + dt * F(x[i], r)
    v_y[i + 1] = v_y[i] + dt * F(y[i], r)
    x[i + 1] = x[i] + dt * v_x[i + 1]
    y[i + 1] = y[i] + dt * v_y[i + 1]

# Plotting Mercury's Orbit
plt.figure(dpi=200)
plt.title("Simulation of Mercury's Orbit")
plt.axis("equal")
plt.plot(x / AU, y / AU, label="Mercury's Orbit")
plt.plot(0, 0, "rx", label="Sun")
plt.xlabel("x Position [AU]")
plt.ylabel("y Position [AU]")
plt.legend()
plt.savefig("mercurys_orbit_newtonian.png")

# General Relativity Part
alpha = 0.01 * AU**2


# Function F_g(x_g, r_g) for General Relativity
def F_g(x_g, r_g):
    """General relativity version of Newtonian orbit equation."""
    return (-(G * Ms * x_g) / (r_g**3)) * (1 + alpha / (r_g**2))


# Arrays for GR simulation
x_g = np.empty(number_steps)
y_g = np.empty(number_steps)
v_x_g = np.empty(number_steps)
v_y_g = np.empty(number_steps)

# Initial conditions
x_g[0] = 0.47 * AU
y_g[0] = 0.0 * AU
v_x_g[0] = 0.0 * AU / E_year
v_y_g[0] = 8.17 * AU / E_year

# Euler-Cromer method for GR simulation
for i in range(number_steps - 1):
    r_g = np.sqrt(x_g[i] ** 2 + y_g[i] ** 2)
    v_x_g[i + 1] = v_x_g[i] + dt * F_g(x_g[i], r_g)
    v_y_g[i + 1] = v_y_g[i] + dt * F_g(y_g[i], r_g)
    x_g[i + 1] = x_g[i] + dt * v_x_g[i + 1]
    y_g[i + 1] = y_g[i] + dt * v_y_g[i + 1]

# Plotting Mercury's Orbit (General Relativity)
plt.figure(dpi=200)
plt.title("Simulation of Mercury's Orbit (General Relativity)")
plt.axis("equal")
plt.plot(x_g / AU, y_g / AU, label="Mercury's Orbit")
plt.plot(0, 0, "rx", label="Sun")
plt.xlabel("x Position [AU]")
plt.ylabel("y Position [AU]")
plt.legend()
plt.savefig("mercurys_orbit_general_relativity.png")
