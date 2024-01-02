"""
Simulating Jupiter's Orbit, Sun-Jupiter System

Author: Dusan Zdravkovic
Purpose: Simulating the orbits of Jupiter and Earth in the Sun-Jupiter system using the Euler-Cromer method.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
Ms = 2.0e30  # Sun's mass in kg
Mj = 1.0e-3 * Ms  # Jupiter's mass in kg
AU = 1.496e11  # Astronomical Unit in meters
G = 6.67e-11  # Gravitational constant
E_year = 365 * 24 * 60 * 60  # Earth year in seconds


# Function F(x, r) for gravitational force
def F(x, r):
    """Function for gravitational force in the x-direction."""
    return -(G * Ms * x) / (r**3)


# Time parameters
dt = 0.0001 * E_year
time_array = np.arange(0, E_year * 10, dt)
number_steps = len(time_array)

# Arrays for Jupiter's position and velocity
x_J = np.empty(number_steps)
y_J = np.empty(number_steps)
v_x_J = np.empty(number_steps)
v_y_J = np.empty(number_steps)

# Initial conditions for Jupiter
x_J[0] = 5.2 * AU
y_J[0] = 0.0 * AU
v_x_J[0] = 0.0 * AU / E_year
v_y_J[0] = 2.63 * AU / E_year

# Euler-Cromer method for Jupiter's orbit
for i in range(number_steps - 1):
    r_J = np.sqrt(x_J[i] ** 2 + y_J[i] ** 2)
    v_x_J[i + 1] = v_x_J[i] + dt * F(x_J[i], r_J)
    v_y_J[i + 1] = v_y_J[i] + dt * F(y_J[i], r_J)
    x_J[i + 1] = x_J[i] + dt * v_x_J[i + 1]
    y_J[i + 1] = y_J[i] + dt * v_y_J[i + 1]


# Function for gravitational force considering both Sun and Jupiter
def F_new(x_s, r_s, x_j, r_j):
    """Combined gravitational force from Sun and Jupiter."""
    F_s = -(G * Ms * x_s) / r_s**3
    F_j = -(G * Mj * x_j) / r_j**3
    return F_s + F_j


# Arrays for Earth's position and velocity
x = np.empty(number_steps)
y = np.empty(number_steps)
v_x = np.empty(number_steps)
v_y = np.empty(number_steps)

# Initial conditions for Earth
x[0] = 1.0 * AU
y[0] = 0.0 * AU
v_x[0] = 0.0 * AU / E_year
v_y[0] = 6.18 * AU / E_year

# Euler-Cromer method for Earth's orbit considering Jupiter
for i in range(number_steps - 1):
    r_s = np.sqrt(x[i] ** 2 + y[i] ** 2)
    x_EJ = x[i] - x_J[i]
    y_EJ = y[i] - y_J[i]
    r_EJ = np.sqrt(x_EJ**2 + y_EJ**2)
    v_x[i + 1] = v_x[i] + dt * F_new(x[i], r_s, x_EJ, r_EJ)
    v_y[i + 1] = v_y[i] + dt * F_new(y[i], r_s, y_EJ, r_EJ)
    x[i + 1] = x[i] + dt * v_x[i + 1]
    y[i + 1] = y[i] + dt * v_y[i + 1]

# Plotting the orbits of Jupiter and Earth
plt.figure(dpi=200)
plt.title("Simulation of the Orbit of Jupiter and Earth")
plt.axis("equal")
plt.plot(x_J / AU, y_J / AU, label="Jupiter's Orbit")
plt.plot(x / AU, y / AU, label="Earth's Orbit")
plt.plot(0, 0, "rx", label="Sun")
plt.xlabel("x Position [AU]")
plt.ylabel("y Position [AU]")
plt.legend()
plt.savefig("orbits_jupiter_earth.png")
