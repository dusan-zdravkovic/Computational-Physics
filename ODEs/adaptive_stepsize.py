"""
Simulation of Particles under Lennard-Jones Potential with Adaptive Step-Size

Author: Dusan Zdravkovic
Date: Oct. 4

Purpose: Explore adaptive step-size in simulations under the Lennard-Jones Potential.
"""

import numpy as np
import matplotlib.pyplot as plt


def rhs(r):
    """The right-hand-side of the equations"""
    M = 10.0
    L = 2.0
    x, vx, y, vy = r
    r2 = x**2 + y**2
    Fx, Fy = -M * np.array([x, y], float) / (r2 * np.sqrt(r2 + 0.25 * L**2))
    return np.array([vx, Fx, vy, Fy], float)


a = 0.0
b = 10.0
N = 1000
h = (b - a) / N

# Adaptive stepsize
tpoints = []
xpoints = []
vxpoints = []
ypoints = []
vypoints = []
r = np.array([1.0, 0.0, 0.0, 1.0], float)
h_arr = []

from time import time

start_adaptive = time()

t = a
while t < 10:
    # Calculations for adaptive stepsize
    k1 = h * rhs(r)
    k2 = h * rhs(r + 0.5 * k1)
    k3 = h * rhs(r + 0.5 * k2)
    k4 = h * rhs(r + k3)
    r_h = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    k1 = h * rhs(r_h)
    k2 = h * rhs(r_h + 0.5 * k1)
    k3 = h * rhs(r_h + 0.5 * k2)
    k4 = h * rhs(r_h + k3)
    r_1 = r_h + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    k1 = 2 * h * rhs(r)
    k2 = 2 * h * rhs(r + 0.5 * k1)
    k3 = 2 * h * rhs(r + 0.5 * k2)
    k4 = 2 * h * rhs(r + k3)
    r_2 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    error_x = 1 / 30 * (r_1[0] - r_2[0])
    error_y = 1 / 30 * (r_1[2] - r_2[2])
    total_error = np.sqrt(error_x**2 + error_y**2)
    delta = 1e-6
    rho = h * delta / total_error
    h = h * rho ** (1 / 4)

    if rho > 1:
        tpoints.append(t)
        r = r_h
        h_arr.append(h)
        xpoints.append(r[0])
        vxpoints.append(r[1])
        ypoints.append(r[2])
        vypoints.append(r[3])
        t += h

end_adaptive = time()
print("Time taken for adaptive method:", end_adaptive - start_adaptive, "s")

# Fixed stepsize
a = 0.0
b = 10.0
N = 10000
h = (b - a) / N

tpoints2 = np.arange(a, b, h)
xpoints2 = []
vxpoints2 = []
ypoints2 = []
vypoints2 = []
r = np.array([1.0, 0.0, 0.0, 1.0], float)
for t in tpoints2:
    xpoints2.append(r[0])
    vxpoints2.append(r[1])
    ypoints2.append(r[2])
    vypoints2.append(r[3])
    k1 = h * rhs(r)
    k2 = h * rhs(r + 0.5 * k1)
    k3 = h * rhs(r + 0.5 * k2)
    k4 = h * rhs(r + k3)
    r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

plt.figure(dpi=200)
plt.plot(xpoints[::40], ypoints[::40], ".", label="adaptive", ms=10)
plt.plot(xpoints2[::40], ypoints2[::40], ".", label="fixed h", alpha=0.5)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Trajectory of a ball bearing around a space rod.")
plt.axis("equal")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig("trajectory_adaptive.png")
plt.show()

# Step size as a function of time
time_array = np.linspace(0, 0.19378, 754)  # Computation time
plt.figure()
plt.plot(time_array, h_arr)
plt.title("Step size as a function of time")
plt.xlabel("time (s)")
plt.ylabel("step size (s)")
plt.grid()
plt.savefig("step_size_time.png")
plt.show()
