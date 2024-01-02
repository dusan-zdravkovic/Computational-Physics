"""
Author: Dusan Zdravkovic
Purpose: Calculation of periods using Gaussian Quadrature and analysis of oscillatory motion.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc


# Functions for Gaussian Quadrature
def gaussxw(N):
    """
    Calculate integration points and weights for Gaussian quadrature.
    Returns integration points x and weights w for Nth-order approximation.
    """
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3, 4 * N - 1, N) / (4 * N + 2)
    x = np.cos(np.pi * a + 1 / (8 * N * N * np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = np.ones(N, float)
        p1 = np.copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = max(np.abs(dx))

    # Calculate the weights
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)
    return x, w


def gaussxwab(N, a, b):
    """
    Calculate integration points and weights for Gaussian quadrature mapped to interval [a, b].
    """
    x, w = gaussxw(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


# Defining constants
m = 1  # mass in kg
k = 12.0  # spring constant in N/m
c = sc.speed_of_light  # speed of light in m/s
x_o = 0.01  # initial position in meters


# Defining the g(x) function
def g(x):
    """Velocity equation from positive root of the energy equation."""
    numerator = k * (x_o**2 - x**2) * (2 * m * c**2 + k * (x_o**2 - x**2) / 2)
    denominator = 2 * (m * c**2 + k * (x_o**2 - x**2) / 2) ** 2
    return c * np.abs(numerator / denominator) ** (1 / 2)


# Using Gaussian Quadrature to evaluate integral
N = 8
x, w = gaussxwab(N, 0.0, x_o)
T = sum(w[i] / g(x[i]) for i in range(len(x)))
print("The period T for N = 8 is equal to", 4 * T, "seconds")

# Repeat with N1 = 16
N1 = 16
x1, w1 = gaussxwab(N1, 0.0, x_o)
T1 = sum(w1[q] / g(x1[q]) for q in range(len(x1)))
print("The period T for N = 16 is equal to", 4 * T1, "seconds")

# Plotting Integrands vs. Sample Points
plt.figure(dpi=200)
plt.title("Integrands vs. Sample Points")
plt.xlabel("Sampling Points")
plt.ylabel("Integrands")
plt.plot(x, 4 / g(x), label="N=8")
plt.plot(x1, 4 / g(x1), label="N=16")
plt.grid()
plt.legend()
plt.savefig("integrand_sample_points.png")

# Plotting Weighted Values vs. Sample Points
plt.figure(dpi=200)
plt.title("Weighted Values vs. Sample Points")
plt.xlabel("Sampling Points")
plt.ylabel("Weighted Values")
plt.plot(x, (4 * w) / g(x), label="N=8")
plt.plot(x1, (4 * w1) / g(x1), label="N=16")
plt.grid()
plt.legend()
plt.savefig("weighted_values_sample_points.png")


# Function to give T based off x_o
def T(x_o):
    N = 200
    x, w = gaussxwab(N, 0.0, x_o)
    return 4 * sum(w[i] / g(x[i]) for i in range(len(x)))
