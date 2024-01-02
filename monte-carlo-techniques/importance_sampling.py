"""
Importance Sampling in Monte Carlo Methods

Author: Dusan Zdravkovic
Purpose: Demonstrating importance sampling in Monte Carlo integration.
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Function to integrate."""
    return x ** (-1 / 2) / (1 + np.exp(x))


# Q3a. Mean Value Method
N = 10000  # Number of samples
M = 100  # Number of loops/estimations
a, b = 0, 1  # Bounds of integration

# Integrals from the mean value method
I_MV = []
for _ in range(M):
    xs = np.random.uniform(a, b, size=N)
    I_MV.append((b - a) / N * sum(f(xs)))

# Q3b. Importance Sampling Method
I_IS = []


def w(x):
    """Weight function for importance sampling."""
    return x ** (-1 / 2)


def x_transform(z):
    """Transformation from z to x, derived in report."""
    return z**2


int_w = 2.0  # Integral of the weight function in the bounds (analytic)

for _ in range(M):
    zs = np.random.uniform(0, 1, size=N)
    xs = x_transform(zs)
    I_IS.append(1 / N * sum(f(xs) / w(xs)) * int_w)

# Q3c. Plotting Histograms
plt.figure(dpi=200)
plt.hist(I_MV, bins=10, range=(0.80, 0.88), edgecolor="white")
plt.title("Histogram of the Mean Value Method")
plt.xlabel("Value of Estimation for the Integral $I$")
plt.ylabel("Frequency (Count)")
plt.tight_layout()
plt.savefig("histogram_mean_value_method.png")

plt.figure(dpi=200)
plt.hist(I_IS, bins=10, range=(0.80, 0.88), edgecolor="white")
plt.title("Histogram of the Importance Sampling Method")
plt.xlabel("Value of Estimation for the Integral $I$")
plt.ylabel("Frequency (Count)")
plt.tight_layout()
plt.savefig("histogram_importance_sampling_method.png")
