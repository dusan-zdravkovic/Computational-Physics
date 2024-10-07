"""
Volume of a 10-Dimensional Hypersphere

Author: Dusan Zdravkovic
Purpose: Calculating the volume of a 10-dimensional hypersphere using Monte Carlo methods.
"""

import numpy as np
from scipy.special import gamma


# Function to find Euclidean distance
def R(x_vec):
    """Calculate the Euclidean distance from the origin."""
    return np.sqrt(sum(x_vec**2))


# Function to determine if a point is inside the hyper-sphere
def f(x):
    """Determine if the point x is inside the unit hyper-sphere."""
    return 1 if R(x) <= 1 else 0


# Dimension of the hyper-sphere
dim = 10

# Bounds of integration (hyper-cube)
a, b = -1, 1
N = 1000000  # Number of points for Monte Carlo simulation

# Volume of the hyper-cube
vol_c = (b - a) ** dim

# Monte Carlo integration
k = sum(f(np.random.uniform(a, b, dim)) for _ in range(N))

# Monte Carlo estimation of the volume of the unit hyper-sphere
I = vol_c / N * k

# Theoretical volume of the unit hyper-sphere
vol_s = np.pi ** (dim / 2) / gamma(dim / 2 + 1)

print(f"Theoretical volume for {dim}-dimensional unit sphere:", vol_s)
print("Monte Carlo Estimation:", I)
