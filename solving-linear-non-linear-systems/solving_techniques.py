"""
Linear Equation Solving Methods Comparison

Author: Dusan Zdravkovic
Purpose: Compare Gaussian Elimination, Partial Pivoting, and LU Decomposition methods.
"""

# Imports
import SolveLinear as sl
import numpy as np
from numpy.random import rand
from numpy.linalg import solve
from time import time
import matplotlib.pyplot as plt

# Test Matrix and Vector
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], dtype=float)
v = np.array([-4.0, 3.0, 9.0, 7.0], dtype=float)

# Solve using different methods
x_gauss = sl.GaussElim(A, v)
x_pp = sl.PartialPivot(A, v)

# Display Solutions
print("Solutions to Ax = v:\n")
print("Matrix A:\n", A)
print("\nVector v:\n", v)
print("\nGaussian Elimination: x =", x_gauss)
print("Partial Pivot: x =", x_pp)

# Timing and Error Analysis
time_for_gaussian, time_for_partial, time_for_LU = [], [], []
error_for_gaussian, error_for_partial, error_for_LU = [], [], []

# Test each method for varying matrix sizes
for N in range(5, 300):
    v = rand(N)  # Random vector
    A = rand(N, N)  # Random matrix

    # Gaussian Elimination
    start_gauss = time()
    x_sol_gauss = sl.GaussElim(A, v)
    end_gauss = time()
    time_for_gaussian.append(end_gauss - start_gauss)
    error_for_gaussian.append(np.mean(abs(v - np.dot(A, x_sol_gauss))))

    # Partial Pivoting
    start_partial = time()
    x_sol_partial = sl.PartialPivot(A, v)
    end_partial = time()
    time_for_partial.append(end_partial - start_partial)
    error_for_partial.append(np.mean(abs(v - np.dot(A, x_sol_partial))))

    # LU Decomposition
    start_LU = time()
    x_sol_LU = solve(A, v)
    end_LU = time()
    time_for_LU.append(end_LU - start_LU)
    error_for_LU.append(np.mean(abs(v - np.dot(A, x_sol_LU))))

# Plotting Time Taken
N_axis = np.arange(5, 300, 1)

plt.figure(dpi=200)
plt.title("Time for Gaussian, Partial, LU (Log-Log Plot)")
plt.xlabel("Matrix Size N")
plt.ylabel("Time (s)")
plt.loglog(N_axis, time_for_gaussian, color="blue", label="Gaussian")
plt.loglog(N_axis, time_for_partial, color="red", label="Partial")
plt.loglog(N_axis, time_for_LU, color="green", label="LU")
plt.legend()
plt.savefig("Time_Comparison.png")

# Plotting Error
plt.figure(dpi=200)
plt.title("Error for Gaussian, Partial, LU (Log-Log Plot)")
plt.xlabel("Matrix Size N")
plt.ylabel("Error")
plt.loglog(N_axis, error_for_gaussian, color="blue", label="Gaussian")
plt.loglog(N_axis, error_for_partial, color="red", label="Partial")
plt.loglog(N_axis, error_for_LU, color="green", alpha=0.7, label="LU")
plt.legend()
plt.savefig("Error_Comparison.png")
plt.show()
