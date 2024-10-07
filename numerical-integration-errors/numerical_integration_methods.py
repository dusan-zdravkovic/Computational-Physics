"""
Compare and Contrast of Numerical Integration Methods: Trapezoid Rule vs Simpson's Rule

Author: Dusan
Purpose: To compare the Trapezoid rule and Simpson's rule for numerical integration.
"""

import numpy as np
from time import time


# Function to integrate
def f(x):
    return 4 / (1 + x**2)


# Variables for integration
N = 16  # Number of slices
a = 0.0  # Lower limit of integration
b = 1.0  # Upper limit of integration
h = (b - a) / N  # Width of a slice

# Trapezoid Rule Integration
start1 = time()
s = 0.5 * f(a) + 0.5 * f(b)
for k in range(1, N):
    s += f(a + k * h)
trap_value = h * s
end1 = time()
print("Trapezoid rule value:", trap_value)
print("Time taken for Trapezoid rule:", end1 - start1)

# Simpson's Rule Integration
start2 = time()
s_1 = f(a) + f(b)
for k in range(1, N, 2):
    s_1 += 4 * f(a + k * h)
for k in range(2, N, 2):
    s_1 += 2 * f(a + k * h)
simp_value = (h / 3) * s_1
end2 = time()
print("Simpson's rule value:", simp_value)
print("Time taken for Simpsons rule:", end2 - start2)


# Error Calculation
def error(value):
    return abs(value - np.pi)


print("Error in Trapezoid rule:", error(trap_value))
print("Error in Simpson's rule:", error(simp_value))

# Error Estimation
I_32 = 3.141429893174975  # Obtained from previous code with N = 32
I_16 = 3.140941612041389  # Obtained from previous code with N = 16
h = (b - a) / 16
C = ((I_32 - I_16) / h**2) * (4 / 3)
error_est = C * (h / 2) ** 2
print("Error estimation:", error_est)
