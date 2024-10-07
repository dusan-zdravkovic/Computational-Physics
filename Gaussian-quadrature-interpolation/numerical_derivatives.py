"""
Analysis of Numerical Derivatives Script

Author: Dusan Zdravkovic
Purpose: Analysis of numerical derivative methods and comparison with analytical solutions.
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the function f(x)
def f(x):
    return np.exp(-(x**2))


# Compute the analytical answer -e^(-0.25)
real_ans = -np.exp(-0.25)
print("The analytical answer: df/dx(0.5) = {}".format(real_ans))

# Set an array 'h' to be from -16 to 0
h = np.logspace(-16, 0, num=17)  # array of 1e-16, 1e-15, ... 1

print("\nh, \t\t\t df/dx, \t\t\t\t error")

# Compute numerical derivative by the forward difference scheme
forw_diff = (f(0.5 + h) - f(0.5)) / h
fd_err = abs(forw_diff - real_ans)

# Printing results
for h_val, fd, err in zip(h, forw_diff, fd_err):
    print("{} & \t {} & \t {}\\\\".format(h_val, fd, err))

# Plotting
plt.figure(dpi=140)
plt.title("Log-log Plot of the Forward Difference vs. $h$")
plt.xlabel("$h$")
plt.ylabel("Forward Difference")
plt.loglog(h, fd_err)
plt.tight_layout()
plt.savefig("forward_difference.png")
plt.show()

# Compute numerical derivative by the central difference scheme (on h again)
cent_diff = (f(0.5 + h) - f(0.5 - h)) / (2 * h)
cd_err = abs(cent_diff - real_ans)

# Plot the errors for both numerical derivatives vs. h on the same plot
plt.figure(dpi=140)
plt.title("Log-log Plot of two Numerical Derivatives vs. $h$")
plt.xlabel("$h$")
plt.ylabel("Numerical Derivatives")
plt.loglog(h, fd_err, label="Forward Difference")
plt.loglog(h, cd_err, label="Central Difference")
plt.legend()
plt.tight_layout()
plt.savefig("numerical_derivatives_comparison.png")
plt.show()
