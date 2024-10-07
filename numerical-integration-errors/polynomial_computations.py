"""
Numerical Error in Polynomial Computations

Author: Dusan Zdravkovic

Purpose: Determine the roundoff error in numerical computations of polynomials and analyze their propagation.
"""

import numpy as np
import matplotlib.pyplot as plt


def p(u):
    return (1 - u) ** 8


def q(u):
    terms0 = 1 - 8 * u + 28 * u**2 - 56 * u**3
    terms1 = 70 * u**4 - 56 * u**5 + 28 * u**6 - 8 * u**7 + u**8
    return terms0 + terms1


u_array = np.linspace(1 - 0.02, 1 + 0.02, 500)
p_u = p(u_array)
q_u = q(u_array)

# Plotting p(u) and q(u)
fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
fig.suptitle("Plots of $p(u)$ and $q(u)$ vs. $u$")
axs[0].plot(u_array, p_u, "m", label="$p(u)$")
axs[1].plot(u_array, q_u, "c", label="$q(u)$")
axs[0].set_xlabel("$u$")
axs[0].set_ylabel("$p(u)$")
axs[1].set_xlabel("$u$")
axs[1].set_ylabel("$q(u)$")
fig.legend()
plt.savefig("pu_qu_plot.png")
plt.show()

# Difference between p(u) and q(u)
diff = p_u - q_u
plt.figure()
plt.title("The Difference $(p(u) - q(u))$ vs. $u$")
plt.plot(u_array, diff)
plt.xlabel("u")
plt.ylabel("p(u) - q(u)")
plt.savefig("difference_pu_qu.png")
plt.show()

plt.figure()
plt.title("Histogram of $(p(u) - q(u))$")
plt.hist(diff)
plt.xlabel("$(p(u) - q(u))$")
plt.savefig("histogram_pu_qu.png")
plt.show()

sigma = np.std(diff)
C = 1
sigma_est = C * np.sqrt(10) * np.sqrt(np.mean(diff**2))
print("Standard deviation using numpy.std:", sigma)
print("Estimated standard deviation:", sigma_est)

u_array2 = np.linspace(0.980, 1, 500)
q_u2 = q(u_array2)
err_est = C / np.sqrt(9) * np.sqrt(np.mean(q_u2**2)) / np.mean(q_u2)
print("Estimated fractional error for q(u):", err_est)

u_array3 = np.linspace(0.980, 0.983, 500)
p_u3 = p(u_array3)
q_u3 = q(u_array3)
err = abs(p_u3 - q_u3) / abs(p_u3)

plt.figure()
plt.title("Fractional error of $q(u)$ vs. $u$")
plt.plot(u_array3, err)
plt.xlabel("$u$")
plt.ylabel("Fractional error of $q(u)$")
plt.savefig("fractional_error_qu.png")
plt.show()


def f(u):
    return u**8 / ((u**4) * (u**4))


f_u = f(u_array)
plt.figure()
plt.title("$(f(u)-1)$ vs. $u$")
plt.plot(u_array, f_u - 1)
plt.xlabel("$u$")
plt.ylabel("$f(u)-1$")
plt.savefig("fu_minus_1.png")
plt.show()

print("Estimate for error of f(u):", 1e-16 * np.mean(f_u))
