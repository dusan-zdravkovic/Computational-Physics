"""
Matrix Multiplication Performance Comparison

Author: Dusan Zdravkovic
Purpose: To compare the performance of brute force matrix multiplication with numpy's dot product.
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time


def time_matrix_product(N):
    """Records the time to multiply two constant matrices of size N by N using brute force."""
    A = np.ones([N, N], float) * 3
    B = np.ones([N, N], float) * 5
    C = np.zeros([N, N], float)
    start = time()

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

    end = time()
    diff = end - start
    print("N =", N, "N^3 =", N**3, "Time =", diff)
    return diff


def time_np_dot(N):
    """Records the time to multiply two constant matrices of size N by N using numpy.dot()."""
    A = np.ones([N, N], float) * 3
    B = np.ones([N, N], float) * 5
    start = time()
    C = np.dot(A, B)
    end = time()
    return end - start


# Range of N values
Ns = range(2, 301)
num_N = len(Ns)

t_matrix_product = np.zeros(num_N)
t_np_dot = np.zeros(num_N)

for i in range(num_N):
    t_matrix_product[i] = time_matrix_product(Ns[i])
    t_np_dot[i] = time_np_dot(Ns[i])

# Plotting the computation times
Ns3 = np.power(Ns, 3)
fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=100, constrained_layout=True)
fig.suptitle("Computation Time of Matrix Multiplication")

# Time vs N
axs[0].plot(Ns, t_matrix_product, label="Matrix Multiplication")
axs[0].plot(Ns, t_np_dot, label="Numpy.dot()")
axs[0].legend()
axs[0].set_xlabel("$N$")
axs[0].set_ylabel("Computation time [s]")
plt.savefig("computation_time_vs_N.png")

# Time vs N^3
axs[1].plot(Ns3, t_matrix_product, label="Matrix Multiplication")
axs[1].plot(Ns3, t_np_dot, label="Numpy.dot()")
axs[1].legend()
axs[1].set_xlabel("$N^3$")
axs[1].set_ylabel("Computation time [s]")
plt.savefig("computation_time_vs_N3.png")

plt.show()
