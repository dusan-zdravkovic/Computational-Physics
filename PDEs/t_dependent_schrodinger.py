"""
Simulating Time-dependent Schrödinger Equation Using Crank-Nicolson Method

Author: Dusan Zdravkovic

Purpose: Solving time-dependent Schrondinger 
equation using Crank-Nicolson method. 

Note: Code runtime is approximately 4 minutes, 
runs faster by lowering 'P' value on line 37.
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from numpy.linalg import solve
from numpy import eye, diag, matmul, conj, arange, ones  # convenient imports


def simps_int(Rpoints, h):
    """Integral solver by Simpson's method function"""
    N = len(Rpoints)
    integral = Rpoints[0] + Rpoints[-1]
    for k in range(1, N, 2):
        integral += 4 * Rpoints[k]
    for k in range(2, N, 2):
        integral += 2 * Rpoints[k]
    integral *= h / 3
    return integral


# Constants
L = 1e-8  # square well length
m_e = 9.109e-31  # mass of electron in kg

P = 1024  # number of cells in the x-direction
a = L / P  # grid spacing
p = np.arange(1, P, 1)  # array for p range {1,...,P-1} (integer)
x = p * a - L / 2  # array for discretized position xp = pa - L/2


# Discretizing psi in time
N = 3000  # number of time steps
n = np.arange(1, N - 1, 1)  # array for n range {1,...,N-1} (integer)
tau = 1e-18  # time step
T = N * tau  # final time


# Creating arrays and matrices
A = -1 * sc.hbar**2 / (2 * m_e * a**2)


def V(x):
    """Definition of the potential function"""
    if -L / 2 < x or x < -L / 2:
        return 0
    else:
        return np.inf


Bp = np.zeros(len(x))  # diagonal vector
for i in range(len(x)):
    Bp[i] = V(x[i]) - 2 * A


# Making the Hamiltonian
vec_diag = Bp
D = diag(Bp, k=0)  # k=0 means diagonal
Sup = A * eye(len(Bp), k=1)  # matrix on 1st super-diagonal
Sub = A * eye(len(Bp), k=-1)  # matrix on 1st sub-diagonal

HD = D + Sub + Sup  # final hamiltonian matrix


# Making the Identity matrix
diagonal_ones = np.ones_like(Bp)
I = diag(diagonal_ones, k=0)  # rank P-1 identity matrix


# Variables for equation (13) L = psi^(n+1) = v
L_matrix = I + 1j * tau / (2 * sc.hbar) * HD  # L in equation (13)
R = I - 1j * tau / (2 * sc.hbar) * HD  # R in equation (13)

psi_vector = np.zeros_like(p, float)  # vector (phi_1,...,phi_p,...,phi_P-1)


# Initial condition constants
sigma = L / 25
k = 500 / L
x_o = L / 5

psi_o_anal = 1 / (2 * np.pi * sigma**2) ** (1 / 4)  # analytical value for psi naught
psi_initial_unnorm = np.exp(
    -((x - x_o) ** 2) / (4 * sigma**2) + 1j * k * x
)  # unnormalized


# Redefining psi_initial
psi_o = 1 / np.sqrt(simps_int(abs(psi_initial_unnorm) ** 2, a))
psi_initial = psi_o * psi_initial_unnorm
norm0 = simps_int(abs(psi_initial) ** 2, a)


# Initializing for main loop
psi_vector = psi_initial
v = matmul(R, psi_vector)


# Main loop
plt.figure(dpi=200)
epsilon = tau / 100  # threshold for comparision of floating point equality
plt.plot(x, psi_initial.real, label="$t={}$ s".format(round(0, 20)))

integrand = conj(psi_initial) * x * psi_initial  # integrand for <X>

x_mean = []  # initialzing array to store <X> values over time
x_mean.append(simps_int(integrand, a))
norms = []  # initializing array to store normalized values over time
norms.append(norm0)

E = []  # initialzing array to store energy values over time
E_integrand = matmul(HD, psi_initial)
E_integrand2 = E_integrand * conj(psi_initial)
E.append(simps_int(abs(E_integrand2), a))

for i in n:
    psi_vector = solve(L_matrix, v)  # solving the system of equations every n
    v = matmul(R, psi_vector)  # matrix multplication for solution

    # Plotting at specific times
    if abs(tau * i - T / 4) < epsilon:
        plt.plot(x, psi_vector.real, label="$t={}$ s".format(round(tau * i, 20)))
    if abs(tau * i - T / 2) < epsilon:
        plt.plot(x, psi_vector.real, label="$t={}$ s".format(round(tau * i, 20)))
    if abs(tau * i - 3 * T / 4) < epsilon:
        plt.plot(x, psi_vector.real, label="$t={}$ s".format(round(tau * i, 20)))

    # Expectation value storing
    integrand = conj(psi_vector) * x * psi_vector
    x_mean.append(simps_int(integrand, a))

    norms.append(simps_int(abs(psi_vector) ** 2, a))

    # Energy value storing
    E_integrand = matmul(HD, psi_vector)
    E_integrand2 = E_integrand * conj(psi_vector)
    E.append(simps_int(abs(E_integrand2), a))


plt.plot(x, psi_vector.real, label="$t={}$ s".format(round(T, 20)))
plt.legend(loc=2, prop={"size": 6})
plt.xlabel("x")
plt.title("Wave Function (real) at Time: 0,T/4,T/2,3T/4,T")
plt.ylabel("$\Psi(x)$ ")
plt.savefig("wave_function.png")
plt.show()


# Expectation value of the particles position, <X>(t) from t=0 to t=T plot
plt.figure(dpi=200)
plt.plot(n * tau, np.array(x_mean[1:]).real)
plt.title("Expectation Value vs. Time")
plt.xlabel("time (s)")
plt.ylabel("$<X>$")
plt.grid()


# Energy conserved from initial to final print statement
E_integrand = matmul(HD, psi_initial)
E_integrand2 = E_integrand * conj(psi_initial)
E_initial = simps_int(abs(E_integrand2), a)

E_integrand3 = matmul(HD, psi_vector)
E_integrand4 = E_integrand3 * conj(psi_vector)
E_final = simps_int(abs(E_integrand4), a)

print("Energy initial is:", E_initial, "and Energy final is:", E_final)


# Plot Expectation Value
plt.figure(dpi=200)
plt.plot(n * tau, np.array(norms[1:]).real)
plt.title("Normalization vs Time")
plt.xlabel("time (s)")
plt.ylabel("Normalization")
plt.grid()


# Plot Energy
plt.figure(dpi=200)
plt.plot(n * tau, np.array(E[1:]))
plt.title("Energy vs Time")
plt.xlabel("time (s)")
plt.ylabel("Energy")
plt.grid()
