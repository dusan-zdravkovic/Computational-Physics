"""
Stefan-Boltzmann Constant Calculation

Author: Dusan Zdravkovic

Purpose: To compute the Stefan-Boltzmann constant using numerical integration and compare it to the known value.
"""

import scipy.constants as sc
import numpy as np
from scipy.integrate import quad


# Defining the integrand for Stefan-Boltzmann law calculation
def integrand(x):
    """Integrand for the Stefan-Boltzmann law calculation."""
    return (x**3) / (np.exp(x) - 1)


# Computing the integral from 0 to infinity
computed_integral = quad(integrand, 0, np.inf)[0]
print("Computed integral value is", computed_integral)


# Defining the W(T) function using computed_integral
def W(T):
    """Function to calculate W as a function of temperature T."""
    C_1 = (np.pi * T**3 * sc.Boltzmann**3 * 2) / (
        sc.speed_of_light * sc.Planck**2
    )
    return C_1 * computed_integral


# Placeholder temperature, T (will cancel out in calculation)
T = 400

# Computing Stefan-Boltzmann constant using the integral
stefan_boltzmann_computed = W(T) / (T**4)
print("Computed Stefan-Boltzmann value is", stefan_boltzmann_computed)

# Comparing with the known Stefan-Boltzmann constant
print("True Stefan-Boltzmann value is", sc.Stefan_Boltzmann)
print(
    "The error of the computed value is:",
    abs(sc.Stefan_Boltzmann - stefan_boltzmann_computed),
)
