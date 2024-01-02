"""
Standard Deviation Calculation Comparison

Author: Dusan Zdravkovic
Purpose: Implements two separated methods of computing the standard deviation and compares them to the numpy.std() method.
"""

import numpy as np

# Read the array of data given
cdata_array = np.loadtxt("cdata.txt")


def std_2_pass(array):
    """Two-pass method for computing standard deviation."""
    n = len(array)
    mean = sum(array) / n
    s = sum((xi - mean) ** 2 for xi in array)
    sigma = np.sqrt(s / (n - 1))
    return sigma


def std_1_pass(array):
    """One-pass method for computing standard deviation."""
    n = len(array)
    mean = sum(array) / n
    s = sum(xi**2 for xi in array)
    s_total = s - n * mean**2
    if s_total < 0:
        print("Error: square root of a negative number")
        return -1
    sigma = np.sqrt(s_total / (n - 1))
    return sigma


def compare_errors(array, func_2_pass, func_1_pass):
    """Compares relative errors of 2-pass and 1-pass methods with numpy.std()."""
    std_correct = np.std(array, ddof=1)
    std_1 = func_2_pass(array)
    std_2 = func_1_pass(array)
    err_std_1 = (std_1 - std_correct) / std_correct
    err_std_2 = (std_2 - std_correct) / std_correct
    print("Two-pass method error:", err_std_1)
    print("One-pass method error:", err_std_2)


# Analysis
print("For cdata.txt:")
compare_errors(cdata_array, std_2_pass, std_1_pass)

arr_0 = np.random.normal(0.0, 1.0, 2000)
arr_1e7 = np.random.normal(1.0e7, 1.0, 2000)

print("For random normal data with mean = 0.:")
compare_errors(arr_0, std_2_pass, std_1_pass)

print("For random normal data with mean = 1.e7:")
compare_errors(arr_1e7, std_2_pass, std_1_pass)
