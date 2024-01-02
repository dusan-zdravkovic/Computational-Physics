"""
Sea Level Pressure Analysis Script

Author: Dusan Zdravkovic
Purpose: Analysis of a Sea Level Pressure dataset using Fourier Transform
and creating contour plots for data visualization 
"""

import numpy as np
import matplotlib.pyplot as plt


# Read in data
SLP = np.loadtxt("SLP.txt")
Longitude = np.loadtxt("lon.txt")
Times = np.loadtxt("times.txt")


# Plot unfiltered data
plt.figure(figsize=(6, 4), dpi=200)
plt.contourf(Longitude, Times, SLP)
plt.xlabel("Longitude (degrees)")
plt.ylabel("Days since Jan. 1, 2015")
plt.title("Unfiltered SLP Anomaly (hPa)")
plt.colorbar()
plt.savefig("unfiltered_SLP_anomaly.png")


# Perform a Fourier Transform on the SLP data to get the coefficients
A_SLP = np.fft.rfft(SLP, axis=1)


# Filter for wave numbers m = 3,5 by making two separate arrays
# Coefficients only with m = 5
A_3 = np.zeros_like(A_SLP)
A_3[:, 3] = A_SLP[:, 3]


# Coefficients only with m = 5
A_5 = np.zeros_like(A_SLP)
A_5[:, 5] = A_SLP[:, 5]


# Set the filtered SLPs as the inverse fft's of the filtered coefficients
SLP_3 = np.fft.irfft(A_3, axis=1)
SLP_5 = np.fft.irfft(A_5, axis=1)


# Plot contours of the filtered SLPs
# m = 3
plt.figure(figsize=(6, 4), dpi=200)
plt.contourf(Longitude, Times, SLP_3)
plt.xlabel("Longitude (degrees)")
plt.ylabel("Days since Jan. 1, 2015")
plt.title("Filtered SLP Anomaly with m = 3 (hPa)")
plt.colorbar()
plt.savefig("filtered_SLP_anomaly_m3.png")


# m = 5
plt.figure(figsize=(6, 4), dpi=200)
plt.contourf(Longitude, Times, SLP_5)
plt.xlabel("Longitude (degrees)")
plt.ylabel("Days since Jan. 1, 2015")
plt.title("Filtered SLP Anomaly with m = 5 (hPa)")
plt.colorbar()
plt.savefig("filtered_SLP_anomaly_m5.png")
