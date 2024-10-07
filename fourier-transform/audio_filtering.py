"""
Audio Signal Filtering Script

Author: Dusan Zdravkovic
Purpose: Filter out unwanted sound from an audio file and export a new filtered file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from numpy.fft import rfft, irfft, rfftfreq


# Read the audio file
sample, data = read("GraviteaTime.wav")
channel_0, channel_1 = data[:, 0], data[:, 1]
N_Points = len(channel_0)


# Time and frequency domain calculations
dt = 1 / sample  # Time step
T = N_Points * dt  # Length of interval
freq = np.arange(N_Points / 2 + 1) * 2 * np.pi / T  # Converting to angular frequency
t = np.arange(N_Points) * dt  # Dimensional time axis


# Plotting the unfiltered sound
plt.figure(1, figsize=(8, 3), dpi=200)
plt.xlabel("time (s)")
plt.ylabel("Amplitude")
plt.title("channel_0")
plt.grid()
plt.plot(t, channel_0)
plt.savefig("channel_0_unfiltered.png")

plt.figure(2, figsize=(8, 3), dpi=200)
plt.xlabel("time (s)")
plt.ylabel("Amplitude")
plt.title("channel_1")
plt.grid()
plt.plot(t, channel_1)
plt.savefig("channel_1_unfiltered.png")


# Fourier Transforms of channel_0 and channel_1
c_0 = rfft(channel_0)  # Fourier Transform on channel_0
c_1 = rfft(channel_1)  # Fourier Transform on channel_1

f_0 = rfftfreq(N_Points, dt)  # Obtain frequency array from channel 0
f_1 = rfftfreq(N_Points, dt)  # Obtain frequency array from channel 1


# Plotting the Fourier Coefficients Unfiltered
plt.figure(3, figsize=(8, 3), dpi=200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier Coefficient")
plt.title("channel_0 Fourier Coefficients Unfiltered")
plt.grid()
plt.plot(f_0, abs(c_0))
plt.savefig("channel_0_Fourier_unfiltered.png")

plt.figure(4, figsize=(8, 3), dpi=200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier Coefficient")
plt.title("channel_1 Fourier Coefficients Unfiltered")
plt.grid()
plt.plot(f_1, abs(c_1))
plt.savefig("channel_0_Fourier_unfiltered.png")


# Setting the frequencies greater than 880 Hz to 0
c_0filtered = c_0
for i in range(len(c_0)):
    if f_0[i] > 880:
        c_0filtered[i] = 0

c_1filtered = c_1
for i in range(len(c_1)):
    if f_1[i] > 880:
        c_1filtered[i] = 0


# Plotting the filtered Fourier coefficients
plt.figure(5, figsize=(8, 3), dpi=200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier Coefficient")
plt.title("channel_0 Fourier Coefficients Filtered")
plt.grid()
plt.plot(f_0, abs(c_0filtered))
plt.savefig("channel_0_Fourier_filtered.png")

plt.figure(6, figsize=(8, 3), dpi=200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fourier Coefficient")
plt.title("channel_1 Fourier Coefficients Filtered")
plt.grid()
plt.plot(f_1, abs(c_1filtered))
plt.savefig("channel_1_Fourier_filtered.png")


# Converting the filtered Fourier Coefficients back to Amplitude vs. Time
channel_0_out = irfft(c_0filtered)
channel_1_out = irfft(c_1filtered)


# Plotting Amplitude vs. Time of filtered sound
plt.figure(7, figsize=(8, 3), dpi=200)
plt.xlabel("time (s)")
plt.ylabel("Amplitude")
plt.title("channel_0 filtered")
plt.grid()
plt.plot(t, channel_0_out)
plt.savefig("channel_0_filtered.png")

plt.figure(8, figsize=(8, 3), dpi=200)
plt.xlabel("time (s)")
plt.ylabel("Amplitude")
plt.title("channel_1 filtered")
plt.grid()
plt.plot(t, channel_1_out)
plt.savefig("channel_1_filtered.png")


# Exporting the filtered sound
data_out = np.empty(data.shape, dtype=data.dtype)
data_out[:, 0] = channel_0_out
data_out[:, 1] = channel_1_out

# Saving new audio file
write("GraviteaTime_filtered.wav", sample, data_out)
