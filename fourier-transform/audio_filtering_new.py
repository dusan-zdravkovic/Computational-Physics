"""
Audio Signal Filtering Script

Author: Dusan Zdravkovic
Purpose: Filter out unwanted sound from an audio file and export a new filtered file.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from numpy.fft import rfft, irfft, rfftfreq

# Set Plot Aesthetics
plt.style.use("seaborn-darkgrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# Read the audio file
sample, data = read("GraviteaTime.wav")
channel_0, channel_1 = data[:, 0], data[:, 1]
N_Points = len(channel_0)

# Time and frequency domain calculations
dt = 1 / sample  # Time step
T = N_Points * dt  # Length of interval
freq = np.arange(N_Points / 2 + 1) * 2 * np.pi / T  # Angular frequency
t = np.arange(N_Points) * dt  # Dimensional time axis

# Plotting the unfiltered sound
fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=200)

axs[0, 0].plot(t, channel_0, color="tab:blue")
axs[0, 0].set_title("Channel 0 - Unfiltered")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Amplitude")

axs[0, 1].plot(t, channel_1, color="tab:green")
axs[0, 1].set_title("Channel 1 - Unfiltered")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Amplitude")

# Fourier Transforms
c_0 = rfft(channel_0)
c_1 = rfft(channel_1)
f_0 = rfftfreq(N_Points, dt)
f_1 = rfftfreq(N_Points, dt)

# Filter frequencies greater than 880 Hz
c_0filtered = np.where(f_0 > 880, 0, c_0)
c_1filtered = np.where(f_1 > 880, 0, c_1)

# Plotting Fourier Coefficients
axs[1, 0].plot(f_0, abs(c_0filtered), color="tab:red")
axs[1, 0].set_title("Channel 0 - Filtered Fourier Coefficients")
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_ylabel("Fourier Coefficient")

axs[1, 1].plot(f_1, abs(c_1filtered), color="tab:orange")
axs[1, 1].set_title("Channel 1 - Filtered Fourier Coefficients")
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Fourier Coefficient")

# Layout adjustments
plt.tight_layout()
plt.savefig("Audio_Filtering_Analysis.png")
plt.show()

# Convert filtered Fourier coefficients back to time domain
channel_0_out = irfft(c_0filtered)
channel_1_out = irfft(c_1filtered)

# Export filtered audio
data_out = np.column_stack((channel_0_out, channel_1_out))
write("GraviteaTime_filtered.wav", sample, data_out)
