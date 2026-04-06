"""
Estimate PLV on synthetic data for testing

Total number of channels:
        Using 15 channels provides a fairly complete validation set.

Channel description:
    Ch1:  10 Hz reference signal
    Ch2:  Exact copy
    Ch3:  10 Hz with +90° phase shift
    Ch4: Ch1 + Low noise
    Ch5: Ch1 + Medium noise
    Ch6: Ch1 + High noise
    Ch7:  Independent band-limited noise
    Ch8: 10 Hz with independently randomized phase per epoch


This configuration allows testing:
    - High PLV under constant phase differences
    - Independence from specific phase lag values
    - Sensitivity to noise (SNR effects)
    - Null connectivity cases

Federico Ramírez-Toraño
01/04/2026

"""

# Imports
import numpy
import scipy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from sEEGnal.tools.fc_tools import compute_plv, reconstruct_fc_matrix


# Create the synthetic matrix
fs = 500
n_epochs = 20
n_channels = 8
n_seconds = 4
n_samples = n_seconds*fs
data = numpy.zeros((n_epochs, n_channels, n_samples))

# Ch1:  10 Hz reference signal
# Ch2: Exact copy
A = 1
f = 10
fs = 500
phi = 0
t = numpy.linspace(0, n_seconds, n_samples)
ch1 = A * numpy.sin(2 * numpy.pi * f * t + phi)
data[:,0,:] = numpy.matlib.repmat(ch1, n_epochs,1)
data[:,1,:] = numpy.matlib.repmat(ch1, n_epochs,1)

# Ch3:  10 Hz with +90° phase shift
phi = numpy.pi/2
ch3 = A * numpy.sin(2 * numpy.pi * f * t + phi)
data[:,2,:] = numpy.matlib.repmat(ch3, n_epochs,1)

# Ch4: Ch1 + Low noise
noise_ratio = 0.1
noise = numpy.random.normal(0,noise_ratio*A, len(ch1))
ch4 = ch1 + noise
data[:,3,:] = numpy.matlib.repmat(ch4, n_epochs,1)

# Ch5: Ch1 + Low noise
noise_ratio = 0.5
noise = numpy.random.normal(0,noise_ratio*A, len(ch1))
ch5 = ch1 + noise
data[:,4,:] = numpy.matlib.repmat(ch5, n_epochs,1)

# Ch6: Ch1 + Low noise
noise_ratio = 0.8
noise = numpy.random.normal(0,noise_ratio*A, len(ch1))
ch6 = ch1 + noise
data[:,5,:] = numpy.matlib.repmat(ch6, n_epochs,1)

# Ch7:  Independent band-limited noise
noise = numpy.random.normal(0,A, len(ch1))
sos = scipy.signal.butter(5, [8, 12], 'bandpass', fs=fs, output='sos')
ch7 = scipy.signal.sosfiltfilt(sos, noise)
data[:,6,:] = numpy.matlib.repmat(ch7, n_epochs,1)

# Ch8: 10 Hz with independently randomized phase per epoch
for iepoch in range(n_epochs):
    phi = numpy.random.uniform(0, 2 * numpy.pi)
    data[iepoch, 7, :] = A * numpy.sin(2 * numpy.pi * f * t + phi)

# Compute plv
plv_vector = compute_plv(
                data=data,
                average_epochs=True
            )
conn_indices = numpy.triu_indices(n_channels, k=1)
plv_matrix = reconstruct_fc_matrix(plv_vector, conn_indices, n_channels)

print(plv_matrix)
plt.imshow(plv_matrix)
plt.colorbar()
plt.show(block=True)



