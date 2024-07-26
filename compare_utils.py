import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def read_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def normalize(data):
    return data / np.max(np.abs(data))

def compute_mse(data1, data2):
    return np.mean((data1 - data2) ** 2)

def plot_waveforms(data1, data2, sample_rate, title1='DAW Output', title2='Python Library Output'):
    time = np.arange(data1.shape[0]) / sample_rate
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, data1, label=title1)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(time, data2, label=title2)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
daw_output_path = './.mixes/rp-op.wav'
python_output_path = './.mixes/py-op.wav' 

sample_rate_daw, daw_data = read_wav(daw_output_path)
sample_rate_py, py_data = read_wav(python_output_path)

# Ensure the sample rates are the same
assert sample_rate_daw == sample_rate_py, "Sample rates do not match!"
print(daw_data.shape)
print(py_data.shape)
# Normalize the data
daw_data_norm = normalize(daw_data)
py_data_norm = normalize(py_data)

# Compute MSE
# mse = compute_mse(daw_data_norm, py_data_norm)
# print(f"Mean Squared Error (MSE): {mse}")

# Plot the waveforms
plot_waveforms(daw_data_norm, py_data_norm, sample_rate_daw)