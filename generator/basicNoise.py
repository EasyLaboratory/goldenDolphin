import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


def add_AWGN_noise(t: np.ndarray, signal: np.ndarray, snr_db: np.int8)->tuple[np.ndarray,np.ndarray]:
    """
    Generate Additive White Gaussian Noise (AWGN) and add it to a signal.
    :param t:
    :param signal:
    :param snr_db:
    :return:
    """

    # Calculate signal power
    # todo int16 数据溢出，平方后值为1
    print(signal)
    print(signal**2)
    signal_power = np.mean(signal ** 2)
    print(signal_power)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    print(noise)

    # Add noise to the signal
    noisy_signal = signal + noise

    return t, noisy_signal


def generate_rician_fading(num_samples, k_factor_db):
    """
    Generate Rician fading noise.

    Parameters:
    - num_samples: Number of fading samples to generate
    - k_factor_db: K-factor in dB

    Returns:
    - rician_fading: Generated Rician fading samples
    """

    # Convert K-factor from dB to linear scale
    k_factor_linear = 10 ** (k_factor_db / 10)

    # Generate I and Q components of multipath component
    std_dev_multipath = np.sqrt(1 / (2 * (k_factor_linear + 1)))
    i_multipath = std_dev_multipath * np.random.randn(num_samples)
    q_multipath = std_dev_multipath * np.random.randn(num_samples)

    # Generate LOS component
    std_dev_los = np.sqrt(k_factor_linear / (k_factor_linear + 1))
    i_los = std_dev_los
    q_los = 0

    # Combine LOS and multipath components
    i_total = i_los + i_multipath
    q_total = q_los + q_multipath

    # Calculate Rician fading amplitude
    rician_fading = np.sqrt(i_total ** 2 + q_total ** 2)

    return rician_fading


def generate_clock_offset(signal, fs, offset_percent):
    """
    Apply clock offset to a signal.

    Parameters:
    - signal: Original signal
    - fs: Sampling frequency of the signal
    - offset_percent: Clock offset as a percentage of the sampling frequency

    Returns:
    - signal_with_offset: Signal after applying clock offset
    """

    # Create original and offset time vectors
    t_original = np.arange(len(signal)) / fs
    offset = fs * offset_percent / 100
    t_offset = np.arange(len(signal)) / (fs + offset)

    # Interpolate signal to new time vector
    interpolate = interp1d(t_original, signal, kind='linear', fill_value="extrapolate")
    signal_with_offset = interpolate(t_offset)

    return signal_with_offset
