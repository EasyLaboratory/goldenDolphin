import numpy as np


def add_noise_to_signal(signal, snr_db):
    """
    Add Gaussian noise to a signal based on a given signal-to-noise ratio (SNR).

    Parameters:
    - signal: Original signal to which noise will be added
    - snr_db: Desired Signal-to-Noise Ratio in dB (decibels)

    Returns:
    - noisy_signal: Signal with added noise
    """

    # Calculate signal power
    signal_power = np.mean(signal ** 2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise with calculated noise power
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))

    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal



