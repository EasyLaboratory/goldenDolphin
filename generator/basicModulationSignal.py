import numpy as np


def cosine_wave(frequency: float, sampling_rate: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """
       generate cosine carrier signal
       :param frequency: the frequency of signal in HZ
       :param sampling_rate: the sampling rate of the signal
       :param duration:  duration of the signal
       :return:
       """
    t = np.arange(0, duration, 1 / sampling_rate)
    carrier_signal = np.cos(2 * np.pi * frequency * t)  # Cosine signal
    return t, carrier_signal


def digital_random_signal(signal_duration: float, sample_rate: float, choice: list)->np.ndarray:
    return np.random.choice(choice, size=np.uint(signal_duration * sample_rate))



