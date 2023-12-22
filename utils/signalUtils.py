import numpy as np


def down_sampling(t: np.ndarray, signal: np.ndarray, down_sample_factor: int, time_window: tuple[int, int]) -> (
        tuple)[np.ndarray, np.ndarray]:
    """
    down sampling the signal
    :param t:
    :param signal:
    :param down_sample_factor:
    :param time_window:
    :return:
    down_sampling_signal:sampled signal in ndarray
    """
    down_sampling_t = t[time_window[0]:time_window[1]:down_sample_factor]
    down_sampling_signal = signal[time_window[0]:time_window[1]:down_sample_factor]
    return down_sampling_t, down_sampling_signal


def convert_signal_datatype(signal: np.ndarray, data_type: np.dtype) -> np.ndarray:
    converted_signal = (signal * np.iinfo(data_type).max).astype(data_type)
    return converted_signal
