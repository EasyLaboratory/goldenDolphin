import numpy as np


def AM_modulation(t: np.ndarray, modulating_signal: np.ndarray, carrier_freq: np.uint16,
                  modulation_index: np.float32) -> tuple[np.ndarray, np.ndarray]:
    """
    :param t:time vector of the carrier signal
    :param modulating_signal:modulating_signal in np.ndarray
    :param carrier_freq:modulating frequency in HZ
    :param modulation_index:modulation index range from 0-1
    :return:
    modulated_signal: the modulated signal
    """

    # Modulating signal (a cosine wave with changing frequency)
    carrier_signal = np.cos(2 * np.pi * carrier_freq * t)
    # AM signal
    modulated_signal = (1 + modulation_index * modulating_signal) * carrier_signal / 2
    return t, modulated_signal


def FSK_modulation(t: np.ndarray, modulating_signal: np.ndarray, fsk_frequency: np.ndarray):
    modulated_signal = []
    for i, bite_signal in enumerate(modulating_signal):
        f = fsk_frequency[bite_signal]
        modulating_signal = np.sin(2 * np.pi * f * t)
        modulated_signal.append(modulating_signal)
    return np.array(modulated_signal)


def generate_fm_signal(sampling_rate, start_frequency, end_frequency, duration):
    """
    Generate a frequency modulation (FM) signal.

    Parameters:
    sampling_rate (float): Sampling rate in Hz.
    start_frequency (float): Start frequency of the FM signal in Hz.
    end_frequency (float): End frequency of the FM signal in Hz.
    duration (float): Duration of the signal in seconds.

    Returns:
    numpy.ndarray: The generated FM signal as an int16 array.
    """
    # Calculate the number of samples
    num_samples = int(sampling_rate * duration)

    # Create a time array
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Create a frequency modulation (FM) signal
    instantaneous_frequency = np.linspace(start_frequency, end_frequency, num_samples)
    phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / sampling_rate
    fm_signal = np.cos(phase)

    # Convert the signal to int16
    fm_signal_int16 = np.int16(fm_signal * np.iinfo(np.int16).max)

    return fm_signal_int16


import matplotlib.pyplot as plt


def generate_fsk_signal(bit_sequence, f1, f2, sampling_rate, duration):
    """
    Generate a Frequency Shift Keying (FSK) signal.

    Parameters:
    bit_sequence (list): Sequence of bits (0 and 1).
    f1 (float): Frequency for bit 0.
    f2 (float): Frequency for bit 1.
    sampling_rate (int): Sampling rate for the signal.
    duration (float): Duration of each bit.

    Returns:
    np.ndarray: The FSK modulated signal.
    np.ndarray: Time array for the signal.
    """

    # 总的样本数
    total_samples = int(duration * sampling_rate * len(bit_sequence))

    # 创建时间轴
    t = np.linspace(0, duration * len(bit_sequence), total_samples)

    # 生成FSK信号
    fsk_signal = np.zeros(total_samples)
    for i, bit in enumerate(bit_sequence):
        f = f1 if bit == 0 else f2
        start = int(i * duration * sampling_rate)
        end = int((i + 1) * duration * sampling_rate)
        fsk_signal[start:end] = np.sin(2 * np.pi * f * t[start:end])

    return fsk_signal, t


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    """
    设计一个低通滤波器。
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    """
    对数据应用低通滤波器。
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def orthogonal_demodulation(received_signal, fc, fs):
    """
    对接收信号进行正交解调。

    Parameters:
    received_signal (np.ndarray): 接收到的调制信号
    fc (float): 载波频率
    fs (float): 采样频率

    Returns:
    np.ndarray: 解调后的I分量
    np.ndarray: 解调后的Q分量
    """
    # 创建时间轴
    t = np.arange(len(received_signal)) / fs

    # 正交载波
    cos_carrier = np.cos(2 * np.pi * fc * t)
    sin_carrier = np.sin(2 * np.pi * fc * t)

    # 混频
    mixed_I = received_signal * cos_carrier
    mixed_Q = received_signal * sin_carrier

    # 低通滤波
    I = lowpass_filter(mixed_I, fc, fs)
    Q = lowpass_filter(mixed_Q, fc, fs)

    return I, Q


if __name__ == "__main__":
    # 示例使用
    bit_sequence = [0, 1, 1, 0, 1]
    f1, f2 = 1, 3
    sampling_rate = 100
    duration = 1

    fsk_signal, t = generate_fsk_signal(bit_sequence, f1, f2, sampling_rate, duration)

    I, Q = orthogonal_demodulation(fsk_signal, 2, 100)

    plt.scatter(I, Q)
    plt.title('Constellation Diagram')
    plt.xlabel('In-phase Component (I)')
    plt.ylabel('Quadrature Component (Q)')
    plt.grid(True)
    plt.show()

    # # 绘制FSK信号
    # plt.figure(figsize=(10, 4))
    # plt.plot(t, fsk_signal)
    # plt.title('Frequency Shift Keying (FSK) Signal')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()
