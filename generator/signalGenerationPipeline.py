import matplotlib.pyplot as plt

from generator.basicModulationSignal import *
from generator.basicCarrierSignal import *
from generator.basicNoise import *
from utils.signalUtils import *


def AM_Signal_pipeline():
    # generate carrier signal
    frequency = 100e6
    duration = 0.001
    sampling_rate = 512e6
    t, carrier_signal = cosine_wave(frequency, sampling_rate, duration)

    # generate modulating signal
    modulating_freq = np.uint16(50e6)
    modulation_index = np.float32(1)
    t, signal = AM_modulation(t, carrier_signal, modulating_freq, modulation_index)
    t, signal = add_AWGN_noise(t, signal, np.int8(10))
    converted_signal = convert_signal_datatype(signal, data_type=np.int16)
    sampled_t, sampled_signal = down_sampling(t, converted_signal, 10, (100, 16000))
    plt.plot(sampled_t, sampled_signal)
    plt.show()


def FM_signal_pipeline():
    pass


def FSK_signal_pipeline():
    signal_duration = 1
    sampling_rate = 50
    modulating_signal = digital_random_signal(signal_duration, sampling_rate, [0, 1])
    print(modulating_signal)
    FSK_freq = np.array([10e6, 20e6])
    t = np.arange(len(modulating_signal))
    res = FSK_modulation(t, modulating_signal, FSK_freq)
    plt.plot(t, res)
    plt.show()





