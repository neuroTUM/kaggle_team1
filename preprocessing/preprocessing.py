import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def subtract_mean(signal):
    """Remove each channel's mean over the whole recording"""
    return signal - signal.mean(axis=-1, keepdims=True)


def notch(signal, fs, freq=50.0, quality=30.0):
    b, a = iirnotch(freq, Q=quality, fs=fs)
    return filtfilt(b, a, signal, axis=-1)


def bandpass(signal, fs, low, high, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def spectrum(signal, fs):
    """FFT"""
    n = signal.shape[-1]
    freqs = rfftfreq(n, d=1 / fs)
    amplitudes = np.abs(rfft(signal, axis=-1)) * 2 / n
    return freqs, amplitudes


class Preprocessing:
    def __init__(
        self, sampling_rate, notch_freq=50.0, low=1.0, high=80.0, subtract_mean=True
    ):
        self.fs = sampling_rate
        self.notch_freq = notch_freq
        self.low = low
        self.high = high
        self.subtract_mean = subtract_mean

    def __call__(self, signal):
        if self.subtract_mean:
            signal = subtract_mean(signal)
        signal = notch(signal, self.fs, self.notch_freq)
        signal = bandpass(signal, self.fs, self.low, self.high)
        return signal

    def epoch_data(
        self,
        signal: np.ndarray,
        markers: np.ndarray,
        cue_names: list[str],
        tmin: float,
        tmax: float,
        fs: int,
    ):
        """Cut one trial per cue, from cue + tmin to cue + tmax seconds"""
        onsets = np.flatnonzero(np.isin(markers, cue_names))
        start = round(tmin * fs)
        length = round((tmax - tmin) * fs)
        trials = [
            signal[..., onset + start : onset + start + length] for onset in onsets
        ]
        return np.stack(trials), markers[onsets]
