import numpy as np

from data.preprocessing import bandpass


class FilterBank:
    """Split a preprocessed signal (channels, samples) into frequency bands: (bands, channels, samples)"""

    def __init__(self, sampling_rate, bands):
        self.fs = sampling_rate
        self.bands = bands

    def __call__(self, signal):
        filtered = []
        for low, high in self.bands:
            filtered.append(bandpass(signal, self.fs, low, high))
        return np.stack(filtered)
