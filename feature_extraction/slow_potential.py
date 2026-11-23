import numpy as np
from mne.baseline import rescale

from .feature_extraction_class import FeatureExtractionClass


class SlowPotentialFeatures(FeatureExtractionClass):
    """Mean voltage per channel in consecutive windows, relative to a baseline period"""

    def __init__(
        self, sampling_rate, tmin, baseline, start, window_length, channels=None
    ):
        self.sampling_rate = sampling_rate
        self.tmin = tmin
        self.baseline = baseline
        self.start = start
        self.window_length = window_length
        self.channels = channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        times = self.tmin + np.arange(X.shape[-1]) / self.sampling_rate
        if self.channels is not None:
            X = X[..., list(self.channels), :]
        X = rescale(X, times, tuple(self.baseline), mode="mean", verbose=False)
        first = round((self.start - self.tmin) * self.sampling_rate)
        return window_means(
            X[..., first:], round(self.window_length * self.sampling_rate)
        )


def window_means(X, window_samples):
    """Mean of each full window of window_samples along the last axis, one row per trial"""
    n_windows = X.shape[-1] // window_samples
    windows = X[..., : n_windows * window_samples].reshape(
        *X.shape[:-1], n_windows, window_samples
    )
    return windows.mean(axis=-1).reshape(len(X), -1)
