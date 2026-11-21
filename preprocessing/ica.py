from typing import Literal

import mne
import numpy as np


class ICA:
    def __init__(
        self,
        ica_components: float,
        ica_measure: Literal["zscore", "correlation"],
        channels: list[str],
        max_iter: int,
        random_state: int,
        method: str = "picard",
    ):
        self.ica_components = ica_components
        self.ica_measure = ica_measure
        self.channels = channels
        self.method = method
        self.random_state = random_state
        self.max_iter = max_iter
        self.ica = mne.preprocessing.ICA(
            n_components=self.ica_components,
            random_state=self.random_state,
            method=self.method,
            max_iter=self.max_iter,
        )

    def __call__(self, X: mne.io.Raw, cleaned: np.ndarray):
        eeg_picks = mne.pick_types(X.info, eeg=True)
        X = X.copy()
        X._data[eeg_picks] = cleaned

        self.ica.fit(X)
        eog_indices, _ = self.ica.find_bads_eog(
            X, ch_name=self.channels, measure=self.ica_measure
        )
        self.ica.exclude = eog_indices

        new_data = self.ica.apply(X)
        return new_data.get_data(picks="eeg")
