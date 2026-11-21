from typing import List

import numpy as np
import mne


class ICA:
    def __init__(self, ica_components: float, channels: List[str], max_iter: int, random_state: int, method: str="picard"):
        self.ica_components = ica_components
        self.channels = channels
        self.method = method
        self.random_state = random_state
        self.max_iter = max_iter
        self.ica = mne.preprocessing.ICA(
            n_components=self.ica_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

    def __call__(self, X: mne.io.Raw):
        self.ica.fit(X)
        eog_indices, _ = self.ica.find_bads_eog(
            X, ch_name=self.channels
        )
        self.ica.exclude = eog_indices
        
        return self.ica.apply(X)

    
