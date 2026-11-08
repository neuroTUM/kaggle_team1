import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import mne


class EEGData:
    def __init__(self, folder_path: str):
        """
        Load all files of a folder as a list of MNE raw array and labels
        """
        self.folder_path = Path(folder_path)
        self.raw_data_list, self.labels_list = self._load_files(self.folder_path)

    def _load_files(self, folder: Path) -> Tuple[List, List]:
        raw_data_list = []
        labels_list = []
        for file in os.listdir(folder):
            df = pd.read_csv(folder / file)
            data = df.transpose().to_numpy()

            ch_names = list(df.columns[:-1]) # remove last channel
            ch_types = ['misc'] + ['eeg'] * (data.shape[0] - 2)
            sfreq = 250 # Sampling frequency

            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

            raw = mne.io.RawArray(data[:-1], info)
            labels = data[-1]

            raw_data_list.append(raw)
            labels_list.append(labels)

        return raw_data_list, labels_list
            