import os
from pathlib import Path

import mne
import pandas as pd
from tqdm import tqdm


class EEGData:
    def __init__(self, folder_path: str, sampling_rate: float):
        """
        Load all files of a folder as a list of MNE raw array and labels
        """
        self.folder_path = Path(folder_path)
        self.sampling_rate = sampling_rate
        self.subject_list, self.session_list, self.raw_data_list, self.labels_list = (
            self._load_files(self.folder_path)
        )

    def _load_files(self, folder: Path) -> tuple[list, list, list, list]:
        subject_list = []
        session_list = []
        raw_data_list = []
        labels_list = []
        for file in tqdm(os.listdir(folder), desc="Loading files"):
            df = pd.read_csv(folder / file, dtype={"Marker_val": str})
            data = df.transpose().to_numpy()

            ch_names = list(df.columns[:-1])  # remove last channel
            ch_types = ["misc"] + ["eeg"] * (data.shape[0] - 2)
            info = mne.create_info(
                ch_names=ch_names, ch_types=ch_types, sfreq=self.sampling_rate
            )

            subject = file[:4]
            session = int(file[5:8])
            raw = mne.io.RawArray(data[:-1], info)
            labels = data[-1]

            subject_list.append(subject)
            raw_data_list.append(raw)
            labels_list.append(labels)
            session_list.append(session)

        return subject_list, session_list, raw_data_list, labels_list
