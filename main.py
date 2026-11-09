import hydra
import numpy as np
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from data import EEGData


def cut_trials(signal, markers, cue_names, tmin, tmax, fs):
    """Cut one trial per cue, from cue + tmin to cue + tmax seconds"""
    onsets = np.flatnonzero(np.isin(markers, cue_names))
    start = round(tmin * fs)
    length = round((tmax - tmin) * fs)
    trials = [signal[..., onset + start : onset + start + length] for onset in onsets]
    return np.stack(trials), markers[onsets]


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    # Hydra runs inside outputs/<date>/<time>, so the data path is made absolute first
    fs = config.data.sampling_rate
    eeg_data = EEGData(to_absolute_path(config.data.path), fs)

    # instantiate() builds the class named by _target_ in each config file
    preprocessing = instantiate(config.preprocessing)
    filter_bank = instantiate(config.filtering)
    feature_extraction = instantiate(config.feature_extraction)

    raw = eeg_data.raw_data_list[0]
    signal = raw.get_data(picks="eeg")  # (channels, samples), without the time column
    markers = np.array(eeg_data.labels_list[0], dtype=str)

    bands = filter_bank(preprocessing(signal))
    trials, cues = cut_trials(bands, markers, ["move", "rest"], config.window.tmin, config.window.tmax, fs)
    features = feature_extraction.fit_transform(trials)


if __name__ == "__main__":
    main()
