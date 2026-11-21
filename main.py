import hydra
import numpy as np
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from data import EEGData



@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    # Hydra runs inside outputs/<date>/<time>, so the data path is made absolute first
    fs = config.data.sampling_rate
    eeg_data = EEGData(to_absolute_path(config.data.path), fs)

    # instantiate() builds the class named by _target_ in each config file
    preprocessing = instantiate(config.preprocessing)
    artifact_removal = instantiate(config.artifact_removal) #asr
    filter_bank = instantiate(config.filtering)
    feature_extraction = instantiate(config.feature_extraction)

    raw = eeg_data.raw_data_list[0]
    signal = raw.get_data(picks="eeg")  # (channels, samples), without the time column
    markers = np.array(eeg_data.labels_list[0], dtype=str)

    # 1. Filtering: mean removal, notch and band-pass on the raw signal
    filtered = preprocessing(signal)

    # 2. ASR: detect and reconstruct noisy components of the filtered broadband signal
    cleaned = artifact_removal(filtered)
    
    # 3. Other preprocessing: split into frequency bands, then epoch around the cues
    bands = filter_bank(cleaned)
    trials, cues = cut_trials(bands, markers, ["move", "rest"], config.window.tmin, config.window.tmax, fs)

    # 4. Feature extraction
    bands = filter_bank(preprocessing(signal))
    trials, cues = preprocessing.epoch_data(bands, markers, ["move", "rest"], config.window.tmin, config.window.tmax, fs)
    features = feature_extraction.fit_transform(trials)

    # 5. Machine learning model: not implemented yet (models/ is empty)


if __name__ == "__main__":
    main()
