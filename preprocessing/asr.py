import warnings

import asrpy
import mne


class ASR:
    """Artifact Subspace Reconstruction (asrpy) for a filtered signal (channels, samples)

    -> Calibration: find the clean windows of the recording and estimate a rejection threshold
    per principal component.
    -> Processing: in each sliding window, components whose variance
    exceeds the threshold are removed and reconstructed from the remaining clean components.
    """

    def __init__(
        self,
        sampling_rate,
        cutoff=20.0,
        window_length=0.5,
        window_overlap=0.66,
        max_bad_channels=0.1,
        step_size=32,
        lookahead=0.25,
        max_dims=0.66,
    ):
        self.fs = sampling_rate
        self.cutoff = (
            cutoff  # in standard deviations; lower is more aggressive (typical 10-30)
        )
        self.window_length = window_length  # in seconds
        self.window_overlap = window_overlap
        self.max_bad_channels = max_bad_channels  # fraction of bad channels allowed in a clean calibration window
        self.step_size = step_size  # samples between two reconstruction matrix updates
        self.lookahead = lookahead  # in seconds
        self.max_dims = max_dims  # max fraction of components removed per window

    def _to_raw(self, signal):
        info = mne.create_info(signal.shape[0], self.fs, "eeg")
        return mne.io.RawArray(signal, info, verbose=False)

    def fit(self, signal):
        self.asr_ = asrpy.ASR(
            sfreq=self.fs,
            cutoff=self.cutoff,
            win_len=self.window_length,
            win_overlap=self.window_overlap,
            max_bad_chans=self.max_bad_channels,
        )
        with warnings.catch_warnings():
            # asrpy 0.0.8 relies on a numpy conversion deprecated since numpy 1.25
            warnings.simplefilter("ignore", DeprecationWarning)
            self.asr_.fit(self._to_raw(signal))
        return self

    def transform(self, signal):
        if not hasattr(self, "asr_"):
            raise RuntimeError("ASR must be fitted before transform")
        cleaned = self.asr_.transform(
            self._to_raw(signal),
            lookahead=self.lookahead,
            stepsize=self.step_size,
            maxdims=self.max_dims,
        )
        return cleaned.get_data()

    def __call__(self, signal):
        """Calibrate on the recording itself, then clean it"""
        return self.fit(signal).transform(signal)
