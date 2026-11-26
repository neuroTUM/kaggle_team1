"""Deep-learning pipeline: EEGNet trained per subject.

Only DL-allowed preprocessing is used: mean removal, 50 Hz notch, 1-100 Hz band-pass
and per-trial z-scoring. Runs 1-3 are the training data; the test file is predicted.
"""

from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch import nn

EEG_CHANNELS = ["Fz", "C3", "Cz", "C4", "PO7", "Pz", "PO8", "Oz"]
LABELS = {"rest": 0, "move": 1}


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the EEG signal (channels, samples) and the marker column"""
    df = pd.read_csv(path, dtype={"Marker_val": str})
    signal = df[EEG_CHANNELS].to_numpy().T
    markers = df["Marker_val"].fillna("").to_numpy(dtype=str)
    return signal, markers


def epoch(signal, markers, cue_names, preprocessing, window, fs, file_name):
    """Filter the continuous signal, drop cues whose window passes the end, then epoch"""
    filtered = preprocessing(signal)
    markers = markers.copy()
    end = round(window.tmax * fs)
    onsets = np.flatnonzero(np.isin(markers, cue_names))
    too_late = onsets[onsets + end > signal.shape[-1]]
    if len(too_late):
        print(f"  warning: {file_name}: dropping {len(too_late)} cue(s) at end of file")
        markers[too_late] = ""
    return preprocessing.epoch_data(
        filtered, markers, cue_names, window.tmin, window.tmax, fs
    )


def zscore(trials: np.ndarray) -> np.ndarray:
    """Normalise each trial and channel to zero mean, unit variance"""
    mean = trials.mean(axis=-1, keepdims=True)
    std = trials.std(axis=-1, keepdims=True) + 1e-8
    return (trials - mean) / std


def to_tensor(trials: np.ndarray) -> torch.Tensor:
    return torch.tensor(trials, dtype=torch.float32).unsqueeze(1)  # (n, 1, ch, t)


def train(config, X: np.ndarray, y: np.ndarray, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = instantiate(config.model, n_channels=X.shape[1], n_samples=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t, y_t = to_tensor(X), torch.tensor(y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(config.training.epochs):
        order = torch.randperm(len(X_t), generator=generator)
        for i in range(0, len(order), config.training.batch_size):
            batch = order[i : i + config.training.batch_size]
            if len(batch) < 2:  # BatchNorm needs more than one sample
                continue
            optimizer.zero_grad()
            loss_fn(model(X_t[batch]), y_t[batch]).backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(to_tensor(X)).argmax(dim=1).numpy()


@hydra.main(version_base=None, config_path="config", config_name="config_dl")
def main(config: DictConfig):
    fs = config.data.sampling_rate
    seed = config.training.seed
    preprocessing = instantiate(config.preprocessing)
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    train_files = defaultdict(list)
    for path in sorted(Path(to_absolute_path(config.data.path)).glob("*.csv")):
        train_files[path.name[:4]].append(path)  # "S001-001_eeg.csv" -> "S001"
    test_files = {
        p.name[:4]: p
        for p in sorted(Path(to_absolute_path(config.data.test_path)).glob("*.csv"))
    }

    results, predictions = [], []
    for subject in sorted(test_files):
        print(f"{subject}: loading")
        X_parts, y_parts = [], []
        for path in train_files[subject]:
            signal, markers = load_csv(path)
            trials, cues = epoch(
                signal,
                markers,
                list(LABELS),
                preprocessing,
                config.window,
                fs,
                path.name,
            )
            X_parts.append(trials)
            y_parts.append([LABELS[c] for c in cues])
        X, y = zscore(np.concatenate(X_parts)), np.concatenate(y_parts)

        signal, markers = load_csv(test_files[subject])
        X_test, _ = epoch(
            signal,
            markers,
            ["cue_start"],
            preprocessing,
            config.window,
            fs,
            test_files[subject].name,
        )
        X_test = zscore(X_test)

        # Cross-validation on the subject's training trials
        folds = StratifiedKFold(
            config.training.n_folds, shuffle=True, random_state=seed
        )
        fold_acc = []
        for k, (tr, va) in enumerate(folds.split(X, y)):
            model = train(config, X[tr], y[tr], seed + k)
            fold_acc.append(float((predict(model, X[va]) == y[va]).mean()))
        acc = float(np.mean(fold_acc))
        print(
            f"{subject}: {len(y)} train / {len(X_test)} test trials, CV acc {acc:.3f}"
        )
        results.append({"subject": subject, "cv_accuracy": acc})

        # Final model on all training trials, then predict the test trials
        model = train(config, X, y, seed)
        names = {v: k for k, v in LABELS.items()}
        predictions += [names[p] for p in predict(model, X_test)]

    results = pd.DataFrame(results)
    print(f"Mean CV accuracy: {results.cv_accuracy.mean():.3f}")
    results.to_csv(out_dir / "cv_results.csv", index=False)
    pd.DataFrame({"ID": range(len(predictions)), "TARGET": predictions}).to_csv(
        out_dir / "submission.csv", index=False
    )
    print(f"Wrote {len(predictions)} predictions to {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
