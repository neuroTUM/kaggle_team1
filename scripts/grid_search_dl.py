"""Grid search over EEGNet / training settings using per-subject cross-validation.

Reuses the loading, preprocessing and training code from main_dl.py without changing
it. Only cross-validation is run: no final model and no submission. Run from the repo
root as a module so main_dl and models/ can be imported:

    uv run python -m scripts.grid_search_dl
    uv run python -m scripts.grid_search_dl "subjects=[S001,S009]" training.epochs=50

Writes to the Hydra run folder:
- grid_results.csv: one row per (combination, subject), appended as it goes
- grid_summary.csv: combinations ranked by mean CV accuracy across subjects
"""

import copy
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from main_dl import LABELS, epoch, get_device, load_csv, predict, train, zscore

WINDOW_KEYS = ("window.tmin", "window.tmax")


def load_raw(config: DictConfig, subjects: list[str]) -> dict:
    """Read every training CSV once: {subject: [(signal, markers, file_name), ...]}"""
    files = defaultdict(list)
    for path in sorted(Path(to_absolute_path(config.data.path)).glob("*.csv")):
        files[path.name[:4]].append(path)  # "S001-001_eeg.csv" -> "S001"
    raw = {}
    for subject in subjects:
        print(f"{subject}: loading")
        raw[subject] = [(*load_csv(p), p.name) for p in files[subject]]
    return raw


def make_trials(config: DictConfig, raw: dict, preprocessing) -> dict:
    """Filter, epoch and z-score each subject's runs: {subject: (X, y)}"""
    data = {}
    for subject, runs in raw.items():
        X_parts, y_parts = [], []
        for signal, markers, name in runs:
            trials, cues = epoch(
                signal,
                markers,
                list(LABELS),
                preprocessing,
                config.window,
                config.data.sampling_rate,
                name,
            )
            X_parts.append(trials)
            y_parts.append([LABELS[c] for c in cues])
        data[subject] = (zscore(np.concatenate(X_parts)), np.concatenate(y_parts))
    return data


def cross_validate(config, X, y, device) -> float:
    """Same stratified k-fold CV as main_dl.py (same seed, so the same folds)"""
    seed = config.training.seed
    folds = StratifiedKFold(config.training.n_folds, shuffle=True, random_state=seed)
    accs = []
    for k, (tr, va) in enumerate(folds.split(X, y)):
        model = train(config, X[tr], y[tr], seed + k, device)
        accs.append(float((predict(model, X[va], device) == y[va]).mean()))
    return float(np.mean(accs))


@hydra.main(version_base=None, config_path="../config", config_name="grid-search/grid_search_dl")
def main(config: DictConfig):
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    results_path = out_dir / "grid_results.csv"
    device = get_device(config.training.device)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU")

    subjects = config.subjects
    if subjects is None:
        test_dir = Path(to_absolute_path(config.data.test_path))
        subjects = sorted(p.name[:4] for p in test_dir.glob("*.csv"))
    subjects = list(subjects)

    search = OmegaConf.to_container(config.search, resolve=True)
    # Window settings change the data, so make them the outer loop and epoch
    # once per window instead of once per combination.
    grid = sorted(
        ParameterGrid(search),
        key=lambda p: tuple(
            p.get(k, config.window[k.split(".")[1]]) for k in WINDOW_KEYS
        ),
    )
    print(f"{len(grid)} combinations x {len(subjects)} subjects")

    preprocessing = instantiate(config.preprocessing.preprocessing)
    raw = load_raw(config, subjects)
    data, data_window = None, None
    param_names = list(search)

    for i, params in enumerate(grid, 1):
        run_config = copy.deepcopy(config)
        for key, value in params.items():
            OmegaConf.update(run_config, key, value)

        window = (run_config.window.tmin, run_config.window.tmax)
        if window != data_window:
            print(f"Epoching with window {window}")
            data, data_window = make_trials(run_config, raw, preprocessing), window

        start = time.time()
        rows = []
        for subject in subjects:
            X, y = data[subject]
            acc = cross_validate(run_config, X, y, device)
            rows.append(
                {"combination": i, **params, "subject": subject, "cv_accuracy": acc}
            )
        mean = np.mean([r["cv_accuracy"] for r in rows])
        print(
            f"[{i}/{len(grid)}] {params} -> mean CV acc {mean:.3f} "
            f"({time.time() - start:.0f} s)"
        )
        # Append after every combination so a crash keeps the finished ones
        pd.DataFrame(rows).to_csv(
            results_path, mode="a", header=not results_path.exists(), index=False
        )

    results = pd.read_csv(results_path)
    summary = (
        results.groupby(["combination", *param_names])["cv_accuracy"]
        .agg(mean_cv_accuracy="mean", std_across_subjects="std")
        .reset_index()
        .sort_values("mean_cv_accuracy", ascending=False)
    )
    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    print("\nTop combinations:")
    print(summary.head(10).to_string(index=False))
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
