# Deep learning pipeline: EEGNet

A deep-learning entry for the single-subject competition. One EEGNet is trained from scratch per subject, using only that subject's data.

## Run

```sh
uv run python main_dl.py
uv run python main_dl.py training.epochs=100   # change any setting from the command line
```

Results go into the Hydra run folder `outputs/<date>/<time>/`:
- `cv_results.csv`: cross-validation accuracy per subject
- `submission.csv`: 680 predictions (`ID,TARGET`) ready for Kaggle

## Files

| File | Purpose |
|---|---|
| `models/dl/eegnet.py` | The EEGNet model (PyTorch), with no training code |
| `main_dl.py` | Loading, preprocessing, training, cross-validation and submission |
| `config/config_dl.yaml` | Top-level config for the DL pipeline |
| `config/model/eegnet.yaml` | Model settings |
| `config/training/training.yaml` | Epochs, batch size, learning rate, folds, seed |

## What the pipeline does

1. **Load**: for each subject, read the training runs 1–3 and the test file, keeping the 8 EEG channels.
2. **Filter** the continuous recording: mean removal, 50 Hz notch, 1–100 Hz band-pass. This uses the shared `Preprocessing` class.
3. **Epoch**: one trial from 0.5 s to 4.5 s after each cue (1000 samples). Training uses the `move`/`rest` cues and the test file uses `cue_start`.
4. **Normalise**: z-score each trial per channel.
5. **Validate**: stratified 5-fold cross-validation, with a new EEGNet in each fold.
6. **Predict**: retrain on all of the subject's trials and predict its 40 test trials.

**Rules compliance**: the input is only minimally filtered EEG with mean/std normalisation, as the rules allow for DL entries. There's no ICA, ASR, CSP, covariance or filter bank.

## Model

EEGNet-8,2 (Lawhern et al., 2018). It's a small convolutional network with about 2–3k trainable weights:
temporal convolution → depthwise spatial convolution → separable convolution → linear classifier.
It isn't pre-trained: every model starts from random weights.

Training settings: Adam, learning rate 1e-3, cross-entropy loss, 200 epochs, batch size 16, seed 42.

## Results (first run, no hyperparameter tuning)

**Mean 5-fold CV accuracy: 61.4%** (chance level is 50%). The full run takes about 22 minutes on CPU.

| Subject | Acc | Subject | Acc | Subject | Acc |
|---|---|---|---|---|---|
| S001 | 0.636 | S007 | 0.617 | S014 | 0.600 |
| S002 | 0.545 | S009 | 0.736 | S016 | 0.691 |
| S003 | 0.536 | S010 | 0.691 | S017 | 0.727 |
| S004 | 0.645 | S011 | 0.573 | S018 | 0.518 |
| S005 | 0.573 | S012 | 0.664 | S019 | 0.464 |
| S006 | 0.529 | | | S020 | 0.700 |

Notes:
- Each fold has about 22 validation trials, so each subject's accuracy is uncertain by roughly ±10 points.
- S006 has 85 training trials and S007 has 60 (its run 2 is missing). Every other subject has 110.
- The test trials come from run 3, which had exoskeleton feedback, so the leaderboard score may differ from the CV score.

## Next steps

- Fewer epochs or early stopping, since 200 epochs on about 90 trials probably overfits
- Other trial windows, e.g. 0–4 s or 0–5 s after the cue
- Data augmentation with overlapping crops of each trial
- Tuning the learning rate and dropout
