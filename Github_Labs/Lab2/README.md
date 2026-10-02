# GitHub Lab 2: Model Training and Versioning with GitHub Actions

[![Github Lab2 - Model Training](https://github.com/Pu1ni2/MLOps/actions/workflows/github_lab2_model_training.yml/badge.svg)](https://github.com/Pu1ni2/MLOps/actions/workflows/github_lab2_model_training.yml)

A spam-detection model that GitHub Actions trains, calibrates, evaluates and versions automatically. Every change to the code starts the pipeline. If the new model is good enough, it is published as a GitHub Release together with its metrics. If not, nothing is released.

## Pipeline

```
push to main, or the "Run workflow" button
        │
        ▼
test job ── pytest ── a test fails? ──► stop, no training
        │
        ▼
train job ── train random forest ──► train calibrated random forest ──► evaluate both on the test set
        │
        ▼
quality gate ── is F1 ≥ 0.90 for every model? ── no ──► stop, nothing is released
        │ yes
        ▼
GitHub Release "lab2-model-<timestamp>" with both models and their metrics
```

## Dataset

[Spambase](https://doi.org/10.24432/C53G6X): 4,601 emails, 39.4% of them spam. Each email is described by 57 numbers:

- 48 word frequencies, such as how often "free", "money" or "remove" appear
- 6 character frequencies, such as "!" and "$"
- 3 measures of runs of capital letters

20% of the emails (921) are held back as a test set that the models never see during training. The split is stratified, so both sets have the same share of spam. It is also fixed, so every model version is evaluated on the same emails.

## Models

- **Random forest:** 200 decision trees that vote on spam or not spam.
- **Calibrated random forest:** the same kind of model wrapped in `CalibratedClassifierCV` (isotonic, 5-fold), so its predicted spam probabilities match how often emails really are spam. The goal is that when it says 0.8, about 80% of such emails really are spam.

## Results

From one training run. Every Release lists its exact numbers.

| Metric | Random forest | Calibrated random forest |
|--------|---------------|--------------------------|
| Accuracy | 0.95 | 0.94 |
| Precision | 0.95 | 0.94 |
| Recall | 0.91 | 0.91 |
| F1 | 0.93 | 0.93 |
| ROC AUC | 0.98 | 0.98 |
| Brier score (lower is better) | 0.043 | 0.041 |

Calibration lowers the Brier score by about 5%, so the spam probabilities become more trustworthy while accuracy stays about the same.

## Model versions

Each version is a [Release](https://github.com/Pu1ni2/MLOps/releases) tagged `lab2-model-<UTC timestamp>` that contains:

- `model_<timestamp>_rf.joblib`: the random forest
- `model_<timestamp>_rf_calibrated.joblib`: the calibrated random forest
- `<timestamp>_metrics.json`: test-set metrics for both models, plus the quality gate result

To use a downloaded model:

```python
import joblib

model = joblib.load("model_<timestamp>_rf_calibrated.joblib")
spam_probability = model.predict_proba(emails)[:, 1]  # emails: DataFrame with the 57 feature columns
```

## Tests

18 pytest tests check:

- the data: shape, labels, no overlap between the training and test sets, a reproducible split
- training and evaluation: the model beats always guessing "not spam", and metrics and summary files are written
- calibration: probabilities are valid, and the Brier score is lower than without calibration
- the quality gate: it passes at the threshold, and it fails with exit code 1 below it while still saving the metrics

The tests use 10-tree forests, so they finish in about 2 seconds.

## Project structure

```
MLOps/                                        (repo root)
├── .github/workflows/
│   └── github_lab2_model_training.yml        # test → train → calibrate → evaluate → gate → release
└── Github_Labs/Lab2/
    ├── data/spambase.csv
    ├── src/
    │   ├── data.py                           # load data, stratified train/test split
    │   ├── train_model.py                    # random forest
    │   ├── calibrate_model.py                # calibrated random forest
    │   └── evaluate_model.py                 # test-set metrics, summary table, quality gate
    ├── test/                                 # pytest tests
    └── requirements.txt
```

## Run locally

From `Github_Labs/Lab2`:

```
python -m venv lab_02
lab_02\Scripts\activate
pip install -r requirements.txt

pytest
python -m src.train_model --timestamp local
python -m src.calibrate_model --timestamp local
python -m src.evaluate_model --timestamp local --min-f1 0.90
```

Models are saved in `models/` and metrics in `metrics/`. Git ignores both folders.

## My changes

Compared with the [original lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2):

**Data and model**
- **Real dataset:** the UCI Spambase emails replace the original's synthetic data, which had a random size (0 to 2,000 rows) on every run, so results could never be reproduced.
- **Proper evaluation:** the original measured accuracy on the training data and "evaluated" on freshly generated random data. Models are now scored on a fixed, stratified test set they never see during training. Precision, recall, ROC AUC and the Brier score were added to F1.
- **Real calibration:** the original README described model calibration, but the code never did it. A calibrated random forest (`CalibratedClassifierCV`, isotonic, 5-fold) is now trained and compared. It lowers the Brier score from 0.043 to 0.041.
- Models are saved compressed, which brings the random forest from 9.2 MB down to 2.0 MB.

**Pipeline**
- **Quality gate:** a version is only released if every model reaches F1 ≥ 0.90.
- **Tests:** 18 pytest tests (the original `test/` folder was empty). The workflow trains only if they pass.
- **Versioning with GitHub Releases:** the original workflow committed each model into the repo. Its commits were signed with the instructor's name, and its push step lacked the write permission new repos require. Each version is now a Release with both models and their metrics, so the workflow never commits.
- Training runs on code changes and from a manual button. The broken daily schedule was removed.
- MLflow was removed: the original logged to a folder that was deleted when each run finished.

## Dataset credit

Spambase by M. Hopkins, E. Reeber, G. Forman and J. Suermondt (1999), [UCI Machine Learning Repository](https://doi.org/10.24432/C53G6X), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data was converted to CSV with a header row, and the 6 character-frequency columns were renamed for readability (for example `char_freq_!` became `char_freq_exclamation`).
