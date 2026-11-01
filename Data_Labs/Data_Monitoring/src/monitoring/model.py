"""
The model being monitored: predicts whether a person earns more than 50K a year.

    docker compose run --rm monitor python -m monitoring.model
trains it, measures its baseline performance on the reference data, and saves both to models/.
"""
import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from monitoring.data import (
    CATEGORICAL, FEATURES, LAB_DIR, NUMERICAL, POSITIVE_LABEL, SEED, TARGET, load_adult, split_data,
)

MODELS_DIR = LAB_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
REFERENCE_PATH = MODELS_DIR / "reference.csv"
METRICS_PATH = MODELS_DIR / "baseline_metrics.json"

# Column names the monitoring reads predictions from
PREDICTION = "prediction"
PROBABILITY = "probability"  # predicted probability of POSITIVE_LABEL


def build_model():
    """A pipeline: fill missing categories, one-hot encode them, then gradient-boosted trees."""
    categories = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        # Dense output, because HistGradientBoosting doesn't accept sparse input
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    features = ColumnTransformer([("num", "passthrough", NUMERICAL), ("cat", categories, CATEGORICAL)])
    return Pipeline([("features", features), ("classifier", HistGradientBoostingClassifier(random_state=SEED))])


def train_model(train):
    """Trains the pipeline on the training data."""
    return build_model().fit(train[FEATURES], train[TARGET])


def add_predictions(model, df):
    """Returns a copy of df with the model's PREDICTION and PROBABILITY columns added."""
    scored = df.copy()
    scored[PREDICTION] = model.predict(df[FEATURES])
    positive_column = list(model.classes_).index(POSITIVE_LABEL)
    scored[PROBABILITY] = model.predict_proba(df[FEATURES])[:, positive_column]
    return scored


def evaluate(scored):
    """Performance on rows that have both a true label and a prediction, rounded to 4 decimals."""
    truth = scored[TARGET] == POSITIVE_LABEL
    predicted = scored[PREDICTION] == POSITIVE_LABEL
    metrics = {
        "accuracy": accuracy_score(truth, predicted),
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
        "roc_auc": roc_auc_score(truth, scored[PROBABILITY]),
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}


def main():
    train, reference, pool = split_data(load_adult())
    model = train_model(train)
    reference = add_predictions(model, reference)
    metrics = evaluate(reference)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    reference.to_csv(REFERENCE_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Split: {len(train)} train, {len(reference)} reference, {len(pool)} production pool")
    print(f"Baseline performance on the reference data: {metrics}")
    return model, reference, metrics


if __name__ == "__main__":
    main()
