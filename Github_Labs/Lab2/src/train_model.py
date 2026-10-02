"""Train a random forest spam classifier and save it as a versioned model file."""
import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.data import RANDOM_STATE, load_data, split_data


def train_model(X_train, y_train, n_estimators=200, random_state=RANDOM_STATE):
    """
    Trains a random forest classifier.
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed that makes training reproducible.
    Returns:
        RandomForestClassifier: The fitted model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def model_path(model_dir, timestamp, name="rf"):
    """Returns the versioned file path of a model, e.g. models/model_20261002130000_rf.joblib."""
    return Path(model_dir) / f"model_{timestamp}_{name}.joblib"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the spam classifier.")
    parser.add_argument("--timestamp", required=True, help="Version label for the model file, e.g. 20261002130000")
    parser.add_argument("--output-dir", default="models", help="Folder to save the model in")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees in the forest")
    args = parser.parse_args(argv)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, n_estimators=args.n_estimators)

    path = model_path(args.output_dir, args.timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    # compress=3 shrinks the file several times over; loading works the same way
    joblib.dump(model, path, compress=3)
    print(f"Trained on {len(X_train)} emails, saved model to {path}")
    return path


if __name__ == "__main__":
    main()
