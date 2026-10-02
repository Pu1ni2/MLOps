"""Train a calibrated version of the spam classifier, so its spam probabilities can be trusted."""
import argparse

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.data import RANDOM_STATE, load_data, split_data
from src.train_model import model_path


def calibrate_model(X_train, y_train, n_estimators=200, method="isotonic", random_state=RANDOM_STATE):
    """
    Trains a random forest whose predicted probabilities are calibrated.

    A model is calibrated when its probabilities match reality: of all emails it gives a 0.8
    spam probability, about 80% really are spam. Random forests are often off here, because
    averaging many trees pulls their probabilities away from 0 and 1. CalibratedClassifierCV
    splits the training data into 5 parts, trains a forest on 4 of them and uses the 5th to
    learn how to correct its probabilities, then repeats this for each part.
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        n_estimators (int): Number of trees in each forest.
        method (str): "isotonic" (flexible correction) or "sigmoid" (Platt scaling).
        random_state (int): Seed that makes training reproducible.
    Returns:
        CalibratedClassifierCV: The fitted, calibrated model.
    """
    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model = CalibratedClassifierCV(forest, method=method, cv=5)
    model.fit(X_train, y_train)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the calibrated spam classifier.")
    parser.add_argument("--timestamp", required=True, help="Version label for the model file, e.g. 20261002130000")
    parser.add_argument("--output-dir", default="models", help="Folder to save the model in")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees in each forest")
    args = parser.parse_args(argv)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = calibrate_model(X_train, y_train, n_estimators=args.n_estimators)

    path = model_path(args.output_dir, args.timestamp, name="rf_calibrated")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    print(f"Trained calibrated model on {len(X_train)} emails, saved to {path}")
    return path


if __name__ == "__main__":
    main()
