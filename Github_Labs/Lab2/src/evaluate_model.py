"""Evaluate every model of a version on the held-back test set and save the metrics."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import load_data, split_data

# Readable names for the metrics table
MODEL_LABELS = {"rf": "Random forest", "rf_calibrated": "Calibrated random forest"}


def evaluate(model, X_test, y_test):
    """
    Scores a model on emails it never saw during training.
    Args:
        model: Fitted classifier with predict and predict_proba.
        X_test (DataFrame): Test features.
        y_test (Series): Test labels.
    Returns:
        dict: accuracy, precision, recall, f1 and roc_auc (higher is better), and brier
            (average squared error of the spam probabilities, lower is better). All are between 0 and 1.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # predicted probability of spam
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "brier": brier_score_loss(y_test, y_proba),
    }
    return {name: round(float(value), 4) for name, value in scores.items()}


def find_models(model_dir, timestamp):
    """
    Finds all model files of one version, e.g. model_<timestamp>_rf.joblib.
    Returns:
        dict: Model name (the part after the timestamp, e.g. "rf") mapped to its file path.
    """
    prefix = f"model_{timestamp}_"
    paths = sorted(Path(model_dir).glob(f"{prefix}*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No models for version {timestamp} in {model_dir}")
    return {path.stem[len(prefix):]: path for path in paths}


def passes_quality_gate(metrics, min_f1):
    """
    Checks that every model of a version reached the minimum F1 score on the test set.
    Args:
        metrics (dict): Metrics with one entry per model under "models".
        min_f1 (float): Lowest acceptable F1 score.
    Returns:
        bool: True if no model is below min_f1.
    """
    return all(scores["f1"] >= min_f1 for scores in metrics["models"].values())


def metrics_to_markdown(metrics):
    """Formats the metrics as a Markdown table for the job summary and the release notes."""
    names = list(metrics["models"])
    lines = [
        f"### Spam classifier, version {metrics['timestamp']}",
        "",
        f"Trained on {metrics['train_size']} emails and tested on {metrics['test_size']} emails "
        f"the model never saw (scikit-learn {metrics['scikit_learn_version']}).",
        "",
        "| Metric | " + " | ".join(MODEL_LABELS.get(name, name) for name in names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for metric in metrics["models"][names[0]]:
        lines.append(f"| {metric} | " + " | ".join(f"{metrics['models'][name][metric]:.4f}" for name in names) + " |")
    gate = metrics.get("quality_gate")
    if gate:
        result = "passed" if gate["passed"] else "FAILED"
        lines += ["", f"Quality gate: **{result}** (every model needs F1 >= {gate['min_f1']:.2f})"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the models of one version on the test set.")
    parser.add_argument("--timestamp", required=True, help="Version to evaluate, e.g. 20261002130000")
    parser.add_argument("--model-dir", default="models", help="Folder containing the model files")
    parser.add_argument("--output-dir", default="metrics", help="Folder to save the metrics in")
    parser.add_argument("--min-f1", type=float, help="Quality gate: exit with an error if any model's F1 is below this")
    args = parser.parse_args(argv)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    metrics = {
        "timestamp": args.timestamp,
        "dataset": "spambase",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "scikit_learn_version": sklearn.__version__,
        "models": {
            name: evaluate(joblib.load(path), X_test, y_test)
            for name, path in find_models(args.model_dir, args.timestamp).items()
        },
    }
    if args.min_f1 is not None:
        metrics["quality_gate"] = {"min_f1": args.min_f1, "passed": passes_quality_gate(metrics, args.min_f1)}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.timestamp}_metrics.json").write_text(json.dumps(metrics, indent=4) + "\n", encoding="utf-8")
    summary = metrics_to_markdown(metrics)
    (output_dir / f"{args.timestamp}_summary.md").write_text(summary, encoding="utf-8")
    print(summary)

    # The metrics files are written first, so a failed version can still be inspected
    if args.min_f1 is not None and not metrics["quality_gate"]["passed"]:
        print(f"Quality gate failed: a model has F1 below {args.min_f1}, so this version will not be released.")
        sys.exit(1)
    return metrics


if __name__ == "__main__":
    main()
