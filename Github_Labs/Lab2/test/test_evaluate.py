import json

import joblib
import pytest

from src import evaluate_model


def test_evaluate_returns_metrics_between_0_and_1(split, small_model):
    X_train, X_test, y_train, y_test = split
    metrics = evaluate_model.evaluate(small_model, X_test, y_test)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(metrics)
    assert all(0 <= value <= 1 for value in metrics.values())


def test_find_models_errors_when_version_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_model.find_models(tmp_path, "missing")


def test_main_writes_metrics_and_summary(tmp_path, small_model):
    joblib.dump(small_model, tmp_path / "model_test_rf.joblib")
    metrics = evaluate_model.main(["--timestamp", "test", "--model-dir", str(tmp_path), "--output-dir", str(tmp_path)])

    saved = json.loads((tmp_path / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
    assert saved["test_size"] == 921
    assert list(saved["models"]) == ["rf"]

    summary = (tmp_path / "test_summary.md").read_text(encoding="utf-8")
    assert "| f1 |" in summary
