import joblib
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from src import calibrate_model, evaluate_model


@pytest.fixture(scope="module")
def small_calibrated_model(split):
    """A calibrated model built from forests of only 10 trees, so tests run in seconds."""
    X_train, X_test, y_train, y_test = split
    return calibrate_model.calibrate_model(X_train, y_train, n_estimators=10)


def test_calibrated_probabilities_are_valid(split, small_calibrated_model):
    X_train, X_test, y_train, y_test = split
    proba = small_calibrated_model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    # Probabilities of "not spam" and "spam" must add up to 1 for every email
    assert np.allclose(proba.sum(axis=1), 1)


def test_calibration_lowers_brier_score(split, small_model, small_calibrated_model):
    X_train, X_test, y_train, y_test = split
    plain = brier_score_loss(y_test, small_model.predict_proba(X_test)[:, 1])
    calibrated = brier_score_loss(y_test, small_calibrated_model.predict_proba(X_test)[:, 1])
    assert calibrated < plain


def test_main_saves_calibrated_model(tmp_path):
    path = calibrate_model.main(["--timestamp", "test", "--output-dir", str(tmp_path), "--n-estimators", "10"])
    assert path == tmp_path / "model_test_rf_calibrated.joblib"
    assert hasattr(joblib.load(path), "predict_proba")


def test_evaluate_reports_both_models(tmp_path, small_model, small_calibrated_model):
    joblib.dump(small_model, tmp_path / "model_test_rf.joblib")
    joblib.dump(small_calibrated_model, tmp_path / "model_test_rf_calibrated.joblib")
    metrics = evaluate_model.main(["--timestamp", "test", "--model-dir", str(tmp_path), "--output-dir", str(tmp_path)])

    assert list(metrics["models"]) == ["rf", "rf_calibrated"]
    assert all("brier" in scores for scores in metrics["models"].values())
    summary = (tmp_path / "test_summary.md").read_text(encoding="utf-8")
    assert "Calibrated random forest" in summary
