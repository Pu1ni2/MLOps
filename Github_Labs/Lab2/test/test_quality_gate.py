import json

import joblib
import pytest

from src import evaluate_model


def metrics_with_f1(*f1_scores):
    """Builds a minimal metrics dict with one model per F1 score."""
    return {"models": {f"model{i}": {"f1": f1} for i, f1 in enumerate(f1_scores)}}


def test_gate_passes_when_every_model_reaches_threshold():
    assert evaluate_model.passes_quality_gate(metrics_with_f1(0.93, 0.91), min_f1=0.90)


def test_gate_passes_at_exactly_the_threshold():
    assert evaluate_model.passes_quality_gate(metrics_with_f1(0.90), min_f1=0.90)


def test_gate_fails_when_any_model_is_below():
    assert not evaluate_model.passes_quality_gate(metrics_with_f1(0.95, 0.85), min_f1=0.90)


def test_main_stops_with_error_below_threshold(tmp_path, small_model):
    joblib.dump(small_model, tmp_path / "model_test_rf.joblib")
    args = ["--timestamp", "test", "--model-dir", str(tmp_path), "--output-dir", str(tmp_path), "--min-f1", "0.999"]

    with pytest.raises(SystemExit) as error:
        evaluate_model.main(args)
    assert error.value.code == 1

    # The metrics are still saved, so the failed version can be inspected
    saved = json.loads((tmp_path / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved["quality_gate"] == {"min_f1": 0.999, "passed": False}


def test_main_records_passed_gate(tmp_path, small_model):
    joblib.dump(small_model, tmp_path / "model_test_rf.joblib")
    args = ["--timestamp", "test", "--model-dir", str(tmp_path), "--output-dir", str(tmp_path), "--min-f1", "0.5"]

    metrics = evaluate_model.main(args)
    assert metrics["quality_gate"]["passed"]
    assert "Quality gate: **passed**" in (tmp_path / "test_summary.md").read_text(encoding="utf-8")
