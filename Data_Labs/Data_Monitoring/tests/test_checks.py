"""The checks: Evidently reports whose failed tests become the gate's messages, on known scenario days."""
import pytest
import yaml

from monitoring import checks
from monitoring.data import TARGET
from monitoring.gate import decide
from monitoring.model import add_predictions
from monitoring.scenarios import simulate_day


@pytest.fixture(scope="module")
def failed_checks_on(scenario, pool, config, model_and_reference):
    """The failed checks of a scenario day's data report, or of its performance report (labels=True)."""
    model, reference = model_and_reference
    data_report, reference_data = checks.data_report(reference, config), checks.to_dataset(reference)
    performance_report, labeled_reference = (checks.performance_report(reference, config),
                                             checks.to_labeled_dataset(reference))

    def failed(day, labels=False):
        batch = simulate_day(scenario, pool, day)
        predictions = add_predictions(model, batch.data)
        if labels:
            labeled = predictions.assign(**{TARGET: batch.labels})
            return checks.failed_checks(performance_report.run(checks.to_labeled_dataset(labeled), labeled_reference))
        return checks.failed_checks(data_report.run(checks.to_dataset(predictions), reference_data))
    return failed


def test_normal_day_passes_every_check(failed_checks_on):
    assert failed_checks_on(1) == []
    assert failed_checks_on(1, labels=True) == []


def test_broken_feed_fails_data_quality(failed_checks_on, config):
    failures = failed_checks_on(15)
    quality = [message for check, message in failures if check == "data_quality"]
    assert any(message.startswith("occupation") and "missing" in message for message in quality)
    assert any(message.startswith("hours-per-week") and "outside" in message for message in quality)
    assert decide(failures, config["severity"]) == "CRITICAL"


def test_more_self_employed_drifts_workclass(failed_checks_on, config):
    failures = failed_checks_on(21)
    assert [check for check, _ in failures] == ["feature_drift"]
    assert "workclass" in failures[0][1]
    assert decide(failures, config["severity"]) == "WARNING"


def test_concept_drift_shows_only_in_performance(failed_checks_on, config):
    assert failed_checks_on(23) == []  # the people look normal...
    failures = failed_checks_on(23, labels=True)  # ...but their true labels reveal it
    assert [check for check, _ in failures] == ["performance"]
    assert failures[0][1].startswith("f1")
    assert decide(failures, config["severity"]) == "CRITICAL"


def write_config(tmp_path, config):
    path = tmp_path / "monitoring.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_misspelled_severity_is_rejected(config, tmp_path):
    bad = {**config, "severity": {**config["severity"], "data_quality": "CRITCAL"}}
    with pytest.raises(ValueError, match="severity of 'data_quality'"):
        checks.load_config(write_config(tmp_path, bad))


def test_unknown_performance_metric_is_rejected(config, tmp_path):
    bad = {**config, "performance": {"max_drop": {"F1": 0.1}}}
    with pytest.raises(ValueError, match="Unknown performance metric 'F1'"):
        checks.load_config(write_config(tmp_path, bad))
