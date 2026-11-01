"""The gate: a day's status is the severity of its most serious failed check, and alerts are JSON lines."""
import json

import pytest

from monitoring.gate import decide, make_alert, write_alert


@pytest.mark.parametrize("failed, status", [
    ([], "OK"),
    (["feature_drift"], "WARNING"),
    (["prediction_drift"], "WARNING"),
    (["feature_drift", "prediction_drift"], "WARNING"),
    (["data_quality"], "CRITICAL"),
    (["performance"], "CRITICAL"),
    (["feature_drift", "data_quality"], "CRITICAL"),
])
def test_status_is_the_most_serious_failed_check(config, failed, status):
    assert decide([(check, "a message") for check in failed], config["severity"]) == status


def test_alerts_are_added_as_json_lines(tmp_path):
    path = tmp_path / "alerts.jsonl"
    write_alert(path, make_alert(15, "2026-10-15", "CRITICAL", [("data_quality", "occupation 45.0% missing")]))
    write_alert(path, make_alert(16, "2026-10-16", "WARNING", [("feature_drift", "1 of 14 features drifted")]))
    alerts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(alert["day"], alert["status"]) for alert in alerts] == [(15, "CRITICAL"), (16, "WARNING")]
    assert alerts[0]["failed_checks"] == [{"check": "data_quality", "message": "occupation 45.0% missing"}]
