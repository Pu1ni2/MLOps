"""The whole run on a short scenario: reports reach the workspace, and the broken day raises a CRITICAL alert."""
import json

from evidently.ui.workspace import Workspace

from monitoring import dashboard, simulate


def test_short_run_from_start_to_end(scenario, tmp_path, capsys):
    broken_feed = next(event for event in scenario["events"] if event["type"] == "data_quality")
    short = {**scenario, "days": 3, "label_delay_days": 1, "events": [{**broken_feed, "days": [2, 2]}]}
    workspace, alerts = Workspace(str(tmp_path / "workspace")), tmp_path / "alerts.jsonl"
    for _ in range(2):  # a rerun replaces the first run's reports instead of adding to them
        simulate.main(short, workspace, alerts)

    assert "day  4  2026-10-04" in capsys.readouterr().out  # one extra day, for the last labels
    (project,) = workspace.search_project(dashboard.PROJECT_NAME)
    assert len(workspace.list_runs(project.id)) == 3 + 3  # a data report and a performance report per day
    statuses = {alert["day"]: alert["status"] for alert in map(json.loads, alerts.read_text().splitlines())}
    assert statuses[2] == "CRITICAL"
