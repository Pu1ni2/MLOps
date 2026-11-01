"""
Runs the simulated month of production. Each day:
  - the model scores the day's batch, and the data report checks it against the reference data
  - the true labels of an earlier day arrive, and the performance report scores that day's predictions
  - the gate gives the day a status (OK, WARNING or CRITICAL) from everything that failed
Every report goes to the Evidently dashboard (see dashboard.py), and WARNING and CRITICAL days
are also written to alerts.jsonl. After the last batch, the run goes on until that batch's
true labels have arrived too.

    docker compose run --rm monitor python -m monitoring.simulate
"""
import datetime
from collections import Counter

from monitoring import checks, dashboard, gate
from monitoring.data import LAB_DIR, TARGET, load_adult, split_data
from monitoring.model import add_predictions, load_or_train
from monitoring.scenarios import date_of, load_scenario, simulate_day
from monitoring.workspace import open_workspace, project_url

ALERTS_PATH = LAB_DIR / "alerts.jsonl"


def at_midnight(date):
    """A report's timestamp: the start of its simulated day."""
    return datetime.datetime.combine(date, datetime.time())


def publish(workspace, project, snapshot, name, severity):
    """
    Sends a report to the dashboard, named after its own result (for example
    "2026-10-15 data checks: CRITICAL"), and returns its failed checks.
    """
    failures = checks.failed_checks(snapshot)
    snapshot.set_name(f"{name}: {gate.decide(failures, severity)}")
    workspace.add_run(project.id, snapshot, include_data=False)
    return failures


def main(scenario=None, workspace=None, alerts_path=ALERTS_PATH):
    """
    Runs the simulation. By default it uses config/scenario.yaml, sends the reports to the
    dashboard service (see workspace.open_workspace), and writes the alerts to alerts.jsonl.
    """
    scenario = load_scenario() if scenario is None else scenario
    config = checks.load_config()
    model, reference = load_or_train()
    _, _, pool = split_data(load_adult())
    data_report, reference_data = checks.data_report(reference, config), checks.to_dataset(reference)
    performance_report = checks.performance_report(reference, config)
    labeled_reference = checks.to_labeled_dataset(reference)
    severity = config["severity"]

    workspace = open_workspace() if workspace is None else workspace
    project = dashboard.recreate_project(workspace)
    dashboard.save_dashboard(workspace, project, "The simulated month is running...")

    alerts_path.unlink(missing_ok=True)  # each run writes a fresh alerts file
    statuses = Counter()
    waiting = {}  # day -> (batch, its predictions), until the day's true labels arrive
    for day in range(1, scenario["days"] + scenario["label_delay_days"] + 1):
        failures = []
        if day <= scenario["days"]:
            batch = simulate_day(scenario, pool, day)
            predictions = add_predictions(model, batch.data)
            snapshot = data_report.run(checks.to_dataset(predictions), reference_data,
                                       timestamp=at_midnight(batch.date), tags=[dashboard.DATA_TAG])
            failures += publish(workspace, project, snapshot, f"{batch.date} data checks", severity)
            waiting[day] = (batch, predictions)
            # The injected events are only known because this is a simulation; they show what the checks should catch
            note = f"[injected: {', '.join(batch.events)}]" if batch.events else ""
        else:
            note = "[no new batch, only labels]"

        for labeled_day in [d for d, (batch, _) in waiting.items() if batch.labels_known_on == day]:
            labeled_batch, labeled_predictions = waiting.pop(labeled_day)
            labeled = labeled_predictions.assign(**{TARGET: labeled_batch.labels})
            # Dated by the day the predictions were made, so its charts line up with that day's data report
            snapshot = performance_report.run(checks.to_labeled_dataset(labeled), labeled_reference,
                                              timestamp=at_midnight(labeled_batch.date),
                                              tags=[dashboard.PERFORMANCE_TAG])
            performance_failures = publish(workspace, project, snapshot, f"{labeled_batch.date} performance", severity)
            failures += [(check, f"predictions of {labeled_batch.date}: {message}")
                         for check, message in performance_failures]

        status = gate.decide(failures, severity)
        statuses[status] += 1
        date = date_of(scenario, day)
        if status != "OK":
            gate.write_alert(alerts_path, gate.make_alert(day, date, status, failures))
        print(f"day {day:>2}  {date}  {status:<8}  {note}".rstrip())
        for check, message in failures:
            print(f"      {check}: {message}")

    counts = f"{statuses['OK']} OK, {statuses['WARNING']} WARNING, {statuses['CRITICAL']} CRITICAL days"
    dashboard.save_dashboard(workspace, project, (
        f"{counts}. Every day, {scenario['batch_size']} people are scored and their data is checked; "
        f"each day's predictions are scored once their true labels arrive, {scenario['label_delay_days']} days later."
    ))
    print(f"\n{counts}. Alerts written to {alerts_path.name}")
    url = project_url(workspace, project)
    print(f"Dashboard: {url}" if url else "Reports saved in the local Evidently workspace")


if __name__ == "__main__":
    main()
