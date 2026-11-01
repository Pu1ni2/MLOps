"""
Runs the simulated month of production. Each day:
  - the model scores the day's batch, and the data report checks it against the reference data
  - the true labels of an earlier day arrive, and the performance report scores that day's predictions
  - the gate gives the day a status (OK, WARNING or CRITICAL) from everything that failed
WARNING and CRITICAL days are also written to alerts.jsonl. After the last batch, the run goes on
until that batch's true labels have arrived too.

    docker compose run --rm monitor python -m monitoring.simulate
"""
from collections import Counter

from monitoring import checks, gate
from monitoring.data import LAB_DIR, TARGET, load_adult, split_data
from monitoring.model import add_predictions, load_or_train
from monitoring.scenarios import date_of, load_scenario, simulate_day

ALERTS_PATH = LAB_DIR / "alerts.jsonl"


def main():
    scenario, config = load_scenario(), checks.load_config()
    model, reference = load_or_train()
    _, _, pool = split_data(load_adult())
    data_report, reference_data = checks.data_report(reference, config), checks.to_dataset(reference)
    performance_report = checks.performance_report(reference, config)
    labeled_reference = checks.to_labeled_dataset(reference)

    ALERTS_PATH.unlink(missing_ok=True)  # each run writes a fresh alerts file
    statuses = Counter()
    waiting = {}  # day -> (batch, its predictions), until the day's true labels arrive
    for day in range(1, scenario["days"] + scenario["label_delay_days"] + 1):
        failures = []
        if day <= scenario["days"]:
            batch = simulate_day(scenario, pool, day)
            predictions = add_predictions(model, batch.data)
            failures += checks.failed_checks(data_report.run(checks.to_dataset(predictions), reference_data))
            waiting[day] = (batch, predictions)
            # The injected events are only known because this is a simulation; they show what the checks should catch
            note = f"[injected: {', '.join(batch.events)}]" if batch.events else ""
        else:
            note = "[no new batch, only labels]"

        for labeled_day in [d for d, (batch, _) in waiting.items() if batch.labels_known_on == day]:
            labeled_batch, labeled_predictions = waiting.pop(labeled_day)
            labeled = labeled_predictions.assign(**{TARGET: labeled_batch.labels})
            snapshot = performance_report.run(checks.to_labeled_dataset(labeled), labeled_reference)
            failures += [(check, f"predictions of {labeled_batch.date}: {message}")
                         for check, message in checks.failed_checks(snapshot)]

        status = gate.decide(failures, config["severity"])
        statuses[status] += 1
        date = date_of(scenario, day)
        if status != "OK":
            gate.write_alert(ALERTS_PATH, gate.make_alert(day, date, status, failures))
        print(f"day {day:>2}  {date}  {status:<8}  {note}".rstrip())
        for check, message in failures:
            print(f"      {check}: {message}")

    print(f"\n{statuses['OK']} OK, {statuses['WARNING']} WARNING, {statuses['CRITICAL']} CRITICAL days. "
          f"Alerts written to {ALERTS_PATH.name}")


if __name__ == "__main__":
    main()
