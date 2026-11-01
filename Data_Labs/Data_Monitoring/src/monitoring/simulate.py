"""
Runs the simulated month of production. Each day, the model scores the day's batch, the daily
report checks it against the reference data, and the gate gives the day a status: OK, WARNING
or CRITICAL. WARNING and CRITICAL days are also written to alerts.jsonl.

    docker compose run --rm monitor python -m monitoring.simulate
"""
from collections import Counter

from monitoring import checks, gate
from monitoring.data import LAB_DIR, load_adult, split_data
from monitoring.model import add_predictions, load_or_train
from monitoring.scenarios import load_scenario, simulate

ALERTS_PATH = LAB_DIR / "alerts.jsonl"


def main():
    scenario, config = load_scenario(), checks.load_config()
    model, reference = load_or_train()
    _, _, pool = split_data(load_adult())
    report, reference_data = checks.data_report(reference, config), checks.to_dataset(reference)

    ALERTS_PATH.unlink(missing_ok=True)  # each run writes a fresh alerts file
    statuses = Counter()
    for batch in simulate(scenario, pool):
        scored = add_predictions(model, batch.data)
        snapshot = report.run(checks.to_dataset(scored), reference_data)
        failures = checks.failed_checks(snapshot)
        status = gate.decide(failures, config["severity"])
        statuses[status] += 1
        if status != "OK":
            gate.write_alert(ALERTS_PATH, gate.make_alert(batch.day, batch.date, status, failures))

        # The injected events are only known because this is a simulation; they show what the checks should catch
        injected = f"   [injected: {', '.join(batch.events)}]" if batch.events else ""
        print(f"day {batch.day:>2}  {batch.date}  {status:<8}{injected}".rstrip())
        for check, message in failures:
            print(f"      {check}: {message}")

    print(f"\n{statuses['OK']} OK, {statuses['WARNING']} WARNING, {statuses['CRITICAL']} CRITICAL days. "
          f"Alerts written to {ALERTS_PATH.name}")


if __name__ == "__main__":
    main()
