"""The calibration: it measures what the reports compute, the config uses its results, and they hold up."""
import pandas as pd
import pytest

from monitoring import checks
from monitoring.calibrate import RESULTS_DIR, DriftTests, is_drift, simulated_days
from monitoring.data import CATEGORICAL, FEATURES
from monitoring.model import PROBABILITY, add_predictions
from monitoring.scenarios import simulate_day


def configured_tests(config):
    """The drift test and threshold the config gives every monitored column."""
    drift, prediction = config["feature_drift"], config["prediction_drift"]
    setup = {column: (drift["cat_method"] if column in CATEGORICAL else drift["num_method"], threshold)
             for column, threshold in drift["per_column_threshold"].items()}
    setup[PROBABILITY] = (prediction["method"], prediction["threshold"])
    return setup


def test_calibration_scores_are_the_reports_scores(scenario, pool, config, model_and_reference):
    model, reference = model_and_reference
    predictions = add_predictions(model, simulate_day(scenario, pool, 15).data)  # a day with missing values
    snapshot = checks.data_report(reference, config).run(checks.to_dataset(predictions), checks.to_dataset(reference))
    reported = {metric["config"]["column"]: metric["value"] for metric in snapshot.dict()["metrics"]
                if metric["config"]["type"].endswith(":ValueDrift")}
    tests = DriftTests(reference)
    for column, (test, _) in configured_tests(config).items():
        assert tests.score(predictions, column, test) == pytest.approx(reported[column]), column


def test_config_uses_the_calibrated_thresholds(config):
    table = pd.read_csv(RESULTS_DIR / "thresholds.csv")
    calibrated = {(row.column, row.test): row.calibrated_threshold for row in table.itertuples()}
    setup = configured_tests(config)
    assert set(setup) == set(FEATURES + [PROBABILITY])
    for column, (test, threshold) in setup.items():
        assert threshold == pytest.approx(calibrated[(column, test)], rel=1e-3), column


def test_fresh_normal_days_rarely_warn(scenario, pool, config, model_and_reference):
    """The study measured 3.4% of drift-free days with a drift WARNING; check on 200 days it never saw."""
    model, reference = model_and_reference
    tests, setup = DriftTests(reference), configured_tests(config)
    days = [add_predictions(model, day) for day in simulated_days(scenario, pool, seed=7, count=200)]
    warned = [any(is_drift(test, tests.score(day, column, test), threshold) for column, (test, threshold) in setup.items())
              for day in days]
    assert sum(warned) / len(days) < 0.10
