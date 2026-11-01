"""
The daily checks: one Evidently report per day compares the day's data and predictions with the
reference data. Each rule in config/monitoring.yaml becomes an Evidently test in that report.
"""
import yaml
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import MissingValueCount, OutRangeValueCount, ValueDrift
from evidently.presets import DataDriftPreset
from evidently.tests import lte

from monitoring.data import CATEGORICAL, FEATURES, LAB_DIR, NUMERICAL
from monitoring.gate import STATUSES
from monitoring.model import PROBABILITY

CONFIG_PATH = LAB_DIR / "config" / "monitoring.yaml"

# What the checks look at: the features and the model's predicted probability (never the true label)
COLUMNS = FEATURES + [PROBABILITY]
DEFINITION = DataDefinition(numerical_columns=NUMERICAL + [PROBABILITY], categorical_columns=CATEGORICAL)

# The order failed checks are listed in
CHECKS = ["data_quality", "feature_drift", "prediction_drift"]


def load_config(path=CONFIG_PATH):
    """Reads the monitoring rules, checking that every check has a known status as its severity."""
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for check in CHECKS:
        status = config["severity"].get(check)
        if status not in STATUSES:
            raise ValueError(f"The severity of {check!r} must be one of {STATUSES}, not {status!r}")
    return config


def to_dataset(df):
    """The columns the checks look at, as an Evidently dataset."""
    return Dataset.from_pandas(df[COLUMNS], data_definition=DEFINITION)


def data_report(reference, config):
    """
    The daily report, with these tests:
      data quality:     missing values in each column, and numeric values outside the reference's range
      feature drift:    each feature's distribution compared with the reference
      prediction drift: the distribution of the model's predicted probabilities
    """
    quality, drift, prediction = config["data_quality"], config["feature_drift"], config["prediction_drift"]
    metrics = []
    for column in FEATURES:
        limit = reference[column].isna().mean() + quality["max_missing_increase"]
        metrics.append(MissingValueCount(column=column, share_tests=[lte(round(limit, 3))]))
    for column in NUMERICAL:
        low, high = reference[column].min().item(), reference[column].max().item()
        metrics.append(OutRangeValueCount(column=column, left=low, right=high,
                                          share_tests=[lte(quality["max_out_of_range"])]))
    metrics.append(DataDriftPreset(
        columns=FEATURES,
        num_method=drift["num_method"], num_threshold=drift["num_threshold"],
        cat_method=drift["cat_method"], cat_threshold=drift["cat_threshold"],
        per_column_threshold=drift["per_column_threshold"],
        # The report's "share of drifted columns" test then fails exactly when the gate's rule does
        drift_share=drift["min_drifted_features"] / len(FEATURES),
    ))
    metrics.append(ValueDrift(column=PROBABILITY, method=prediction["method"], threshold=prediction["threshold"]))
    return Report(metrics, include_tests=True)


def failed_checks(snapshot):
    """
    The failed tests of a day's report, as (check, message) pairs, for example
    ("data_quality", "occupation 45.0% missing (limit 16.0%)"). Drifted features are listed
    together in one feature_drift message, which only appears when enough features drifted.
    """
    result = snapshot.dict()
    values = {(metric["config"]["type"], metric["config"].get("column")): metric["value"]
              for metric in result["metrics"]}
    failures, drifted, enough_drifted = [], [], False
    for test in result["tests"]:
        if test["status"] != "FAIL":
            continue
        params = test["metric_config"]["params"]
        metric, column = params["type"].rsplit(":", 1)[-1], params.get("column")
        value = values[(params["type"], column)]
        if metric == "MissingValueCount":
            limit = test["test_config"]["threshold"]
            failures.append(("data_quality", f"{column} {value['share']:.1%} missing (limit {limit:.1%})"))
        elif metric == "OutRangeValueCount":
            limit = test["test_config"]["threshold"]
            failures.append(("data_quality", f"{column} {value['share']:.1%} outside "
                                             f"{params['left']} to {params['right']} (limit {limit:.1%})"))
        elif metric == "ValueDrift" and column == PROBABILITY:
            failures.append(("prediction_drift", f"score {value:.2f} (threshold {params['threshold']})"))
        elif metric == "ValueDrift":
            drifted.append(f"{column} {value:.2f}")
        elif metric == "DriftedColumnsCount":
            enough_drifted = True
    if enough_drifted:
        failures.append(("feature_drift", f"{len(drifted)} of {len(FEATURES)} features drifted: {', '.join(drifted)}"))
    return sorted(failures, key=lambda failure: CHECKS.index(failure[0]))
