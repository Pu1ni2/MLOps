"""
The checks, as Evidently reports whose tests come from the rules in config/monitoring.yaml:
  - every day, a data report compares the day's data and predictions with the reference data
  - once a day's true labels arrive, a performance report scores that day's predictions
"""
import yaml
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import (
    Accuracy, F1Score, MissingValueCount, OutRangeValueCount, Precision, Recall, RocAuc, ValueDrift,
)
from evidently.presets import DataDriftPreset
from evidently.tests import gte, lte

from monitoring.data import CATEGORICAL, FEATURES, LAB_DIR, NUMERICAL, POSITIVE_LABEL, TARGET
from monitoring.gate import STATUSES
from monitoring.model import PREDICTION, PROBABILITY, evaluate

CONFIG_PATH = LAB_DIR / "config" / "monitoring.yaml"

# What the data report looks at: the features and the model's predicted probability (never the true label)
COLUMNS = FEATURES + [PROBABILITY]
DEFINITION = DataDefinition(numerical_columns=NUMERICAL + [PROBABILITY], categorical_columns=CATEGORICAL)

# What the performance report looks at: the true label next to the model's prediction
LABELED_COLUMNS = [TARGET, PREDICTION, PROBABILITY]
LABELED_DEFINITION = DataDefinition(
    classification=[BinaryClassification(target=TARGET, prediction_labels=PREDICTION,
                                         prediction_probas=PROBABILITY, pos_label=POSITIVE_LABEL)],
    categorical_columns=[TARGET, PREDICTION],
    numerical_columns=[PROBABILITY],
)
# The Evidently metric for each performance measure, under the names model.evaluate gives them
PERFORMANCE_METRICS = {"accuracy": Accuracy, "precision": Precision, "recall": Recall, "f1": F1Score, "roc_auc": RocAuc}

# The order failed checks are listed in
CHECKS = ["data_quality", "feature_drift", "prediction_drift", "performance"]


def load_config(path=CONFIG_PATH):
    """Reads the monitoring rules, checking that every check has a known severity and every tested metric exists."""
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for check in CHECKS:
        status = config["severity"].get(check)
        if status not in STATUSES:
            raise ValueError(f"The severity of {check!r} must be one of {STATUSES}, not {status!r}")
    for name in config["performance"]["max_drop"]:
        if name not in PERFORMANCE_METRICS:
            raise ValueError(f"Unknown performance metric {name!r}: use one of {list(PERFORMANCE_METRICS)}")
    return config


def to_dataset(df):
    """The columns the data report looks at, as an Evidently dataset."""
    return Dataset.from_pandas(df[COLUMNS], data_definition=DEFINITION)


def to_labeled_dataset(df):
    """The true labels and predictions the performance report looks at, as an Evidently dataset."""
    return Dataset.from_pandas(df[LABELED_COLUMNS], data_definition=LABELED_DEFINITION)


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
        # A test's own default threshold applies to any column without a threshold of its own
        num_method=drift["num_method"], num_threshold=drift.get("num_threshold"),
        cat_method=drift["cat_method"], cat_threshold=drift.get("cat_threshold"),
        per_column_threshold=drift["per_column_threshold"],
        # The report's "share of drifted columns" test then fails exactly when the gate's rule does
        drift_share=drift["min_drifted_features"] / len(FEATURES),
    ))
    metrics.append(ValueDrift(column=PROBABILITY, method=prediction["method"], threshold=prediction["threshold"]))
    return Report(metrics, include_tests=True)


def performance_report(reference, config):
    """
    The report on a day's predictions once its true labels arrive. It measures every metric in
    PERFORMANCE_METRICS, and tests the ones listed under performance.max_drop against their
    baseline on the reference data. The others get no test (tests=[]), so Evidently doesn't add its own.
    """
    baseline, max_drop = evaluate(reference), config["performance"]["max_drop"]
    metrics = []
    for name, metric in PERFORMANCE_METRICS.items():
        tests = [gte(round(baseline[name] - max_drop[name], 3))] if name in max_drop else []
        metrics.append(metric(tests=tests))
    return Report(metrics, include_tests=True)


def failed_checks(snapshot):
    """
    The failed tests of a data or performance report, as (check, message) pairs, for example
    ("data_quality", "occupation 45.0% missing (limit 16.0%)"). Drifted features are listed
    together in one feature_drift message, which only appears when enough features drifted.
    """
    performance_names = {metric.__name__: name for name, metric in PERFORMANCE_METRICS.items()}
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
            failures.append(("prediction_drift", f"score {value:.3g} (threshold {params['threshold']:.3g})"))
        elif metric == "ValueDrift":
            drifted.append(f"{column} {value:.3g}")
        elif metric == "DriftedColumnsCount":
            enough_drifted = True
        elif metric in performance_names:
            limit = test["test_config"]["threshold"]
            failures.append(("performance", f"{performance_names[metric]} {value:.3f} (limit {limit:.3f})"))
    if enough_drifted:
        failures.append(("feature_drift", f"{len(drifted)} of {len(FEATURES)} features drifted: {', '.join(drifted)}"))
    return sorted(failures, key=lambda failure: CHECKS.index(failure[0]))
