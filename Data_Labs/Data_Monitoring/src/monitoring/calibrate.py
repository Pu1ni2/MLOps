"""
Compares drift tests and calibrates the thresholds of the daily drift checks. It runs the same
Evidently drift tests the reports run, on thousands of simulated days:

  1. Normal days, without drift: each test's score for every monitored column. Each column's
     threshold is set so that only 1 normal day in 500 crosses it.
  2. Days with drift of growing strength, made with the scenario's own drift events: how often
     each test catches the drift with its calibrated threshold. The test that catches the most
     is the one to use.
  3. Fresh normal days: how often the daily checks would raise a false drift WARNING, with
     Evidently's default thresholds and with the calibrated ones.
  4. The normal days scored against their true labels: how far each performance metric falls
     below its baseline when nothing has drifted, and how often the performance rule fires.

    docker compose run --rm monitor python -m monitoring.calibrate
writes the results to results/ (tables, and the chart calibration.png) and prints the settings
for config/monitoring.yaml. It takes about 5 minutes.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently.legacy.calculations.stattests import get_stattest
from evidently.legacy.core import ColumnType

from monitoring.checks import load_config
from monitoring.data import CATEGORICAL, LAB_DIR, NUMERICAL, TARGET, load_adult, split_data
from monitoring.model import PROBABILITY, add_predictions, evaluate, load_or_train
from monitoring.scenarios import load_scenario, simulate_day

RESULTS_DIR = LAB_DIR / "results"

# The drift tests compared, by their Evidently names. KS and Wasserstein only work on numbers,
# chi-square only on categories.
NUMERIC_TESTS = ["ks", "wasserstein", "psi", "jensenshannon"]
CATEGORICAL_TESTS = ["chisquare", "psi", "jensenshannon"]
# These give a p-value: drift means a score BELOW the threshold. The others give a distance:
# drift means a score at or above the threshold.
P_VALUE_TESTS = {"ks", "chisquare"}
# What Evidently uses by default for a reference this size (the daily checks before calibration)
DEFAULT_TESTS = {"numeric": "wasserstein", "categorical": "jensenshannon"}

# The monitored columns: the features, and the model's predicted probability
NUMERIC_COLUMNS = NUMERICAL + [PROBABILITY]
CATEGORICAL_COLUMNS = CATEGORICAL

ALLOWED_FALSE_ALARMS = 1 / 500  # share of normal days on which a column's calibrated threshold is crossed
NORMAL_DAYS = 2000              # for calibrating the thresholds
FRESH_NORMAL_DAYS = 1000        # for checking them on days they weren't calibrated on
DRIFT_DAYS = 100                # per column and drift strength
# Drift strengths: the tilt of a numeric_shift event, and the factor of a category_shift event
TILTS = [0.1, 0.2, 0.3, 0.4, 0.5]
FACTORS = [1.5, 2, 3, 4]
# A different seed for each set of simulated days, none of them the scenario's
SEEDS = {"normal": 101, "fresh": 102, "drift": 103}

# Chart colors: one per test, in every panel, from a palette checked for color-blind readers
TEST_COLORS = {"ks": "#2a78d6", "wasserstein": "#eb6834", "psi": "#1baf7a", "jensenshannon": "#eda100",
               "chisquare": "#e87ba4"}
TEST_NAMES = {"ks": "Kolmogorov-Smirnov", "wasserstein": "Wasserstein", "psi": "PSI",
              "jensenshannon": "Jensen-Shannon", "chisquare": "Chi-square"}
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def kind_of(column):
    return "categorical" if column in CATEGORICAL_COLUMNS else "numeric"


def pairs():
    """Every (column, test) pair compared: each monitored column with each test for its kind."""
    return ([(column, test) for column in NUMERIC_COLUMNS for test in NUMERIC_TESTS]
            + [(column, test) for column in CATEGORICAL_COLUMNS for test in CATEGORICAL_TESTS])


class DriftTests:
    """Evidently's drift tests, scoring one column of a day against the reference data."""

    def __init__(self, reference):
        self.reference = reference
        self.tests = {}
        for kind, names in (("numeric", NUMERIC_TESTS), ("categorical", CATEGORICAL_TESTS)):
            column_type = ColumnType.Numerical if kind == "numeric" else ColumnType.Categorical
            for name in names:
                empty = pd.Series(dtype=float)
                self.tests[(name, kind)] = (get_stattest(empty, empty, column_type, name), column_type)

    def score(self, day, column, test):
        """The test's score for this column, ignoring missing values as the reports do."""
        stattest, column_type = self.tests[(test, kind_of(column))]
        # Chi-square divides by zero when a day has a category the reference never saw. It then
        # reports certain drift (p-value 0), which shows up in its false-alarm rate.
        with np.errstate(divide="ignore"):
            return stattest(self.reference[column].dropna(), day[column].dropna(), column_type, None).drift_score

    def default_threshold(self, column, test):
        return self.tests[(test, kind_of(column))][0].default_threshold


def is_drift(test, score, threshold):
    return score < threshold if test in P_VALUE_TESTS else score >= threshold


def simulated_days(scenario, pool, seed, count, events=()):
    """count simulated days (their data and true labels), with the given drift events active on every day."""
    setup = {**scenario, "seed": seed, "events": [{**event, "days": [1, count]} for event in events]}
    for day in range(1, count + 1):
        batch = simulate_day(setup, pool, day)
        yield batch.data.assign(**{TARGET: batch.labels})


def normal_day_results(scenario, pool, model, tests, seed, count):
    """For each normal day: every (column, test) score, and the performance metrics of its predictions."""
    scores, performance = [], []
    for day in simulated_days(scenario, pool, seed, count):
        day = add_predictions(model, day)
        scores.append({pair: tests.score(day, *pair) for pair in pairs()})
        performance.append(evaluate(day))
    return pd.DataFrame(scores), pd.DataFrame(performance)


def calibrate(scores, tests):
    """Per (column, test): its default threshold, how often normal days cross it, and the calibrated threshold."""
    rows = []
    for column, test in pairs():
        values, default = scores[(column, test)], tests.default_threshold(column, test)
        quantile = ALLOWED_FALSE_ALARMS if test in P_VALUE_TESTS else 1 - ALLOWED_FALSE_ALARMS
        rows.append({
            "column": column, "kind": kind_of(column), "test": test, "default_threshold": default,
            "false_alarms_at_default_%": 100 * np.mean([is_drift(test, v, default) for v in values]),
            "calibrated_threshold": float(np.quantile(values, quantile)),
        })
    return pd.DataFrame(rows)


def drift_events(pool):
    """For every feature, drift events of growing strength that change only that feature's distribution."""
    for column in NUMERICAL:
        for tilt in TILTS:
            yield column, f"higher {column}", tilt, {"name": "calibration", "type": "numeric_shift",
                                                     "tilt": {column: tilt}}
    for column in CATEGORICAL:
        category = pool[column].value_counts().index[1]  # the second most common: the most common barely moves
        for factor in FACTORS:
            yield column, f"more {category}", factor, {"name": "calibration", "type": "category_shift",
                                                       "weights": {column: {category: factor}}}


def detection_rates(scenario, pool, tests, thresholds):
    """How often each test, with its calibrated threshold, catches drift of each strength in each feature."""
    rows = []
    for column, change, strength, event in drift_events(pool):
        days = list(simulated_days(scenario, pool, SEEDS["drift"], DRIFT_DAYS, [event]))
        for test in (NUMERIC_TESTS if kind_of(column) == "numeric" else CATEGORICAL_TESTS):
            threshold = thresholds[(column, test)]
            caught = [is_drift(test, tests.score(day, column, test), threshold) for day in days]
            rows.append({"column": column, "kind": kind_of(column), "drift": change, "strength": strength,
                         "test": test, "detected_%": 100 * np.mean(caught)})
    return pd.DataFrame(rows)


def false_warning_rates(scores, setups):
    """
    On each day, whether any feature drifts and whether the predictions drift, under each setup
    (a test and threshold per column). Returns the % of days with each kind of false WARNING.
    """
    rows = []
    for name, setup in setups.items():
        drifted = pd.DataFrame({column: [is_drift(test, v, threshold) for v in scores[(column, test)]]
                                for column, (test, threshold) in setup.items()})
        features, predictions = drifted.drop(columns=PROBABILITY).any(axis=1), drifted[PROBABILITY]
        rows.append({"setup": name, "feature_drift_%": 100 * features.mean(),
                     "prediction_drift_%": 100 * predictions.mean(),
                     "any_drift_warning_%": 100 * (features | predictions).mean()})
    return pd.DataFrame(rows)


def performance_noise(performance, baseline, max_drop):
    """
    How far each metric fell below its baseline on normal days, and, for the metrics the
    performance rule tests (max_drop), on how many normal days the rule would fire.
    """
    drops = baseline - performance[list(baseline)]
    # The same limit as the performance report's test (see checks.performance_report)
    false_alarms = {name: 100 * (performance[name] < round(baseline[name] - drop, 3)).mean()
                    for name, drop in max_drop.items()}
    return pd.DataFrame({
        "metric": list(baseline), "baseline": list(baseline.values()),
        "daily_std": performance.std().tolist(),
        "drop_exceeded_on_1%_of_days": drops.quantile(0.99).tolist(),
        "largest_drop": drops.max().tolist(),
        "max_drop_in_config": [max_drop.get(name) for name in baseline],
        "false_alarms_%": [false_alarms.get(name) for name in baseline],
    })


def add_titles(axes, title, subtitle):
    """A title and a two-line subtitle above a panel, aligned with its left edge."""
    axes.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                  fontsize=9, color=SECONDARY, va="bottom", linespacing=1.5)
    axes.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 44), textcoords="offset points",
                  fontsize=12, color=INK, fontweight="bold", va="bottom")


def plot_false_alarms(axes, thresholds):
    """Ranked bars: on what share of normal days each column 'drifted' with Evidently's default test and threshold."""
    defaults = thresholds[thresholds["test"] == thresholds["kind"].map(DEFAULT_TESTS)]
    defaults = defaults.sort_values("false_alarms_at_default_%")
    rows = range(len(defaults))
    axes.barh(rows, defaults["false_alarms_at_default_%"], height=0.5,
              color=[TEST_COLORS[test] for test in defaults["test"]])
    for row, value in zip(rows, defaults["false_alarms_at_default_%"]):
        if value >= 1:  # label the columns that matter; the axis and the CSV carry the rest
            axes.annotate(f"{value:.1f}%", xy=(value, row), xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=9, color=SECONDARY)
    names = defaults["column"].replace({PROBABILITY: "predicted probability"})
    axes.set_yticks(list(rows), names, color=SECONDARY, fontsize=9.5)
    axes.set_xlim(0, 100)
    axes.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    axes.grid(axis="x")
    axes.spines["left"].set_visible(True)
    axes.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TEST_COLORS[test]) for test in DEFAULT_TESTS.values()],
                labels=["Wasserstein (numeric columns)", "Jensen-Shannon (categorical columns)"],
                loc="lower right")
    add_titles(axes, "False alarms with Evidently's default threshold of 0.1",
               f"Share of {NORMAL_DAYS:,} drift-free days on which each column was flagged.\n"
               f"With calibrated thresholds: {100 * ALLOWED_FALSE_ALARMS:.1f}% of days for every column.")


def plot_detection(axes, detection, kind, title, subtitle, x_label, tick_format):
    """Lines: the share of drifted days each test caught, by drift strength, averaged over the features."""
    rates = (detection[detection["kind"] == kind].groupby(["test", "strength"])["detected_%"].mean()
             .unstack("test"))
    ranking = rates.mean().sort_values(ascending=False).index  # best test first
    for test in reversed(ranking):  # the best test is drawn last, on top
        axes.plot(rates.index, rates[test], color=TEST_COLORS[test], linewidth=2, marker="o", markersize=7,
                  markeredgecolor=SURFACE, markeredgewidth=1.6, label=TEST_NAMES[test],
                  solid_capstyle="round", solid_joinstyle="round")
    # The lines meet at the strongest drift, so the best test is named where the tests are furthest
    # apart; the legend, listed best first, names them all
    widest = (rates.max(axis=1) - rates.min(axis=1)).idxmax()
    axes.annotate(TEST_NAMES[ranking[0]], xy=(widest, rates.loc[widest, ranking[0]]), xytext=(-7, 5),
                  textcoords="offset points", ha="right", va="bottom", fontsize=9.5, color=INK)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[::-1], labels[::-1], loc="lower right")
    axes.set_xticks(rates.index, [tick_format(strength) for strength in rates.index])
    axes.set_ylim(0, 105)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    axes.set_xlabel(x_label, color=MUTED, fontsize=9)
    axes.grid(axis="y")
    axes.spines["bottom"].set_visible(True)
    add_titles(axes, title, subtitle)


def plot(results_dir=RESULTS_DIR):
    """Draws results/calibration.png from the result tables."""
    thresholds = pd.read_csv(results_dir / "thresholds.csv")
    detection = pd.read_csv(results_dir / "detection.csv")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
        "legend.fontsize": 9, "legend.labelcolor": SECONDARY,
    })
    figure = plt.figure(figsize=(12.5, 8))
    grid = figure.add_gridspec(2, 2, width_ratios=[1, 1.1], hspace=0.78, wspace=0.3,
                               left=0.135, right=0.98, top=0.85, bottom=0.08)
    plot_false_alarms(figure.add_subplot(grid[:, 0]), thresholds)
    same_rate = f"Each test's thresholds flag {100 * ALLOWED_FALSE_ALARMS:.1f}% of drift-free days."
    plot_detection(figure.add_subplot(grid[0, 1]), detection, "numeric", "Numeric drift caught by each test",
                   f"Average over the {len(NUMERICAL)} numeric features, {DRIFT_DAYS} drifted days per point.\n"
                   + same_rate, "Drift strength: tilt toward higher values", lambda strength: f"{strength:g}")
    plot_detection(figure.add_subplot(grid[1, 1]), detection, "categorical", "Category drift caught by each test",
                   f"Average over the {len(CATEGORICAL)} categorical features, {DRIFT_DAYS} drifted days per point.\n"
                   + same_rate, "Drift strength: one category made this many times as common",
                   lambda strength: f"{strength:g}×")
    figure.savefig(results_dir / "calibration.png", dpi=150)
    plt.close(figure)


def main():
    start = time.time()
    scenario = load_scenario()
    model, reference = load_or_train()
    _, _, pool = split_data(load_adult())
    tests = DriftTests(reference)

    print(f"1. Scoring {NORMAL_DAYS} normal days...")
    scores, performance = normal_day_results(scenario, pool, model, tests, SEEDS["normal"], NORMAL_DAYS)
    table = calibrate(scores, tests)
    thresholds = {(row.column, row.test): row.calibrated_threshold for row in table.itertuples()}

    print(f"2. Measuring detection on days with drift ({time.time() - start:.0f}s so far)...")
    detection = detection_rates(scenario, pool, tests, thresholds)
    average = detection.groupby(["kind", "test"])["detected_%"].mean()
    best = {kind: average[kind].idxmax() for kind in ("numeric", "categorical")}
    table = table.merge(detection.groupby(["column", "test"], as_index=False)["detected_%"].mean()
                        .rename(columns={"detected_%": "average_detection_%"}), how="left")

    print(f"3. Checking on {FRESH_NORMAL_DAYS} fresh normal days ({time.time() - start:.0f}s so far)...")
    fresh_scores, _ = normal_day_results(scenario, pool, model, tests, SEEDS["fresh"], FRESH_NORMAL_DAYS)
    setups = {
        "Evidently defaults": {c: (DEFAULT_TESTS[kind_of(c)], tests.default_threshold(c, DEFAULT_TESTS[kind_of(c)]))
                               for c in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS},
        "calibrated": {c: (best[kind_of(c)], thresholds[(c, best[kind_of(c)])])
                       for c in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS},
    }
    false_warnings = false_warning_rates(fresh_scores, setups)
    noise = performance_noise(performance, evaluate(reference), load_config()["performance"]["max_drop"])

    RESULTS_DIR.mkdir(exist_ok=True)
    # Significant digits rather than decimals: some p-value thresholds are tiny
    for name, frame in (("thresholds", table), ("detection", detection), ("false_warnings", false_warnings),
                        ("performance_noise", noise)):
        frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False, float_format="%.4g")
    plot()

    print(f"\nDone in {time.time() - start:.0f}s. Average detection rate by test (at the same false-alarm rate):")
    print(average.round(1).to_string())
    print(f"\nFalse WARNINGs on fresh normal days:\n{false_warnings.round(1).to_string(index=False)}")
    print(f"\nPerformance on normal days:\n{noise.to_string(index=False)}")
    print("\nSettings for config/monitoring.yaml:")
    print(f"  num_method: {best['numeric']}\n  cat_method: {best['categorical']}\n  per_column_threshold:")
    for column in NUMERICAL + CATEGORICAL:
        print(f"    {column}: {thresholds[(column, best[kind_of(column)])]:.4g}")
    print(f"  prediction drift: method {best['numeric']}, threshold "
          f"{thresholds[(PROBABILITY, best['numeric'])]:.4g}")


if __name__ == "__main__":
    main()
