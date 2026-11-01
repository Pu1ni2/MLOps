"""
The monitoring dashboard, defined in code: an Evidently project whose panels chart the daily
reports over time. The dashboard service shows it at http://localhost:8000.

The whole dashboard is saved as one model in a single call. In Evidently 0.7.23, adding panels
one by one with project.dashboard.add_panel lists a panel twice when its tab already exists, and
the open-source service allows only one tab, so text panels split the dashboard into sections.
"""
from evidently.sdk.models import DashboardModel, DashboardTabModel, PanelMetric
from evidently.sdk.panels import bar_plot_panel, counter_panel, line_plot_panel, text_panel

from monitoring.checks import PERFORMANCE_METRICS
from monitoring.data import CATEGORICAL, NUMERICAL
from monitoring.model import PROBABILITY

PROJECT_NAME = "Census income model: daily monitoring"
PROJECT_DESCRIPTION = "A simulated month of production for the census income model, with drift injected on purpose"

# Every report is tagged with its kind, so each chart reads only the reports it is about
DATA_TAG = "data"
PERFORMANCE_TAG = "performance"

# The columns that have missing values in this data
COLUMNS_WITH_MISSING = ["workclass", "occupation", "native-country"]


def recreate_project(workspace):
    """Creates the project, first deleting the one an earlier run made, so a rerun never mixes in old reports."""
    for project in workspace.search_project(PROJECT_NAME):
        workspace.delete_project(project.id)
    return workspace.create_project(PROJECT_NAME, description=PROJECT_DESCRIPTION)


def series(metric, legend, tag, **labels):
    """One line (or bar) in a chart: an Evidently metric's value in every report with this tag."""
    return PanelMetric(legend=legend, metric=metric, metric_labels=labels, tags=[tag])


def dashboard_model(summary):
    """Every panel, from top to bottom, in one tab. The summary is the text at the very top."""
    drifted_features = series("DriftedColumnsCount", "drifted features", DATA_TAG, value_type="count")
    f1 = series("F1Score", "F1", PERFORMANCE_TAG)
    panels = [
        text_panel("Census income model: a simulated month of production", description=summary),
        counter_panel("F1 on the latest day with true labels", [f1], size="half"),
        counter_panel("Drifted features on the latest day", [drifted_features], size="half"),
        bar_plot_panel("Drifted features per day", [drifted_features], size="half"),
        line_plot_panel("Prediction drift: score per day", [
            series("ValueDrift", "predicted probability", DATA_TAG, column=PROBABILITY),
        ], size="half"),
        line_plot_panel("Data quality: share of rows with a problem", [
            series("MissingValueCount", "occupation missing", DATA_TAG, column="occupation", value_type="share"),
            series("OutRangeValueCount", "hours-per-week out of range", DATA_TAG,
                   column="hours-per-week", value_type="share"),
        ], size="half"),
        line_plot_panel("F1 per day of predictions", [f1], size="half"),

        text_panel("Feature drift", description="How far each feature's distribution is from the reference "
                                                "data. A higher score means more drift."),
        line_plot_panel("Numeric features: drift score per day",
                        [series("ValueDrift", column, DATA_TAG, column=column) for column in NUMERICAL], size="half"),
        line_plot_panel("Categorical features: drift score per day",
                        [series("ValueDrift", column, DATA_TAG, column=column) for column in CATEGORICAL], size="half"),

        text_panel("Data quality", description="Missing values, and numbers outside the range seen in the "
                                               "reference data."),
        line_plot_panel("Missing values: share of rows",
                        [series("MissingValueCount", column, DATA_TAG, column=column, value_type="share")
                         for column in COLUMNS_WITH_MISSING], size="half"),
        line_plot_panel("Values outside the reference range: share of rows",
                        [series("OutRangeValueCount", column, DATA_TAG, column=column, value_type="share")
                         for column in NUMERICAL], size="half"),

        text_panel("Performance", description="Each day's predictions, scored once their true labels arrive "
                                              "and dated by the day the predictions were made."),
        line_plot_panel("Performance per day of predictions",
                        [series(metric.__name__, name, PERFORMANCE_TAG) for name, metric in PERFORMANCE_METRICS.items()]),
    ]
    return DashboardModel(tabs=[DashboardTabModel(title="Monitoring", panels=[panel.id for panel in panels])],
                          panels=panels)


def save_dashboard(workspace, project, summary):
    """Saves the whole dashboard, replacing what the project had, so saving again never duplicates panels."""
    workspace.save_dashboard(project.id, dashboard_model(summary))
