"""The dashboard: one tab, no duplicates on reruns, and every chart line reads a metric the reports contain."""
from evidently.ui.workspace import Workspace

from monitoring import checks, dashboard
from monitoring.data import TARGET
from monitoring.model import add_predictions
from monitoring.scenarios import simulate_day


def test_one_tab_lists_every_panel_once():
    model = dashboard.dashboard_model("summary")
    assert len(model.tabs) == 1  # the open-source service rejects more
    assert sorted(model.tabs[0].panels) == sorted(panel.id for panel in model.panels)


def test_rerunning_the_setup_never_duplicates(tmp_path):
    workspace = Workspace(str(tmp_path))
    for _ in range(2):
        project = dashboard.recreate_project(workspace)
        dashboard.save_dashboard(workspace, project, "first save")
        dashboard.save_dashboard(workspace, project, "second save")
    assert len(workspace.search_project(dashboard.PROJECT_NAME)) == 1
    saved = workspace.get_dashboard(project.id)
    assert len(saved.panels) == len(dashboard.dashboard_model("summary").panels)
    assert len(saved.tabs[0].panels) == len(set(saved.tabs[0].panels)) == len(saved.panels)


def test_every_chart_line_finds_its_metric(scenario, pool, config, model_and_reference):
    model, reference = model_and_reference
    batch = simulate_day(scenario, pool, 1)
    predictions = add_predictions(model, batch.data)
    data = checks.data_report(reference, config).run(checks.to_dataset(predictions), checks.to_dataset(reference))
    labeled = predictions.assign(**{TARGET: batch.labels})
    performance = checks.performance_report(reference, config).run(checks.to_labeled_dataset(labeled),
                                                                    checks.to_labeled_dataset(reference))
    reports = {dashboard.DATA_TAG: data.dict()["metrics"], dashboard.PERFORMANCE_TAG: performance.dict()["metrics"]}
    for panel in dashboard.dashboard_model("summary").panels:
        for line in panel.values:
            (tag,) = line.tags
            # value_type picks the count or the share inside a metric's value; the other labels are its settings
            settings = {key: value for key, value in line.metric_labels.items() if key != "value_type"}
            found = [metric for metric in reports[tag] if metric["config"]["type"] == line.metric
                     and all(str(metric["config"].get(key)) == value for key, value in settings.items())]
            assert found, f"{panel.title}: {line.legend}"
            if "value_type" in line.metric_labels:
                assert line.metric_labels["value_type"] in found[0]["value"]
