"""
The original lab's analysis, kept as a baseline to compare against:
split the census data by education level and compare the two halves with
Evidently's DataDriftPreset. The split guarantees drift, because the "current"
half contains only 3 education levels that the "reference" half never has.

    docker compose run --rm monitor python -m monitoring.baseline
"""
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from monitoring.data import CATEGORICAL, LAB_DIR, NUMERICAL, TARGET, load_adult
from monitoring.workspace import get_or_create_project, open_workspace

REPORTS_DIR = LAB_DIR / "reports"
PROJECT_NAME = "Baseline: original lab (education split)"


def main():
    df = load_adult()
    in_current = df["education"].isin(["Some-college", "HS-grad", "Bachelors"])
    reference, current = df[~in_current], df[in_current]

    # As in the original lab, the income label is (wrongly) included as if it were a feature
    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL + [TARGET])
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    html_path = REPORTS_DIR / "baseline_education_split.html"
    snapshot.save_html(str(html_path))

    workspace = open_workspace()
    project = get_or_create_project(workspace, PROJECT_NAME)
    workspace.add_run(project.id, snapshot, include_data=False)

    print(f"Reference: {len(reference)} rows, current: {len(current)} rows")
    print(f"Report saved to {html_path} and to the dashboard project '{PROJECT_NAME}'")
    return snapshot


if __name__ == "__main__":
    main()
