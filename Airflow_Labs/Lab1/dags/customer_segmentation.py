"""
Customer segmentation pipeline (Airflow 3, TaskFlow API).

Groups credit-card customers into clusters with K-Means. Each task writes its
output to a file in this run's folder and passes only the file PATH to the
next task, so Airflow's database (XCom) only stores short strings.
The k range, quality bar and features are Params: form fields shown when the
DAG is triggered, so every run can use different settings without code changes.
"""
import os
import re
from datetime import timedelta

import pendulum
from airflow.sdk import Param, TriggerRule, dag, task, get_current_context
from airflow.sdk.exceptions import AirflowFailException

from src import lab

# Inside the containers, dags/ is /opt/airflow/dags and working_data/ is /opt/airflow/working_data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WORKING_DIR = "/opt/airflow/working_data"
# Always holds the current official model; only replaced when a new model passes the quality gate
LATEST_DIR = os.path.join(WORKING_DIR, "latest")


def run_folder():
    """
    Returns this run's folder, /opt/airflow/working_data/<run id>, creating it if needed.
    A run id looks like manual__2026-10-17T21:09:12+00:00. The ':' and '+' are replaced
    with '_' because Windows can't store those characters in folder names.
    Only works inside a running task, because that's when Airflow knows the run id.
    """
    run_id = get_current_context()["run_id"]
    folder = os.path.join(WORKING_DIR, re.sub(r"[^A-Za-z0-9_.-]", "_", run_id))
    os.makedirs(folder, exist_ok=True)
    return folder


def setting(name):
    """
    Returns one Param's value for this run: what was entered in the trigger form,
    or the default if it was left unchanged. Only works inside a running task.
    """
    return get_current_context()["params"][name]


@dag(
    dag_id="customer_segmentation",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,  # only runs when triggered
    catchup=False,
    tags=["lab1", "clustering"],
    # Applied to every task: on failure, try up to 2 more times, 30 seconds apart
    default_args={
        "owner": "Sai Punith Kolla",
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
    },
    # Form fields shown when you click Trigger. Airflow checks every value against
    # these rules (type, minimum, maximum, ...) before the run is allowed to start.
    params={
        "k_min": Param(
            lab.K_MIN,
            type="integer",
            minimum=2,
            maximum=20,
            title="Smallest number of clusters",
            description="At least 2, because the silhouette score needs 2 or more clusters.",
        ),
        "k_max": Param(
            lab.K_MAX,
            type="integer",
            minimum=3,
            maximum=20,
            title="Largest number of clusters",
            description="Every k from the smallest to the largest is trained as its own parallel task.",
        ),
        "min_silhouette": Param(
            lab.MIN_SILHOUETTE,
            type="number",
            minimum=-1,
            maximum=1,
            title="Quality bar (minimum silhouette score)",
            description="The new model only replaces the official one if its silhouette score is at least this.",
        ),
        "features": Param(
            lab.FEATURES,
            type="array",
            examples=lab.FEATURES,
            items={"type": "string", "enum": lab.FEATURES},
            minItems=2,
            uniqueItems=True,
            title="Customer columns to cluster on",
            description="Pick at least 2.",
        ),
    },
)
def customer_segmentation():

    @task
    def load_data():
        """Copy the raw data into this run's folder and return its path."""
        lab_data_path = lab.load_data(os.path.join(DATA_DIR, "file.csv"), run_folder())
        return lab_data_path

    @task
    def validate_data(raw_path):
        """Check the data before modelling. Bad data fails the run at once, with no retries."""
        try:
            report = lab.validate_data(raw_path, run_folder(), features=setting("features"))
        except ValueError as err:
            # Bad data won't fix itself, so fail now instead of retrying
            raise AirflowFailException(f"Data check failed: {err}") from err
        print(f"Kept {report['rows_out']} of {report['rows_in']} rows. Missing values: {report['missing']}")
        return report

    @task
    def preprocess(validated):
        """Scale the validated data; return the paths of the scaled data and the scaler."""
        preprocessed_data = lab.preprocess(validated["clean_path"], run_folder(), features=setting("features"))
        return preprocessed_data

    @task
    def make_k_values():
        """List the numbers of clusters to try. Each one becomes its own parallel task."""
        k_min, k_max = setting("k_min"), setting("k_max")
        if k_min >= k_max:
            # A wrong setting won't fix itself either, so fail now instead of retrying
            raise AirflowFailException(f"k_min ({k_min}) must be smaller than k_max ({k_max})")
        k_values = list(range(k_min, k_max + 1))
        print(f"Trying k = {k_values}")
        return k_values

    # map_index_template labels each parallel copy in the UI as k=2, k=3, ... instead of 0, 1, ...
    @task(map_index_template="{{ k_label }}")
    def train_kmeans(prepared, k):
        """Train and score one K-Means model. Airflow runs one copy of this task per k, in parallel."""
        get_current_context()["k_label"] = f"k={k}"
        result = lab.train_kmeans(prepared["data_path"], run_folder(), k)
        print(f"k={k}: SSE {result['sse']:.2f}, silhouette {result['silhouette']:.3f}")
        return result

    @task
    def select_best_k(results):
        """Collect the results of all parallel train_kmeans tasks and pick the best k."""
        best = lab.select_best_k(list(results), run_folder())
        print(f"Best k: {best['best_k']} (by {best['method']}), silhouette {best['silhouette']:.3f}")
        return best

    # A branch task returns the task_id(s) to run next; Airflow skips the other path
    @task.branch
    def quality_gate(best):
        """Decide which path runs next: promote the model, or flag it as too weak."""
        min_silhouette = setting("min_silhouette")
        if lab.passes_quality_gate(best["silhouette"], min_silhouette):
            print(f"Silhouette {best['silhouette']:.3f} >= {min_silhouette}: promoting the model")
            return "promote_model"
        print(f"Silhouette {best['silhouette']:.3f} < {min_silhouette}: flagging the model as low quality")
        return "flag_low_quality"

    @task
    def promote_model(best, prepared):
        """Make the chosen model the official latest model."""
        promoted = lab.promote_model(best, prepared["scaler_path"], LATEST_DIR, features=setting("features"))
        print(f"Promoted the k={best['best_k']} model to {LATEST_DIR}")
        return promoted

    @task
    def predict(promoted, validated):
        """Use the promoted model: assign the test customers to clusters and describe every cluster."""
        features = setting("features")
        clusters = lab.predict(
            promoted["model_path"], promoted["scaler_path"], os.path.join(DATA_DIR, "test.csv"), features=features
        )
        profile_path = lab.profile_clusters(
            promoted["model_path"], promoted["scaler_path"], validated["clean_path"], run_folder(), features=features
        )
        print(f"Clusters: {clusters}")
        print(f"Cluster profile saved to {profile_path}")
        return {"test_clusters": clusters, "profile_path": profile_path}

    @task
    def flag_low_quality(best):
        """Record why the model was not promoted. The previous latest model stays in place."""
        report_path = lab.write_low_quality_report(best, setting("min_silhouette"), run_folder())
        print(f"Model NOT promoted, see {report_path}")
        return report_path

    # Runs once one path has succeeded and nothing has failed; the skipped path doesn't block it
    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def summarize(best):
        """Write the run summary, whichever path ran."""
        # xcom_pull reads another task's result; it's None when that task was skipped
        predictions = get_current_context()["ti"].xcom_pull(task_ids="predict")
        summary = {
            "settings": {name: setting(name) for name in ("k_min", "k_max", "min_silhouette", "features")},
            "best_k": best["best_k"],
            "method": best["method"],
            "silhouette": best["silhouette"],
            "decision": "promoted" if predictions else "flagged as low quality",
            "test_clusters": predictions["test_clusters"] if predictions else None,
        }
        lab.save_json(summary, os.path.join(run_folder(), "summary.json"))
        print(f"Run summary: {summary}")
        return summary

    # Passing one task's result into another task creates the arrow between them
    raw = load_data()
    validated = validate_data(raw)
    prepared = preprocess(validated)

    # Fan out: one train_kmeans task per k.
    # partial() = arguments shared by every copy; expand() = one copy per value in the list
    k_values = make_k_values()
    results = train_kmeans.partial(prepared=prepared).expand(k=k_values)

    # Fan in: select_best_k receives the list of all the parallel results
    best = select_best_k(results)

    # Branch: quality_gate picks ONE of the two paths; the other is skipped
    promoted = promote_model(best, prepared)
    predicted = predict(promoted, validated)
    flagged = flag_low_quality(best)
    quality_gate(best) >> [promoted, flagged]

    # Join: summarize runs after whichever path ran
    [predicted, flagged] >> summarize(best)


customer_segmentation()
