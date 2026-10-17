"""
Customer segmentation pipeline (Airflow 3, TaskFlow API).

Groups credit-card customers into clusters with K-Means. Each task writes its
output to a file in this run's folder and passes only the file PATH to the
next task, so Airflow's database (XCom) only stores short strings.
"""
import os
import re
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task, get_current_context
from airflow.sdk.exceptions import AirflowFailException

from src import lab

# Inside the containers, dags/ is /opt/airflow/dags and working_data/ is /opt/airflow/working_data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WORKING_DIR = "/opt/airflow/working_data"


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
            report = lab.validate_data(raw_path, run_folder())
        except ValueError as err:
            # Bad data won't fix itself, so fail now instead of retrying
            raise AirflowFailException(f"Data check failed: {err}") from err
        print(f"Kept {report['rows_out']} of {report['rows_in']} rows. Missing values: {report['missing']}")
        return report

    @task
    def preprocess(validated):
        """Scale the validated data; return the paths of the scaled data and the scaler."""
        preprocessed_data = lab.preprocess(validated["clean_path"], run_folder())
        return preprocessed_data

    @task
    def make_k_values():
        """List the numbers of clusters to try. Each one becomes its own parallel task."""
        return list(range(lab.K_MIN, lab.K_MAX + 1))

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

    @task
    def predict(model_info, prepared):
        """Assign the customers in test.csv to clusters."""
        clusters = lab.predict(model_info["model_path"], prepared["scaler_path"], os.path.join(DATA_DIR, "test.csv"))
        print(f"Clusters: {clusters}")
        return clusters

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
    predict(best, prepared)


customer_segmentation()
