"""
Customer segmentation pipeline (Airflow 3, TaskFlow API).

Groups credit-card customers into clusters with K-Means. Each task writes its
output to a file in this run's folder and passes only the file PATH to the
next task, so Airflow's database (XCom) only stores short strings.
"""
import os
import re

import pendulum
from airflow.sdk import dag, task, get_current_context

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
    default_args={"owner": "Sai Punith Kolla"},
)
def customer_segmentation():

    @task
    def load_data():
        """Copy the raw data into this run's folder and return its path."""
        lab_data_path = lab.load_data(os.path.join(DATA_DIR, "file.csv"), run_folder())
        return lab_data_path

    @task
    def preprocess(raw_path):
        """Select, clean and scale the data; return the paths of the scaled data and the scaler."""
        preprocessed_data = lab.preprocess(raw_path, run_folder())
        return preprocessed_data

    @task
    def build_model(prepared):
        """Pick the best number of clusters and save that model."""
        model_info = lab.build_model(prepared["data_path"], run_folder())
        print(f"Best k: {model_info['best_k']}")
        return model_info

    @task
    def predict(model_info, prepared):
        """Assign the customers in test.csv to clusters."""
        clusters = lab.predict(model_info["model_path"], prepared["scaler_path"], os.path.join(DATA_DIR, "test.csv"))
        print(f"Clusters: {clusters}")
        return clusters

    # Passing one task's result into another task creates the arrow between them
    raw = load_data()
    prepared = preprocess(raw)
    model_info = build_model(prepared)
    predict(model_info, prepared)


customer_segmentation()
