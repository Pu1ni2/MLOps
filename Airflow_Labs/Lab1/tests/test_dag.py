"""
DAG integrity tests: load the DAG file the way Airflow does and check its structure.
They catch broken imports, wrong task connections and wrong settings before any run.
"""
import os
from datetime import timedelta

import pytest
from airflow.dag_processing.dagbag import DagBag
from airflow.sdk.exceptions import ParamValidationError

DAGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dags"))


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=DAGS_DIR)


@pytest.fixture(scope="module")
def dag(dagbag):
    return dagbag.dags["customer_segmentation"]


def test_dag_loads_without_import_errors(dagbag):
    assert dagbag.import_errors == {}
    assert "customer_segmentation" in dagbag.dags


def test_dag_has_the_expected_tasks(dag):
    assert set(dag.task_ids) == {
        "load_data", "validate_data", "preprocess", "make_k_values", "train_kmeans", "select_best_k",
        "quality_gate", "promote_model", "predict", "flag_low_quality", "summarize",
    }


@pytest.mark.parametrize(
    "task_id, downstream",
    [
        ("load_data", {"validate_data"}),
        ("validate_data", {"preprocess", "predict"}),  # predict also profiles the clean data
        ("preprocess", {"train_kmeans", "promote_model"}),
        ("make_k_values", {"train_kmeans"}),
        ("train_kmeans", {"select_best_k"}),
        ("select_best_k", {"quality_gate", "promote_model", "flag_low_quality", "summarize"}),
        ("quality_gate", {"promote_model", "flag_low_quality"}),
        ("promote_model", {"predict"}),
        ("predict", {"summarize"}),
        ("flag_low_quality", {"summarize"}),
        ("summarize", set()),
    ],
)
def test_task_connections(dag, task_id, downstream):
    assert dag.get_task(task_id).downstream_task_ids == downstream


def test_training_runs_as_parallel_mapped_tasks(dag):
    assert dag.get_task("train_kmeans").is_mapped


def test_summary_runs_after_whichever_branch_ran(dag):
    assert dag.get_task("summarize").trigger_rule == "none_failed_min_one_success"


def test_every_task_retries_twice_30_seconds_apart(dag):
    for task in dag.tasks:
        assert task.retries == 2, task.task_id
        assert task.retry_delay == timedelta(seconds=30), task.task_id


def test_runs_only_when_triggered(dag):
    assert dag.schedule is None
    assert dag.catchup is False


def test_param_defaults(dag):
    assert dag.params["k_min"] == 2
    assert dag.params["k_max"] == 10
    assert dag.params["min_silhouette"] == 0.40
    assert dag.params["features"] == ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]


@pytest.mark.parametrize(
    "name, bad_value",
    [
        ("k_min", 1),  # the silhouette score needs at least 2 clusters
        ("k_max", 50),  # above the maximum of 20
        ("min_silhouette", 2),  # silhouette is between -1 and 1
        ("features", ["BALANCE"]),  # fewer than 2 columns
        ("features", ["BALANCE", "PAYMENTS"]),  # PAYMENTS is not in test.csv
    ],
)
def test_params_reject_invalid_values(dag, name, bad_value):
    with pytest.raises(ParamValidationError):
        dag.params.get_param(name).resolve(bad_value)
