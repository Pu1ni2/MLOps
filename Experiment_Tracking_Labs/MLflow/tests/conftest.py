"""
Shared test setup. Tests that use MLflow get a fresh MLflow store in a temporary folder (a SQLite database
plus a local artifacts folder), so they never touch the tracking server, its registry or the lab's files.
"""
import copy

import mlflow
import pytest

from wine_quality.config import load_config
from wine_quality.data import prepare


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def splits(config):
    return prepare(config)[0]


@pytest.fixture
def small_config(config):
    """The lab's settings, made small so a test runs in seconds: fewer trees, attempts and re-draws."""
    small = copy.deepcopy(config)
    small["baselines"]["random_forest"]["n_estimators"] = 30
    small["tuning"].update(trials=3, max_trees=40, early_stopping_rounds=10)
    small["registry"]["resamples"] = 200
    return small


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """A temporary MLflow store, used by everything that talks to MLflow during the test."""
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)  # artifacts go to ./mlruns, inside the temporary folder
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("tests")
    yield uri
    mlflow.end_run()
    mlflow.set_tracking_uri(previous)
