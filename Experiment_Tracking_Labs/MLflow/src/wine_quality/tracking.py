"""
Helpers for the MLflow tracking server. Inside docker-compose, MLFLOW_TRACKING_URI points at the
server (http://mlflow:5000), and MLFLOW_UI_URL is the same server's address in your browser.
"""
import contextlib
import os
import warnings

import mlflow
from mlflow.utils import logging_utils

# MLflow hints, for every table with integer columns, that integers can't hold missing values. The wine
# data has no missing values, and its integer columns (is_red, quality and the label) never will.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")


def use_experiment(name):
    """Sends the next runs to this experiment, creating it the first time."""
    return mlflow.set_experiment(name)


@contextlib.contextmanager
def quiet_mlflow():
    """Silences MLflow's own messages (e.g. "Created version '2'...") inside the block: the lab prints its own."""
    logging_utils.disable_logging()
    try:
        yield
    finally:
        logging_utils.enable_logging()


def run_link(run):
    """The run's page in the MLflow UI, or just its ID when the UI's address isn't known."""
    base = os.environ.get("MLFLOW_UI_URL")
    if not base:
        return run.info.run_id
    return f"{base}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"


def model_link(name, version=None):
    """A registered model's page (or one version's page) in the MLflow UI, or its name if the UI's address isn't known."""
    base = os.environ.get("MLFLOW_UI_URL")
    page = f"models/{name}" + (f"/versions/{version}" if version else "")
    return f"{base}/#/{page}" if base else page
