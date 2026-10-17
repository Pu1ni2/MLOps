"""
Shared test setup.

Inside Airflow, the dags/ folder is automatically on Python's import path, which
is how `from src import lab` works in the DAG. pytest runs as a plain Python
process, so the folder is added here.
"""
import os
import sys

import pandas as pd
import pytest
from sklearn.datasets import make_blobs

DAGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dags"))
DATA_DIR = os.path.join(DAGS_DIR, "data")
sys.path.insert(0, DAGS_DIR)

from src import lab  # noqa: E402  (needs DAGS_DIR on the path first)


@pytest.fixture
def real_csv():
    """The lab's real customer data (8,950 customers)."""
    return os.path.join(DATA_DIR, "file.csv")


@pytest.fixture
def test_csv():
    """The lab's test customers to assign to clusters."""
    return os.path.join(DATA_DIR, "test.csv")


@pytest.fixture
def blob_csv(tmp_path):
    """Small made-up data with exactly 3 obvious, well-separated groups (300 rows)."""
    points, _ = make_blobs(n_samples=300, centers=3, n_features=len(lab.FEATURES), cluster_std=0.5, random_state=0)
    path = tmp_path / "blobs.csv"
    pd.DataFrame(points, columns=lab.FEATURES).to_csv(path, index=False)
    return str(path)
