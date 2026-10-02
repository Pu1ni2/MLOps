"""Shared test fixtures: the data split and a small, fast model."""
import pytest

from src.data import load_data, split_data
from src.train_model import train_model


@pytest.fixture(scope="session")
def split():
    """X_train, X_test, y_train, y_test, loaded once for all tests."""
    return split_data(*load_data())


@pytest.fixture(scope="session")
def small_model(split):
    """A random forest with only 10 trees, so tests run in seconds."""
    X_train, X_test, y_train, y_test = split
    return train_model(X_train, y_train, n_estimators=10)
