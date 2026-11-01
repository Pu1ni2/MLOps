"""Shared, read-only test data, loaded once per test run because it takes a few seconds to build."""
import pytest

from monitoring import checks
from monitoring.data import load_adult, split_data
from monitoring.model import load_or_train
from monitoring.scenarios import load_scenario


@pytest.fixture(scope="session")
def scenario():
    return load_scenario()


@pytest.fixture(scope="session")
def config():
    return checks.load_config()


@pytest.fixture(scope="session")
def pool():
    _, _, pool = split_data(load_adult())
    return pool


@pytest.fixture(scope="session")
def model_and_reference():
    """The saved model and the scored reference data (trained first if they don't exist yet)."""
    return load_or_train()
