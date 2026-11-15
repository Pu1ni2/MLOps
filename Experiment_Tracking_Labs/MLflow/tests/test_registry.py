"""The registry gate: every rule, the aliases it moves, and the rollback."""
import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from wine_quality import registry
from wine_quality.data import FEATURES, TARGET
from wine_quality.models import build_model, log_model

SETTINGS = {"min_win_share": 0.9, "max_brier_increase": 0.0}


def test_the_first_model_becomes_champion():
    promoted, reason = registry.decide({"auc": 0.7, "brier": 0.2}, None, None, SETTINGS)
    assert promoted and "first model" in reason


def test_a_win_that_could_be_luck_is_rejected():
    promoted, reason = registry.decide({"auc": 0.85, "brier": 0.10}, {"auc": 0.84, "brier": 0.11}, 0.6, SETTINGS)
    assert not promoted and "only 60%" in reason


def test_a_worse_brier_score_is_rejected():
    promoted, reason = registry.decide({"auc": 0.90, "brier": 0.12}, {"auc": 0.84, "brier": 0.11}, 1.0, SETTINGS)
    assert not promoted and "Brier" in reason


def test_a_clear_and_calibrated_win_is_promoted():
    promoted, _ = registry.decide({"auc": 0.90, "brier": 0.10}, {"auc": 0.84, "brier": 0.11}, 0.95, SETTINGS)
    assert promoted


def test_win_share_counts_redraws_won():
    truth = np.array([0, 1] * 50)
    perfect, useless = truth.astype(float), np.full(100, 0.5)
    assert registry.win_share(truth, perfect, useless, 200, seed=1) == 1.0
    assert registry.win_share(truth, perfect, perfect, 200, seed=1) == 0.0  # a tie is not a win


def log(family, splits, shuffle_labels=False):
    """Trains and logs a small model; shuffled labels give a model that knows nothing (AUC about 0.5)."""
    train = splits["train"]
    labels = train[TARGET].sample(frac=1, random_state=0).to_numpy() if shuffle_labels else train[TARGET]
    model = build_model(family, {"n_estimators": 30} if family == "random_forest" else {}, seed=42)
    model.fit(train[FEATURES], labels)
    with mlflow.start_run() as run:
        info = log_model(model, family, train[FEATURES])
    return info.model_uri, {"run_id": run.info.run_id, "model_family": family, "validation_auc": 0.0}


def aliases(name):
    return {alias: int(version) for alias, version in MlflowClient().get_registered_model(name).aliases.items()}


def test_gate_promotes_rejects_and_rolls_back(small_config, splits, tracking, capsys):
    name = small_config["registry"]["model_name"]
    weak = log("logistic_regression", splits, shuffle_labels=True)
    strong = log("random_forest", splits)

    first = registry.gate(*weak, splits, small_config)
    assert first["promoted"] and aliases(name) == {"champion": 1}

    second = registry.gate(*strong, splits, small_config)
    assert second["promoted"] and aliases(name) == {"champion": 2, "previous": 1}
    version = MlflowClient().get_model_version(name, "2")
    assert version.tags["gate.decision"] == "promoted" and version.description.startswith("Promoted")

    again = registry.gate(*strong, splits, small_config)  # the same model again: a tie, so rejected
    assert not again["promoted"] and aliases(name) == {"champion": 2, "previous": 1, "challenger": 3}

    registry.rollback(small_config)
    assert aliases(name)["champion"] == 1 and aliases(name)["previous"] == 2
    registry.status(small_config)
    assert "@champion" in capsys.readouterr().out


def test_rollback_without_a_previous_version_stops(small_config, splits, tracking):
    registry.gate(*log("logistic_regression", splits), splits, small_config)
    with pytest.raises(SystemExit, match="Nothing to roll back to"):
        registry.rollback(small_config)
