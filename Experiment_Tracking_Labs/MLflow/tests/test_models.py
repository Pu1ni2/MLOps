"""Every model family is logged safely (skops) and loads back answering with predict_proba."""
import mlflow
import numpy as np
import pytest

from wine_quality.data import FEATURES, TARGET
from wine_quality.evaluation import scores
from wine_quality.models import TRUSTED_TYPES, build_model, log_model

SMALL = {"logistic_regression": {"max_iter": 500}, "random_forest": {"n_estimators": 20},
         "xgboost": {"n_estimators": 20, "max_depth": 3}}


@pytest.mark.parametrize("family", list(TRUSTED_TYPES))
def test_each_family_loads_back_with_the_same_probabilities(family, splits, tracking):
    train, validation = splits["train"], splits["validation"]
    model = build_model(family, SMALL[family], seed=42).fit(train[FEATURES], train[TARGET])
    with mlflow.start_run():
        info = log_model(model, family, train[FEATURES])
    loaded = mlflow.pyfunc.load_model(info.model_uri).predict(validation[FEATURES])
    assert loaded.shape == (len(validation), 2)
    assert np.allclose(loaded, model.predict_proba(validation[FEATURES]))


def test_an_unknown_family_is_refused():
    with pytest.raises(ValueError, match="Unknown model family"):
        build_model("svm", {}, seed=42)


def test_scores_are_perfect_for_a_perfect_model():
    truth = np.array([0, 0, 1, 1])
    result = scores(truth, np.array([0.0, 0.0, 1.0, 1.0]))
    assert result["auc"] == 1 and result["brier"] == 0 and result["precision"] == 1 and result["recall"] == 1
