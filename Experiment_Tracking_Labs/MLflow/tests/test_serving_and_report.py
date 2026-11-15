"""The serving request format, batch lineage, and the original lab's score reproduced exactly."""
import sys

import numpy as np
import pandas as pd

from wine_quality import batch, registry, serving
from wine_quality.data import FEATURES
from wine_quality.report import auc_by_group, original_split
from tests.test_registry import log


class FakeResponse:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return {"predictions": [[0.25, 0.75]] * self.rows}


def test_the_server_gets_the_features_and_its_answer_is_read(splits, monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, payload=json)
        return FakeResponse(len(json["dataframe_split"]["data"]))

    monkeypatch.setattr(serving.requests, "post", fake_post)
    answer = serving.served_chances("http://serve:5001", splits["test"].head(4))
    assert sent["url"] == "http://serve:5001/invocations"
    assert sent["payload"]["dataframe_split"]["columns"] == FEATURES
    assert np.allclose(answer, 0.75)


def test_batch_scoring_records_the_model_version(config, splits, tracking, tmp_path, monkeypatch):
    registry.gate(*log("random_forest", splits), splits, config)  # version 1 becomes the champion
    monkeypatch.setattr(batch, "OUTPUT_PATH", tmp_path / "predictions.csv")
    monkeypatch.setattr(sys, "argv", ["batch"])
    batch.main()
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert len(predictions) == len(splits["test"])
    assert set(predictions["model_version"]) == {1} and set(predictions["model_name"]) == {"wine_quality"}
    assert predictions["chance_high_quality"].between(0, 1).all()


def test_the_original_labs_score_is_reproduced_and_explained():
    from sklearn.ensemble import RandomForestClassifier

    train, test = original_split(seed=123)
    model = RandomForestClassifier(n_estimators=10, random_state=np.random.RandomState(123))
    result = auc_by_group(model.fit(train[FEATURES], train["high_quality"]), train, test)
    assert result["auc_all"] == 0.8540300975814177  # exactly what the original notebook printed
    assert result["seen_wines"] == 269 and result["auc_seen"] == 1.0
    assert round(result["auc_new"], 3) == 0.785
