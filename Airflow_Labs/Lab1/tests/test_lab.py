"""Unit tests for dags/src/lab.py: the pipeline logic, tested without Airflow."""
import json
import os

import joblib
import pandas as pd
import pytest

from src import lab


def test_load_data_copies_file_into_run_folder(tmp_path, real_csv):
    path = lab.load_data(real_csv, str(tmp_path))
    assert path == os.path.join(str(tmp_path), "raw.csv")
    with open(path, "rb") as copy, open(real_csv, "rb") as original:
        assert copy.read() == original.read()


def test_validate_data_on_real_data(tmp_path, real_csv):
    report = lab.validate_data(real_csv, str(tmp_path))
    assert report["rows_in"] == 8950
    assert report["rows_out"] == 8949  # one customer has no CREDIT_LIMIT
    assert report["missing"] == {"BALANCE": 0, "PURCHASES": 0, "CREDIT_LIMIT": 1}
    assert list(pd.read_csv(report["clean_path"]).columns) == lab.FEATURES


@pytest.mark.parametrize(
    "break_data, message",
    [
        (lambda df: df.drop(columns=["PURCHASES"]), "Missing columns"),
        (lambda df: df.assign(BALANCE="lots"), "must be numeric"),
        (lambda df: df.head(50), "complete rows left"),
    ],
    ids=["missing column", "text column", "too few rows"],
)
def test_validate_data_rejects_bad_data(tmp_path, real_csv, break_data, message):
    bad_csv = tmp_path / "bad.csv"
    break_data(pd.read_csv(real_csv)).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match=message):
        lab.validate_data(str(bad_csv), str(tmp_path))


def test_validate_data_uses_chosen_features(tmp_path, real_csv):
    report = lab.validate_data(real_csv, str(tmp_path), features=["BALANCE", "PURCHASES"])
    assert report["rows_out"] == 8950  # the missing CREDIT_LIMIT no longer matters
    assert list(report["missing"]) == ["BALANCE", "PURCHASES"]


def test_preprocess_scales_every_column_to_0_1(tmp_path, real_csv):
    clean_csv = lab.validate_data(real_csv, str(tmp_path))["clean_path"]
    prepared = lab.preprocess(clean_csv, str(tmp_path))
    scaled = pd.read_csv(prepared["data_path"])
    assert scaled.min().tolist() == pytest.approx([0, 0, 0])
    assert scaled.max().tolist() == pytest.approx([1, 1, 1])
    assert os.path.exists(prepared["scaler_path"])


def test_train_kmeans_scores_and_saves_one_model(tmp_path, blob_csv):
    result = lab.train_kmeans(blob_csv, str(tmp_path), 3)
    assert result["k"] == 3
    assert result["sse"] > 0
    assert result["silhouette"] > 0.7  # 3 clean groups are clearly separated
    assert joblib.load(result["model_path"]).n_clusters == 3


def test_select_best_k_finds_the_elbow(tmp_path, blob_csv):
    # Parallel tasks can finish in any order, so train k = 2..8 in shuffled order
    results = [lab.train_kmeans(blob_csv, str(tmp_path), k) for k in [5, 2, 8, 3, 7, 4, 6]]
    best = lab.select_best_k(results, str(tmp_path))
    assert best["best_k"] == 3  # the made-up data has exactly 3 groups
    assert best["method"] == "elbow"
    scores = pd.read_csv(best["scores_path"])
    assert scores["k"].tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert list(scores.columns) == ["k", "sse", "silhouette", "model_path"]


def test_select_best_k_falls_back_to_silhouette_without_elbow(tmp_path):
    # With only 2 values of k there is no bend to find
    results = [
        {"k": 2, "sse": 100.0, "silhouette": 0.6, "model_path": "kmeans_k2.joblib"},
        {"k": 3, "sse": 80.0, "silhouette": 0.4, "model_path": "kmeans_k3.joblib"},
    ]
    best = lab.select_best_k(results, str(tmp_path))
    assert best["method"] == "silhouette"
    assert best["best_k"] == 2


@pytest.mark.parametrize("silhouette, passes", [(0.45, True), (0.40, True), (0.39, False)])
def test_quality_gate_rule(silhouette, passes):
    assert lab.passes_quality_gate(silhouette, min_silhouette=0.40) is passes


def test_promote_model_copies_model_and_records_metrics(tmp_path, blob_csv):
    result = lab.train_kmeans(blob_csv, str(tmp_path), 3)
    best = {"best_k": 3, "method": "elbow", "silhouette": result["silhouette"], "sse": result["sse"],
            "model_path": result["model_path"]}
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump("any object", scaler_path)  # promote_model only copies the file

    promoted = lab.promote_model(best, str(scaler_path), str(tmp_path / "latest"))
    assert os.path.exists(promoted["model_path"])
    assert os.path.exists(promoted["scaler_path"])
    with open(promoted["metrics_path"], encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["best_k"] == 3
    assert metrics["features"] == lab.FEATURES


def test_predict_and_profile_on_real_data(tmp_path, real_csv, test_csv):
    clean_csv = lab.validate_data(real_csv, str(tmp_path))["clean_path"]
    prepared = lab.preprocess(clean_csv, str(tmp_path))
    model_path = lab.train_kmeans(prepared["data_path"], str(tmp_path), 5)["model_path"]

    clusters = lab.predict(model_path, prepared["scaler_path"], test_csv)
    assert len(clusters) == len(pd.read_csv(test_csv))
    assert all(isinstance(c, int) and 0 <= c < 5 for c in clusters)

    profile = pd.read_csv(lab.profile_clusters(model_path, prepared["scaler_path"], clean_csv, str(tmp_path)))
    assert len(profile) == 5
    assert profile["customers"].sum() == 8949  # every clean customer is in exactly one cluster
    assert list(profile.columns) == ["cluster", "customers", *lab.FEATURES]


def test_low_quality_report_explains_the_decision(tmp_path):
    path = lab.write_low_quality_report({"best_k": 5, "silhouette": 0.30}, 0.40, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["min_silhouette"] == 0.40
    assert "below the minimum" in report["reason"]
