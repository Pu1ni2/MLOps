"""Tuning makes one nested run per attempt; the whole pipeline runs twice without promoting the same model again."""
import mlflow
from mlflow import MlflowClient

from wine_quality import pipeline
from wine_quality.tuning import tune_xgboost


def test_tuning_logs_one_nested_run_per_attempt(small_config, splits, tracking):
    run, result, model_info = tune_xgboost(splits, small_config)
    children = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'")
    assert len(children) == small_config["tuning"]["trials"]
    best = children["metrics.validation_auc"].max()
    assert abs(result["auc"] - best) < 1e-9  # the retrained winner scores what its attempt scored
    assert model_info.model_uri.startswith("models:/")


def test_pipeline_twice_keeps_the_champion(small_config, tracking, capsys):
    pipeline.main(small_config)
    pipeline.main(small_config)
    output = capsys.readouterr().out
    assert output.count("1. Data") == 2
    name = small_config["registry"]["model_name"]
    versions = MlflowClient().search_model_versions(f"name = '{name}'")
    assert len(versions) == 3  # the best baseline, then one tuned challenger per run
    third = MlflowClient().get_model_version(name, "3")
    assert third.tags["gate.decision"] == "rejected"  # identical to version 2, so never a real win
    assert "champion" in MlflowClient().get_registered_model(name).aliases
