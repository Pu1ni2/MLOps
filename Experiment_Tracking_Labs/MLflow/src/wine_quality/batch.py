"""
Batch scoring: the champion scores a table of wines, and every prediction records which model version made it.

    docker compose run --rm trainer python -m wine_quality.batch [wines.csv]

The CSV needs the 12 feature columns (see data.FEATURES). Without one, the test wines (without their labels)
stand in for a batch of new wines. Writes outputs/predictions.csv and logs the job as an MLflow run.
"""
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

from wine_quality.config import LAB_DIR, load_config
from wine_quality.data import FEATURES, prepare
from wine_quality.registry import CHAMPION, chances, version_with_alias
from wine_quality.tracking import run_link, use_experiment

OUTPUT_PATH = LAB_DIR / "outputs" / "predictions.csv"


def load_batch(config, path=None):
    """The wines to score: the given CSV, or the test wines without their labels."""
    if path is None:
        return prepare(config)[0]["test"][FEATURES]
    wines = pd.read_csv(path)
    missing = [column for column in FEATURES if column not in wines.columns]
    if missing:
        sys.exit(f"{path} is missing the columns {missing}")
    return wines[FEATURES]


def score(wines, name, version):
    """The wines with the model's chance of high quality, its call (at 50%), and the model that made them."""
    chance = chances(f"models:/{name}/{version}", wines)
    return wines.assign(chance_high_quality=chance.round(4), predicted_high_quality=(chance >= 0.5).astype(int),
                        model_name=name, model_version=int(version))


def main():
    config = load_config()
    name = config["registry"]["model_name"]
    # Resolve @champion once, so the whole batch uses one version even if the alias moves meanwhile
    champion = version_with_alias(MlflowClient(), name, CHAMPION)
    if champion is None:
        sys.exit(f"{name} has no @champion yet: run the pipeline first.")
    wines = load_batch(config, sys.argv[1] if len(sys.argv) > 1 else None)
    predictions = score(wines, name, champion.version)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)

    use_experiment(config["experiment"])
    with mlflow.start_run(run_name="batch-scoring") as run:
        mlflow.set_tags({"stage": "batch", "model_version": champion.version})
        mlflow.log_input(mlflow.data.from_pandas(wines, name="wines-batch"), context="batch")
        mlflow.log_metrics({"wines_scored": len(predictions),
                            "share_called_high_quality": round(float(predictions["predicted_high_quality"].mean()), 4)})
        mlflow.log_artifact(OUTPUT_PATH)
    print(f"Scored {len(predictions):,} wines with {name} version {champion.version} (the champion): "
          f"{predictions['predicted_high_quality'].mean():.1%} called high quality")
    print(f"Predictions: outputs/{OUTPUT_PATH.name}")
    print(f"Recorded in MLflow: {run_link(run)}")


if __name__ == "__main__":
    main()
