"""
The report: tables and charts in results/, built from what MLflow tracked (the runs and the model registry),
plus a study that recreates the original lab's setup to measure how much its duplicate wines inflated its score.

    docker compose run --rm trainer python -m wine_quality.report

Run it after the pipeline. The study is logged to MLflow too, in its own experiment, "leakage-study".
"""
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from wine_quality import plots
from wine_quality.config import LAB_DIR, load_config
from wine_quality.data import FEATURES, TARGET, add_label, load_wines
from wine_quality.tracking import use_experiment

RESULTS_DIR = LAB_DIR / "results"


def latest_runs(experiment_id, stage):
    """The runs of one stage (e.g. baseline), newest first, as MLflow's search_runs table."""
    return mlflow.search_runs([experiment_id], filter_string=f"tags.stage = '{stage}'",
                              order_by=["attributes.start_time DESC"])


def registry_table(config):
    """Every version of the model: its family, aliases, test scores and the gate's decision and reason."""
    name, client = config["registry"]["model_name"], MlflowClient()
    aliases = {}
    for alias, number in client.get_registered_model(name).aliases.items():
        aliases.setdefault(str(number), []).append(f"@{alias}")
    versions = sorted(client.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version))
    return pd.DataFrame([{
        "version": int(v.version), "aliases": " ".join(sorted(aliases.get(str(v.version), []))),
        "model_family": v.tags.get("model_family"), "validation_auc": float(v.tags.get("validation_auc", "nan")),
        "test_auc": float(v.tags.get("gate.test_auc", "nan")), "test_brier": float(v.tags.get("gate.test_brier", "nan")),
        "decision": v.tags.get("gate.decision"), "reason": v.tags.get("gate.reason"),
    } for v in versions])


def models_table(experiment_id):
    """The latest baseline and tuned runs, with their validation scores."""
    columns = ["auc", "average_precision", "log_loss", "brier"]
    rows = []
    for stage in ("baseline", "tuning"):
        runs = latest_runs(experiment_id, stage).drop_duplicates("tags.model_family")
        for _, run in runs.iterrows():
            rows.append({"model": run["tags.model_family"] + (" (tuned)" if stage == "tuning" else ""),
                         **{f"validation_{c}": round(run[f"metrics.validation_{c}"], 4) for c in columns}})
    return pd.DataFrame(rows).sort_values("validation_auc", ascending=False)


def tuning_attempts(experiment_id):
    """The attempts (nested runs) of the latest tuning run, in order, with their validation AUC."""
    parent = latest_runs(experiment_id, "tuning").iloc[0]
    trials = mlflow.search_runs([experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent['run_id']}'",
                                order_by=["attributes.start_time ASC"], max_results=1000)
    return pd.DataFrame({"attempt": np.arange(1, len(trials) + 1), "run_name": trials["tags.mlflow.runName"],
                         "validation_auc": trials["metrics.validation_auc"], "trees": trials["params.trees"]})


def original_split(seed):
    """The original lab's data and split: duplicates kept, 60/20/20, not stratified, with its random_state."""
    wines = add_label(load_wines(), 7)
    train, rest = train_test_split(wines, train_size=0.6, random_state=seed)
    _, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, test


def auc_by_group(model, train, test):
    """Test AUC on all test wines, on the ones also in the training data, and on the new ones."""
    chance = model.predict_proba(test[FEATURES])[:, 1]
    training_wines = set(map(tuple, train[FEATURES].to_numpy()))
    seen = np.array([tuple(wine) in training_wines for wine in test[FEATURES].to_numpy()])
    truth = test[TARGET].to_numpy()
    return {"auc_all": roc_auc_score(truth, chance), "auc_seen": roc_auc_score(truth[seen], chance[seen]),
            "auc_new": roc_auc_score(truth[~seen], chance[~seen]), "test_wines": len(test),
            "seen_wines": int(seen.sum()), "new_wines": int((~seen).sum())}


def leakage_study(config, champion_test_auc, test_wines):
    """
    The original lab's setup, and the same split with this lab's random forest, each scored on the test
    wines it had already seen and on the new ones; both are logged as runs. This lab's champion is the last
    row: its test set (test_wines wines) has no duplicates, so all its test wines are new.
    """
    train, test = original_split(seed=123)
    setups = [
        ("Original lab: 10 trees, duplicates kept",
         RandomForestClassifier(n_estimators=10, random_state=np.random.RandomState(123))),
        ("300 trees, duplicates kept",
         RandomForestClassifier(**config["baselines"]["random_forest"], random_state=config["seed"], n_jobs=-1)),
    ]
    use_experiment("leakage-study")
    rows = []
    for setup, model in setups:
        with mlflow.start_run(run_name=setup):
            mlflow.set_tag("stage", "leakage-study")
            mlflow.log_params({"trees": model.n_estimators, "duplicates": "kept", "split_random_state": 123})
            result = auc_by_group(model.fit(train[FEATURES], train[TARGET]), train, test)
            mlflow.log_metrics(result)
        rows.append({"setup": setup, **{key: round(value, 4) for key, value in result.items()}})
    rows.append({"setup": "This lab: 300 trees, duplicates removed", "auc_all": champion_test_auc,
                 "auc_seen": None, "auc_new": champion_test_auc, "test_wines": test_wines, "seen_wines": 0,
                 "new_wines": test_wines})
    return pd.DataFrame(rows)


def save_chart(figure, name):
    figure.savefig(RESULTS_DIR / name, dpi=150)
    plt.close(figure)


def main():
    config = load_config()
    experiment = mlflow.get_experiment_by_name(config["experiment"])
    if experiment is None:
        raise SystemExit("No runs to report on yet: run the pipeline first.")
    RESULTS_DIR.mkdir(exist_ok=True)

    models = models_table(experiment.experiment_id)
    registry = registry_table(config)
    attempts = tuning_attempts(experiment.experiment_id)
    champion = registry[registry["aliases"].str.contains("@champion")].iloc[0]
    test_wines = int(latest_runs(experiment.experiment_id, "data").iloc[0]["metrics.test_wines"])
    leakage = leakage_study(config, float(champion["test_auc"]), test_wines)
    for name, table in (("models", models), ("registry", registry), ("tuning_attempts", attempts), ("leakage", leakage)):
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    best_baseline = models[~models["model"].str.endswith("(tuned)")]["validation_auc"].max()
    save_chart(plots.tuning_chart(attempts["validation_auc"].to_numpy(), best_baseline), "tuning_progress.png")
    save_chart(plots.leakage_chart(leakage.to_dict("records")), "leakage.png")

    print("Models (validation):")
    print(models.to_string(index=False))
    print("\nRegistry:")
    print(registry.drop(columns="reason").to_string(index=False))
    print("\nLeakage study (test AUC):")
    print(leakage.to_string(index=False))
    print(f"\nWritten to results/: {', '.join(sorted(path.name for path in RESULTS_DIR.iterdir()))}")


if __name__ == "__main__":
    main()
