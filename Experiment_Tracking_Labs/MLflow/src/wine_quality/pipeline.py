"""
The whole pipeline in one command: the data, the baselines, the tuning, then the registry gate.
Every step is tracked in MLflow.

    docker compose run --rm trainer python -m wine_quality.pipeline

The first time, the best baseline becomes the first champion. After that, every run's tuned model
challenges the champion, and replaces it only if it is clearly better on the test wines.
"""
from mlflow import MlflowClient

from wine_quality import data
from wine_quality.baselines import print_result, train_baseline
from wine_quality.config import load_config
from wine_quality.registry import CHAMPION, gate, version_with_alias
from wine_quality.tracking import model_link, use_experiment
from wine_quality.tuning import print_tuned, tune_xgboost


def print_outcome(outcome, label):
    decision = "PROMOTED to @champion" if outcome["promoted"] else "rejected"
    print(f"  version {outcome['version']} ({label}): {decision}")
    print(f"    because {outcome['reason']}")


def main():
    config = load_config()
    splits, summary = data.prepare(config)
    use_experiment(config["experiment"])
    print("1. Data")
    data.print_summary(splits, summary, config, data.record(splits, summary, config))

    print(f"\n2. Baselines, judged on the {len(splits['validation']):,} validation wines")
    baselines = []
    for family, params in config["baselines"].items():
        run, result, model_info = train_baseline(family, params, splits, config["seed"])
        print_result(family, result, run)
        baselines.append({"auc": result["auc"], "family": family, "run": run, "model": model_info})

    print(f"\n3. Tuning XGBoost: {config['tuning']['trials']} attempts, judged on the validation wines")
    tuned_run, tuned, tuned_model = tune_xgboost(splits, config)
    print_tuned(tuned_run, tuned)

    print(f"\n4. Registry gate, judged on the {len(splits['test']):,} test wines")
    name = config["registry"]["model_name"]
    if version_with_alias(MlflowClient(), name, CHAMPION) is None:
        best = max(baselines, key=lambda baseline: baseline["auc"])
        candidate = {"run_id": best["run"].info.run_id, "model_family": best["family"], "validation_auc": best["auc"]}
        print_outcome(gate(best["model"].model_uri, candidate, splits, config), f"{best['family']}, the best baseline")
    candidate = {"run_id": tuned_run.info.run_id, "model_family": "xgboost", "validation_auc": tuned["auc"]}
    print_outcome(gate(tuned_model.model_uri, candidate, splits, config), "tuned xgboost")
    print(f"  Registry: {model_link(name)}")


if __name__ == "__main__":
    main()
