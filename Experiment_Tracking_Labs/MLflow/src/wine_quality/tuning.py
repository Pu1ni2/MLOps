"""
Tuning XGBoost with Optuna. A parent run, "tune-xgboost", holds the search and the winning model; each
attempt (trial) is a child run nested under it, with its own settings and validation scores.

    docker compose run --rm trainer python -m wine_quality.tuning
"""
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd

from wine_quality import plots
from wine_quality.config import load_config
from wine_quality.data import FEATURES, TARGET, log_datasets, prepare
from wine_quality.evaluation import evaluate
from wine_quality.models import build_model, log_model
from wine_quality.tracking import run_link, use_experiment


def suggest_settings(trial):
    """The settings Optuna searches. Trees stay at most 10 levels deep: deeper ones memorize the wines."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
    }


def fit_with_early_stopping(settings, splits, config):
    """Trains XGBoost, adding trees until the validation AUC stops improving. Returns the model and its tree count."""
    tuning, train, validation = config["tuning"], splits["train"], splits["validation"]
    model = build_model("xgboost", {**settings, "n_estimators": tuning["max_trees"], "eval_metric": "auc",
                                    "early_stopping_rounds": tuning["early_stopping_rounds"]}, config["seed"])
    model.fit(train[FEATURES], train[TARGET], eval_set=[(validation[FEATURES], validation[TARGET])], verbose=False)
    return model, model.best_iteration + 1


def search(splits, config):
    """Runs Optuna's search inside the current run, with each attempt as a nested run. Returns the study."""
    def attempt(trial):
        settings = suggest_settings(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number + 1:02d}", nested=True):
            mlflow.set_tags({"stage": "tuning-trial", "model_family": "xgboost"})
            model, trees = fit_with_early_stopping(settings, splits, config)
            mlflow.log_params({**settings, "trees": trees})
            result = evaluate(model, splits["validation"], "validation", config["seed"], charts=False)
        trial.set_user_attr("trees", trees)
        return result["auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Seeded and one attempt at a time, so the search tries the same settings every time
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"]))
    study.optimize(attempt, n_trials=config["tuning"]["trials"])
    return study


def best_baseline_auc(experiment_id):
    """The best validation AUC among the baseline runs already logged in this experiment, if any."""
    runs = mlflow.search_runs([experiment_id], filter_string="tags.stage = 'baseline'",
                              order_by=["metrics.validation_auc DESC"], max_results=1)
    return None if runs.empty else float(runs.iloc[0]["metrics.validation_auc"])


def tune_xgboost(splits, config):
    """
    The whole search in a parent run. The winning settings are then retrained, with the number of trees
    early stopping found, and logged on the parent run with their scores, charts and model.
    Returns the parent run and the winner's validation scores.
    """
    seed, train = config["seed"], splits["train"]
    with mlflow.start_run(run_name="tune-xgboost") as run:
        mlflow.set_tags({"stage": "tuning", "model_family": "xgboost"})
        log_datasets({part: splits[part] for part in ("train", "validation")})
        study = search(splits, config)
        settings = {**study.best_params, "n_estimators": study.best_trial.user_attrs["trees"]}
        mlflow.log_params({"seed": seed, "trials": config["tuning"]["trials"],
                           "best_trial": study.best_trial.number + 1, **settings})
        model = build_model("xgboost", settings, seed).fit(train[FEATURES], train[TARGET])
        result = evaluate(model, splits["validation"], "validation", seed)
        attempts = pd.DataFrame([{"attempt": trial.number + 1, "validation_auc": round(trial.value, 4),
                                  "trees": trial.user_attrs["trees"], **trial.params} for trial in study.trials])
        mlflow.log_table(attempts, "tuning/attempts.json")
        figure = plots.tuning_chart(attempts["validation_auc"].to_numpy(), best_baseline_auc(run.info.experiment_id))
        mlflow.log_figure(figure, "tuning/progress.png")
        plt.close(figure)
        log_model(model, "xgboost", train[FEATURES])
    return run, result


def main():
    config = load_config()
    splits, _ = prepare(config)
    use_experiment(config["experiment"])
    print(f"Tuning XGBoost: {config['tuning']['trials']} attempts, each a nested run of 'tune-xgboost'...")
    run, result = tune_xgboost(splits, config)
    params = mlflow.get_run(run.info.run_id).data.params
    print(f"  best attempt {params['best_trial']}: validation AUC {result['auc']:.3f}   average precision "
          f"{result['average_precision']:.3f}   log loss {result['log_loss']:.3f}   Brier {result['brier']:.3f}")
    shown = ("max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree")
    print("  settings: " + ", ".join(f"{key} {float(params[key]):.3g}" for key in shown))
    print(f"  {run_link(run)}")


if __name__ == "__main__":
    main()
