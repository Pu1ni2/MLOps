"""
The baseline models, the simple models a tuned model has to beat. Each is trained on the training wines,
judged on the validation wines (the test wines stay unseen until the registry gate), and logged as a run.

    docker compose run --rm trainer python -m wine_quality.baselines
"""
import mlflow

from wine_quality.config import load_config
from wine_quality.data import FEATURES, TARGET, log_datasets, prepare
from wine_quality.evaluation import evaluate
from wine_quality.models import build_model, log_model
from wine_quality.tracking import run_link, use_experiment


def train_baseline(family, params, splits, seed):
    """
    Trains one baseline model in its own run: settings, data, validation scores, charts and the model.
    Returns the run, its validation scores and the logged model's ModelInfo.
    """
    with mlflow.start_run(run_name=f"baseline-{family}") as run:
        mlflow.set_tags({"stage": "baseline", "model_family": family})
        mlflow.log_params({"seed": seed, **params})
        log_datasets({part: splits[part] for part in ("train", "validation")})
        train = splits["train"]
        model = build_model(family, params, seed).fit(train[FEATURES], train[TARGET])
        result = evaluate(model, splits["validation"], "validation", seed)
        model_info = log_model(model, family, train[FEATURES])
    return run, result, model_info


def print_result(name, result, run):
    print(f"  {name:<20} AUC {result['auc']:.3f}   average precision {result['average_precision']:.3f}   "
          f"log loss {result['log_loss']:.3f}   Brier {result['brier']:.3f}")
    print(f"  {'':<20} {run_link(run)}")


def main():
    config = load_config()
    splits, _ = prepare(config)
    use_experiment(config["experiment"])
    print(f"Baselines, judged on the {len(splits['validation']):,} validation wines:")
    for family, params in config["baselines"].items():
        run, result, _ = train_baseline(family, params, splits, config["seed"])
        print_result(family, result, run)


if __name__ == "__main__":
    main()
