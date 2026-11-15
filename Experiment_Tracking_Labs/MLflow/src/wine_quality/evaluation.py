"""How a model is judged: scores on held-out wines, plus charts and a feature-importance table, logged to MLflow."""
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_score, recall_score,
                             roc_auc_score)

from wine_quality import plots
from wine_quality.data import FEATURES, TARGET


def scores(truth, chance):
    """
    A model's scores, from the true labels and the model's chance that each wine is high quality:
      auc, average_precision: how well it ranks high-quality wines above the rest (1 is perfect)
      log_loss, brier: how close its chances are to what really happened (lower is better)
      precision, recall: for the wines it calls high quality (a chance of 50% or more)
    """
    called = chance >= 0.5
    return {
        "auc": roc_auc_score(truth, chance),
        "average_precision": average_precision_score(truth, chance),
        "log_loss": log_loss(truth, chance),
        "brier": brier_score_loss(truth, chance),
        "precision": precision_score(truth, called, zero_division=0),
        "recall": recall_score(truth, called, zero_division=0),
    }


def evaluate(model, wines, part, seed, charts=True):
    """
    Scores the model on one part of the data and logs the scores to the current run as <part>_<score>.
    With charts=True, it also logs the ROC, precision-recall, confusion-matrix and feature-importance
    charts, and the importance table, under <part>/. Returns the scores.
    """
    chance = model.predict_proba(wines[FEATURES])[:, 1]
    result = {name: round(float(value), 4) for name, value in scores(wines[TARGET], chance).items()}
    mlflow.log_metrics({f"{part}_{name}": value for name, value in result.items()})
    if charts:
        shuffled = permutation_importance(model, wines[FEATURES], wines[TARGET], scoring="roc_auc", n_repeats=5,
                                          random_state=seed)
        table = pd.DataFrame({"feature": FEATURES, "auc_drop": shuffled.importances_mean.round(4),
                              "spread": shuffled.importances_std.round(4)}).sort_values("auc_drop", ascending=False)
        mlflow.log_table(table, f"{part}/feature_importance.json")
        for name, figure in {
            "roc_curve": plots.roc_chart(wines[TARGET], chance, result["auc"]),
            "precision_recall": plots.precision_recall_chart(wines[TARGET], chance, result["average_precision"]),
            "confusion_matrix": plots.confusion_chart(wines[TARGET], chance),
            "feature_importance": plots.importance_chart(table),
        }.items():
            mlflow.log_figure(figure, f"{part}/{name}.png")
            plt.close(figure)
    return result
