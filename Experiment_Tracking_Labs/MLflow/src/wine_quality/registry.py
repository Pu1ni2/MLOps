"""
The model registry: one registered model (wine_quality), whose versions carry aliases:
  @champion    the version in use: what is served and what batch scoring uses
  @challenger  the latest candidate, judged against the champion on the test wines
  @previous    the champion before the current one, kept for a rollback

The gate promotes a challenger only if it beats the champion on the test wines, and not by luck (see
config/pipeline.yaml). Each decision is its own MLflow run, and is also written on the model version.

    docker compose run --rm trainer python -m wine_quality.registry status     lists the versions
    docker compose run --rm trainer python -m wine_quality.registry rollback   makes @previous the champion again
"""
import sys

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

from wine_quality.config import load_config
from wine_quality.data import FEATURES, TARGET, log_datasets
from wine_quality.evaluation import scores
from wine_quality.tracking import model_link, quiet_mlflow

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"


def version_with_alias(client, name, alias):
    """The model version that has this alias, or None (also before the model is registered at all)."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def chances(model_uri, wines):
    """A registered model's chance that each wine is high quality: the second column of its predict_proba."""
    return np.asarray(mlflow.pyfunc.load_model(model_uri).predict(wines[FEATURES]))[:, 1]


def win_share(truth, challenger, champion, resamples, seed):
    """
    The share of random re-draws of the wines (with replacement, the same re-draw for both models)
    in which the challenger's AUC is higher than the champion's.
    """
    truth, rng, wins, valid = np.asarray(truth), np.random.default_rng(seed), 0, 0
    for _ in range(resamples):
        rows = rng.integers(0, len(truth), len(truth))
        if truth[rows].min() == truth[rows].max():  # AUC needs both kinds of wine in the re-draw
            continue
        valid += 1
        wins += roc_auc_score(truth[rows], challenger[rows]) > roc_auc_score(truth[rows], champion[rows])
    return wins / valid


def decide(challenger, champion, share, settings):
    """
    Whether the challenger replaces the champion, and why. challenger and champion are their test scores
    (champion is None when there is no champion yet), and share is the challenger's win share.
    """
    if champion is None:
        return True, "first model: there was no champion yet"
    if share < settings["min_win_share"]:
        return False, (f"its test AUC ({challenger['auc']:.3f}, champion {champion['auc']:.3f}) was higher in only "
                       f"{share:.0%} of the re-drawn test sets, below the {settings['min_win_share']:.0%} required")
    if challenger["brier"] > champion["brier"] + settings["max_brier_increase"]:
        return False, (f"its Brier score ({challenger['brier']:.3f}) is worse than the champion's "
                       f"({champion['brier']:.3f})")
    return True, (f"its test AUC ({challenger['auc']:.3f}, champion {champion['auc']:.3f}) was higher in {share:.0%} "
                  f"of the re-drawn test sets, and its Brier score is no worse "
                  f"({challenger['brier']:.3f}, champion {champion['brier']:.3f})")


def gate(model_uri, candidate, splits, config):
    """
    Registers a logged model as a new version, then judges it against the champion on the test wines,
    inside a "registry-gate" run. candidate describes it: its run_id, model_family and validation_auc.
    Returns the outcome: the version, whether it was promoted, the reason and the test scores.
    """
    settings, test, client = config["registry"], splits["test"], MlflowClient()
    name = settings["model_name"]
    with mlflow.start_run(run_name="registry-gate") as run:
        mlflow.set_tags({"stage": "gate", "candidate_run": candidate["run_id"], "model_family": candidate["model_family"]})
        mlflow.log_params({key: value for key, value in settings.items()})
        log_datasets({"test": test})
        champion = version_with_alias(client, name, CHAMPION)
        with quiet_mlflow():  # registering waits until the version is READY, without sleeping
            version = mlflow.register_model(model_uri, name, tags={key: str(value) for key, value in candidate.items()}).version

        challenger_chances = chances(f"models:/{name}/{version}", test)
        challenger = {key: round(float(value), 4) for key, value in scores(test[TARGET], challenger_chances).items()}
        mlflow.log_metrics({f"challenger_test_{key}": value for key, value in challenger.items()})
        if champion is None:
            promoted, reason = decide(challenger, None, None, settings)
        else:
            client.set_registered_model_alias(name, CHALLENGER, version)
            champion_chances = chances(f"models:/{name}@{CHAMPION}", test)
            current = {key: round(float(value), 4) for key, value in scores(test[TARGET], champion_chances).items()}
            share = win_share(test[TARGET], challenger_chances, champion_chances, settings["resamples"], config["seed"])
            mlflow.log_metrics({**{f"champion_test_{key}": value for key, value in current.items()},
                                "challenger_win_share": share})
            promoted, reason = decide(challenger, current, share, settings)

        if promoted:
            if champion is not None:
                client.set_registered_model_alias(name, PREVIOUS, champion.version)
                client.delete_registered_model_alias(name, CHALLENGER)
            client.set_registered_model_alias(name, CHAMPION, version)
        decision = "promoted" if promoted else "rejected"
        for key, value in {"gate.decision": decision, "gate.reason": reason, "gate.run": run.info.run_id,
                           "gate.test_auc": challenger["auc"], "gate.test_brier": challenger["brier"],
                           "gate.champion_version": champion.version if champion else "none"}.items():
            client.set_model_version_tag(name, version, key, str(value))
        client.update_model_version(name, version, description=f"{decision.capitalize()}: {reason}.")
        mlflow.set_tags({"decision": decision, "version": version})
    return {"version": version, "promoted": promoted, "reason": reason, "test": challenger,
            "champion_version": champion.version if champion else None}


def status(config):
    """Prints every version of the model with its aliases and its gate decision."""
    name, client = config["registry"]["model_name"], MlflowClient()
    versions = sorted(client.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version))
    if not versions:
        print(f"No versions of {name} yet: run the pipeline first.")
        return
    # The registered model lists its aliases; search_model_versions leaves each version's aliases empty.
    # Version numbers are strings from the server but integers from a local store, so compare them as strings.
    aliases = {}
    for alias, number in client.get_registered_model(name).aliases.items():
        aliases.setdefault(str(number), []).append(f"@{alias}")
    print(f"{name}: {len(versions)} versions")
    for version in versions:
        aliases_shown = ", ".join(sorted(aliases.get(str(version.version), []))) or "-"
        print(f"  version {version.version:>2}  {aliases_shown:<22} {version.tags.get('model_family', '?'):<20} "
              f"test AUC {version.tags.get('gate.test_auc', '?'):<7} {version.tags.get('gate.decision', '?')}")
    print(model_link(name))


def rollback(config):
    """Makes the @previous version the champion again; the current champion becomes @previous."""
    name, client = config["registry"]["model_name"], MlflowClient()
    previous, champion = version_with_alias(client, name, PREVIOUS), version_with_alias(client, name, CHAMPION)
    if previous is None or champion is None:
        sys.exit("Nothing to roll back to: no version has the @previous alias.")
    client.set_registered_model_alias(name, CHAMPION, previous.version)
    client.set_registered_model_alias(name, PREVIOUS, champion.version)
    client.set_model_version_tag(name, champion.version, "gate.rolled_back", "true")
    print(f"Rolled back: version {previous.version} is the champion again (version {champion.version} is @previous).")


def main():
    commands = {"status": status, "rollback": rollback}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(f"Usage: python -m wine_quality.registry [{' | '.join(commands)}]")
    commands[sys.argv[1]](load_config())


if __name__ == "__main__":
    main()
