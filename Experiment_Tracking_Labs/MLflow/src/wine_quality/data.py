"""
The wine data: red and white wines with their quality scores, labelled, without duplicates, and split.

    docker compose run --rm trainer python -m wine_quality.data
prints a summary of the data and the split, and records the split in MLflow as a run.
"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from wine_quality.config import LAB_DIR, load_config
from wine_quality.tracking import run_link, use_experiment

DATA_DIR = LAB_DIR / "data"
FILES = {"red": DATA_DIR / "winequality-red.csv", "white": DATA_DIR / "winequality-white.csv"}

# The label: is the wine high quality (1) or not (0)?
TARGET = "high_quality"
# The 11 lab measurements of each wine, plus its colour (is_red). The quality score itself is not a
# feature: the label is made from it.
FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide",
    "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "is_red",
]
PARTS = ["train", "validation", "test"]


def read_wines(path):
    """Reads one file. The red wines' file separates values with commas, the white wines' with semicolons."""
    with open(path, encoding="utf-8") as file:
        header = file.readline()
    return pd.read_csv(path, sep=";" if header.count(";") > header.count(",") else ",")


def load_wines():
    """All the wines in one table: their measurements, is_red (1 for red, 0 for white) and quality score."""
    tables = [read_wines(path).assign(is_red=int(colour == "red")) for colour, path in FILES.items()]
    return pd.concat(tables, ignore_index=True).rename(columns=lambda name: name.replace(" ", "_"))


def add_label(wines, high_quality_from):
    """Adds the label: 1 if the wine's quality score is at least high_quality_from, else 0."""
    return wines.assign(**{TARGET: (wines["quality"] >= high_quality_from).astype(int)})


def split(wines, shares, seed):
    """Splits the wines into train, validation and test, each with the same share of high-quality wines."""
    train, rest = train_test_split(wines, train_size=shares["train"], stratify=wines[TARGET], random_state=seed)
    validation, test = train_test_split(rest, train_size=shares["validation"] / (shares["validation"] + shares["test"]),
                                        stratify=rest[TARGET], random_state=seed)
    return {part: table.reset_index(drop=True) for part, table in zip(PARTS, [train, validation, test])}


def seen_in_training(train, test):
    """How many test wines also appear, with exactly the same measurements, in the training data."""
    training_wines = set(map(tuple, train[FEATURES].to_numpy()))
    return sum(tuple(wine) in training_wines for wine in test[FEATURES].to_numpy())


def prepare(config):
    """
    Loads and labels the wines, removes the duplicates (if configured), and splits them.
    Returns the splits and a summary of what was done.
    """
    settings = config["data"]
    wines = add_label(load_wines(), settings["high_quality_from"])
    total, duplicates = len(wines), int(wines.duplicated().sum())
    # The split with the duplicates kept, as the original lab did, to measure how many test wines it shares
    with_duplicates = split(wines, settings["split"], settings["seed"])
    if settings["remove_duplicates"]:
        wines = wines.drop_duplicates(ignore_index=True)
    splits = split(wines, settings["split"], settings["seed"])
    summary = {
        "wines": total,
        "duplicate_wines": duplicates,
        "wines_used": len(wines),
        "high_quality_share": round(float(wines[TARGET].mean()), 4),
        **{f"{part}_wines": len(table) for part, table in splits.items()},
        "test_wines_seen_in_training": seen_in_training(splits["train"], splits["test"]),
        "test_wines_if_duplicates_kept": len(with_duplicates["test"]),
        "test_wines_seen_in_training_if_duplicates_kept": seen_in_training(with_duplicates["train"],
                                                                           with_duplicates["test"]),
    }
    return splits, summary


def log_datasets(splits):
    """
    Records the splits as the current run's input datasets. MLflow keeps a fingerprint (digest) of each,
    so every run shows exactly which data it used.
    """
    for part, table in splits.items():
        mlflow.log_input(mlflow.data.from_pandas(table, name=f"wines-{part}", targets=TARGET), context=part)


def main():
    config = load_config()
    splits, summary = prepare(config)
    use_experiment(config["experiment"])
    with mlflow.start_run(run_name="prepare-data") as run:
        mlflow.set_tag("stage", "data")
        settings = {key: value for key, value in config["data"].items() if key != "split"}
        settings.update({f"split.{part}": share for part, share in config["data"]["split"].items()})
        mlflow.log_params({f"data.{key}": value for key, value in settings.items()})
        mlflow.log_metrics(summary)
        log_datasets(splits)

    kept_or_removed = "removed" if config["data"]["remove_duplicates"] else "kept"
    print(f"Wines: {summary['wines']:,}, of which {summary['duplicate_wines']:,} are exact copies of another wine "
          f"({kept_or_removed}), so {summary['wines_used']:,} are used")
    print(f"High quality (score >= {config['data']['high_quality_from']}): {summary['high_quality_share']:.1%} of them")
    print("Split: " + ", ".join(f"{part} {len(table):,} ({table[TARGET].mean():.1%} high quality)"
                                for part, table in splits.items()))
    print(f"Test wines also in the training data: {summary['test_wines_seen_in_training']} of "
          f"{summary['test_wines']:,}. With the duplicates kept, as in the original lab: "
          f"{summary['test_wines_seen_in_training_if_duplicates_kept']} of {summary['test_wines_if_duplicates_kept']:,}")
    print(f"Recorded in MLflow: {run_link(run)}")


if __name__ == "__main__":
    main()
