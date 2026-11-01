"""Load the UCI Adult census data, describe its columns, and split it for the simulation."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = LAB_DIR / "data" / "adult.csv.gz"

# The label: does the person earn more than 50K a year (">50K") or not ("<=50K")?
TARGET = "class"
POSITIVE_LABEL = ">50K"

NUMERICAL = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
FEATURES = NUMERICAL + CATEGORICAL

# How the 48,842 people are divided (the rest, about 28,800, form the production pool)
TRAIN_SIZE = 15_000      # the model learns from these
REFERENCE_SIZE = 5_000   # held out: the model's baseline performance, and what drift is compared against
SEED = 42


def load_adult(path=DATA_PATH):
    """
    Loads the census data: one row per person, the 14 FEATURES plus the TARGET.
    Missing values (in workclass, occupation and native-country) stay as NaN.
    """
    return pd.read_csv(path)


def split_data(df, train_size=TRAIN_SIZE, reference_size=REFERENCE_SIZE, seed=SEED):
    """
    Splits the data into three parts, keeping the same share of high earners in each:
      train:     what the model is trained on
      reference: held-out data the model never trained on; it gives the model's baseline
                 performance and is the "normal" data that daily batches are compared against
      pool:      everything else; the simulated production days are drawn from it
    Returns:
        tuple of DataFrames: train, reference, pool
    """
    train, rest = train_test_split(df, train_size=train_size, stratify=df[TARGET], random_state=seed)
    reference, pool = train_test_split(rest, train_size=reference_size, stratify=rest[TARGET], random_state=seed)
    return train.reset_index(drop=True), reference.reset_index(drop=True), pool.reset_index(drop=True)
