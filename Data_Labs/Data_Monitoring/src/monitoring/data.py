"""Load the UCI Adult census data and describe its columns."""
from pathlib import Path

import pandas as pd

LAB_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = LAB_DIR / "data" / "adult.csv.gz"

# The label: does the person earn more than 50K a year (">50K") or not ("<=50K")?
TARGET = "class"

NUMERICAL = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
FEATURES = NUMERICAL + CATEGORICAL


def load_adult(path=DATA_PATH):
    """
    Loads the census data: one row per person, the 14 FEATURES plus the TARGET.
    Missing values (in workclass, occupation and native-country) stay as NaN.
    """
    df = pd.read_csv(path)
    df[CATEGORICAL + [TARGET]] = df[CATEGORICAL + [TARGET]].astype("string")
    return df
