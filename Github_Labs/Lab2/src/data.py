"""Load the spam dataset and split it into training and test sets."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spambase.csv"
TARGET = "is_spam"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """
    Loads the spam dataset.
    Args:
        path (str/Path): CSV file with one row per email and an is_spam column.
    Returns:
        tuple: Features X (DataFrame, 57 columns) and labels y (Series, 1 = spam).
    """
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """
    Splits the data into a training set and a test set the model never sees during training.
    Stratifying keeps the same share of spam in both sets, and the fixed random_state makes
    the split identical on every run, so training and evaluation always agree on the test set.
    Args:
        X (DataFrame): Features.
        y (Series): Labels.
        test_size (float): Share of rows held back for testing.
        random_state (int): Seed that makes the split reproducible.
    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
