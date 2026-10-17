"""
Customer segmentation logic: plain Python functions, no Airflow.

Every function reads its input from a file, writes its output to a file, and
returns the path(s). The Airflow tasks only pass these short paths to each
other, never the data itself. Keeping Airflow out of this file also means it
can be tested on its own.
"""
import os
import shutil

import joblib
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# The 3 customer columns used for clustering
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]

# K-Means settings; random_state makes every run give the same result
KMEANS_SETTINGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}


def load_data(source_csv, out_dir):
    """
    Copies the raw customer CSV into this run's folder, so every run keeps
    the exact data it used.
    Args:
        source_csv (str): Path of the original CSV (dags/data/file.csv).
        out_dir (str): This run's folder.
    Returns:
        str: Path of the copy: <out_dir>/raw.csv
    """
    raw_path = os.path.join(out_dir, "raw.csv")
    shutil.copy(source_csv, raw_path)
    return raw_path


def preprocess(raw_csv, out_dir):
    """
    Keeps the FEATURES columns, drops rows with missing values, and scales
    every column to the range 0-1.
    Saves the scaled data AND the fitted scaler, because predict() needs the
    same scaler later.
    Args:
        raw_csv (str): Path returned by load_data.
        out_dir (str): This run's folder.
    Returns:
        dict: {"data_path": <out_dir>/scaled.csv, "scaler_path": <out_dir>/scaler.joblib}
    """
    df = pd.read_csv(raw_csv)[FEATURES].dropna()

    # fit = learn each column's min and max; transform = squeeze the column into 0-1
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    data_path = os.path.join(out_dir, "scaled.csv")
    pd.DataFrame(scaled, columns=FEATURES).to_csv(data_path, index=False)

    scaler_path = os.path.join(out_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    return {"data_path": data_path, "scaler_path": scaler_path}


def build_model(data_path, out_dir, max_k=49):
    """
    Tries k = 1..max_k clusters, picks the best k with the elbow method, then
    trains and saves the model for THAT k.
    Args:
        data_path (str): Scaled data CSV from preprocess.
        out_dir (str): This run's folder.
        max_k (int): Largest number of clusters to try.
    Returns:
        dict: {"best_k": int, "model_path": <out_dir>/model.joblib}
    """
    data = pd.read_csv(data_path)
    ks = range(1, max_k + 1)

    # SSE (inertia_) = how far customers are from their cluster centre; it always drops as k grows
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, **KMEANS_SETTINGS)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # The elbow: the k after which adding more clusters stops reducing SSE by much
    best_k = KneeLocator(ks, sse, curve="convex", direction="decreasing").elbow

    model = KMeans(n_clusters=best_k, **KMEANS_SETTINGS).fit(data)
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(model, model_path)

    # int(): Airflow can only pass plain Python values between tasks, not numpy numbers
    return {"best_k": int(best_k), "model_path": model_path}


def predict(model_path, scaler_path, test_csv):
    """
    Assigns each customer in test_csv to a cluster.
    The test data is scaled with the SAME scaler that was fitted on the
    training data, so it's on the same 0-1 scale the model learned.
    Args:
        model_path (str): Saved model from build_model.
        scaler_path (str): Saved scaler from preprocess.
        test_csv (str): Customers to assign (dags/data/test.csv).
    Returns:
        list: One cluster number per customer, as plain ints.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # transform only, NOT fit_transform: reuse the training data's min and max
    test = pd.read_csv(test_csv)[FEATURES]
    scaled = pd.DataFrame(scaler.transform(test), columns=FEATURES)

    clusters = model.predict(scaled)
    return [int(c) for c in clusters]
