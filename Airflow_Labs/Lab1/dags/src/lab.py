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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

# The 3 customer columns used for clustering
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]

# K-Means settings; random_state makes every run give the same result
KMEANS_SETTINGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

# Fewer clean rows than this means something is wrong with the data
MIN_ROWS = 100

# Numbers of clusters to try; each k is trained as its own parallel Airflow task.
# The silhouette score needs at least 2 clusters, so k starts at 2.
K_MIN = 2
K_MAX = 10

# The silhouette score compares customers with each other, so it's computed on a sample to stay fast
SILHOUETTE_SAMPLE = 2000

# CPU threads per training task. By default scikit-learn uses EVERY CPU thread in every
# task, so 9 parallel tasks fight over the CPU. Measured on 9 parallel models:
# 26.7 s with all threads per task vs 0.5 s with 1 thread per task, same results.
TRAIN_THREADS = 1


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


def validate_data(raw_csv, out_dir, min_rows=MIN_ROWS):
    """
    Checks the raw data before any modelling and saves a clean copy.
    Raises ValueError with a clear message if the data can't be used:
      - a FEATURES column is missing
      - a FEATURES column is not numeric
      - fewer than min_rows rows are left after dropping incomplete rows
    Args:
        raw_csv (str): Path returned by load_data.
        out_dir (str): This run's folder.
        min_rows (int): Smallest acceptable number of clean rows.
    Returns:
        dict: {"clean_path": <out_dir>/clean.csv, "rows_in": int, "rows_out": int,
               "missing": {column: number of missing values}}
    """
    df = pd.read_csv(raw_csv)

    missing_columns = [col for col in FEATURES if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    for col in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric, but has type {df[col].dtype}")

    missing = {col: int(df[col].isna().sum()) for col in FEATURES}

    clean = df[FEATURES].dropna()
    if len(clean) < min_rows:
        raise ValueError(f"Only {len(clean)} complete rows left, need at least {min_rows}")

    clean_path = os.path.join(out_dir, "clean.csv")
    clean.to_csv(clean_path, index=False)

    return {"clean_path": clean_path, "rows_in": len(df), "rows_out": len(clean), "missing": missing}


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


def train_kmeans(data_path, out_dir, k):
    """
    Trains ONE K-Means model with k clusters, scores it, and saves it.
    In the pipeline, Airflow runs one copy of this per k, all in parallel.
    Args:
        data_path (str): Scaled data CSV from preprocess.
        out_dir (str): This run's folder.
        k (int): Number of clusters.
    Returns:
        dict: {"k": int, "sse": float, "silhouette": float,
               "model_path": <out_dir>/models/kmeans_k<k>.joblib}
              sse: how far customers are from their cluster centre (lower = tighter clusters)
              silhouette: how well separated the clusters are, from -1 to 1 (higher = better)
    """
    data = pd.read_csv(data_path)
    with threadpool_limits(limits=TRAIN_THREADS):
        model = KMeans(n_clusters=k, **KMEANS_SETTINGS).fit(data)
        silhouette = silhouette_score(data, model.labels_, sample_size=SILHOUETTE_SAMPLE, random_state=42)

    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"kmeans_k{k}.joblib")
    joblib.dump(model, model_path)

    # float()/int(): Airflow can only pass plain Python values between tasks, not numpy numbers
    return {"k": int(k), "sse": float(model.inertia_), "silhouette": float(silhouette), "model_path": model_path}


def select_best_k(results, out_dir):
    """
    Picks the best number of clusters from all the train_kmeans results.
    Uses the elbow method on SSE: the k after which adding more clusters stops
    reducing SSE by much. If there's no clear elbow, falls back to the k with
    the highest silhouette score. Saves every k's scores as k_scores.csv.
    Args:
        results (list): One train_kmeans result dict per k.
        out_dir (str): This run's folder.
    Returns:
        dict: {"best_k": int, "method": "elbow" or "silhouette", "silhouette": float,
               "sse": float, "model_path": str, "scores_path": <out_dir>/k_scores.csv}
    """
    # Parallel tasks can finish in any order, so sort by k; fix the column order for a readable CSV
    scores = pd.DataFrame(results)[["k", "sse", "silhouette", "model_path"]]
    scores = scores.sort_values("k").reset_index(drop=True)
    scores_path = os.path.join(out_dir, "k_scores.csv")
    scores.to_csv(scores_path, index=False)

    elbow = KneeLocator(scores["k"], scores["sse"], curve="convex", direction="decreasing").elbow
    if elbow is not None:
        best, method = scores[scores["k"] == elbow].iloc[0], "elbow"
    else:
        best, method = scores.loc[scores["silhouette"].idxmax()], "silhouette"

    return {
        "best_k": int(best["k"]),
        "method": method,
        "silhouette": float(best["silhouette"]),
        "sse": float(best["sse"]),
        "model_path": str(best["model_path"]),
        "scores_path": scores_path,
    }


def predict(model_path, scaler_path, test_csv):
    """
    Assigns each customer in test_csv to a cluster.
    The test data is scaled with the SAME scaler that was fitted on the
    training data, so it's on the same 0-1 scale the model learned.
    Args:
        model_path (str): Saved model of the chosen k (from select_best_k).
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
