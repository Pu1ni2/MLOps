# The MLflow tracking server: the official image, plus the PostgreSQL driver it doesn't include.
# psycopg (version 3) is the driver SQLAlchemy 2.1, used by this image, picks for postgresql:// addresses.
FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir "psycopg[binary]==3.3.9"
