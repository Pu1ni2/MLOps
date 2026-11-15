# Image for the lab's jobs (training, tuning, the registry gate, tests) and for serving the model
FROM python:3.12-slim
# No .pyc files, unbuffered logs, and no usage telemetry from MLflow.
# GIT_PYTHON_REFRESH=quiet: the container has no git repository, so MLflow can't record a commit; don't warn about it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true \
    GIT_PYTHON_REFRESH=quiet
WORKDIR /lab
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The lab folder is mounted at /lab by docker-compose, so code changes need no rebuild
ENV PYTHONPATH=/lab/src
