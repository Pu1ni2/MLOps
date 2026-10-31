# Image for the monitoring jobs (data checks, simulation, tests)
FROM python:3.12-slim

# No .pyc files, unbuffered logs, and no usage telemetry from Evidently
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DO_NOT_TRACK=1 \
    EVIDENTLY_DISABLE_TELEMETRY=1

WORKDIR /lab
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The lab folder is mounted at /lab by docker-compose, so code changes need no rebuild
ENV PYTHONPATH=/lab/src
