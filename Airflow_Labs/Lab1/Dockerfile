# Official Airflow 3.3.2 image (Python 3.12) plus this lab's Python packages
FROM apache/airflow:3.3.2-python3.12
COPY requirements.txt /requirements.txt
# Listing apache-airflow keeps pip from changing Airflow's own version while installing the rest
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
