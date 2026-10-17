# MLOps Labs

Hands-on MLOps projects covering automated testing, CI/CD, ML workflows and pipeline orchestration. Each lab has its own folder.

| Lab | Folder | What it covers |
|-----|--------|----------------|
| GitHub Lab 1 | [Github_Labs/Lab1](Github_Labs/Lab1) | Unit testing with pytest and unittest, run by GitHub Actions on every push and pull request |
| GitHub Lab 2 | [Github_Labs/Lab2](Github_Labs/Lab2) | Spam-detection model that GitHub Actions trains, calibrates, checks against a quality gate and publishes as a versioned GitHub Release |
| Airflow Lab 1 | [Airflow_Labs/Lab1](Airflow_Labs/Lab1) | Customer-segmentation pipeline orchestrated with Apache Airflow 3: data checks with retries, parallel training with dynamic task mapping, a quality-gate branch and settings chosen at trigger time |

## Repository layout

- Each lab has its own folder, grouped by topic (`Github_Labs/`, `Airflow_Labs/`, more to come).
- All GitHub Actions workflows are in [.github/workflows](.github/workflows), because GitHub only runs workflows from there. Each workflow runs inside its own lab's folder and only starts when that lab's files change.
- Airflow labs run locally in Docker. Each one has its own `docker-compose.yaml` and a README with setup and run steps.
