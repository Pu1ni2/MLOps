# MLOps Labs

Hands-on MLOps projects covering automated testing, CI/CD and ML workflows. Each lab has its own folder.

| Lab | Folder | What it covers |
|-----|--------|----------------|
| GitHub Lab 1 | [Github_Labs/Lab1](Github_Labs/Lab1) | Unit testing with pytest and unittest, run by GitHub Actions on every push and pull request |
| GitHub Lab 2 | [Github_Labs/Lab2](Github_Labs/Lab2) | Spam-detection model that GitHub Actions trains, calibrates, checks against a quality gate and publishes as a versioned GitHub Release |

## Repository layout

- Each lab has its own folder, grouped by topic (`Github_Labs/` now, more to come).
- All GitHub Actions workflows are in [.github/workflows](.github/workflows), because GitHub only runs workflows from there. Each workflow runs inside its own lab's folder and only starts when that lab's files change.
