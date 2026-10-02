# GitHub Lab 1: Testing with Pytest, Unittest and GitHub Actions

[![Github Lab1 - Pytest](https://github.com/Pu1ni2/MLOps/actions/workflows/github_lab1_pytest_action.yml/badge.svg)](https://github.com/Pu1ni2/MLOps/actions/workflows/github_lab1_pytest_action.yml)
[![Github Lab1 - Unittest](https://github.com/Pu1ni2/MLOps/actions/workflows/github_lab1_unittest_action.yml/badge.svg)](https://github.com/Pu1ni2/MLOps/actions/workflows/github_lab1_unittest_action.yml)

A Python calculator module with a complete automated test suite. GitHub Actions runs the tests on four Python versions on every push and pull request, and fails the build if test coverage drops below 90%.

## Calculator

`src/calculator.py`:

| Function | Returns | Raises |
|----------|---------|--------|
| `fun1(x, y)` | `x + y` | `ValueError` if an input is not a number |
| `fun2(x, y)` | `x - y` | `ValueError` if an input is not a number |
| `fun3(x, y)` | `x * y` | `ValueError` if an input is not a number |
| `fun4(x, y, z)` | `x + y + z` | `ValueError` if an input is not a number |
| `fun5(x, y)` | `x / y` | `ValueError` if an input is not a number, `ZeroDivisionError` if `y` is 0 |
| `fun6(x, y)` | `x ** y` | `ValueError` if an input is not a number |
| `fun7(numbers)` | average of a list | `ValueError` if the list is empty or contains a non-number |

## Tests

The same checks are written in two frameworks: pytest (`test/test_pytest.py`) and unittest (`test/test_unittest.py`).

- **Correct results**, including negative numbers and zero. Decimal results are compared with `pytest.approx` / `assertAlmostEqual`, because floating point is not exact. For example, `(0.1 + 0.2) / 2` is `0.15000000000000002`.
- **Bad input:** every function must reject non-numbers, divide-by-zero and invalid lists. In pytest, one parametrized test runs 5 functions × 3 bad inputs (15 cases). In unittest, `subTest` reports each combination separately.
- **Coverage:** 100% of `src/` is exercised by the tests.

## CI with GitHub Actions

Both workflows live in the repo-root `.github/workflows/`:

- **Triggers:** pushes and pull requests to `main` that touch this lab, plus a manual **Run workflow** button.
- **Python version matrix:** tests run on Python 3.11, 3.12, 3.13 and 3.14 in parallel.
- **Coverage gate:** `pytest-cov` measures coverage, and the build fails below 90%.
- **Reports:** each Python version uploads its JUnit test report and coverage XML as an artifact.
- **Speed:** pip downloads are cached between runs.

## Project structure

```
MLOps/                                   (repo root)
├── .github/workflows/
│   ├── github_lab1_pytest_action.yml    # pytest + coverage gate, uploads reports
│   └── github_lab1_unittest_action.yml  # unittest
└── Github_Labs/Lab1/
    ├── data/
    ├── src/
    │   └── calculator.py                # fun1-fun7
    ├── test/
    │   ├── test_pytest.py
    │   └── test_unittest.py
    └── requirements.txt
```

## Run the tests locally

From `Github_Labs/Lab1`:

```
python -m venv lab_01
lab_01\Scripts\activate
pip install -r requirements.txt

pytest --cov=src --cov-report=term-missing
python -m unittest test.test_unittest -v
```
