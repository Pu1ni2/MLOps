# GitHub Lab 1: Testing with Pytest, Unittest and GitHub Actions

A small Python calculator module tested with both pytest and unittest. GitHub Actions runs both test suites every time this lab's code is pushed to `main`.

## Project structure

```
MLOps/                                   (repo root)
├── .github/workflows/
│   ├── github_lab1_pytest_action.yml    # runs pytest, uploads an XML test report
│   └── github_lab1_unittest_action.yml  # runs unittest
└── Github_Labs/Lab1/
    ├── data/
    ├── src/
    │   └── calculator.py
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

pytest
python -m unittest test.test_unittest
```
