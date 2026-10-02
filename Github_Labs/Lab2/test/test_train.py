import joblib

from src import train_model


def test_model_beats_always_guessing_not_spam(split, small_model):
    X_train, X_test, y_train, y_test = split
    # Always answering "not spam" would be right about 61% of the time
    baseline = 1 - y_test.mean()
    assert small_model.score(X_test, y_test) > baseline + 0.25


def test_main_saves_versioned_model(tmp_path):
    path = train_model.main(["--timestamp", "test", "--output-dir", str(tmp_path), "--n-estimators", "10"])
    assert path == tmp_path / "model_test_rf.joblib"
    model = joblib.load(path)
    assert model.n_estimators == 10
