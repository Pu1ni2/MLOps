from src.data import TARGET, load_data, split_data


def test_load_data_shape_and_labels():
    X, y = load_data()
    assert X.shape == (4601, 57)
    assert TARGET not in X.columns
    assert set(y.unique()) == {0, 1}
    assert not X.isna().any().any()


def test_split_holds_back_20_percent():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 921
    assert len(X_train) + len(X_test) == len(X)
    # No email may be in both sets, or the test score would be too optimistic
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_keeps_spam_share():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_is_reproducible():
    X, y = load_data()
    first_test_rows = split_data(X, y)[1].index
    second_test_rows = split_data(X, y)[1].index
    assert list(first_test_rows) == list(second_test_rows)
