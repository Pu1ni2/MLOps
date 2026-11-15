"""The settings are checked when loaded, and the data is labelled, de-duplicated and split as described."""
import copy

import pytest
import yaml

from wine_quality.config import load_config
from wine_quality.data import FEATURES, TARGET, load_wines, prepare, seen_in_training


def write(tmp_path, config):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("section, key, value, message", [
    ("data", "split", {"train": 0.7, "validation": 0.2, "test": 0.2}, "data.split needs"),
    ("baselines", "svm", {"C": 1}, "Unknown model family 'svm'"),
    ("tuning", "trials", 0, "tuning.trials must be"),
    ("registry", "resamples", 10, "registry.resamples must be"),
    ("registry", "min_win_share", 0.3, "registry.min_win_share must be"),
    ("registry", "max_brier_increase", -0.1, "registry.max_brier_increase can't be negative"),
])
def test_bad_settings_are_rejected(config, tmp_path, section, key, value, message):
    bad = copy.deepcopy(config)
    bad[section][key] = value
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, bad))


def test_both_files_are_read_despite_different_separators():
    wines = load_wines()
    assert len(wines) == 6497
    assert wines["is_red"].sum() == 1599
    assert set(FEATURES) <= set(wines.columns) and "quality" in wines.columns
    assert wines.isna().sum().sum() == 0


def test_duplicates_are_removed_and_the_split_is_stratified(config, splits):
    sizes = {part: len(table) for part, table in splits.items()}
    assert sizes == {"train": 3192, "validation": 1064, "test": 1064}
    shares = [round(table[TARGET].mean(), 2) for table in splits.values()]
    assert shares == [0.19, 0.19, 0.19]
    assert seen_in_training(splits["train"], splits["test"]) == 0
    assert seen_in_training(splits["train"], splits["validation"]) == 0


def test_the_split_is_the_same_every_time(config, splits):
    again = prepare(config)[0]
    assert all(again[part].equals(splits[part]) for part in splits)


def test_keeping_duplicates_leaks_test_wines_into_training(config):
    kept = copy.deepcopy(config)
    kept["data"]["remove_duplicates"] = False
    leaky, summary = prepare(kept)
    assert summary["wines_used"] == 6497
    assert seen_in_training(leaky["train"], leaky["test"]) == 313


def test_the_quality_score_is_not_a_feature():
    assert "quality" not in FEATURES and TARGET not in FEATURES
