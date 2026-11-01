"""The simulated days: reproducible, drifting only on the configured days, with labels kept apart."""
import re

import pytest
import yaml

from monitoring.data import FEATURES, POSITIVE_LABEL
from monitoring.scenarios import active_events, event_strength, load_scenario, simulate_day


def test_same_scenario_gives_the_same_day(scenario, pool):
    first, second = simulate_day(scenario, pool, 12), simulate_day(scenario, pool, 12)
    assert first.data.equals(second.data)
    assert first.labels.equals(second.labels)


def test_events_change_only_their_own_days(scenario, pool):
    quiet = {**scenario, "events": []}
    event_days = {day for event in scenario["events"] for day in range(event["days"][0], event["days"][1] + 1)}
    for day in range(1, scenario["days"] + 1):
        with_events, without = simulate_day(scenario, pool, day), simulate_day(quiet, pool, day)
        unchanged = with_events.data.equals(without.data) and with_events.labels.equals(without.labels)
        assert unchanged == (day not in event_days), f"day {day}"


def test_each_event_is_active_only_on_its_days(scenario):
    for event in scenario["events"]:
        first, last = event["days"]
        for day in range(1, scenario["days"] + 1):
            assert (event in active_events(scenario, day)) == (first <= day <= last)


def test_gradual_drift_grows_to_full_strength(scenario):
    gradual = next(event for event in scenario["events"] if event.get("ramp"))
    first, last = gradual["days"]
    strengths = [event_strength(gradual, day) for day in range(first, last + 1)]
    assert strengths == sorted(strengths)
    assert strengths[0] < 0.2 and strengths[-1] == 1.0


def test_labels_are_kept_apart_until_they_arrive(scenario, pool):
    batch = simulate_day(scenario, pool, 5)
    assert list(batch.data.columns) == FEATURES  # what production sees: no income column
    assert batch.labels.index.equals(batch.data.index)
    assert batch.labels_known_on == 5 + scenario["label_delay_days"]


def test_broken_feed_corrupts_the_data(scenario, pool):
    data = simulate_day(scenario, pool, 15).data
    assert data["hours-per-week"].max() > 99 * 10  # hours sent in minutes
    assert data["occupation"].isna().mean() > 0.3


def test_concept_drift_changes_the_labels_but_not_the_people(scenario, pool):
    quiet = {**scenario, "events": []}
    drifted, normal = simulate_day(scenario, pool, 25), simulate_day(quiet, pool, 25)
    assert drifted.data.equals(normal.data)
    assert (drifted.labels == POSITIVE_LABEL).sum() < (normal.labels == POSITIVE_LABEL).sum()


@pytest.mark.parametrize("change, message", [
    ({"type": "typo_shift"}, "has an unknown type"),
    ({"days": [15, 31]}, "they must be [first, last] within 1-30"),
    ({"days": [20, 16]}, "they must be [first, last] within 1-30"),
])
def test_bad_scenario_is_rejected(scenario, tmp_path, change, message):
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump({**scenario, "events": [{**scenario["events"][0], **change}]}))
    with pytest.raises(ValueError, match=re.escape(message)):
        load_scenario(path)
