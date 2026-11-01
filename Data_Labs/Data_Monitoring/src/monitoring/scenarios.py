"""
Simulated production: every day a batch of people arrives and the model scores them. The
batches are drawn from the production pool, and the drift events in config/scenario.yaml
change them on their days.

    docker compose run --rm monitor python -m monitoring.scenarios
prints a few numbers for every simulated day, so each drift event can be seen in them.
"""
import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from monitoring.data import FEATURES, LAB_DIR, POSITIVE_LABEL, TARGET, load_adult, split_data

SCENARIO_PATH = LAB_DIR / "config" / "scenario.yaml"


@dataclass
class DayBatch:
    """One simulated day of production."""
    day: int                 # 1, 2, ... up to the scenario's number of days
    date: datetime.date
    events: list             # names of the drift events active on this day
    data: pd.DataFrame       # the people who arrived: only their FEATURES, as production sees them
    labels: pd.Series        # their true incomes, which only become known on labels_known_on
    labels_known_on: int     # the day these labels arrive


def load_scenario(path=SCENARIO_PATH):
    """Reads the scenario, checking that every event has a known type and days inside the simulation."""
    with open(path, encoding="utf-8") as file:
        scenario = yaml.safe_load(file)
    for event in scenario["events"]:
        if event["type"] not in SAMPLING_EVENTS | ROW_EVENTS:
            raise ValueError(f"Event {event['name']!r} has an unknown type: {event['type']!r}")
        first, last = event["days"]
        if not 1 <= first <= last <= scenario["days"]:
            raise ValueError(f"Event {event['name']!r} has days {event['days']}: "
                             f"they must be [first, last] within 1-{scenario['days']}")
    return scenario


def active_events(scenario, day):
    """The events that are active on this day."""
    return [event for event in scenario["events"] if event["days"][0] <= day <= event["days"][1]]


def date_of(scenario, day):
    """The calendar date of a simulated day: day 1 is the scenario's start_date."""
    start = datetime.date.fromisoformat(str(scenario["start_date"]))
    return start + datetime.timedelta(days=day - 1)


def event_strength(event, day):
    """1.0 on every day of the event, or, if it ramps up, a share that grows to 1.0 on its last day."""
    if not event.get("ramp"):
        return 1.0
    first, last = event["days"]
    return (day - first + 1) / (last - first + 1)


# Events that change WHO arrives: each gives every person in the pool a sampling weight.

def numeric_shift(pool, event):
    """Favours people with high values in the tilted columns (low values if the tilt is negative)."""
    weights = np.ones(len(pool))
    for column, tilt in event["tilt"].items():
        values = pool[column]
        weights *= np.exp(tilt * ((values - values.mean()) / values.std()).to_numpy())
    return weights


def category_shift(pool, event):
    """Makes the listed categories more common (factor above 1) or rarer (below 1)."""
    weights = np.ones(len(pool))
    for column, factors in event["weights"].items():
        for category, factor in factors.items():
            weights[(pool[column] == category).to_numpy()] *= factor
    return weights


# Events that change the ROWS that arrived: each edits the day's data or labels in place.

def data_quality(data, labels, event, strength, rng):
    """Corrupts a share of the rows, as a broken upstream system would."""
    for column, change in event.get("multiply", {}).items():
        rows = rng.random(len(data)) < change["share"] * strength
        data[column] = data[column].mask(rows, data[column] * change["factor"])
    for column, share in event.get("missing", {}).items():
        rows = rng.random(len(data)) < share * strength
        data[column] = data[column].mask(rows)


def concept_drift(data, labels, event, strength, rng):
    """Changes the true label of a share of the people in a segment, while their features stay the same."""
    segment, flip = event["segment"], event["flip"]
    in_segment = data[segment["column"]].between(segment.get("min", -np.inf), segment.get("max", np.inf))
    rows = in_segment & (labels == flip["from"]) & (rng.random(len(data)) < flip["share"] * strength)
    labels[rows] = flip["to"]


SAMPLING_EVENTS = {"numeric_shift": numeric_shift, "category_shift": category_shift}
ROW_EVENTS = {"data_quality": data_quality, "concept_drift": concept_drift}


def simulate_day(scenario, pool, day):
    """
    Builds one day of production: draws batch_size people from the pool (the shift events change
    who is likely to be drawn), then applies the data-quality and concept-drift events to them.
    Each day has its own random seed, so the same scenario always gives the same day.
    """
    rng = np.random.default_rng([scenario["seed"], day])
    events = active_events(scenario, day)

    weights = np.ones(len(pool))
    for event in events:
        if event["type"] in SAMPLING_EVENTS:
            weights *= SAMPLING_EVENTS[event["type"]](pool, event) ** event_strength(event, day)
    drawn = rng.choice(len(pool), size=scenario["batch_size"], replace=False, p=weights / weights.sum())
    people = pool.iloc[drawn].reset_index(drop=True)
    data, labels = people[FEATURES].copy(), people[TARGET].copy()

    for event in events:
        if event["type"] in ROW_EVENTS:
            ROW_EVENTS[event["type"]](data, labels, event, event_strength(event, day), rng)

    return DayBatch(
        day=day,
        date=date_of(scenario, day),
        events=[event["name"] for event in events],
        data=data,
        labels=labels,
        labels_known_on=day + scenario["label_delay_days"],
    )


def simulate(scenario, pool):
    """Every day of the scenario, in order."""
    for day in range(1, scenario["days"] + 1):
        yield simulate_day(scenario, pool, day)


def describe(data, labels):
    """A few numbers that move when the drift events happen."""
    degree = data["education-num"] >= 13  # bachelor's degree or higher
    high_income = labels == POSITIVE_LABEL
    return {
        "age": data["age"].mean(),
        "hours": data["hours-per-week"].mean(),
        "max hours": data["hours-per-week"].max(),
        "self-emp %": 100 * data["workclass"].isin(["Self-emp-not-inc", "Self-emp-inc"]).mean(),
        "no occupation %": 100 * data["occupation"].isna().mean(),
        ">50K %": 100 * high_income.mean(),
        "degree >50K %": 100 * high_income[degree].mean(),
    }


def main():
    scenario = load_scenario()
    _, _, pool = split_data(load_adult())
    print(f"{scenario['days']} days of {scenario['batch_size']} people. The true labels arrive "
          f"{scenario['label_delay_days']} days late in the monitoring, but are shown here straight away.\n")
    rows = {"pool (no drift)": {**describe(pool[FEATURES], pool[TARGET]), "events": ""}}
    for batch in simulate(scenario, pool):
        row = {**describe(batch.data, batch.labels), "events": ", ".join(batch.events)}
        rows[f"day {batch.day:>2}  {batch.date}"] = row
    print(pd.DataFrame.from_dict(rows, orient="index").round(1).to_string())


if __name__ == "__main__":
    main()
