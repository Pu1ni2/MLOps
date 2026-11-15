"""The pipeline's settings, read from config/pipeline.yaml and checked when they are loaded."""
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = LAB_DIR / "config" / "pipeline.yaml"


def load_config(path=CONFIG_PATH):
    """Reads the settings, stopping with a clear message on any that would break a run later."""
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    shares = config["data"]["split"]
    if (set(shares) != {"train", "validation", "test"} or not all(0 < share < 1 for share in shares.values())
            or abs(sum(shares.values()) - 1) > 1e-9):
        raise ValueError(f"data.split needs train, validation and test shares between 0 and 1 "
                         f"that add up to 1, not {shares}")
    from wine_quality.models import TRUSTED_TYPES  # imported here: the models module loads the ML libraries
    for family in config["baselines"]:
        if family not in TRUSTED_TYPES:
            raise ValueError(f"Unknown model family {family!r} in baselines: use one of {list(TRUSTED_TYPES)}")
    for key, value in config["tuning"].items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"tuning.{key} must be a whole number of at least 1, not {value!r}")
    registry = config["registry"]
    if not isinstance(registry["resamples"], int) or registry["resamples"] < 100:
        raise ValueError(f"registry.resamples must be a whole number of at least 100, not {registry['resamples']!r}")
    if not 0.5 <= registry["min_win_share"] <= 1:
        raise ValueError(f"registry.min_win_share must be between 0.5 and 1, not {registry['min_win_share']!r}")
    if registry["max_brier_increase"] < 0:
        raise ValueError(f"registry.max_brier_increase can't be negative: {registry['max_brier_increase']!r}")
    return config
