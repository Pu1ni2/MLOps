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
    return config
