"""
Checks the served model. The serve container answers REST requests with models:/wine_quality@champion.
This sends it the test wines, and checks that its answers are exactly what the registry's champion predicts.

    docker compose --profile serve up -d serve
    docker compose run --rm trainer python -m wine_quality.serving

The serve container reads @champion once, when it starts. If the champion changes later, it keeps serving
the old model until it is restarted, and this check says so.
"""
import os
import sys
import time

import numpy as np
import requests
from mlflow import MlflowClient

from wine_quality.config import load_config
from wine_quality.data import FEATURES, TARGET, prepare
from wine_quality.registry import CHAMPION, chances, version_with_alias

SERVE_URL = os.environ.get("SERVE_URL", "http://localhost:5001")


def wait_until_ready(url, timeout=120):
    """Waits until the server answers its /ping health check; the first start can take a few seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/ping", timeout=5).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    sys.exit(f"The model server at {url} didn't answer. Start it with: docker compose --profile serve up -d serve")


def served_chances(url, wines):
    """Sends the wines to the server's /invocations endpoint; returns its chance that each is high quality."""
    payload = {"dataframe_split": wines[FEATURES].to_dict(orient="split", index=False)}
    response = requests.post(f"{url}/invocations", json=payload, timeout=60)
    response.raise_for_status()
    return np.asarray(response.json()["predictions"])[:, 1]


def main():
    config = load_config()
    test = prepare(config)[0]["test"]
    name = config["registry"]["model_name"]
    champion = version_with_alias(MlflowClient(), name, CHAMPION)
    if champion is None:
        sys.exit(f"{name} has no @champion yet: run the pipeline first.")

    wait_until_ready(SERVE_URL)
    served = served_chances(SERVE_URL, test)
    expected = chances(f"models:/{name}/{champion.version}", test)
    difference = float(np.abs(served - expected).max())
    if difference > 1e-9:
        sys.exit(f"The server's answers differ from the champion's (version {champion.version}) by up to "
                 f"{difference:.4f}: it is probably serving an older champion. Restart it with: "
                 f"docker compose --profile serve restart serve")
    print(f"The server answered for all {len(test):,} test wines, exactly as the champion (version "
          f"{champion.version}) predicts.")
    print("For example:")
    for row in range(3):
        actual = "high quality" if test[TARGET].iloc[row] else "ordinary"
        print(f"  wine {row + 1}: {served[row]:.0%} chance of high quality (actually {actual})")


if __name__ == "__main__":
    main()
