"""
Downloads the UCI Adult census dataset from OpenML and saves it as data/adult.csv.gz.

The compressed CSV is committed to the repo, so the lab never needs the internet at run
time. This script documents exactly where the data came from and can recreate the file:
    docker compose run --rm monitor python -m monitoring.fetch_data
"""
from sklearn.datasets import fetch_openml

from monitoring.data import DATA_PATH

OPENML_ID = 1590  # "adult" version 2: 48,842 people, 14 features + the income label


def main():
    frame = fetch_openml(data_id=OPENML_ID, as_frame=True).frame
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 makes the gzip file identical every time it's recreated
    frame.to_csv(DATA_PATH, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Saved {len(frame)} rows x {frame.shape[1]} columns to {DATA_PATH}")


if __name__ == "__main__":
    main()
