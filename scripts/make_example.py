"""Generate artificial trip records for a local CPU smoke run."""
from pathlib import Path
import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/example.pkl"))
    args = parser.parse_args()
    rows = []
    for user in range(30):
        for event in range(12):
            rows.append({"userID": user, "startTime": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=12 * event, minutes=user), "origin": event % 4 + 1, "destination": (event + 1) % 4 + 1})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(args.output)
    print(f"Wrote artificial example to {args.output}")


if __name__ == "__main__":
    main()
