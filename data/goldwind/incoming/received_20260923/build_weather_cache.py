from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from weather_api import get_archival_weather

TURBINES = {
    # Mapping follows upload order; swap IDs here if organisers identify the files oppositely.
    "T1": (43.645150, 78.535604),
    "T2": (43.643198, 78.538828),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-01-31")
    p.add_argument("--end", default="2026-02-28")
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--run-hour-utc", type=int, default=0)
    p.add_argument("--cache-dir", default="data/weather_cache")
    p.add_argument("--combined", default="data/processed/weather_test_cache.csv")
    args = p.parse_args()

    frames = []
    failures = []
    for d in pd.date_range(args.start, args.end, freq="D"):
        for tid, (lat, lon) in TURBINES.items():
            try:
                x = get_archival_weather(
                    lat, lon, d, args.horizon,
                    run_hour_utc=args.run_hour_utc,
                    cache_dir=args.cache_dir,
                )
                x["turbine_id"] = tid
                frames.append(x)
                print(f"OK {d.date()} {tid}: {len(x)} rows")
            except Exception as e:
                failures.append((str(d.date()), tid, str(e)))
                print(f"FAIL {d.date()} {tid}: {e}")

    if frames:
        out = pd.concat(frames, ignore_index=True)
        Path(args.combined).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.combined, index=False)
        print(f"Saved {len(out)} rows -> {args.combined}")
    if failures:
        pd.DataFrame(failures, columns=["run_date", "turbine_id", "error"]).to_csv(
            Path(args.combined).with_name("weather_cache_failures.csv"), index=False
        )
        raise SystemExit(f"{len(failures)} requests failed; see weather_cache_failures.csv")


if __name__ == "__main__":
    main()
