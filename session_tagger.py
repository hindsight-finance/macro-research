#!/usr/bin/env python3
"""
Session Tagger (ET-naïve) for NQ 1-minute data.

Adds session and window tags to minute-bar data.
  - session: ASIA / LONDON / NYAM / LUNCH / PM / OTHER
  - window:  H3PM / MACRO / POST / NONE
"""
import os
from pathlib import Path

import pandas as pd

# =============================
# HARD-CODED PATHS (edit these)
# =============================
INPUT_FILES = [
    "data/nq_1m.csv",
    "data/es_1m.csv",
]
OUTPUT_DIR = "outputs"


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic schema checks
    needed = {"DateTime_ET", "Open", "High", "Low", "Close", "Volume"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    # Parse timestamp (ET-naïve)
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])
    df = df.sort_values("DateTime_ET").reset_index(drop=True)
    return df


def minutes_since_midnight(dt: pd.Timestamp) -> int:
    return dt.hour * 60 + dt.minute


def tag_session(row_minutes: int) -> str:
    """
    Coarse session tags by clock-only (ET-naïve). Ranges are half-open [start, end).
    """
    t = row_minutes
    # Asia: 19:00–24:00
    if 19*60 <= t < 24*60:
        return "ASIA"
    # London: 02:00–05:00
    if 2*60 <= t < 5*60:
        return "LONDON"
    # NY AM: 09:30–11:00
    if (9*60 + 30) <= t < 11*60:
        return "NYAM"
    # Lunch: 12:00–13:00
    if 12*60 <= t < 13*60:
        return "LUNCH"
    # PM: 13:00–15:00
    if 13*60 <= t < 15*60:
        return "PM"
    return "OTHER"


def tag_window(row_minutes: int) -> str:
    """
    Fine windows relevant to the study.
    - H3PM: 15:00–15:49
    - MACRO: 15:50–15:59
    - POST: 16:00–16:10
    """
    t = row_minutes
    if 15*60 <= t <= 15*60 + 49:
        return "H3PM"
    if 15*60 + 50 <= t <= 15*60 + 59:
        return "MACRO"
    if 16*60 <= t <= 16*60 + 10:
        return "POST"
    return "NONE"


def process_file(input_path: str, output_dir: str):
    """Process a single CSV file and save as tagged parquet."""
    df = read_csv(input_path)

    # Compute minute-of-day and tags
    mins = df["DateTime_ET"].map(minutes_since_midnight)
    df["session"] = mins.map(tag_session)
    df["window"] = mins.map(tag_window)

    # Order columns
    first_cols = ["DateTime_ET", "session", "window"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    # Build output path: data/nq_1m.csv -> outputs/nq_1m.parquet
    stem = Path(input_path).stem  # e.g. "nq_1m"
    out_path = Path(output_dir) / f"{stem}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write parquet
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df):,} rows → {out_path}")
    return out_path


def main():
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print(f"[WARN] File not found, skipping: {input_file}")
            continue
        process_file(input_file, OUTPUT_DIR)

    print("\nDone. Columns: DateTime_ET, session, window, Open, High, Low, Close, Volume")


if __name__ == "__main__":
    main()
