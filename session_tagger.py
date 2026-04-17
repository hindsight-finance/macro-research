#!/usr/bin/env python3
"""
Build canonical UTC minute-base parquet files.

Legacy input is accepted, including ET-local timestamps, but output is always:
  datetime_utc, Open, High, Low, Close, Volume[, instrument]
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.minute_bars import load_minute_bars

INPUT_FILES = [
    "input-data/nq_1m.csv",
    "input-data/es_1m.csv",
]
OUTPUT_DIR = "outputs"


def build_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    stem = Path(input_path).stem.replace("_1m", "")
    return Path(output_dir) / f"{stem}_minute_base.parquet"


def process_file(input_path: str, output_dir: str):
    """Normalize one input file into the canonical UTC minute-base schema."""
    df = load_minute_bars(input_path)
    out_path = build_output_path(input_path, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows → {out_path}")
    return out_path


def main():
    wrote_any = False
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print(f"[WARN] File not found, skipping: {input_file}")
            continue
        process_file(input_file, OUTPUT_DIR)
        wrote_any = True

    if wrote_any:
        print("\nDone. Columns: datetime_utc, Open, High, Low, Close, Volume[, instrument]")
    else:
        print("\nNo output written.")


if __name__ == "__main__":
    main()
