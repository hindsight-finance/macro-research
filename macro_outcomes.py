"""
Compute daily macro-window outcomes from canonical UTC minute-base bars.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

from utils.minute_bars import (
    build_market_time_columns,
    derive_session_window,
    load_minute_bars,
    normalize_minute_bars,
)

INPUT_PATH = Path("outputs/nq_minute_base.parquet")
OUTPUT_PATH = Path("outputs/nq_macro_outcomes.parquet")
MACRO_WINDOW_NAME = "MACRO"
POST_WINDOW_CANDIDATES = {"post", "postclose", "post_close", "postmacro", "POST", "Post", "PostClose"}


def _pct(numer, denom):
    return (numer / denom) * 100.0 if denom != 0 else math.nan


def _prepare_macro_bars(df: pl.DataFrame) -> pl.DataFrame:
    return (
        derive_session_window(build_market_time_columns(normalize_minute_bars(df)))
        .with_columns(DateTime_ET=pl.col("datetime_et"), date=pl.col("date_et"))
        .sort("DateTime_ET")
    )


def _null_to_nan(value):
    return value if value is not None else math.nan


def compute_macro_outcomes(df: pl.DataFrame, macro_window_name: str) -> pl.DataFrame:
    required = {"datetime_utc", "Open", "High", "Low", "Close"}
    normalized = normalize_minute_bars(df)
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = _prepare_macro_bars(normalized)
    macro = work.filter(pl.col("window") == macro_window_name)
    if macro.is_empty():
        raise ValueError(f'No rows found where window == "{macro_window_name}".')

    post = work.filter(pl.col("window").is_in(list(POST_WINDOW_CANDIDATES)))
    out_rows = []

    for (d,), g in macro.group_by("date", maintain_order=True):
        g = g.sort("DateTime_ET")
        open_price = g.item(0, "Open")
        close_price = g.item(g.height - 1, "Close")
        high_price = g.select(pl.col("High").max()).item()
        low_price = g.select(pl.col("Low").min()).item()
        range_points = high_price - low_price
        dir_points = close_price - open_price

        range_pct = _pct(range_points, open_price)
        dir_pct = _pct(dir_points, open_price)

        if range_points > 0:
            up_impulse = high_price - open_price
            down_impulse = open_price - low_price
            skew_ratio = (up_impulse - down_impulse) / range_points
            close_in_range = (close_price - low_price) / range_points
        else:
            skew_ratio = math.nan
            close_in_range = math.nan

        t0 = g.item(0, "DateTime_ET")
        high_row = g.filter(pl.col("High") == high_price).row(0, named=True)
        low_row = g.filter(pl.col("Low") == low_price).row(0, named=True)
        macro_high_time = int((high_row["DateTime_ET"] - t0).total_seconds() // 60)
        macro_low_time = int((low_row["DateTime_ET"] - t0).total_seconds() // 60)

        post_d = post.filter(pl.col("date") == d)
        post_range_points = math.nan
        post_range_pct = math.nan
        if not post_d.is_empty():
            post_high = post_d.select(pl.col("High").max()).item()
            post_low = post_d.select(pl.col("Low").min()).item()
            post_range_points = post_high - post_low
            post_range_pct = _pct(post_range_points, open_price)

        out_rows.append(
            {
                "date": d,
                "macro_open": float(open_price),
                "macro_close": float(close_price),
                "macro_high": float(high_price),
                "macro_low": float(low_price),
                "macro_range_points": float(range_points),
                "macro_range_pct": float(range_pct),
                "macro_dir_points": float(dir_points),
                "macro_dir_pct": float(dir_pct),
                "skew_ratio": float(skew_ratio) if not np.isnan(skew_ratio) else math.nan,
                "close_in_range": float(close_in_range) if not np.isnan(close_in_range) else math.nan,
                "macro_high_time": int(macro_high_time),
                "macro_low_time": int(macro_low_time),
                "postclose_range_points": float(post_range_points) if not np.isnan(post_range_points) else math.nan,
                "postclose_range_pct": float(post_range_pct) if not np.isnan(post_range_pct) else math.nan,
                "macro_type": "UNLABELED",
            }
        )

    return pl.DataFrame(out_rows).with_columns(pl.col("macro_type").cast(pl.Categorical)).sort("date")


def main():
    in_path = INPUT_PATH
    out_path = OUTPUT_PATH

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    df = load_minute_bars(in_path)
    feats = compute_macro_outcomes(df, macro_window_name=MACRO_WINDOW_NAME)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(out_path)
    print(f"[OK] Wrote macro outcomes → {out_path}  (rows={feats.height})")


if __name__ == "__main__":
    main()
