"""
Compute daily macro-window outcomes from canonical UTC minute-base bars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    return np.where(denom != 0, (numer / denom) * 100.0, np.nan)


def _prepare_macro_bars(df: pd.DataFrame) -> pd.DataFrame:
    work = derive_session_window(build_market_time_columns(normalize_minute_bars(df)))
    work["DateTime_ET"] = work["datetime_et"]
    work["date"] = work["date_et"]
    return work.sort_values("DateTime_ET").reset_index(drop=True)


def compute_macro_outcomes(df: pd.DataFrame, macro_window_name: str) -> pd.DataFrame:
    required = {"datetime_utc", "Open", "High", "Low", "Close"}
    normalized = normalize_minute_bars(df)
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = _prepare_macro_bars(normalized)
    macro = work[work["window"] == macro_window_name].copy()
    if macro.empty:
        raise ValueError(f'No rows found where window == "{macro_window_name}".')

    post = work[work["window"].isin(POST_WINDOW_CANDIDATES)].copy()
    out_rows = []

    for d, g in macro.groupby("date"):
        g = g.sort_values("DateTime_ET")
        if g.empty:
            continue

        open_price = g["Open"].iloc[0]
        close_price = g["Close"].iloc[-1]
        high_price = g["High"].max()
        low_price = g["Low"].min()
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
            skew_ratio = np.nan
            close_in_range = np.nan

        t0 = g["DateTime_ET"].iloc[0]
        idx_high = g["High"].idxmax()
        idx_low = g["Low"].idxmin()
        t_high = g.loc[idx_high, "DateTime_ET"]
        t_low = g.loc[idx_low, "DateTime_ET"]
        macro_high_time = int((t_high - t0).total_seconds() // 60)
        macro_low_time = int((t_low - t0).total_seconds() // 60)

        post_range_points = np.nan
        post_range_pct = np.nan
        post_d = post[post["date"] == d]
        if not post_d.empty:
            post_high = post_d["High"].max()
            post_low = post_d["Low"].min()
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
                "skew_ratio": float(skew_ratio) if pd.notna(skew_ratio) else np.nan,
                "close_in_range": float(close_in_range) if pd.notna(close_in_range) else np.nan,
                "macro_high_time": int(macro_high_time),
                "macro_low_time": int(macro_low_time),
                "postclose_range_points": float(post_range_points) if pd.notna(post_range_points) else np.nan,
                "postclose_range_pct": float(post_range_pct) if pd.notna(post_range_pct) else np.nan,
                "macro_type": "UNLABELED",
            }
        )

    feats = pd.DataFrame(out_rows)
    feats["date"] = pd.to_datetime(feats["date"])
    feats["macro_type"] = feats["macro_type"].astype("category")
    return feats.sort_values("date").reset_index(drop=True)


def main():
    in_path = INPUT_PATH
    out_path = OUTPUT_PATH

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    df = load_minute_bars(in_path)
    feats = compute_macro_outcomes(df, macro_window_name=MACRO_WINDOW_NAME)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[OK] Wrote macro outcomes → {out_path}  (rows={len(feats)})")


if __name__ == "__main__":
    main()
