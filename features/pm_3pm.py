# src/features/pm_hr3.py
import argparse
import glob
import os
import sys
from datetime import time
from pathlib import Path
from typing import List

import numpy as np
import polars as pl

try:
    from utils.minute_bars import build_market_time_columns, load_minute_bars
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.minute_bars import build_market_time_columns, load_minute_bars

PM_START = time(13, 0)
PM_END = time(15, 0)
HR3_START = time(15, 0)
HR3_END = time(15, 50)


def _load_minutes(paths: List[str]) -> pl.DataFrame:
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No input files matched: {paths}")

    dfs = []
    for f in sorted(files):
        df = build_market_time_columns(load_minute_bars(f)).with_columns(
            __srcfile=pl.lit(os.path.basename(f)),
            timestamp=pl.col("datetime_et"),
            date=pl.col("datetime_et").dt.date(),
            time=pl.col("datetime_et").dt.time(),
        )
        rename_map = {
            old: new
            for old, new in {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "VOLUME": "volume",
            }.items()
            if old in df.columns
        }
        df = df.rename(rename_map)
        dfs.append(df)

    out = pl.concat(dfs, how="diagonal").sort("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")
    return out


def _mask_window(df: pl.DataFrame, start_t, end_t) -> pl.Series:
    return df.select(((pl.col("time") >= start_t) & (pl.col("time") < end_t)).alias("mask"))["mask"]


def _pick_session_masks(df: pl.DataFrame):
    if "session" in df.columns:
        pm_mask = df.select(pl.col("session").is_in(["PM"]).alias("mask"))["mask"]
        hr3_mask = df.select(pl.col("session").is_in(["HR3", "3PM", "HR_3PM", "3_PM"]).alias("mask"))["mask"]
        if pm_mask.sum() == 0:
            pm_mask = _mask_window(df, PM_START, PM_END)
        if hr3_mask.sum() == 0:
            hr3_mask = _mask_window(df, HR3_START, HR3_END)
    else:
        pm_mask = _mask_window(df, PM_START, PM_END)
        hr3_mask = _mask_window(df, HR3_START, HR3_END)
    return pm_mask, hr3_mask


def _first_time_offset_minutes(sub: pl.DataFrame, extreme: str, session_start_time) -> float:
    if sub.is_empty():
        return np.nan
    idx = int(sub["high"].arg_max() if extreme == "high" else sub["low"].arg_min())
    t = sub.item(idx, "timestamp")
    delta = (t - t.replace(hour=session_start_time.hour, minute=session_start_time.minute, second=0, microsecond=0)).total_seconds() / 60.0
    return int(delta)


def _dir_state(dir_val: float, close_pos: float, up_thr=0.7, dn_thr=0.3) -> str:
    if dir_val is None or close_pos is None or np.isnan(dir_val) or np.isnan(close_pos):
        return "neutral"
    if (dir_val > 0) and (close_pos > up_thr):
        return "bullish"
    if (dir_val < 0) and (close_pos < dn_thr):
        return "bearish"
    return "neutral"


def _compute_day_features(day_df: pl.DataFrame) -> dict:
    pm_mask, hr3_mask = _pick_session_masks(day_df)
    pm = day_df.filter(pm_mask)
    hr3 = day_df.filter(hr3_mask)

    out = {"date": day_df.item(0, "date"), "bars_pm": pm.height, "bars_hr3": hr3.height}

    if not pm.is_empty():
        pm_open = pm.item(0, "open")
        pm_close = pm.item(pm.height - 1, "close")
        pm_high = pm["high"].max()
        pm_low = pm["low"].min()
        pm_range = pm_high - pm_low
        pm_dir = pm_close - pm_open
        pm_close_pos = (pm_close - pm_low) / pm_range if pm_range > 0 else np.nan
        pm_dir_ratio = abs(pm_dir) / pm_range if pm_range > 0 else np.nan
        out.update(
            {
                "pm_open": pm_open,
                "pm_high": pm_high,
                "pm_low": pm_low,
                "pm_close": pm_close,
                "pm_range": pm_range,
                "pm_dir": pm_dir,
                "pm_close_pos": pm_close_pos,
                "pm_dir_ratio": pm_dir_ratio,
                "pm_volume": pm["volume"].sum(),
                "pm_high_time": _first_time_offset_minutes(pm, "high", PM_START),
                "pm_low_time": _first_time_offset_minutes(pm, "low", PM_START),
            }
        )
        out["pm_trend_state"] = _dir_state(pm_dir, pm_close_pos)
    else:
        for k in ["pm_open", "pm_high", "pm_low", "pm_close", "pm_range", "pm_dir", "pm_close_pos", "pm_dir_ratio", "pm_volume", "pm_high_time", "pm_low_time"]:
            out[k] = np.nan
        out["pm_trend_state"] = "neutral"

    if not hr3.is_empty():
        hr3_open = hr3.item(0, "open")
        hr3_close = hr3.item(hr3.height - 1, "close")
        hr3_high = hr3["high"].max()
        hr3_low = hr3["low"].min()
        hr3_range = hr3_high - hr3_low
        hr3_dir = hr3_close - hr3_open
        hr3_dir_ratio = abs(hr3_dir) / hr3_range if hr3_range > 0 else np.nan
        hr3_close_pos = (hr3_close - hr3_low) / hr3_range if hr3_range > 0 else np.nan
        hr3_high_time = _first_time_offset_minutes(hr3, "high", HR3_START)
        hr3_low_time = _first_time_offset_minutes(hr3, "low", HR3_START)
        out.update(
            {
                "hr3_open": hr3_open,
                "hr3_high": hr3_high,
                "hr3_low": hr3_low,
                "hr3_close": hr3_close,
                "hr3_range": hr3_range,
                "hr3_dir": hr3_dir,
                "hr3_dir_ratio": hr3_dir_ratio,
                "hr3_close_pos": hr3_close_pos,
                "hr3_high_time": hr3_high_time,
                "hr3_low_time": hr3_low_time,
                "hr3_high_first": bool(hr3_high_time <= hr3_low_time) if not np.isnan(hr3_high_time) and not np.isnan(hr3_low_time) else np.nan,
                "hr3_trend_state": _dir_state(hr3_dir, hr3_close_pos),
                "pre_macro_price": hr3_close,
                "pre_macro_pos_in_hr3": hr3_close_pos,
            }
        )
    else:
        for k in ["hr3_open", "hr3_high", "hr3_low", "hr3_close", "hr3_range", "hr3_dir", "hr3_dir_ratio", "hr3_close_pos", "hr3_high_time", "hr3_low_time", "pre_macro_price", "pre_macro_pos_in_hr3"]:
            out[k] = np.nan
        out["hr3_high_first"] = np.nan
        out["hr3_trend_state"] = "neutral"

    out["pre_macro_pos_in_PD"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute PM + 3pm hour features (per-day) from ET-naive 1m data.")
    ap.add_argument("--input", nargs="+", required=True, help="Input files (CSV/Parquet) or globs with 1m OHLCV.")
    ap.add_argument("--out", required=True, help="Output Parquet path for per-day features.")
    ap.add_argument("--instrument", default="NQ", help="Instrument label to include in output.")
    args = ap.parse_args()

    df = _load_minutes(args.input)
    features = [_compute_day_features(day_df) for (_,), day_df in df.group_by("date", maintain_order=True)]
    feat = pl.DataFrame(features).with_columns(instrument=pl.lit(args.instrument), date=pl.col("date").cast(pl.Date))

    cols = [
        "date", "instrument", "bars_pm", "bars_hr3", "pm_open", "pm_high", "pm_low", "pm_close", "pm_range", "pm_dir", "pm_close_pos", "pm_dir_ratio", "pm_volume", "pm_high_time", "pm_low_time", "pm_trend_state", "hr3_open", "hr3_high", "hr3_low", "hr3_close", "hr3_range", "hr3_dir", "hr3_dir_ratio", "hr3_close_pos", "hr3_high_time", "hr3_low_time", "hr3_high_first", "hr3_trend_state", "pre_macro_price", "pre_macro_pos_in_hr3", "pre_macro_pos_in_PD",
    ]
    feat = feat.select([c for c in cols if c in feat.columns])

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    feat.write_parquet(args.out)
    print(f"Wrote {feat.height} rows → {args.out}")


if __name__ == "__main__":
    main()
