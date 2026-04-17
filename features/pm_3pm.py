# src/features/pm_hr3.py
import argparse
import os
import glob
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from utils.minute_bars import build_market_time_columns, load_minute_bars
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.minute_bars import build_market_time_columns, load_minute_bars

warnings.filterwarnings("ignore", category=FutureWarning)

PM_START = pd.Timestamp("13:00:00").time()   # 13:00 inclusive
PM_END   = pd.Timestamp("15:00:00").time()   # 15:00 exclusive
HR3_START = pd.Timestamp("15:00:00").time()  # 15:00 inclusive
HR3_END   = pd.Timestamp("15:50:00").time()  # 15:50 exclusive

def _load_minutes(paths: List[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No input files matched: {paths}")
    dfs = []
    for f in sorted(files):
        df = build_market_time_columns(load_minute_bars(f))
        df["__srcfile"] = os.path.basename(f)
        df["timestamp"] = df["datetime_et"]
        df["date"] = df["datetime_et"].dt.date
        df["time"] = df["datetime_et"].dt.time
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)

    colmap = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close", "VOLUME": "volume"
    }
    out = out.rename(columns=colmap)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")
    return out

def _has_session_column(df: pd.DataFrame) -> bool:
    return "session" in df.columns

def _mask_window(df: pd.DataFrame, start_t, end_t) -> pd.Series:
    # half-open [start, end)
    return (df["time"] >= start_t) & (df["time"] < end_t)

def _pick_session_masks(df: pd.DataFrame):
    if _has_session_column(df):
        pm_mask  = df["session"].isin(["PM"])
        hr3_mask = df["session"].isin(["HR3", "3PM", "HR_3PM", "3_PM"])
        if pm_mask.sum() == 0:
            pm_mask = _mask_window(df, PM_START, PM_END)
        if hr3_mask.sum() == 0:
            hr3_mask = _mask_window(df, HR3_START, HR3_END)
    else:
        pm_mask  = _mask_window(df, PM_START, PM_END)
        hr3_mask = _mask_window(df, HR3_START, HR3_END)
    return pm_mask, hr3_mask

def _first_time_offset_minutes(sub: pd.DataFrame, extreme: str, session_start_time: pd.Timestamp) -> float:
    if sub.empty:
        return np.nan
    if extreme == "high":
        idx = sub["high"].idxmax()
    else:
        idx = sub["low"].idxmin()
    t = sub.loc[idx, "timestamp"]
    delta = (t - pd.Timestamp.combine(t.date(), session_start_time)).total_seconds() / 60.0
    return int(delta)

def _dir_state(dir_val: float, close_pos: float, up_thr=0.7, dn_thr=0.3) -> str:
    if pd.isna(dir_val) or pd.isna(close_pos):
        return "neutral"
    if (dir_val > 0) and (close_pos > up_thr):
        return "bullish"
    if (dir_val < 0) and (close_pos < dn_thr):
        return "bearish"
    return "neutral"

def _compute_day_features(day_df: pd.DataFrame) -> dict:
    pm_mask, hr3_mask = _pick_session_masks(day_df)
    pm = day_df.loc[pm_mask].copy()
    hr3 = day_df.loc[hr3_mask].copy()

    out = {
        "date": day_df["date"].iloc[0],
        "bars_pm": int(pm.shape[0]),
        "bars_hr3": int(hr3.shape[0]),
    }

    # --- PM (13:00–15:00) ---
    if not pm.empty:
        pm_open = pm["open"].iloc[0]
        pm_close = pm["close"].iloc[-1]
        pm_high = pm["high"].max()
        pm_low  = pm["low"].min()
        pm_range = pm_high - pm_low
        pm_dir = pm_close - pm_open
        pm_close_pos = (pm_close - pm_low) / pm_range if pm_range > 0 else np.nan
        pm_dir_ratio = abs(pm_dir) / pm_range if pm_range > 0 else np.nan
        pm_volume = pm["volume"].sum()

        out.update({
            "pm_open": pm_open,
            "pm_high": pm_high,
            "pm_low": pm_low,
            "pm_close": pm_close,
            "pm_range": pm_range,
            "pm_dir": pm_dir,
            "pm_close_pos": pm_close_pos,
            "pm_dir_ratio": pm_dir_ratio,
            "pm_volume": pm_volume,
        })

        pm_high_time = _first_time_offset_minutes(pm, "high", PM_START)
        pm_low_time  = _first_time_offset_minutes(pm, "low", PM_START)
        out.update({
            "pm_high_time": pm_high_time,
            "pm_low_time": pm_low_time,
        })

        pm_trend_state = _dir_state(pm_dir, pm_close_pos)
        out["pm_trend_state"] = pm_trend_state
    else:
        for k in [
            "pm_open","pm_high","pm_low","pm_close",
            "pm_range","pm_dir","pm_close_pos","pm_dir_ratio",
            "pm_volume","pm_high_time","pm_low_time"
        ]:
            out[k] = np.nan
        out["pm_trend_state"] = "neutral"

    # --- 3 pm hour (15:00–15:49) ---
    if not hr3.empty:
        hr3_open = hr3["open"].iloc[0]
        hr3_close = hr3["close"].iloc[-1]
        hr3_high = hr3["high"].max()
        hr3_low  = hr3["low"].min()
        hr3_range = hr3_high - hr3_low
        hr3_dir = hr3_close - hr3_open
        hr3_dir_ratio = abs(hr3_dir) / hr3_range if hr3_range > 0 else np.nan
        hr3_close_pos = (hr3_close - hr3_low) / hr3_range if hr3_range > 0 else np.nan

        out.update({
            "hr3_open": hr3_open,
            "hr3_high": hr3_high,
            "hr3_low": hr3_low,
            "hr3_close": hr3_close,
            "hr3_range": hr3_range,
            "hr3_dir": hr3_dir,
            "hr3_dir_ratio": hr3_dir_ratio,
            "hr3_close_pos": hr3_close_pos,
        })

        hr3_high_time = _first_time_offset_minutes(hr3, "high", HR3_START)
        hr3_low_time  = _first_time_offset_minutes(hr3, "low", HR3_START)
        out.update({
            "hr3_high_time": hr3_high_time,
            "hr3_low_time": hr3_low_time,
        })

        if (pd.notna(hr3_high_time) and pd.notna(hr3_low_time)):
            out["hr3_high_first"] = bool(hr3_high_time <= hr3_low_time)
        else:
            out["hr3_high_first"] = np.nan

        hr3_trend_state = _dir_state(hr3_dir, hr3_close_pos)
        out["hr3_trend_state"] = hr3_trend_state

        out["pre_macro_price"] = hr3_close
        out["pre_macro_pos_in_hr3"] = hr3_close_pos
    else:
        for k in [
            "hr3_open","hr3_high","hr3_low","hr3_close",
            "hr3_range","hr3_dir","hr3_dir_ratio","hr3_close_pos",
            "hr3_high_time","hr3_low_time","pre_macro_price",
            "pre_macro_pos_in_hr3"
        ]:
            out[k] = np.nan
        out["hr3_high_first"] = np.nan
        out["hr3_trend_state"] = "neutral"

    # placeholder for later PD-relative stuff
    out["pre_macro_pos_in_PD"] = np.nan

    return out

def main():
    ap = argparse.ArgumentParser(description="Compute PM + 3pm hour features (per-day) from ET-naive 1m data.")
    ap.add_argument("--input", nargs="+", required=True,
                    help="Input files (CSV/Parquet) or globs with 1m OHLCV (ET-naive).")
    ap.add_argument("--out", required=True, help="Output Parquet path for per-day features.")
    ap.add_argument("--instrument", default="NQ", help="Instrument label to include in output.")
    args = ap.parse_args()

    df = _load_minutes(args.input)

    features = []
    for d, day_df in df.groupby("date", sort=True):
        features.append(_compute_day_features(day_df))
    feat = pd.DataFrame(features)

    feat["instrument"] = args.instrument
    feat["date"] = pd.to_datetime(feat["date"])

    cols = [
        "date","instrument",
        "bars_pm","bars_hr3",
        "pm_open","pm_high","pm_low","pm_close",
        "pm_range","pm_dir","pm_close_pos","pm_dir_ratio","pm_volume",
        "pm_high_time","pm_low_time","pm_trend_state",
        "hr3_open","hr3_high","hr3_low","hr3_close",
        "hr3_range","hr3_dir","hr3_dir_ratio","hr3_close_pos",
        "hr3_high_time","hr3_low_time","hr3_high_first","hr3_trend_state",
        "pre_macro_price","pre_macro_pos_in_hr3","pre_macro_pos_in_PD",
    ]
    cols = [c for c in cols if c in feat.columns]
    feat = feat[cols]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    feat.to_parquet(args.out, index=False)
    print(f"Wrote {len(feat)} rows → {args.out}")

if __name__ == "__main__":
    main()
