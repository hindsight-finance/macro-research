from __future__ import annotations

from pathlib import Path

import pandas as pd

MARKET_TZ = "America/New_York"
UTC = "UTC"
BASE_COLUMNS = ["datetime_utc", "Open", "High", "Low", "Close", "Volume"]
OPTIONAL_BASE_COLUMNS = ["instrument"]


def _read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def _coerce_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _coerce_et_to_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        return dt.dt.tz_convert(UTC)
    localized = dt.dt.tz_localize(
        MARKET_TZ,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    if localized.isna().any() and dt.notna().any():
        raise ValueError(
            "Legacy ET timestamps contain ambiguous DST-fallback values; provide datetime_utc instead."
        )
    return localized.dt.tz_convert(UTC)


def normalize_minute_bars(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "datetime_utc" in work.columns:
        datetime_utc = _coerce_utc(work["datetime_utc"])
    elif "DateTime_UTC" in work.columns:
        datetime_utc = _coerce_utc(work["DateTime_UTC"])
    elif "DateTime_ET" in work.columns:
        datetime_utc = _coerce_et_to_utc(work["DateTime_ET"])
    elif "datetime_et" in work.columns:
        datetime_utc = _coerce_et_to_utc(work["datetime_et"])
    else:
        raise ValueError("Expected one of: datetime_utc, DateTime_UTC, DateTime_ET, datetime_et")

    if datetime_utc.isna().any():
        raise ValueError("Timestamp column contains unparsable values")

    missing = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work["datetime_utc"] = datetime_utc
    columns = BASE_COLUMNS + [column for column in OPTIONAL_BASE_COLUMNS if column in work.columns]
    out = work[columns].copy()
    out = out.sort_values("datetime_utc").reset_index(drop=True)

    if out["datetime_utc"].duplicated().any():
        raise ValueError("Duplicate datetime_utc values after normalization")

    return out


def load_minute_bars(path: str | Path) -> pd.DataFrame:
    return normalize_minute_bars(_read_any(path))


def build_market_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "datetime_utc" not in work.columns:
        raise ValueError("Expected canonical datetime_utc column")

    datetime_et = pd.to_datetime(work["datetime_utc"], utc=True).dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    work["datetime_et"] = datetime_et
    work["date_et"] = datetime_et.dt.normalize()
    work["time_et"] = datetime_et.dt.time
    work["minute_of_day_et"] = datetime_et.dt.hour * 60 + datetime_et.dt.minute
    return work


def derive_session_window(df: pd.DataFrame) -> pd.DataFrame:
    work = build_market_time_columns(df) if "minute_of_day_et" not in df.columns else df.copy()
    mins = work["minute_of_day_et"]

    work["session"] = "OTHER"
    work.loc[(mins >= 19 * 60) & (mins < 24 * 60), "session"] = "ASIA"
    work.loc[(mins >= 2 * 60) & (mins < 5 * 60), "session"] = "LONDON"
    work.loc[(mins >= 9 * 60 + 30) & (mins < 11 * 60), "session"] = "NYAM"
    work.loc[(mins >= 12 * 60) & (mins < 13 * 60), "session"] = "LUNCH"
    work.loc[(mins >= 13 * 60) & (mins < 15 * 60), "session"] = "PM"

    work["window"] = "NONE"
    work.loc[(mins >= 15 * 60) & (mins <= 15 * 60 + 49), "window"] = "H3PM"
    work.loc[(mins >= 15 * 60 + 50) & (mins <= 15 * 60 + 59), "window"] = "MACRO"
    work.loc[(mins >= 16 * 60) & (mins <= 16 * 60 + 10), "window"] = "POST"
    return work
