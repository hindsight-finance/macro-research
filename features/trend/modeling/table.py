from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from features.trend.efficiency_ratio import analyze_efficiency_ratio
from features.trend.modeling.target import build_descriptive_target
from features.trend.state_detector import ADXIndicator, DRAIndicator, IRRIndicator, MSSIndicator
from features.trend.variance_ratio import analyze_variance_ratio


SESSION_WINDOWS = {
    "1pm-3pm": (time(13, 0), time(15, 0)),
    "3pm-3:50pm": (time(15, 0), time(15, 50)),
    "3:50pm-4pm": (time(15, 50), time(16, 0)),
}
DEFAULT_SESSION_NAMES = tuple(SESSION_WINDOWS.keys())
VR_LAG = 4


def _normalize_input_bars(bars: pd.DataFrame) -> pd.DataFrame:
    timestamp_col = "timestamp" if "timestamp" in bars.columns else "DateTime_ET"
    if timestamp_col not in bars.columns:
        raise ValueError("Input bars must contain 'timestamp' or 'DateTime_ET'.")

    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(bars[timestamp_col]),
            "open": bars["open"] if "open" in bars.columns else bars["Open"],
            "high": bars["high"] if "high" in bars.columns else bars["High"],
            "low": bars["low"] if "low" in bars.columns else bars["Low"],
            "close": bars["close"] if "close" in bars.columns else bars["Close"],
        }
    )

    return normalized.sort_values("timestamp").reset_index(drop=True)


def _select_session_window(day_bars: pd.DataFrame, session_name: str) -> pd.DataFrame:
    if session_name not in SESSION_WINDOWS:
        raise ValueError(f"Unknown session_name: {session_name}")

    start_time, end_time = SESSION_WINDOWS[session_name]
    times = day_bars["timestamp"].dt.time
    mask = (times >= start_time) & (times < end_time)
    return day_bars.loc[mask].reset_index(drop=True)


def _compute_mss(window_bars: pd.DataFrame) -> tuple[float, dict]:
    result = MSSIndicator().calculate(window_bars)
    if result.error:
        raise ValueError(result.error)
    return float(result.signal), result.metadata


def _compute_adx(window_bars: pd.DataFrame, session_name: str) -> dict:
    result = ADXIndicator().calculate(window_bars, session=session_name)
    if result.error:
        raise ValueError(result.error)

    components = result.metadata["components"]
    return {
        "adx_quality": float(result.signal),
        "adx_strength": float(components["strength"]),
        "adx_persistence": float(components["persistence"]),
        "adx_persistence_margin": float(components["persistence_margin_weighted"]),
        "adx_persistence_control": float(components["persistence_time_in_control"]),
        "adx_persistence_recency": float(components["persistence_recency_weighted"]),
        "adx_crossover": float(components["crossover"]),
    }


def _compute_irr(window_bars: pd.DataFrame) -> float:
    result = IRRIndicator().calculate(window_bars)
    if result.error:
        raise ValueError(result.error)
    if result.raw_value is None:
        raise ValueError("IRR raw value unavailable")
    return float(result.raw_value)


def _compute_er(window_bars: pd.DataFrame) -> dict:
    result = analyze_efficiency_ratio(window_bars)
    return {
        "er": float(result.efficiency_ratio),
        "er_net_change": float(result.net_change),
        "er_path_length": float(result.path_length),
    }


def _compute_vr(window_bars: pd.DataFrame) -> dict:
    result = analyze_variance_ratio(window_bars, lag=VR_LAG)
    return {
        "log_vr": float(result.log_variance_ratio),
        "vr_raw": float(result.variance_ratio),
        "vr_one_period_variance": float(result.one_period_variance),
        "vr_multi_period_variance": float(result.multi_period_variance),
    }


def _compute_dra(window_bars: pd.DataFrame) -> float:
    if len(window_bars) < 15:
        return np.nan

    result = DRAIndicator().calculate(window_bars, reference_bars=window_bars.iloc[:15].copy())
    if result.error:
        raise ValueError(result.error)
    if result.raw_value is None:
        return np.nan
    return float(result.raw_value)


def _build_session_row(window_bars: pd.DataFrame, instrument: str, session_name: str) -> dict:
    row = {
        "instrument": instrument,
        "trade_date": window_bars["timestamp"].iloc[0].date(),
        "session_name": session_name,
        "window_start_ts": window_bars["timestamp"].iloc[0],
        "window_end_ts": window_bars["timestamp"].iloc[-1],
        "n_bars_raw": int(len(window_bars)),
    }
    feature_errors: list[str] = []

    try:
        row["mss"], _ = _compute_mss(window_bars)
    except Exception as exc:
        row["mss"] = np.nan
        feature_errors.append(f"mss:{exc}")

    try:
        row.update(_compute_adx(window_bars, session_name))
    except Exception as exc:
        row["adx_quality"] = np.nan
        row["adx_strength"] = np.nan
        row["adx_persistence"] = np.nan
        row["adx_persistence_margin"] = np.nan
        row["adx_persistence_control"] = np.nan
        row["adx_persistence_recency"] = np.nan
        row["adx_crossover"] = np.nan
        feature_errors.append(f"adx:{exc}")

    try:
        row["irr"] = _compute_irr(window_bars)
    except Exception as exc:
        row["irr"] = np.nan
        feature_errors.append(f"irr:{exc}")

    try:
        row.update(_compute_er(window_bars))
    except Exception as exc:
        row["er"] = np.nan
        row["er_net_change"] = np.nan
        row["er_path_length"] = np.nan
        feature_errors.append(f"er:{exc}")

    try:
        row.update(_compute_vr(window_bars))
    except Exception as exc:
        row["log_vr"] = np.nan
        row["vr_raw"] = np.nan
        row["vr_one_period_variance"] = np.nan
        row["vr_multi_period_variance"] = np.nan
        feature_errors.append(f"vr:{exc}")

    try:
        row["dra"] = _compute_dra(window_bars)
    except Exception:
        row["dra"] = np.nan

    try:
        row.update(
            build_descriptive_target(
                open_=window_bars["open"].to_numpy(),
                high=window_bars["high"].to_numpy(),
                low=window_bars["low"].to_numpy(),
                close=window_bars["close"].to_numpy(),
            )
        )
    except Exception as exc:
        row.update(
            {
                "target_strength": np.nan,
                "target_consistency": np.nan,
                "target_smoothness": np.nan,
                "target_retention": np.nan,
                "descriptive_target": np.nan,
                "target_status": f"error:{exc}",
            }
        )

    row["feature_status"] = "ok" if not feature_errors else ";".join(feature_errors)
    return row


def build_modeling_table(
    input_path: str | Path,
    instrument: str,
    session_names: Iterable[str] = DEFAULT_SESSION_NAMES,
) -> pd.DataFrame:
    bars = pd.read_parquet(input_path)
    normalized = _normalize_input_bars(bars)
    normalized["trade_date"] = normalized["timestamp"].dt.date

    rows: list[dict] = []
    for _, day_bars in normalized.groupby("trade_date", sort=True):
        for session_name in session_names:
            window_bars = _select_session_window(day_bars, session_name)
            if window_bars.empty:
                continue
            rows.append(_build_session_row(window_bars, instrument=instrument, session_name=session_name))

    return pd.DataFrame(rows)


def write_modeling_table_cache(table: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
