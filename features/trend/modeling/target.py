from __future__ import annotations

import numpy as np
import pandas as pd


INITIAL_BALANCE_BARS = {
    "1pm-3pm": 15,
    "3pm-3:50pm": 10,
    "3:50pm-4pm": 10,
}


def _coerce_ohlc_arrays(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    open_arr = np.asarray(open_, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)

    if not (open_arr.size and high_arr.size and low_arr.size and close_arr.size):
        raise ValueError("open_, high, low, and close must be non-empty")

    if not (open_arr.shape == high_arr.shape == low_arr.shape == close_arr.shape):
        raise ValueError("open_, high, low, and close must have matching shapes")

    if not (
        np.isfinite(open_arr).all()
        and np.isfinite(high_arr).all()
        and np.isfinite(low_arr).all()
        and np.isfinite(close_arr).all()
    ):
        raise ValueError("open_, high, low, and close must contain only finite values")

    if np.any(close_arr <= 0):
        raise ValueError("close must contain only positive values")

    return open_arr, high_arr, low_arr, close_arr


def _compute_realized_range_and_close_pos(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
) -> tuple[float, np.ndarray]:
    realized_range = float(high_arr.max() - low_arr.min())
    close_pos = (close_arr - low_arr.min()) / (realized_range + 1e-12)
    return realized_range, close_pos


def _coerce_volume_array(volume: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    volume_arr = np.asarray(volume, dtype=float)

    if volume_arr.shape != expected_shape:
        raise ValueError("volume must have matching shape")

    if not np.isfinite(volume_arr).all() or np.any(volume_arr < 0):
        raise ValueError("volume must contain only finite non-negative values")

    return volume_arr


def _collapse_directional_swings(close_arr: np.ndarray) -> list[tuple[float, float]]:
    deltas = np.diff(close_arr)
    swings: list[tuple[float, float]] = []
    current_sign = 0.0
    current_magnitude = 0.0

    for delta in deltas:
        sign = float(np.sign(delta))
        if sign == 0.0:
            continue

        if current_sign == 0.0 or sign == current_sign:
            current_sign = sign
            current_magnitude += abs(float(delta))
            continue

        swings.append((current_sign, current_magnitude))
        current_sign = sign
        current_magnitude = abs(float(delta))

    if current_sign != 0.0:
        swings.append((current_sign, current_magnitude))

    return swings


def build_descriptive_target(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict:
    open_arr, high_arr, low_arr, close_arr = _coerce_ohlc_arrays(
        open_=open_,
        high=high,
        low=low,
        close=close,
    )

    returns = np.diff(np.log(close_arr))
    n_returns = returns.size

    if n_returns:
        return_std = np.std(returns, ddof=1) if n_returns > 1 else 0.0
        target_strength = 1.0 - np.exp(
            -abs(np.sum(returns)) / (return_std * np.sqrt(n_returns) + 1e-12)
        )
    else:
        target_strength = 0.0

    nonzero_returns = returns[returns != 0]
    if nonzero_returns.size:
        up_share = float(np.mean(nonzero_returns > 0))
        down_share = float(np.mean(nonzero_returns < 0))
        target_consistency = 2.0 * max(up_share, down_share) - 1.0
        sign_changes = np.count_nonzero(np.sign(nonzero_returns[1:]) != np.sign(nonzero_returns[:-1]))
        target_smoothness = 1.0 - (sign_changes / (nonzero_returns.size - 1) if nonzero_returns.size > 1 else 0.0)
    else:
        target_consistency = 0.0
        target_smoothness = 1.0

    path_range = float(high_arr.max() - low_arr.min())
    target_retention = np.clip(
        abs(float(close_arr[-1] - open_arr[0])) / (path_range + 1e-12),
        0.0,
        1.0,
    )

    descriptive_target = (
        0.35 * target_strength
        + 0.25 * target_consistency
        + 0.20 * target_smoothness
        + 0.20 * target_retention
    )

    return {
        "target_strength": float(target_strength),
        "target_consistency": float(target_consistency),
        "target_smoothness": float(target_smoothness),
        "target_retention": float(target_retention),
        "descriptive_target": float(np.clip(descriptive_target, 0.0, 1.0)),
        "target_status": "ok",
    }


def build_containment_target(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict:
    open_arr, high_arr, low_arr, close_arr = _coerce_ohlc_arrays(
        open_=open_,
        high=high,
        low=low,
        close=close,
    )

    realized_range, close_pos = _compute_realized_range_and_close_pos(
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )

    containment_displacement = 1.0 - np.clip(
        abs(float(close_arr[-1] - open_arr[0])) / (realized_range + 1e-12),
        0.0,
        1.0,
    )

    upper_share = float(np.mean(close_pos >= (2.0 / 3.0)))
    lower_share = float(np.mean(close_pos <= (1.0 / 3.0)))
    containment_edge_balance = float(np.clip(2.0 * min(upper_share, lower_share), 0.0, 1.0))

    containment_inside_share = float(np.mean((close_pos >= 0.05) & (close_pos <= 0.95)))

    path_length = float(np.sum(np.abs(np.diff(close_arr))))
    path_waste = float(np.clip(path_length / (realized_range + 1e-12), 0.0, 4.0) / 4.0)
    containment_path_efficiency = float(1.0 - path_waste)

    containment_target = (
        0.30 * containment_displacement
        + 0.30 * containment_edge_balance
        + 0.25 * containment_inside_share
        + 0.15 * containment_path_efficiency
    )

    return {
        "containment_displacement": float(containment_displacement),
        "containment_edge_balance": containment_edge_balance,
        "containment_inside_share": containment_inside_share,
        "containment_path_efficiency": containment_path_efficiency,
        "containment_target": float(np.clip(containment_target, 0.0, 1.0)),
        "containment_status": "ok",
    }


def build_containment_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict:
    _, high_arr, low_arr, close_arr = _coerce_ohlc_arrays(
        open_=open_,
        high=high,
        low=low,
        close=close,
    )

    realized_range, close_pos = _compute_realized_range_and_close_pos(
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )

    lower_overshoot = np.clip(0.05 - close_pos, 0.0, None)
    upper_overshoot = np.clip(close_pos - 0.95, 0.0, None)
    containment_overshoot_ratio = float(np.mean(lower_overshoot + upper_overshoot))

    block_indexes = [
        block
        for block in np.array_split(np.arange(close_arr.size), min(4, close_arr.size))
        if block.size
    ]
    block_ranges = [
        float(high_arr[idx].max() - low_arr[idx].min()) / (realized_range + 1e-12)
        for idx in block_indexes
    ]
    containment_range_stability = float(
        1.0 - np.clip(np.std(block_ranges, ddof=0), 0.0, 1.0)
    )

    mid = 0.5 * float(high_arr.max() + low_arr.min())
    side = np.sign(close_arr - mid)
    side = side[side != 0]
    crosses = np.count_nonzero(side[1:] != side[:-1]) if side.size > 1 else 0
    containment_mid_cross_count = float(crosses / max(side.size - 1, 1))

    swings = _collapse_directional_swings(close_arr)
    positive_total = sum(magnitude for sign, magnitude in swings if sign > 0.0)
    negative_total = sum(magnitude for sign, magnitude in swings if sign < 0.0)
    if positive_total <= 0.0 or negative_total <= 0.0:
        containment_swing_symmetry = 0.0
    else:
        containment_swing_symmetry = float(
            1.0
            - abs(positive_total - negative_total)
            / (positive_total + negative_total + 1e-12)
        )

    return {
        "containment_overshoot_ratio": containment_overshoot_ratio,
        "containment_range_stability": containment_range_stability,
        "containment_mid_cross_count": containment_mid_cross_count,
        "containment_swing_symmetry": containment_swing_symmetry,
    }


def build_containment_expansion_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    session_name: str,
) -> dict:
    open_arr, high_arr, low_arr, close_arr = _coerce_ohlc_arrays(
        open_=open_,
        high=high,
        low=low,
        close=close,
    )
    volume_arr = _coerce_volume_array(volume=volume, expected_shape=close_arr.shape)

    realized_range, _ = _compute_realized_range_and_close_pos(
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )

    ib_bars = min(INITIAL_BALANCE_BARS.get(session_name, close_arr.size), close_arr.size)
    ib_high = float(high_arr[:ib_bars].max())
    ib_low = float(low_arr[:ib_bars].min())
    ib_range = ib_high - ib_low
    ext_up = max(0.0, float(high_arr.max()) - ib_high) / (ib_range + 1e-12)
    ext_dn = max(0.0, ib_low - float(low_arr.min())) / (ib_range + 1e-12)
    containment_ib_extension_ratio = float(ext_up + ext_dn)
    containment_ib_asymmetry = float(
        abs(ext_up - ext_dn) / (ext_up + ext_dn + 1e-12)
    )

    window = min(10, close_arr.size)
    close_series = pd.Series(close_arr)
    rolling_mean = close_series.rolling(window=window, min_periods=window).mean()
    rolling_std = close_series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = rolling_mean + 2.0 * rolling_std
    lower = rolling_mean - 2.0 * rolling_std
    bandwidth = (upper - lower) / (rolling_mean.abs() + 1e-12)
    bandwidth_valid = bandwidth.dropna()
    if bandwidth_valid.empty:
        containment_bandwidth_squeeze = 1.0
    else:
        containment_bandwidth_squeeze = float(1.0 / (1.0 + bandwidth_valid.mean()))

    typical_price = (high_arr + low_arr + close_arr) / 3.0
    cumulative_volume = np.cumsum(volume_arr)
    vwap = np.cumsum(typical_price * volume_arr) / (cumulative_volume + 1e-12)
    vwap_distance = np.abs(close_arr - vwap) / (realized_range + 1e-12)
    containment_vwap_acceptance = float(1.0 - np.clip(np.mean(vwap_distance), 0.0, 1.0))

    bar_range = high_arr - low_arr
    upper_tail = (high_arr - np.maximum(open_arr, close_arr)) / (bar_range + 1e-12)
    lower_tail = (np.minimum(open_arr, close_arr) - low_arr) / (bar_range + 1e-12)
    upper_reject_share = float(np.mean(upper_tail >= 0.4))
    lower_reject_share = float(np.mean(lower_tail >= 0.4))
    containment_excess_rejection = float(2.0 * min(upper_reject_share, lower_reject_share))

    return {
        "containment_ib_extension_ratio": containment_ib_extension_ratio,
        "containment_ib_asymmetry": containment_ib_asymmetry,
        "containment_bandwidth_squeeze": containment_bandwidth_squeeze,
        "containment_vwap_acceptance": containment_vwap_acceptance,
        "containment_excess_rejection": containment_excess_rejection,
    }
