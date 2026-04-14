from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


Direction = Literal["UP", "DOWN", "NEUTRAL"]


@dataclass(frozen=True)
class EfficiencyRatioResult:
    """Path-efficiency summary for a realized price sequence."""

    efficiency_ratio: float
    net_change: float
    path_length: float
    direction: Direction
    n_prices: int


def _coerce_prices(prices: Sequence[float]) -> np.ndarray:
    array = np.asarray(prices, dtype=float)

    if array.ndim != 1:
        raise ValueError("Efficiency Ratio expects a one-dimensional price sequence.")
    if array.size < 2:
        raise ValueError("Efficiency Ratio requires at least two prices.")
    if np.isnan(array).any():
        raise ValueError("Efficiency Ratio prices cannot contain NaN values.")

    return array


def _direction_from_change(net_change: float) -> Direction:
    if net_change > 0:
        return "UP"
    if net_change < 0:
        return "DOWN"
    return "NEUTRAL"


def calculate_efficiency_ratio(prices: Sequence[float]) -> EfficiencyRatioResult:
    """
    Calculate Kaufman Efficiency Ratio from a price path.

    ER = |final - initial| / sum(|delta price|)
    """

    price_array = _coerce_prices(prices)
    deltas = np.diff(price_array)
    net_change = float(price_array[-1] - price_array[0])
    path_length = float(np.abs(deltas).sum())

    if path_length == 0.0:
        efficiency_ratio = 0.0
    else:
        efficiency_ratio = float(np.clip(abs(net_change) / path_length, 0.0, 1.0))

    return EfficiencyRatioResult(
        efficiency_ratio=efficiency_ratio,
        net_change=net_change,
        path_length=path_length,
        direction=_direction_from_change(net_change),
        n_prices=int(price_array.size),
    )


def analyze_efficiency_ratio(df: pd.DataFrame, price_col: str = "close") -> EfficiencyRatioResult:
    """Calculate Efficiency Ratio from a DataFrame column."""

    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")

    return calculate_efficiency_ratio(df[price_col].to_numpy())
