from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


Interpretation = Literal["persistent", "mean_reverting", "neutral"]
_LOG_EPS = 1e-12


@dataclass(frozen=True)
class VarianceRatioResult:
    """Variance-ratio summary for a realized return sequence."""

    variance_ratio: float
    log_variance_ratio: float
    lag: int
    one_period_variance: float
    multi_period_variance: float
    n_returns: int
    interpretation: Interpretation


def _coerce_returns(returns: Sequence[float], lag: int) -> np.ndarray:
    array = np.asarray(returns, dtype=float)

    if lag < 2:
        raise ValueError("Variance Ratio lag must be at least 2.")
    if array.ndim != 1:
        raise ValueError("Variance Ratio expects a one-dimensional return sequence.")
    if array.size < lag + 1:
        raise ValueError("Variance Ratio requires at least lag + 1 returns.")
    if np.isnan(array).any():
        raise ValueError("Variance Ratio returns cannot contain NaN values.")

    return array


def _coerce_prices(prices: Sequence[float]) -> np.ndarray:
    array = np.asarray(prices, dtype=float)

    if array.ndim != 1:
        raise ValueError("Variance Ratio expects a one-dimensional price sequence.")
    if array.size < 3:
        raise ValueError("Variance Ratio requires at least three prices.")
    if np.isnan(array).any():
        raise ValueError("Variance Ratio prices cannot contain NaN values.")
    if np.any(array <= 0):
        raise ValueError("Variance Ratio requires strictly positive prices for log returns.")

    return array


def _classify_variance_ratio(variance_ratio: float, tolerance: float = 0.05) -> Interpretation:
    if variance_ratio > 1.0 + tolerance:
        return "persistent"
    if variance_ratio < 1.0 - tolerance:
        return "mean_reverting"
    return "neutral"


def calculate_variance_ratio_from_returns(
    returns: Sequence[float],
    lag: int = 4,
) -> VarianceRatioResult:
    """
    Calculate an overlapping variance ratio on demeaned returns.

    VR(q) = Var(sum_{i=0}^{q-1} r_{t+i}) / (q * Var(r_t))
    """

    return_array = _coerce_returns(returns, lag=lag)
    mean_return = float(return_array.mean())
    demeaned_returns = return_array - mean_return
    one_period_variance = float(np.var(demeaned_returns, ddof=1))

    if np.isclose(one_period_variance, 0.0):
        return VarianceRatioResult(
            variance_ratio=1.0,
            log_variance_ratio=0.0,
            lag=lag,
            one_period_variance=0.0,
            multi_period_variance=0.0,
            n_returns=int(return_array.size),
            interpretation="neutral",
        )

    aggregate_returns = np.array(
        [return_array[start : start + lag].sum() for start in range(return_array.size - lag + 1)],
        dtype=float,
    )
    demeaned_aggregate_returns = aggregate_returns - (lag * mean_return)
    multi_period_variance = float(np.var(demeaned_aggregate_returns, ddof=1))
    variance_ratio = max(multi_period_variance / (lag * one_period_variance), 0.0)
    log_variance_ratio = float(np.log(max(variance_ratio, _LOG_EPS)))

    return VarianceRatioResult(
        variance_ratio=float(variance_ratio),
        log_variance_ratio=log_variance_ratio,
        lag=lag,
        one_period_variance=one_period_variance,
        multi_period_variance=multi_period_variance,
        n_returns=int(return_array.size),
        interpretation=_classify_variance_ratio(variance_ratio),
    )


def calculate_variance_ratio(prices: Sequence[float], lag: int = 4) -> VarianceRatioResult:
    """Calculate Variance Ratio from a price path using log returns."""

    price_array = _coerce_prices(prices)
    log_returns = np.diff(np.log(price_array))
    return calculate_variance_ratio_from_returns(log_returns, lag=lag)


def analyze_variance_ratio(
    df: pd.DataFrame,
    price_col: str = "close",
    lag: int = 4,
) -> VarianceRatioResult:
    """Calculate Variance Ratio from a DataFrame column."""

    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")

    return calculate_variance_ratio(df[price_col].to_numpy(), lag=lag)
