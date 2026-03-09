# features/trend/di_indicators.py
"""
Directional Indicators (+DI and -DI) Calculation Module

Computes the raw directional indicators from OHLC price data.

Components:
    - TR: True Range
    - +DM: Positive Directional Movement = max(current_high - prev_high, 0)
    - -DM: Negative Directional Movement = max(prev_low - current_low, 0)
    - +DI: Positive Directional Indicator = (Smoothed +DM / Smoothed TR) * 100
    - -DI: Negative Directional Indicator = (Smoothed -DM / Smoothed TR) * 100

Notes:
    - This is the foundation for ADX/DX calculations
    - First valid values appear at bar `period`
    - Supports Wilder's smoothing (default) or EMA
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Literal


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Calculate True Range (TR).
    
    TR = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
        
    Returns
    -------
    pd.Series
        True Range values
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def directional_movement(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Positive (+DM) and Negative (-DM) Directional Movement.
    
    +DM = max(current_high - prev_high, 0) if > -DM, else 0
    -DM = max(prev_low - current_low, 0) if > +DM, else 0
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (+DM, -DM) tuple
    """
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )
    
    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )
    
    return pd.Series(plus_dm, index=high.index), pd.Series(minus_dm, index=high.index)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing method (also called SMMA or Modified Moving Average).
    
    This is the original smoothing method used by Welles Wilder.
    Unlike standard EMA:
        - Uses alpha = 1/period (not 2/(period+1))
        - First smoothed value is the SUM of first `period` values
        - Subsequent: smoothed[i] = smoothed[i-1] - smoothed[i-1]/period + value[i]
    
    Parameters
    ----------
    series : pd.Series
        Input series to smooth
    period : int
        Smoothing period
        
    Returns
    -------
    pd.Series
        Wilder smoothed series (as running sum, not average)
    """
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)
    
    # Find first valid (non-NaN) index
    first_valid = 0
    while first_valid < n and np.isnan(values[first_valid]):
        first_valid += 1
    
    # Need at least `period` valid values
    if first_valid + period > n:
        return pd.Series(result, index=series.index)
    
    # First smoothed value = sum of first `period` valid values
    result[first_valid + period - 1] = np.nansum(values[first_valid:first_valid + period])
    
    # Apply Wilder's smoothing formula
    for i in range(first_valid + period, n):
        result[i] = result[i-1] - (result[i-1] / period) + values[i]
    
    return pd.Series(result, index=series.index)


def ema_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA) smoothing.
    
    Uses standard EMA formula:
        - alpha = 2 / (period + 1)
        - First value = SMA of first `period` values
        - Subsequent: ema[i] = value[i] * alpha + ema[i-1] * (1 - alpha)
    
    Parameters
    ----------
    series : pd.Series
        Input series to smooth
    period : int
        Smoothing period
        
    Returns
    -------
    pd.Series
        EMA smoothed series (as running sum form for consistency)
    """
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)
    
    # Find first valid (non-NaN) index
    first_valid = 0
    while first_valid < n and np.isnan(values[first_valid]):
        first_valid += 1
    
    # Need at least `period` valid values
    if first_valid + period > n:
        return pd.Series(result, index=series.index)
    
    # First smoothed value = sum of first `period` valid values (sum form)
    result[first_valid + period - 1] = np.nansum(values[first_valid:first_valid + period])
    
    # Apply EMA smoothing formula (adapted to sum form)
    alpha = 2.0 / (period + 1)
    for i in range(first_valid + period, n):
        result[i] = values[i] * alpha * period + result[i-1] * (1 - alpha)
    
    return pd.Series(result, index=series.index)


def calculate_di(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smoothing: Literal["wilder", "ema"] = "wilder"
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Directional Indicators (+DI and -DI).
    
    +DI = (Smoothed +DM / Smoothed TR) * 100
    -DI = (Smoothed -DM / Smoothed TR) * 100
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default 14
        Smoothing period
    smoothing : {"wilder", "ema"}, default "wilder"
        Smoothing method to use
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (+DI, -DI) tuple
        
    Notes
    -----
    First valid values appear at index `period - 1` (0-indexed),
    which corresponds to bar number `period`.
    """
    # Step 1: Calculate True Range
    tr = true_range(high, low, close)
    
    # Step 2: Calculate +DM and -DM
    plus_dm, minus_dm = directional_movement(high, low)
    
    # Step 3: Smooth TR, +DM, -DM
    smooth_fn = wilder_smooth if smoothing == "wilder" else ema_smooth
    
    smoothed_tr = smooth_fn(tr, period)
    smoothed_plus_dm = smooth_fn(plus_dm, period)
    smoothed_minus_dm = smooth_fn(minus_dm, period)
    
    # Step 4: Calculate +DI and -DI
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100
    
    # Handle division by zero
    plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan)
    
    return plus_di, minus_di


def calculate_di_from_df(
    df: pd.DataFrame,
    period: int = 14,
    smoothing: Literal["wilder", "ema"] = "wilder",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> Tuple[pd.Series, pd.Series]:
    """
    Convenience function to calculate +DI and -DI from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC price data
    period : int, default 14
        Smoothing period
    smoothing : {"wilder", "ema"}, default "wilder"
        Smoothing method to use
    high_col : str, default "high"
        Column name for high prices
    low_col : str, default "low"
        Column name for low prices
    close_col : str, default "close"
        Column name for close prices
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (+DI, -DI) tuple
    """
    return calculate_di(
        df[high_col],
        df[low_col],
        df[close_col],
        period=period,
        smoothing=smoothing
    )


def calculate_di_full(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smoothing: Literal["wilder", "ema"] = "wilder"
) -> pd.DataFrame:
    """
    Calculate +DI and -DI with all intermediate components.
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default 14
        Smoothing period
    smoothing : {"wilder", "ema"}, default "wilder"
        Smoothing method to use
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: TR, +DM, -DM, Smoothed_TR, Smoothed_+DM, Smoothed_-DM, +DI, -DI
    """
    # Step 1: True Range
    tr = true_range(high, low, close)
    
    # Step 2: Directional Movement
    plus_dm, minus_dm = directional_movement(high, low)
    
    # Step 3: Smooth components
    smooth_fn = wilder_smooth if smoothing == "wilder" else ema_smooth
    
    smoothed_tr = smooth_fn(tr, period)
    smoothed_plus_dm = smooth_fn(plus_dm, period)
    smoothed_minus_dm = smooth_fn(minus_dm, period)
    
    # Step 4: Calculate +DI and -DI
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100
    
    # Handle division by zero
    plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan)
    
    return pd.DataFrame({
        "TR": tr,
        "+DM": plus_dm,
        "-DM": minus_dm,
        "Smoothed_TR": smoothed_tr,
        "Smoothed_+DM": smoothed_plus_dm,
        "Smoothed_-DM": smoothed_minus_dm,
        "+DI": plus_di,
        "-DI": minus_di
    }, index=high.index)


def calculate_di_full_from_df(
    df: pd.DataFrame,
    period: int = 14,
    smoothing: Literal["wilder", "ema"] = "wilder",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pd.DataFrame:
    """
    Convenience function to calculate +DI and -DI with all components from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC price data
    period : int, default 14
        Smoothing period
    smoothing : {"wilder", "ema"}, default "wilder"
        Smoothing method to use
    high_col : str, default "high"
        Column name for high prices
    low_col : str, default "low"
        Column name for low prices
    close_col : str, default "close"
        Column name for close prices
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all DI components
    """
    return calculate_di_full(
        df[high_col],
        df[low_col],
        df[close_col],
        period=period,
        smoothing=smoothing
    )

