# features/trend/adx_calc.py
"""
ADX (Average Directional Index) Calculation Module

Computes ADX indicator and its components from OHLC price data.

Components:
    - TR: True Range
    - +DM: Positive Directional Movement  
    - -DM: Negative Directional Movement
    - ATR: Smoothed True Range (Average True Range)
    - +DI: Positive Directional Indicator = (Smoothed +DM / ATR) * 100
    - -DI: Negative Directional Indicator = (Smoothed -DM / ATR) * 100
    - DX: Directional Movement Index = |+DI - -DI| / (+DI + -DI) * 100
    - ADX: Average Directional Index = Smoothed DX
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from .di_indicators import (
    true_range,
    directional_movement,
    wilder_smooth,
)


def wilder_smooth_avg(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing in average form (for DX -> ADX).
    
    Formula:
        First value = mean of first `period` values
        Subsequent: avg[i] = avg[i-1] * (period-1)/period + value[i]/period
    
    Parameters
    ----------
    series : pd.Series
        Input series to smooth
    period : int
        Smoothing period
        
    Returns
    -------
    pd.Series
        Wilder smoothed average
    """
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)
    
    # Find first valid (non-NaN) index
    first_valid = 0
    while first_valid < n and np.isnan(values[first_valid]):
        first_valid += 1
    
    # Need at least `period` valid values after first_valid
    if first_valid + period > n:
        return pd.Series(result, index=series.index)
    
    # First smoothed value = mean of first `period` valid values
    result[first_valid + period - 1] = np.nanmean(values[first_valid:first_valid + period])
    
    # Apply Wilder's smoothing formula (average form)
    alpha = 1.0 / period
    for i in range(first_valid + period, n):
        result[i] = result[i-1] * (1 - alpha) + values[i] * alpha
    
    return pd.Series(result, index=series.index)


def directional_indicators(
    plus_dm_smooth: pd.Series, 
    minus_dm_smooth: pd.Series, 
    atr: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Directional Indicators (+DI, -DI).
    
    +DI = (Smoothed +DM / ATR) * 100
    -DI = (Smoothed -DM / ATR) * 100
    
    Parameters
    ----------
    plus_dm_smooth : pd.Series
        Smoothed Positive Directional Movement
    minus_dm_smooth : pd.Series
        Smoothed Negative Directional Movement
    atr : pd.Series
        Average True Range (smoothed TR)
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (+DI, -DI) tuple
    """
    plus_di = (plus_dm_smooth / atr) * 100
    minus_di = (minus_dm_smooth / atr) * 100
    
    # Handle division by zero
    plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan)
    
    return plus_di, minus_di


def directional_index(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    """
    Calculate Directional Movement Index (DX).
    
    DX = |+DI - -DI| / (+DI + -DI) * 100
    
    Parameters
    ----------
    plus_di : pd.Series
        Positive Directional Indicator
    minus_di : pd.Series
        Negative Directional Indicator
        
    Returns
    -------
    pd.Series
        DX values
    """
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    
    dx = (di_diff / di_sum) * 100
    
    # Handle division by zero
    dx = dx.replace([np.inf, -np.inf], np.nan)
    
    return dx


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength regardless of direction.
    Values typically interpreted as:
        - 0-25: Weak/absent trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices  
    close : pd.Series
        Close prices
    period : int, default 14
        Smoothing period for ADX calculation
        
    Returns
    -------
    pd.Series
        ADX values
    """
    # Step 1: Calculate True Range
    tr = true_range(high, low, close)
    
    # Step 2: Calculate +DM and -DM
    plus_dm, minus_dm = directional_movement(high, low)
    
    # Step 3: Smooth TR, +DM, -DM using Wilder's method (sum form)
    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Step 4: Calculate +DI and -DI
    plus_di, minus_di = directional_indicators(plus_dm_smooth, minus_dm_smooth, atr)
    
    # Step 5: Calculate DX
    dx = directional_index(plus_di, minus_di)
    
    # Step 6: Smooth DX to get ADX using Wilder's method (average form)
    adx = wilder_smooth_avg(dx, period)
    
    return adx


def calculate_adx_full(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate ADX with all intermediate components.
    
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
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: TR, +DM, -DM, ATR, +DI, -DI, DX, ADX
    """
    # Step 1: True Range
    tr = true_range(high, low, close)
    
    # Step 2: Directional Movement
    plus_dm, minus_dm = directional_movement(high, low)
    
    # Step 3: Wilder smoothing (sum form for TR/DM)
    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Step 4: Directional Indicators
    plus_di, minus_di = directional_indicators(plus_dm_smooth, minus_dm_smooth, atr)
    
    # Step 5: DX
    dx = directional_index(plus_di, minus_di)
    
    # Step 6: ADX (Wilder smoothing average form)
    adx = wilder_smooth_avg(dx, period)
    
    return pd.DataFrame({
        "TR": tr,
        "+DM": plus_dm,
        "-DM": minus_dm,
        "ATR": atr,
        "+DI": plus_di,
        "-DI": minus_di,
        "DX": dx,
        "ADX": adx
    }, index=high.index)


def calculate_dx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Directional Index (DX) without ADX smoothing.
    
    DX is more responsive but volatile compared to ADX.
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices  
    close : pd.Series
        Close prices
    period : int, default 14
        Smoothing period for TR/DM components
        
    Returns
    -------
    pd.Series
        DX values (unsmoothed)
    """
    tr = true_range(high, low, close)
    plus_dm, minus_dm = directional_movement(high, low)
    
    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    plus_di, minus_di = directional_indicators(plus_dm_smooth, minus_dm_smooth, atr)
    dx = directional_index(plus_di, minus_di)
    
    return dx


def calculate_dx_from_df(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pd.Series:
    """
    Convenience function to calculate DX from a DataFrame.
    """
    return calculate_dx(
        df[high_col],
        df[low_col],
        df[close_col],
        period=period
    )


def calculate_dx_full_from_df(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pd.DataFrame:
    """
    Calculate DX with all components (excludes ADX).
    """
    result = calculate_adx_full(
        df[high_col],
        df[low_col],
        df[close_col],
        period=period
    )
    return result.drop(columns=["ADX"])


def calculate_adx_from_df(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pd.Series:
    """
    Convenience function to calculate ADX from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC price data
    period : int, default 14
        Smoothing period
    high_col : str, default "high"
        Column name for high prices
    low_col : str, default "low"
        Column name for low prices
    close_col : str, default "close"
        Column name for close prices
        
    Returns
    -------
    pd.Series
        ADX values
    """
    return calculate_adx(
        df[high_col],
        df[low_col],
        df[close_col],
        period=period
    )


def calculate_adx_full_from_df(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pd.DataFrame:
    """
    Convenience function to calculate ADX with all components from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC price data
    period : int, default 14
        Smoothing period
    high_col : str, default "high"
        Column name for high prices
    low_col : str, default "low"
        Column name for low prices
    close_col : str, default "close"
        Column name for close prices
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all ADX components
    """
    return calculate_adx_full(
        df[high_col],
        df[low_col],
        df[close_col],
        period=period
    )
