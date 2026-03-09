import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Session configurations
SESSIONS = {
    '1pm-3pm': {
        'bar_size': '5min',
        'total_bars': 24,
        'atr_period': 10,
    },
    '3pm-3:50pm': {
        'bar_size': '2min',
        'total_bars': 25,
        'atr_period': 8,
    },
    '3:50-4pm': {
        'bar_size': '1min',
        'total_bars': 10,
        'atr_period': 3,
    }
}


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate ATR (Average True Range) over a rolling window.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period
        
    Returns:
        Series of ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the rolling mean of True Range
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    return atr


def calculate_ratio(df: pd.DataFrame, atr_period: int) -> Tuple[float, float, float]:
    """
    Calculate ATR/Range ratio for a session.
    
    Args:
        df: DataFrame with OHLC data for the session window
        atr_period: Period for ATR calculation
        
    Returns:
        Tuple of (raw_ratio, median_atr, total_range)
    """
    # Calculate total range
    total_range = df['high'].max() - df['low'].min()
    
    # Handle edge case of zero range
    if total_range == 0:
        return np.nan, 0, 0
    
    # Calculate rolling ATR
    atr = calculate_atr(df, atr_period)
    
    # Take median ATR across the window (robust to outliers)
    median_atr = atr.median()
    
    # Calculate ratio
    raw_ratio = median_atr / total_range
    
    return raw_ratio, median_atr, total_range


def analyze_session(df: pd.DataFrame, session_name: str) -> Dict:
    """
    Analyze a trading session and return trend/consolidation signal.
    
    Args:
        df: DataFrame with OHLC data for the session
        session_name: Name of the session (e.g., '1pm-3pm')
        
    Returns:
        Dict with analysis results including raw_ratio, median_atr, total_range, and signal
    """
    config = SESSIONS[session_name]
    
    # Calculate raw ratio
    raw_ratio, median_atr, total_range = calculate_ratio(df, config['atr_period'])
    
    if np.isnan(raw_ratio):
        return {
            'session': session_name,
            'raw_ratio': None,
            'median_atr': None,
            'total_range': None,
            'signal': 'NO_RANGE'
        }
    
    # Lower ratio = trending (ATR small relative to range)
    # Higher ratio = consolidating (ATR large relative to range)
    if raw_ratio < 0.5:
        signal = 'TRENDING'
    elif raw_ratio > 0.8:
        signal = 'CONSOLIDATING'
    else:
        signal = 'NEUTRAL'
    
    return {
        'session': session_name,
        'raw_ratio': round(raw_ratio, 4),
        'median_atr': round(median_atr, 4),
        'total_range': round(total_range, 4),
        'signal': signal
    }
