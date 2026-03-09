"""
Swing Point Density Detection for Historical Trend/Consolidation Analysis

Detects choppy vs smooth price action by counting local highs/lows.
High density = consolidation/chop, low density = trending.
"""

from typing import List, Dict, Tuple
from datetime import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Time windows for intraday analysis
WINDOWS = {
    "early_afternoon": {
        "start": time(13, 0),  # 1:00 PM
        "end": time(15, 0),    # 3:00 PM
        "name": "Early Afternoon (1pm-3pm)"
    },
    "late_afternoon": {
        "start": time(15, 0),  # 3:00 PM
        "end": time(15, 50),   # 3:50 PM
        "name": "Late Afternoon (3pm-3:50pm)"
    },
    "close": {
        "start": time(15, 50), # 3:50 PM
        "end": time(16, 0),    # 4:00 PM
        "name": "Close (3:50pm-4pm)"
    }
}

# Swing density thresholds (raw count)
THRESHOLDS = {
    "trending": 5,      # < 5 swings = smooth/trending
    "chop": 8           # > 8 swings = choppy/consolidation
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def detect_swing_highs(bars: List[Dict]) -> List[int]:
    """
    Detect swing highs where high[i-1] < high[i] > high[i+1]
    
    Args:
        bars: List of OHLC bars with 'high' key
        
    Returns:
        List of indices where swing highs occur
    """
    swing_highs = []
    
    for i in range(1, len(bars) - 1):
        if bars[i-1]['high'] < bars[i]['high'] > bars[i+1]['high']:
            swing_highs.append(i)
    
    return swing_highs


def detect_swing_lows(bars: List[Dict]) -> List[int]:
    """
    Detect swing lows where low[i-1] > low[i] < low[i+1]
    
    Args:
        bars: List of OHLC bars with 'low' key
        
    Returns:
        List of indices where swing lows occur
    """
    swing_lows = []
    
    for i in range(1, len(bars) - 1):
        if bars[i-1]['low'] > bars[i]['low'] < bars[i+1]['low']:
            swing_lows.append(i)
    
    return swing_lows


def filter_bars_by_time(bars: List[Dict], start_time: time, end_time: time) -> List[Dict]:
    """
    Filter bars to those within the specified time window.
    
    Args:
        bars: List of OHLC bars with 'time' or 'timestamp' key
        start_time: Start of window (inclusive)
        end_time: End of window (exclusive)
        
    Returns:
        Filtered list of bars within time window
    """
    filtered = []
    
    for bar in bars:
        # Handle both 'time' and 'timestamp' keys
        bar_time = bar.get('time') or bar.get('timestamp')
        
        # Extract time component if datetime object
        if hasattr(bar_time, 'time'):
            bar_time = bar_time.time()
        
        if start_time <= bar_time < end_time:
            filtered.append(bar)
    
    return filtered


def classify_density(count: int) -> str:
    """
    Classify market state based on swing point count.
    
    Args:
        count: Total swing points (highs + lows)
        
    Returns:
        Classification string: 'trending', 'mixed', or 'chop'
    """
    if count < THRESHOLDS['trending']:
        return 'trending'
    elif count > THRESHOLDS['chop']:
        return 'chop'
    else:
        return 'mixed'


def get_swing_density(bars: List[Dict], start_time: time, end_time: time) -> Dict:
    """
    Calculate swing point density for a specific time window.
    
    Args:
        bars: List of OHLC bars
        start_time: Start of analysis window
        end_time: End of analysis window
        
    Returns:
        Dictionary with density analysis results
    """
    # Filter to time window
    window_bars = filter_bars_by_time(bars, start_time, end_time)
    
    if len(window_bars) < 3:
        return {
            'count': 0,
            'swing_highs': [],
            'swing_lows': [],
            'classification': 'insufficient_data',
            'bars_analyzed': len(window_bars)
        }
    
    # Detect swings
    swing_highs = detect_swing_highs(window_bars)
    swing_lows = detect_swing_lows(window_bars)
    
    total_swings = len(swing_highs) + len(swing_lows)
    
    return {
        'count': total_swings,
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'swing_high_count': len(swing_highs),
        'swing_low_count': len(swing_lows),
        'classification': classify_density(total_swings),
        'bars_analyzed': len(window_bars)
    }


def analyze_all_windows(bars: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze swing density for all configured time windows.
    
    Args:
        bars: List of OHLC bars for the entire day
        
    Returns:
        Dictionary with results for each window
    """
    results = {}
    
    for window_name, window_config in WINDOWS.items():
        results[window_name] = {
            **get_swing_density(bars, window_config['start'], window_config['end']),
            'window_name': window_config['name']
        }
    
    return results