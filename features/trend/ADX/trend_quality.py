"""
Trend Quality Score Calculator

Combines ADX/DX, DI persistence, and DI crossovers into a single
trend quality score (0-1) that distinguishes clean trends from chop.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .adx_calc import calculate_adx_from_df, calculate_dx_from_df
from .di_indicators import calculate_di_from_df
from .di_persistence import calculate_di_persistence
from .di_crossovers import calculate_crossover_penalty


# ============================================================================
# CONFIGURATION
# ============================================================================

# Window-specific configurations
WINDOW_CONFIGS = {
    "1pm-3pm": {
        "bar_size": "5m",
        "period": 12,
        "use_dx": False,  # Use ADX
        "weights": {
            "strength": 0.50,
            "persistence": 0.30,
            "crossover": 0.20
        },
        "min_bars_required": 20  # period + buffer
    },
    "3pm-3:50pm": {
        "bar_size": "2m",
        "period": 12,
        "use_dx": False,  # Use ADX
        "weights": {
            "strength": 0.50,
            "persistence": 0.30,
            "crossover": 0.20
        },
        "min_bars_required": 20
    },
    "3:50pm-4pm": {
        "bar_size": "1m",
        "period": 5,
        "use_dx": True,  # Use DX (not enough bars for ADX)
        "weights": {
            "strength": 0.40,  # Slightly higher - EOD moves matter
            "persistence": 0.30,
            "crossover": 0.30
        },
        "min_bars_required": 7  # period + small buffer
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_strength(strength: float) -> float:
    """
    Normalize ADX/DX value from 0-100 scale to 0-1 scale.
    
    Args:
        strength: ADX or DX value (0-100)
        
    Returns:
        Normalized value (0-1)
    """
    return strength / 100.0


def validate_bars(bars: pd.DataFrame, min_required: int) -> bool:
    """
    Check if we have sufficient bars for calculation.
    
    Args:
        bars: Price data
        min_required: Minimum bars needed
        
    Returns:
        True if sufficient data, False otherwise
    """
    if bars is None or len(bars) < min_required:
        return False
    
    # Check required columns exist
    required_cols = ['high', 'low', 'close']
    if not all(col in bars.columns for col in required_cols):
        return False
    
    return True


def get_valid_data_window(
    strength: np.ndarray,
    plus_di: np.ndarray,
    minus_di: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract only valid (non-NaN) data for scoring.
    
    Since ADX/DX have a warm-up period, we only use bars where
    all indicators are valid.
    
    Args:
        strength: ADX or DX array
        plus_di: +DI array
        minus_di: -DI array
        
    Returns:
        Tuple of (valid_strength, valid_plus_di, valid_minus_di)
    """
    # Find first valid index (where all are non-NaN)
    valid_mask = ~(np.isnan(strength) | np.isnan(plus_di) | np.isnan(minus_di))
    
    return (
        strength[valid_mask],
        plus_di[valid_mask],
        minus_di[valid_mask]
    )


# ============================================================================
# MAIN CALCULATION FUNCTION
# ============================================================================

def calculate_trend_quality(
    bars: pd.DataFrame,
    window_name: str,
    config: Dict = None
) -> Dict:
    """
    Calculate trend quality score for a given time window.
    
    This combines:
    - Directional strength (ADX or DX)
    - DI persistence (trend consistency)
    - Crossover penalty (chop filter)
    
    Into a single score where:
    - 1.0 = Clean, strong trend
    - 0.5 = Moderate/unclear
    - 0.0 = Choppy or consolidating
    
    Args:
        bars: DataFrame with OHLC data
        window_name: One of "1pm-3pm", "3pm-3:50pm", "3:50pm-4pm"
        config: Optional custom config (overrides defaults)
        
    Returns:
        Dict containing:
        - quality_score: Final trend quality (0-1)
        - components: Breakdown of individual scores
        - metadata: Additional context (bar count, etc.)
    """
    # Get configuration
    if config is None:
        if window_name not in WINDOW_CONFIGS:
            raise ValueError(f"Unknown window: {window_name}")
        config = WINDOW_CONFIGS[window_name]
    
    # Validate sufficient data
    if not validate_bars(bars, config['min_bars_required']):
        return {
            'quality_score': None,
            'components': None,
            'metadata': {'error': 'Insufficient data'}
        }
    
    # ========================================================================
    # STEP 1: Calculate strength metric (ADX or DX)
    # ========================================================================
    if config['use_dx']:
        strength = calculate_dx_from_df(bars, period=config['period'])
    else:
        strength = calculate_adx_from_df(bars, period=config['period'])
    
    # ========================================================================
    # STEP 2: Calculate DI indicators
    # ========================================================================
    plus_di, minus_di = calculate_di_from_df(
        bars,
        period=config['period']
    )
    
    # ========================================================================
    # STEP 3: Extract valid data window (skip NaN warm-up period)
    # ========================================================================
    # Convert pandas Series to numpy arrays for processing
    strength_array = strength.values
    plus_di_array = plus_di.values
    minus_di_array = minus_di.values
    
    valid_strength, valid_plus_di, valid_minus_di = get_valid_data_window(
        strength_array, plus_di_array, minus_di_array
    )
    
    if len(valid_strength) == 0:
        return {
            'quality_score': None,
            'components': None,
            'metadata': {'error': 'No valid data after warm-up'}
        }
    
    # ========================================================================
    # STEP 4: Calculate component scores
    # ========================================================================
    
    # Use the most recent (current) strength value
    strength_score = normalize_strength(valid_strength[-1])
    
    # Persistence over the valid window
    persistence_score = calculate_di_persistence(valid_plus_di, valid_minus_di)
    
    # Crossover penalty over the valid window
    crossover_score = calculate_crossover_penalty(valid_plus_di, valid_minus_di)
    
    # ========================================================================
    # STEP 5: Combine into weighted quality score
    # ========================================================================
    weights = config['weights']
    
    quality_score = (
        weights['strength'] * strength_score +
        weights['persistence'] * persistence_score +
        weights['crossover'] * crossover_score
    )
    
    # Ensure score is in valid range
    quality_score = np.clip(quality_score, 0.0, 1.0)
    
    # ========================================================================
    # STEP 6: Package results
    # ========================================================================
    return {
        'quality_score': quality_score,
        'components': {
            'strength': strength_score,
            'persistence': persistence_score,
            'crossover': crossover_score,
            'strength_raw': valid_strength[-1],  # Keep raw ADX/DX value
            'dominant_di': 'plus' if valid_plus_di[-1] > valid_minus_di[-1] else 'minus'
        },
        'metadata': {
            'window': window_name,
            'total_bars': len(bars),
            'valid_bars': len(valid_strength),
            'period': config['period'],
            'indicator_type': 'DX' if config['use_dx'] else 'ADX'
        }
    }


# ============================================================================
# BATCH PROCESSING (Optional)
# ============================================================================

def calculate_all_windows(
    bars_dict: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Calculate trend quality for all windows at once.
    
    Args:
        bars_dict: Dictionary mapping window names to bar data
                   e.g., {"1pm-3pm": df1, "3pm-3:50pm": df2, ...}
        
    Returns:
        Dictionary mapping window names to results
    """
    results = {}
    
    for window_name, bars in bars_dict.items():
        results[window_name] = calculate_trend_quality(bars, window_name)
    
    return results


# ============================================================================
# UTILITY: Update weights for optimization
# ============================================================================

def update_window_weights(
    window_name: str,
    new_weights: Dict[str, float]
) -> None:
    """
    Update weights for a specific window (for backtesting/optimization).
    
    Args:
        window_name: Window to update
        new_weights: New weight values (must sum to 1.0)
    """
    if abs(sum(new_weights.values()) - 1.0) > 0.001:
        raise ValueError("Weights must sum to 1.0")
    
    WINDOW_CONFIGS[window_name]['weights'] = new_weights


# ============================================================================
# EXAMPLE USAGE (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example: Load your bar data
    # bars = load_bars_for_window("3:50pm-4pm")
    
    # Calculate trend quality
    # result = calculate_trend_quality(bars, "3:50pm-4pm")
    
    # print(f"Trend Quality: {result['quality_score']:.3f}")
    # print(f"Components: {result['components']}")
    # print(f"Metadata: {result['metadata']}")
    
    pass