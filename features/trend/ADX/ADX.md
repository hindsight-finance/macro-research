# ADX (Average Directional Index) Module

## State Detector Integration Status

**Readiness Score: 85% ✅ READY** (Revised Assessment)

**Current Status:**
- ✅ **Sophisticated composite system** (`trend_quality.py`)
- ✅ DataFrame-based API
- ✅ Returns 0-1 quality score (perfect for integration!)
- ✅ Session-specific configurations with smart defaults
- ✅ Combines 3 metrics: ADX/DX + DI persistence + crossover penalty
- ✅ Robust error handling and validation
- ✅ Production-tested on real NQ data
- ⚠️ Function-based (but well-designed, just needs wrapper)
- ❌ Missing `TrendIndicator` interface

**Why This Module is Better Than Initially Assessed:**

The `trend_quality.py` module is already a **mini state detector** for ADX! It intelligently combines:
1. **Strength** (50%): Raw ADX/DX directional movement
2. **Persistence** (30%): How long one DI stays dominant (trend consistency)
3. **Crossovers** (20%): Penalty for DI flip-flops (chop filter)

This is **significantly more sophisticated** than raw ADX and doesn't need major refactoring.

**Required Changes for Integration:**

**RECOMMENDED: Minimal Wrapper Approach (45 min)**
**1. Create Thin `ADXIndicator` Wrapper (45 min)**
   
   Wrap existing `calculate_trend_quality()` function - **no refactoring needed!**
   
   ```python
   class ADXIndicator(TrendIndicator):
       def __init__(self, config=None):
           from features.trend.ADX.trend_quality import calculate_trend_quality
           self._calculate_fn = calculate_trend_quality
           self._last_result = None
       
       def calculate(self, df: pd.DataFrame, session: SessionName = "auto") -> IndicatorResult:
           # Map session names
           session_map = {
               "1pm-3pm": "1pm-3pm",
               "3pm-3:50pm": "3pm-3:50pm",
               "3:50pm-4pm": "3:50pm-4pm",
               "auto": "3pm-3:50pm"
           }
           
           window_name = session_map.get(session, "3pm-3:50pm")
           result = self._calculate_fn(df, window_name)
           
           if result['quality_score'] is None:
               signal = 0.5  # Neutral if insufficient data
           else:
               signal = result['quality_score']  # Already 0-1!
           
           self._last_result = IndicatorResult(
               signal=signal,
               raw_value=result.get('components', {}).get('strength_raw'),
               metadata=result
           )
           return self._last_result
       
       def get_signal(self) -> float:
           return self._last_result.signal if self._last_result else 0.5
       
       @property
       def name(self) -> str:
           return "adx"
   ```

**2. Alternative: Full Class Refactor (120 min)** - *Not Recommended*
   
   If you really want pure class-based architecture:
   
   ```python
   class ADXAnalyzer:
       """Comprehensive ADX-based trend quality analyzer."""
       
       def __init__(self, period: int = 14, use_dx: bool = False):
           self.period = period
           self.use_dx = use_dx
       
       def calculate_trend_quality(self, df: pd.DataFrame, 
                                  weights: Optional[Dict] = None) -> Dict:
           """
           Calculate comprehensive trend quality score.
           
           Returns:
               Dict with quality_score (0-1), components, and metadata
           """
           if weights is None:
               weights = {
                   'strength': 0.50,
                   'persistence': 0.30,
                   'crossover': 0.20
               }
           
           # Calculate ADX/DX
           if self.use_dx:
               strength = calculate_dx_from_df(df, period=self.period)
               strength_value = strength.iloc[-1]
           else:
               strength = calculate_adx_from_df(df, period=self.period)
               strength_value = strength.iloc[-1]
           
           # Normalize to 0-1
           strength_score = strength_value / 100.0
           
           # Calculate DI persistence
           plus_di, minus_di = calculate_di_from_df(df, period=self.period)
           persistence_score = calculate_di_persistence(plus_di, minus_di)
           
           # Calculate crossover penalty
           crossover_penalty = calculate_crossover_penalty(plus_di, minus_di)
           crossover_score = 1.0 - crossover_penalty
           
           # Composite score
           quality_score = (
               weights['strength'] * strength_score +
               weights['persistence'] * persistence_score +
               weights['crossover'] * crossover_score
           )
           
           return {
               'quality_score': quality_score,
               'components': {
                   'strength': strength_score,
                   'strength_raw': strength_value,
                   'persistence': persistence_score,
                   'crossover': crossover_score,
                   'crossover_penalty': crossover_penalty
               },
               'adx_value': strength_value,
               'plus_di_final': plus_di.iloc[-1],
               'minus_di_final': minus_di.iloc[-1]
           }
   ```

2. **Create `ADXIndicator` Wrapper (45 min)**
   ```python
   class ADXIndicator(TrendIndicator):
       def __init__(self, config=None):
           period = config.get('period', 14) if config else 14
           use_dx = config.get('use_dx', False) if config else False
           self.analyzer = ADXAnalyzer(period=period, use_dx=use_dx)
           self._last_result = None
           
           # Session-specific weights
           self.session_weights = {
               '1pm-3pm': {'strength': 0.50, 'persistence': 0.30, 'crossover': 0.20},
               '3pm-3:50pm': {'strength': 0.50, 'persistence': 0.30, 'crossover': 0.20},
               '3:50pm-4pm': {'strength': 0.40, 'persistence': 0.30, 'crossover': 0.30}
           }
       
       def calculate(self, df: pd.DataFrame, 
                    session: SessionName = "auto") -> IndicatorResult:
           # Get session-specific weights
           weights = self.session_weights.get(session, 
                                             {'strength': 0.50, 'persistence': 0.30, 'crossover': 0.20})
           
           result = self.analyzer.calculate_trend_quality(df, weights=weights)
           
           self._last_result = IndicatorResult(
               signal=result['quality_score'],  # Already 0-1
               raw_value=result['adx_value'],
               metadata=result
           )
           return self._last_result
       
       def get_signal(self) -> float:
           return self._last_result.signal if self._last_result else 0.5
   ```

3. **Signal Mapping:**
   - Use `quality_score` directly (already 0-1)
   - Combines ADX strength + DI persistence + crossover penalty
   - Higher score = higher trend quality

4. **Session Handling:**
   - Use session-specific weights from `WINDOW_CONFIGS`
   - Use DX (not ADX) for very short windows (< 10 bars)
   - Adjust period based on session length

5. **Migration Strategy:**
   - Wrap existing `calculate_trend_quality()` function
   - Keep original functions for backward compatibility
   - Class provides cleaner interface

**2. Signal Mapping:**
   - Use `quality_score` from `trend_quality.py` directly (already 0-1)
   - Higher = stronger trend quality
   - Composite of strength + persistence + (1 - crossovers)

**3. Session Handling:**
   - Use session names as-is: "1pm-3pm", "3pm-3:50pm", "3:50pm-4pm"
   - Configs automatically select ADX vs DX based on window size
   - Weights auto-adjust per session

**4. Why Wrapper Approach is Best:**
   - ✅ Preserves battle-tested logic
   - ✅ Minimal refactoring risk
   - ✅ Existing test suite still works
   - ✅ Fast integration (45 min vs 2+ hours)
   - ✅ Can refactor to full class later if needed

**Integration Priority: HIGH** - This is actually one of your **strongest modules**. The `trend_quality.py` composite system is sophisticated, well-tested, and production-ready. Just needs a thin wrapper.

---

## Overview

This module implements the complete ADX indicator system developed by J. Welles Wilder. It measures **trend strength regardless of direction** and provides additional metrics for trend quality assessment.

---

## File Structure

| File | Purpose |
|------|---------|
| `adx_calc.py` | Core ADX/DX calculation |
| `di_indicators.py` | +DI and -DI directional indicators |
| `di_crossovers.py` | DI crossover counting and penalty scoring |
| `di_persistence.py` | DI dominance persistence analysis |
| `trend_quality.py` | Composite trend quality score |

---

## Function Reference

### `adx_calc.py`

#### `wilder_smooth_avg(series, period) -> pd.Series`
Wilder's smoothing in **average form** (used for DX → ADX smoothing).

- **Formula**: `avg[i] = avg[i-1] * (period-1)/period + value[i]/period`
- **First value**: Mean of first `period` values
- **Use case**: Final ADX smoothing step

#### `directional_indicators(plus_dm_smooth, minus_dm_smooth, atr) -> Tuple[pd.Series, pd.Series]`
Calculates +DI and -DI from smoothed directional movement components.

- **Formula**: `+DI = (Smoothed +DM / ATR) * 100`
- **Returns**: Tuple of (+DI, -DI) Series

#### `directional_index(plus_di, minus_di) -> pd.Series`
Calculates DX (Directional Movement Index).

- **Formula**: `DX = |+DI - -DI| / (+DI + -DI) * 100`
- **Range**: 0-100 (higher = stronger directional movement)

#### `calculate_adx(high, low, close, period=14) -> pd.Series`
**Main ADX calculation function**. Computes ADX from OHLC data.

- **Steps**: TR → +DM/-DM → Wilder smooth → +DI/-DI → DX → Wilder smooth → ADX
- **Interpretation**:
  - 0-25: Weak/absent trend
  - 25-50: Strong trend
  - 50-75: Very strong trend
  - 75-100: Extremely strong trend

#### `calculate_adx_full(high, low, close, period=14) -> pd.DataFrame`
Returns ADX with all intermediate components (TR, +DM, -DM, ATR, +DI, -DI, DX, ADX).

#### `calculate_dx(high, low, close, period=14) -> pd.Series`
Returns unsmoothed DX (more responsive but volatile).

#### DataFrame convenience functions:
- `calculate_dx_from_df(df, period, high_col, low_col, close_col)`
- `calculate_dx_full_from_df(df, period, high_col, low_col, close_col)`
- `calculate_adx_from_df(df, period, high_col, low_col, close_col)`
- `calculate_adx_full_from_df(df, period, high_col, low_col, close_col)`

---

### `di_indicators.py`

#### `true_range(high, low, close) -> pd.Series`
Calculates True Range using Wilder's formula.

- **Formula**: `TR = max(high-low, |high-prev_close|, |low-prev_close|)`

#### `directional_movement(high, low) -> Tuple[pd.Series, pd.Series]`
Calculates raw +DM and -DM.

- **+DM**: `max(current_high - prev_high, 0)` if > -DM, else 0
- **-DM**: `max(prev_low - current_low, 0)` if > +DM, else 0

#### `wilder_smooth(series, period) -> pd.Series`
Wilder's smoothing in **sum form** (for TR/DM components).

- **First value**: Sum of first `period` values
- **Formula**: `smoothed[i] = smoothed[i-1] - smoothed[i-1]/period + value[i]`

#### `ema_smooth(series, period) -> pd.Series`
Alternative EMA smoothing (sum form for consistency).

- **Alpha**: `2 / (period + 1)`

#### `calculate_di(high, low, close, period=14, smoothing="wilder") -> Tuple[pd.Series, pd.Series]`
Main function to calculate +DI and -DI with configurable smoothing method.

#### `calculate_di_full(high, low, close, period=14, smoothing="wilder") -> pd.DataFrame`
Returns DI with all intermediate components.

---

### `di_crossovers.py`

#### `count_di_crossovers(plus_di, minus_di) -> int`
Counts number of times +DI and -DI cross (switch dominance).

- **Returns**: Number of sign changes in `(+DI - -DI)`

#### `calculate_crossover_penalty(plus_di, minus_di, max_expected=None) -> float`
Calculates crossover penalty score.

- **Range**: 0 (many crossovers/choppy) to 1 (no crossovers/smooth)
- **Default max_expected**: `len(array) / 2`
- **Formula**: `1 - (crossovers / max_expected)`

---

### `di_persistence.py`

#### `calculate_di_persistence(plus_di, minus_di) -> float`
Calculates what percentage of bars show consistent DI dominance based on **maximum consecutive run**.

- **Range**: 0 (perfect alternation) to 1 (same DI led entire period)
- **Formula**: `max_consecutive / total_bars`

#### `calculate_di_persistence_avg(plus_di, minus_di) -> float`
Alternative persistence metric using **average run length**.

- More sensitive to multiple trend attempts
- **Formula**: `avg_run_length / total_bars`

---

### `trend_quality.py`

#### Configuration: `WINDOW_CONFIGS`
Pre-configured settings for different trading windows:

| Window | Bar Size | Period | Use DX | Min Bars |
|--------|----------|--------|--------|----------|
| 1pm-3pm | 5m | 12 | No (ADX) | 20 |
| 3pm-3:50pm | 2m | 12 | No (ADX) | 20 |
| 3:50pm-4pm | 1m | 5 | Yes (DX) | 7 |

#### `normalize_strength(strength) -> float`
Converts ADX/DX (0-100) to 0-1 scale.

#### `validate_bars(bars, min_required) -> bool`
Validates sufficient data exists for calculation.

#### `get_valid_data_window(strength, plus_di, minus_di) -> Tuple`
Extracts only valid (non-NaN) data after warm-up period.

#### `calculate_trend_quality(bars, window_name, config=None) -> Dict`
**Main composite scoring function**. Combines:

1. **Strength** (ADX or DX): Trend magnitude
2. **Persistence**: DI dominance consistency
3. **Crossover penalty**: Chop filter

**Weights** (default):
- Strength: 50%
- Persistence: 30%
- Crossover: 20%

**Returns**:
```python
{
    'quality_score': 0.0-1.0,  # Final composite score
    'components': {
        'strength': float,
        'persistence': float,
        'crossover': float,
        'strength_raw': float,
        'dominant_di': 'plus' | 'minus'
    },
    'metadata': {...}
}
```

#### `calculate_all_windows(bars_dict) -> Dict`
Batch process multiple time windows.

#### `update_window_weights(window_name, new_weights) -> None`
Update weights for optimization/backtesting.

---

## Shortcomings & Improvement Suggestions

### 1. **Not Class-Based** ⚠️ HIGH PRIORITY
**Issue**: Module uses standalone functions; not compatible with planned `state_detector.py` architecture.

**Improvement**:
- Create `ADXIndicator` class implementing `TrendIndicator` interface
- Wrap `calculate_trend_quality` as primary method
- Add `get_signal() -> float` returning normalized trend quality score

### 2. **Warm-up Period Handling**
**Issue**: ADX requires `2 * period` bars before producing valid output, which is significant for short intraday windows.

**Improvement**: 
- Implement a "fast" ADX variant using EMA instead of Wilder smoothing
- Provide hybrid approach: use DX for early bars, transition to ADX as data accumulates

### 2. **Static Period Configuration**
**Issue**: Period is fixed regardless of market conditions.

**Improvement**:
- Implement adaptive period selection based on volatility (shorter in high-vol, longer in low-vol)
- Add `auto_period` parameter that selects based on available bars

### 3. **Equal Weighting in Composite Score**
**Issue**: `trend_quality.py` uses predefined weights that may not be optimal for all market conditions.

**Improvement**:
- Add weight optimization via backtesting
- Consider dynamic weighting based on recent regime (e.g., increase persistence weight after failed breakouts)

### 4. **No Trend Direction Signal**
**Issue**: ADX measures strength but not direction; `dominant_di` is a point-in-time signal.

**Improvement**:
- Add `trend_direction_score` (-1 to +1) based on +DI vs -DI spread
- Include DI slope for trend acceleration/deceleration

### 5. **Missing Trend Change Detection**
**Issue**: No explicit signal for trend exhaustion or reversal warning.

**Improvement**:
- Implement ADX divergence detection (price makes new high/low but ADX decreases)
- Add "hook" pattern detection (ADX turn after extended move)

### 6. **Crossover Penalty Sensitivity**
**Issue**: `max_expected` default of `len/2` may be too lenient for trending markets.

**Improvement**:
- Add regime-adaptive `max_expected` (lower for trending windows)
- Consider weighting recent crossovers more heavily

### 7. **No Vectorized Implementation**
**Issue**: Wilder smoothing uses Python loops, slow for large datasets.

**Improvement**:
- Implement numba-accelerated versions
- Use `pandas.ewm` with adjusted alpha for approximation

---

## Usage Example

```python
import pandas as pd
from features.trend.ADX.trend_quality import calculate_trend_quality

# Load your OHLC data
bars = pd.DataFrame({
    'high': [...],
    'low': [...],
    'close': [...]
})

# Calculate trend quality for 3pm-3:50pm window
result = calculate_trend_quality(bars, "3pm-3:50pm")

print(f"Trend Quality: {result['quality_score']:.3f}")
print(f"Dominant Direction: {result['components']['dominant_di']}")
```

