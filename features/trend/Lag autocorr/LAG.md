# Lag Autocorrelation & Hurst Exponent Module

## State Detector Integration Status

**Readiness Score: 75% ⚠️ PARTIALLY READY**

**Current Status:**
- ✅ Class-based architecture (`IntradayTemporalFeatures`)
- ✅ Comprehensive temporal analysis
- ✅ Multiple methods (ACF, Hurst simple, Hurst R/S)
- ✅ Regime classification built-in
- ⚠️ No `get_signal()` method
- ⚠️ Missing `TrendIndicator` interface

**Required Changes for Integration:**

1. **Add `get_signal()` Method to IntradayTemporalFeatures Class (15 min)**
   ```python
   def get_signal(self) -> float:
       """
       Get trending signal from Hurst exponent.
       
       Returns:
           Trending signal (0-1): 0 = mean-reverting, 0.5 = random, 1 = trending
       """
       features = self.get_all_features()
       hurst = features.get('hurst_exponent')
       
       if hurst is None or np.isnan(hurst):
           return 0.5  # Neutral if can't calculate
       
       # Map Hurst to 0-1: H=0 → 0, H=0.5 → 0.5, H=1 → 1
       # This is already the natural mapping
       return float(hurst)
   ```

2. **Create `LagIndicator` Wrapper (30 min)**
   ```python
   class LagIndicator(TrendIndicator):
       def __init__(self, config=None):
           self.config = config or {}
           self._last_features = None
       
       def calculate(self, df: pd.DataFrame) -> IndicatorResult:
           prices = df['close'].values
           
           # Create feature calculator
           tf = IntradayTemporalFeatures(prices, compute_returns=True)
           
           # Get all features (auto-selects Hurst method)
           features = tf.get_all_features()
           regime = tf.get_regime_signal()
           
           hurst = features.get('hurst_exponent', 0.5)
           if hurst is None or np.isnan(hurst):
               hurst = 0.5
           
           self._last_features = features
           
           return IndicatorResult(
               signal=float(hurst),  # 0-1 scale naturally
               raw_value=features.get('acf_lag1'),
               metadata={
                   **features,
                   'regime': regime,
                   'n_bars': len(prices)
               }
           )
       
       def get_signal(self) -> float:
           if self._last_features is None:
               return 0.5
           hurst = self._last_features.get('hurst_exponent', 0.5)
           return float(hurst) if hurst is not None else 0.5
   ```

3. **Signal Mapping:**
   - Use Hurst exponent directly (already 0-1)
   - H < 0.5 = mean-reverting (low trending signal)
   - H = 0.5 = random walk (neutral)
   - H > 0.5 = trending (high trending signal)

4. **Alternative: ACF-Based Signal**
   - Could also use: `signal = (acf_lag1 + 1) / 2` to map [-1,1] → [0,1]
   - Or combine: `signal = 0.7 * hurst + 0.3 * ((acf_lag1 + 1) / 2)`

5. **Session Handling:**
   - Auto-selects Hurst method based on bar count
   - No session parameter needed (works on any window)

**Integration Priority: MEDIUM-HIGH** - Statistical foundation complements technical indicators.

---

## Overview

This module computes **temporal dependence features** optimized for intraday minute-bar data. It answers: "Do price moves persist (trend) or reverse (mean-revert)?"

**Key Metrics**:
1. **Autocorrelation (ACF)**: Correlation between returns at time t and t-lag
2. **Hurst Exponent**: Measure of long-range dependence (trending vs mean-reverting)

Both metrics are adapted for small sample sizes (50-120 bars) typical of intraday trading sessions.

---

## File Structure

| File | Purpose |
|------|---------|
| `lag.py` | Autocorrelation, Hurst exponent, and regime detection |

---

## Function Reference

### `lag.py`

#### Dataclass: `SessionConfig`
Configuration for intraday session analysis.

| Method | Session | Expected Bars |
|--------|---------|---------------|
| `afternoon_session()` | 1pm-3pm | 120 bars |
| `close_session()` | 3pm-3:50pm | 50 bars |

---

### Class: `IntradayTemporalFeatures`

Main class for computing temporal dependence features.

##### `__init__(self, prices, compute_returns=True)`
Initialize with price array.

- **prices**: 1D array of prices (e.g., close prices)
- **compute_returns**: If True, internally converts to log returns for stationarity

##### `autocorrelation(self, lag) -> float`
Calculate autocorrelation at specified lag.

**Formula**:
```python
x = data - mean(data)  # Demean
c0 = dot(x, x) / n     # Variance
c_lag = dot(x[lag:], x[:-lag]) / n  # Autocovariance
acf = c_lag / c0
```

**Returns**: ACF coefficient [-1, 1] or `np.nan` if insufficient data

**Interpretation**:
| ACF Value | Meaning |
|-----------|---------|
| > 0.2 | Positive persistence (trending) |
| < -0.1 | Negative autocorrelation (mean-reverting) |
| ≈ 0 | Random walk |

**Minimum requirements**: 10 observations total, 5 after lag.

##### `autocorrelations(self, lags=[1, 2, 5]) -> Dict`
Calculate multiple ACF values efficiently.

**Returns**: `{'acf_lag1': 0.15, 'acf_lag2': 0.08, 'acf_lag5': -0.02}`

##### `hurst_exponent_simple(self) -> float`
Hurst exponent using **variance method** - suitable for small samples.

**Method**: Fit power law to `Var(lag-differences) ~ lag^(2H)`

**Returns**: Hurst exponent [0, 1] or `np.nan` if insufficient data (< 20 bars)

**Adaptive lag range**: `max_lag = min(n // 4, 20)`

##### `hurst_exponent_rs(self) -> float`
Hurst exponent using **Rescaled Range (R/S) method** - more robust but needs more data.

**Method**: Classic R/S analysis with adaptive window sizing.

**Returns**: Hurst [0, 1] or `np.nan` if insufficient data (< 30 bars)

**Use case**: 120-bar afternoon sessions.

##### `get_all_features(self, session_config=None) -> Dict`
Calculate all temporal features, auto-selecting Hurst method based on session length.

**Logic**:
- Close session (short): Use `hurst_exponent_simple()`
- Afternoon session (long): Try `hurst_exponent_rs()`, fallback to simple

**Returns**:
```python
{
    'acf_lag1': float,
    'acf_lag2': float,
    'acf_lag5': float,
    'hurst_exponent': float
}
```

##### `get_regime_signal(self) -> str`
Quick regime classification based on ACF and Hurst.

| Condition | Regime |
|-----------|--------|
| H > 0.6 and ACF1 > 0.2 | `trending` |
| H < 0.4 and ACF1 < -0.1 | `mean_reverting` |
| \|ACF1\| < 0.1 and 0.45 ≤ H ≤ 0.55 | `random_walk` |
| H > 0.5 | `weak_trend` |
| H < 0.5 | `weak_reversion` |

---

### Convenience Functions

#### `analyze_intraday_session(df, price_col='close', session_config=None) -> Dict`
Analyze a single session from DataFrame.

**Returns**:
```python
{
    'acf_lag1': float,
    'acf_lag2': float,
    'acf_lag5': float,
    'hurst_exponent': float,
    'regime': str,
    'n_bars': int,
    'price_change_pct': float
}
```

#### `batch_analyze_sessions(df, date_col, time_col, price_col) -> pd.DataFrame`
Analyze multiple days of intraday data.

**Automatically splits by**:
- Afternoon: 13:00-15:00
- Close: 15:00-15:50

**Returns**: DataFrame with one row per day, columns prefixed with session name.

---

## Theoretical Background

### Hurst Exponent Interpretation

| H Value | Behavior | Market Implication |
|---------|----------|-------------------|
| 0.0-0.4 | Mean-reverting | Sell rallies, buy dips |
| 0.4-0.6 | Random walk | No exploitable pattern |
| 0.6-1.0 | Trending/persistent | Follow breakouts |

### Autocorrelation Decay

- **Trending**: ACF decays slowly (lag1 ≈ lag5)
- **Mean-reverting**: ACF alternates sign (lag1 +, lag2 -)
- **Random walk**: ACF ≈ 0 at all lags

---

## Shortcomings & Improvement Suggestions

### 1. **Missing `get_signal()` Method** ⚠️ MEDIUM PRIORITY
**Issue**: Class exists but lacks standardized signal output for `state_detector.py` integration.

**Improvement**:
- Add `get_signal() -> float` method returning Hurst-based trending signal
- Map: H > 0.5 → positive (trending), H < 0.5 → negative (reverting)
- Consider: `signal = (hurst - 0.5) * 2` for -1 to 1 scale

### 2. **Log Returns Assumption**
**Issue**: Always uses log returns for stationarity; may not suit all use cases.

**Improvement**:
- Add option for simple returns
- Auto-detect if data is already stationary (via ADF test)

### 2. **Fixed Lag Selection**
**Issue**: Lags [1, 2, 5] are hardcoded; may not capture relevant cycles.

**Improvement**:
- Add auto-lag selection based on first significant ACF
- Include partial autocorrelation (PACF) for more insight

### 3. **Hurst Method Selection**
**Issue**: Binary choice based on session type; doesn't consider actual data quality.

**Improvement**:
- Auto-select based on actual bar count and variance stability
- Ensemble Hurst: average multiple methods for robustness

### 4. **No Confidence Intervals**
**Issue**: Point estimates without uncertainty quantification.

**Improvement**:
- Add bootstrap confidence intervals for ACF
- Report Hurst estimation standard error

### 5. **Regime Classification Thresholds**
**Issue**: Hardcoded thresholds may not generalize across instruments.

**Improvement**:
- Add calibration mode using historical regime annotations
- Use fuzzy/probabilistic regime classification

### 6. **No Structural Break Detection**
**Issue**: Assumes stationary regime throughout window.

**Improvement**:
- Implement rolling Hurst to detect regime shifts
- Add CUSUM or similar for detecting breakpoints

### 7. **Time Column Handling**
**Issue**: `batch_analyze_sessions` assumes string time comparison ('13:00').

**Improvement**:
- Support datetime objects
- Handle timezone-aware timestamps

### 8. **Missing Error Handling**
**Issue**: Division by zero and edge cases handled silently with NaN.

**Improvement**:
- Add warnings for insufficient data
- Return structured result with error status

### 9. **No Integration with Price Level**
**Issue**: Only analyzes returns; doesn't consider absolute price levels.

**Improvement**:
- Add support analysis (price relative to VWAP, MA)
- Combine with DRA for range-aware regime detection

### 10. **Computational Efficiency**
**Issue**: Pure Python/NumPy; may be slow for large batch analysis.

**Improvement**:
- Add numba JIT compilation for core loops
- Vectorize across multiple sessions

---

## Usage Example

```python
import pandas as pd
import numpy as np
from features.trend.Lag_autocorr.lag import (
    IntradayTemporalFeatures,
    SessionConfig,
    analyze_intraday_session
)

# Single session analysis
prices = np.array([100, 100.5, 101.2, 101.0, 101.8, 102.3, ...])  # 50+ bars

tf = IntradayTemporalFeatures(prices, compute_returns=True)
features = tf.get_all_features(SessionConfig.close_session())

print(f"ACF(1): {features['acf_lag1']:.3f}")
print(f"Hurst: {features['hurst_exponent']:.3f}")
print(f"Regime: {tf.get_regime_signal()}")

# DataFrame analysis
df = pd.DataFrame({
    'close': prices,
    'timestamp': pd.date_range('2024-01-15 15:00', periods=len(prices), freq='1min')
})

result = analyze_intraday_session(df, price_col='close', session_config=SessionConfig.close_session())
```

---

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| ADX | Both detect trend; ACF/Hurst are statistical, ADX is price-action based |
| IRR | IRR = intraperiod (within-bar) reversion; ACF = inter-bar dependence; complementary |
| MSS | MSS = multi-scale slope; Hurst = multi-scale variance; similar concept, different approach |

---

## References

- Hurst, H.E. (1951). "Long-term storage capacity of reservoirs"
- Mandelbrot, B.B. (1971). "Analysis of long-run dependence in economics"
- Lo, A.W. (1991). "Long-term memory in stock market prices"

