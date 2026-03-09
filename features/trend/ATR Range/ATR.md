# ATR/Range Ratio Module

## State Detector Integration Status

**Readiness Score: 50% ⚠️ NEEDS WORK**

**Current Status:**
- ❌ Function-based (not class-based)
- ✅ DataFrame-based API
- ✅ Simple, clear metric
- ✅ Session configurations defined
- ❌ No class structure
- ❌ Missing `TrendIndicator` interface

**Required Changes for Integration:**

1. **Create `ATRRangeAnalyzer` Class (60 min)**
   ```python
   class ATRRangeAnalyzer:
       """ATR/Range ratio analyzer for trend detection."""
       
       def __init__(self, atr_period: int = 10):
           self.atr_period = atr_period
       
       def calculate_ratio(self, df: pd.DataFrame) -> Tuple[float, float, float]:
           """
           Calculate ATR/Range ratio.
           
           Returns:
               Tuple of (ratio, median_atr, total_range)
           """
           total_range = df['high'].max() - df['low'].min()
           
           if total_range == 0:
               return np.nan, 0, 0
           
           # Calculate ATR
           high = df['high']
           low = df['low']
           close = df['close']
           
           tr1 = high - low
           tr2 = abs(high - close.shift(1))
           tr3 = abs(low - close.shift(1))
           tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
           atr = tr.rolling(window=self.atr_period, min_periods=1).mean()
           
           median_atr = atr.median()
           ratio = median_atr / total_range
           
           return ratio, median_atr, total_range
       
       def analyze(self, df: pd.DataFrame) -> Dict:
           """Analyze session and return results."""
           ratio, median_atr, total_range = self.calculate_ratio(df)
           
           # Classify
           if np.isnan(ratio):
               signal = 'NO_RANGE'
           elif ratio < 0.5:
               signal = 'TRENDING'
           elif ratio > 0.8:
               signal = 'CONSOLIDATING'
           else:
               signal = 'NEUTRAL'
           
           return {
               'ratio': ratio,
               'median_atr': median_atr,
               'total_range': total_range,
               'signal': signal
           }
   ```

2. **Create `ATRRangeIndicator` Wrapper (30 min)**
   ```python
   class ATRRangeIndicator(TrendIndicator):
       def __init__(self, config=None):
           atr_period = config.get('atr_period', 10) if config else 10
           self.analyzer = ATRRangeAnalyzer(atr_period=atr_period)
           self._last_result = None
       
       def calculate(self, df: pd.DataFrame, 
                    session: SessionName = "auto") -> IndicatorResult:
           result = self.analyzer.analyze(df)
           
           # Invert ratio: low ratio = trending → high signal
           if np.isnan(result['ratio']):
               signal = 0.5
           else:
               # Map [0, 1] ratio to [1, 0] signal (inverse)
               signal = 1.0 - min(1.0, result['ratio'])
           
           self._last_result = IndicatorResult(
               signal=signal,
               raw_value=result['ratio'],
               metadata=result
           )
           return self._last_result
       
       def get_signal(self) -> float:
           return self._last_result.signal if self._last_result else 0.5
   ```

3. **Signal Mapping:**
   - **Invert ratio**: `signal = 1.0 - ratio`
   - Low ATR/Range (trending) → High signal
   - High ATR/Range (choppy) → Low signal
   - Clamp ratio to [0, 1] before inversion

4. **Session Configs:**
   - Could use session-specific ATR periods from `SESSIONS` dict
   - Or make ATR period auto-adaptive based on bar count

5. **Migration Path:**
   - Keep existing functions for backward compatibility
   - New class calls same calculation logic
   - Gradually phase out standalone functions

**Integration Priority: MEDIUM** - Simple but effective metric, worth the refactor effort.

---

## Overview

This module analyzes the relationship between **Average True Range (ATR)** and **Total Session Range** to distinguish trending from consolidating markets. The core insight: in a trending market, ATR (local volatility) should be small relative to total range (directional movement).

**Key Metric**: `ATR / Total Range`
- **Low ratio** (< 0.5): Trending - price moved directionally with relatively small bar-to-bar volatility
- **High ratio** (> 0.8): Consolidating - price oscillated within a range with high relative volatility

---

## File Structure

| File | Purpose |
|------|---------|
| `atr.py` | ATR calculation and trend/consolidation detection |

---

## Function Reference

### `atr.py`

#### Configuration: `SESSIONS`

Pre-configured session parameters:

| Session | Bar Size | Total Bars | ATR Period |
|---------|----------|------------|------------|
| 1pm-3pm | 5min | 24 | 10 |
| 3pm-3:50pm | 2min | 25 | 8 |
| 3:50-4pm | 1min | 10 | 3 |

#### `calculate_atr(df, period) -> pd.Series`
Calculates rolling ATR using standard True Range formula.

- **True Range**: `max(H-L, |H-prev_close|, |L-prev_close|)`
- **ATR**: Rolling mean of TR with `min_periods=1`
- **Returns**: Series of ATR values

#### `calculate_ratio(df, atr_period) -> Tuple[float, float, float]`
Computes the core ATR/Range ratio.

**Calculation**:
1. Total Range = `max(highs) - min(lows)` over entire window
2. Median ATR = median of rolling ATR values (robust to outliers)
3. Ratio = `median_atr / total_range`

**Returns**: `(raw_ratio, median_atr, total_range)`

**Edge case**: Returns `(np.nan, 0, 0)` if total_range is zero.

#### `analyze_session(df, session_name) -> Dict`
Main analysis function for a trading session.

**Signal Logic**:
| Ratio | Signal | Interpretation |
|-------|--------|----------------|
| < 0.5 | TRENDING | ATR small vs range → directional |
| 0.5-0.8 | NEUTRAL | Mixed behavior |
| > 0.8 | CONSOLIDATING | ATR large vs range → choppy |

**Returns**:
```python
{
    'session': str,
    'raw_ratio': float,  # The core metric
    'median_atr': float,
    'total_range': float,
    'signal': 'TRENDING' | 'NEUTRAL' | 'CONSOLIDATING' | 'NO_RANGE'
}
```

---

## Intuition

### Why This Works

**Trending Market**:
```
Price: ──────────────────────▲
       ↗ small bars moving up
       
Total Range: Large (200 points)
Median ATR: Small (15 points)  
Ratio: 15/200 = 0.075 → TRENDING
```

**Consolidating Market**:
```
Price: ─────╱╲──────╱╲───────
       ↔ large oscillating bars
       
Total Range: Small (50 points)
Median ATR: Large (30 points)
Ratio: 30/50 = 0.60 → NEUTRAL/CONSOLIDATING
```

---

## Shortcomings & Improvement Suggestions

### 1. **Not Class-Based** ⚠️ HIGH PRIORITY
**Issue**: Module uses standalone functions; not compatible with planned `state_detector.py` architecture.

**Improvement**:
- Create `ATRRangeIndicator` class implementing `TrendIndicator` interface
- Wrap `analyze_session` as primary method
- Add `get_signal() -> float` returning inverted ratio (higher = more trending)

### 2. **Arbitrary Threshold Values**
**Issue**: The 0.5 and 0.8 thresholds are hardcoded without empirical validation.

**Improvement**:
- Add calibration function that determines thresholds from historical data
- Use percentile-based thresholds (e.g., bottom 25% = trending, top 25% = consolidating)
- Consider regime-dependent thresholds

### 2. **No Time-Weighting**
**Issue**: All bars in the window contribute equally; doesn't capture *when* trending occurred.

**Improvement**:
- Add time-weighted variant that emphasizes recent bars
- Implement rolling ratio to track trend evolution within session

### 3. **Single ATR Period**
**Issue**: Fixed ATR period may not suit all volatility regimes.

**Improvement**:
- Implement adaptive ATR period based on bar count
- Use multiple ATR periods and compare (multi-scale analysis)

### 4. **Median vs Mean ATR**
**Issue**: Median is used for robustness, but doesn't indicate ATR distribution shape.

**Improvement**:
- Add ATR stability metric (std of ATR / mean ATR)
- Include ATR trend (is volatility expanding or contracting?)

### 5. **No Directional Information**
**Issue**: Ratio is unsigned; doesn't indicate trend direction.

**Improvement**:
- Add signed ratio: `sign(close[-1] - close[0]) * ratio`
- Include directional breakout detection

### 6. **Vulnerable to Outlier Bars**
**Issue**: Total range can be dominated by single spike bar.

**Improvement**:
- Use robust range: trim extreme 5% of highs/lows before computing range
- Add "effective range" that excludes obvious false breakouts (wicks that immediately reverse)

### 7. **No Confidence Score**
**Issue**: Binary signal without indication of strength.

**Improvement**:
- Convert ratio to continuous score (0-1 scale)
- Add confidence metric based on how far ratio is from thresholds

### 8. **Missing Session Configurations**
**Issue**: `SESSIONS` dict uses inconsistent key naming (`'3:50-4pm'` vs other formats).

**Improvement**:
- Standardize session naming convention
- Add validation for session_name input

---

## Usage Example

```python
import pandas as pd
from features.trend.ATR_Range.atr import analyze_session

# Load your OHLC data for the session
bars = pd.DataFrame({
    'high': [...],
    'low': [...],
    'close': [...]
})

# Analyze the 3pm-3:50pm session
result = analyze_session(bars, '3pm-3:50pm')

print(f"Signal: {result['signal']}")
print(f"ATR/Range Ratio: {result['raw_ratio']:.4f}")
```

---

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| ADX | Both measure trend strength; ATR/Range is simpler, ADX is more sophisticated |
| MSS | ATR is used in MSS for normalization; could share ATR calculation |
| DRA | Complementary - DRA measures range breakout, ATR/Range measures within-range behavior |

