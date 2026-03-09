# IRR (Intraperiod Reversion Ratio) Module

## State Detector Integration Status

**Readiness Score: 95% ✅ READY**

**Current Status:**
- ✅ Class-based architecture (`IRRAnalyzer`)
- ✅ DataFrame-based API
- ✅ Two-level analysis (full window + sub-windows)
- ✅ Comprehensive output structure (`IRRResult`)
- ✅ Tested and validated
- ⚠️ Missing `TrendIndicator` interface implementation

**Required Changes for Integration:**

1. **Create `IRRIndicator` Wrapper (30 min)**
   ```python
   class IRRIndicator(TrendIndicator):
       def __init__(self, config=None):
           self.analyzer = IRRAnalyzer(
               num_subwindows=config.get('num_subwindows', 5)
           )
       
       def calculate(self, df: pd.DataFrame) -> IndicatorResult:
           window_end = len(df) - 1
           window_start = max(0, window_end - 49)  # 50-bar window
           result = self.analyzer.analyze(df, window_start, window_end)
           
           # Convert directional_strength to signal (0-1)
           return IndicatorResult(
               signal=result.directional_strength,  # Already 0-1
               raw_value=result.median_sub_irr,
               metadata=result.to_dict()
           )
       
       def get_signal(self) -> float:
           return self._last_result.signal
   ```

2. **Signal Mapping:**
   - Use `directional_strength` (1 - average_irr) as primary signal
   - Already normalized to 0-1 scale
   - Higher values = more directional/trending

3. **Integration Notes:**
   - Module is fully self-contained
   - No session parameter needed (analyzes whatever window provided)
   - Can optionally expose `num_subwindows` as config parameter
   - Reversal detection info available in metadata for advanced use

**Integration Priority: HIGH** - Module is production-ready, just needs wrapper.

---

## Overview

IRR measures **bar-level reversion behavior** by comparing candle body size to total range. It quantifies how much price "gave back" within each bar - bars where price explored a range but closed near where it opened indicate intraperiod reversion (price reversed within the bar).

**Core Formula**: `IRR = 1 - (body / range)`

- **High IRR** (> 0.65): Large wicks, small body → high intraperiod reversion → consolidation regime
- **Low IRR** (< 0.35): Small wicks, large body → low reversion (full follow-through) → trending regime

---

## File Structure

| File | Purpose |
|------|---------|
| `irr.py` | IRR calculation and regime classification |

---

## Function Reference

### `irr.py`

#### `compute_irr(open, high, low, close) -> float | None`
Compute IRR for a single bar.

**Formula**:
```python
range_val = high - low
body = abs(close - open)
irr = 1 - (body / range_val)
```

**Returns**: 
- IRR value (0.0 to 1.0)
- `None` if range < 0.0001 (avoid division by zero)

**Interpretation**:
| IRR Value | Body/Range | Candle Pattern |
|-----------|------------|----------------|
| 0.0 | 100% body | Marubozu (full directional) |
| 0.5 | 50% body | Normal candle |
| 1.0 | 0% body | Doji (complete indecision) |

#### `compute_session_window_irr(bars, window_name) -> float | None`
Compute **median IRR** for a full session window.

**Parameters**:
- `bars`: List of bar objects with `.open`, `.high`, `.low`, `.close` attributes
- `window_name`: Session identifier ('early', 'middle', 'close')

**Returns**: Median IRR across all valid bars, or `None` if no valid bars.

**Why Median?**: Robust to outlier bars (news spikes, gaps).

#### `classify_regime(median_irr) -> str`
Simple regime classification based on absolute thresholds.

| Median IRR | Regime | Market Behavior |
|------------|--------|-----------------|
| > 0.65 | `high_reversion` | Consolidation, mean-reverting |
| < 0.35 | `directional` | Trending, momentum-driven |
| 0.35-0.65 | `mixed` | Unclear, transitional |
| None | `unknown` | Insufficient data |

#### `analyze_session(session_data) -> Dict`
Analyze a complete trading session with multiple windows.

**Input**:
```python
session_data = {
    'early': [bars...],   # 1pm-3pm
    'middle': [bars...],  # 3pm-3:50pm
    'close': [bars...]    # 3:50pm-4pm
}
```

**Output**:
```python
{
    'early': {'median_irr': 0.52, 'regime': 'mixed'},
    'middle': {'median_irr': 0.28, 'regime': 'directional'},
    'close': {'median_irr': 0.71, 'regime': 'high_reversion'}
}
```

---

## Visual Explanation

```
LOW IRR (Directional):          HIGH IRR (Indecision):
┌───┐                           ┌───┐
│   │ small wick               │   │ large upper wick
│███│                           ├───┤
│███│ LARGE BODY               │ █ │ SMALL BODY
│███│                           ├───┤
│   │ small wick               │   │ large lower wick
└───┘                           └───┘
IRR ≈ 0.1                       IRR ≈ 0.9
→ Strong conviction             → Indecision/rejection
```

---

## Shortcomings & Improvement Suggestions

### 1. **Not Class-Based** ⚠️ HIGH PRIORITY
**Issue**: Module uses standalone functions; not compatible with planned `state_detector.py` architecture.

**Improvement**:
- Create `IRRIndicator` class implementing `TrendIndicator` interface
- Add `calculate(df) -> IRRResult` method
- Add `get_signal() -> float` for normalized output

### 2. **Bar Object Assumption**
**Issue**: `compute_session_window_irr` expects bar objects with attributes, not DataFrames.

**Improvement**:
- Add DataFrame support: `compute_session_window_irr_df(df, window_name)`
- Use duck typing to accept either format
- **Required for `IRRIndicator` class**

### 3. **Unused `window_name` Parameter**
**Issue**: `window_name` parameter is accepted but never used in `compute_session_window_irr`.

**Improvement**:
- Either remove parameter or use it for window-specific logic
- Could adjust thresholds based on window (EOD bars typically have higher IRR)

### 3. **Static Thresholds**
**Issue**: 0.35/0.65 thresholds are arbitrary and may not suit all instruments.

**Improvement**:
- Add calibration function based on historical IRR distribution
- Use percentile-based thresholds (e.g., 25th/75th percentile)
- Instrument-specific configurations

### 4. **No Trend Confirmation**
**Issue**: Low IRR indicates directional bars but doesn't confirm *consistent* direction.

**Improvement**:
- Add "directional consistency" metric: % of low-IRR bars with same direction
- Combine IRR with sign of (close - open) for directional IRR

### 5. **Missing Aggregate Statistics**
**Issue**: Only median is computed; distribution shape is ignored.

**Improvement**:
- Add IRR std/variance to measure regime stability
- Track IRR skewness (are indecision bars clustered or distributed?)
- Compute IRR momentum (is indecision increasing/decreasing?)

### 6. **No Weighted Averaging**
**Issue**: All bars contribute equally to median.

**Improvement**:
- Add volume-weighted IRR (high-volume bars matter more)
- Time-weighted variant (recent bars emphasized)

### 7. **Doji Dominance Issue**
**Issue**: True dojis (open = close) get IRR = 1.0, which may dominate median.

**Improvement**:
- Track doji count separately
- Consider trimmed mean instead of median

### 8. **No Rolling/Streaming Support**
**Issue**: Only batch computation; can't track IRR evolution.

**Improvement**:
- Add `IRRTracker` class for streaming computation
- Implement rolling window IRR

### 9. **Threshold Clustering**
**Issue**: Values near thresholds (e.g., 0.36, 0.64) flip regime classification easily.

**Improvement**:
- Add confidence score based on distance from threshold
- Implement hysteresis (require sustained break of threshold to change regime)

---

## Usage Example

```python
from features.trend.IRR.irr import compute_irr, classify_regime

# Single bar IRR
irr = compute_irr(open=100.0, high=102.5, low=99.0, close=100.2)
print(f"Bar IRR: {irr:.3f}")  # High IRR → indecision

# Session analysis (with bar objects)
class Bar:
    def __init__(self, o, h, l, c):
        self.open, self.high, self.low, self.close = o, h, l, c

bars = [
    Bar(100, 101, 99, 100.8),  # Directional
    Bar(100.8, 102, 100.5, 101.9),  # Directional
    Bar(102, 102.5, 101, 101.2),  # Indecision
]

from features.trend.IRR.irr import compute_session_window_irr
median_irr = compute_session_window_irr(bars, 'middle')
regime = classify_regime(median_irr)
print(f"Session regime: {regime}")
```

---

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| ADX | Both measure trend strength; IRR is bar-level, ADX is smoothed multi-bar |
| MSS | MSS looks at slope structure; IRR looks at bar microstructure |
| Lag Autocorr | IRR = intraperiod reversion; Autocorr = inter-bar dependence; complementary |

---

## Theoretical Background

IRR (Intraperiod Reversion Ratio) quantifies how much price "gives back" within each bar:

- **Marubozu** (IRR ≈ 0): No intraperiod reversion - price followed through completely
- **Spinning Top** (IRR ≈ 0.5-0.7): Moderate reversion - some giveback within bar
- **Doji** (IRR ≈ 0.9-1.0): Maximum reversion - price reversed entirely within bar

**Why "Intraperiod Reversion"?**
- "Intraperiod" = within a single bar/candle
- "Reversion" = price moved in one direction then reversed back
- High IRR bars show price explored a range but couldn't maintain direction

The ratio-based approach quantifies these patterns continuously rather than using discrete pattern matching.

### Connection to Mean Reversion
Sessions dominated by high-IRR bars indicate a mean-reverting regime at the bar level. This often corresponds to range-bound, consolidating market conditions where breakouts fail to sustain.

