# MSS (Multi-Scale Slope) Analysis Module

## State Detector Integration Status

**Readiness Score: 90% ✅ READY**

**Current Status:**
- ✅ Class-based architecture (`MultiScaleSlope`)
- ✅ DataFrame-based API
- ✅ Sophisticated multi-scale analysis
- ✅ Returns trending_score directly (0-1)
- ✅ Comprehensive diagnostics
- ⚠️ Missing `TrendIndicator` interface implementation

**Required Changes for Integration:**

1. **Create `MSSIndicator` Wrapper (30 min)**
   ```python
   class MSSIndicator(TrendIndicator):
       def __init__(self, config=None):
           self.mss = MultiScaleSlope(
               num_subwindows=config.get('num_subwindows', 4),
               atr_period=config.get('atr_period', 14),
               prominence_lookback=config.get('prominence_lookback', 3)
           )
           self._last_result = None
       
       def calculate(self, df: pd.DataFrame) -> IndicatorResult:
           window_end = len(df) - 1
           window_start = max(14, window_end - 49)  # 50-bar window + preload
           
           result = self.mss.calculate_trending_score(
               df, 
               window_start=window_start,
               window_end=window_end,
               preload_bars=14
           )
           
           self._last_result = IndicatorResult(
               signal=result.trending_score,  # Already 0-1
               raw_value=result.main_slope,
               metadata=result.to_dict()
           )
           return self._last_result
       
       def get_signal(self) -> float:
           return self._last_result.signal if self._last_result else 0.0
   ```

2. **Signal Mapping:**
   - Use `trending_score` directly (already 0-1)
   - Score combines 4 components with weights
   - Higher values = stronger trend quality

3. **Preload Requirements:**
   - Needs 14 bars before analysis window for ATR warm-up
   - Handle gracefully if insufficient data (fallback to range-based estimate)

4. **Optional Enhancement:**
   - Could add signed variant: `sign(main_slope) * trending_score` for directional signal
   - Expose component weights as config parameters

**Integration Priority: HIGH** - Module is most sophisticated, provides rich metadata.

---

## Overview

MSS is the most sophisticated trend detection module in the system. It analyzes price windows at **multiple scales** to determine trend quality by examining:

1. **Main window slope** (volatility-normalized)
2. **Sub-window slopes** (4-6 equal divisions)
3. **Extrema-based path slopes** (swing point analysis)
4. **Cross-scale consistency and alignment**

**Key Insight**: A true trend should show consistent slope direction across all scales and through prominent swing points.

---

## File Structure

| File | Purpose |
|------|---------|
| `mss.py` | Multi-Scale Slope analysis with composite scoring |

---

## Configuration Constants

| Constant | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_ATR_PERIOD` | 14 | ATR calculation period |
| `DEFAULT_NUM_SUBWINDOWS` | 4 | Sub-window divisions |
| `DEFAULT_PROMINENCE_LOOKBACK` | 3 | Bars for extrema detection |
| `DEFAULT_AMPLITUDE_THRESHOLD_FACTOR` | 0.5 | ATR multiplier for swing significance |

### Composite Score Weights (v2.0)

| Component | Weight | Purpose |
|-----------|--------|---------|
| Directional Consistency | 0.35 | % of sub-windows matching main direction |
| Slope Alignment | 0.20 | How tightly clustered sub-slopes are |
| Extrema Coherence | 0.20 | Swing structure alignment with trend |
| Magnitude | 0.25 | Absolute slope strength |

### Floor Values

| Floor | Value | Purpose |
|-------|-------|---------|
| `FLOOR_SLOPE_ALIGNMENT` | 0.1 | Prevent alignment collapse |
| `FLOOR_EXTREMA_COHERENCE` | 0.2 | Prevent coherence collapse |

---

## Function Reference

### `mss.py`

#### Dataclasses

##### `ExtremaInfo`
Information about a prominent high or low.
- `index`: Bar index
- `price`: Price at extremum
- `prominence`: Swing amplitude
- `is_high`: True for highs, False for lows

##### `Diagnostics`
Debug/calibration information.
- `main_slope_raw`: Percent change (before normalization)
- `internal_volatility`: Std dev of log returns
- `sub_slope_mean/std/range`: Sub-window statistics
- `num_positive_subs/num_negative_subs`: Directional breakdown
- `atr_value`: ATR at analysis time

##### `MSSResult`
Complete analysis result.
- `trending_score`: Final composite score (0-1)
- `main_slope`: Normalized main window slope
- `sub_slopes`: List of sub-window slopes
- `components`: Individual component scores
- `extrema`: Detected swing points
- `interpretation`: Human-readable regime
- `diagnostics`: Optional debug info

---

### Class: `MultiScaleSlope`

Main analysis class.

##### `__init__(self, atr_period, num_subwindows, prominence_lookback, amplitude_threshold_factor)`
Initialize with configurable parameters.

##### `calculate_atr(self, df, period=None) -> pd.Series`
Standard ATR calculation.

##### `calculate_normalized_slope(self, prices, start_idx, end_idx, return_components=False) -> float | Tuple`
**Core slope calculation** with volatility normalization.

**Formula**:
```python
percent_change = (end_price - start_price) / start_price
raw_slope = percent_change / num_bars
normalized_slope = raw_slope / std(log_returns)
```

**Returns**: Risk-adjusted slope ("% per bar per unit volatility")
- High value = strong directional with low chop → trending
- Low value = weak move OR high internal chop → consolidating

##### `calculate_sub_window_slopes(self, prices, window_start, window_end) -> List[float]`
Divide window into `num_subwindows` equal parts and calculate normalized slope for each.

##### `find_prominent_extrema(self, df, window_start, window_end, atr_value) -> Tuple[ExtremaInfo, ExtremaInfo]`
Find most prominent high and low within window.

**Prominence criteria**:
1. Must be highest/lowest within `prominence_lookback` bars on EACH side
2. Swing amplitude must exceed `amplitude_threshold_factor * ATR`
3. Select most prominent among qualifying

##### `calculate_extrema_path_coherence(self, prices, main_slope, window_start, window_end, prominent_high, prominent_low) -> Tuple[float, List[float]]`
**Path coherence analysis** - does the swing structure support the trend?

**For uptrend (main_slope > 0)**:
- Expected path: Start → High → Low → End (pullback then continuation)
- Score based on: entry slope positive, exit slope positive

**For downtrend (main_slope < 0)**:
- Expected path: Start → Low → High → End (bounce then continuation)
- Score based on: entry slope negative, exit slope negative

**V2.0 improvements**:
- Continuous scoring (tanh-based), not binary
- Neutral fallback (0.5) when no extrema found
- Handles partial information gracefully

##### `calculate_composite_score(self, main_slope, sub_slopes, extrema_coherence) -> Dict`
**Combine all components** into final trending score.

**Component calculations**:

1. **Directional Consistency**: `matching_sub_signs / total_subs`

2. **Slope Alignment** (V2.0 - exponential decay):
```python
relative_variance = sub_std / (|main_slope| * sqrt(num_subwindows))
alignment = exp(-relative_variance)
```

3. **Extrema Coherence**: From path analysis (with floor)

4. **Magnitude** (adaptive scaling):
```python
magnitude = tanh(|main_slope| * 5.0)
```

##### `interpret_score(score) -> str`
Convert score to human-readable interpretation.

| Score Range | Interpretation |
|-------------|----------------|
| ≥ 0.8 | STRONG_TREND |
| 0.6-0.8 | TRENDING_WITH_NOISE |
| 0.4-0.6 | MIXED_TRANSITIONAL |
| 0.2-0.4 | CHOPPY_CONSOLIDATING |
| < 0.2 | HIGHLY_CHOPPY |

##### `calculate_trending_score(self, df, window_start, window_end, preload_bars=0, include_diagnostics=True) -> MSSResult`
**Main entry point** for analysis.

**Steps**:
1. Calculate ATR (with preload for warm-up)
2. Compute main window normalized slope
3. Calculate sub-window slopes
4. Find prominent extrema
5. Calculate extrema path coherence
6. Combine into composite score

---

### Convenience Functions

##### `analyze_session(df, session_name='default', window_size=None) -> Dict`
Analyze a trading session with pre-configured settings.

**Session configs**:
| Session | Window Size | Preload |
|---------|-------------|---------|
| 3pm | 50 | 14 |
| london | 60 | 14 |
| default | 50 | 14 |

##### `batch_analyze_windows(df, window_size=50, step_size=1, preload_bars=14, include_diagnostics=False) -> pd.DataFrame`
Rolling window analysis for backtesting.

---

## Visual Explanation

### Multi-Scale Hierarchy

```
MAIN WINDOW (50 bars):
├──────────────────────────────────────────────────┤
main_slope = +0.15 (normalized)

SUB-WINDOWS (4 divisions):
├────────────┼────────────┼────────────┼───────────┤
   +0.18        +0.12        +0.08        +0.22
   
EXTREMA PATH:
Start ──▲ High ──▼ Low ──▲ End
        ↑           ↑
        prominent   prominent
        
→ All positive sub-slopes + coherent path = HIGH TRENDING SCORE
```

### Component Breakdown

```
trending_score = 0.75

├─ directional_consistency: 1.00 (4/4 subs positive)
│  Weight: 0.35 → Contribution: 0.35
│
├─ slope_alignment: 0.82 (subs tightly clustered)
│  Weight: 0.20 → Contribution: 0.164
│
├─ extrema_coherence: 0.68 (path supports trend)
│  Weight: 0.20 → Contribution: 0.136
│
└─ magnitude_score: 0.50 (moderate slope strength)
   Weight: 0.25 → Contribution: 0.125
```

---

## Shortcomings & Improvement Suggestions

### 1. **Missing `get_signal()` Method** ⚠️ MEDIUM PRIORITY
**Issue**: Class exists but lacks standardized signal output for `state_detector.py` integration.

**Improvement**:
- Add `get_signal() -> float` method returning `trending_score` directly (already 0-1)
- Add optional signed variant: `sign(main_slope) * trending_score`
- Implement `TrendIndicator` interface for consistency

### 2. **Fixed Number of Sub-Windows**
**Issue**: 4 sub-windows may not suit all window sizes; too few for long windows, too many for short.

**Improvement**:
- Adaptive sub-window count: `num_subs = max(3, window_size // 15)`
- Multi-resolution: compute at 2, 4, 8 divisions and ensemble

### 2. **Extrema Detection Sensitivity**
**Issue**: `prominence_lookback=3` may miss larger swings or catch noise.

**Improvement**:
- Adaptive lookback based on ATR/volatility
- Multi-scale extrema: detect at multiple lookback values

### 3. **Path Coherence Assumptions**
**Issue**: Assumes specific swing sequences (High→Low or Low→High).

**Improvement**:
- Handle multiple intermediate swings
- Score based on overall path efficiency, not just entry/exit

### 4. **Weight Rigidity**
**Issue**: Fixed weights may not be optimal across all market conditions.

**Improvement**:
- Add weight optimization framework
- Context-adaptive weights (e.g., increase magnitude weight in volatile markets)

### 5. **No Trend Direction Output**
**Issue**: Score measures quality, not direction.

**Improvement**:
- Add `trend_direction`: sign of main_slope (-1, 0, +1)
- Add `signed_trending_score`: direction * quality

### 6. **Tanh Saturation**
**Issue**: `tanh(slope * 5.0)` saturates quickly; may lose granularity for strong moves.

**Improvement**:
- Use softer scaling: `tanh(slope * 2.0)` or adaptive multiplier
- Report pre-saturation value in diagnostics

### 7. **No Time-Weighting**
**Issue**: All bars contribute equally; doesn't emphasize recent action.

**Improvement**:
- Add time-weighted sub-window scores (recent subs matter more)
- Exponential decay for older bars

### 8. **Preload Requirement**
**Issue**: Needs `preload_bars` for ATR warm-up; may not always be available.

**Improvement**:
- Implement ATR approximation for short histories
- Fall back to range-based volatility estimate

### 9. **No Confidence Metric**
**Issue**: Single score without uncertainty estimate.

**Improvement**:
- Add bootstrap confidence interval for trending_score
- Report component variance as confidence proxy

### 10. **Computational Cost**
**Issue**: Multiple passes over data; may be slow for batch analysis.

**Improvement**:
- Cache ATR and intermediate calculations
- Vectorize sub-window slope calculation

### 11. **Missing V1 vs V2 Switch**
**Issue**: V2.0 changes are hardcoded; no way to compare with original behavior.

**Improvement**:
- Add version parameter to use original formulas
- A/B testing support for scoring variants

---

## Usage Example

```python
import pandas as pd
from features.trend.MSS_tan.mss import MultiScaleSlope, analyze_session

# Single window analysis
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'timestamp': pd.date_range('2024-01-15 15:00', periods=70, freq='1min')
})

mss = MultiScaleSlope(num_subwindows=4)
result = mss.calculate_trending_score(
    df, 
    window_start=14,  # Start after preload
    window_end=63,    # 50-bar window
    preload_bars=14
)

print(f"Trending Score: {result.trending_score:.3f}")
print(f"Interpretation: {result.interpretation}")
print(f"Main Slope: {result.main_slope:.6f}")
print(f"Sub-Slopes: {result.sub_slopes}")
print(f"Components: {result.components}")

# Quick session analysis
session_result = analyze_session(df, session_name='3pm')
```

---

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| ADX | Both measure trend strength; MSS is slope-based, ADX is DI-based |
| ATR Range | MSS uses ATR internally; could share ATR calculation |
| Lag/Hurst | Hurst = multi-scale variance; MSS = multi-scale slope; complementary |
| IRR | IRR = intraperiod (bar-level) reversion; MSS = window-level slope; different granularity |

---

## Version History

- **V1.0**: Original implementation with binary scoring
- **V2.0**: Improved calibration for real market data
  - Exponential decay slope alignment (vs hard cutoff)
  - Continuous extrema coherence (vs binary)
  - Adaptive magnitude scaling
  - Floor values to prevent collapse
  - Diagnostic output for calibration

