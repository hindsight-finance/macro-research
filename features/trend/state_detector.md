# State Detector Documentation

## Overview

The `StateDetector` is a unified system for detecting market regime by combining signals from 7 complementary trend identification modules. It provides a single interface to classify market state as **STRONG_TREND**, **WEAK_TREND**, **CONSOLIDATION**, **CHOPPY**, or **UNCERTAIN**.

### Purpose

Instead of manually combining signals from multiple modules, `StateDetector`:
- Runs all enabled indicators in parallel
- Normalizes their outputs to a common 0-1 scale
- Combines them using weighted averaging
- Classifies the final state with confidence scoring
- Handles failures gracefully with automatic reweighting

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    StateDetector                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ...        │
│  │   ADX    │  │  ATR/    │  │   DRA    │             │
│  │Indicator │  │ Range    │  │Indicator │             │
│  └──────────┘  └──────────┘  └──────────┘             │
│       │             │             │                    │
│       └─────────────┴─────────────┘                    │
│                    │                                    │
│              ┌─────▼─────┐                             │
│              │  Signals  │                             │
│              │  (0-1)    │                             │
│              └─────┬─────┘                             │
│                    │                                    │
│              ┌─────▼─────┐                             │
│              │ Weighted  │                             │
│              │  Average  │                             │
│              └─────┬─────┘                             │
│                    │                                    │
│         ┌──────────┴──────────┐                        │
│         │                     │                        │
│    ┌────▼────┐         ┌─────▼─────┐                 │
│    │ State   │         │Confidence │                  │
│    │Classification│    │  Score    │                  │
│    └─────────┘         └───────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### Type Definitions

```python
MarketState = Literal["STRONG_TREND", "WEAK_TREND", "CONSOLIDATION", "CHOPPY", "UNCERTAIN"]
TrendDirection = Literal["UP", "DOWN", "NEUTRAL"]
SessionName = Literal["1pm-3pm", "3pm-3:50pm", "3:50pm-4pm", "auto"]
```

### Dataclasses

#### `IndicatorResult`
Result from a single trend indicator calculation.

| Field | Type | Description |
|-------|------|-------------|
| `signal` | `float` | Normalized signal (0-1, higher = more trending) |
| `raw_value` | `Optional[float]` | Raw indicator value for reference |
| `metadata` | `Dict` | Additional indicator-specific information |
| `error` | `Optional[str]` | Error message if calculation failed |

#### `StateResult`
Complete market state detection result.

| Field | Type | Description |
|-------|------|-------------|
| `state` | `MarketState` | Classified market state |
| `direction` | `TrendDirection` | Trend direction (UP/DOWN/NEUTRAL) |
| `confidence` | `float` | Confidence score (0-1) |
| `signals` | `Dict[str, float]` | Individual indicator signals |
| `weights` | `Dict[str, float]` | Actual weights used (after reweighting) |
| `warnings` | `List[str]` | Warnings from calculation |
| `metadata` | `Dict` | Additional context (session, failed indicators, etc.) |

---

## Base Classes

### `TrendIndicator` (Abstract Base Class)

All indicator wrappers must implement this interface:

```python
class TrendIndicator(ABC):
    @abstractmethod
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional configuration."""
        pass
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate indicator from OHLC DataFrame."""
        pass
    
    @abstractmethod
    def get_signal(self) -> float:
        """Get normalized signal from last calculation."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return indicator name."""
        pass
```

---

## Indicator Wrappers

### Current Status: **STUBS** (To Be Implemented)

All 7 indicator wrappers are currently placeholder implementations. Each has TODO comments indicating what function to call from the existing modules.

| Indicator | Module | Status | Signal Normalization |
|-----------|--------|--------|---------------------|
| `ADXIndicator` | `ADX/trend_quality.py` | ⚠️ Stub | `quality_score` (0-1) |
| `ATRRangeIndicator` | `ATR Range/atr.py` | ⚠️ Stub | `1 - ratio` (inverted) |
| `DRAIndicator` | `DRA/dra.py` | ⚠️ Stub | `1 - overlap` (inverted) |
| `IRRIndicator` | `IRR/irr.py` | ⚠️ Stub | `1 - median_irr` (inverted) |
| `LagIndicator` | `Lag autocorr/lag.py` | ⚠️ Stub | `(hurst - 0.5) * 2` normalized |
| `MSSIndicator` | `MSS tan/mss.py` | ⚠️ Stub | `trending_score` (0-1) |
| `SPDIndicator` | `Swing Point Density/spd.py` | ⚠️ Stub | `1 - density_ratio` (inverted) |

**Signal Normalization Philosophy**: All signals are normalized to 0-1 scale where:
- **Higher (closer to 1.0)** = More trending
- **Lower (closer to 0.0)** = More consolidating/choppy

This allows direct weighted averaging across all indicators.

---

## Main Classes

### `StateDetector`

Main class for unified state detection.

#### Initialization

```python
detector = StateDetector(
    config: Optional[Dict] = None,
    enabled_indicators: Optional[Set[str]] = None,
    confidence_threshold: float = 0.3
)
```

**Parameters**:
- `config`: Optional configuration dict for individual indicators
  ```python
  config = {
      'adx': {'period': 14},
      'mss': {'num_subwindows': 4},
      'weights': {...}  # Override default weights
  }
  ```
- `enabled_indicators`: Set of indicator names to use (None = all)
  ```python
  enabled_indicators = {'adx', 'mss', 'spd'}  # Only use these 3
  ```
- `confidence_threshold`: Minimum confidence to return a state (default: 0.3)
  - If confidence < threshold, returns `UNCERTAIN`

#### Default Weights

```python
DEFAULT_WEIGHTS = {
    'adx': 0.18,        # 18%
    'atr_range': 0.10,  # 10%
    'dra': 0.12,        # 12%
    'irr': 0.12,        # 12%
    'lag': 0.12,        # 12%
    'mss': 0.24,        # 24% (highest - most sophisticated)
    'spd': 0.12,        # 12%
}
```

#### Main Method: `detect()`

```python
result = detector.detect(
    df: pd.DataFrame,
    session: SessionName = "auto",
    reference_bars: Optional[pd.DataFrame] = None
) -> StateResult
```

**Parameters**:
- `df`: DataFrame with `'open'`, `'high'`, `'low'`, `'close'`, `'timestamp'` columns
- `session`: Session name (`"1pm-3pm"`, `"3pm-3:50pm"`, `"3:50pm-4pm"`) or `"auto"` for auto-detection
- `reference_bars`: Optional reference bars for DRA (defaults to first 15 bars)

**Returns**: `StateResult` object

---

## Helper Functions

### `detect_session(df, timestamp_col='timestamp') -> SessionName`

Auto-detects session from DataFrame timestamps.

**Logic**:
- Checks time range of data
- Returns `"1pm-3pm"`, `"3pm-3:50pm"`, `"3:50pm-4pm"`, or `"auto"` if cannot determine

### `classify_state(signals, weights, confidence_threshold=0.3) -> Tuple[MarketState, float]`

Classifies market state from weighted signals.

**Classification Logic**:

| Composite Signal | State | Confidence Calculation |
|-----------------|-------|------------------------|
| ≥ 0.7 | `STRONG_TREND` | `abs(signal - 0.5) * 2` |
| 0.55 - 0.7 | `WEAK_TREND` | (maps 0.5→0, 0/1→1) |
| 0.45 - 0.55 | `WEAK_TREND` | |
| 0.3 - 0.45 | `CONSOLIDATION` | |
| < 0.3 | `CHOPPY` | |
| (any) | `UNCERTAIN` | If confidence < threshold |

### `determine_direction(df) -> TrendDirection`

Determines trend direction from price movement.

**Logic**:
- Calculates `(close[-1] - close[0]) / close[0]`
- Returns `UP` if change > 0.1%
- Returns `DOWN` if change < -0.1%
- Returns `NEUTRAL` otherwise

### `detect_state()` (Convenience Function)

One-off state detection without creating a detector instance.

```python
result = detect_state(
    df,
    session="auto",
    reference_bars=None,
    enabled_indicators=None,
    confidence_threshold=0.3
)
```

---

## Usage Examples

### Basic Usage

```python
import pandas as pd
from features.trend.state_detector import StateDetector

# Load OHLC data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'timestamp': pd.date_range('2024-01-15 15:00', periods=50, freq='2min')
})

# Create detector
detector = StateDetector()

# Detect state
result = detector.detect(df)

# Access results
print(f"State: {result.state}")           # e.g., "STRONG_TREND"
print(f"Direction: {result.direction}")   # e.g., "UP"
print(f"Confidence: {result.confidence:.2f}")  # e.g., 0.85
print(f"Signals: {result.signals}")       # Individual indicator values
print(f"Warnings: {result.warnings}")     # Any calculation warnings
```

### Custom Configuration

```python
# Custom weights and indicator config
config = {
    'weights': {
        'adx': 0.25,
        'mss': 0.30,
        'spd': 0.15,
        # ... other indicators
    },
    'adx': {'period': 14},
    'mss': {'num_subwindows': 6}
}

detector = StateDetector(
    config=config,
    confidence_threshold=0.4  # Higher threshold = more conservative
)
```

### Selective Indicators

```python
# Only use ADX, MSS, and SPD
detector = StateDetector(
    enabled_indicators={'adx', 'mss', 'spd'}
)

result = detector.detect(df)
# Weights automatically renormalize to sum to 1.0
```

### Explicit Session

```python
# Force specific session (bypasses auto-detection)
result = detector.detect(df, session="3pm-3:50pm")
```

### Custom DRA Reference

```python
# Use custom reference bars for DRA
reference_bars = df.iloc[10:25]  # Bars 10-24 as reference

result = detector.detect(df, reference_bars=reference_bars)
```

### Convenience Function

```python
from features.trend.state_detector import detect_state

# One-liner for quick analysis
result = detect_state(df, session="auto", confidence_threshold=0.3)
```

---

## Error Handling

### Graceful Degradation

If an indicator fails:
1. **Warning is logged** in `result.warnings`
2. **Indicator is skipped** (not included in signals)
3. **Weights are automatically reweighted** to sum to 1.0
4. **Detection continues** with remaining indicators

### All Indicators Fail

If all indicators fail:
- Returns `StateResult` with:
  - `state = "UNCERTAIN"`
  - `confidence = 0.0`
  - `signals = {}`
  - `warnings` includes "All indicators failed"

### Input Validation

- Raises `ValueError` if required columns (`open`, `high`, `low`, `close`) are missing
- Warns if weights don't sum to 1.0 (auto-normalizes)

---

## State Classification Details

### Signal Combination

1. **Calculate weighted average**:
   ```python
   composite_signal = Σ(signal[i] * weight[i]) / Σ(weight[i])
   ```

2. **Calculate confidence**:
   ```python
   confidence = abs(composite_signal - 0.5) * 2
   ```
   - Maximum confidence (1.0) at extremes (0 or 1)
   - Minimum confidence (0.0) at neutral (0.5)

3. **Classify state** based on composite signal thresholds

4. **Check confidence threshold** - if below threshold, return `UNCERTAIN`

### State Thresholds

| State | Signal Range | Typical Interpretation |
|-------|-------------|------------------------|
| `STRONG_TREND` | ≥ 0.7 | Clear directional movement, high confidence |
| `WEAK_TREND` | 0.55 - 0.7 | Directional but noisy or weak |
| `WEAK_TREND` | 0.45 - 0.55 | Borderline trending |
| `CONSOLIDATION` | 0.3 - 0.45 | Range-bound, mean-reverting |
| `CHOPPY` | < 0.3 | Frequent reversals, no clear direction |
| `UNCERTAIN` | Any | Confidence too low to classify |

---

## Integration Status

### ✅ Completed

- [x] Architecture design
- [x] Base classes and interfaces
- [x] State classification logic
- [x] Error handling and reweighting
- [x] Session auto-detection
- [x] Direction determination
- [x] Convenience functions

### ⚠️ Pending Implementation

- [ ] `ADXIndicator.calculate()` - Wrap `calculate_trend_quality()`
- [ ] `ATRRangeIndicator.calculate()` - Wrap `analyze_session()`
- [ ] `DRAIndicator.calculate()` - Wrap `DRA` class
- [ ] `IRRIndicator.calculate()` - Wrap `compute_session_window_irr()`
- [ ] `LagIndicator.calculate()` - Wrap `IntradayTemporalFeatures`
- [ ] `MSSIndicator.calculate()` - Wrap `MultiScaleSlope.calculate_trending_score()`
- [ ] `SPDIndicator.calculate()` - Wrap `get_swing_density()`

### 🔮 Future Enhancements

- [ ] Session-specific weight optimization
- [ ] Caching for shared calculations (ATR, etc.)
- [ ] Streaming/real-time support
- [ ] Backtesting framework integration
- [ ] Confidence interval calculation
- [ ] State transition detection

---

## Implementation Guide

### Adding a New Indicator

1. **Create wrapper class** inheriting from `TrendIndicator`:
   ```python
   class NewIndicator(TrendIndicator):
       def __init__(self, config: Optional[Dict] = None):
           self.config = config or {}
           self._last_result: Optional[IndicatorResult] = None
       
       @property
       def name(self) -> str:
           return "new_indicator"
       
       def calculate(self, df: pd.DataFrame) -> IndicatorResult:
           # Call your module's function
           raw_value = your_module_function(df)
           
           # Normalize to 0-1 (higher = more trending)
           signal = normalize_function(raw_value)
           
           self._last_result = IndicatorResult(
               signal=signal,
               raw_value=raw_value,
               metadata={...}
           )
           return self._last_result
       
       def get_signal(self) -> float:
           if self._last_result is None:
               return 0.0
           return self._last_result.signal
   ```

2. **Add to StateDetector**:
   ```python
   self.indicators['new_indicator'] = NewIndicator(self.config.get('new_indicator'))
   ```

3. **Add to DEFAULT_WEIGHTS** (ensure sum = 1.0)

---

## Testing Checklist

Before using in production:

- [ ] All 7 indicator wrappers implemented
- [ ] Test with sample data for each session
- [ ] Verify signal normalization (all 0-1, higher = trending)
- [ ] Test error handling (missing data, failed indicators)
- [ ] Validate reweighting logic
- [ ] Check confidence threshold behavior
- [ ] Test session auto-detection
- [ ] Verify direction determination
- [ ] Test with various data sizes (edge cases)

---

## Notes

- **Signal Normalization**: All indicators must output 0-1 where higher = more trending. Inverted signals (ATR/Range, DRA, IRR, SPD) are handled in their wrappers.
- **Reweighting**: Automatic reweighting ensures weights always sum to 1.0 even when indicators fail.
- **Confidence**: Based on distance from neutral (0.5), not absolute signal value.
- **Session Awareness**: Some indicators (ADX, SPD) use session-specific configurations.
- **DRA Reference**: Defaults to first 15 bars if not provided, but can be customized.

---

## References

- See individual module documentation:
  - `ADX.md` - ADX module
  - `ATR Range.md` - ATR/Range module
  - `DRA.md` - DRA module
  - `IRR.md` - IRR module
  - `Lag autocorr.md` - Lag/Hurst module
  - `MSS tan.md` - MSS module
  - `spd.md` - SPD module
- `SYSTEM_OVERVIEW.md` - System-wide architecture

