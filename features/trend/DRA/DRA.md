# DRA (Dynamic Range Analysis) Module

## State Detector Integration Status

**Readiness Score: 70% ⚠️ PARTIALLY READY**

**Current Status:**
- ✅ Class-based architecture (`DRA`)
- ✅ DataFrame and object support
- ✅ Streaming/stateful design
- ⚠️ No `get_signal()` method
- ⚠️ Missing `TrendIndicator` interface
- ⚠️ Requires separate reference range setup

**Required Changes for Integration:**

1. **Add `get_signal()` Method to DRA Class (15 min)**
   ```python
   def get_signal(self) -> float:
       """
       Get trending signal (inverted overlap).
       
       Returns:
           Breakout signal (0-1): 0 = inside range, 1 = full breakout
       """
       if not self.overlaps:
           return 0.5  # Neutral if no data
       
       recent = self.overlaps[-self.window:]
       rolling_overlap = sum(recent) / len(recent)
       
       # Invert: low overlap = high breakout signal
       return 1.0 - rolling_overlap
   ```

2. **Create `DRAIndicator` Wrapper (45 min)**
   ```python
   class DRAIndicator(TrendIndicator):
       def __init__(self, config=None):
           self.dra = DRA(window=config.get('window', 10))
           self._reference_set = False
       
       def calculate(self, df: pd.DataFrame, 
                    reference_bars: Optional[pd.DataFrame] = None) -> IndicatorResult:
           # Set reference range if not already done
           if not self._reference_set and reference_bars is not None:
               self.dra.set_initial_range(reference_bars)
               self._reference_set = True
           elif not self._reference_set:
               # Use first 15 bars as default reference
               self.dra.set_initial_range(df.iloc[:15])
               self._reference_set = True
           
           # Process remaining bars
           for _, bar in df.iloc[15:].iterrows():
               self.dra.update(bar)
           
           signal = self.dra.get_signal()
           
           return IndicatorResult(
               signal=signal,
               raw_value=self.dra.overlaps[-1] if self.dra.overlaps else None,
               metadata={
                   'initial_range': self.dra.initial_range,
                   'num_overlaps': len(self.dra.overlaps)
               }
           )
       
       def get_signal(self) -> float:
           return self.dra.get_signal()
   ```

3. **Signal Mapping:**
   - Inverted overlap: `1.0 - rolling_overlap`
   - 0.0 = fully inside range (consolidation)
   - 1.0 = fully outside range (trending/breakout)

4. **Reference Range Handling:**
   - state_detector.py needs to pass `reference_bars` parameter
   - Default to first 15 bars if not provided
   - Consider adding reset method for multi-session analysis

**Integration Priority: MEDIUM** - Unique range-based approach, complements other indicators.

---

## Overview

DRA tracks **overlap between current price bars and an initial reference range**. It answers the question: "Is price staying within the opening range or breaking out?"

**Core Concept**: Establish a reference range (e.g., 3pm-3:10pm), then measure how much each subsequent bar overlaps with that range. High overlap indicates consolidation within range; low overlap indicates breakout/trending behavior.

---

## File Structure

| File | Purpose |
|------|---------|
| `dra.py` | DRA class for streaming range overlap analysis |

---

## Function Reference

### `dra.py`

#### Configuration Constants

| Constant | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_WINDOW` | 10 | Bars for rolling average calculation |

#### Class: `DRA`

A stateful class for tracking range overlap over time.

##### `__init__(self, window=10)`
Initialize DRA calculator.

- **window**: Number of bars for rolling average (default: 10)

**Instance Attributes**:
- `initial_high`: High of reference range
- `initial_low`: Low of reference range
- `initial_range`: Reference range size
- `overlaps`: List of all overlap values
- `window`: Rolling average window size

##### `set_initial_range(self, bars_3pm_to_310pm)`
Set the reference range from initial bars.

**Input Options**:
1. `pd.DataFrame` with 'high' and 'low' columns
2. List of bar objects with `.high` and `.low` attributes

**Calculation**:
```python
initial_high = max(all highs)
initial_low = min(all lows)
initial_range = initial_high - initial_low
```

##### `update(self, bar) -> float`
Process a new bar and return rolling average overlap.

**Overlap Calculation**:
```python
overlap_high = min(bar_high, initial_high)
overlap_low = max(bar_low, initial_low)
overlap = (overlap_high - overlap_low) / initial_range
```

**Returns**: Rolling average of last `window` overlap values

**Interpretation**:
- **1.0**: Bar completely within reference range (consolidating)
- **0.0**: Bar completely outside reference range (breakout)
- **0.5**: Bar half-in, half-out

---

## Visual Explanation

```
Reference Range (3pm-3:10pm):
├─────────────────────────┤
initial_low              initial_high

Bar completely INSIDE (overlap = 1.0):
├─────────────────────────┤  Range
    ├─────────┤              Bar
    
Bar partially OUTSIDE (overlap = 0.6):
├─────────────────────────┤  Range
              ├─────────────────┤  Bar
              ↑overlap zone↑
              
Bar completely OUTSIDE (overlap = 0.0):
├─────────────────────────┤  Range
                               ├─────────┤  Bar
```

---

## Shortcomings & Improvement Suggestions

### 1. **Missing `get_signal()` Method** ⚠️ MEDIUM PRIORITY
**Issue**: Class exists but lacks standardized signal output for `state_detector.py` integration.

**Improvement**:
- Add `get_signal() -> float` method returning inverted overlap (1 - overlap = breakout strength)
- Implement `TrendIndicator` interface for consistency

### 2. **Fixed Reference Range**
**Issue**: Once `set_initial_range` is called, the reference never updates. Markets that gradually expand don't get captured.

**Improvement**:
- Add `update_range` method for dynamic range expansion
- Implement "expanding range" variant that tracks max/min seen
- Add option to re-anchor range at key breakout points

### 2. **Simple Rolling Average**
**Issue**: All bars in window equally weighted; recent action not emphasized.

**Improvement**:
- Add exponential moving average variant for overlap
- Implement decay-weighted average (recent bars count more)

### 3. **No Direction Awareness**
**Issue**: Overlap is unsigned; doesn't distinguish breakout above vs below range.

**Improvement**:
- Add signed overlap: positive for breakouts above, negative for below
- Track separate metrics: `overlap_above_range`, `overlap_below_range`

### 4. **No Failed Breakout Detection**
**Issue**: If price breaks out then returns, this is treated same as gradual drift.

**Improvement**:
- Track breakout attempts (first bar with overlap < threshold)
- Measure "breakout persistence" - how long price stays outside range
- Detect failed breakouts: overlap → low → back to high

### 5. **Edge Case: Zero Initial Range**
**Issue**: Returns 0.0 for all subsequent bars if initial range is zero (flat open).

**Improvement**:
- Use minimum range threshold (e.g., 1 tick)
- Fall back to ATR-based range if initial range too small

### 6. **No State Persistence**
**Issue**: `overlaps` list grows unbounded for long sessions.

**Improvement**:
- Add `max_history` parameter to limit memory usage
- Implement circular buffer for overlap history

### 7. **Missing Aggregate Metrics**
**Issue**: Only returns rolling average; no other summary statistics.

**Improvement**:
- Add `get_stats()` method returning min, max, std of overlaps
- Track time spent inside vs outside range
- Add "range efficiency" metric: how clean was the breakout?

### 8. **No Integration with Other Signals**
**Issue**: Standalone metric without connection to trend confirmation.

**Improvement**:
- Combine with ADX: low overlap + high ADX = confirmed breakout
- Add method to classify: `INSIDE_RANGE`, `TESTING_BREAKOUT`, `CONFIRMED_BREAKOUT`

---

## Usage Example

```python
import pandas as pd
from features.trend.DRA.dra import DRA

# Initialize DRA
dra = DRA(window=10)

# Set reference range from 3pm-3:10pm bars
opening_bars = pd.DataFrame({
    'high': [100.5, 101.0, 100.8, 101.2, 100.9],
    'low': [99.5, 99.8, 99.6, 100.0, 99.7]
})
dra.set_initial_range(opening_bars)

# Process subsequent bars
for _, bar in subsequent_bars.iterrows():
    rolling_overlap = dra.update(bar)
    
    if rolling_overlap < 0.3:
        print("Breakout in progress")
    elif rolling_overlap > 0.7:
        print("Consolidating within range")
```

---

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| ATR/Range | DRA uses range explicitly; ATR/Range uses ratio approach |
| ADX | DRA detects breakout, ADX confirms trend strength post-breakout |
| MSS | Both analyze trend structure; DRA is range-centric, MSS is slope-centric |

