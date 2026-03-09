# SPD (Swing Point Density) Module

## State Detector Integration Status

**Readiness Score: 50% ⚠️ NEEDS WORK**

**Current Status:**
- ❌ Function-based (not class-based)
- ⚠️ List[Dict] input format (not DataFrame)
- ✅ Clear swing detection logic
- ✅ Time-window aware
- ❌ No class structure
- ❌ Missing `TrendIndicator` interface

**Required Changes for Integration:**

1. **Create `SwingDensityAnalyzer` Class (90 min)**
   ```python
   class SwingDensityAnalyzer:
       """Swing point density analyzer for trend detection."""
       
       def __init__(self, lookback: int = 1, normalize: bool = True):
           """
           Args:
               lookback: Bars on each side for swing detection
               normalize: Normalize count by bar count
           """
           self.lookback = lookback
           self.normalize = normalize
       
       def detect_swing_highs(self, df: pd.DataFrame) -> List[int]:
           """Detect swing highs in DataFrame."""
           highs = df['high'].values
           swings = []
           
           for i in range(self.lookback, len(highs) - self.lookback):
               is_swing = True
               for j in range(1, self.lookback + 1):
                   if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                       is_swing = False
                       break
               if is_swing:
                   swings.append(i)
           
           return swings
       
       def detect_swing_lows(self, df: pd.DataFrame) -> List[int]:
           """Detect swing lows in DataFrame."""
           lows = df['low'].values
           swings = []
           
           for i in range(self.lookback, len(lows) - self.lookback):
               is_swing = True
               for j in range(1, self.lookback + 1):
                   if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                       is_swing = False
                       break
               if is_swing:
                   swings.append(i)
           
           return swings
       
       def calculate_density(self, df: pd.DataFrame) -> Dict:
           """Calculate swing density for DataFrame."""
           if len(df) < 2 * self.lookback + 1:
               return {
                   'count': 0,
                   'density': 0.0,
                   'classification': 'insufficient_data',
                   'swing_high_count': 0,
                   'swing_low_count': 0
               }
           
           swing_highs = self.detect_swing_highs(df)
           swing_lows = self.detect_swing_lows(df)
           
           total_count = len(swing_highs) + len(swing_lows)
           
           if self.normalize:
               density = total_count / len(df)
           else:
               density = total_count
           
           # Classify (normalized thresholds)
           if self.normalize:
               threshold_low = 0.1  # 10% of bars are swings
               threshold_high = 0.15  # 15% of bars are swings
           else:
               threshold_low = 5
               threshold_high = 8
           
           if density < threshold_low:
               classification = 'trending'
           elif density > threshold_high:
               classification = 'chop'
           else:
               classification = 'mixed'
           
           return {
               'count': total_count,
               'density': density,
               'swing_high_count': len(swing_highs),
               'swing_low_count': len(swing_lows),
               'classification': classification,
               'swing_highs': swing_highs,
               'swing_lows': swing_lows
           }
   ```

2. **Create `SPDIndicator` Wrapper (30 min)**
   ```python
   class SPDIndicator(TrendIndicator):
       def __init__(self, config=None):
           lookback = config.get('lookback', 1) if config else 1
           self.analyzer = SwingDensityAnalyzer(
               lookback=lookback, 
               normalize=True
           )
           self._last_result = None
       
       def calculate(self, df: pd.DataFrame, 
                    session: SessionName = "auto") -> IndicatorResult:
           result = self.analyzer.calculate_density(df)
           
           # Invert density: low density = trending → high signal
           if result['classification'] == 'insufficient_data':
               signal = 0.5
           else:
               density = result['density']
               # Map density to signal (inverted and clamped)
               # High density (0.2) → low signal (0.0)
               # Low density (0.05) → high signal (1.0)
               signal = max(0.0, min(1.0, 1.0 - (density * 5)))
           
           self._last_result = IndicatorResult(
               signal=signal,
               raw_value=result['density'],
               metadata=result
           )
           return self._last_result
       
       def get_signal(self) -> float:
           return self._last_result.signal if self._last_result else 0.5
   ```

3. **Signal Mapping:**
   - Invert density: **low density → high trending signal**
   - Normalize density by bar count for consistency
   - Suggested mapping: `signal = 1.0 - (density * 5)` clamped to [0,1]

4. **DataFrame Migration:**
   - Refactor to use DataFrame with 'high'/'low' columns
   - Remove time-filtering logic (state_detector provides pre-filtered data)
   - Keep original functions for backward compatibility

5. **Enhancement Opportunities:**
   - Configurable lookback (currently hardcoded to 1)
   - ATR-based amplitude filtering
   - Multi-scale swing detection

**Integration Priority: MEDIUM** - Unique signal, but requires significant refactor.

---

## Overview

SPD detects **choppy vs smooth price action** by counting local swing highs and lows within a time window. The core insight: trending markets have few reversals (low swing density), while consolidating markets oscillate frequently (high swing density).

**Core Concept**: Count how many times price forms a local high or low. More swings = more chop = consolidation.

---

## File Structure

| File | Purpose |
|------|---------|
| `spd.py` | Swing point detection and density classification |

---

## Configuration

### Time Windows (`WINDOWS`)

| Window | Time Range | Purpose |
|--------|------------|---------|
| `early_afternoon` | 1:00 PM - 3:00 PM | Main session analysis |
| `late_afternoon` | 3:00 PM - 3:50 PM | Pre-close behavior |
| `close` | 3:50 PM - 4:00 PM | Final auction period |

### Density Thresholds (`THRESHOLDS`)

| Threshold | Value | Classification |
|-----------|-------|----------------|
| `trending` | < 5 swings | Smooth directional movement |
| `chop` | > 8 swings | Choppy consolidation |
| (between) | 5-8 swings | Mixed/transitional |

---

## Function Reference

### `spd.py`

#### `detect_swing_highs(bars) -> List[int]`
Detect swing highs using 3-bar pattern.

**Pattern**: `high[i-1] < high[i] > high[i+1]`

**Returns**: List of bar indices where swing highs occur.

```
        ●  ← Swing High (bar i)
       / \
      ●   ●
    i-1   i+1
```

#### `detect_swing_lows(bars) -> List[int]`
Detect swing lows using 3-bar pattern.

**Pattern**: `low[i-1] > low[i] < low[i+1]`

**Returns**: List of bar indices where swing lows occur.

```
      ●   ●
       \ /
        ●  ← Swing Low (bar i)
```

#### `filter_bars_by_time(bars, start_time, end_time) -> List[Dict]`
Filter bars to those within specified time window.

**Input**: Bars with `'time'` or `'timestamp'` key (handles both datetime and time objects).

**Returns**: Filtered list of bars within `[start_time, end_time)`.

#### `classify_density(count) -> str`
Classify market state based on total swing count.

| Count | Classification |
|-------|----------------|
| < 5 | `'trending'` |
| 5-8 | `'mixed'` |
| > 8 | `'chop'` |

#### `get_swing_density(bars, start_time, end_time) -> Dict`
Calculate swing point density for a specific time window.

**Returns**:
```python
{
    'count': int,              # Total swings (highs + lows)
    'swing_highs': List[int],  # Indices of swing highs
    'swing_lows': List[int],   # Indices of swing lows
    'swing_high_count': int,
    'swing_low_count': int,
    'classification': str,     # 'trending', 'mixed', 'chop', or 'insufficient_data'
    'bars_analyzed': int
}
```

**Edge case**: Returns `'insufficient_data'` if fewer than 3 bars in window.

#### `analyze_all_windows(bars) -> Dict[str, Dict]`
Analyze swing density for all configured time windows.

**Returns**: Dictionary keyed by window name with density results for each.

---

## Visual Explanation

### Low Density (Trending)
```
Price
  │         ●
  │        /
  │       /
  │      ●
  │     /
  │    /
  │   ●
  └────────────── Time
  
Swing points: 0 highs, 0 lows = 0 total
→ Classification: TRENDING
```

### High Density (Choppy)
```
Price
  │    ●     ●     ●
  │   / \   / \   / \
  │  /   \ /   \ /   \
  │ ●     ●     ●     ●
  └────────────────────── Time
  
Swing points: 3 highs, 4 lows = 7 total
→ Classification: MIXED (borderline CHOP)
```

---

## Shortcomings & Improvement Suggestions

### 1. **Not Class-Based** ⚠️ HIGH PRIORITY
**Issue**: Module uses standalone functions; not compatible with planned `state_detector.py` architecture.

**Improvement**:
- Create `SPDIndicator` class implementing `TrendIndicator` interface
- Add `calculate(df) -> SPDResult` method
- Add `get_signal() -> float` returning normalized density (inverted: low density = high trending signal)

### 2. **Fixed 3-Bar Pattern**
**Issue**: Only detects immediate reversals; misses broader swing structure.

**Improvement**:
- Add configurable lookback: `detect_swing_highs(bars, lookback=1)`
- Multi-scale detection: count swings at multiple lookback values
- Use prominence-based detection (like MSS module)

### 3. **Raw Count vs Normalized**
**Issue**: Raw swing count doesn't account for window size; 5 swings in 10 bars ≠ 5 swings in 100 bars.

**Improvement**:
- Add `density_ratio = count / bars_analyzed`
- Normalize thresholds based on bar count
- Report swings-per-bar metric

### 4. **Arbitrary Thresholds**
**Issue**: Thresholds (5, 8) are hardcoded without empirical validation.

**Improvement**:
- Calibrate from historical data
- Use percentile-based thresholds
- Add threshold configuration parameter

### 5. **No Direction Information**
**Issue**: Counts swings but doesn't indicate trend direction.

**Improvement**:
- Track net direction: `(last_price - first_price) / range`
- Add `swing_bias`: more highs = uptrend exhaustion, more lows = downtrend exhaustion
- Detect swing sequence patterns (higher highs, lower lows)

### 6. **Equal vs Strict Comparison**
**Issue**: Uses strict inequality (`<`, `>`); equal highs/lows are missed.

**Improvement**:
- Add option for `<=`, `>=` comparisons
- Handle plateaus (consecutive equal values)

### 7. **No Swing Amplitude Filtering**
**Issue**: All swings counted equally; tiny noise swings = significant reversals.

**Improvement**:
- Filter by minimum amplitude (ATR-based threshold)
- Weight swings by prominence
- Distinguish "significant" vs "minor" swings

### 8. **Time Handling Assumptions**
**Issue**: Assumes bars have `'time'` or `'timestamp'` key with specific format.

**Improvement**:
- Add DataFrame support with configurable column names
- Handle timezone-aware timestamps
- Validate time format on input

### 9. **No Swing Timing Analysis**
**Issue**: Only counts total swings; doesn't analyze when they occur.

**Improvement**:
- Track swing clustering (many swings in short period = high chop)
- Detect swing acceleration/deceleration
- Report time between swings

### 10. **Missing Integration with Price Levels**
**Issue**: Doesn't consider where swings occur relative to key levels.

**Improvement**:
- Combine with DRA: swings at range boundaries vs mid-range
- Track if swings respect support/resistance levels

---

## Usage Example

```python
from features.trend.Swing_Point_Density.spd import (
    get_swing_density,
    analyze_all_windows,
    WINDOWS
)
from datetime import time

# Single window analysis
bars = [
    {'high': 101, 'low': 99, 'time': time(13, 0)},
    {'high': 102, 'low': 100, 'time': time(13, 1)},
    {'high': 101, 'low': 99, 'time': time(13, 2)},  # Swing high at index 1
    # ... more bars
]

result = get_swing_density(
    bars,
    start_time=time(13, 0),
    end_time=time(15, 0)
)

print(f"Total swings: {result['count']}")
print(f"Classification: {result['classification']}")

# Full day analysis
all_results = analyze_all_windows(bars)
for window, data in all_results.items():
    print(f"{data['window_name']}: {data['classification']} ({data['count']} swings)")
```

---

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| ADX | Both detect chop; ADX uses DI spread, SPD uses swing count |
| IRR | IRR = within-bar reversion; SPD = between-bar reversals; complementary |
| MSS | MSS uses prominent extrema; SPD counts all swings; different granularity |
| DRA | SPD can enhance DRA: high density at range = consolidation confirmed |

---

## Theoretical Background

Swing Point Density is based on the observation that:

1. **Trending markets** move in sustained directions with few reversals
2. **Consolidating markets** oscillate frequently, creating many swing points
3. **The transition** from consolidation to trend often shows decreasing swing density

This is related to:
- **Market microstructure**: Order flow imbalance creates trends
- **Technical analysis**: Swing highs/lows are key structural points
- **Volatility clustering**: Chop often precedes breakouts

### Connection to Other Concepts

| Concept | SPD Equivalent |
|---------|----------------|
| Fractal patterns | 3-bar swing detection |
| Williams %R overbought/oversold | Swing high/low formation |
| Donchian channels | Swing high = channel top test |

