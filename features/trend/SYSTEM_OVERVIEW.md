# Trend Identification System - Complete Overview

## State Detector Integration Readiness

**Overall System Readiness: 76% (4/7 modules need work)** ⬆️ Revised Up

**Last Updated: 2026-01-21** (Revised after ADX analysis)

| Module | Readiness | Status | Architecture | Priority | Est. Effort |
|--------|-----------|--------|--------------|----------|-------------|
| **IRR** | 95% ✅ | READY | Class-based (IRRAnalyzer) | HIGH | 30 min |
| **MSS** | 90% ✅ | READY | Class-based (MultiScaleSlope) | HIGH | 30 min |
| **ADX** | 85% ✅ | READY | Composite system (trend_quality) | HIGH | 45 min |
| **Lag/Hurst** | 75% ⚠️ | PARTIAL | Class-based (IntradayTemporalFeatures) | MED-HIGH | 45 min |
| **DRA** | 70% ⚠️ | PARTIAL | Class-based (DRA) | MEDIUM | 60 min |
| **ATR/Range** | 50% ⚠️ | NEEDS WORK | Function-based | MEDIUM | 90 min |
| **SPD** | 50% ⚠️ | NEEDS WORK | Function-based | MEDIUM | 120 min |

**Total Integration Effort: ~7 hours** (to get all modules production-ready)

### Quick Integration Path (Start with ready modules)

**Phase 1 - Immediate (2 hours):** Get state detector working with 3 best modules
1. Create `IRRIndicator` wrapper → Working state detector with IRR (30 min)
2. Create `MSSIndicator` wrapper → Add MSS to ensemble (30 min)
3. Create `ADXIndicator` wrapper → Add industry-standard ADX composite (45 min)
   - **Note:** ADX is better than initially assessed - has sophisticated trend_quality system!

**Phase 2 - Short-term (2 hours):** Add statistical + range signals
4. Add `get_signal()` to `IntradayTemporalFeatures` + wrapper (45 min)
5. Add `get_signal()` to `DRA` + wrapper (60 min)

**Phase 3 - Medium-term (3.5 hours):** Complete the ensemble
6. Refactor ATR/Range to class-based + wrapper (90 min)
7. Refactor SPD to class-based + wrapper (120 min)

### Key Integration Requirements

All modules must implement the `TrendIndicator` interface:
```python
class TrendIndicator(ABC):
    def calculate(self, df: pd.DataFrame) -> IndicatorResult
    def get_signal(self) -> float  # Returns 0-1 normalized signal
    @property
    def name(self) -> str
```

**Signal Convention:** Higher values (→1.0) = more trending, Lower values (→0.0) = more consolidation/choppy

### Module-Specific Notes

- **IRR**: Use `directional_strength` (1 - average_irr) as signal
- **MSS**: Use `trending_score` directly (already 0-1)
- **ADX**: Use composite `quality_score` from `trend_quality.py` (already 0-1)
  - **Best module!** Combines ADX strength + DI persistence + crossover penalty
  - Session-aware with smart defaults
  - Battle-tested on real data
- **Lag**: Use `hurst_exponent` directly (H > 0.5 = trending)
- **DRA**: Invert overlap: `1.0 - rolling_overlap` (breakout strength)
- **ATR/Range**: Invert ratio: `1.0 - ratio` (low ATR/Range = trending)
- **SPD**: Invert density: `1.0 - normalized_density` (few swings = trending)

---

## Executive Summary

This trend identification system provides **7 complementary approaches** to detecting trending vs. consolidating market regimes. Each module captures different aspects of market behavior, enabling robust regime classification through signal combination.

**End Goal**: Build a unified `state_detector.py` script that weighs and combines all modules to determine the current market state (trending, consolidating, transitional, etc.).

| Module | Core Question | Approach |
|--------|--------------|----------|
| **ADX** | How strong is directional movement? | Smoothed DI spread analysis |
| **ATR/Range** | Is volatility small relative to progress? | Ratio of local vs total movement |
| **DRA** | Has price escaped the opening range? | Overlap tracking |
| **IRR** | Is there intraperiod reversion (price giving back gains within bars)? | Candle body/range ratio |
| **Lag/Hurst** | Do returns persist or reverse? | Statistical time-series analysis |
| **MSS** | Is slope consistent across scales? | Hierarchical slope alignment |
| **SPD** | How many swing reversals occurred? | Swing point counting |

---

## System Architecture

```
                           OHLC Price Data
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │ Price-Action│        │ Statistical │        │   Range-    │
   │   Based     │        │   Based     │        │   Based     │
   └─────────────┘        └─────────────┘        └─────────────┘
         │                       │                       │
    ┌────┴────┐             ┌────┴────┐             ┌────┴────┐
    │    │    │             │         │             │         │
    ▼    ▼    ▼             ▼         ▼             ▼         ▼
  ┌───┐┌───┐┌───┐      ┌─────┐   ┌─────┐      ┌─────┐   ┌─────┐
  │ADX││MSS││SPD│      │Hurst│   │ ACF │      │ DRA │   │ATR/ │
  └───┘└───┘└───┘      └─────┘   └─────┘      └─────┘   │Range│
    │    │    │            │         │             │    └─────┘
    │    │    │            └────┬────┘             │        │
    │    │    │                 │                  │        │
    ▼    ▼    ▼                 ▼                  ▼        ▼
  ┌────────────────┐      ┌─────────────┐    ┌─────────────────┐
  │ Trend Quality  │      │   Regime    │    │ Breakout Status │
  │     Score      │      │   Signal    │    │     Signal      │
  └────────────────┘      └─────────────┘    └─────────────────┘
         │                       │                    │
         └───────────────────────┴────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Combined Regime │
                        │  Classification │
                        └─────────────────┘
```

---

## State Detector: Integration Goal

### Purpose

The end goal is to create `state_detector.py` - a unified script that:
1. Accepts OHLC price data
2. Runs all 7 trend modules
3. Weighs and combines their outputs
4. Returns a single market state classification

### Target Market States

| State | Description | Typical Signal Combination |
|-------|-------------|---------------------------|
| `STRONG_TREND` | Clear directional movement | High ADX, high MSS, low IRR, low DRA overlap, low SPD |
| `WEAK_TREND` | Directional but noisy | Moderate ADX, positive Hurst, mixed IRR |
| `CONSOLIDATION` | Range-bound, mean-reverting | Low ADX, high DRA overlap, high IRR, low Hurst, high SPD |
| `BREAKOUT` | Transitioning from range | DRA overlap dropping, ADX rising, SPD decreasing |
| `CHOPPY` | No clear regime | Conflicting signals across modules, high SPD |

### Required Module Interface

For `state_detector.py` to work cleanly, each module should expose a consistent class-based interface:

```python
class TrendIndicator(ABC):
    """Base class all trend modules should implement."""
    
    @abstractmethod
    def __init__(self, config: dict = None):
        """Initialize with optional configuration."""
        pass
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> TrendResult:
        """
        Calculate indicator from OHLC DataFrame.
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns
            
        Returns:
            TrendResult with standardized fields
        """
        pass
    
    @abstractmethod
    def get_signal(self) -> float:
        """Return normalized signal (-1 to 1 or 0 to 1)."""
        pass
```

### Current Class-Based Status

| Module | Current State | Refactoring Needed |
|--------|--------------|-------------------|
| ADX | Function-based | ✅ **Yes** - wrap in `ADXIndicator` class |
| ATR/Range | Function-based | ✅ **Yes** - wrap in `ATRRangeIndicator` class |
| DRA | ✅ Class-based (`DRA`) | Minor - add `get_signal()` method |
| IRR | Function-based | ✅ **Yes** - wrap in `IRRIndicator` class |
| Lag/Hurst | ✅ Class-based (`IntradayTemporalFeatures`) | Minor - add standard interface |
| MSS | ✅ Class-based (`MultiScaleSlope`) | Minor - add `get_signal()` method |
| SPD | Function-based | ✅ **Yes** - wrap in `SPDIndicator` class |

### Proposed State Detector Architecture

```python
# features/trend/state_detector.py

class StateDetector:
    """
    Unified market state detection using weighted indicator ensemble.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize all indicators
        self.indicators = {
            'adx': ADXIndicator(self.config.get('adx')),
            'atr_range': ATRRangeIndicator(self.config.get('atr_range')),
            'dra': DRAIndicator(self.config.get('dra')),
            'irr': IRRIndicator(self.config.get('irr')),
            'lag': LagIndicator(self.config.get('lag')),
            'mss': MSSIndicator(self.config.get('mss')),
            'spd': SPDIndicator(self.config.get('spd')),
        }
        
        # Default weights (can be optimized)
        self.weights = self.config.get('weights', {
            'adx': 0.18,
            'atr_range': 0.10,
            'dra': 0.12,
            'irr': 0.12,
            'lag': 0.12,
            'mss': 0.24,
            'spd': 0.12,
        })
    
    def detect(self, df: pd.DataFrame) -> StateResult:
        """
        Detect market state from OHLC data.
        
        Returns:
            StateResult with state classification and confidence
        """
        # Run all indicators
        signals = {}
        for name, indicator in self.indicators.items():
            try:
                result = indicator.calculate(df)
                signals[name] = indicator.get_signal()
            except Exception as e:
                signals[name] = None  # Handle gracefully
        
        # Combine signals
        state, confidence = self._classify_state(signals)
        
        return StateResult(
            state=state,
            confidence=confidence,
            signals=signals,
            weights=self.weights
        )
    
    def _classify_state(self, signals: dict) -> Tuple[str, float]:
        """Classify market state from individual signals."""
        # Weighted combination logic here
        pass
```

### Implementation Roadmap

1. **Phase 1: Standardize Interfaces**
   - Create `TrendIndicator` base class
   - Create `TrendResult` and `StateResult` dataclasses
   - Define standard column names

2. **Phase 2: Refactor Function-Based Modules**
   - Wrap ADX functions in `ADXIndicator` class
   - Wrap ATR/Range functions in `ATRRangeIndicator` class
   - Wrap IRR functions in `IRRIndicator` class
   - Wrap SPD functions in `SPDIndicator` class

3. **Phase 3: Update Existing Classes**
   - Add `get_signal()` to DRA, MSS
   - Standardize input/output formats

4. **Phase 4: Build State Detector**
   - Implement `StateDetector` class
   - Add weighted signal combination
   - Add state classification logic

5. **Phase 5: Optimization**
   - Backtesting framework for weight optimization
   - Threshold calibration from historical data

---

## Module Summary

### 1. ADX (Average Directional Index) Module
**Location**: `features/trend/ADX/`

**Purpose**: Measure trend strength using Wilder's directional movement system.

**Key Components**:
- `adx_calc.py`: Core ADX calculation
- `di_indicators.py`: +DI/-DI directional indicators
- `di_crossovers.py`: DI crossover penalty scoring
- `di_persistence.py`: DI dominance persistence
- `trend_quality.py`: **Composite trend quality score**

**Output**: Trend quality score (0-1) combining strength, persistence, and crossover metrics.

**Strengths**:
- Industry-standard indicator
- Captures both trend strength and direction
- Well-tested composite scoring

**Weaknesses**:
- Long warm-up period (2 × period)
- Static parameters
- **Not class-based** (needs refactoring for state_detector integration)

---

### 2. ATR/Range Module
**Location**: `features/trend/ATR Range/`

**Purpose**: Compare local volatility (ATR) to total directional movement (range).

**Key Metric**: `median_ATR / total_range`
- Low ratio → Trending (small bars, big progress)
- High ratio → Consolidating (big bars, little progress)

**Output**: TRENDING / NEUTRAL / CONSOLIDATING signal

**Strengths**:
- Simple and intuitive
- Quick computation
- Robust to outliers (uses median)

**Weaknesses**:
- No direction information
- Arbitrary thresholds
- Vulnerable to single spike bars
- **Not class-based** (needs refactoring for state_detector integration)

---

### 3. DRA (Dynamic Range Analysis) Module
**Location**: `features/trend/DRA/`

**Purpose**: Track how much price overlaps with an initial reference range (e.g., opening 10 minutes).

**Key Metric**: Rolling average overlap (0-1)
- High overlap → Still in opening range (consolidating)
- Low overlap → Broke out of range (trending)

**Output**: Continuous overlap score enabling breakout detection.

**Strengths**:
- Excellent for breakout trading
- Streaming/real-time capable
- Clear reference point

**Weaknesses**:
- Fixed reference range
- No direction awareness
- Doesn't detect failed breakouts

---

### 4. IRR (Intraperiod Reversion Ratio) Module
**Location**: `features/trend/IRR/`

**Purpose**: Quantify intraperiod reversion - how much price "gives back" within each bar.

**Key Metric**: `1 - (body / range)` per bar
- Low IRR → Full-body candles (low reversion, full follow-through)
- High IRR → Wick-heavy candles (high reversion, price reversed within bar)

**Output**: Regime classification (directional / high_reversion / mixed)

**Strengths**:
- Bar-level granularity
- Based on proven candlestick concepts
- Fast computation

**Weaknesses**:
- Doesn't confirm consistent direction
- Single-bar focus misses inter-bar patterns
- Static thresholds
- **Not class-based** (needs refactoring for state_detector integration)

---

### 5. Lag Autocorrelation / Hurst Module
**Location**: `features/trend/Lag autocorr/`

**Purpose**: Statistical analysis of return persistence using time-series methods.

**Key Metrics**:
- **Autocorrelation**: Correlation between returns at lag
- **Hurst Exponent**: Long-range dependence (0=reverting, 0.5=random, 1=persistent)

**Output**: Regime signal (trending / mean_reverting / random_walk / weak_*)

**Strengths**:
- Statistically rigorous
- Captures temporal dependence
- Theoretically grounded

**Weaknesses**:
- Requires sufficient data (50+ bars)
- Sensitive to parameter choices
- No confidence intervals

---

### 6. MSS (Multi-Scale Slope) Module
**Location**: `features/trend/MSS tan/`

**Purpose**: Most sophisticated analysis - checks if slope is consistent across multiple time scales.

**Key Approach**:
1. Calculate main window normalized slope
2. Divide into sub-windows, calculate each slope
3. Find prominent swings, analyze path coherence
4. Combine into weighted composite score

**Output**: Trending score (0-1) with component breakdown and interpretation.

**Strengths**:
- Multi-scale analysis captures trend quality
- Includes swing structure analysis
- Detailed diagnostics for debugging

**Weaknesses**:
- Complex with many parameters
- Computationally expensive
- Requires tuning for different instruments

---

### 7. SPD (Swing Point Density) Module
**Location**: `features/trend/Swing Point Density/`

**Purpose**: Detect choppy vs smooth price action by counting local swing highs/lows.

**Key Metric**: Total swing count (highs + lows) in window
- Low count (< 5) → Trending (smooth directional movement)
- High count (> 8) → Choppy (frequent reversals)

**Output**: Classification (trending / mixed / chop) with swing counts.

**Strengths**:
- Simple and intuitive
- Fast computation
- Direct measure of "choppiness"
- Pre-configured time windows

**Weaknesses**:
- Raw count not normalized for window size
- Fixed 3-bar pattern detection
- No amplitude filtering (noise = significant swings)
- **Not class-based** (needs refactoring for state_detector integration)

---

## Cross-Module Comparison

### Signal Correlation Matrix

| Module A | Module B | Expected Correlation | Notes |
|----------|----------|---------------------|-------|
| ADX | MSS | High | Both measure trend strength |
| ADX | ATR/Range | Moderate | Different approaches, same goal |
| ADX | SPD | High (inverse) | Low SPD ↔ High ADX (both = trending) |
| DRA | ATR/Range | Moderate | Range-based, different focus |
| IRR | ACF | Low-Moderate | Bar vs inter-bar structure |
| IRR | SPD | Moderate | Both detect chop, different scale |
| Hurst | MSS | Moderate | Multi-scale concepts |
| DRA | MSS | Low | Breakout vs slope quality |
| SPD | MSS | Moderate (inverse) | SPD counts swings, MSS uses them |

### Computation Speed

| Module | Relative Speed | Notes |
|--------|----------------|-------|
| IRR | ⚡⚡⚡⚡⚡ | Simple bar-level calc |
| SPD | ⚡⚡⚡⚡⚡ | Simple 3-bar pattern scan |
| ATR/Range | ⚡⚡⚡⚡ | Rolling calculation |
| DRA | ⚡⚡⚡⚡ | Streaming, single pass |
| ADX | ⚡⚡⚡ | Multiple smoothing passes |
| Lag/Hurst | ⚡⚡ | Statistical computations |
| MSS | ⚡ | Multi-pass, extrema search |

### Data Requirements

| Module | Minimum Bars | Optimal Bars | Notes |
|--------|--------------|--------------|-------|
| IRR | 1 | Any | Per-bar metric |
| SPD | 3 | 20+ | Needs 3 bars for swing detection |
| ATR/Range | 10 | 25+ | Needs ATR period |
| DRA | 5 (reference) | 15+ (reference + analysis) | Reference + trailing |
| ADX | 28 | 50+ | 2×period warm-up |
| Lag/Hurst | 20 | 50-120 | Hurst needs 30+ |
| MSS | 20 | 50+ | Sub-windows need data |

---

## Recommended Signal Combinations

### 1. Fast Regime Detection (Real-time)
```
Primary: DRA (breakout status)
Confirm: IRR (bar structure)
Threshold: DRA overlap < 0.4 AND IRR < 0.45 → TRENDING
```

### 2. Robust Trend Confirmation
```
Primary: ADX trend_quality_score
Secondary: MSS trending_score  
Agreement: Both > 0.6 → HIGH CONFIDENCE TREND
Disagreement: Investigate with Hurst
```

### 3. Regime Change Detection
```
Monitor: DRA overlap transition (high → low)
Confirm: ADX strength rising
Validate: MSS directional_consistency > 0.75
```

### 4. Mean-Reversion Identification
```
Primary: Hurst < 0.4 AND ACF(1) < -0.1
Secondary: IRR > 0.6 (high intraperiod reversion)
Confirm: ATR/Range > 0.7 (high ratio) AND SPD > 8 (choppy)
```

### 5. Chop Detection
```
Primary: SPD classification = 'chop'
Confirm: IRR > 0.5 AND ADX < 0.4
Validate: DRA overlap > 0.6 (still in range)
```

---

## Session-Specific Configuration

The system is designed for intraday US equity markets with these session windows:

| Session | Time | Bar Size | Primary Modules |
|---------|------|----------|-----------------|
| Early Afternoon | 1pm-3pm | 5min | ADX, MSS, Hurst |
| Pre-Close | 3pm-3:50pm | 2min | ADX, ATR/Range, DRA |
| Close Auction | 3:50pm-4pm | 1min | DRA, IRR, DX (not ADX) |

**Note**: Shorter windows (3:50-4pm) use DX instead of ADX due to insufficient bars for ADX warm-up.

---

## Known Limitations & Future Work

### Priority: State Detector Integration

The immediate priority is building `state_detector.py`. This requires:

| Task | Priority | Effort |
|------|----------|--------|
| Create `TrendIndicator` base class | 🔴 High | Low |
| Wrap ADX in class | 🔴 High | Medium |
| Wrap ATR/Range in class | 🔴 High | Low |
| Wrap IRR in class | 🔴 High | Low |
| Wrap SPD in class | 🔴 High | Low |
| Add `get_signal()` to existing classes | 🟡 Medium | Low |
| Build `StateDetector` core | 🔴 High | Medium |
| Weight optimization framework | 🟡 Medium | High |

### System-Wide Issues

1. **No Unified Interface** ← *Blocking state_detector*
   - Each module has different input/output formats
   - No common `TrendIndicator` base class
   - **Fix**: Create unified interface with `calculate(df) -> TrendResult`

2. **Parameter Fragmentation**
   - Periods, thresholds scattered across modules
   - No centralized configuration
   - **Fix**: Create `TrendConfig` class with all parameters

3. **Missing Signal Ensemble** ← *Core of state_detector*
   - No built-in signal combination logic
   - Users must implement their own fusion
   - **Fix**: Build `StateDetector` class with weighted voting

4. **No Backtesting Framework**
   - Individual modules work, but no integrated backtester
   - **Fix**: Create `TrendBacktester` with standardized metrics

5. **Inconsistent Data Handling** ← *Blocking state_detector*
   - Some modules expect Series, others DataFrames
   - Column naming varies
   - **Fix**: Standardize on DataFrame input with configurable column names

6. **No Caching/Optimization**
   - ATR calculated separately in multiple modules
   - No shared computation
   - **Fix**: Implement shared feature cache in `StateDetector`

### Module-Specific Priorities (for state_detector)

| Module | Refactoring Needed | Priority Fix |
|--------|-------------------|--------------|
| ADX | **Create `ADXIndicator` class** | Wrap `calculate_trend_quality` |
| ATR/Range | **Create `ATRRangeIndicator` class** | Wrap `analyze_session` |
| DRA | Add `get_signal()` method | Return normalized overlap |
| IRR | **Create `IRRIndicator` class** | Add DataFrame support |
| Lag/Hurst | Add `get_signal()` method | Return Hurst-based signal |
| MSS | Add `get_signal()` method | Return normalized trending_score |
| SPD | **Create `SPDIndicator` class** | Add DataFrame support, normalize count |

---

## Quick Start Guide

### Current Usage (Individual Modules)

```python
import pandas as pd

# Load your OHLC data
df = pd.DataFrame({
    'open': [...],
    'high': [...], 
    'low': [...],
    'close': [...],
    'timestamp': [...]
})

# 1. ADX-based trend quality
from features.trend.ADX.trend_quality import calculate_trend_quality
adx_result = calculate_trend_quality(df, "3pm-3:50pm")
print(f"ADX Quality: {adx_result['quality_score']}")

# 2. MSS multi-scale analysis
from features.trend.MSS_tan.mss import analyze_session
mss_result = analyze_session(df, session_name='3pm')
print(f"MSS Score: {mss_result['trending_score']}")

# 3. Statistical regime
from features.trend.Lag_autocorr.lag import analyze_intraday_session
stats_result = analyze_intraday_session(df)
print(f"Statistical Regime: {stats_result['regime']}")

# 4. Manual combination (current approach)
if (adx_result['quality_score'] > 0.6 and 
    mss_result['trending_score'] > 0.6 and
    stats_result['regime'] in ['trending', 'weak_trend']):
    print("HIGH CONFIDENCE: TRENDING")
else:
    print("UNCERTAIN or CONSOLIDATING")
```

### Future Usage (After state_detector.py)

```python
import pandas as pd
from features.trend.state_detector import StateDetector

# Load your OHLC data
df = pd.DataFrame({
    'open': [...],
    'high': [...], 
    'low': [...],
    'close': [...],
})

# One-line state detection
detector = StateDetector()
result = detector.detect(df)

print(f"Market State: {result.state}")        # e.g., "STRONG_TREND"
print(f"Confidence: {result.confidence:.2f}") # e.g., 0.85
print(f"Individual Signals: {result.signals}")

# Access individual indicator results if needed
print(f"ADX Signal: {result.signals['adx']}")
print(f"MSS Signal: {result.signals['mss']}")
```

---

## File Index

```
features/trend/
│
├── state_detector.py        # 🚧 PLANNED: Unified state detection
├── base.py                  # 🚧 PLANNED: TrendIndicator base class
├── SYSTEM_OVERVIEW.md       # This document
│
├── ADX/
│   ├── adx_calc.py          # Core ADX calculation
│   ├── di_crossovers.py     # DI crossover analysis
│   ├── di_indicators.py     # +DI/-DI calculation
│   ├── di_persistence.py    # DI dominance tracking
│   ├── trend_quality.py     # Composite trend score
│   ├── indicator.py         # 🚧 PLANNED: ADXIndicator class wrapper
│   ├── testing/
│   │   └── test_adx.py      # Unit tests
│   └── ADX.md               # Module documentation
│
├── ATR Range/
│   ├── atr.py               # ATR/Range ratio analysis
│   ├── indicator.py         # 🚧 PLANNED: ATRRangeIndicator class wrapper
│   └── ATR Range.md         # Module documentation
│
├── DRA/
│   ├── dra.py               # Dynamic Range Analysis (already class-based)
│   ├── test/
│   │   └── test_dra.py      # Unit tests
│   └── DRA.md               # Module documentation
│
├── IRR/
│   ├── irr.py               # Intraperiod Reversion Ratio
│   ├── indicator.py         # 🚧 PLANNED: IRRIndicator class wrapper
│   └── IRR.md               # Module documentation
│
├── Lag autocorr/
│   ├── lag.py               # Autocorrelation & Hurst (already class-based)
│   └── Lag autocorr.md      # Module documentation
│
├── MSS tan/
│   ├── mss.py               # Multi-Scale Slope analysis (already class-based)
│   └── MSS tan.md           # Module documentation
│
└── Swing Point Density/
    ├── spd.py               # Swing point counting and classification
    ├── indicator.py         # 🚧 PLANNED: SPDIndicator class wrapper
    └── spd.md               # Module documentation
```

---

## Changelog

- **Current State**: 7 independent modules, each with separate documentation
- **V1.0**: Initial implementation of all modules
- **V2.0** (MSS only): Improved calibration for real market data

---

## References

- Wilder, J.W. (1978). "New Concepts in Technical Trading Systems" - ADX
- Hurst, H.E. (1951). "Long-term storage capacity of reservoirs" - Hurst Exponent
- Mandelbrot, B.B. (1971). "Analysis of long-run dependence in economics"
- Standard candlestick analysis literature - IRR concepts

