# Regime Detector — Calibration Status

## Round 1 Fixes Applied

Three wrapper bugs were identified via code review and full-population
diagnostics (1,304 PM sessions). All three were **timeframe mismatch**
issues — the underlying indicator modules are correct, but the wrappers
were feeding them 1-minute data when they were designed for 5-minute bars
or shorter sessions.

### What Was Wrong

**DRA** — Used the final 10-bar rolling overlap average as the signal.
Over 120 bars, price almost always permanently leaves the 15-minute opening
range, making the rolling average collapse to 0 and the signal pin at 1.0
for 58% of sessions.

**ATR/Range** — Fed raw 1-minute bars to a module configured for 5-minute
data (`bar_size: '5min'`, `atr_period: 10`). The ratio
`median_10min_ATR / total_2hr_range` is structurally near zero (std 0.027)
because a 10-minute rolling range is tiny relative to 120 minutes of
cumulative range.

**SPD** — Ran 3-bar swing detection on raw 1-minute data. At 1-min
resolution, noise produces 40–50 swing points per session regardless of
regime. The module's thresholds (< 5 = trending, > 8 = chop) were
calibrated for 5-minute or coarser bars.

### What Was Fixed

| Indicator | Fix | Rationale |
|-----------|-----|-----------|
| **DRA** | Signal = fraction of post-reference bars with zero overlap (breakout fraction) | Preserves temporal info — a session that escapes in 7 min is different from one that stays inside for 55% of bars |
| **ATR/Range** | Resample to 5-min OHLC before calling `analyze_session()` | Matches the module's design: ATR period 10 on 5-min bars = 50-min lookback, producing meaningful ratio variance |
| **SPD** | Resample to 5-min OHLC before swing detection | Smooths noise-driven false swings; aligns with the module's count thresholds |

All fixes are in `state_detector.py` wrappers only — no changes to the
underlying indicator modules.

---

## Signal Distributions: Before vs After

```
BEFORE (broken):
Indicator      Mean    Std    Min    P10    P50    P90    Max
--------------------------------------------------------------
dra           0.941  0.091  0.497  0.796  1.000  1.000  1.000  <-- pinned
atr_range     0.907  0.027  0.785  0.872  0.908  0.941  0.977  <-- flat
spd           0.171  0.076  0.000  0.076  0.160  0.277  0.412  <-- compressed

AFTER (fixed):
Indicator      Mean    Std    Min    P10    P50    P90    Max
--------------------------------------------------------------
dra           0.549  0.278  0.000  0.143  0.576  0.905  0.990  <-- 3x std, full range
atr_range     0.786  0.062  0.553  0.706  0.788  0.862  0.960  <-- 2.3x std
spd           0.227  0.147  0.000  0.043  0.217  0.391  0.739  <-- 2x std, wider range
```

Healthy indicators (unchanged):
```
Indicator      Mean    Std    Min    P10    P50    P90    Max
--------------------------------------------------------------
adx           0.403  0.085  0.210  0.302  0.391  0.512  0.867
mss           0.483  0.155  0.131  0.305  0.455  0.705  0.884
irr           0.452  0.113  0.091  0.302  0.455  0.596  0.874
lag           0.600  0.095  0.260  0.483  0.599  0.720  0.935
```

---

## Regime Distribution: Before vs After

```
BEFORE:                      AFTER:
UNCERTAIN       1268 (97%)   UNCERTAIN        1216 (93%)
WEAK_TREND        31          CONSOLIDATION      51
STRONG_TREND       5          WEAK_TREND         25
                              STRONG_TREND        6
                              CHOPPY              6
```

Key improvements:
- **CONSOLIDATION** and **CHOPPY** categories now appear (were completely absent)
- All 5 regime types are represented in the 20-chart sample set
- UNCERTAIN still dominant — remaining improvement depends on weight
  tuning and classification threshold adjustment (Phase 2)

---

## Chart Samples (Round 1)

20 charts in `test/regime_charts/`, selected to span all detected regimes:

| # | Date | Regime | Dir | Conf | Hurst | Move |
|---|------|--------|-----|------|-------|------|
| 1 | 2021-03-15 | WEAK_TREND | UP | 34.9% | 0.541 | +0.53% |
| 2 | 2021-12-01 | UNCERTAIN | DOWN | 0.1% | 0.522 | -0.30% |
| 3 | 2022-10-03 | STRONG_TREND | UP | 42.2% | 0.474 | +1.17% |
| 4 | 2022-11-08 | WEAK_TREND | DOWN | 32.3% | 0.918 | -1.46% |
| 5 | 2023-04-28 | CHOPPY | NEUTRAL | 50.1% | 0.423 | +0.03% |
| 6 | 2023-06-15 | STRONG_TREND | UP | 46.6% | 0.792 | +0.83% |
| 7 | 2023-11-01 | UNCERTAIN | UP | 16.3% | 0.621 | +0.17% |
| 8 | 2023-12-06 | CONSOLIDATION | NEUTRAL | 40.0% | 0.481 | -0.05% |
| 9 | 2023-12-12 | WEAK_TREND | UP | 39.0% | 0.604 | +0.32% |
| 10 | 2024-04-04 | STRONG_TREND | DOWN | 49.5% | 0.532 | -1.78% |
| 11 | 2024-04-22 | STRONG_TREND | UP | 40.4% | 0.862 | +0.83% |
| 12 | 2024-05-13 | CONSOLIDATION | NEUTRAL | 35.3% | 0.698 | -0.01% |
| 13 | 2024-07-02 | WEAK_TREND | UP | 30.2% | 0.724 | +0.25% |
| 14 | 2024-08-02 | CONSOLIDATION | NEUTRAL | 32.2% | 0.641 | +0.07% |
| 15 | 2024-10-28 | CHOPPY | NEUTRAL | 41.5% | 0.430 | -0.01% |
| 16 | 2025-04-02 | UNCERTAIN | DOWN | 29.9% | 0.793 | -1.06% |
| 17 | 2025-09-19 | CONSOLIDATION | NEUTRAL | 30.1% | 0.669 | -0.02% |
| 18 | 2025-10-08 | UNCERTAIN | NEUTRAL | 7.6% | 0.558 | +0.03% |
| 19 | 2025-10-14 | CHOPPY | NEUTRAL | 40.1% | 0.537 | -0.09% |
| 20 | 2025-11-18 | CHOPPY | NEUTRAL | 48.6% | 0.368 | -0.01% |

---

## Remaining Issues / Next Steps

### ATR/Range still high-biased (mean 0.786)

Even after the 5-min resample, ATR/Range leans toward "trending" because
the session-total range is always larger than any rolling ATR window. The
signal distribution sits 0.55–0.96 which is functional (has variance) but
asymmetric. Possible further fixes:
- Adjust classification thresholds to account for the bias
- Use a rolling range denominator instead of session total

### UNCERTAIN dominance (93%)

The confidence threshold is 0.3 and most sessions produce composite scores
near 0.5, yielding sub-0.3 confidence and an UNCERTAIN classification.
Options:
- Lower `confidence_threshold` (e.g., 0.15) to let more sessions classify
- Adjust the composite → state mapping thresholds
- Re-weight indicators after visual calibration

### Phase 2: Visual Calibration Loop

Review the 20 charts and flag disagreements:
- "This is clearly trending but flagged UNCERTAIN"
- "This looks consolidating, not WEAK_TREND"
- Specific indicator signals that look wrong

Each round of feedback tightens the calibration.
