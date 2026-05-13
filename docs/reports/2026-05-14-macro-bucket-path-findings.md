# Macro Bucket Path Findings

Date: 2026-05-14
Branch/worktree: `feat/macro-delta-reversal`

## Study Scope

Question: does first-10-second volume-delta conviction inside the 15:50 and 15:59 ET macro candles predict continuation, fade, or churn after the impulse?

Input:

- `outputs/nq_macro_volume_delta_5s.parquet`

Outputs:

- `outputs/nq_macro_bucket_path.parquet`
- `outputs/nq_macro_bucket_path_summary.parquet`

## Candle Definitions

- `k350`: 15:50:00-15:50:59 ET, macro buckets `0..11`.
- `k359`: 15:59:00-15:59:59 ET, macro buckets `108..119`.

Both candles are normalized to relative 5-second buckets `0..11` before path features are computed.

## Runtime Shapes

- study shape: `(1682, 185)`
- summary shape: `(78, 61)`
- complete candles: 841 days for `k350`, 841 days for `k359`

## Important Metric Distinction

Full-candle and early-30s continuation include the first-10-second predictor, so those metrics measure whether the early impulse remains dominant in inclusive cumulative windows. Residual continuation excludes the predictor:

- `post_10s_to_30s`: buckets `2..5` only.
- `post_10s`: buckets `2..11` only.

Use residual metrics for “after early 10s / rest of candle” claims. Use full-candle metrics only for inclusive path anchoring.

Signed early-conviction categories are sign guarded: negative labels only occur when `early_10s_sign < 0`, positive labels only when `early_10s_sign > 0`, zero maps to neutral.

## Baseline Continuation/Fade

| Candle | Signal Days Full | Incl. Continue 30s | Incl. Fade 30s | Residual Signal 10-30s | Residual Continue 10-30s | Residual Fade 10-30s | Residual Signal Post10 | Residual Continue Post10 | Residual Fade Post10 | Continue Late 30s | Fade Late 30s | Incl. Continue Full | Incl. Fade Full | Median Early 10s | Median Late 30s | Median Full | Median Path Efficiency | Median Sign Flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k350` | 834 | 88.2% | 11.8% | 835 | 53.3% | 46.7% | 835 | 52.8% | 47.2% | 48.7% | 51.3% | 80.0% | 20.0% | +9.0 | -3.0 | +1.0 | +0.002 | 0 |
| `k359` | 836 | 74.8% | 25.2% | 833 | 58.5% | 41.5% | 835 | 56.8% | 43.2% | 55.4% | 44.6% | 64.4% | 35.6% | -2.0 | -25.0 | -25.0 | -0.037 | 1 |

Interpretation: inclusive full-candle anchoring is much stronger at 15:50 than 15:59, but residual post-10-second flow is near balanced at 15:50 and only modestly continuation-biased at 15:59. Do not read the 80.0% 15:50 full-inclusive continuation as “rest of candle continues”; residual post10 continuation is 52.8%.

## First-10s Signed Conviction

Signed early-conviction buckets are ranked separately for each candle, then guarded by actual early sign.

### 15:50 (`k350`)

| Early Category | n | Residual Continue Post10 | Residual Fade Post10 | Incl. Continue Full | Incl. Fade Full | Median Early 10s | Median Late 30s | Median Full | Median Path Efficiency | Median Sign Flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strong negative | 169 | 51.8% | 48.2% | 91.1% | 8.9% | -200.0 | +3.0 | -225.0 | -0.473 | 0 |
| weak negative | 237 | 58.2% | 41.8% | 72.6% | 27.4% | -57.0 | -3.0 | -75.0 | -0.294 | 0 |
| neutral | 4 | n/a | n/a | n/a | n/a | 0.0 | -33.0 | -43.0 | -0.359 | 1.5 |
| weak positive | 263 | 54.6% | 45.4% | 71.9% | 28.1% | +68.0 | +3.0 | +85.0 | +0.299 | 0 |
| strong positive | 168 | 43.5% | 56.5% | 91.7% | 8.3% | +218.0 | -13.0 | +222.0 | +0.428 | 0 |

Read: 15:50 first-10-second high conviction strongly anchors inclusive full-candle sign, but post10 residual flow does not strongly continue. Strong positive residual post10 actually fades 56.5%, while full-inclusive still closes same-signed 91.7% because the first 10 seconds dominate the candle total.

### 15:59 (`k359`)

| Early Category | n | Residual Continue Post10 | Residual Fade Post10 | Incl. Continue Full | Incl. Fade Full | Median Early 10s | Median Late 30s | Median Full | Median Path Efficiency | Median Sign Flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strong negative | 169 | 65.1% | 34.9% | 82.2% | 17.8% | -143.0 | -72.0 | -290.0 | -0.419 | 0 |
| weak negative | 257 | 53.7% | 46.3% | 56.4% | 43.6% | -41.0 | -40.0 | -94.0 | -0.194 | 1 |
| neutral | 4 | n/a | n/a | n/a | n/a | 0.0 | -80.0 | -40.5 | -0.042 | 2.5 |
| weak positive | 243 | 55.1% | 44.9% | 56.8% | 43.2% | +31.0 | +13.0 | +84.0 | +0.109 | 2 |
| strong positive | 168 | 55.4% | 44.6% | 69.5% | 30.5% | +133.5 | +24.5 | +208.5 | +0.294 | 0 |

Read: 15:59 has modest residual continuation, strongest for strong negative early flow. Full-inclusive continuation remains higher than residual because it includes the early impulse.

## First-10s Absolute Conviction

Absolute conviction uses absolute first-10-second delta deciles, separately by candle.

| Candle | Abs Category | n | Residual Continue Post10 | Residual Fade Post10 | Incl. Continue Full | Incl. Fade Full | Median Early 10s | Median Full | Median Path Efficiency | Median Early Abs Flow Share | Median Sign Flips |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k350` | low | 253 | 53.2% | 46.8% | 62.6% | 37.4% | -2.0 | -2.0 | -0.006 | 0.108 | 1 |
| `k350` | mid | 420 | 53.9% | 46.1% | 83.6% | 16.4% | +67.5 | -0.5 | -0.001 | 0.315 | 0 |
| `k350` | high | 168 | 49.4% | 50.6% | 96.4% | 3.6% | +218.0 | +100.0 | +0.151 | 0.488 | 0 |
| `k359` | low | 253 | 49.8% | 50.2% | 51.8% | 48.2% | +5.0 | +27.0 | +0.040 | 0.024 | 2 |
| `k359` | mid | 420 | 58.1% | 41.9% | 64.2% | 35.8% | -39.0 | -50.0 | -0.076 | 0.100 | 1 |
| `k359` | high | 168 | 63.9% | 36.1% | 83.3% | 16.7% | -144.5 | -53.5 | -0.054 | 0.229 | 0 |

Read: high absolute first-10-second conviction mostly means early impulse dominates the inclusive full-candle total. It is not equivalent to residual follow-through at 15:50. At 15:59, high absolute conviction has stronger residual continuation than 15:50.

## Strongest Rows

Inclusive full-candle anchors:

- `k350` high absolute-conviction category: 96.4% continue to full, median full delta +100.
- `k350` strong positive category: 91.7% continue to full, median full delta +222.
- `k350` strong negative category: 91.1% continue to full, median full delta -225.
- `k359` high absolute-conviction category: 83.3% continue to full, median full delta -53.5.
- `k359` strong negative category: 82.2% continue to full, median full delta -290.

Residual post10 rows:

- `k359` strong negative: 65.1% residual continuation after first 10s.
- `k359` high absolute-conviction: 63.9% residual continuation after first 10s.
- `k350` strong positive: 56.5% residual fade after first 10s despite 91.7% inclusive full continuation.

No large stable residual fade group appears at category level except mild 15:50 strong-positive mean reversion after the opening impulse.

## Current Best Read

Most useful formulation:

> First-10-second volume delta is a strong inclusive path anchor for 15:50 and a moderate anchor for 15:59. Residual follow-through after the first 10 seconds is much weaker; use post10 residual metrics when testing actual continuation after the impulse.

Operational implications from this sample:

1. At 15:50, high-conviction first-10-second delta usually defines the full-candle signed total, but post10 flow is close to balanced.
2. At 15:59, early conviction helps and residual continuation is modestly positive, strongest for strong negative/high-absolute early flow.
3. Weak/neutral early flow is much less useful; neutral rows are rare after sign guarding.
4. High absolute conviction matters for inclusive full-candle dominance more than for residual follow-through.
5. This is volume-flow structure only and should be joined to price outcomes before trade use.

## Caveats

- Study uses existing 5-second volume-delta parquet, not raw tick/order-type data.
- No price target is included.
- Findings describe volume-flow path behavior, not trade recommendations.
