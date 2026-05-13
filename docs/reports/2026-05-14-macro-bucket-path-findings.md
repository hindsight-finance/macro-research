# Macro Bucket Path Findings

Date: 2026-05-14
Branch/worktree: `feat/macro-bucket-path`

## Study Scope

Question: does early 5-second volume-delta conviction inside the 15:50 and 15:59 ET macro candles predict continuation, fade, or churn through the rest of the candle?

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

- study shape: `(1682, 171)`
- summary shape: `(78, 51)`
- complete candles: 841 days for `k350`, 841 days for `k359`

## Baseline Continuation/Fade

Continuation/fade compares the sign of first-10-second delta with later cumulative/window signs.

| Candle | Signal Days Full | Continue to 30s | Fade to 30s | Continue to Late 30s | Fade to Late 30s | Continue to Full | Fade to Full | Median Early 10s | Median Full | Median Path Efficiency | Median Sign Flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k350` | 834 | 88.2% | 11.8% | 48.7% | 51.3% | 80.0% | 20.0% | +9 | +1 | +0.002 | 0 |
| `k359` | 836 | 74.8% | 25.2% | 55.4% | 44.6% | 64.4% | 35.6% | -2 | -25 | -0.037 | 1 |

Interpretation: first-10-second flow is much more persistent at 15:50 than at 15:59. The 15:50 candle usually keeps the early sign through the full minute, while 15:59 is more mixed/choppy.

## First-10s Signed Conviction

Signed early-conviction buckets are ranked separately for each candle.

### 15:50 (`k350`)

| Early Category | n | Continue Full | Fade Full | Median Early 10s | Median Late 30s | Median Full | Median Path Efficiency | Median Sign Flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strong negative | 169 | 91.1% | 8.9% | -200.0 | +3.0 | -225.0 | -0.473 | 0 |
| weak negative | 168 | 78.0% | 22.0% | -79.5 | -2.0 | -88.5 | -0.325 | 0 |
| neutral | 168 | 62.3% | 37.7% | +9.0 | +5.5 | +15.0 | +0.048 | 1 |
| weak positive | 168 | 76.0% | 24.0% | +87.0 | -9.0 | +92.5 | +0.300 | 0 |
| strong positive | 168 | 91.7% | 8.3% | +218.0 | -13.0 | +222.0 | +0.428 | 0 |

Read: 15:50 first-10-second high conviction is strongly persistent. Strong positive and strong negative both continue to full minute over 91% of signal days.

### 15:59 (`k359`)

| Early Category | n | Continue Full | Fade Full | Median Early 10s | Median Late 30s | Median Full | Median Path Efficiency | Median Sign Flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strong negative | 169 | 82.2% | 17.8% | -143.0 | -72.0 | -290.0 | -0.419 | 0 |
| weak negative | 168 | 61.9% | 38.1% | -54.0 | -72.0 | -182.5 | -0.309 | 1 |
| neutral | 168 | 50.0% | 50.0% | -2.0 | +27.5 | +35.5 | +0.046 | 2 |
| weak positive | 168 | 57.7% | 42.3% | +41.0 | +14.0 | +89.0 | +0.126 | 1 |
| strong positive | 168 | 69.5% | 30.5% | +133.5 | +24.5 | +208.5 | +0.294 | 0 |

Read: 15:59 first-10-second conviction still helps, but persistence is weaker and more asymmetric. Strong negative early flow continues more reliably than strong positive early flow.

## First-10s Absolute Conviction

Absolute conviction uses absolute first-10-second delta deciles, separately by candle.

| Candle | Abs Category | n | Continue Full | Fade Full | Median Early 10s | Median Full | Median Path Efficiency | Median Early Abs Flow Share | Median Sign Flips |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `k350` | low | 253 | 62.6% | 37.4% | -2.0 | -2.0 | -0.006 | 0.108 | 1 |
| `k350` | mid | 420 | 83.6% | 16.4% | +67.5 | -0.5 | -0.001 | 0.315 | 0 |
| `k350` | high | 168 | 96.4% | 3.6% | +218.0 | +100.0 | +0.151 | 0.488 | 0 |
| `k359` | low | 253 | 51.8% | 48.2% | +5.0 | +27.0 | +0.040 | 0.024 | 2 |
| `k359` | mid | 420 | 64.2% | 35.8% | -39.0 | -50.0 | -0.076 | 0.100 | 1 |
| `k359` | high | 168 | 83.3% | 16.7% | -144.5 | -53.5 | -0.054 | 0.229 | 0 |

Read: high absolute first-10-second conviction is meaningful for both candles, but stronger at 15:50. At 15:50, high abs conviction almost locks in the full-candle sign. At 15:59, high abs conviction helps but still fades about 17% of the time.

## Strongest Rows

The strongest continuation rows were mostly 15:50 high-conviction groups:

- `k350` high absolute-conviction category: 96.4% continue to full, median full delta +100.
- `k350` strong positive category: 91.7% continue to full, median full delta +222.
- `k350` strong negative category: 91.1% continue to full, median full delta -225.
- `k359` strong negative category: 82.2% continue to full, median full delta -290.
- `k359` high absolute-conviction category: 83.3% continue to full, median full delta -53.5.

No large stable fade group appears at category level. The strongest fade rows in deciles were small edges around 15:59 and near neutral/low-conviction behavior, not a robust reversal profile.

## Current Best Read

Most useful formulation:

> First-10-second volume delta is a strong path anchor for 15:50 and a moderate path anchor for 15:59. High absolute first-10-second conviction strongly predicts same-signed full-candle delta, especially at 15:50.

Operational implications from this sample:

1. At 15:50, high-conviction first-10-second delta usually defines the candle direction.
2. At 15:59, early conviction helps, but the candle is more prone to churn/fade.
3. Weak/neutral early flow is much less useful, especially at 15:59 where neutral behavior is effectively 50/50 and has more sign flips.
4. High absolute conviction matters more than weak signed direction; size of early impulse is important.
5. This is volume-flow structure only and should be joined to price outcomes before trade use.

## Caveats

- Study uses existing 5-second volume-delta parquet, not raw tick/order-type data.
- No price target is included.
- Findings describe volume-flow path behavior, not trade recommendations.
