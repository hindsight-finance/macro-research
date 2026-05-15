# Macro Delta Reversal Findings

Date: 2026-05-13
Branch/worktree: `feat/macro-delta-reversal`

## Study Scope

Question: does accumulated volume delta before the 15:59 ET close candle predict opposite-signed delta into the 15:59 close flow?

Inputs used:

- `outputs/nq_globex_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_5s.parquet`

Main outputs:

- `outputs/nq_macro_delta_reversal.parquet`
- `outputs/nq_macro_delta_reversal_summary.parquet`

Runtime dataset after seconds extension:

- study shape: `(841, 211)`
- summary shape: `(1744, 37)`

## Main Predictors

Primary cumulative imbalance definitions:

- `eth_rth_pre59`: ETH + RTH through 15:49 ET.
- `eth_rth_macro_pre59`: ETH + RTH + 15:50:00-15:58:59 ET.
- `rth_macro_pre59`: RTH + 15:50:00-15:58:59 ET.
- `eth_rth_pre_35940`: ETH + RTH + 15:50:00-15:59:39 ET.
- `rth_pre_35940`: RTH + 15:50:00-15:59:39 ET.
- `eth_rth_pre_35950`: ETH + RTH + 15:50:00-15:59:49 ET.

Primary 15:59 target windows:

- `k359`: full 15:59 candle.
- `k359_00_29`: 15:59:00-15:59:29.
- `k359_30_59`: 15:59:30-15:59:59.
- `k359_40_59`: 15:59:40-15:59:59.
- `k359_45_59`: 15:59:45-15:59:59.
- `k359_50_59`: 15:59:50-15:59:59.

## Full-Minute Baseline

The original full-minute result supports a weak reversal effect:

| Predictor | Target | Opposite Rate | Correlation | Positive Predictor Median Target | Negative Predictor Median Target |
|---|---:|---:|---:|---:|---:|
| `eth_rth_pre59` | `k359` | 55.8% | -0.160 | -69.5 | +21.0 |
| `eth_rth_macro_pre59` | `k359` | 57.1% | -0.146 | -92.0 | +28.5 |
| `rth_macro_pre59` | `k359` | 56.0% | -0.132 | -91.0 | +23.0 |

Interpretation: full-minute 15:59 has reversal tendency, but signal is modest.

## Seconds-Level Finding

Reversal effect is concentrated late in the 15:59 candle. First 30 seconds show no useful reversal signal.

For `eth_rth_macro_pre59`:

| Target | Opposite Rate | Correlation | Positive Predictor Median Target | Negative Predictor Median Target |
|---|---:|---:|---:|---:|
| `k359_00_29` | 50.1% | +0.052 | -14 | -16 |
| `k359_30_59` | 59.4% | -0.208 | -79 | +45 |
| `k359_45_59` | 62.1% | -0.249 | -93 | +50 |
| `k359_50_59` | 61.8% | -0.249 | -73 | +49 |

Interpretation: 15:59:00-15:59:29 is noise/mixed auction. Signal appears in 15:59:30-15:59:59, strongest in final 15/10 seconds.

## Best Last-10s Comparison

For `k359_50_59`:

| Predictor | Opposite Rate | Correlation | Positive Predictor Median Target | Negative Predictor Median Target |
|---|---:|---:|---:|---:|
| `eth_rth_pre59` | 59.5% | -0.249 | -68.5 | +46.0 |
| `eth_rth_macro_pre59` | 61.8% | -0.249 | -73.0 | +49.0 |
| `rth_macro_pre59` | 59.7% | -0.226 | -62.0 | +46.5 |

Interpretation: ETH+RTH+15:50-15:58 is best by sign rate. ETH+RTH alone is nearly same by correlation.

## Tail Findings

For `eth_rth_macro_pre59 -> k359_50_59`:

| Tail | n | Opposite Rate | Median Target | Target IQR |
|---|---:|---:|---:|---|
| positive top 20% | 81 | 72.2% | -128 | [-287, +29] |
| positive top 10% | 41 | 65.9% | -147 | [-294, +92] |
| negative bottom 20% | 88 | 59.1% | +74.5 | [-63, +232] |
| negative bottom 10% | 45 | 62.2% | +133 | [-28, +269] |

Interpretation: positive accumulated imbalance is the cleaner reversal setup. Negative imbalance reversal exists but is weaker/noisier.

## Imbalance-Decile Findings

For `eth_rth_macro_pre59 -> k359_50_59`:

| Imbalance Decile | Mean Cum Delta | Median Last-10s Delta | Opposite Rate |
|---:|---:|---:|---:|
| 1 most negative | -8087 | +100 | 68.2% |
| 8 positive | +2896 | -92.5 | 63.1% |
| 9 positive | +4492 | -115.5 | 76.8% |
| 10 most positive | +7842 | -96.5 | 70.2% |

Interpretation: effect is better validated away from the mean. Decile 9/10 positive imbalance are strongest practical zones.

## Macro Pre-59 Context

For `k359_50_59`:

| Condition | Predictor | n | Opposite Rate | Median Target |
|---|---|---:|---:|---:|
| macro pre59 same as ETH+RTH | `eth_rth_macro_pre59` | 441 | 61.3% | -11.0 |
| macro pre59 opposes ETH+RTH | `eth_rth_macro_pre59` | 400 | 62.4% | -15.0 |
| macro pre59 same as RTH | `rth_macro_pre59` | 436 | 61.8% | -10.5 |
| macro pre59 opposes RTH | `rth_macro_pre59` | 405 | 57.4% | -18.0 |

Interpretation: whether 15:50-15:58 continues or opposes prior imbalance matters less than expected. The combined unresolved imbalance still drives final-seconds reversal tendency.

## Pre-15:59:40 Extension

Tested using first 40 seconds of 15:59 as part of the predictor:

- `eth_rth_pre_35940`: ETH + RTH + 15:50:00-15:59:39.
- Target: `k359_40_59`, 15:59:40-15:59:59.

Main results:

| Predictor | Target | Opposite Rate | Correlation | Positive Predictor Median Target | Negative Predictor Median Target |
|---|---|---:|---:|---:|---:|
| `eth_rth_pre_35940` | `k359_40_59` | 59.8% | -0.218 | -93.0 | +34.5 |
| `eth_rth_pre_35940` | `k359_50_59` | 61.4% | -0.242 | -71.0 | +48.0 |
| `rth_pre_35940` | `k359_40_59` | 55.6% | -0.187 | -71.0 | +17.0 |
| `rth_pre_35940` | `k359_50_59` | 59.9% | -0.219 | -61.0 | +46.5 |

Interpretation: adding 15:59:00-15:59:39 into the predictor does not improve the signal. It slightly weakens last-20s results and is roughly comparable for last-10s.

Tail results for `eth_rth_pre_35940 -> k359_40_59`:

| Tail | n | Opposite Rate | Median Target |
|---|---:|---:|---:|
| positive top 20% | 81 | 67.9% | -124 |
| positive top 10% | 41 | 63.4% | -78 |
| negative bottom 20% | 89 | 60.7% | +74 |
| negative bottom 10% | 45 | 64.4% | +82 |

Imbalance-decile highlights for `eth_rth_pre_35940 -> k359_40_59`:

| Decile | Mean Cum Delta | Median Target | Opposite Rate |
|---:|---:|---:|---:|
| 1 most negative | -8115 | +95 | 67.1% |
| 8 positive | +2849 | -125 | 71.4% |
| 9 positive | +4500 | -127 | 65.5% |
| 10 most positive | +7803 | -92.5 | 66.7% |

## Current Best Read

Most useful formulation:

> Accumulated ETH+RTH+15:50-15:58 volume delta predicts opposite-signed final 15/10-second 15:59 flow, especially when the accumulated imbalance is strongly positive.

Operational implications from this sample:

1. Do not use 15:59 full-minute average alone; signal is diluted.
2. First 30 seconds of 15:59 are not predictive reversal flow.
3. Final 15/10 seconds carry the strongest reversal behavior.
4. Positive accumulated imbalance has the cleanest reversal profile.
5. Adding 15:59:00-15:59:39 to the predictor does not improve the edge.
6. 15:50-15:58 is best treated as auction/context, not as a direct 15:59 reversal predictor.

## Caveats

- Study uses 5-second buckets, not raw tick or order-type-level NOII/EOII separation.
- No price outcome target included here.
- Findings are volume-flow relationships, not trade recommendations.
