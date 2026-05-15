# Macro 15:50 Delta Impulse Findings

Date: 2026-05-13
Branch/worktree: `feat/macro-delta-reversal`

## Study Scope

Question: does accumulated ETH-only, RTH-only, or ETH+RTH volume delta before 15:50 ET predict the initial 15:50:00-15:50:09 ET volume-delta impulse?

Inputs used:

- `outputs/nq_globex_volume_delta_1m.parquet`
- `outputs/nq_macro_volume_delta_5s.parquet`

Outputs:

- `outputs/nq_macro_1550_delta_impulse.parquet`
- `outputs/nq_macro_1550_delta_impulse_summary.parquet`

Runtime dataset:

- study shape: `(841, 110)`
- summary shape: `(411, 28)`

## Main Predictors

- `eth_only_pre350`: Globex/session minute index `0..929`.
- `rth_only_pre350`: session minute index `930..1309`, RTH through 15:49 ET.
- `eth_rth_pre350`: session minute index `0..1309`, full pre-15:50 accumulated delta.

## Primary Target

- `k350_00_09`: 15:50:00-15:50:09 ET, macro 5-second buckets `0..1`.

Support windows:

- `k350_00_04`: first 5 seconds.
- `k350_05_09`: second 5 seconds.
- `k350_00_29`: first 30 seconds.
- `k350_00_59`: full 15:50 minute.

## First-10s Result

For `k350_00_09`:

| Predictor | Signal Days | Opposite Rate | Correlation | Positive Predictor Median Target | Negative Predictor Median Target |
|---|---:|---:|---:|---:|---:|
| `eth_only_pre350` | 837 | 51.1% | -0.018 | +3.0 | +10.0 |
| `eth_rth_pre350` | 837 | 54.6% | -0.114 | -15.5 | +22.0 |
| `rth_only_pre350` | 837 | 55.0% | -0.114 | -14.0 | +24.0 |

Interpretation: ETH-only pre-RTH delta has almost no predictive relationship with the first 10 seconds of 15:50. RTH-only and ETH+RTH show a modest opposite-signed/reversal tendency, with RTH-only slightly best by sign rate and essentially tied by correlation.

## Support Window Comparison

Opposite rates by target window:

| Target | `eth_only_pre350` | `eth_rth_pre350` | `rth_only_pre350` |
|---|---:|---:|---:|
| `k350_00_04` | 50.1% | 54.0% | 55.1% |
| `k350_05_09` | 50.6% | 54.0% | 53.7% |
| `k350_00_09` | 51.1% | 54.6% | 55.0% |
| `k350_00_29` | 50.5% | 53.7% | 55.0% |
| `k350_00_59` | 50.0% | 54.8% | 56.3% |

Interpretation: the RTH-only signal persists across the first minute and is not isolated to one 5-second bucket. ETH-only remains near random across all support windows.

## Tail Findings

For `k350_00_09`:

| Predictor | Tail | n | Opposite Rate | Median Target | Target IQR |
|---|---|---:|---:|---:|---|
| `eth_rth_pre350` | positive top 20% | 81 | 60.5% | -84.0 | [-186.0, +117.0] |
| `eth_rth_pre350` | positive top 10% | 41 | 61.0% | -66.0 | [-164.0, +111.0] |
| `eth_rth_pre350` | negative bottom 20% | 89 | 55.1% | +17.0 | [-60.0, +128.0] |
| `eth_rth_pre350` | negative bottom 10% | 45 | 64.4% | +30.0 | [-58.0, +118.0] |
| `rth_only_pre350` | positive top 20% | 77 | 59.7% | -66.0 | [-166.0, +111.0] |
| `rth_only_pre350` | positive top 10% | 39 | 53.8% | -41.0 | [-161.0, +146.0] |
| `rth_only_pre350` | negative bottom 20% | 93 | 57.6% | +17.0 | [-53.0, +114.0] |
| `rth_only_pre350` | negative bottom 10% | 48 | 56.2% | +24.0 | [-86.0, +87.0] |

Interpretation: tails improve the ETH+RTH signal more clearly than the base sign rate, especially for the most negative ETH+RTH imbalance. RTH-only positive top 20% is useful, but the top 10% is noisier.

## Decile Highlights

For `k350_00_09` raw predictor deciles:

| Predictor | Decile | Mean Predictor Delta | Median Target | Opposite Rate |
|---|---:|---:|---:|---:|
| `eth_rth_pre350` | 1 most negative | -8124 | +12.0 | 54.1% |
| `eth_rth_pre350` | 9 positive | +4329 | -20.5 | 59.5% |
| `eth_rth_pre350` | 10 most positive | +7841 | -88.5 | 60.7% |
| `rth_only_pre350` | 1 most negative | -7727 | +24.0 | 58.3% |
| `rth_only_pre350` | 9 positive | +3751 | -25.0 | 60.2% |
| `rth_only_pre350` | 10 most positive | +7180 | -41.5 | 57.1% |

Interpretation: positive unresolved imbalance in RTH or ETH+RTH tends to precede negative first-10s impulse. Negative imbalance also reverses, but less consistently for ETH+RTH decile 1 than its negative bottom 10% tail.

## Current Best Read

Most useful formulation:

> RTH-only accumulated volume delta through 15:49 ET, and nearly equivalently ETH+RTH accumulated delta, modestly predict opposite-signed 15:50 opening impulse delta. ETH-only does not add useful signal.

Operational implications from this sample:

1. Use RTH-only or ETH+RTH pre-15:50 delta, not ETH-only.
2. The first 10 seconds show a modest reversal tendency, not a strong standalone edge.
3. Positive pre-15:50 imbalance is the cleaner practical setup for expecting negative impulse flow.
4. The effect persists through first 30/60 seconds rather than disappearing after the first 10 seconds.
5. Treat this as volume-flow context only until joined with price outcomes.

## Caveats

- Study uses 5-second volume-delta buckets, not raw order-type-level imbalance.
- No price target is included.
- Findings describe volume-flow relationships, not trade recommendations.
